"""ONNX Runtime + TensorRT execution provider — the SmolVLA inference backend.

This is the *only* deployment path. ORT's TensorRT EP partitions the ONNX graph,
builds a TensorRT engine for each supported subgraph (cached to disk on first run),
and hands anything TRT can't take to the CUDA EP — so the whole SmolVLA graph runs
fast without the all-or-nothing failure of a single monolithic `trtexec` build, and
without that build's memory peak (which OOM'd on the Orin's 8 GB).

Precision on the Orin Nano (compute 8.7)
----------------------------------------
`tools/probe_precision.py` reports `platform_has_fast_fp16 = True` but
`platform_has_fast_bf16 = n/a`. So:

  * fp16 (default): the graph is FP32; `trt_fp16_enable` lets TRT lower compatible
    subgraphs to FP16 while `trt_layer_norm_fp32_fallback` keeps the precision-
    sensitive normalisations in FP32. Ops TRT rejects fall back to the CUDA EP at
    FP32 (the ONNX graph's dtype). This is the validated deployment mode.
  * bf16 (experimental): `trt_bf16_enable`. Keep it ONLY if logs show TRT actually
    builds BF16 tactics on this image AND it beats fp16 on latency + action error
    vs the PyTorch-BF16 reference. Benchmarked separately, never with fp16.

Requires onnxruntime-gpu built for CUDA 13 / cp312 (no PyPI aarch64 GPU wheel — see
requirements.txt for the Jetson AI Lab index / source-build note).
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

from ..io_spec import TensorSpec, resolve_io
from ..preprocess import InputBuilder
from .base import PredictResult

LOG = logging.getLogger("smolvla_runtime.backends.ort")

# Conservative defaults for the Orin Nano's 8 GB unified memory.
_DEFAULT_TRT_WORKSPACE = 1 * 1024 * 1024 * 1024   # 1 GiB
_DEFAULT_CUDA_MEM_LIMIT = 3 * 1024 * 1024 * 1024  # 3 GiB


def _ort_shape(shape) -> tuple[int, ...]:
    # ORT reports symbolic dims as strings / None — treat those as dynamic (-1).
    return tuple(d if isinstance(d, int) else -1 for d in shape)


_ORT_TYPE_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
    "tensor(uint8)": np.uint8,
}


def build_providers(
    cache_dir: str,
    precision: str = "fp16",
    trt_workspace: int = _DEFAULT_TRT_WORKSPACE,
    cuda_mem_limit: int = _DEFAULT_CUDA_MEM_LIMIT,
):
    """The canonical Orin-Nano provider stack: TensorRT EP -> CUDA EP -> CPU EP.

    precision selects the TRT reduced-precision mode (fp16 default, bf16 experiment).
    Engine + timing caches live under cache_dir so the (slow) first build is paid once.

    Build-peak overrides (env, for fitting the first build into 8 GB unified memory):
      TRT_WORKSPACE_MB   cap per-tactic GPU scratch (default 1024).
      TRT_OPT_LEVEL      TensorRT builder optimization level 0-5 (default 3). Lower
                         explores far fewer tactics -> smaller build peak + faster
                         build, at a modest runtime-latency cost. Only the *build* is
                         affected; a cached engine reloads identically either way.
    These shrink the one-time engine build; they do not change a cached engine.
    """
    os.makedirs(cache_dir, exist_ok=True)
    ws_mb = os.environ.get("TRT_WORKSPACE_MB")
    if ws_mb:
        trt_workspace = int(ws_mb) * 1024 * 1024
    trt_opts = {
        "device_id": 0,
        # fp16 is the Orin deploy path; bf16 is gated/experimental (see module docstring).
        "trt_fp16_enable": precision == "fp16",
        "trt_bf16_enable": precision == "bf16",
        # Keep the precision-sensitive norms in FP32 even when fp16 is on.
        "trt_layer_norm_fp32_fallback": True,
        # Cache the built engines + timing so the first-run compile is one-time.
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache_dir,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": cache_dir,
        # Conservative on 8 GB unified memory.
        "trt_max_workspace_size": trt_workspace,
        # Avoid carving the graph into many tiny TRT islands.
        "trt_min_subgraph_size": 5,
    }
    opt_level = os.environ.get("TRT_OPT_LEVEL")
    if opt_level:
        trt_opts["trt_builder_optimization_level"] = int(opt_level)
    cuda_opts = {
        "device_id": 0,
        "gpu_mem_limit": cuda_mem_limit,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "do_copy_in_default_stream": True,
    }
    providers: list = [("TensorrtExecutionProvider", trt_opts)]
    # TRT_DROP_CUDA_EP=1 omits the CUDA EP. Its CUDA context + cuDNN/cuBLAS
    # workspaces reserve several hundred MB of *non-swappable* memory at session
    # creation — memory the TRT engine *build* doesn't need but which can tip an
    # 8 GB board into OOM during the build. Use only to build/cache the engine;
    # a cached engine then loads cheaply with the full stack re-enabled. With the
    # whole graph on TRT, CPU EP is enough to catch any stray op.
    if not os.environ.get("TRT_DROP_CUDA_EP"):
        providers.append(("CUDAExecutionProvider", cuda_opts))
    providers.append("CPUExecutionProvider")
    return providers


class ORTBackend:
    def __init__(
        self,
        onnx_path: str,
        model_id: str,
        engine_cache_dir: str = "/tmp/smolvla_trt_cache",
        precision: str = "fp16",
        fixed_noise: bool = False,
    ):
        import onnxruntime as ort

        if precision not in ("fp16", "bf16"):
            raise ValueError(f"precision must be 'fp16' or 'bf16', got {precision!r}")

        providers = build_providers(engine_cache_dir, precision=precision)
        LOG.info(
            "Creating ORT session: %s  precision=%s "
            "(first run builds + caches the TRT engine into %s — be patient, minutes).",
            onnx_path, precision, engine_cache_dir,
        )
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        active = self._session.get_providers()
        if active[0] != "TensorrtExecutionProvider":
            LOG.warning(
                "Active providers are %s — TensorRT EP did not register (libnvinfer "
                "missing or graph rejected). Inference will use the slower fallback.",
                active,
            )
        else:
            LOG.info("Active providers: %s", active)

        inputs = [
            TensorSpec(i.name, np.dtype(_ORT_TYPE_TO_NP.get(i.type, np.float32)), _ort_shape(i.shape))
            for i in self._session.get_inputs()
        ]
        outputs = [
            TensorSpec(o.name, np.dtype(_ORT_TYPE_TO_NP.get(o.type, np.float32)), _ort_shape(o.shape))
            for o in self._session.get_outputs()
        ]
        self.io = resolve_io(inputs, outputs)
        self._builder = InputBuilder(
            model_id=model_id,
            image_size=self.io.image_size,
            lang_max_len=self.io.lang_max_len,
            state_dim=self.io.state_dim,
            chunk_size=self.io.chunk_size,
            action_dim=self.io.action_dim,
            fixed_noise=fixed_noise,
        )
        self.description = (
            f"ort precision={precision} providers={active[0]} onnx={onnx_path.split('/')[-1]}"
        )

    def predict(self, image_rgb, instruction, state) -> PredictResult:
        logical = self._builder.build(image_rgb, instruction, state)
        feeds = {self.io.role_to_name[role]: arr for role, arr in logical.items()}
        t0 = time.perf_counter()
        out = self._session.run([self.io.primary_output], feeds)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        actions = np.asarray(out[0], dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return PredictResult(actions=actions, latency_ms=latency_ms, backend="ort")
