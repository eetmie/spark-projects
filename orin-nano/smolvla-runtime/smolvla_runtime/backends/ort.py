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
import math
import os
import time
from pathlib import Path

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


# --- split-engine backend (the 8 GB Orin deploy path) ------------------------
# The monolithic graph can't TRT-build on the 8 GB board (node-count-independent
# ~6 GB weight floor). The Spark exports 9 per-component graphs instead
# (export_split_onnx.py); each carries only its weight slice so each builds a clean
# FP16 engine in <=60 s. The flow-matching denoise loop runs here in Python:
# prefill the VLM KV cache once, then decode x_t += dt*v_t for num_steps. This loop
# is verified bit-for-bit against the monolith on the Spark (parity_split_onnx.py:
# cosine 1.0000000, max_abs 2.1e-6).
_SPLIT_GRAPHS = {
    "vision": "smolvlm_vision.onnx",
    "text": "smolvlm_text.onnx",
    "state": "state_projector.onnx",
    "action_in": "action_in_projector.onnx",
    "action_out": "action_out_projector.onnx",
    "time_in": "time_in_projector.onnx",
    "time_out": "time_out_projector.onnx",
    "prefill": "smolvlm_expert_prefill.onnx",
    "decode": "smolvlm_expert_decode.onnx",
}


def _silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def _make_att_2d_masks(pad_masks: np.ndarray, att_masks: np.ndarray) -> np.ndarray:
    cumsum = np.cumsum(att_masks, axis=1)
    att = cumsum[:, None, :] <= cumsum[:, :, None]
    pad = pad_masks[:, None, :] & pad_masks[:, :, None]
    return att & pad


def _sinusoidal_time_emb(time_v: np.ndarray, dim: int, min_period: float, max_period: float) -> np.ndarray:
    frac = np.linspace(0.0, 1.0, dim // 2, dtype=np.float64)
    period = min_period * (max_period / min_period) ** frac
    scale = 1.0 / period * 2 * math.pi
    x = scale[None, :] * time_v[:, None]
    return np.concatenate([np.sin(x), np.cos(x)], axis=1)


class SplitORTBackend:
    """SmolVLA over the 9 split graphs + a Python flow-matching loop.

    Same `predict(image_rgb, instruction, state)` contract as ORTBackend, so it
    drops into the pipeline unchanged. Each graph gets its own TRT engine cache
    subdir, so the per-component engines build + cache independently (the whole
    point — none of them hits the monolith's 8 GB build wall).
    """

    def __init__(
        self,
        split_dir: str,
        model_id: str,
        engine_cache_dir: str = "/tmp/smolvla_trt_cache_split",
        precision: str = "fp16",
        num_steps: int = 10,
        min_period: float = 0.004,
        max_period: float = 4.0,
        fixed_noise: bool = False,
        providers=None,
    ):
        import onnxruntime as ort

        if precision not in ("fp16", "bf16"):
            raise ValueError(f"precision must be 'fp16' or 'bf16', got {precision!r}")
        self.num_steps = int(num_steps)
        self.min_period = float(min_period)
        self.max_period = float(max_period)

        d = Path(split_dir)
        self._sess: dict[str, object] = {}
        self._inames: dict[str, list[str]] = {}
        for key, fname in _SPLIT_GRAPHS.items():
            path = d / fname
            if not path.exists():
                raise SystemExit(f"split graph missing: {path}")
            provs = providers or build_providers(
                os.path.join(engine_cache_dir, key), precision=precision)
            LOG.info("Loading split graph %-10s %s", key, fname)
            s = ort.InferenceSession(str(path), providers=provs)
            self._sess[key] = s
            self._inames[key] = [i.name for i in s.get_inputs()]
        active = self._sess["prefill"].get_providers()
        if active[0] != "TensorrtExecutionProvider":
            LOG.warning("Split graphs not on TensorRT EP (active=%s) — slower fallback.", active)

        # Derive dims straight off the graph I/O (auto-config to whatever the Spark baked).
        def _shape(key, idx=0, out=False):
            t = (self._sess[key].get_outputs() if out else self._sess[key].get_inputs())[idx]
            return [int(x) if isinstance(x, int) else -1 for x in t.shape]

        image_size = _shape("vision")[2]
        n_img = _shape("vision", out=True)[1]
        # prefill input order is [attention_mask, position_ids, vlm_embeds]
        prefix_len = _shape("prefill", 2)[1]
        lang_max_len = prefix_len - n_img - 1               # 113 - 64 - 1 = 48
        chunk_size, action_dim = _shape("action_in")[1], _shape("action_in")[2]
        self._exp_dim = _shape("action_in", out=True)[2]    # 720
        state_dim = _shape("state")[1]
        self._n_layers = (len(self._inames["decode"]) - 3) // 2   # (N inputs - mask,pos,emb)/2

        self._builder = InputBuilder(
            model_id=model_id, image_size=image_size, lang_max_len=lang_max_len,
            state_dim=state_dim, chunk_size=chunk_size, action_dim=action_dim,
            fixed_noise=fixed_noise,
        )
        self.description = (
            f"ort-split precision={precision} providers={active[0]} steps={self.num_steps} "
            f"graphs={split_dir.rstrip('/').split('/')[-1]} prefix={prefix_len} layers={self._n_layers}"
        )
        LOG.info("Split dims: image=%d n_img=%d lang=%d state=%d chunk=%d action=%d exp=%d steps=%d",
                 image_size, n_img, lang_max_len, state_dim, chunk_size, action_dim,
                 self._exp_dim, self.num_steps)

    def _run(self, key: str, *args):
        return self._sess[key].run(None, dict(zip(self._inames[key], args)))

    def _embed_prefix(self, image, lang_tokens, lang_masks, state):
        img_emb = self._run("vision", image)[0]
        img_emb = img_emb * img_emb.shape[-1] ** 0.5
        lang_emb = self._run("text", lang_tokens)[0]
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        state_emb = self._run("state", state)[0]
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs = np.concatenate([img_emb, lang_emb, state_emb], axis=1).astype(np.float32)
        pad = np.concatenate([
            np.ones((1, img_emb.shape[1]), dtype=bool), lang_masks,
            np.ones((1, state_emb.shape[1]), dtype=bool)], axis=1)
        att = np.array([0] * img_emb.shape[1] + [0] * lang_emb.shape[1]
                       + [1] * state_emb.shape[1], dtype=bool)[None, :]
        return embs, pad, att

    def _embed_suffix(self, x_t, t):
        action_emb = self._run("action_in", x_t.astype(np.float32))[0]
        time_emb = _sinusoidal_time_emb(np.broadcast_to(t, 1), self._exp_dim,
                                        self.min_period, self.max_period)
        time_emb = np.broadcast_to(time_emb[:, None, :], action_emb.shape).copy()
        at = np.concatenate([action_emb, time_emb], axis=2).astype(np.float32)
        at = self._run("time_in", at)[0]
        at = _silu(at)
        at = self._run("time_out", at.astype(np.float32))[0]
        pad = np.ones((1, at.shape[1]), dtype=bool)
        att = np.ones(at.shape[1], dtype=bool)[None, :]
        return at.astype(np.float32), pad, att

    def _sample_actions(self, logical: dict) -> np.ndarray:
        image, lang_tokens = logical["image"], logical["lang_tokens"]
        lang_masks, state, noise = logical["lang_masks"], logical["state"], logical["noise"]
        pe, pp, pa = self._embed_prefix(image, lang_tokens, lang_masks, state)
        pmask2d = _make_att_2d_masks(pp, pa)
        ppos = (np.cumsum(pp, axis=1) - 1).astype(np.int64)
        kv = self._run("prefill", pmask2d, ppos, pe)          # 2*n_layers KV tensors

        dt = np.array(-1.0 / self.num_steps, dtype=np.float32)
        x_t = noise.astype(np.float32).copy()
        t = np.array(1.0, dtype=np.float32)
        chunk = x_t.shape[1]
        while t >= -dt / 2:
            se, sp, sa = self._embed_suffix(x_t, t)
            slen, plen = sp.shape[1], pp.shape[1]
            pref2d = np.broadcast_to(pp[:, None, :], (1, slen, plen)).copy()
            full = np.concatenate([pref2d, _make_att_2d_masks(sp, sa)], axis=2)
            pos = (np.sum(pp, axis=-1)[:, None] + np.cumsum(sp, axis=1) - 1).astype(np.int64)
            out = self._run("decode", full, pos, se, *kv)[0]
            v_t = self._run("action_out", out[:, -chunk:].astype(np.float32))[0]
            x_t = x_t + dt * v_t
            t = t + dt
        return x_t

    def predict(self, image_rgb, instruction, state) -> PredictResult:
        logical = self._builder.build(image_rgb, instruction, state)
        t0 = time.perf_counter()
        actions = self._sample_actions(logical)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return PredictResult(actions=actions, latency_ms=latency_ms, backend="ort-split")
