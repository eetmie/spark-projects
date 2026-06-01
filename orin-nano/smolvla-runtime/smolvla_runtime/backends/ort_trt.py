"""ONNX Runtime + TensorRT-EP backend — kept as a DIAGNOSTIC / fallback path.

The primary path is the pure `.engine` runtime (trt_engine.py). This one exists
because pure TRT has one nasty failure mode: if `trtexec` hits an op it can't
build, the whole engine build aborts with no detail about *which* op. ORT's
TensorRT EP, by contrast, partitions the graph and silently falls back to the
CUDA EP for unsupported subgraphs — so running this backend (and watching ORT's
verbose log for "falling back to CUDA") tells you exactly what to fix in the
export. It also compiles + caches a TRT engine itself, so it's a working
inference path even when the pure build won't go.

Requires onnxruntime-gpu from the Jetson AI Lab index (no aarch64 GPU wheel on
PyPI). See requirements.txt.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..io_spec import TensorSpec, resolve_io
from ..preprocess import InputBuilder
from .base import PredictResult

LOG = logging.getLogger("smolvla_runtime.backends.ort_trt")


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


class ORTBackend:
    def __init__(
        self,
        onnx_path: str,
        model_id: str,
        engine_cache_dir: str = "/tmp/smolvla_trt_cache",
        fp16: bool = True,
        fixed_noise: bool = False,
    ):
        import onnxruntime as ort
        import os

        os.makedirs(engine_cache_dir, exist_ok=True)
        trt_opts = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": engine_cache_dir,
            "trt_fp16_enable": fp16,
            "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,
        }
        providers = [
            ("TensorrtExecutionProvider", trt_opts),
            ("CUDAExecutionProvider", {}),
        ]
        LOG.info("Creating ORT session: %s (first run compiles TRT engine — be patient)", onnx_path)
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        active = self._session.get_providers()[0]
        if active != "TensorrtExecutionProvider":
            LOG.warning("Active ORT provider is %s, not TensorrtExecutionProvider — "
                        "libnvinfer not found or graph rejected; running slower fallback.", active)

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
        self.description = f"ort-trt-ep provider={active} onnx={onnx_path.split('/')[-1]}"

    def predict(self, image_rgb, instruction, state) -> PredictResult:
        logical = self._builder.build(image_rgb, instruction, state)
        feeds = {self.io.role_to_name[role]: arr for role, arr in logical.items()}
        t0 = time.perf_counter()
        out = self._session.run([self.io.primary_output], feeds)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        actions = np.asarray(out[0], dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return PredictResult(actions=actions, latency_ms=latency_ms, backend="ort-trt-ep")
