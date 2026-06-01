"""Pure TensorRT runtime backend (the primary, performance path).

Loads a prebuilt `.engine` (see build_engine.py) and drives it with the
TensorRT 10.x tensor-I/O API + cuda-python device buffers. No ONNX Runtime in
this path — the serialized engine *is* the runtime.

Notes that matter on the Orin Nano:
  * The engine is locked to this GPU (sm_87) + TRT version + CUDA version. Build
    it on this board; never copy one in from the Spark. ONNX is the portable
    source of truth, the engine is a local cache.
  * Static shapes only. If the engine has a dynamic dim we set it once at load.
  * Device buffers are allocated once and reused every frame — no per-call malloc.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..io_spec import TensorSpec, resolve_io
from ..preprocess import InputBuilder
from .base import PredictResult

LOG = logging.getLogger("smolvla_runtime.backends.trt_engine")


# --- TensorRT dtype <-> numpy ------------------------------------------------
def _trt_to_np(trt, dt):
    import numpy as _np
    return {
        trt.DataType.FLOAT: _np.float32,
        trt.DataType.HALF: _np.float16,
        trt.DataType.INT32: _np.int32,
        trt.DataType.INT64: _np.int64,
        trt.DataType.BOOL: _np.bool_,
        trt.DataType.INT8: _np.int8,
        trt.DataType.UINT8: _np.uint8,
    }[dt]


def _cuda_check(cudart, err, what: str = "cuda call"):
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{what} failed: {cudart.cudaGetErrorString(err)}")


class TRTEngineRunner:
    """Low-level: load engine, bind device buffers, run one inference."""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        try:
            # cuda-python >= 12.x preferred path (cuda.cudart is deprecated, gone in 13.x)
            from cuda.bindings import runtime as cudart
        except ImportError:
            from cuda import cudart

        self._trt = trt
        self._cudart = cudart

        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        runtime = trt.Runtime(logger)
        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(
                f"Failed to deserialize {engine_path}. An engine is locked to the "
                "exact GPU/TRT/CUDA it was built on — rebuild on this board if the "
                "stack changed."
            )
        self._ctx = self._engine.create_execution_context()

        err, self._stream = cudart.cudaStreamCreate()
        _cuda_check(cudart, err, "cudaStreamCreate")

        self._inputs: list[TensorSpec] = []
        self._outputs: list[TensorSpec] = []
        self._dev: dict[str, int] = {}      # tensor name -> device ptr
        self._host_out: dict[str, np.ndarray] = {}
        self._nbytes: dict[str, int] = {}

        self._setup_bindings()
        self.io = resolve_io(self._inputs, self._outputs)

    def _setup_bindings(self) -> None:
        trt, cudart = self._trt, self._cudart
        eng = self._engine
        for i in range(eng.num_io_tensors):
            name = eng.get_tensor_name(i)
            dtype = _trt_to_np(trt, eng.get_tensor_dtype(name))
            shape = tuple(self._ctx.get_tensor_shape(name))
            # Resolve any dynamic dim to its build-time optimum (engine min==opt==max
            # for our static export, but be defensive).
            if any(d < 0 for d in shape):
                shape = tuple(eng.get_tensor_profile_shape(name, 0)[1])  # opt profile
                self._ctx.set_input_shape(name, shape)
            spec = TensorSpec(name=name, np_dtype=np.dtype(dtype), shape=shape)
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            self._nbytes[name] = nbytes
            err, ptr = cudart.cudaMalloc(nbytes)
            _cuda_check(cudart, err, f"cudaMalloc({name})")
            self._dev[name] = int(ptr)
            self._ctx.set_tensor_address(name, int(ptr))
            if eng.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._inputs.append(spec)
            else:
                self._outputs.append(spec)
                self._host_out[name] = np.empty(shape, dtype=dtype)

    def infer(self, feeds_by_name: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """feeds_by_name maps *engine tensor names* -> contiguous ndarrays."""
        cudart = self._cudart
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost

        for spec in self._inputs:
            arr = np.ascontiguousarray(feeds_by_name[spec.name], dtype=spec.np_dtype)
            err, = cudart.cudaMemcpyAsync(
                self._dev[spec.name], arr.ctypes.data, self._nbytes[spec.name], H2D, self._stream
            )
            _cuda_check(cudart, err, f"H2D({spec.name})")

        if not self._ctx.execute_async_v3(self._stream):
            raise RuntimeError("execute_async_v3 returned False")

        for spec in self._outputs:
            host = self._host_out[spec.name]
            err, = cudart.cudaMemcpyAsync(
                host.ctypes.data, self._dev[spec.name], self._nbytes[spec.name], D2H, self._stream
            )
            _cuda_check(cudart, err, f"D2H({spec.name})")

        _cuda_check(cudart, cudart.cudaStreamSynchronize(self._stream)[0], "streamSync")
        return {name: arr.copy() for name, arr in self._host_out.items()}

    def close(self) -> None:
        cudart = self._cudart
        for ptr in self._dev.values():
            cudart.cudaFree(ptr)
        cudart.cudaStreamDestroy(self._stream)


class TRTBackend:
    """High-level pure-TRT backend: RGB + instruction (+ state) -> action chunk."""

    def __init__(self, engine_path: str, model_id: str, fixed_noise: bool = False):
        self._runner = TRTEngineRunner(engine_path)
        io = self._runner.io
        self._builder = InputBuilder(
            model_id=model_id,
            image_size=io.image_size,
            lang_max_len=io.lang_max_len,
            state_dim=io.state_dim,
            chunk_size=io.chunk_size,
            action_dim=io.action_dim,
            fixed_noise=fixed_noise,
        )
        self._role_to_name = io.role_to_name
        self._primary_output = io.primary_output
        self.description = (
            f"trt-engine path={engine_path.split('/')[-1]} "
            f"image={io.image_size} lang={io.lang_max_len} state={io.state_dim} "
            f"chunk={io.chunk_size} action={io.action_dim}"
        )

    def predict(self, image_rgb, instruction, state) -> PredictResult:
        logical = self._builder.build(image_rgb, instruction, state)
        feeds = {self._role_to_name[role]: arr for role, arr in logical.items()}
        t0 = time.perf_counter()
        out = self._runner.infer(feeds)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        actions = np.asarray(out[self._primary_output], dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return PredictResult(actions=actions, latency_ms=latency_ms, backend="trt-engine")
