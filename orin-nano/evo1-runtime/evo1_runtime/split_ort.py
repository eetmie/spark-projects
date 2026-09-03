"""Run the fixed-shape EVO1 split bundle with ONNX Runtime + TensorRT EP.

This first-stage runtime intentionally accepts only the deterministic bootstrap
bundle.  Its action head is randomly initialized, so the only supported entry
point consumes the bundled parity fixture; it is not a robot-control backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

_BOOTSTRAP_WARNING = (
    "EVO1 bootstrap bundle: the action head is deterministic random initialization; "
    "never use these actions to control a robot."
)


def verify_bundle(bundle_dir: str | Path) -> dict:
    """Verify every checksummed artifact and the non-deployable bundle contract."""
    root = Path(bundle_dir).resolve()
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        raise ValueError(f"bundle manifest is missing: {manifest}")
    checked = set()
    for line in manifest.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"manifest path escapes bundle: {relative}")
        if not path.is_file():
            raise ValueError(f"manifest artifact is missing: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"bundle identity mismatch: {relative}")
        checked.add(relative)

    bundle = json.loads((root / "bundle.json").read_text())
    required = {"bundle.json", bundle["fixture"]["file"]}
    required.update(graph["file"] for graph in bundle["graphs"])
    missing = sorted(required - checked)
    if missing:
        raise ValueError(f"bundle artifacts absent from manifest: {missing}")
    if bundle.get("model") != "evo1" or bundle.get("schema_version") != 1:
        raise ValueError("unsupported EVO1 bundle schema")
    if bundle.get("deployable") or not bundle.get("random_action_head"):
        raise ValueError("initial runtime accepts only a marked bootstrap bundle")
    if bundle.get("max_views") != 1 or bundle.get("valid_views") != 1:
        raise ValueError("initial runtime requires the one-camera export profile")
    return bundle


def make_session_options() -> "ort.SessionOptions":
    """Disable pre-EP fusions that manufacture unsupported FP16 contrib ops."""
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_cpu_mem_arena = False
    return options


def build_providers(cache_dir: str | Path, precision: str = "fp16") -> list:
    """Return the same TensorRT -> CUDA -> CPU stack used by X-VLA."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    available = set(ort.get_available_providers())
    providers: list = []
    if "TensorrtExecutionProvider" in available:
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_fp16_enable": precision == "fp16",
                    "trt_bf16_enable": precision == "bf16",
                    "trt_layer_norm_fp32_fallback": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache),
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": str(cache),
                    "trt_max_workspace_size": int(
                        os.environ.get("TRT_WORKSPACE_MB", "512")
                    )
                    * (1 << 20),
                    "trt_builder_optimization_level": int(
                        os.environ.get("TRT_OPT_LEVEL", "2")
                    ),
                    "trt_min_subgraph_size": 5,
                },
            )
        )
    if "CUDAExecutionProvider" in available and not os.environ.get("TRT_DROP_CUDA_EP"):
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "gpu_mem_limit": 3 << 30,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "do_copy_in_default_stream": True,
                },
            )
        )
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if not providers:
        raise RuntimeError(f"no usable ONNX Runtime provider in {sorted(available)}")
    return providers


def _cpu_session(path: Path) -> "ort.InferenceSession":
    return ort.InferenceSession(
        str(path),
        sess_options=make_session_options(),
        providers=["CPUExecutionProvider"],
    )


class Evo1SplitPolicy:
    """Fixed one-camera/320-token inference for bootstrap parity testing only."""

    def __init__(
        self,
        bundle_dir: str | Path,
        cache_dir: str | Path,
        precision: str = "fp16",
        embedding_device: str = "cpu",
        action_hot: str | Path | None = None,
        device_resident_action: bool = False,
        *,
        allow_bootstrap: bool = False,
    ) -> None:
        if not allow_bootstrap:
            raise ValueError(_BOOTSTRAP_WARNING + " Pass allow_bootstrap=True for parity testing.")
        self.root = Path(bundle_dir).resolve()
        self.bundle = verify_bundle(self.root)
        self.cache_dir = Path(cache_dir).resolve()
        self.precision = precision
        if embedding_device not in {"cpu", "cuda"}:
            raise ValueError("embedding_device must be 'cpu' or 'cuda'")
        self.embedding_device = embedding_device
        self.action_hot_path = Path(action_hot).resolve() if action_hot else None
        self.device_resident_action = bool(device_resident_action)
        self.graphs = {graph["name"]: graph for graph in self.bundle["graphs"]}
        self.load_timings_s: dict[str, float] = {}
        self.sessions: dict[str, ort.InferenceSession] = {}

        self._load(
            "token_embedding",
            cpu_only=embedding_device == "cpu",
            cuda_only=embedding_device == "cuda",
        )
        for name in (
            "vision_0",
            "vision_1",
            "vision_2",
            "vision_3",
            "language_0",
            "language_1",
            "language_2",
            "action_context",
        ):
            self._load(name, cpu_only=False)
        if self.action_hot_path:
            self._load(
                "action_hot", cpu_only=False, onnx_path=self.action_hot_path
            )
        else:
            self._load("action_step", cpu_only=False)
            self._load("action_output", cpu_only=False)
        if self.device_resident_action:
            action_names = (
                ("action_hot",)
                if self.action_hot_path
                else ("action_step", "action_output")
            )
            gpu = {"TensorrtExecutionProvider", "CUDAExecutionProvider"}
            if not all(
                gpu.intersection(self.sessions[name].get_providers())
                for name in action_names
            ):
                raise ValueError("device-resident action requires GPU action sessions")
            self._action_io = {
                name: self.sessions[name].io_binding() for name in action_names
            }

    def _load(
        self,
        name: str,
        *,
        cpu_only: bool,
        cuda_only: bool = False,
        onnx_path: str | Path | None = None,
    ) -> None:
        path = (
            Path(onnx_path).resolve()
            if onnx_path
            else self.root / self.graphs[name]["file"]
        )
        started = time.perf_counter()
        if cpu_only:
            session = _cpu_session(path)
        elif cuda_only:
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDA embedding requested but CUDA EP is unavailable")
            session = ort.InferenceSession(
                str(path),
                sess_options=make_session_options(),
                providers=[
                    ("CUDAExecutionProvider", {"device_id": 0}),
                    "CPUExecutionProvider",
                ],
            )
        else:
            session = ort.InferenceSession(
                str(path),
                sess_options=make_session_options(),
                providers=build_providers(self.cache_dir, self.precision),
            )
        self.sessions[name] = session
        self.load_timings_s[name] = time.perf_counter() - started

    @property
    def warning(self) -> str:
        return _BOOTSTRAP_WARNING

    def _chain(self, names: tuple[str, ...], feed: dict) -> np.ndarray:
        value = None
        for index, name in enumerate(names):
            if index:
                inputs = self.sessions[name].get_inputs()
                accepted = {item.name for item in inputs}
                feed = {key: item for key, item in feed.items() if key in accepted}
                feed[inputs[0].name] = value
            value = self.sessions[name].run(None, feed)[0]
        assert value is not None
        return value

    def run_fixture(
        self, fixture: "np.lib.npyio.NpzFile", steps: int | None = None
    ) -> dict:
        """Run the exact native-reference inputs through all eleven split graphs."""
        timings: dict[str, float] = {}

        started = time.perf_counter()
        image_features = self._chain(
            ("vision_0", "vision_1", "vision_2", "vision_3"),
            {"pixel_values": np.asarray(fixture["pixel_values"], dtype=np.float32)},
        )
        timings["vision"] = time.perf_counter() - started

        started = time.perf_counter()
        input_ids = np.asarray(fixture["input_ids"], dtype=np.int64)
        merged = self.sessions["token_embedding"].run(
            None, {"input_ids": input_ids}
        )[0].copy()
        positions = np.flatnonzero(
            input_ids.reshape(-1) == int(self.bundle["image_token_id"])
        )
        expected = int(self.bundle["image_seq_length"])
        if len(positions) != expected:
            raise ValueError(f"fixture has {len(positions)} image tokens, expected {expected}")
        merged.reshape(-1, int(self.bundle["hidden_size"]))[positions] = (
            image_features.reshape(-1, int(self.bundle["hidden_size"]))
        )
        fused_tokens = self._chain(
            ("language_0", "language_1", "language_2"),
            {
                "hidden_in": merged,
                "causal_mask": np.asarray(fixture["causal_mask"], dtype=np.float32),
            },
        )
        timings[f"language_with_{self.embedding_device}_embedding"] = (
            time.perf_counter() - started
        )

        started = time.perf_counter()
        cached = self.sessions["action_context"].run(
            None,
            {
                "fused_tokens": fused_tokens,
                "context_mask": np.asarray(fixture["context_mask"], dtype=bool),
                "state": np.asarray(fixture["state"], dtype=np.float32),
            },
        )
        timings["action_context"] = time.perf_counter() - started

        action = np.asarray(fixture["initial_noise"], dtype=np.float32).copy()
        steps = int(self.bundle["num_inference_timesteps"] if steps is None else steps)
        if steps <= 0:
            raise ValueError("steps must be positive")
        hot_name = "action_hot" if "action_hot" in self.sessions else "action_step"
        step_inputs = [item.name for item in self.sessions[hot_name].get_inputs()]
        cached_names = step_inputs[2:]
        if len(cached_names) != len(cached):
            raise ValueError("action cache graph contract does not match action step")
        if self.device_resident_action:
            uploaded = time.perf_counter()
            cached_device = {
                name: ort.OrtValue.ortvalue_from_numpy(
                    np.ascontiguousarray(value), "cuda", 0
                )
                for name, value in zip(cached_names, cached, strict=True)
            }
            timings["action_cache_upload"] = time.perf_counter() - uploaded
            cached_feed = None
        else:
            cached_device = None
            cached_feed = dict(zip(cached_names, cached, strict=True))

        step_total = 0.0
        output_total = 0.0
        for index in range(steps):
            time_index = np.asarray(
                [min(int((index / steps) * 999), 999)], dtype=np.int64
            )
            started = time.perf_counter()
            if self.device_resident_action:
                io = self._action_io[hot_name]
                io.clear_binding_inputs()
                io.clear_binding_outputs()
                io.bind_cpu_input("action", np.ascontiguousarray(action))
                io.bind_cpu_input("time_index", time_index)
                for name, value in cached_device.items():
                    io.bind_ortvalue_input(name, value)
                output_device = "cpu" if hot_name == "action_hot" else "cuda"
                io.bind_output(
                    self.sessions[hot_name].get_outputs()[0].name, output_device, 0
                )
                self.sessions[hot_name].run_with_iobinding(io)
                first_output = io.get_outputs()[0]
                step_total += time.perf_counter() - started
                if hot_name == "action_hot":
                    velocity = first_output.numpy()
                else:
                    started = time.perf_counter()
                    output_io = self._action_io["action_output"]
                    output_io.clear_binding_inputs()
                    output_io.clear_binding_outputs()
                    output_io.bind_ortvalue_input("action_hidden", first_output)
                    output_io.bind_output("velocity", "cpu", 0)
                    self.sessions["action_output"].run_with_iobinding(output_io)
                    velocity = output_io.get_outputs()[0].numpy()
                    output_total += time.perf_counter() - started
            elif hot_name == "action_hot":
                velocity = self.sessions[hot_name].run(
                    None,
                    {"action": action, "time_index": time_index, **cached_feed},
                )[0]
                step_total += time.perf_counter() - started
            else:
                action_hidden = self.sessions[hot_name].run(
                    None,
                    {"action": action, "time_index": time_index, **cached_feed},
                )[0]
                step_total += time.perf_counter() - started
                started = time.perf_counter()
                velocity = self.sessions["action_output"].run(
                    None, {"action_hidden": action_hidden}
                )[0]
                output_total += time.perf_counter() - started
            action += velocity / steps
        timings[f"{hot_name}_x{steps}"] = step_total
        if hot_name == "action_step":
            timings[f"action_output_x{steps}"] = output_total
        timings["total"] = sum(timings.values())
        return {
            "vision": image_features,
            "fused": fused_tokens,
            "action": action,
            "timings_s": timings,
        }


def _mem_available_kb() -> int:
    with open("/proc/meminfo") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    return -1


class _MemorySampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.baseline_kb = _mem_available_kb()
        self.minimum_kb = self.baseline_kb
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.wait(0.2):
            self.minimum_kb = min(self.minimum_kb, _mem_available_kb())

    def finish(self) -> dict:
        self._halt.set()
        self.join(timeout=2)
        return {
            "baseline_available_gb": round(self.baseline_kb / 1e6, 3),
            "minimum_available_gb": round(self.minimum_kb / 1e6, 3),
            "consumed_gb": round((self.baseline_kb - self.minimum_kb) / 1e6, 3),
        }


_BUILD_ONE = r'''
import json, os, resource, sys, time
from pathlib import Path
import numpy as np
import onnxruntime as ort

graph, cache, precision = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
cache.mkdir(parents=True, exist_ok=True)
options = ort.SessionOptions()
options.log_severity_level = 3
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.enable_cpu_mem_arena = False
trt = {
    "device_id": 0,
    "trt_fp16_enable": precision == "fp16",
    "trt_bf16_enable": precision == "bf16",
    "trt_layer_norm_fp32_fallback": True,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": str(cache),
    "trt_timing_cache_enable": True,
    "trt_timing_cache_path": str(cache),
    "trt_max_workspace_size": int(os.environ.get("TRT_WORKSPACE_MB", "512")) * (1 << 20),
    "trt_builder_optimization_level": int(os.environ.get("TRT_OPT_LEVEL", "2")),
    "trt_min_subgraph_size": 5,
}
providers = [("TensorrtExecutionProvider", trt), "CPUExecutionProvider"]
started = time.perf_counter()
session = ort.InferenceSession(str(graph), sess_options=options, providers=providers)
load_s = time.perf_counter() - started
types = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(bool)": bool,
}
feed = {}
for item in session.get_inputs():
    if any(not isinstance(dim, int) for dim in item.shape):
        raise ValueError(f"dynamic build input is unsupported: {item.name} {item.shape}")
    feed[item.name] = np.zeros(item.shape, dtype=types[item.type])
started = time.perf_counter()
session.run(None, feed)
run_s = time.perf_counter() - started
print("BUILD_RESULT " + json.dumps({
    "graph": graph.stem,
    "load_s": round(load_s, 3),
    "first_run_s": round(run_s, 3),
    "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3),
    "providers": session.get_providers(),
}))
'''


def prebuild_engines(
    bundle_dir: str | Path,
    cache_dir: str | Path,
    precision: str = "fp16",
    only: list[str] | None = None,
) -> list[dict]:
    """Build each TRT engine in a fresh subprocess and report unified-memory peaks."""
    root = Path(bundle_dir).resolve()
    bundle = verify_bundle(root)
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    results = []
    for graph in bundle["graphs"]:
        name = graph["name"]
        if name == "token_embedding" or (only and name not in only):
            continue
        print(f"prebuild {name} ({graph['size_mb']} MB)...", flush=True)
        sampler = _MemorySampler()
        sampler.start()
        environment = dict(os.environ, TRT_DROP_CUDA_EP="1")
        process = subprocess.run(
            [sys.executable, "-c", _BUILD_ONE, str(root / graph["file"]), str(cache), precision],
            capture_output=True,
            text=True,
            env=environment,
        )
        memory = sampler.finish()
        if process.returncode:
            print(process.stdout, end="")
            print(process.stderr, end="", file=sys.stderr)
            raise RuntimeError(f"TensorRT prebuild failed for {name}")
        line = next(
            value for value in process.stdout.splitlines() if value.startswith("BUILD_RESULT ")
        )
        result = json.loads(line.removeprefix("BUILD_RESULT "))
        result["memory"] = memory
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    engines = sorted(cache.glob("*.engine"))
    if not engines:
        raise RuntimeError("TensorRT prebuild produced no engine files")
    summary = {
        "precision": precision,
        "graphs": results,
        "engine_files": [
            {"name": path.name, "size": path.stat().st_size} for path in engines
        ],
    }
    (cache / "evo1_engine_cache.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return results


def process_memory() -> dict:
    return {
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3),
        "available_gb": round(_mem_available_kb() / 1e6, 3),
    }
