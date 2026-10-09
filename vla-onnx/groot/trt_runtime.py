#!/usr/bin/env python3
"""TensorRT-only runtime for a GR00T split bundle on the Orin. System python3 + numpy.

No ONNX Runtime and no torch on purpose. ORT's TensorRT EP keeps every ONNX initializer
in host memory beside the engine's own copy, and on unified memory both count, which is
~5.5 bytes/param measured for X-VLA here. GR00T's ~2.3 B deployed params cannot afford
that. Engines hold their weights once; all execution contexts share ONE scratch buffer
sized to the largest engine, since the stages run strictly one after another.

    python3 trt_runtime.py build --bundle ~/bundles/groot-n16-split --cache ~/groot-trt
    python3 trt_runtime.py bench --bundle ~/bundles/groot-n16-split --cache ~/groot-trt
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import Bundle, infer  # noqa: E402

_cudart = ctypes.CDLL("libcudart.so.13")
_H2D, _D2H = 1, 2


def _ck(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed: cudaError {rc}")


def cuda_malloc(n: int) -> int:
    p = ctypes.c_void_p()
    _ck(_cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(max(n, 1))), f"cudaMalloc({n})")
    return p.value


def cuda_mem_info() -> tuple[int, int]:
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    _ck(_cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)), "cudaMemGetInfo")
    return free.value, total.value


def meminfo() -> dict:
    d = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":")
        d[k] = int(v.split()[0]) // 1024
    rss = int([l for l in Path("/proc/self/status").read_text().splitlines()
               if l.startswith("VmRSS")][0].split()[1]) // 1024
    return {"avail_mb": d["MemAvailable"], "used_mb": d["MemTotal"] - d["MemAvailable"],
            "swap_used_mb": d["SwapTotal"] - d["SwapFree"], "rss_mb": rss}


class MemWatch:
    """Samples system memory every 50 ms; keeps the floor of MemAvailable and peak swap."""

    def __init__(self):
        self.min_avail, self.max_swap = 1 << 30, 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop.is_set():
            m = meminfo()
            self.min_avail = min(self.min_avail, m["avail_mb"])
            self.max_swap = max(self.max_swap, m["swap_used_mb"])
            time.sleep(0.05)

    def stop(self):
        self._stop.set()
        self._t.join()
        return {"min_avail_mb": self.min_avail, "max_swap_used_mb": self.max_swap}


# --------------------------------------------------------------------------------------
# build


def build_one(onnx_path: Path, engine_path: Path, timing_cache: Path, opt_level: int,
              workspace_mb: int) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # Strongly typed: the mixed-FP16 ONNX already says which ops stay FP32 (norms,
    # softmax, RMSNorm's Pow/ReduceMean/Sqrt). Weakly typed + FP16 flag would let TRT
    # re-pick those in FP16, and a Qwen3 residual squared in FP16 overflows.
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise SystemExit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    cfg.builder_optimization_level = opt_level
    cache = cfg.create_timing_cache(timing_cache.read_bytes() if timing_cache.exists() else b"")
    cfg.set_timing_cache(cache, ignore_mismatch=False)
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        raise SystemExit(f"build failed: {onnx_path.name}")
    engine_path.write_bytes(blob)
    timing_cache.write_bytes(cfg.get_timing_cache().serialize())


def cmd_build(a):
    bundle = Bundle(a.bundle)
    cache = Path(a.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    log = []
    for name in bundle.names:
        eng = cache / f"{name}.engine"
        if eng.exists() and not a.force:
            continue
        watch = MemWatch()
        t0 = time.time()
        p = subprocess.run([sys.executable, __file__, "_build_one", str(bundle.root / f"{name}.onnx"),
                            str(eng), str(cache / "timing.cache"), str(a.opt_level), str(a.workspace_mb)])
        mem = watch.stop()
        peak_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024
        row = {"graph": name, "ok": p.returncode == 0, "s": round(time.time() - t0, 1),
               "engine_mb": round(eng.stat().st_size / 1e6, 1) if eng.exists() else None,
               "children_peak_rss_mb": peak_rss, **mem}
        print(json.dumps(row), flush=True)
        log.append(row)
        if p.returncode != 0:
            raise SystemExit(f"{name}: build failed")
    with open(cache / "build_log.jsonl", "a") as f:
        for row in log:
            f.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------------------
# run


class Engines:
    def __init__(self, bundle: Bundle, cache: str | Path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        cache = Path(cache).expanduser()
        self.engines, self.contexts = {}, {}
        for name in bundle.names:
            blob = (cache / f"{name}.engine").read_bytes()
            self.engines[name] = self.runtime.deserialize_cuda_engine(blob)
            del blob
        scratch = max(e.device_memory_size_v2 for e in self.engines.values())
        self.scratch = cuda_malloc(scratch)
        self.scratch_mb = scratch / 2**20
        for name, e in self.engines.items():
            ctx = e.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
            ctx.set_device_memory(self.scratch, scratch)
            self.contexts[name] = ctx
        self.stream = ctypes.c_void_p()
        _ck(_cudart.cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")
        # Device buffers shared by tensor name+shape across engines; an input whose host
        # array is the same object as the last upload (vl, biases, temb) is not re-copied.
        # The object itself is held, not its id(): a freed array's id can be reused.
        self.buffers: dict[tuple, int] = {}
        self.last_upload: dict[tuple, object] = {}
        self.io = {n: self._io(e) for n, e in self.engines.items()}

    def _io(self, e):
        out = []
        for i in range(e.num_io_tensors):
            n = e.get_tensor_name(i)
            shape = tuple(e.get_tensor_shape(n))
            dt = trt.nptype(e.get_tensor_dtype(n))
            mode = e.get_tensor_mode(n)
            key = (n if mode == trt.TensorIOMode.INPUT else "out:" + n, shape, np.dtype(dt).str)
            if key not in self.buffers:
                self.buffers[key] = cuda_malloc(int(np.prod(shape)) * np.dtype(dt).itemsize)
            out.append((n, mode, shape, np.dtype(dt), key))
        return out

    def run(self, name, feeds):
        ctx, outs = self.contexts[name], {}
        for n, mode, shape, dt, key in self.io[name]:
            ptr = self.buffers[key]
            ctx.set_tensor_address(n, ptr)
            if mode == trt.TensorIOMode.INPUT:
                x = feeds[n]
                if self.last_upload.get(key) is x:
                    continue
                x = np.ascontiguousarray(x, dtype=dt)
                assert x.shape == shape, (name, n, x.shape, shape)
                _ck(_cudart.cudaMemcpyAsync(ctypes.c_void_p(ptr), x.ctypes.data_as(ctypes.c_void_p),
                                            ctypes.c_size_t(x.nbytes), _H2D, self.stream), "H2D")
                self.last_upload[key] = feeds[n]
            else:
                outs[n] = (ptr, np.empty(shape, dt))
        if not ctx.execute_async_v3(self.stream.value):
            raise RuntimeError(f"{name}: execute failed")
        for n, (ptr, arr) in outs.items():
            _ck(_cudart.cudaMemcpyAsync(arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(ptr),
                                        ctypes.c_size_t(arr.nbytes), _D2H, self.stream), "D2H")
        _ck(_cudart.cudaStreamSynchronize(self.stream), "sync")
        return {n: arr for n, (ptr, arr) in outs.items()}


def cmd_bench(a):
    idle = meminfo()
    print("idle", json.dumps(idle), flush=True)
    bundle = Bundle(a.bundle)
    watch = MemWatch()
    t0 = time.time()
    eng = Engines(bundle, a.cache)
    loaded = meminfo()
    print(f"loaded {len(eng.engines)} engines in {time.time() - t0:.1f} s, "
          f"shared scratch {eng.scratch_mb:.0f} MB", json.dumps(loaded), flush=True)
    b = bundle.b
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (b["views"], *b["image_hw"], 3), dtype=np.uint8)
    pv = bundle.normalize(img)
    state = rng.standard_normal((1, 1, b["state_dim"])).astype(np.float32)
    lat, parts = [], []
    for i in range(a.warmup + a.iters):
        noise = rng.standard_normal((1, b["action_horizon"], b["action_dim"])).astype(np.float32)
        t = {}
        act = infer(bundle, eng.run, pv, state, noise, timings=t)
        if i >= a.warmup:
            lat.append(t["total_ms"])
            parts.append(t)
    mem = watch.stop()
    after = meminfo()
    lat = np.array(lat)
    res = {
        "engines": len(eng.engines), "views": b["views"], "seq_len": b["seq_len"],
        "p50_ms": round(float(np.percentile(lat, 50)), 1),
        "p95_ms": round(float(np.percentile(lat, 95)), 1),
        "hz": round(1000 / float(np.percentile(lat, 50)), 2),
        "vision_ms": round(float(np.median([p["vision_ms"] for p in parts])), 1),
        "backbone_ms": round(float(np.median([p["backbone_ms"] for p in parts])), 1),
        "dit_ms": round(float(np.median([p["dit_ms"] for p in parts])), 1),
        "finite": bool(np.isfinite(act).all()),
        "sys_used_idle_mb": idle["used_mb"], "sys_used_loaded_mb": loaded["used_mb"],
        "sys_used_after_mb": after["used_mb"], "rss_after_mb": after["rss_mb"], **mem,
    }
    print(json.dumps(res), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_build_one":
        build_one(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5]),
                  int(sys.argv[6]))
        return
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--bundle", required=True)
    b.add_argument("--cache", required=True)
    b.add_argument("--opt-level", type=int, default=3)
    b.add_argument("--workspace-mb", type=int, default=512)
    b.add_argument("--force", action="store_true")
    r = sub.add_parser("bench")
    r.add_argument("--bundle", required=True)
    r.add_argument("--cache", required=True)
    r.add_argument("--warmup", type=int, default=3)
    r.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    {"build": cmd_build, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    main()
