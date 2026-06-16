#!/usr/bin/env python3
"""Probe whether ONE ONNX builds a TensorRT engine on this board, within 8 GB.

Generic across the SmolVLA split graphs (vision / text / expert_prefill / decode /
projectors). Creates an ORT TensorRT-EP session (same provider stack the runtime
deploys), feeds STATIC-shaped dummy inputs read off the graph, and runs once to
trigger + time the per-subgraph engine build. Reports build time, the active
providers, and whether a TRT engine actually got cached.

Build validation ONLY — inputs are dummy zeros, so outputs are meaningless. This
answers "does this graph build + cache a TRT engine on 8 GB" — the thing the
monolithic SmolVLA export could not do.

    python tools/build_probe.py exports/ainekko_base_split/smolvlm_expert_prefill.onnx
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smolvla_runtime.backends.ort import build_providers  # noqa: E402

_T = {
    "tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64,
    "tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(bool)": np.bool_,
    "tensor(uint8)": np.uint8,
}


def dummy(shape, tname):
    dt = _T.get(tname, np.float32)
    dims = [d if isinstance(d, int) and d > 0 else 1 for d in shape]  # sub 1 for dynamic
    if dt == np.bool_:
        return np.ones(dims, dtype=bool)
    if np.issubdtype(dt, np.integer):
        return np.ones(dims, dtype=dt)
    return np.zeros(dims, dtype=dt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--cache-dir", default="/tmp/smolvla_split_cache")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--runs", type=int, default=0,
                    help="After the build, time N steady-state inferences (engine loaded from cache).")
    args = ap.parse_args()

    import onnxruntime as ort

    p = Path(args.onnx)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"== build probe: {p.name} ({p.stat().st_size/1e6:.0f} MB), precision={args.precision}")

    sess = ort.InferenceSession(str(p), providers=build_providers(args.cache_dir, precision=args.precision))
    print("registered providers:", sess.get_providers())

    feeds = {}
    print("inputs:")
    for i in sess.get_inputs():
        feeds[i.name] = dummy(i.shape, i.type)
        print(f"  {i.name:<30} {i.type:<16} {i.shape}")
    print("outputs:")
    for o in sess.get_outputs():
        print(f"  {o.name:<30} {o.type:<16} {o.shape}")

    print("running first inference (this builds + caches the TRT engine)...")
    t0 = time.perf_counter()
    outs = sess.run(None, feeds)
    dt = time.perf_counter() - t0
    print(f"\nfirst run (build+infer): {dt:.1f}s")
    print("output shapes:", [tuple(np.asarray(o).shape) for o in outs])

    finite = all(bool(np.all(np.isfinite(np.asarray(o)))) for o in outs
                 if np.issubdtype(np.asarray(o).dtype, np.floating))
    print("outputs finite:", finite)

    if args.runs > 0:
        # Steady-state latency: the engine is built/cached now, so these are pure
        # inference. TRT-fast = ms; a silent CPU fallback would show up as 10-100x slower.
        lat = []
        for _ in range(args.runs):
            t = time.perf_counter()
            sess.run(None, feeds)
            lat.append((time.perf_counter() - t) * 1000.0)
        lat.sort()
        mean = sum(lat) / len(lat)
        p50 = lat[len(lat) // 2]
        p95 = lat[max(0, int(len(lat) * 0.95) - 1)]
        print(f"\ninference latency over {args.runs} runs (ms): "
              f"mean={mean:.2f}  p50={p50:.2f}  p95={p95:.2f}  min={lat[0]:.2f}")

    engines = list(Path(args.cache_dir).glob("*.engine"))
    built = len(engines) > 0
    print(f"\ncached engines in {args.cache_dir}: {len(engines)}")
    print(f"VERDICT: {'TRT ENGINE BUILT + CACHED ✅' if built else 'NO ENGINE CACHED ⚠️ (CPU fallback?)'}")
    return 0 if built else 2


if __name__ == "__main__":
    raise SystemExit(main())
