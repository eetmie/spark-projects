#!/usr/bin/env python3
"""Build a TensorRT `.engine` from a SmolVLA ONNX, on this Orin Nano.

The engine is hardware/version specific — build it here, never copy one from the
Spark. First build is slow (20-60+ min) and memory-sensitive on the 8 GB Nano;
once built it loads in ~1 s.

Examples
--------
    # Standard FP16 build:
    python build_engine.py --onnx exports/smolvla.onnx --engine exports/smolvla.engine

    # If FP16 NaNs out (parity check fails), keep sensitive layers in FP32:
    python build_engine.py --onnx ... --engine ... \
        --layer-precisions "*softmax*:fp32,*norm*:fp32"

    # If the build OOMs, shrink the workspace and lower the optimization level:
    python build_engine.py --onnx ... --engine ... --workspace-mib 2048 --opt-level 2

On an unsupported-op failure (hard abort with no clear cause), run the ORT
TensorRT-EP backend instead — it reports which subgraph falls back to CUDA.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time


def find_trtexec() -> str:
    for cand in (shutil.which("trtexec"), "/usr/src/tensorrt/bin/trtexec"):
        if cand and os.path.exists(cand):
            return cand
    sys.exit("trtexec not found. Add /usr/src/tensorrt/bin to PATH or install tensorrt.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="Input ONNX (with .onnx.data sidecar if split).")
    ap.add_argument("--engine", required=True, help="Output .engine path.")
    ap.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 (default).")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--workspace-mib", type=int, default=4096,
                    help="Builder workspace cap. Lower this if the build OOMs.")
    ap.add_argument("--opt-level", type=int, default=3, choices=range(0, 6),
                    help="builderOptimizationLevel; lower = less build memory/time.")
    ap.add_argument("--layer-precisions", default=None,
                    help="Comma list name_glob:type, e.g. '*softmax*:fp32'. Pins precision.")
    ap.add_argument("--timeout-s", type=int, default=7200)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Anything after --extra is passed straight to trtexec.")
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        sys.exit(f"ONNX not found: {args.onnx}")
    os.makedirs(os.path.dirname(os.path.abspath(args.engine)) or ".", exist_ok=True)

    trtexec = find_trtexec()
    cmd = [
        trtexec,
        f"--onnx={args.onnx}",
        f"--saveEngine={args.engine}",
        f"--memPoolSize=workspace:{args.workspace_mib}MiB",
        f"--builderOptimizationLevel={args.opt_level}",
    ]
    if args.fp16:
        cmd.append("--fp16")
    if args.layer_precisions:
        cmd += ["--precisionConstraints=obey", f"--layerPrecisions={args.layer_precisions}"]
    cmd += args.extra

    print("==> Building TensorRT engine (this is the slow, memory-sensitive step):")
    print("   ", " ".join(cmd))
    print("    Tip: build headless and close other GPU users; ~20-60 min on Orin Nano.\n")

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, timeout=args.timeout_s)
    dt = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"\n==> trtexec FAILED (rc={proc.returncode}) after {dt/60:.1f} min.", file=sys.stderr)
        print("    Likely causes: unsupported op (run the ORT TRT-EP backend to find it), "
              "or builder OOM (retry with --workspace-mib 2048 --opt-level 2).", file=sys.stderr)
        return proc.returncode

    size_mb = os.path.getsize(args.engine) / 1e6
    print(f"\n==> Engine built: {args.engine}  ({size_mb:.1f} MB) in {dt/60:.1f} min.")
    print("    Next: parity-check it against the ONNX before trusting actions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
