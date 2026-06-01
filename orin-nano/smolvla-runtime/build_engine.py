#!/usr/bin/env python3
"""Build a TensorRT `.engine` from a SmolVLA ONNX, on this Orin Nano.

The engine is hardware/version specific — build it here, never copy one from the
Spark. First build is slow (20-60+ min) and memory-sensitive on the 8 GB Nano;
once built it loads in ~1 s.

PRECISION — read this before you build
--------------------------------------
A precision sweep on the Spark (FP32-ONNX reference vs TRT engine, identical
inputs; see notes/findings.md) settled the choice:

    bf16  ->  cosine 0.9974   RECOMMENDED (default here)
    fp32  ->  cosine 0.99999  correct but ~3-4x slower than bf16
    fp16  ->  cosine 0.805    BROKEN — do not use for SmolVLA

FP16 collapses because the SmolVLM **vision tower** has constants that overflow
FP16's narrow exponent range (literal `inf` attention-mask values in every
self-attn layer). BF16 has FP32's exponent range, so it doesn't overflow, while
still using the tensor cores. Per-layer FP32 pinning via `--layer-precisions`
does NOT rescue FP16 here: TensorRT's myelin fusion erases the softmax/norm layer
names into `__myl_*` supernodes, so name globs match nothing. BF16 is the fix.

Examples
--------
    # Recommended build (BF16, static batch=1 profile auto-derived from the ONNX):
    python build_engine.py --onnx exports/smolvla.onnx --engine exports/smolvla.engine \
        --static-batch

    # Correct-but-slow fallback if BF16 parity is ever insufficient:
    python build_engine.py --onnx ... --engine ... --precision fp32 --static-batch

    # If the build OOMs on 8 GB, shrink workspace and lower the optimization level:
    python build_engine.py --onnx ... --engine ... --static-batch \
        --workspace-mib 2048 --opt-level 2

On an unsupported-op failure (hard abort with no clear cause), run the ORT
TensorRT-EP backend instead — it reports which subgraph falls back to CUDA.
(For this SmolVLA export the pure-TRT build is known to succeed — NonZero/ScatterND
from the vision-tower masked indexing are accepted by TRT 10.x.)
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


def static_batch_profile(onnx_path: str, batch: int = 1) -> str | None:
    """Build a min=opt=max shape profile string pinning the batch dim to `batch`.

    SmolVLA's ONNX has a dynamic batch dim; trtexec aborts on a dynamic input
    unless given an optimization profile. All other dims are static, so we read
    each input's shape from the ONNX and substitute `batch` for any dynamic dim.
    Returns a string like 'image0:1x3x512x512,img_mask0:1,...' or None if onnx
    isn't importable (caller then prints the manual --extra flags).
    """
    try:
        import onnx
    except ImportError:
        return None
    g = onnx.load(onnx_path, load_external_data=False).graph
    parts = []
    for inp in g.input:
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            dims.append(str(d.dim_value) if d.HasField("dim_value") and d.dim_value > 0
                        else str(batch))
        parts.append(f"{inp.name}:{'x'.join(dims)}")
    return ",".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="Input ONNX (with .onnx.data sidecar if split).")
    ap.add_argument("--engine", required=True, help="Output .engine path.")
    ap.add_argument("--precision", choices=("bf16", "fp32", "fp16"), default="bf16",
                    help="bf16 (default, recommended) | fp32 (correct, slow) | fp16 (BROKEN for SmolVLA).")
    ap.add_argument("--static-batch", action="store_true",
                    help="Auto-derive a batch=1 min/opt/max profile from the ONNX "
                         "(required — the SmolVLA graph has a dynamic batch dim).")
    ap.add_argument("--batch", type=int, default=1, help="Batch size for --static-batch profile.")
    ap.add_argument("--workspace-mib", type=int, default=4096,
                    help="Builder workspace cap. Lower this if the build OOMs.")
    ap.add_argument("--opt-level", type=int, default=3, choices=range(0, 6),
                    help="builderOptimizationLevel; lower = less build memory/time.")
    ap.add_argument("--layer-precisions", default=None,
                    help="Comma list name:type. NOTE: unreliable here — myelin fusion "
                         "erases names, and the wildcard is default-only, not substring.")
    ap.add_argument("--timeout-s", type=int, default=7200)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Anything after --extra is passed straight to trtexec.")
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        sys.exit(f"ONNX not found: {args.onnx}")
    os.makedirs(os.path.dirname(os.path.abspath(args.engine)) or ".", exist_ok=True)

    if args.precision == "fp16":
        print("WARNING: fp16 produces cosine ~0.805 on SmolVLA (vision-tower overflow). "
              "Use --precision bf16 unless you know what you're doing.\n", file=sys.stderr)

    trtexec = find_trtexec()
    cmd = [
        trtexec,
        f"--onnx={args.onnx}",
        f"--saveEngine={args.engine}",
        f"--memPoolSize=workspace:{args.workspace_mib}MiB",
        f"--builderOptimizationLevel={args.opt_level}",
    ]
    if args.precision == "bf16":
        cmd.append("--bf16")
    elif args.precision == "fp16":
        cmd.append("--fp16")
    # fp32: nothing (TF32 stays on by default)

    if args.static_batch:
        prof = static_batch_profile(args.onnx, args.batch)
        if prof is None:
            sys.exit("--static-batch needs the `onnx` package to read input shapes. "
                     "pip install onnx, or pass the profile yourself via --extra "
                     "--minShapes=... --optShapes=... --maxShapes=...")
        cmd += [f"--minShapes={prof}", f"--optShapes={prof}", f"--maxShapes={prof}"]

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
        print("    Likely causes: missing shape profile (use --static-batch), builder OOM "
              "(retry --workspace-mib 2048 --opt-level 2), or an unsupported op (run the "
              "ORT TRT-EP backend to find it).", file=sys.stderr)
        return proc.returncode

    size_mb = os.path.getsize(args.engine) / 1e6
    print(f"\n==> Engine built: {args.engine}  ({size_mb:.1f} MB) in {dt/60:.1f} min.")
    print("    Next: parity-check it against the ONNX before trusting actions:")
    print(f"      python parity.py --onnx {args.onnx} --engine {args.engine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
