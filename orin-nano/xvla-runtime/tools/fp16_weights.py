#!/usr/bin/env python3
"""Write mixed-FP16 copies of the split graphs, to halve what stays resident.

Different goal from SmolVLA's `--fp16-weights`, same mechanism. There it was aimed at the
*build*; the smolvla-runtime findings then measured that it does **not** help the build
(TRT imports weights as FP32 working copies regardless of file dtype) and "only halves the
deployed/loaded footprint". For X-VLA the build is already solved by splitting, and the
loaded footprint is exactly the problem: 12 sessions sit at ~6.7 GB on a 7.4 GB board with
the camera and control loop still to come.

Precision recipe is copied deliberately rather than reinvented, because the Orin has
already burned this once: a *blanket* FP16 cast overflowed SmolVLA's vision tower
(cos 0.805). Keeping `LayerNormalization` and `Softmax` in FP32 mirrors the
`trt_layer_norm_fp32_fallback` the runtime already sets, and `keep_io_types` leaves graph
inputs/outputs FP32 so the runtime feeds nothing differently.

BF16 is not an option here: on Orin (compute 8.7) `platform_has_fast_bf16` is n/a — no
hardware fast path — which is why the whole stack is FP16.

    python tools/fp16_weights.py --split-dir exports/split --out-dir exports/split_fp16
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The two ops a blanket FP16 cast overflowed on this stack; kept FP32.
FP16_SENSITIVE_OPS = ["LayerNormalization", "Softmax"]


CHILD = r'''
import sys
from pathlib import Path
import onnx
from onnxruntime.transformers import float16
from onnxruntime.transformers.onnx_model import OnnxModel

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
sensitive = sys.argv[3].split(",") if sys.argv[3] else []

model = onnx.load(str(src))
block = list(float16.DEFAULT_OP_BLOCK_LIST) + sensitive
fp16 = float16.convert_float_to_float16(
    model, keep_io_types=True, op_block_list=block
)
# keep_io_types inserts cast nodes without re-sorting; onnx.checker demands
# topological order even though ORT tolerates it.
om = OnnxModel(fp16)
om.topological_sort()
onnx.save(om.model, str(dst))
print(f"OK {dst.stat().st_size / 1e6:.1f}")
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", type=Path, default=REPO / "exports" / "split")
    ap.add_argument("--out-dir", type=Path, default=REPO / "exports" / "split_fp16")
    ap.add_argument("--only", nargs="*", default=None,
                    help="graph names to convert; others are copied as-is")
    args = ap.parse_args()

    bundle = json.loads((args.split_dir / "bundle.json").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    child = args.out_dir / "_convert_one.py"
    child.write_text(CHILD)
    sensitive = ",".join(FP16_SENSITIVE_OPS)

    total_src = total_dst = 0.0
    for graph in bundle["graphs"]:
        src = args.split_dir / graph["file"]
        dst = args.out_dir / graph["file"]
        if args.only and graph["name"] not in args.only:
            shutil.copy2(src, dst)
            print(f"  {graph['name']:16s} copied (fp32)")
            continue
        # One subprocess per graph: the converter holds the FP32 model and the FP16
        # copy at once, and these are up to 412 MB each.
        proc = subprocess.run(
            [sys.executable, str(child), str(src), str(dst), sensitive],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-8:]
            sys.exit(f"fp16 conversion failed for {graph['name']}:\n" + "\n".join(tail))
        src_mb = src.stat().st_size / 1e6
        dst_mb = dst.stat().st_size / 1e6
        total_src += src_mb
        total_dst += dst_mb
        print(f"  {graph['name']:16s} {src_mb:6.0f} MB -> {dst_mb:6.0f} MB "
              f"({dst_mb / src_mb:.2f}x)")

    child.unlink(missing_ok=True)
    (args.out_dir / "bundle.json").write_text(json.dumps(bundle, indent=2))
    print(f"\nconverted {total_src:.0f} MB -> {total_dst:.0f} MB "
          f"({total_dst / total_src:.2f}x)" if total_src else "")
    print(f"wrote {args.out_dir}")
    print("\nNext: rebuild engines against this dir, then re-run parity.py -- an FP16 "
          "weight cast is exactly the kind of change parity exists to catch.")


if __name__ == "__main__":
    main()
