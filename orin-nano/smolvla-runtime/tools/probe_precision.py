#!/usr/bin/env python3
"""Print what reduced-precision math TensorRT can actually *accelerate* on this board.

This is the empirical basis for the FP16-default / BF16-experimental decision in
the runtime. On the Orin Nano (compute capability 8.7, JetPack 7.2, TensorRT 10.16)
the result is:

    platform_has_fast_fp16 = True      <- deploy here
    platform_has_fast_int8 = True
    platform_has_fast_bf16 = n/a       <- NOT a hardware fast path on 8.7

BF16 is SmolVLA's reference dtype numerically, but it is not accelerated on Orin,
so the deployment path uses FP16 (with FP32 fallback for sensitive ops). Re-run
this on any new image before trusting BF16 as an accelerated mode.

    python tools/probe_precision.py
"""
from __future__ import annotations


def main() -> int:
    try:
        import tensorrt as trt
    except Exception as exc:  # noqa: BLE001
        print(f"tensorrt import failed: {exc!r}")
        print("(venv must be created with --system-site-packages to see JetPack's tensorrt)")
        return 1

    print(f"TensorRT: {trt.__version__}")
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    for attr in (
        "platform_has_fast_fp16",
        "platform_has_fast_int8",
        "platform_has_tf32",
        "platform_has_fast_bf16",
        "platform_has_bf16",
    ):
        print(f"  {attr:<26} {getattr(builder, attr, 'n/a')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
