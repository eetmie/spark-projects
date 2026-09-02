#!/usr/bin/env python3
"""Prebuild the bootstrap EVO1 TensorRT engines one subprocess at a time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evo1_runtime.split_ort import prebuild_engines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("bundle"))
    parser.add_argument("--cache", type=Path, default=Path("trt_cache"))
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    results = prebuild_engines(args.bundle, args.cache, args.precision, args.only)
    print(json.dumps({"status": "PASS", "built": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
