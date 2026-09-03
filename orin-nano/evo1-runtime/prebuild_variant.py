#!/usr/bin/env python3
"""Build one experimental ONNX graph in an isolated TensorRT subprocess."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from evo1_runtime.split_ort import _BUILD_ONE, _MemorySampler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()
    sampler = _MemorySampler()
    sampler.start()
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            _BUILD_ONE,
            str(args.graph.resolve()),
            str(args.cache.resolve()),
            args.precision,
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ, TRT_DROP_CUDA_EP="1"),
    )
    memory = sampler.finish()
    if process.returncode:
        print(process.stdout, end="")
        print(process.stderr, end="", file=sys.stderr)
        raise SystemExit(process.returncode)
    line = next(
        value for value in process.stdout.splitlines() if value.startswith("BUILD_RESULT ")
    )
    result = json.loads(line.removeprefix("BUILD_RESULT "))
    result["memory"] = memory
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
