#!/usr/bin/env python3
"""Create a mixed-FP16 EVO1 split bundle for Orin resident-memory testing."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from export_split_onnx import write_manifest

ROOT = Path(__file__).resolve().parent
SENSITIVE_OPS = ("LayerNormalization", "Softmax")

CHILD = r'''
import sys
from pathlib import Path

import onnx
from onnxruntime.transformers import float16
from onnxruntime.transformers.onnx_model import OnnxModel

source, target = Path(sys.argv[1]), Path(sys.argv[2])
model = onnx.load(str(source))
blocked = list(float16.DEFAULT_OP_BLOCK_LIST) + ["LayerNormalization", "Softmax"]
converted = float16.convert_float_to_float16(
    model,
    keep_io_types=True,
    op_block_list=blocked,
)
wrapped = OnnxModel(converted)
wrapped.topological_sort()
onnx.checker.check_model(wrapped.model)
onnx.save(wrapped.model, str(target))
print(target.stat().st_size)
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=ROOT / "exports" / "split-bootstrap",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "exports" / "split-bootstrap-fp16",
    )
    args = parser.parse_args()

    source = args.split_dir.resolve()
    target = args.out_dir.resolve()
    bundle = json.loads((source / "bundle.json").read_text())
    if bundle.get("deployable") or not bundle.get("random_action_head"):
        raise ValueError("bootstrap converter refuses an unrecognized bundle")

    target.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer",):
        destination = target / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source / name, destination)
    fixture = bundle.get("fixture") or {}
    if fixture.get("file"):
        shutil.copy2(source / fixture["file"], target / fixture["file"])

    child = target / "_convert_one.py"
    child.write_text(CHILD)
    total_source = 0
    total_target = 0
    for graph in bundle["graphs"]:
        source_graph = source / graph["file"]
        target_graph = target / graph["file"]
        source_size = source_graph.stat().st_size
        if graph["name"] == "token_embedding":
            shutil.copy2(source_graph, target_graph)
            mode = "FP32 CPU"
        else:
            process = subprocess.run(
                [sys.executable, str(child), str(source_graph), str(target_graph)],
                capture_output=True,
                text=True,
            )
            if process.returncode:
                tail = (process.stderr or process.stdout).splitlines()[-12:]
                raise SystemExit(
                    f"conversion failed for {graph['name']}:\n" + "\n".join(tail)
                )
            mode = "mixed FP16"
        target_size = target_graph.stat().st_size
        total_source += source_size
        total_target += target_size
        graph["size_mb"] = round(target_size / 1e6, 1)
        graph["weight_precision"] = mode
        print(
            f"{graph['name']:18s} {source_size / 1e6:8.1f} MB -> "
            f"{target_size / 1e6:8.1f} MB  {mode}",
            flush=True,
        )
    child.unlink()

    bundle["weight_precision"] = "mixed_fp16_keep_io_fp32"
    bundle["provenance"]["fp16_sensitive_ops"] = list(SENSITIVE_OPS)
    (target / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    count = write_manifest(target)
    print(
        f"\nconverted {total_source / 1e9:.3f} GB -> "
        f"{total_target / 1e9:.3f} GB"
    )
    print(f"wrote {target / 'MANIFEST.sha256'} ({count} files)")


if __name__ == "__main__":
    main()
