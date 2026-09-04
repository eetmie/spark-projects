#!/usr/bin/env python3
"""Export a fused EVO1 action-step + output graph for measured Orin testing."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from export_split_onnx import _build_policy, _build_wrappers, _validate_onnx
from fp16_weights import CHILD as FP16_CHILD

ROOT = Path(__file__).resolve().parent


def manifest(directory: Path) -> None:
    lines = []
    for path in sorted(item for item in directory.iterdir() if item.is_file()):
        if path.name == "MANIFEST.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (directory / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=ROOT / "models" / "InternVL3-1B-hf"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "exports" / "perf-action-hot"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=320)
    args = parser.parse_args()

    import torch
    from torch import nn

    policy = _build_policy(args.base.resolve(), args.seed, args.seq_len)
    *_, ActionContext, ActionStep, ActionOutput = _build_wrappers()
    head = policy.model.action_head

    class ActionHot(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.step = ActionStep(head)
            self.output = ActionOutput(head)

        def forward(self, action, time_index, key_mask, *key_values):
            hidden = self.step(action, time_index, key_mask, *key_values)
            return self.output(hidden)

    context = ActionContext(head).eval().float()
    with torch.no_grad():
        cached = context(
            torch.zeros(1, args.seq_len, 896),
            torch.ones(1, args.seq_len, dtype=torch.bool),
            torch.zeros(1, 24),
        )
    action = torch.zeros(1, 50, 24)
    time_index = torch.zeros(1, dtype=torch.long)
    input_names = ["action", "time_index", "key_mask"]
    for index in range(8):
        input_names.extend((f"key_{index}", f"value_{index}"))

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / ".action_hot_fp32.onnx"
    target = out / "action_hot.onnx"
    hot = ActionHot().eval().float()
    with torch.no_grad():
        torch.onnx.export(
            hot,
            (action, time_index, *cached),
            str(temporary),
            input_names=input_names,
            output_names=["velocity"],
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
    _validate_onnx(temporary)
    conversion = subprocess.run(
        [sys.executable, "-c", FP16_CHILD, str(temporary), str(target)],
        capture_output=True,
        text=True,
    )
    if conversion.returncode:
        raise SystemExit(conversion.stderr or conversion.stdout)
    temporary.unlink()
    _validate_onnx(target)

    state_elements = sum(tensor.numel() for tensor in hot.state_dict().values())
    record = {
        "schema_version": 1,
        "variant": "fused_action_hot",
        "deployable": False,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "state_elements": state_elements,
        "weight_precision": "mixed_fp16_keep_io_fp32",
        "file": target.name,
        "size": target.stat().st_size,
        "inputs": input_names,
        "outputs": ["velocity"],
    }
    (out / "variant.json").write_text(json.dumps(record, indent=2) + "\n")
    manifest(out)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
