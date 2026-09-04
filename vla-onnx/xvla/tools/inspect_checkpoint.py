#!/usr/bin/env python3
"""Parameter accounting for an X-VLA safetensors checkpoint, straight from the header.

safetensors stores its JSON index at the head of the file, so this reports exact
per-component parameter counts without loading (or even fully downloading) weights.
The point is to size each prospective TensorRT engine BEFORE building anything: on
this 8 GB board the build peak tracks the FP32 weight slice a single engine carries.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from vla_common.safetensors_header import read_header

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}

# Prospective engine boundaries: (label, predicate on the tensor key).
# Order matters -- first match wins.
COMPONENTS: list[tuple[str, str]] = [
    ("vlm.vision_tower (DaViT)", "model.vlm.vision_tower."),
    ("vlm.multi_modal_projector", "model.vlm.multi_modal_projector."),
    ("vlm.language_model.shared/embed", "model.vlm.language_model.shared."),
    ("vlm.language_model.encoder", "model.vlm.language_model.encoder."),
    ("policy transformer blocks", "model.transformer.blocks."),
    ("policy transformer io/proj", "model.transformer."),
]



def classify(key: str) -> str:
    for label, prefix in COMPONENTS:
        if key.startswith(prefix):
            return label
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--detail", action="store_true", help="also break the vision tower down by stage")
    args = ap.parse_args()

    header = read_header(args.checkpoint)
    header.pop("__metadata__", None)

    params: dict[str, int] = defaultdict(int)
    on_disk: dict[str, int] = defaultdict(int)
    dtypes: dict[str, set[str]] = defaultdict(set)
    stages: dict[str, int] = defaultdict(int)

    for key, meta in header.items():
        n = 1
        for d in meta["shape"]:
            n *= d
        label = classify(key)
        params[label] += n
        on_disk[label] += n * DTYPE_BYTES.get(meta["dtype"], 4)
        dtypes[label].add(meta["dtype"])
        if key.startswith("model.vlm.vision_tower.blocks."):
            stages[f"  vision stage {key.split('.')[4]}"] += n
        elif key.startswith("model.vlm.vision_tower.convs."):
            stages[f"  vision conv-embed {key.split('.')[4]}"] += n

    total = sum(params.values())
    order = [lbl for lbl, _ in COMPONENTS if lbl in params] + (["other"] if "other" in params else [])

    print(f"\n{args.checkpoint}")
    print(f"{'component':34s} {'params':>12s} {'FP32':>9s} {'FP16':>9s}  dtype")
    print("-" * 78)
    for label in order:
        n = params[label]
        print(
            f"{label:34s} {n:12,d} {n * 4 / 1e9:8.2f}G {n * 2 / 1e9:8.2f}G  "
            f"{','.join(sorted(dtypes[label]))}"
        )
        if args.detail and label.startswith("vlm.vision_tower"):
            for skey in sorted(stages):
                sn = stages[skey]
                print(f"{skey:34s} {sn:12,d} {sn * 4 / 1e9:8.2f}G {sn * 2 / 1e9:8.2f}G")
    print("-" * 78)
    print(f"{'TOTAL':34s} {total:12,d} {total * 4 / 1e9:8.2f}G {total * 2 / 1e9:8.2f}G")
    print(f"\non-disk checkpoint: {sum(on_disk.values()) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
