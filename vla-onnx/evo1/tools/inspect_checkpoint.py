#!/usr/bin/env python3
"""Report exact EVO1 serialized tensor sizes and a first Orin split-engine profile.

The input may be a complete safetensors checkpoint or a file containing only its
header range. No tensors are loaded. Both the raw InternVL3 base key layout and a
LeRobot EVO1 policy checkpoint are supported.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "F32": 4,
    "I64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    elements: int


def read_header(path: Path) -> list[TensorInfo]:
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"{path} is too short to contain a safetensors header")
        (header_size,) = struct.unpack("<Q", raw_size)
        raw_header = handle.read(header_size)
    if len(raw_header) != header_size:
        raise ValueError(
            f"{path} contains {len(raw_header):,} of {header_size:,} required header bytes"
        )
    header = json.loads(raw_header)
    tensors = []
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        shape = tuple(metadata["shape"])
        elements = 1
        for dimension in shape:
            elements *= dimension
        tensors.append(TensorInfo(name, shape, metadata["dtype"], elements))
    return tensors


def canonical_name(name: str) -> str:
    for prefix in ("model.embedder.model.", "embedder.model."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    for prefix in ("model.action_head.", "action_head."):
        if name.startswith(prefix):
            return "action_head." + name[len(prefix) :]
    return name


def component(name: str) -> str:
    name = canonical_name(name)
    match = re.match(r"vision_tower\.encoder\.layer\.(\d+)\.", name)
    if match:
        return f"vision.layer.{int(match.group(1)):02d}"
    if name.startswith("vision_tower."):
        return "vision.embeddings"
    if name.startswith("multi_modal_projector."):
        return "vision.projector"
    if name.startswith("language_model.embed_tokens."):
        return "language.token_embedding"
    match = re.match(r"language_model\.layers\.(\d+)\.", name)
    if match:
        return f"language.layer.{int(match.group(1)):02d}"
    if name.startswith("language_model.norm."):
        return "language.norm"
    if name.startswith("language_model.lm_head."):
        return "language.lm_head_unused"
    match = re.match(r"action_head\.transformer_blocks\.(\d+)\.", name)
    if match:
        return f"action.block.{int(match.group(1)):02d}"
    if name.startswith("action_head.action_encoder."):
        return "action.encoder"
    if name.startswith("action_head.state_encoder."):
        return "action.state_encoder"
    if name.startswith("action_head.time_pos_enc."):
        return "action.time_embedding"
    if name.startswith(
        ("action_head.norm_out.", "action_head.seq_pool_proj.", "action_head.mlp_head.")
    ):
        return "action.output"
    return "other"


def grouped_elements(tensors: list[TensorInfo]) -> dict[str, int]:
    groups: dict[str, int] = defaultdict(int)
    for tensor in tensors:
        groups[component(tensor.name)] += tensor.elements
    return dict(groups)


def pack_layers(
    groups: dict[str, int],
    prefix: str,
    budget: int,
    initial: tuple[str, ...] = (),
    final: tuple[str, ...] = (),
) -> list[tuple[str, int]]:
    layers = sorted(
        (name, count) for name, count in groups.items() if name.startswith(prefix + ".layer.")
    )
    chunks: list[tuple[list[str], int]] = []
    names = list(initial)
    count = sum(groups.get(name, 0) for name in initial)
    for name, layer_count in layers:
        if names and count + layer_count > budget:
            chunks.append((names, count))
            names, count = [], 0
        names.append(name)
        count += layer_count
    final_count = sum(groups.get(name, 0) for name in final)
    if names and count + final_count > budget:
        chunks.append((names, count))
        names, count = [], 0
    names.extend(final)
    count += final_count
    if names:
        chunks.append((names, count))

    packed = []
    for names, count in chunks:
        layer_names = [name for name in names if ".layer." in name]
        if layer_names:
            first = int(layer_names[0].rsplit(".", 1)[1])
            last = int(layer_names[-1].rsplit(".", 1)[1])
            label = f"{prefix}[{first:02d}:{last + 1:02d}]"
        else:
            label = "+".join(names)
        if any(name.endswith("embeddings") for name in names):
            label += "+head"
        present_final = [name for name in final if name in names]
        if present_final:
            label += "+" + "+".join(name.rsplit(".", 1)[-1] for name in present_final)
        packed.append((label, count))
    return packed


def action_kv_elements(tensors: list[TensorInfo]) -> int:
    """K and V are two thirds of nn.MultiheadAttention's packed in-projection."""
    total = 0
    for tensor in tensors:
        name = canonical_name(tensor.name)
        if re.match(
            r"action_head\.transformer_blocks\.\d+\.attn\.in_proj_(weight|bias)$", name
        ):
            if tensor.shape[0] % 3:
                raise ValueError(f"packed QKV tensor has unexpected shape: {name} {tensor.shape}")
            total += tensor.elements * 2 // 3
    return total


def split_profile(
    tensors: list[TensorInfo], groups: dict[str, int], budget: int
) -> list[tuple[str, str, int]]:
    profile: list[tuple[str, str, int]] = []
    for label, count in pack_layers(
        groups,
        "vision",
        budget,
        initial=("vision.embeddings",),
        final=("vision.projector",),
    ):
        profile.append(("TRT cold", label, count))

    token_count = groups.get("language.token_embedding", 0)
    if token_count:
        profile.append(("ORT CPU", "language.token_embedding", token_count))
    for label, count in pack_layers(
        groups,
        "language",
        budget,
        final=("language.norm",),
    ):
        profile.append(("TRT cold", label, count))

    action_blocks = sum(
        count for name, count in groups.items() if name.startswith("action.block.")
    )
    if action_blocks:
        kv_count = action_kv_elements(tensors)
        context_count = groups.get("action.state_encoder", 0) + kv_count
        step_count = (
            groups.get("action.encoder", 0)
            + groups.get("action.time_embedding", 0)
            + action_blocks
            - kv_count
        )
        output_count = groups.get("action.output", 0)
        profile.extend(
            [
                ("TRT cold", "action.context+per-block-KV", context_count),
                ("TRT hot x32", "action.velocity.blocks[00:08]", step_count),
                ("TRT hot x32", "action.velocity.output", output_count),
            ]
        )
    return profile


def print_count(label: str, count: int) -> None:
    print(f"{label:46s} {count:12,d}  {count * 4 / 1e9:7.3f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--budget-elements",
        type=int,
        default=100_000_000,
        help="maximum tensor elements per TRT engine (default: 100M, from existing Orin probes)",
    )
    parser.add_argument("--detail", action="store_true", help="print every source component")
    args = parser.parse_args()

    tensors = read_header(args.checkpoint)
    groups = grouped_elements(tensors)
    total = sum(tensor.elements for tensor in tensors)
    on_disk = sum(t.elements * DTYPE_BYTES.get(t.dtype, 4) for t in tensors)

    print(f"\n{args.checkpoint}")
    print(f"{'component':46s} {'elements':>12s}  {'FP32 data':>10s}")
    print("-" * 76)
    if args.detail:
        for name, count in sorted(groups.items()):
            print_count(name, count)
        print("-" * 76)
    print_count("TOTAL", total)
    print(f"on-disk tensors: {on_disk / 1e9:.3f} GB")

    print(f"\nproposed profile (budget {args.budget_elements:,} tensor elements / engine)")
    print(f"{'backend':12s} {'graph':46s} {'elements':>12s}  {'FP32 data':>10s}")
    print("-" * 92)
    profile = split_profile(tensors, groups, args.budget_elements)
    for backend, label, count in profile:
        status = (
            "  OVER BUDGET"
            if backend.startswith("TRT") and count > args.budget_elements
            else ""
        )
        print(
            f"{backend:12s} {label:46s} {count:12,d}  "
            f"{count * 4 / 1e9:7.3f} GB{status}"
        )
    trt_count = sum(backend.startswith("TRT") for backend, _, _ in profile)
    print(f"\n{trt_count} proposed TRT engines; token lookup stays outside TensorRT.")


if __name__ == "__main__":
    main()
