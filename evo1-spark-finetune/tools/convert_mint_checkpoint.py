#!/usr/bin/env python3
"""Convert an original MINT EVO1 DeepSpeed checkpoint to LeRobot 0.6.1 keys.

The MINT releases wrap the policy state dictionary in ``checkpoint["module"]`` and
use the original trust-remote-code InternVL3 namespace. LeRobot 0.6.1 uses the
native Transformers InternVL implementation, whose vision attention has separate
Q/K/V parameters. This converter performs only those deterministic renames and
splits; it does not fabricate processor configuration or normalization statistics.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def _add(output: dict[str, torch.Tensor], name: str, value: torch.Tensor) -> None:
    if name in output:
        raise ValueError(f"duplicate converted key: {name}")
    output[name] = value


def convert_module_state(
    source: dict[str, torch.Tensor], *, policy_prefix: bool = True
) -> dict[str, torch.Tensor]:
    """Map an original MINT ``Evo1Model`` state to native-HF LeRobot keys."""
    output: dict[str, torch.Tensor] = {}
    prefix = "model." if policy_prefix else ""

    for name, value in source.items():
        if name.startswith("action_head."):
            _add(output, prefix + name, value)
            continue

        language = "embedder.model.language_model.model."
        if name.startswith(language):
            suffix = name.removeprefix(language)
            _add(output, prefix + "embedder.model.language_model." + suffix, value)
            continue

        projector = {
            "embedder.model.mlp1.0.": "embedder.model.multi_modal_projector.layer_norm.",
            "embedder.model.mlp1.1.": "embedder.model.multi_modal_projector.linear_1.",
            "embedder.model.mlp1.3.": "embedder.model.multi_modal_projector.linear_2.",
        }
        projector_prefix = next((item for item in projector if name.startswith(item)), None)
        if projector_prefix is not None:
            mapped = projector[projector_prefix] + name.removeprefix(projector_prefix)
            _add(output, prefix + mapped, value)
            continue

        embeddings = {
            "class_embedding": "cls_token",
            "position_embedding": "position_embeddings",
            "patch_embedding.weight": "patch_embeddings.projection.weight",
            "patch_embedding.bias": "patch_embeddings.projection.bias",
        }
        embedding_prefix = "embedder.model.vision_model.embeddings."
        if name.startswith(embedding_prefix):
            suffix = name.removeprefix(embedding_prefix)
            try:
                mapped = embeddings[suffix]
            except KeyError as exc:
                raise ValueError(f"unsupported vision embedding key: {name}") from exc
            _add(output, prefix + "embedder.model.vision_tower.embeddings." + mapped, value)
            continue

        layer_prefix = "embedder.model.vision_model.encoder.layers."
        if name.startswith(layer_prefix):
            suffix = name.removeprefix(layer_prefix)
            layer, separator, field = suffix.partition(".")
            if not separator or not layer.isdigit():
                raise ValueError(f"unsupported vision layer key: {name}")
            target = prefix + f"embedder.model.vision_tower.encoder.layer.{layer}."
            if field in {"attn.qkv.weight", "attn.qkv.bias"}:
                parameter = field.rsplit(".", 1)[1]
                if value.shape[0] % 3:
                    raise ValueError(f"packed QKV is not divisible by three: {name}")
                for component, tensor in zip(("q_proj", "k_proj", "v_proj"), value.chunk(3, dim=0)):
                    _add(output, target + f"attention.{component}.{parameter}", tensor)
                continue
            if field in {"ls1", "ls2"}:
                mapped = {"ls1": "lambda_1", "ls2": "lambda_2"}[field]
                _add(output, target + mapped, value)
                continue
            renames = {
                "attn.proj": "attention.projection_layer",
                "mlp.fc1": "mlp.fc1",
                "mlp.fc2": "mlp.fc2",
                "norm1": "layernorm_before",
                "norm2": "layernorm_after",
            }
            stem, dot, parameter = field.rpartition(".")
            if not dot or stem not in renames:
                raise ValueError(f"unsupported vision layer field: {name}")
            _add(output, target + renames[stem] + "." + parameter, value)
            continue

        raise ValueError(f"unsupported MINT EVO1 key: {name}")

    return output


def _signature(state: dict[str, torch.Tensor]) -> str:
    payload = "\n".join(
        f"{name}\t{','.join(map(str, tensor.shape))}\t{tensor.dtype}"
        for name, tensor in sorted(state.items())
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, help="optional model.safetensors output")
    parser.add_argument(
        "--model-only-keys",
        action="store_true",
        help="omit the outer 'model.' prefix used by Evo1Policy.state_dict()",
    )
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=True
    )
    if "module" not in checkpoint or not isinstance(checkpoint["module"], dict):
        raise ValueError("expected a DeepSpeed checkpoint containing checkpoint['module']")
    source = checkpoint["module"]
    converted = convert_module_state(source, policy_prefix=not args.model_only_keys)
    report = {
        "source": str(args.checkpoint.resolve()),
        "source_tensors": len(source),
        "converted_tensors": len(converted),
        "source_elements": sum(tensor.numel() for tensor in source.values()),
        "converted_elements": sum(tensor.numel() for tensor in converted.values()),
        "converted_dtypes": dict(
            collections.Counter(str(tensor.dtype) for tensor in converted.values())
        ),
        "key_shape_signature_sha256": _signature(converted),
        "output": str(args.output.resolve()) if args.output else None,
    }
    if report["source_elements"] != report["converted_elements"]:
        raise ValueError("conversion changed the number of tensor elements")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {name: tensor.contiguous() for name, tensor in converted.items()},
            str(args.output),
            metadata={"format": "pt", "source": str(args.checkpoint)},
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
