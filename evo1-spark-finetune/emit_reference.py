#!/usr/bin/env python3
"""Emit a deterministic native-LeRobot EVO1 reference fixture for split parity."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from export_split_onnx import _build_policy, write_manifest

ROOT = Path(__file__).resolve().parent


def _verify_manifest(bundle_dir: Path) -> None:
    import hashlib

    manifest = bundle_dir / "MANIFEST.sha256"
    for line in manifest.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = bundle_dir / relative
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"bundle identity mismatch: {relative}")


def _causal_mask(valid: np.ndarray) -> np.ndarray:
    length = valid.shape[1]
    causal = np.tril(np.ones((length, length), dtype=bool))
    allowed = causal[None, None, :, :] & valid[:, None, None, :].astype(bool)
    return np.where(allowed, 0.0, -10000.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=ROOT / "exports" / "split-bootstrap",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "models" / "InternVL3-1B-hf",
    )
    parser.add_argument("--task", default="move sand to the container")
    parser.add_argument("--noise-seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    from lerobot.policies.evo1.internvl3_embedder import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        _batched_pixel_values,
    )

    bundle_dir = args.bundle_dir.resolve()
    _verify_manifest(bundle_dir)
    bundle = json.loads((bundle_dir / "bundle.json").read_text())
    if bundle.get("deployable") or not bundle.get("random_action_head"):
        raise ValueError("this fixture emitter is only for the bootstrap bundle")

    start = time.time()
    policy = _build_policy(
        args.base.resolve(),
        int(bundle["seed"]),
        int(bundle["seq_len"]),
    )
    policy.to(args.device).eval()
    owner = policy.model.embedder.model
    tokenizer = policy.model.embedder.tokenizer

    image_rng = np.random.default_rng(20260902)
    raw_image = image_rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    image = (
        torch.from_numpy(raw_image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(args.device, dtype=torch.float32)
        / 255.0
    )
    mean = torch.tensor(IMAGENET_MEAN, device=args.device, dtype=torch.float32)
    std = torch.tensor(IMAGENET_STD, device=args.device, dtype=torch.float32)
    pixel_values = _batched_pixel_values(
        [image],
        max_views=1,
        image_size=int(bundle["image_size"]),
        mean=mean,
        std=std,
        dtype=torch.float32,
        device=args.device,
    )

    prompt = policy.model.embedder._build_multimodal_prompts(
        [[1]],
        [args.task],
    )
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=int(bundle["seq_len"]),
    ).to(args.device)
    input_ids = encoded["input_ids"]
    context_mask = encoded["attention_mask"].to(torch.bool)
    image_tokens = int(
        (input_ids == int(bundle["image_token_id"])).sum().item()
    )
    if image_tokens != int(bundle["image_seq_length"]):
        raise ValueError(
            f"prompt has {image_tokens} image tokens, expected "
            f"{bundle['image_seq_length']}"
        )

    with torch.no_grad():
        vision_start = time.time()
        image_features = owner.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
            return_dict=True,
        ).pooler_output
        vision_s = time.time() - vision_start

        token_embeddings = owner.language_model.embed_tokens(input_ids)
        image_mask = (input_ids == int(bundle["image_token_id"])).unsqueeze(-1)
        image_mask = image_mask.expand_as(token_embeddings)
        merged = token_embeddings.masked_scatter(
            image_mask,
            image_features.to(token_embeddings.dtype),
        )
        language_start = time.time()
        fused_tokens = owner.language_model(
            attention_mask=encoded["attention_mask"],
            inputs_embeds=merged,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        language_s = time.time() - language_start

        state = torch.linspace(
            -0.5,
            0.5,
            int(bundle["max_state_dim"]),
            device=args.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        action_mask = torch.ones(
            1,
            int(bundle["max_action_dim"]),
            device=args.device,
            dtype=torch.bool,
        )
        embodiment = torch.zeros(1, device=args.device, dtype=torch.long)

        torch.manual_seed(args.noise_seed)
        initial_noise = torch.rand(
            1,
            int(bundle["chunk_size"]) * int(bundle["max_action_dim"]),
            device=args.device,
            dtype=fused_tokens.dtype,
        )
        initial_noise = initial_noise.mul(2).sub(1).view(
            1,
            int(bundle["chunk_size"]),
            int(bundle["max_action_dim"]),
        )
        torch.manual_seed(args.noise_seed)
        action_start = time.time()
        expected_action = policy.model.action_head.get_action(
            fused_tokens,
            state=state,
            action_mask=action_mask,
            embodiment_id=embodiment,
            context_mask=context_mask,
        ).view(
            1,
            int(bundle["chunk_size"]),
            int(bundle["max_action_dim"]),
        )
        action_s = time.time() - action_start

    fixture_path = bundle_dir / "parity_fixture.npz"
    np.savez_compressed(
        fixture_path,
        raw_image=raw_image,
        pixel_values=pixel_values.cpu().numpy().astype(np.float32),
        input_ids=input_ids.cpu().numpy().astype(np.int64),
        context_mask=context_mask.cpu().numpy().astype(bool),
        causal_mask=_causal_mask(context_mask.cpu().numpy()),
        state=state.cpu().numpy().astype(np.float32),
        initial_noise=initial_noise.cpu().numpy().astype(np.float32),
        expected_vision=image_features.cpu().numpy().astype(np.float32),
        expected_fused=fused_tokens.cpu().numpy().astype(np.float32),
        expected_action=expected_action.cpu().numpy().astype(np.float32),
    )
    bundle["fixture"] = {
        "file": fixture_path.name,
        "task": args.task,
        "noise_seed": args.noise_seed,
        "reference": "native LeRobot 0.6.1 EVO1, FP32 CUDA",
    }
    bundle["reference_timings_s"] = {
        "vision": round(vision_s, 4),
        "language": round(language_s, 4),
        "action_32_steps": round(action_s, 4),
        "total_with_load": round(time.time() - start, 4),
    }
    (bundle_dir / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    files = write_manifest(bundle_dir)
    print(f"reference fixture: {fixture_path}")
    print(
        f"native timings: vision={vision_s:.3f}s language={language_s:.3f}s "
        f"action32={action_s:.3f}s"
    )
    print(
        f"expected action range: {expected_action.min().item():.6f} "
        f"to {expected_action.max().item():.6f}"
    )
    print(f"rewrote MANIFEST.sha256 ({files} files)")


if __name__ == "__main__":
    main()
