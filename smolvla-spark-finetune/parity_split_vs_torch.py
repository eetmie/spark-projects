#!/usr/bin/env python
"""Split 9-graph ORT pipeline vs the PyTorch checkpoint on identical seeded inputs.

The repo's parity_split_onnx.py compares the split graphs against a MONOLITH ONNX of
lerobot/smolvla_base — useless for a finetuned checkpoint, whose monolith does not exist.
This compares against the finetuned torch policy itself, which is the real gold and is
what the excav_E17500 PARITY.txt recorded.

  python parity_split_vs_torch.py --split-dir exports-split-digIR20k \
      --model-id outputs/digging/ir/checkpoints/020000/pretrained_model
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from parity_split_onnx import SplitPipeline


def cosine(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(args.model_id).eval().float().to(args.device)
    for p in policy.parameters():
        p.requires_grad_(False)
    cfg = policy.config
    dev = torch.device(args.device)
    print(f"torch policy device: {dev}")

    rng = np.random.default_rng(args.seed)
    image = rng.standard_normal((1, 3, *cfg.resize_imgs_with_padding)).astype(np.float32)
    lang_tokens = np.ones((1, cfg.tokenizer_max_length), dtype=np.int64)
    lang_masks = np.ones((1, cfg.tokenizer_max_length), dtype=bool)
    noise = rng.standard_normal((1, cfg.chunk_size, cfg.max_action_dim)).astype(np.float32)

    # This model is STATE-FED (3 real joints padded to max_state_dim), so the meaningful
    # condition is a realistic normalized state, not zeros. Check both anyway.
    cases = {
        "random state": rng.standard_normal((1, cfg.max_state_dim)).astype(np.float32),
        "zero state": np.zeros((1, cfg.max_state_dim), dtype=np.float32),
    }

    split = SplitPipeline(Path(args.split_dir), cfg)
    ok = True
    for label, state in cases.items():
        s_out = split.sample_actions(image, lang_tokens, lang_masks, state, noise)
        with torch.no_grad():
            t_out = policy.model.sample_actions(
                [torch.from_numpy(image).to(dev)],
                [torch.ones(1, dtype=torch.bool, device=dev)],
                torch.from_numpy(lang_tokens).to(dev),
                torch.from_numpy(lang_masks).to(dev),
                torch.from_numpy(state).to(dev),
                noise=torch.from_numpy(noise).to(dev),
            ).cpu().numpy()
        cos = cosine(t_out, s_out)
        absd = np.abs(t_out.astype(np.float64) - s_out.astype(np.float64))
        passed = cos >= 0.999 and absd.max() <= 1e-2
        ok &= passed
        print(f"\n--- {label} ---")
        print(f"  split {s_out.shape}  torch {t_out.shape}")
        print(f"  cosine        = {cos:.7f}")
        print(f"  max_abs_diff  = {absd.max():.6e}")
        print(f"  mean_abs_diff = {absd.mean():.6e}")
        print(f"  {'PASS' if passed else 'FAIL'}")

    print(f"\nRESULT: {'PASS - split matches the checkpoint, Orin-ready' if ok else 'FAIL - investigate'}")


if __name__ == "__main__":
    main()
