#!/usr/bin/env python3
"""Does the rearranged N1.7 pipeline (groot17_split.run_split) reproduce stock GR00T?

Feeds the reference's own input_ids / pixels / state / noise, right-padded to --seq-len,
and compares backbone features (real tokens only) and the final action chunk. The
pixels are rebuilt from the processor's flattened patches, and the host normalization
of the eval-transformed frames is checked against them.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from groot17_split import MERGE, PATCH, Contract, layer_plans, load_policy, run_split


def unpatchify(pv: np.ndarray, n: int, gh: int, gw: int) -> np.ndarray:
    """Processor patches [N*P, 3*2*16*16] -> [N,3,H,W] (temporal copy 0)."""
    x = pv.reshape(n, gh // MERGE, gw // MERGE, MERGE, MERGE, 3, 2, PATCH, PATCH)[..., 0, :, :]
    x = x.transpose(0, 5, 1, 3, 6, 2, 4, 7)       # N, C, bh, mh, ph, bw, mw, pw
    return np.ascontiguousarray(x.reshape(n, 3, gh * PATCH, gw * PATCH))


def report(name, a, b):
    a, b = a.double().flatten(), b.double().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    d = (a - b).abs()
    rng = (b.max() - b.min()).item()
    print(f"{name:20s} cos {cos:.7f}  max|d| {d.max().item():.3e}  "
          f"max/range {d.max().item() / rng * 100:.4f} %")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--seq-len", type=int, default=448)
    ap.add_argument("--budget-m", type=float, default=105.0, help="params per chunk, millions")
    a = ap.parse_args()

    r = np.load(a.ref)
    meta = json.loads(str(r["meta"]))
    policy = load_policy(meta["checkpoint"], meta["vlm_files"], torch.float32)
    plans = layer_plans(policy, int(a.budget_m * 1e6))
    for k, v in plans.items():
        print(f"{k}: {len(v)} chunks {v}")

    thw = r["image_grid_thw"]
    n = thw.shape[0]
    gh, gw = int(thw[0, 1]), int(thw[0, 2])
    assert (thw == thw[0]).all()
    c = Contract(meta["embodiment"], int(r["embodiment_id"]), meta["views"], meta["frames"],
                 (gh * PATCH, gw * PATCH), a.seq_len)
    pix = unpatchify(r["pixel_values"], n, gh, gw)
    host = (r["stacked"].astype(np.float32) / 255.0 - 0.5) / 0.5
    print("host normalize vs processor pixels: shape", host.shape, pix.shape,
          " max|d|", float(np.abs(host - pix).max()) if host.shape == pix.shape else "n/a")

    ids = torch.from_numpy(r["input_ids"])
    n_real = ids.shape[1]
    assert n_real <= a.seq_len, f"prompt is {n_real} tokens, more than --seq-len {a.seq_len}"
    pad_id = policy.backbone.model.config.text_config.eos_token_id
    ids_p = torch.cat([ids, torch.full((1, a.seq_len - n_real), pad_id, dtype=ids.dtype)], 1)
    print(f"prompt {n_real} tokens, {n} images x {c.tokens_per_image} tokens, padded to {a.seq_len}")

    with torch.no_grad():
        act, feats = run_split(policy, plans, c, torch.from_numpy(pix), ids_p, n_real,
                               torch.from_numpy(r["state"]), torch.from_numpy(r["noise"]))
    # reference.py's backbone_features is post vlln + VL self-attention: get_action
    # overwrites the dict entry in place.
    report("conditioning (vl)", feats[:, :n_real], torch.from_numpy(r["backbone_features"]))
    report("action_pred", act, torch.from_numpy(r["action_pred"]))


if __name__ == "__main__":
    main()
