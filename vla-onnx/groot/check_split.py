#!/usr/bin/env python3
"""Does the rearranged pipeline (groot_split.run_split) reproduce stock GR00T?

Feeds the reference's own input_ids / pixels / state / noise, right-padded to --seq-len,
and compares backbone features (real tokens only) and the final action chunk.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from groot_split import Contract, layer_plans, load_policy, run_split


def pad(ids, attn, seq_len, pad_id):
    n = ids.shape[1]
    assert n <= seq_len, f"prompt is {n} tokens, more than --seq-len {seq_len}"
    ids = torch.cat([ids, torch.full((1, seq_len - n), pad_id, dtype=ids.dtype)], 1)
    attn = torch.cat([attn, torch.zeros(1, seq_len - n, dtype=attn.dtype)], 1)
    return ids, attn


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
    ap.add_argument("--seq-len", type=int, default=320)
    ap.add_argument("--budget-m", type=float, default=105.0, help="params per chunk, millions")
    ap.add_argument("--dtype", default="float32")
    a = ap.parse_args()

    r = np.load(a.ref)
    meta = json.loads(str(r["meta"]))
    dtype = getattr(torch, a.dtype)
    policy = load_policy(meta["checkpoint"], torch.float32)
    if dtype != torch.float32:
        policy = policy.to(dtype)
    plans = layer_plans(policy, int(a.budget_m * 1e6))
    for k, v in plans.items():
        print(f"{k}: {len(v)} chunks {v}")

    ids = torch.from_numpy(r["input_ids"])
    attn = torch.from_numpy(r["attention_mask"])
    n_real = ids.shape[1]
    pv = torch.from_numpy(r["pixel_values"])
    tok = policy.backbone.model.config.image_token_index
    c = Contract(meta["embodiment"], int(r["embodiment_id"]), meta["views"],
                 tuple(pv.shape[-2:]), int((ids == tok).sum()) // meta["views"], a.seq_len)
    ids_p, attn_p = pad(ids, attn, a.seq_len, policy.backbone.model.config.text_config.eos_token_id)

    with torch.no_grad():
        act, feats = run_split(policy, plans, c, pv, ids_p, attn_p,
                               torch.from_numpy(r["state"]), torch.from_numpy(r["noise"]), dtype)
        print("non-finite in LLM output:", (~torch.isfinite(feats)).sum().item(),
              " max |h|:", feats.float().abs().max().item())
        feats = policy.action_head.vlln(feats.to(next(policy.action_head.vlln.parameters()).dtype))
    report("backbone_features", feats[:, :n_real].float(), torch.from_numpy(r["backbone_features"]))
    report("action_pred", act.float(), torch.from_numpy(r["action_pred"]))


if __name__ == "__main__":
    main()
