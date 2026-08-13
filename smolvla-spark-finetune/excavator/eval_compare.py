#!/usr/bin/env python
"""Compare SmolVLA runs trained at different control rates, on a common time base.

The runs cannot be compared by training loss: each was trained on a different
temporal decimation of the same data, so their action distributions, their
normalization statistics and their loss scales all differ. A model at 6 fps
predicting 50 actions is describing 8.3 s of machine motion; a model at 30 fps
predicting 50 actions is describing 1.67 s. Those numbers are not commensurable.

So this script evaluates every run the same way:

  * the observation is taken from the ORIGINAL 30 fps dataset, so every model
    sees byte-identical images and states,
  * each model predicts its action chunk at its own rate,
  * the chunk is expanded to a common 30 Hz grid by zero-order hold, which is
    how it would actually execute on the machine,
  * it is scored against the original 30 fps ground-truth actions over a fixed
    wall-clock horizon.

Because the actions are rate commands, the headline metric is displacement
error: the integral of (predicted - actual) command over the horizon. That is
what decides whether the bucket ends up in the right place. Per-sample MAE is
reported too, but a model can have decent MAE and still drift.

Two trivial baselines are included. If a run does not beat them it has not
learned the task, and the comparison between runs is moot.

Usage:
    python eval_compare.py                       # all runs found under outputs/excavator
    python eval_compare.py --runs A B --horizon 1.5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

SRC = Path("/home/masi-pgx/Desktop/masi_kaivuri_juusto")
OUT = Path("/home/masi-pgx/smolvla/outputs/excavator")
TASK = "scoop blocks and dump it to the left"
JOINTS = ["slew", "lift", "tilt", "scoop"]
SRC_FPS = 30
VAL_EPISODES = [3, 11, 19, 27]
# Start points land on multiples of 15 so the same source frame exists in the
# 30/10/6 fps variants; every model is scored from identical observations.
STRIDE_EVAL = 15


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", default=None, help="run names (default: every run with a checkpoint)")
    p.add_argument("--checkpoint", default="last", help="checkpoint to load (default: last)")
    p.add_argument("--horizons", type=float, nargs="+", default=[1.5, 4.0], help="horizons in seconds")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", type=Path, default=OUT, help="sweep output dir holding the runs")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()
    if args.json_out is None:
        args.json_out = args.out_dir / "comparison.json"
    return args


def load_ground_truth():
    df = pd.read_parquet(SRC / "data" / "chunk-000" / "file-000.parquet")
    eps = pd.read_parquet(SRC / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
    bounds = {
        int(r["episode_index"]): (int(r["dataset_from_index"]), int(r["dataset_to_index"]))
        for _, r in eps.iterrows()
    }
    return actions, bounds


def eval_points(bounds, max_horizon_s):
    """Source-frame indices to evaluate from, with a full horizon left in the episode."""
    need = int(round(max_horizon_s * SRC_FPS))
    points = []
    for ep in VAL_EPISODES:
        start, stop = bounds[ep]
        points += [i for i in range(start, stop - need, STRIDE_EVAL)]
    return points


def predict_chunks(ckpt, src_ds, points, batch_size, device):
    """Run one model over all eval points. Returns (N, chunk, 4) unnormalized actions and its fps."""
    # SmolVLA's flow-matching sampler draws random noise, so repeated evals of the same
    # checkpoint differ by ~0.001 disp_err. Seed per model to make the numbers reproducible;
    # without this the sampling jitter is a meaningful fraction of the gaps between runs.
    torch.manual_seed(0)
    policy = SmolVLAPolicy.from_pretrained(ckpt)
    policy.eval().to(device)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt)
    fps = json.loads((ckpt / "train_config.json").read_text())["dataset"]["repo_id"]
    fps = {"local/masi_kaivuri_juusto": 30, "local/masi_kaivuri_10fps": 10, "local/masi_kaivuri_6fps": 6}[fps]

    chunks = []
    for i in range(0, len(points), batch_size):
        group = points[i : i + batch_size]
        items = [src_ds[j] for j in group]
        batch = {
            "observation.state": torch.stack([it["observation.state"] for it in items]),
            "observation.images.cam1": torch.stack([it["observation.images.cam1"] for it in items]),
            "task": [TASK] * len(group),
        }
        with torch.no_grad():
            chunks.append(post(policy.predict_action_chunk(pre(batch))).float().cpu())
    del policy
    torch.cuda.empty_cache()
    return torch.cat(chunks).numpy(), fps


def to_30hz(chunk, fps, n_samples):
    """Zero-order hold a chunk predicted at `fps` onto the 30 Hz grid."""
    reps = SRC_FPS // fps
    expanded = np.repeat(chunk, reps, axis=1)
    if expanded.shape[1] < n_samples:
        raise ValueError(f"chunk covers {expanded.shape[1] / SRC_FPS:.2f}s, need {n_samples / SRC_FPS:.2f}s")
    return expanded[:, :n_samples]


def score(pred, gt):
    """pred, gt: (N, T, 4) rate commands on the 30 Hz grid."""
    dt = 1.0 / SRC_FPS
    mae = np.abs(pred - gt).mean()
    # Integrated command over the horizon: where the joint actually ends up.
    disp_err = np.abs((pred - gt).sum(axis=1) * dt)
    return {
        "mae": float(mae),
        "disp_err": float(disp_err.mean()),
        "disp_err_per_joint": {j: float(v) for j, v in zip(JOINTS, disp_err.mean(axis=0))},
        "move_ratio": float(np.abs(pred).mean() / np.abs(gt).mean()),
    }


def main():
    args = parse_args()
    actions, bounds = load_ground_truth()
    max_h = max(args.horizons)
    points = eval_points(bounds, max_h)
    print(f"evaluating on {len(points)} start points from held-out episodes {VAL_EPISODES}\n")

    runs = args.runs
    if runs is None:
        runs = sorted(d.name for d in args.out_dir.iterdir() if (d / "checkpoints" / args.checkpoint).exists())
    if not runs:
        raise SystemExit(f"no runs with a '{args.checkpoint}' checkpoint under {args.out_dir}")

    src_ds = LeRobotDataset(repo_id="local/src", root=SRC, video_backend="torchcodec")

    # Ground truth on the 30 Hz grid, per horizon.
    n_max = int(round(max_h * SRC_FPS))
    gt_full = np.stack([actions[p : p + n_max] for p in points])

    results = {}
    for name in runs:
        ckpt = args.out_dir / name / "checkpoints" / args.checkpoint / "pretrained_model"
        if not ckpt.exists():
            print(f"[{name}] no checkpoint, skipping")
            continue
        print(f"[{name}] running inference ...", flush=True)
        chunk, fps = predict_chunks(ckpt, src_ds, points, args.batch_size, args.device)
        covered = chunk.shape[1] / fps
        results[name] = {"fps": fps, "chunk": int(chunk.shape[1]), "covers_s": covered, "horizons": {}}
        for h in args.horizons:
            n = int(round(h * SRC_FPS))
            if covered + 1e-6 < h:
                continue
            results[name]["horizons"][f"{h}s"] = score(to_30hz(chunk, fps, n), gt_full[:, :n])

    # Trivial baselines on the same points and horizons. The mean is taken over the
    # training episodes only, so the baseline gets no more information than the models.
    train_mask = np.ones(len(actions), bool)
    for ep in VAL_EPISODES:
        train_mask[slice(*bounds[ep])] = False
    train_mean = actions[train_mask].mean(axis=0)
    for label, const in [("zero-action", np.zeros(4, np.float32)), ("mean-action", train_mean)]:
        results[label] = {"fps": SRC_FPS, "chunk": n_max, "covers_s": max_h, "horizons": {}}
        for h in args.horizons:
            n = int(round(h * SRC_FPS))
            pred = np.broadcast_to(const, (len(points), n, 4))
            results[label]["horizons"][f"{h}s"] = score(pred, gt_full[:, :n])

    for h in args.horizons:
        key = f"{h}s"
        rows = [(n, r) for n, r in results.items() if key in r["horizons"]]
        if not rows:
            continue
        print(f"\n{'=' * 78}\nhorizon {h}s   (lower is better; move_ratio near 1.0 = right amount of motion)")
        print(f"{'run':<14}{'fps':>5}{'chunk':>7}{'covers':>8}{'MAE':>9}{'disp_err':>10}{'move_ratio':>12}")
        print("-" * 78)
        for name, r in sorted(rows, key=lambda x: x[1]["horizons"][key]["disp_err"]):
            s = r["horizons"][key]
            print(
                f"{name:<14}{r['fps']:>5}{r['chunk']:>7}{r['covers_s']:>7.1f}s"
                f"{s['mae']:>9.4f}{s['disp_err']:>10.4f}{s['move_ratio']:>12.2f}"
            )
        print("\nper-joint displacement error:")
        print(f"{'run':<14}" + "".join(f"{j:>10}" for j in JOINTS))
        for name, r in sorted(rows, key=lambda x: x[1]["horizons"][key]["disp_err"]):
            pj = r["horizons"][key]["disp_err_per_joint"]
            print(f"{name:<14}" + "".join(f"{pj[j]:>10.4f}" for j in JOINTS))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
