#!/usr/bin/env python
"""Plot predicted vs actual joystick commands on a held-out episode.

The comparison table gives a ranking; this gives the reason for it. Each model
replans open-loop at its own chunk boundary from the real observation at that
instant, and the concatenated trace is drawn against ground truth.

Left column is the rate command the model would send. Right column is that
command integrated over time, which is the quantity that decides where the
boom actually ends up -- a model can track the left column loosely and still
drift badly on the right.

Note this is not a closed-loop rollout: observations always come from the real
recording, so errors never compound the way they would on the machine. It shows
whether a model has learned the shape and timing of the task, not whether it
would survive its own mistakes.

Usage:
    python plot_predictions.py                  # episode 3, all runs with a checkpoint
    python plot_predictions.py --episode 11
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
FPS_BY_REPO = {"local/masi_kaivuri_juusto": 30, "local/masi_kaivuri_10fps": 10, "local/masi_kaivuri_6fps": 6}

# Validated categorical slots 1-4 (see dataviz palette reference); ground truth is
# deliberately neutral because it is the reference, not a peer series.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GT_COLOR = "#0b0b0b"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#dcdcd8"
SURFACE = "#fcfcfb"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", type=int, default=3, help="held-out episode to plot (default: 3)")
    p.add_argument("--runs", nargs="*", default=None)
    p.add_argument("--checkpoint", default="last")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def rollout(ckpt, src_ds, start, stop, device):
    """Open-loop replan across an episode. Returns (T, 4) commands on the 30 Hz grid."""
    policy = SmolVLAPolicy.from_pretrained(ckpt)
    policy.eval().to(device)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt)
    fps = FPS_BY_REPO[json.loads((ckpt / "train_config.json").read_text())["dataset"]["repo_id"]]
    reps = SRC_FPS // fps

    trace, t = [], start
    while t < stop:
        item = src_ds[t]
        batch = {
            "observation.state": item["observation.state"].unsqueeze(0),
            "observation.images.cam1": item["observation.images.cam1"].unsqueeze(0),
            "task": [TASK],
        }
        with torch.no_grad():
            chunk = post(policy.predict_action_chunk(pre(batch))).float().cpu().numpy()[0]
        trace.append(np.repeat(chunk, reps, axis=0))  # hold each command at 30 Hz
        t += chunk.shape[0] * reps

    del policy
    torch.cuda.empty_cache()
    return np.concatenate(trace)[: stop - start], fps


def main():
    args = parse_args()
    eps = pd.read_parquet(SRC / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    row = eps[eps["episode_index"] == args.episode]
    if row.empty:
        raise SystemExit(f"episode {args.episode} not found")
    start, stop = int(row.iloc[0]["dataset_from_index"]), int(row.iloc[0]["dataset_to_index"])

    df = pd.read_parquet(SRC / "data" / "chunk-000" / "file-000.parquet")
    gt = np.stack(df["action"].to_numpy()).astype(np.float32)[start:stop]

    runs = args.runs
    if runs is None:
        runs = sorted(d.name for d in OUT.iterdir() if (d / "checkpoints" / args.checkpoint).exists())
    if not runs:
        raise SystemExit(f"no runs with a '{args.checkpoint}' checkpoint under {OUT}")

    src_ds = LeRobotDataset(repo_id="local/src", root=SRC, video_backend="torchcodec")
    traces = {}
    for name in runs:
        ckpt = OUT / name / "checkpoints" / args.checkpoint / "pretrained_model"
        if not ckpt.exists():
            continue
        print(f"[{name}] rolling out episode {args.episode} ...", flush=True)
        trace, fps = rollout(ckpt, src_ds, start, stop, args.device)
        traces[f"{name} ({fps}fps)"] = trace

    time = np.arange(stop - start) / SRC_FPS
    dt = 1.0 / SRC_FPS

    fig, axes = plt.subplots(4, 2, figsize=(15, 11), sharex=True, facecolor=SURFACE)
    for r, joint in enumerate(JOINTS):
        for c, integrated in enumerate([False, True]):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            gt_y = np.cumsum(gt[:, r]) * dt if integrated else gt[:, r]
            ax.plot(time, gt_y, color=GT_COLOR, lw=2.0, label="ground truth", zorder=5)
            for i, (label, tr) in enumerate(traces.items()):
                y = np.cumsum(tr[:, r]) * dt if integrated else tr[:, r]
                color = SERIES_COLORS[i % len(SERIES_COLORS)]
                ax.plot(time, y, color=color, lw=2.0, alpha=0.9, label=label)
                # Direct end label: the palette's contrast WARN requires visible labels.
                ax.annotate(
                    label.split()[0],
                    xy=(time[-1], y[-1]),
                    xytext=(4, 0),
                    textcoords="offset points",
                    color=TEXT_SECONDARY,
                    fontsize=8,
                    va="center",
                    clip_on=False,
                )
            ax.grid(True, color=GRID_COLOR, lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID_COLOR)
            ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
            if c == 0:
                ax.set_ylabel(joint, color=TEXT_PRIMARY, fontsize=11)
            if r == 0:
                # The integral is command x seconds; per joint that is proportional to
                # displacement, but the velocity gain differs per joint so it is not degrees.
                ax.set_title(
                    "integrated command (proportional to joint displacement)"
                    if integrated
                    else "rate command",
                    color=TEXT_SECONDARY,
                    fontsize=10,
                )
            if r == 3:
                ax.set_xlabel("time (s)", color=TEXT_SECONDARY, fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=10,
        labelcolor=TEXT_SECONDARY,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        f"Open-loop replan on held-out episode {args.episode}  ·  checkpoint '{args.checkpoint}'",
        color=TEXT_PRIMARY,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    out = args.out or OUT / "plots" / f"episode_{args.episode}_{args.checkpoint}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
