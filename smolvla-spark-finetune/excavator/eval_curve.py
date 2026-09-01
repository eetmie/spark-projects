#!/usr/bin/env python
"""Score every saved checkpoint of every run, so we can see where held-out
performance peaks instead of assuming the last step is the best one.

With a few dozen training episodes and tens of epochs, overfitting is expected.
This turns that into data: it plots held-out displacement error against training
step for each run, and reports the best checkpoint per run rather than the final one.

Reuses eval_compare's scoring so the numbers are the same ones the headline table
reports -- identical source-rate observations for every model, each model fed only
the cameras it was trained on, each chunk zero-order held onto a common grid,
scored on integrated command error.

Usage:
    python eval_curve.py --preset digging
    python eval_curve.py --preset kaivuri --out-dir <sweep dir>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_compare import (
    PRESETS,
    eval_points,
    load_ground_truth,
    predict_chunks,
    score,
    source_meta,
    tasks_for_points,
    to_30hz,
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
GRID_COLOR, SURFACE = "#dcdcd8", "#fcfcfb"
BASELINE_COLOR = "#52514e"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), default="digging")
    p.add_argument("--out-dir", type=Path, default=None, help="override the preset's sweep dir")
    p.add_argument("--runs", nargs="*", default=None)
    p.add_argument("--horizon", type=float, default=1.5, help="scoring horizon in seconds")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-draws", type=int, default=1,
                   help="average this many flow-matching noise draws per eval point "
                        "(default 1; see predict_chunks). X-VLA gains ~18%% from 4, "
                        "SmolVLA ~0%% -- so a cross-architecture curve wants 4, and a "
                        "curve compared against earlier results wants 1")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    preset = PRESETS[args.preset]
    if args.out_dir is None:
        args.out_dir = preset.out_dir
    src_fps, joints = source_meta(preset)
    actions, bounds = load_ground_truth(preset)
    points = eval_points(preset, bounds, args.horizon, src_fps)
    # On a multi-task preset each point carries its own episode's instruction.
    point_tasks = tasks_for_points(preset, bounds, points)
    n = int(round(args.horizon * src_fps))
    gt = np.stack([actions[p : p + n] for p in points])
    act_dim = gt.shape[-1]

    runs = args.runs or sorted(d.name for d in args.out_dir.iterdir() if (d / "checkpoints").is_dir())
    src_ds = LeRobotDataset(repo_id="local/src", root=preset.src, video_backend="torchcodec")

    # Zero-action reference line: anything above it has not learned the task.
    baseline = score(np.zeros((len(points), n, act_dim), np.float32), gt, joints, src_fps)["disp_err"]
    draws_note = f", {args.n_draws} noise draws averaged" if args.n_draws > 1 else ""
    print(f"{len(points)} eval points, horizon {args.horizon}s{draws_note}, "
          f"zero-action baseline disp_err={baseline:.4f}\n")

    curves = {}
    for name in runs:
        ck_dir = args.out_dir / name / "checkpoints"
        steps = sorted(d.name for d in ck_dir.iterdir() if d.name.isdigit()) if ck_dir.is_dir() else []
        if not steps:
            continue
        curves[name] = []
        for s in steps:
            ckpt = ck_dir / s / "pretrained_model"
            if not (ckpt / "model.safetensors").exists():
                continue
            chunk, fps, cams, ptype = predict_chunks(
                ckpt, src_ds, points, args.batch_size, args.device, preset, src_fps, point_tasks,
                n_draws=args.n_draws)
            if chunk.shape[1] / fps + 1e-6 < args.horizon:
                continue
            sc = score(to_30hz(chunk, fps, n, src_fps), gt, joints, src_fps)
            cam_short = [c.rsplit(".", 1)[-1] for c in cams]
            curves[name].append({"step": int(s), "fps": fps, "cameras": cam_short,
                                 "policy": ptype, **sc})
            print(f"  [{name}] step {int(s):>6}  disp_err={sc['disp_err']:.4f}  mae={sc['mae']:.4f}  move={sc['move_ratio']:.2f}", flush=True)

    if not curves:
        raise SystemExit(f"no checkpoints found under {args.out_dir}")

    print(f"\n{'=' * 72}\nbest checkpoint per run (held-out disp_err, horizon {args.horizon}s)")
    print(f"{'run':<8}{'best step':>11}{'disp_err':>11}{'final step':>12}{'final':>10}{'overfit?':>10}")
    print("-" * 72)
    summary = {}
    for name, pts in curves.items():
        best = min(pts, key=lambda r: r["disp_err"])
        final = pts[-1]
        degraded = final["disp_err"] - best["disp_err"]
        flag = "yes" if degraded > 0.005 else "no"
        summary[name] = {"best": best, "final": final}
        print(
            f"{name:<8}{best['step']:>11}{best['disp_err']:>11.4f}"
            f"{final['step']:>12}{final['disp_err']:>10.4f}{flag:>10}"
        )

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # The baseline is usually ~3x the model error, and plotting it to scale squashes
    # the differences we actually care about into a sliver. Draw it only when it is
    # close enough to be informative; otherwise say where it is in words.
    all_y = [r["disp_err"] for pts in curves.values() for r in pts]
    baseline_on_scale = baseline <= max(all_y) * 1.3
    if baseline_on_scale:
        ax.axhline(baseline, color=BASELINE_COLOR, lw=2.0, ls="--", label="zero-action baseline")
    baseline_note = ""
    if not baseline_on_scale:
        span = max(all_y) - min(all_y)
        ax.set_ylim(min(all_y) - span * 0.25, max(all_y) + span * 0.25)
        baseline_note = f"\nzero-action baseline = {baseline:.4f} — off scale, {baseline / max(all_y):.1f}x worse than every run"
    for i, (name, pts) in enumerate(sorted(curves.items())):
        xs = [r["step"] for r in pts]
        ys = [r["disp_err"] for r in pts]
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        cams = "+".join(pts[0].get("cameras") or [])
        tag = cams if len({tuple(p.get("cameras") or []) for c in curves.values() for p in c}) > 1 \
            else f"{pts[0]['fps']}fps"
        ax.plot(xs, ys, color=color, lw=2.0, marker="o", ms=5, label=f"{name} ({tag})")
        best = min(pts, key=lambda r: r["disp_err"])
        ax.annotate(
            name,
            xy=(xs[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=TEXT_SECONDARY,
            fontsize=9,
            va="center",
        )
        ax.plot([best["step"]], [best["disp_err"]], marker="o", ms=10, mfc="none", mec=color, mew=2)

    ax.set_xlabel("training step", color=TEXT_SECONDARY)
    ax.set_ylabel(f"held-out displacement error ({args.horizon}s horizon)", color=TEXT_SECONDARY)
    ax.set_title(
        "Held-out error vs training length  ·  ring marks each run's best checkpoint" + baseline_note,
        color=TEXT_PRIMARY,
        fontsize=12,
    )
    ax.grid(True, color=GRID_COLOR, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.legend(frameon=False, labelcolor=TEXT_SECONDARY)
    fig.tight_layout()

    png = args.out_dir / "curve.png"
    fig.savefig(png, dpi=140, facecolor=SURFACE)
    (args.out_dir / "curve.json").write_text(json.dumps({"baseline": baseline, "curves": curves}, indent=2))
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
