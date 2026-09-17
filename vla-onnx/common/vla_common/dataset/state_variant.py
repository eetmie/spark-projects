#!/usr/bin/env python
"""Build a LeRobot v3 dataset variant with columns dropped from `observation.state`.

`lerobot-train` has no flag for "train on a subset of the proprio channels": the state
vector the policy sees is whatever `observation.state` holds, and its width comes from
`meta/info.json`. So dropping a channel -- because the sensor behind it should not be an
input -- has to happen in the dataset, exactly as `camera_variant` drops a camera.

WHY YOU WOULD DROP ONE. An absolute joint angle is only an input worth having if its
origin means the same thing in every episode. Slew yaw does not: it has no tare, so its
zero moves across power cycles, and a recording session can also simply start each episode
from a different heading. The komatsu `local_manual_test` set opens its episodes anywhere
across 134 degrees of yaw while doing the same job each time -- with 31 episodes a policy
will happily key on that number instead of on what it can see. The masi excavator
recordings already ship `observation.state` as [lift, tilt, scoop] for this reason;
`trim_variant` documents the same call ("slew was dropped because its yaw origin drifts
across power cycles").

Dropping it here is what makes slew *camera-derived*: the swing is then visible only in the
images, and the model has to read it off the scene like the operator does.

WHAT IS NOT TOUCHED, AND WHY.

  * `action` keeps every channel. The machine still has to be *commanded* to slew; what
    is being removed is the feedback, not the degree of freedom.
  * `feedback.*` and `clock.*` keep theirs -- `feedback.imu_quaternion` still carries the
    slew quaternion. They are diagnostics and they never reach the policy:
    `dataset_to_policy_features` keeps `observation*` and `action*` and skips everything
    else (feature_utils.py, `else: continue`). Rewriting them would cost a schema the
    recorder and the eval tools both already read, for no change in what is trained.
    `camera_variant` leaves `clock.*` alone on the same grounds.

STATS ARE SLICED, NOT RECOMPUTED. min/max/mean/std/q* are per-channel, so the surviving
channels' numbers are bit-identical to the source's -- taking column 1: of each array is
the whole job, in `meta/stats.json` and in the per-episode `stats/observation.state/*`
columns of `meta/episodes/*.parquet`. `count` is a frame count and stays. Recomputing
would only introduce a difference where there is none.

The result is a standalone recording, not a view: it belongs under `$VLA_DATASETS`
alongside its source and is what `--dataset` points at. `data/` is rewritten (the state
column is narrower), video is copied so the variant survives its source being moved --
pass `--link-videos` to symlink it instead, when the source is large and staying put.

Usage:
    python -m vla_common.dataset.state_variant --src ~/Desktop/<rec> \
        --dst ~/Desktop/<rec>_nogyro --drop slew
    python -m vla_common.dataset.state_variant --src <rec> --dst <dst> --keep lift tilt scoop
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

STATE = "observation.state"
STAT_PREFIX = f"stats/{STATE}/"
# Per-channel stats: one entry per state dimension, so they slice with the data.
# `count` is a frame count -- one entry whatever the width -- and must not be sliced.
PER_CHANNEL = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="source LeRobot v3 dataset root")
    p.add_argument("--dst", type=Path, required=True, help="variant root to create")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--drop", nargs="+", metavar="NAME",
                   help="observation.state channels to remove, by name (e.g. slew)")
    g.add_argument("--keep", nargs="+", metavar="NAME",
                   help="observation.state channels to keep, in the source's own order")
    p.add_argument("--link-videos", action="store_true",
                   help="symlink videos/ instead of copying it")
    p.add_argument("--force", action="store_true", help="replace an existing --dst")
    return p.parse_args()


def resolve_channels(names: list[str], drop, keep) -> list[int]:
    """Indices of the surviving channels, in source order."""
    unknown = sorted(set(drop or keep or []) - set(names))
    if unknown:
        raise SystemExit(f"{STATE} has no channel {unknown}: it is {names}")
    gone = set(drop) if drop else set(names) - set(keep)
    if not gone:
        raise SystemExit(f"nothing to drop -- {STATE} is already {names}")
    survivors = [i for i, n in enumerate(names) if n not in gone]
    if not survivors:
        raise SystemExit(f"that would empty {STATE}; a policy needs at least one channel")
    return survivors


def slice_stats(block: dict, idx: list[int]) -> dict:
    """One feature's stats dict, narrowed to `idx`."""
    return {k: ([v[i] for i in idx] if k in PER_CHANNEL else v) for k, v in block.items()}


def main():
    args = parse_args()
    src, dst = args.src.expanduser().resolve(), args.dst.expanduser().resolve()

    info = json.loads((src / "meta" / "info.json").read_text())
    feat = info["features"].get(STATE)
    if feat is None:
        raise SystemExit(f"{src} has no {STATE}")
    names = list(feat["names"])
    if len(names) != feat["shape"][0]:
        raise SystemExit(f"{STATE} names {names} disagree with shape {feat['shape']}")
    idx = resolve_channels(names, args.drop, args.keep)
    kept = [names[i] for i in idx]
    dropped = [n for n in names if n not in kept]

    if dst.exists():
        if not args.force:
            raise SystemExit(f"{dst} exists (use --force to replace)")
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)

    # --- meta/info.json: the width the policy will be built from ---------------------
    feat = dict(feat)
    feat["shape"] = [len(idx)]
    feat["names"] = kept
    info["features"] = {**info["features"], STATE: feat}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    # --- meta/stats.json: slice, never recompute -------------------------------------
    stats_path = src / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        if STATE in stats:
            stats[STATE] = slice_stats(stats[STATE], idx)
        (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))

    # Everything else the recorder put in meta/ rides along untouched -- for this dataset
    # that is `recording_contract.json`, which is what says cam1 is rgb8 and the action
    # channels are bar. A variant that loses it stops being a recording you can read back.
    for extra in sorted((src / "meta").iterdir()):
        if extra.name in ("info.json", "stats.json", "episodes"):
            continue
        (shutil.copytree if extra.is_dir() else shutil.copy2)(extra, dst / "meta" / extra.name)

    # --- meta/episodes/*: the same slice, per episode --------------------------------
    # Shard layout is preserved: each episode row carries its own data/ and videos/
    # locators, and collapsing shards here would make every one of them a lie.
    for pq in sorted((src / "meta" / "episodes").rglob("*.parquet")):
        df = pd.read_parquet(pq)
        for col in [c for c in df.columns if c.startswith(STAT_PREFIX)]:
            if col[len(STAT_PREFIX):] not in PER_CHANNEL:
                continue
            df[col] = [np.asarray(v)[idx] for v in df[col]]
        out = dst / "meta" / "episodes" / pq.relative_to(src / "meta" / "episodes")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)

    # --- data/: the only frame-level rewrite -----------------------------------------
    n_frames = 0
    for pq in sorted((src / "data").rglob("*.parquet")):
        df = pd.read_parquet(pq)
        df[STATE] = [np.asarray(v, dtype=np.float32)[idx] for v in df[STATE]]
        out = dst / "data" / pq.relative_to(src / "data")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        n_frames += len(df)

    # --- videos/: copied, so the variant is a recording and not a pointer ------------
    if args.link_videos:
        (dst / "videos").symlink_to((src / "videos").resolve(), target_is_directory=True)
    else:
        shutil.copytree(src / "videos", dst / "videos", symlinks=False)

    print(f"{dst}\n  {STATE}  {names} -> {kept}"
          f"\n  dropped {dropped}   (action and feedback.* keep every channel)"
          f"\n  {info['total_episodes']} episodes, {n_frames} frames"
          f"\n  videos/ {'symlinked to ' + str((src / 'videos').resolve()) if args.link_videos else 'copied'}")


if __name__ == "__main__":
    main()
