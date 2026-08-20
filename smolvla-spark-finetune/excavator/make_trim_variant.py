#!/usr/bin/env python
"""Build a LeRobot v3 dataset variant with the dead air trimmed off every episode.

Every recording starts with the operator stationary -- record is pressed, then movement
begins -- and ends the same way. Measured over masi_digging's 189 episodes, the first
0.5 s carries 89% less action than mid-episode, and about 10% of all frames sit within
1.0 s of a boundary. That is not merely wasted data: an episode's first frames show an
ordinary mid-task scene paired with an action near zero, while the same visual state
elsewhere is paired with real digging. The model gets contradictory supervision on
observations it cannot tell apart, and the symptom on the machine is hesitation.

SmolVLA already masks PADDED actions out of the loss (modeling_smolvla.py, `losses =
losses * in_episode_bound`), so chunks overhanging the end contribute no gradient. That
protects the tail. It does not protect the head: an episode's opening frames are real,
fully supervised, and wrong.

    python make_trim_variant.py --src ~/Desktop/masi_digging --dst datasets/masi_digging_trim
    python make_trim_variant.py --src ... --dst ... --apply

Report-only by default -- it prints what it would cut and changes nothing.

WHERE THE CUT POINTS COME FROM
------------------------------
    start = min(first frame the state actually moves, first frame slew is commanded)
    end   = last frame any channel is commanded

Both halves of that start rule are load-bearing, and the excavator is why:

  * A joystick command that moves nothing is not the start. Episode 4's tilt stick is
    active from 1.40 s but the boom does not move until 1.83 s -- below the hydraulic
    threshold, the command does nothing. Starting from the command keeps 0.43 s of dead
    air; starting from measured state motion does not.
  * State motion alone misses slew. `observation.state` is [lift, tilt, scoop] -- slew
    was dropped because its yaw origin drifts across power cycles -- so a swing is
    INVISIBLE in state. Episode 5 starts with slew at 0.53 s and the state does not
    stir until 1.17 s.

Validated against seven hand-labelled episodes spanning both recording batches
(0, 4, 5, 10, 20, 90, 144): five within 0.15 s, worst case 0.5 s, and every
end within 0.1 s. Errors skew negative -- the rule starts a shade early and keeps a few
extra frames, which is the safe direction.

WHY NO EPISODE-INDEX CUTOFF
---------------------------
The obvious shortcut is "clean the first hundred, the later ones are fine". The data
disagrees. Cut fraction by block: 13.6% (0-19), 7.8% (20-39), 5.3% (40-59), 5.0%
(60-81), 9.4% (82-99), 5.6% (100-119), 2.6% (120-144), 2.3% (145-188). The artifact
does shrink as recording technique improves -- and then jumps back up at 82-99, the
start of the second recording batch. It is not "early episodes", it is the start of
every session. A signal-driven rule scales itself to each one and will handle the
next session's warm-up without anyone remembering to check.

WHAT THIS DOES NOT FIX
----------------------
The trim takes the head dead zone from -89% to -56% of mid-episode action, not to zero.
The residual is the operator genuinely accelerating from rest: real task content,
correctly labelled. Cutting past it would delete data, not noise.

VIDEO IS NOT RE-ENCODED
-----------------------
LeRobot v3 addresses video by [from_timestamp, to_timestamp] into shared per-chunk mp4s,
so trimming is a timestamp edit. `videos/` is symlinked; only the ~8 MB of parquet is
rewritten.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

FPS_FALLBACK = 30.0


# --------------------------------------------------------------------------- detection
def episode_bounds(action, state, fps, action_th, vel_th, smooth):
    """(start, end) inclusive frame indices of the moving part, or None if silent."""
    vel = np.abs(np.diff(state, axis=0)) * fps           # deg/s per joint
    vel = np.convolve(vel.max(axis=1), np.ones(smooth) / smooth, mode="same")
    moving = np.where(vel > vel_th)[0]
    slew = np.where(np.abs(action[:, 0]) > action_th)[0]  # no state feedback for slew
    commanded = np.where((np.abs(action) > action_th).any(axis=1))[0]
    if not len(commanded):
        return None
    firsts = [x[0] for x in (moving, slew) if len(x)]
    start = min(firsts) if firsts else commanded[0]
    return int(start), int(commanded[-1])


def plan(src: Path, action_th, vel_th, smooth, exclude):
    """Per-episode trim plan, computed from the source data."""
    info = json.loads((src / "meta" / "info.json").read_text())
    fps = float(info.get("fps", FPS_FALLBACK))
    data = pd.concat(
        [pd.read_parquet(p) for p in sorted((src / "data").rglob("*.parquet"))]
    ).sort_values(["episode_index", "frame_index"])

    rows = []
    for ep, d in data.groupby("episode_index"):
        ep = int(ep)
        n = len(d)
        if ep in exclude:
            rows.append(dict(ep=ep, n=n, start=0, end=n - 1, keep=0, reason="excluded"))
            continue
        b = episode_bounds(
            np.stack(d["action"].values),
            np.stack(d["observation.state"].values),
            fps, action_th, vel_th, smooth,
        )
        if b is None:
            rows.append(dict(ep=ep, n=n, start=0, end=n - 1, keep=0, reason="silent"))
            continue
        s, e = b
        rows.append(dict(ep=ep, n=n, start=s, end=e, keep=e - s + 1, reason=""))
    return pd.DataFrame(rows), data, info, fps


# --------------------------------------------------------------------------- reporting
def report(pl, fps, min_frames):
    kept = pl[pl.keep > 0]
    dropped = pl[pl.keep == 0]
    cut = int((pl.n - pl.keep).sum())
    print(f"episodes           : {len(pl)}  ->  {len(kept)} kept, {len(dropped)} dropped")
    print(f"frames             : {int(pl.n.sum())}  ->  {int(pl.keep.sum())}"
          f"  ({cut} cut = {100 * cut / pl.n.sum():.1f}%)")
    if len(kept):
        head = kept.start
        tail = kept.n - 1 - kept.end
        print(f"head trim (frames) : median {head.median():.0f}"
              f" ({head.median() / fps:.2f}s), max {head.max()}")
        print(f"tail trim (frames) : median {tail.median():.0f}"
              f" ({tail.median() / fps:.2f}s), max {tail.max()}")
        print(f"shortest survivor  : {int(kept.keep.min())} frames"
              f" ({kept.keep.min() / fps:.1f}s)")
    short = kept[kept.keep < min_frames]
    if len(short):
        print(f"\nWARNING: {len(short)} episode(s) fall below --min-frames {min_frames} "
              f"and are dropped: {short.ep.tolist()}")
    for reason in ("excluded", "silent"):
        g = dropped[dropped.reason == reason]
        if len(g):
            print(f"dropped ({reason}): {g.ep.tolist()}")


# ------------------------------------------------------------------------------- write
def stat_block(values, prefix, out):
    """Per-episode stats in the layout meta/episodes uses."""
    v = np.asarray(values, dtype=np.float64)
    v = v.reshape(len(v), -1) if v.ndim > 1 else v.reshape(-1, 1)
    out[f"stats/{prefix}/min"] = v.min(axis=0).tolist()
    out[f"stats/{prefix}/max"] = v.max(axis=0).tolist()
    out[f"stats/{prefix}/mean"] = v.mean(axis=0).tolist()
    out[f"stats/{prefix}/std"] = v.std(axis=0).tolist()
    out[f"stats/{prefix}/count"] = [len(v)]
    for q, name in ((0.01, "q01"), (0.10, "q10"), (0.50, "q50"), (0.90, "q90"), (0.99, "q99")):
        out[f"stats/{prefix}/{name}"] = np.quantile(v, q, axis=0).tolist()


def apply_trim(src: Path, dst: Path, pl, data, info, fps, min_frames, force):
    if dst.exists():
        if not force:
            raise SystemExit(f"{dst} exists -- pass --force to replace it")
        shutil.rmtree(dst)
    (dst / "meta" / "episodes").mkdir(parents=True)
    (dst / "data" / "chunk-000").mkdir(parents=True)

    keep = pl[(pl.keep > 0) & (pl.keep >= min_frames)].set_index("ep")
    src_eps = pd.concat(
        [pd.read_parquet(p) for p in sorted((src / "meta" / "episodes").rglob("*.parquet"))]
    ).set_index("episode_index", drop=False)

    # Videos are addressed by timestamp into shared mp4s -- symlink, never re-encode.
    (dst / "videos").symlink_to((src / "videos").resolve(), target_is_directory=True)

    out_rows, out_meta, cursor, new_ep = [], [], 0, 0
    for ep in sorted(keep.index):
        r = keep.loc[ep]
        d = data[data.episode_index == ep].iloc[int(r.start): int(r.end) + 1].copy()
        n = len(d)

        # Rewrite the per-frame bookkeeping so the episode is self-consistent from 0.
        d["episode_index"] = new_ep
        d["frame_index"] = np.arange(n, dtype=d["frame_index"].dtype)
        d["timestamp"] = (np.arange(n) / fps).astype(d["timestamp"].dtype)
        d["index"] = np.arange(cursor, cursor + n, dtype=d["index"].dtype)
        out_rows.append(d)

        m = src_eps.loc[ep].to_dict()
        m["episode_index"] = new_ep
        m["length"] = n
        m["dataset_from_index"] = cursor
        m["dataset_to_index"] = cursor + n
        for key in [k for k in m if k.startswith("videos/") and k.endswith("/from_timestamp")]:
            base = key[: -len("/from_timestamp")]
            m[key] = float(m[key]) + int(r.start) / fps
            m[f"{base}/to_timestamp"] = float(m[f"{base}/to_timestamp"]) - (int(r.n) - 1 - int(r.end)) / fps
        # Per-episode stats must describe the trimmed span, not the original.
        for col, name in (("observation.state", "observation.state"), ("action", "action")):
            stat_block(np.stack(d[col].values), name, m)
        for col in [c for c in d.columns if c.startswith("clock.")] + ["timestamp", "frame_index", "index"]:
            stat_block(d[col].values, col, m)
        out_meta.append(m)

        cursor += n
        new_ep += 1

    out = pd.concat(out_rows, ignore_index=True)
    out.to_parquet(dst / "data" / "chunk-000" / "file-000.parquet", index=False)
    meta_df = pd.DataFrame(out_meta)
    meta_df["data/chunk_index"] = 0
    meta_df["data/file_index"] = 0
    meta_df["meta/episodes/chunk_index"] = 0
    meta_df["meta/episodes/file_index"] = 0
    (dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    meta_df.to_parquet(dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)

    shutil.copy2(src / "meta" / "tasks.parquet", dst / "meta" / "tasks.parquet")

    info = dict(info)
    info["total_episodes"] = new_ep
    info["total_frames"] = int(cursor)
    if isinstance(info.get("splits"), dict):
        info["splits"] = {"train": f"0:{new_ep}"}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    # Global stats drive normalization -- recompute over the trimmed frames.
    src_stats = json.loads((src / "meta" / "stats.json").read_text())
    stats = {}
    for key, val in src_stats.items():
        if key in out.columns:
            v = np.stack(out[key].values) if isinstance(out[key].iloc[0], np.ndarray) \
                else np.asarray(out[key].values).reshape(-1, 1)
            v = v.astype(np.float64)
            stats[key] = {
                "min": v.min(axis=0).tolist(), "max": v.max(axis=0).tolist(),
                "mean": v.mean(axis=0).tolist(), "std": v.std(axis=0).tolist(),
                "count": [len(v)],
                **{n: np.quantile(v, q, axis=0).tolist()
                   for q, n in ((0.01, "q01"), (0.10, "q10"), (0.50, "q50"),
                                (0.90, "q90"), (0.99, "q99"))},
            }
        else:
            stats[key] = val          # image stats: frames are a subset, keep source
    (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))

    print(f"\nwrote {dst}")
    print(f"  {new_ep} episodes, {cursor} frames")
    print(f"  videos/ symlinked to {(src / 'videos').resolve()} (not re-encoded)")
    print("  NOTE: episodes were RENUMBERED 0..N-1 after drops -- recompute your "
          "held-out split against the new indices.")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="source LeRobot v3 dataset root")
    p.add_argument("--dst", type=Path, required=True, help="variant root to create")
    p.add_argument("--apply", action="store_true", help="actually write it (default: report only)")
    p.add_argument("--force", action="store_true", help="replace an existing --dst")
    p.add_argument("--action-th", type=float, default=0.05,
                   help="joystick magnitude counted as a command (default 0.05)")
    p.add_argument("--vel-th", type=float, default=2.0,
                   help="joint speed in deg/s counted as real motion (default 2.0)")
    p.add_argument("--smooth", type=int, default=5,
                   help="frames of moving-average on the speed signal (default 5)")
    p.add_argument("--min-frames", type=int, default=50,
                   help="drop episodes shorter than this after trimming; keep >= the "
                        "policy chunk size, since a shorter episode cannot fill one "
                        "(default 50)")
    p.add_argument("--exclude-episodes", type=int, nargs="*", default=[],
                   help="source episode indices to skip entirely, e.g. a run of dirty "
                        "episodes that do not represent the task")
    args = p.parse_args()

    pl, data, info, fps = plan(args.src, args.action_th, args.vel_th, args.smooth,
                               set(args.exclude_episodes))
    print(f"source {args.src}  ({len(pl)} episodes, {int(pl.n.sum())} frames, {fps:g} fps)")
    print(f"thresholds: action>{args.action_th}, speed>{args.vel_th} deg/s, "
          f"smooth {args.smooth} frames\n")
    report(pl, fps, args.min_frames)

    if not args.apply:
        print("\n(report only -- pass --apply to write the variant)")
        return
    apply_trim(args.src, args.dst, pl, data, info, fps, args.min_frames, args.force)


if __name__ == "__main__":
    main()
