#!/usr/bin/env python
"""Merge LeRobot v3 recordings of one task into one dataset, without re-encoding video.

Built for the new-IMU retrain: `masi_digging_new_imu` + `_edge_cases` (6-wide action,
tracks appended) plus the sand half of `masi_digging_dry_2` (4-wide, recorded before the
tracks were drivable). lerobot's own `delete_episodes` cannot drop dry_2's rock
episodes cheaply: they are interleaved with sand inside every video file, so it would
re-encode almost all of them to AV1 — slow, and lossy on the frames the policy sees.

Here nothing is decoded. Each source gets its own output chunk (`chunk-00<i>`); its
video files are SYMLINKED in under that chunk and unkept episodes simply stop being
referenced. Only the parquet is rewritten: episode_index / index renumbered, one
instruction for everything (`--task`), and a narrower action zero-padded to the widest
source's layout. Padding is refused unless the narrow layout is a strict PREFIX of the
wide one by name, so [slew,lift,tilt,scoop] can become [..,trackL,trackR] and nothing
else can. Zero is a claim about the recording: that those actuators were not commanded.

Stats are re-aggregated from the kept episodes' own per-episode stats (the padded
channels get zero stats with the episode's count), so the rock frames dry_2's global
stats.json includes do not leak into the normaliser.

Usage:
    python -m vla_common.dataset.merge --out datasets/sand_mix \
        --task "scoop the sand and put it to the container" \
        --src ~/Desktop/kaivurin_datasetit/masi_digging_new_imu \
        --src ~/Desktop/kaivurin_datasetit/masi_digging_new_imu_edge_cases \
        --src "~/Desktop/kaivurin_datasetit/masi_digging_dry_2::move sand to container"

`ROOT::INSTRUCTION` keeps only that source's episodes with that instruction.

`--action-norm trackL=0,0.15` pins a channel's MEAN_STD stats. The tracks need it: they
move in ~1% of frames, so their measured std is ~0.035 and a normal track command lands
at 15-30 sigma -- once the arm channels are fitted, those rare spikes are most of the
MSE and every batch that holds one gets a gradient spike. 0.15 puts a typical command
at ~4 sigma while X-VLA's ~0.1-sigma idle jitter stays at ~0.015, under the valves' 2%
deadzone. The pin is written into stats.json, so the exported bundle unnormalises with
the same numbers.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

INDEX_STATS = ("episode_index", "index", "task_index")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", action="append", required=True,
                   help="ROOT or ROOT::INSTRUCTION (keep only that task); repeat, in output order")
    p.add_argument("--out", type=Path, required=True, help="merged dataset root to create")
    p.add_argument("--task", required=True, help="the one instruction every kept episode gets")
    p.add_argument("--action-norm", action="append", default=[], metavar="NAME=MEAN,STD",
                   help="pin one action channel's normalisation stats (repeatable)")
    p.add_argument("--force", action="store_true", help="replace an existing --out")
    return p.parse_args()


def to_np(v):
    """Episode-stat cells come back from parquet as nested object arrays."""
    if isinstance(v, np.ndarray) and v.dtype == object:
        return np.stack([to_np(x) for x in v])
    return np.asarray(v)


def read_episodes(root: Path) -> pd.DataFrame:
    shards = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


def main():
    args = parse_args()
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import write_info, write_stats, write_tasks

    srcs = []
    for spec in args.src:
        root, _, keep = spec.partition("::")
        root = Path(root).expanduser().resolve()
        info = json.loads((root / "meta" / "info.json").read_text())
        srcs.append({"root": root, "keep": keep or None, "info": info})

    # --- schema: widest action wins, everything else must match exactly ------------
    ref = max(srcs, key=lambda s: s["info"]["features"]["action"]["shape"][0])["info"]
    feats = ref["features"]
    names = feats["action"]["names"]
    for s in srcs:
        f = s["info"]["features"]
        other = {k: v for k, v in f.items() if k != "action"}
        if other != {k: v for k, v in feats.items() if k != "action"}:
            raise SystemExit(f"{s['root'].name}: non-action features differ from {names}")
        if names[:len(f["action"]["names"])] != f["action"]["names"]:
            raise SystemExit(f"{s['root'].name}: action {f['action']['names']} is not a "
                             f"prefix of {names}; refusing to pad")
        if s["info"]["fps"] != ref["fps"]:
            raise SystemExit(f"{s['root'].name}: fps {s['info']['fps']} != {ref['fps']}")
        for sub in ("data", *[f"videos/{k}" for k, v in f.items() if v["dtype"] == "video"]):
            chunks = sorted(p.name for p in (s["root"] / sub).iterdir())
            if chunks != ["chunk-000"]:
                raise SystemExit(f"{s['root'].name}/{sub}: expected only chunk-000, got {chunks}")
    width = len(names)
    video_keys = [k for k, v in feats.items() if v["dtype"] == "video"]
    # Write against the recording's own arrow schema (scalars are plain columns there,
    # not the length-1 lists get_hf_features_from_features would declare), with the
    # huggingface metadata regenerated from it: the fold tool left new_imu's saying
    # action length 4 plus a separate action.tracks.
    wide = next(s for s in srcs if s["info"] is ref)
    schema = pq.read_schema(next((wide["root"] / "data" / "chunk-000").glob("*.parquet")))
    schema = schema.remove_metadata()
    schema = schema.with_metadata({"huggingface": json.dumps(
        {"info": {"features": datasets.Features.from_arrow_schema(schema).to_dict()}})})

    out = args.out.expanduser().resolve()
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} exists (use --force to replace)")
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    ep_rows, ep_stats, provenance = [], [], []
    next_ep = next_index = 0
    for ci, s in enumerate(srcs):
        root = s["root"]
        eps = read_episodes(root).sort_values("episode_index").reset_index(drop=True)
        if any(len(t) != 1 for t in eps["tasks"]):
            raise SystemExit(f"{root.name}: an episode carries more than one instruction")
        eps["task"] = [t[0] for t in eps["tasks"]]
        if s["keep"] is not None:
            if s["keep"] not in set(eps["task"]):
                raise SystemExit(f"{root.name}: no episodes with {s['keep']!r}; "
                                 f"has {sorted(set(eps['task']))}")
            eps = eps[eps["task"] == s["keep"]].reset_index(drop=True)
        pad = width - s["info"]["features"]["action"]["shape"][0]

        # old episode -> (new episode, index shift)
        remap, first_ep = {}, next_ep
        for _, e in eps.iterrows():
            remap[int(e["episode_index"])] = (next_ep, next_index - int(e["dataset_from_index"]))
            next_ep += 1
            next_index += int(e["length"])

        # --- data: filter, renumber, pad; one output file per source file ---------
        written = 0
        for f in sorted((root / "data" / "chunk-000").glob("*.parquet")):
            df = pd.read_parquet(f)
            df = df[df["episode_index"].isin(remap)]
            if df.empty:
                continue
            new_ep = df["episode_index"].map(lambda e: remap[e][0])
            df["index"] = df["index"] + df["episode_index"].map(lambda e: remap[e][1])
            df["episode_index"] = new_ep
            df["task_index"] = 0
            if pad:
                df["action"] = [np.concatenate([np.asarray(a, np.float32), np.zeros(pad, np.float32)])
                                for a in df["action"]]
            if not df["index"].is_monotonic_increasing:
                raise SystemExit(f"{f}: rows not in index order after filtering")
            dst = out / "data" / f"chunk-{ci:03d}" / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            pq.write_table(table, dst, compression="snappy")
            written += len(df)
        if written != sum(int(l) for l in eps["length"]):
            raise SystemExit(f"{root.name}: wrote {written} rows, episodes claim "
                             f"{int(eps['length'].sum())}")

        # --- videos: symlink every file under this source's chunk ----------------
        for vk in video_keys:
            vdir = out / "videos" / vk / f"chunk-{ci:03d}"
            vdir.mkdir(parents=True, exist_ok=True)
            for v in sorted((root / "videos" / vk / "chunk-000").glob("*.mp4")):
                (vdir / v.name).symlink_to(v)

        # --- episode metadata -------------------------------------------------------
        for _, e in eps.iterrows():
            new_ep, shift = remap[int(e["episode_index"])]
            row = e.drop(labels=["task"]).to_dict()
            row.update({"episode_index": new_ep, "tasks": [args.task],
                        "data/chunk_index": ci,
                        "dataset_from_index": int(e["dataset_from_index"]) + shift,
                        "dataset_to_index": int(e["dataset_to_index"]) + shift,
                        "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0})
            for vk in video_keys:
                row[f"videos/{vk}/chunk_index"] = ci
            stats = {}
            for col in [c for c in e.index if c.startswith("stats/")]:
                _, feat, stat = col.split("/", 2)
                v = to_np(e[col]).astype(np.float64) if stat != "count" else to_np(e[col])
                if feat == "action" and pad and stat != "count":
                    v = np.concatenate([v, np.zeros(pad)])
                if feat in INDEX_STATS and stat != "count":
                    v = {"episode_index": np.full_like(v, new_ep),
                         "index": v + shift,
                         "task_index": np.zeros_like(v)}[feat]
                row[col] = v
                stats.setdefault(feat, {})[stat] = v
            ep_rows.append(row)
            ep_stats.append(stats)
        provenance.append({"src": str(root), "keep_task": s["keep"], "chunk": ci,
                           "episodes": [first_ep, next_ep - 1], "n_episodes": next_ep - first_ep,
                           "frames": int(eps["length"].sum()), "action_padded": pad})
        print(f"  chunk-{ci:03d}  {root.name:<34} eps {first_ep:>3}-{next_ep - 1:<3} "
              f"({next_ep - first_ep}, {int(eps['length'].sum())} frames)"
              + (f"  action +{pad} zero" if pad else ""))

    # --- meta ----------------------------------------------------------------------
    ep_df = pd.DataFrame(ep_rows)
    for col in [c for c in ep_df.columns if c.startswith("stats/")]:
        ep_df[col] = [np.asarray(v).tolist() for v in ep_df[col]]
    dst = out / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    dst.parent.mkdir(parents=True)
    ep_df.to_parquet(dst, index=False)

    stats = aggregate_stats(ep_stats)
    pinned = {}
    for spec in args.action_norm:
        name, _, vals = spec.partition("=")
        if name not in names:
            raise SystemExit(f"--action-norm {name}: not an action channel {names}")
        i = names.index(name)
        mean, std = (float(x) for x in vals.split(","))
        pinned[name] = {"measured": [float(stats["action"]["mean"][i]), float(stats["action"]["std"][i])],
                        "pinned": [mean, std]}
        stats["action"]["mean"][i], stats["action"]["std"][i] = mean, std
    write_stats(stats, out)
    write_tasks(pd.DataFrame({"task_index": [0]}, index=pd.Index([args.task], name="task")), out)
    info = dict(ref)
    info.update({"total_episodes": next_ep, "total_frames": next_index, "total_tasks": 1,
                 "splits": {"train": f"0:{next_ep}"}})
    write_info(info, out)
    (out / ".merge.json").write_text(json.dumps(
        {"task": args.task, "sources": provenance, "action_norm_pinned": pinned,
         "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2))
    print(f"{out}\n  {next_ep} episodes, {next_index} frames, action {names}")


if __name__ == "__main__":
    main()
