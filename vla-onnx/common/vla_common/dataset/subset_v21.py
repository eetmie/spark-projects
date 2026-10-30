#!/usr/bin/env python
"""Write the training half of a LeRobot v2.1 dataset: every episode except the held-out ones.

    python -m vla_common.dataset.subset_v21 --src pi05/datasets/digging_complete \
        --dst pi05/datasets/digging_complete_train            # holds out every 10th from 5
    python -m vla_common.dataset.subset_v21 --src SRC --dst DST --val "5 15 25"

WHY A COPY. OpenPI and GR00T train on a whole v2.1 dataset, and neither takes an episode
list. The LeRobot that OpenPI pins does have `LeRobotDataset(episodes=...)`, but with
`delta_timestamps` it looks an episode's frame range up by episode id in a table built
only for the selected episodes, so any subset that skips an episode reads another
episode's frames. A smaller dataset avoids that, and both trainers can read it.

The held-out rule is `vla_common.dataset.split`'s (every 10th episode from 5), the one the
X-VLA runs use, so the held-out episodes keep their ids in SRC. Scoring runs on SRC.

WHAT IS WRITTEN. Kept episodes are renumbered 0..K-1. Their parquet is rewritten with
the new `episode_index` and a contiguous global `index`. Their videos are symlinks to
SRC's files, because a v2.1 video holds one episode and carries no index. meta/ gets
episodes.jsonl and episodes_stats.jsonl for the kept episodes (the index stats are
shifted), tasks.jsonl, info.json, and subset.json, which records SRC, the held-out ids
and the old id of every kept episode. SRC's stats.json is not copied, because it includes
the held-out frames. The loaders aggregate episodes_stats.jsonl instead.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from vla_common.dataset.split import split


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def subset(src: Path, dst: Path, val: list[int] | None = None) -> dict:
    info = json.loads((src / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise SystemExit(f"{src} is {info.get('codebase_version')}, not v2.1")
    n, held_out, train = split(src, val)
    chunks = int(info["chunks_size"])
    video_keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    episodes = {e["episode_index"]: e for e in read_jsonl(src / "meta" / "episodes.jsonl")}
    ep_stats = {e["episode_index"]: e for e in read_jsonl(src / "meta" / "episodes_stats.jsonl")}
    if sorted(episodes) != list(range(n)):
        raise SystemExit(f"{src}: episodes.jsonl does not list episodes 0..{n - 1}")

    (dst / "meta").mkdir(parents=True)
    out_eps, out_stats, offset = [], [], 0
    for new, old in enumerate(train):
        length = int(episodes[old]["length"])
        f = src / info["data_path"].format(episode_chunk=old // chunks, episode_index=old)
        table = pq.read_table(f)
        if table.num_rows != length:
            raise SystemExit(f"{f}: {table.num_rows} rows, episodes.jsonl says {length}")
        cols = {name: table.column(name) for name in table.column_names}
        cols["episode_index"] = pa.array([new] * length, type=table.schema.field("episode_index").type)
        cols["index"] = pa.array(range(offset, offset + length), type=table.schema.field("index").type)
        out = dst / info["data_path"].format(episode_chunk=new // chunks, episode_index=new)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(cols, schema=table.schema), out)

        for key in video_keys:
            v = src / info["video_path"].format(episode_chunk=old // chunks, video_key=key, episode_index=old)
            link = dst / info["video_path"].format(episode_chunk=new // chunks, video_key=key, episode_index=new)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(v.resolve())

        out_eps.append({**episodes[old], "episode_index": new})
        stats = json.loads(json.dumps(ep_stats[old]["stats"]))
        stats["episode_index"].update({"min": [new], "max": [new], "mean": [float(new)], "std": [0.0]})
        old_from = int(stats["index"]["min"][0])
        for k in ("min", "max", "mean"):
            stats["index"][k] = [type(v)(v - old_from + offset) for v in stats["index"][k]]
        out_stats.append({"episode_index": new, "stats": stats})
        offset += length

    write_jsonl(dst / "meta" / "episodes.jsonl", out_eps)
    write_jsonl(dst / "meta" / "episodes_stats.jsonl", out_stats)
    shutil.copy2(src / "meta" / "tasks.jsonl", dst / "meta" / "tasks.jsonl")
    k = len(train)
    info.update({"total_episodes": k, "total_frames": offset, "total_videos": k * len(video_keys),
                 "total_chunks": (k - 1) // chunks + 1, "splits": {"train": f"0:{k}"}})
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    record = {"source": str(src), "source_episodes": n, "held_out": held_out,
              "kept_source_episodes": train, "frames": offset}
    (dst / "meta" / "subset.json").write_text(json.dumps(record, indent=1))
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", type=Path, required=True, help="LeRobot v2.1 dataset")
    p.add_argument("--dst", type=Path, required=True, help="training subset to create")
    p.add_argument("--val", help="space-separated held-out ids (default: every 10th from 5)")
    p.add_argument("--force", action="store_true", help="replace an existing --dst")
    a = p.parse_args()
    src, dst = a.src.expanduser().resolve(), a.dst.expanduser().resolve()
    if dst.exists():
        if not a.force:
            raise SystemExit(f"{dst} exists (use --force to replace)")
        shutil.rmtree(dst)
    val = [int(x) for x in a.val.split()] if a.val else None
    try:
        r = subset(src, dst, val)
    except BaseException:
        shutil.rmtree(dst, ignore_errors=True)
        raise
    print(f"{dst}\n  {len(r['kept_source_episodes'])} of {r['source_episodes']} episodes, "
          f"{r['frames']} frames\n  held out {len(r['held_out'])}: {' '.join(map(str, r['held_out']))}")


if __name__ == "__main__":
    main()
