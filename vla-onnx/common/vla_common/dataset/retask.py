#!/usr/bin/env python
"""Rename an instruction string in a LeRobot v3 dataset, in place.

Needed because the excavator recordings disagree on phrasing for what is the same job:
masi_digging/_clean say "move the sand to the container", the dry sets say "move sand to
container". The policy conditions on the language embedding, so merging them as-is trains
two unrelated instructions on what should be one, splitting the sand data between them.

The string lives in exactly two places, and `data/` is not one of them -- the frame tables
carry only the integer `task_index`. So this is a metadata-only rewrite: no video is
touched, no frame is rewritten, and it takes a second on a 100k-frame dataset.

  meta/tasks.parquet          the string is the INDEX, task_index the column
  meta/episodes/*.parquet     a `tasks` column, one-element array of strings per episode

Refuses to merge two instructions into one. Renaming A -> B where B already exists would
collapse two task_index values, which changes what `task_index` means in every data shard
and is a different operation from renaming; it is not done silently here.

Usage:
    python retask_dataset.py --root ~/Desktop/masi_digging_clean \
        --from "move the sand to the container" --to "move sand to container"
    python retask_dataset.py --root <ds> --from A --to B --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="dataset root")
    p.add_argument("--from", dest="old", required=True, help="instruction to replace")
    p.add_argument("--to", dest="new", required=True, help="replacement instruction")
    p.add_argument("--dry-run", action="store_true", help="report, change nothing")
    return p.parse_args()


def main():
    a = parse_args()
    root = a.root.expanduser().resolve()
    tasks_f = root / "meta" / "tasks.parquet"
    if not tasks_f.is_file():
        raise SystemExit(f"no tasks.parquet under {root}")

    tasks = pd.read_parquet(tasks_f)
    names = list(tasks.index) if tasks.index.name == "task" else list(tasks["task"])
    if a.old not in names:
        raise SystemExit(f"{root} has no task {a.old!r}; it has {names}")
    if a.new in names:
        raise SystemExit(
            f"{root} already has {a.new!r}. Renaming onto an existing instruction would "
            f"merge two task_index values and silently change what task_index means in "
            f"every data shard -- do that deliberately, not with this script.")

    eps_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    n_eps = 0
    for f in eps_files:
        df = pd.read_parquet(f)
        n_eps += int(df["tasks"].apply(lambda v: a.old in list(v)).sum())

    print(f"{root}\n  {a.old!r}\n    -> {a.new!r}")
    print(f"  tasks.parquet: 1 entry | meta/episodes: {n_eps} episodes across "
          f"{len(eps_files)} shard(s) | data/: untouched (holds task_index only)")
    if a.dry_run:
        print("  dry run — nothing written")
        return

    tasks = tasks.rename(index={a.old: a.new}) if tasks.index.name == "task" \
        else tasks.assign(task=[a.new if t == a.old else t for t in tasks["task"]])
    tasks.to_parquet(tasks_f)

    for f in eps_files:
        df = pd.read_parquet(f)
        df["tasks"] = df["tasks"].apply(
            lambda v: [a.new if t == a.old else t for t in list(v)])
        df.to_parquet(f, index=False)

    print("  written")


if __name__ == "__main__":
    main()
