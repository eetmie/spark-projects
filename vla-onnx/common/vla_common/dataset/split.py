#!/usr/bin/env python
"""Derive a train/held-out episode split from a LeRobot v3 dataset.

This was nine byte-identical lines of bash in two places -- `split_for()` in both
`smolvla/excavator/run_digging.sh` and `xvla/excavator/run_digging.sh`. It has zero
model coupling: it reads `meta/info.json` and counts.

THE RULE. Held out is every 10th episode from 5, which lands at roughly 10% spread
across the recording rather than one stretch of it. Everything else trains.

THE EPISODE COUNT IS READ, NEVER TYPED. That is the whole point of the file. The rule
used to be spelled `range(82)` inline; when `masi_digging` grew to 189 episodes that
would have trained on episodes 0-81 only and silently thrown away every new one, with
no error and a plausible-looking loss curve. These recordings do grow -- masi_digging
82 -> 189, masi_digging_dry_2 78 -> 242 -- so a literal is a bug with a delay fuse.

DROPPING EPISODES RENUMBERS THE SURVIVORS. A trimmed or filtered variant does not hold
out the same recordings as its source under the same rule, so a hand-written override
that was correct for one dataset is quietly wrong for the other. `--val` exists for
that case, and `split()` validates it against the episode count rather than trusting it.

Usage:
    python -m vla_common.dataset.split --root ~/Desktop/masi_digging_dry_2
    python -m vla_common.dataset.split --root <ds> --emit train    # for --dataset.episodes
    python -m vla_common.dataset.split --root <ds> --val "5 15 25"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HOLDOUT_START = 5
HOLDOUT_EVERY = 10


def episode_count(root: Path) -> int:
    """`total_episodes` from the dataset's own metadata."""
    info = json.loads((Path(root) / "meta" / "info.json").read_text())
    return int(info["total_episodes"])


def holdout(n: int, start: int = HOLDOUT_START, every: int = HOLDOUT_EVERY) -> list[int]:
    """The held-out ids for an `n`-episode dataset: every `every`th from `start`."""
    return list(range(start, n, every))


def split(root: Path, val: list[int] | None = None) -> tuple[int, list[int], list[int]]:
    """Return (n_episodes, held_out, train) for a dataset root.

    `val` overrides the rule. It is checked against the episode count because the
    failure it guards is silent: an override carried over from a differently sized
    variant names episodes that do not exist, and the complement then trains on
    everything -- including the recordings the override meant to protect.
    """
    n = episode_count(root)
    val = holdout(n) if val is None else sorted(set(val))
    out_of_range = [e for e in val if not 0 <= e < n]
    if out_of_range:
        raise SystemExit(
            f"{root}: held-out episodes {out_of_range} are outside 0..{n - 1}. "
            f"The dataset has {n} episodes -- an override from a different variant?"
        )
    train = [e for e in range(n) if e not in set(val)]
    if not train:
        raise SystemExit(f"{root}: the held-out set covers every episode, nothing to train on")
    return n, val, train


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="LeRobot v3 dataset root")
    p.add_argument("--val", default=None,
                   help="space-separated held-out ids, overriding the every-10th-from-5 rule")
    p.add_argument("--emit", choices=("summary", "train", "val", "count"), default="summary",
                   help="train prints a JSON list for --dataset.episodes; val prints ids")
    return p.parse_args()


def main():
    args = parse_args()
    val = [int(x) for x in args.val.split()] if args.val else None
    n, val, train = split(args.root.expanduser(), val)

    if args.emit == "count":
        print(n)
    elif args.emit == "train":
        # JSON list, which is the spelling draccus wants for --dataset.episodes.
        print("[" + ",".join(str(e) for e in train) + "]")
    elif args.emit == "val":
        print(" ".join(str(e) for e in val))
    else:
        print(f"{args.root}\n  {n} episodes\n  train    {len(train)}\n"
              f"  held out {len(val)}: {' '.join(str(e) for e in val)}")


if __name__ == "__main__":
    main()
