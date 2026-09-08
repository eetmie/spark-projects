#!/usr/bin/env python
"""Resolve a camera-subset view of a recording, building it only when it must be built.

`--cameras cam1` on the trainer has to become a `--dataset.root`, because lerobot-train
has no camera-subset flag: every `observation.images.*` in `meta/info.json` becomes a
policy input. `camera_variant` makes that view; this decides whether the one on disk is
still usable, and it is the part that used to be done by hand and got skipped.

WHY A STAMP. `camera_variant` SYMLINKS `data/` and the kept `videos/<cam>/` but COPIES
`meta/` -- info.json, stats.json, tasks.parquet, and every rewritten episodes shard. That
asymmetry is what makes a view cost a few hundred kB instead of 594 MB, and it is also a
trap: when the source recording grows, new frames appear through the symlink while the
episode table stays frozen at the old count. Training then derives its split from the
stale count and silently never touches the new episodes -- no error, a plausible loss
curve. These recordings do grow (masi_digging 82 -> 189, masi_digging_dry_2 78 -> 242),
so the view records what the source looked like when it was built, and we compare.

GROWN IS REBUILT, SHRUNK IS REFUSED. Appending preserves episode indices, so a rebuild
after growth leaves every existing checkpoint's `dataset.episodes` naming the same
recordings -- safe, automatic. Dropping episodes RENUMBERS the survivors, which makes
every split derived from an older checkpoint a lie about which recordings were held out.
That is the failure the old `x if x < 83 else x-8` remap existed to paper over, so here it
stops the pipeline instead of being documented.

Usage:
    python -m vla_common.dataset.view --src ~/Desktop/masi_digging_dry_2 --cameras cam1
    python -m vla_common.dataset.view --src <ds> --cameras cam1 --if-stale refuse
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from vla_common.paths import VLA_ONNX

PREFIX = "observation.images."
STAMP = ".variant.json"

# Fields of the source's info.json that decide whether a view still describes it.
FINGERPRINT_KEYS = ("total_episodes", "total_frames", "total_tasks")


def views_root() -> Path:
    """Where camera views live: shared, not owned by one playbook.

    They used to sit in `smolvla/datasets/`, which is why the X-VLA runner reached across
    into another playbook's tree to find its training data.
    """
    return VLA_ONNX / "datasets"


def _cam_keys(cameras) -> list[str]:
    return sorted(c if c.startswith(PREFIX) else PREFIX + c for c in cameras)


def _short(cam: str) -> str:
    return cam[len(PREFIX):] if cam.startswith(PREFIX) else cam


def view_name(src: Path, cameras) -> str:
    """Deterministic from (source, cameras), so two callers never disagree.

    `__` separates because source names contain single underscores; the old hand-written
    `masi_digging_dry2_ir` both mangled the source name and baked in one robot's word for
    a camera, which the pipeline is supposed to know nothing about.
    """
    return f"{Path(src).name}__{'+'.join(_short(c) for c in _cam_keys(cameras))}"


def source_cameras(src: Path) -> list[str]:
    info = json.loads((Path(src) / "meta" / "info.json").read_text())
    return sorted(k for k in info["features"] if k.startswith(PREFIX))


def fingerprint(src: Path) -> dict:
    info = json.loads((Path(src) / "meta" / "info.json").read_text())
    return {k: info.get(k) for k in FINGERPRINT_KEYS}


def read_stamp(view: Path) -> dict | None:
    p = Path(view) / STAMP
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def write_stamp(view: Path, src: Path, keep: list[str]) -> None:
    (Path(view) / STAMP).write_text(json.dumps({
        "src": str(Path(src).resolve()),
        "keep": keep,
        "src_fingerprint": fingerprint(src),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))


def source_of(view: Path) -> Path:
    """The recording a view was cut from.

    Stamp first; otherwise follow `data/`, which `camera_variant` symlinks into the
    source. A plain (non-view) dataset resolves to itself, so callers need no special case.
    """
    view = Path(view)
    stamp = read_stamp(view)
    if stamp and stamp.get("src"):
        return Path(stamp["src"])
    return (view / "data").resolve().parent


def staleness(view: Path, src: Path, cameras) -> str:
    """fresh | grew | shrank | retasked | recut | unstamped | missing"""
    view = Path(view)
    if not view.exists():
        return "missing"
    stamp = read_stamp(view)
    if stamp is None:
        return "unstamped"
    if stamp.get("keep") != _cam_keys(cameras):
        return "recut"
    was, now = stamp.get("src_fingerprint") or {}, fingerprint(src)
    if was == now:
        return "fresh"
    if (now.get("total_tasks") or 0) < (was.get("total_tasks") or 0):
        return "retasked"
    if all((now.get(k) or 0) >= (was.get(k) or 0) for k in FINGERPRINT_KEYS):
        return "grew"
    return "shrank"


def build(src: Path, dst: Path, keep: list[str], force: bool) -> None:
    cmd = [sys.executable, "-m", "vla_common.dataset.camera_variant",
           "--src", str(src), "--dst", str(dst), "--keep", *keep]
    if force:
        cmd.append("--force")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"camera_variant failed:\n{r.stdout}{r.stderr}")
    write_stamp(dst, src, keep)


def resolve(src: Path, cameras=None, if_stale: str = "rebuild") -> Path:
    """The dataset root to train on. Builds or rebuilds the view when needed.

    Returns the SOURCE itself when the requested cameras are all of them -- there is
    nothing to drop, and `camera_variant` rightly refuses to make an identity view.
    """
    src = Path(src).expanduser().resolve()
    if not (src / "meta" / "info.json").exists():
        raise SystemExit(f"{src} is not a LeRobot v3 dataset (no meta/info.json)")

    have = source_cameras(src)
    keep = _cam_keys(cameras) if cameras else have
    missing = [c for c in keep if c not in have]
    if missing:
        raise SystemExit(f"{src.name} has {[_short(c) for c in have]}, "
                         f"not {[_short(c) for c in missing]}")
    if set(keep) == set(have):
        return src

    dst = views_root() / view_name(src, keep)
    state = staleness(dst, src, keep)

    if state == "fresh":
        return dst
    if state in ("shrank", "retasked") and if_stale != "force":
        raise SystemExit(
            f"!! {src.name} has FEWER episodes/tasks than when {dst.name} was built:\n"
            f"     built against {read_stamp(dst)['src_fingerprint']}\n"
            f"     source is now {fingerprint(src)}\n"
            f"   Dropping episodes renumbers the survivors, so every checkpoint trained on\n"
            f"   this view now names different recordings than it did. Rebuilding makes\n"
            f"   those runs uninterpretable rather than merely stale.\n"
            f"   Rebuild anyway with --if-stale force if that is what you mean.")
    if state == "unstamped" and if_stale == "reuse":
        return dst
    if state == "unstamped" and if_stale == "refuse":
        raise SystemExit(f"!! {dst} predates the build stamp, so it cannot be checked "
                         f"against {src.name}. Rebuild it, or pass --if-stale reuse.")

    verb = {"missing": "building", "grew": "rebuilding (source grew)",
            "recut": "rebuilding (different cameras)",
            "unstamped": "rebuilding (no build stamp)",
            "shrank": "rebuilding (FORCED over a shrunk source)",
            "retasked": "rebuilding (FORCED over a changed task set)"}[state]
    print(f"{verb}: {dst.name}", file=sys.stderr)
    if state == "grew":
        was = read_stamp(dst)["src_fingerprint"]
        print(f"  {was.get('total_episodes')} -> {fingerprint(src)['total_episodes']} episodes",
              file=sys.stderr)
    views_root().mkdir(parents=True, exist_ok=True)
    build(src, dst, keep, force=state != "missing")
    return dst


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="source recording")
    p.add_argument("--cameras", nargs="*", default=None,
                   help="camera short names to keep (default: all of them, no view built)")
    p.add_argument("--if-stale", choices=("rebuild", "reuse", "refuse", "force"), default="rebuild",
                   help="unstamped view: rebuild (default), reuse it, or refuse. "
                        "force also rebuilds over a SHRUNK source, which invalidates every "
                        "split derived from a checkpoint trained on the old view")
    return p.parse_args()


def main():
    args = parse_args()
    cams = args.cameras
    if cams and len(cams) == 1 and "," in cams[0]:
        cams = cams[0].split(",")
    # stdout is the resolved root and nothing else, so a shell can capture it.
    print(resolve(args.src, cams, args.if_stale))


if __name__ == "__main__":
    main()
