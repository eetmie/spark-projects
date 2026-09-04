#!/usr/bin/env python
"""Build a camera-subset view of a LeRobot v3 dataset, without copying video.

`lerobot-train` has no flag for "train on a subset of the cameras": the policy's
input features come from `dataset_to_policy_features(ds_meta.features)`, so every
`observation.images.*` in `meta/info.json` becomes a visual input. `empty_cameras`
only *adds* slots, it never removes one.

So to train an IR-only model on a two-camera recording, the dataset itself has to
present one camera. This builds that view: `meta/` is rewritten with the unwanted
cameras dropped, while `data/` and the kept `videos/` are **symlinked** to the
source. A 594 MB dataset becomes a few hundred kB of metadata, and the frames the
two runs see stay byte-identical because they are literally the same files.

The `clock.*` columns are left alone: they are diagnostics, and
`dataset_to_policy_features` skips any key that is not `observation*`/`action*`
(feature_utils.py: `else: continue`), so they never reach the policy. Dropping
`clock.cam2_age` from info.json while it remains in the shared parquet would only
risk a schema mismatch for no gain.

Usage:
    python make_camera_variant.py --src ~/Desktop/masi_digging \
        --dst datasets/masi_digging_ir --keep cam1
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

PREFIX = "observation.images."


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="source LeRobot v3 dataset root")
    p.add_argument("--dst", type=Path, required=True, help="variant root to create")
    p.add_argument("--keep", nargs="+", required=True,
                   help="camera short names to keep, e.g. cam1 (or full feature keys)")
    p.add_argument("--force", action="store_true", help="replace an existing --dst")
    return p.parse_args()


def main():
    args = parse_args()
    src, dst = args.src.expanduser().resolve(), args.dst.expanduser().resolve()
    keep = {k if k.startswith(PREFIX) else PREFIX + k for k in args.keep}

    info = json.loads((src / "meta" / "info.json").read_text())
    cams = {k for k in info["features"] if k.startswith(PREFIX)}
    missing = keep - cams
    if missing:
        raise SystemExit(f"cameras {sorted(missing)} not in {src}: has {sorted(cams)}")
    drop = cams - keep
    if not drop:
        raise SystemExit(f"nothing to drop — {src} already has exactly {sorted(keep)}")

    if dst.exists():
        if not args.force:
            raise SystemExit(f"{dst} exists (use --force to replace)")
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)

    # --- meta/info.json: drop the unwanted camera features -------------------
    info["features"] = {k: v for k, v in info["features"].items() if k not in drop}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    # --- meta/stats.json: same ----------------------------------------------
    stats_path = src / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        stats = {k: v for k, v in stats.items() if k not in drop}
        (dst / "meta" / "stats.json").write_text(json.dumps(stats))

    shutil.copy2(src / "meta" / "tasks.parquet", dst / "meta" / "tasks.parquet")

    # --- meta/episodes/*: drop the per-camera video-locator columns ----------
    for pq in sorted((src / "meta" / "episodes").rglob("*.parquet")):
        df = pd.read_parquet(pq)
        cols = [c for c in df.columns
                if not any(c.startswith(f"videos/{d}/") for d in drop)]
        out = dst / "meta" / "episodes" / pq.relative_to(src / "meta" / "episodes")
        out.parent.mkdir(parents=True, exist_ok=True)
        df[cols].to_parquet(out, index=False)

    # --- data/ and the kept videos/: symlink, never copy ---------------------
    (dst / "data").symlink_to(src / "data", target_is_directory=True)
    (dst / "videos").mkdir()
    for cam in sorted(keep):
        (dst / "videos" / cam).symlink_to(src / "videos" / cam, target_is_directory=True)

    print(f"{dst}\n  kept    {sorted(keep)}\n  dropped {sorted(drop)}"
          f"\n  data/ and videos/{sorted(keep)[0]} symlinked to {src}")


if __name__ == "__main__":
    main()
