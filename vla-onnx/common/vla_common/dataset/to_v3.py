#!/usr/bin/env python
"""Write a LeRobot v3.0 copy of a LeRobot v2.1 dataset, and check it.

    python -m vla_common.dataset.to_v3 SRC [--dst SRC_v3]

The other half of `to_v21`. `from_native` writes v2.1 because OpenPI and GR00T read only
that; the LeRobot policies (SmolVLA, X-VLA, EVO1) and `lerobot-dataset-viz` read v3.
`from_native --v3` calls this after writing.

Uses LeRobot's own `convert_dataset_v21_to_v30` steps, pointed at SRC (read-only) and DST.
Its `convert_dataset()` entry point is not used: it converts in place, renaming SRC.
Video is joined with a stream copy, not re-encoded. Extra meta (native_source.json) and
native/ are copied over unchanged.

CHECK. LeRobot loads DST; frame/episode counts, tasks, every state/action/clock value, and
the first and last frame of every episode's video must match SRC exactly.

Needs lerobot >= 0.4 (the lerobot 0.6.1 venv).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def convert(src: Path, dst: Path) -> None:
    from lerobot.datasets.utils import DEFAULT_DATA_FILE_SIZE_IN_MB, DEFAULT_VIDEO_FILE_SIZE_IN_MB
    from lerobot.scripts import convert_dataset_v21_to_v30 as c

    c.validate_local_dataset_version(src)
    c.convert_info(src, dst, DEFAULT_DATA_FILE_SIZE_IN_MB, DEFAULT_VIDEO_FILE_SIZE_IN_MB)
    c.convert_tasks(src, dst)
    episodes = c.convert_data(src, dst, DEFAULT_DATA_FILE_SIZE_IN_MB)
    videos = c.convert_videos(src, dst, DEFAULT_VIDEO_FILE_SIZE_IN_MB)
    c.convert_episodes_metadata(src, dst, episodes, videos)

    known = {"info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl", "stats.json"}
    for f in (src / "meta").iterdir():
        if f.is_file() and f.name not in known:
            shutil.copy2(f, dst / "meta" / f.name)
    if (src / "native").is_dir():
        shutil.copytree(src / "native", dst / "native")


def decode(mp4: Path, t: float) -> np.ndarray:
    import av

    with av.open(str(mp4)) as box:
        s = box.streams.video[0]
        box.seek(int(t / s.time_base), stream=s, backward=True)
        for frame in box.decode(s):
            if frame.time >= t - 1e-4:
                return frame.to_ndarray(format="rgb24")
    raise ValueError(f"{mp4}: no frame at {t}")


def check(src: Path, dst: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = json.loads((src / "meta/info.json").read_text())
    eps = [json.loads(l) for l in (src / "meta/episodes.jsonl").read_text().splitlines()]
    ds = LeRobotDataset(dst.name, root=dst)
    assert ds.meta.total_episodes == len(eps), (ds.meta.total_episodes, len(eps))
    assert ds.meta.total_frames == info["total_frames"], (ds.meta.total_frames, info["total_frames"])

    cols = [k for k, v in info["features"].items() if v["dtype"] not in ("video", "image")]
    new = ds.hf_dataset.with_format("numpy")
    for e in eps:
        ep = e["episode_index"]
        chunk = ep // info["chunks_size"]
        old = pq.read_table(src / info["data_path"].format(episode_chunk=chunk, episode_index=ep))
        m = ds.meta.episodes[ep]
        assert m["tasks"] == e["tasks"], (ep, m["tasks"], e["tasks"])
        rows = new[m["dataset_from_index"]:m["dataset_to_index"]]
        for k in cols:
            a = np.asarray(old[k].to_pylist())
            b = np.asarray(rows[k]).reshape(a.shape)
            assert np.array_equal(a, b), f"episode {ep} {k} differs"
        for cam in ds.meta.video_keys:
            mp4_old = src / info["video_path"].format(episode_chunk=chunk, video_key=cam, episode_index=ep)
            mp4_new = dst / ds.meta.get_video_file_path(ep, cam)
            t0 = m[f"videos/{cam}/from_timestamp"]
            for t in (0.0, (e["length"] - 1) / info["fps"]):
                assert np.array_equal(decode(mp4_old, t), decode(mp4_new, t0 + t)), f"episode {ep} {cam} t={t}"
    print(f"check OK: {len(eps)} episodes, {info['total_frames']} frames, {len(cols)} columns, "
          f"first/last frame of {len(ds.meta.video_keys)} cameras per episode")


def to_v3(src: Path, dst: Path | None = None) -> Path:
    src = src.resolve()
    dst = (dst or src.with_name(f"{src.name}_v3")).resolve()
    if dst.exists():
        raise SystemExit(f"{dst} exists; choose a new destination")
    convert(src, dst)
    check(src, dst)
    size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    print(f"wrote {dst}: {size / 2**30:.2f} GB")
    return dst


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src", type=Path, help="LeRobot v2.1 dataset")
    p.add_argument("--dst", type=Path, help="default: SRC_v3 next to SRC")
    a = p.parse_args()
    to_v3(a.src, a.dst)


if __name__ == "__main__":
    main()
