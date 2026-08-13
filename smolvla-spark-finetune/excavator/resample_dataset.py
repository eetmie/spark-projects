#!/usr/bin/env python
"""Temporally decimate a LeRobot v3.0 dataset.

Why: this excavator data is recorded at 30 fps but the machine moves slowly
(median joint speed 2-3 deg/s), so consecutive frames carry very little new
visual information. Decimating by stride k gives the policy a bigger visual
delta per step and stretches the wall-clock horizon covered by one action chunk.

The action signal here is a *rate* command (joystick in [-1, 1], correlated
r=0.71-0.88 with measured joint velocity), not a position target. Under
zero-order hold at the lower control rate each kept action is applied for k
times longer, so taking every k-th sample would scale the executed motion.
Averaging the action over the window preserves the integral instead:

    integral(a dt) over window ~= mean(a) * (k / fps_src)

which is why --action-agg defaults to "mean". Use "first" to reproduce naive
subsampling if you want that as a comparison point.

Usage:
    python resample_dataset.py --src <src_root> --dst <dst_root> --stride 3
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from torchcodec.decoders import VideoDecoder

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Set by add_frame/save_episode, so they must not be passed back in.
AUTO_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="source dataset root")
    p.add_argument("--dst", type=Path, required=True, help="destination dataset root")
    p.add_argument("--stride", type=int, required=True, help="keep every k-th frame")
    p.add_argument("--repo-id", default=None, help="repo_id for the new dataset (default: dst dir name)")
    p.add_argument(
        "--action-agg",
        choices=["mean", "first"],
        default="mean",
        help="how to aggregate actions inside a decimation window (default: mean, correct for rate commands)",
    )
    p.add_argument("--vcodec", default="h264", help="video codec for the output (default: h264)")
    p.add_argument("--overwrite", action="store_true", help="delete --dst if it already exists")
    return p.parse_args()


def main():
    args = parse_args()
    src, dst, k = args.src, args.dst, args.stride

    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} already exists (pass --overwrite to replace it)")
        shutil.rmtree(dst)

    src_ds = LeRobotDataset(repo_id="local/src", root=src, video_backend="torchcodec")
    meta = src_ds.meta
    fps_src = meta.fps
    if fps_src % k:
        raise SystemExit(f"stride {k} does not divide source fps {fps_src} evenly; pick a divisor")
    fps_dst = fps_src // k

    video_keys = list(meta.video_keys)
    if len(video_keys) != 1:
        raise SystemExit(f"this script assumes exactly one video key, got {video_keys}")
    vkey = video_keys[0]

    features = {n: f for n, f in meta.features.items() if n not in AUTO_FEATURES}
    df = pd.read_parquet(src / "data" / "chunk-000" / "file-000.parquet")
    episodes = pd.read_parquet(src / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    task = str(src_ds.meta.tasks.index[0])

    print(f"source : {fps_src} fps, {len(df)} frames, {len(episodes)} episodes")
    print(f"output : {fps_dst} fps, stride {k}, action-agg={args.action_agg}")

    out = LeRobotDataset.create(
        repo_id=args.repo_id or dst.name,
        fps=fps_dst,
        features=features,
        root=dst,
        robot_type=meta.robot_type,
        use_videos=True,
        video_backend="torchcodec",
        vcodec=args.vcodec,
    )

    # All episodes live in one mp4 whose frame i lines up with parquet row i, so
    # decoding sequentially per episode beats random access by a wide margin.
    video_path = src / "videos" / vkey / "chunk-000" / "file-000.mp4"
    decoder = VideoDecoder(str(video_path), device="cpu")
    if len(decoder) != len(df):
        raise SystemExit(f"video has {len(decoder)} frames but parquet has {len(df)} rows")

    total_kept = 0
    for _, ep in episodes.iterrows():
        start, stop = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        keep = list(range(start, stop, k))
        frames = decoder.get_frames_at(keep).data  # (N, C, H, W) uint8

        for n, idx in enumerate(keep):
            window_end = min(idx + k, stop)
            if args.action_agg == "mean":
                action = actions[idx:window_end].mean(axis=0)
            else:
                action = actions[idx]
            out.add_frame(
                {
                    vkey: frames[n].permute(1, 2, 0).numpy(),  # CHW -> HWC uint8
                    "observation.state": states[idx],
                    "action": action.astype(np.float32),
                    "task": task,
                }
            )
        out.save_episode()
        total_kept += len(keep)
        print(f"  ep {int(ep['episode_index']):>3}: {stop - start:>4} -> {len(keep):>4} frames", flush=True)

    out.finalize()
    print(f"\ndone: {len(df)} -> {total_kept} frames ({total_kept / len(df):.1%}) at {fps_dst} fps -> {dst}")


if __name__ == "__main__":
    main()
