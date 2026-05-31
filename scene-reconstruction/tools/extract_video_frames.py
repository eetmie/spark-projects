#!/usr/bin/env python3
"""Extract an evenly-sampled JPEG frame set from a video.

Creates the full workspace directory structure and populates images/.

Usage:
    python tools/extract_video_frames.py video.MOV
    python tools/extract_video_frames.py video.MOV --workspace /path/to/scene
    python tools/extract_video_frames.py video.MOV --frames 150 --max-width 1280
    python tools/extract_video_frames.py video.MOV --clear
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, check=True, text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def video_info(video: Path) -> tuple[float, int, int]:
    result = run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(video)],
        capture=True,
    )
    meta = json.loads(result.stdout)
    duration = float(meta["format"]["duration"])
    for s in meta["streams"]:
        if s.get("codec_type") == "video":
            return duration, int(s["width"]), int(s["height"])
    raise RuntimeError("No video stream found")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", help="Input video file")
    parser.add_argument(
        "--workspace", "-w",
        help="Workspace root directory (default: parent directory of video)",
    )
    parser.add_argument(
        "--frames", type=int, default=300,
        help="Approximate number of frames to extract (default: 300)",
    )
    parser.add_argument(
        "--max-width", type=int, default=None,
        help="Scale down to this width if source is wider (default: native resolution)",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Delete existing frame_*.jpg in images/ before extracting",
    )
    args = parser.parse_args()

    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        print(f"Error: video not found: {video}", file=sys.stderr)
        return 1

    workspace = Path(args.workspace).expanduser().resolve() if args.workspace else video.parent
    images_dir = workspace / "images"

    for tool in ("ffmpeg", "ffprobe"):
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            print(f"Error: {tool} not found on PATH", file=sys.stderr)
            return 1

    # Create workspace structure
    for d in ("images", "models", "output"):
        (workspace / d).mkdir(parents=True, exist_ok=True)

    if args.clear:
        removed = sum(1 for f in images_dir.glob("frame_*.jpg") if f.unlink() or True)
        if removed:
            print(f"Removed {removed} existing frames.")

    duration, width, height = video_info(video)
    fps = args.frames / duration
    vf = f"fps={fps:.8f}"
    if args.max_width and width > args.max_width:
        vf += f",scale={args.max_width}:-2"

    print(f"Source    : {video.name}  ({width}x{height}, {duration:.1f}s)")
    print(f"Workspace : {workspace}")
    print(f"Target    : ~{args.frames} frames at {fps:.3f} fps → {images_dir}")

    run([
        "ffmpeg", "-hide_banner", "-y",
        "-i", str(video),
        "-map", "0:v:0",
        "-vf", vf,
        "-q:v", "2",
        "-start_number", "1",
        str(images_dir / "frame_%04d.jpg"),
    ])

    frames = sorted(images_dir.glob("frame_*.jpg"))
    for extra in frames[args.frames:]:
        extra.unlink()
    frames = frames[:args.frames]

    print(f"\nExtracted {len(frames)} frames to {images_dir}/")
    print(f"\nNext: python tools/colmap.py {workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
