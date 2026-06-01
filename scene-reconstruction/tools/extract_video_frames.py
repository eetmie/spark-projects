#!/usr/bin/env python3
"""Extract a JPEG frame set from a video for COLMAP/3DGRUT.

Creates the image folder and populates it with frames for COLMAP/3DGRUT.

Usage:
    python tools/extract_video_frames.py video.MOV
    python tools/extract_video_frames.py video.MOV --workspace /path/to/scene
    python tools/extract_video_frames.py video.MOV --frames 150 --max-width 1280
    python tools/extract_video_frames.py video.MOV --frames 200 --select sharp
    python tools/extract_video_frames.py video.MOV --clear
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
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


def clear_frames(images_dir: Path) -> int:
    removed = 0
    for frame in images_dir.glob("frame_*.jpg"):
        frame.unlink()
        removed += 1
    return removed


def extract_even(video: Path, images_dir: Path, frames: int, max_width: int | None, duration: float) -> list[Path]:
    fps = frames / duration
    vf = f"fps={fps:.8f}"
    if max_width:
        vf += f",scale='min(iw,{max_width})':-2"

    run([
        "ffmpeg", "-hide_banner", "-y",
        "-i", str(video),
        "-map", "0:v:0",
        "-vf", vf,
        "-q:v", "2",
        "-start_number", "1",
        str(images_dir / "frame_%04d.jpg"),
    ])

    extracted = sorted(images_dir.glob("frame_*.jpg"))
    for extra in extracted[frames:]:
        extra.unlink()
    return extracted[:frames]


def extract_candidates(video: Path, candidate_dir: Path, max_width: int | None, candidate_step: int) -> list[Path]:
    vf_parts = []
    if candidate_step > 1:
        vf_parts.append(f"select='not(mod(n\\,{candidate_step}))'")
    if max_width:
        vf_parts.append(f"scale='min(iw,{max_width})':-2")
    vf = ",".join(vf_parts) if vf_parts else "null"

    run([
        "ffmpeg", "-hide_banner", "-y",
        "-i", str(video),
        "-map", "0:v:0",
        "-vf", vf,
        "-vsync", "vfr",
        "-q:v", "2",
        "-start_number", "1",
        str(candidate_dir / "candidate_%06d.jpg"),
    ])
    return sorted(candidate_dir.glob("candidate_*.jpg"))


def score_frame(path: Path, sift) -> tuple[int, float]:
    import cv2

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return 0, 0.0
    keypoints = sift.detect(gray, None)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return len(keypoints), float(laplacian)


def choose_sharp_frames(candidates: list[Path], target_count: int, min_sharpness: float) -> tuple[list[Path], int]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("--select sharp requires OpenCV Python (cv2)") from exc

    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("--select sharp requires an OpenCV build with SIFT support")

    sift = cv2.SIFT_create()
    selected: list[Path] = []
    weak_buckets = 0
    candidate_count = len(candidates)
    bucket_count = min(target_count, candidate_count)

    for bucket in range(bucket_count):
        start = bucket * candidate_count // bucket_count
        end = (bucket + 1) * candidate_count // bucket_count
        bucket_candidates = candidates[start:end] or [candidates[min(start, candidate_count - 1)]]

        scored = [(score_frame(path, sift), path) for path in bucket_candidates]
        strong = [item for item in scored if item[0][1] >= min_sharpness]
        if not strong:
            weak_buckets += 1
            strong = scored
        selected.append(max(strong, key=lambda item: item[0])[1])

    return selected, weak_buckets


def write_selected_frames(selected: list[Path], images_dir: Path) -> None:
    clear_frames(images_dir)
    for index, src in enumerate(selected, start=1):
        shutil.copy2(src, images_dir / f"frame_{index:04d}.jpg")


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
    parser.add_argument(
        "--select", choices=["even", "sharp"], default="even",
        help="Frame selection mode: even = uniform sampling; sharp = best SIFT/Laplacian frame per time bucket (default: even)",
    )
    parser.add_argument(
        "--candidate-step", type=int, default=1, metavar="N",
        help="For --select sharp, score every N-th decoded frame (default: 1)",
    )
    parser.add_argument(
        "--min-sharpness", type=float, default=0.0,
        help="For --select sharp, prefer frames with Laplacian variance at least this value (default: 0)",
    )
    args = parser.parse_args()

    if args.frames <= 0:
        print("Error: --frames must be positive", file=sys.stderr)
        return 1
    if args.candidate_step <= 0:
        print("Error: --candidate-step must be positive", file=sys.stderr)
        return 1
    if args.min_sharpness < 0:
        print("Error: --min-sharpness must be non-negative", file=sys.stderr)
        return 1

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

    images_dir.mkdir(parents=True, exist_ok=True)

    if args.clear:
        removed = clear_frames(images_dir)
        if removed:
            print(f"Removed {removed} existing frames.")

    duration, width, height = video_info(video)
    print(f"Source    : {video.name}  ({width}x{height}, {duration:.1f}s)")
    print(f"Workspace : {workspace}")

    if args.select == "even":
        fps = args.frames / duration
        print(f"Target    : ~{args.frames} even frames at {fps:.3f} fps -> {images_dir}")
        frames = extract_even(video, images_dir, args.frames, args.max_width, duration)
    else:
        print(f"Target    : {args.frames} sharp frames across full video -> {images_dir}")
        with tempfile.TemporaryDirectory(prefix="scene_frames_") as tmp:
            candidate_dir = Path(tmp)
            candidates = extract_candidates(video, candidate_dir, args.max_width, args.candidate_step)
            if not candidates:
                print("Error: no candidate frames extracted", file=sys.stderr)
                return 1
            print(f"Candidates: {len(candidates)} split into {min(args.frames, len(candidates))} time buckets")
            print("Scoring   : SIFT + Laplacian sharpness; this can take a few minutes for long/high-res videos...")
            try:
                selected, weak_buckets = choose_sharp_frames(candidates, args.frames, args.min_sharpness)
            except RuntimeError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            write_selected_frames(selected, images_dir)
            frames = sorted(images_dir.glob("frame_*.jpg"))
            if weak_buckets and args.min_sharpness > 0:
                print(f"Warning   : {weak_buckets} buckets had no frame above --min-sharpness; kept their best frame")

    print(f"\nExtracted {len(frames)} frames to {images_dir}/")
    print(f"\nNext: python tools/colmap.py {workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
