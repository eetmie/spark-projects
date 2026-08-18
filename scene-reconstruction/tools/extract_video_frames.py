#!/usr/bin/env python3
"""Extract a JPEG frame set from any video for COLMAP/3DGRUT.

Camera-agnostic: works for iPhone/Android, GoPro and other action cams, drones,
or a plain .mp4/.mov. Populates images/ with frames and writes a
capture_metadata.json provenance sidecar next to it — container/codec/resolution/
fps, camera identity, and any telemetry stream (GoPro GPMD, Android/Insta360
CAMM, Apple mebx, Sony rtmd) — plus a suggested COLMAP camera model.

Usage:
    python tools/extract_video_frames.py video.MOV
    python tools/extract_video_frames.py video.MOV --workspace /path/to/scene
    python tools/extract_video_frames.py video.MOV --frames 150 --max-width 1280
    python tools/extract_video_frames.py video.MOV --frames 200 --select sharp
    python tools/extract_video_frames.py video.MOV --frames 200 --select sharp --score-scale 2
    python tools/extract_video_frames.py video.MOV --clear
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, check=True, text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


# Timed-metadata / telemetry stream identifiers (matched on codec_tag_string),
# mapped to a human-readable description. Any camera that writes one of these is
# handled the same way — this is what makes the pipeline camera-agnostic.
TELEMETRY_TAGS = {
    "gpmd": "GoPro GPMD telemetry (GPS, gyro, accelerometer)",
    "camm": "Camera Motion Metadata (IMU; Android / Insta360 / Google)",
    "mebx": "Apple QuickTime timed metadata (may include motion / quaternion)",
    "rtmd": "Sony real-time metadata",
}


def probe(video: Path) -> dict:
    """Return the full ffprobe format+streams JSON for a video."""
    result = run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(video)],
        capture=True,
    )
    return json.loads(result.stdout)


def video_dimensions(meta: dict) -> tuple[float, int, int]:
    duration = float(meta["format"].get("duration", 0.0))
    for s in meta["streams"]:
        if s.get("codec_type") == "video":
            return duration, int(s["width"]), int(s["height"])
    raise RuntimeError("No video stream found")


def video_info(video: Path) -> tuple[float, int, int]:
    return video_dimensions(probe(video))


def _clean_tag(value: object) -> str:
    """Strip the control chars GoPro/QuickTime prepend to handler_name."""
    return value.strip().strip("\x0b").strip() if isinstance(value, str) else ""


def _fps(stream: dict) -> float | None:
    for key in ("avg_frame_rate", "r_frame_rate"):
        val = stream.get(key)
        if val and "/" in val:
            num, den = val.split("/")
            if float(den) != 0:
                return round(float(num) / float(den), 3)
    return None


def describe_capture(video: Path, meta: dict) -> dict:
    """Build a camera-agnostic provenance record from ffprobe metadata.

    Captures container/codec/resolution/fps, camera identity, and any
    telemetry/data streams, then suggests a COLMAP camera model. Nothing here
    is camera-specific — GoPro, iPhone, drone and plain files are all described
    the same way; only the detected label and suggestion differ.
    """
    fmt = meta.get("format", {})
    ftags = fmt.get("tags", {})
    streams = meta.get("streams", [])
    vstream = next((s for s in streams if s.get("codec_type") == "video"), {})

    make = ftags.get("make") or ftags.get("com.apple.quicktime.make")
    model = ftags.get("model") or ftags.get("com.apple.quicktime.model")
    firmware = ftags.get("firmware")
    handlers = " ".join(_clean_tag(s.get("tags", {}).get("handler_name")) for s in streams)
    blob = " ".join(str(x) for x in (make, model, firmware, handlers)).lower()

    if "gopro" in blob or (firmware or "").startswith("HD"):
        detected = "GoPro"
    elif "insta360" in blob:
        detected = "Insta360"
    elif "dji" in blob:
        detected = "DJI"
    elif "apple" in blob or "iphone" in blob:
        detected = "Apple/iPhone"
    else:
        detected = make or "unknown"

    telemetry, data_streams = [], []
    for s in streams:
        if s.get("codec_type") != "data":
            continue
        tag = (s.get("codec_tag_string") or "").strip()
        entry = {
            "index": s.get("index"),
            "codec_tag": tag,
            "handler": _clean_tag(s.get("tags", {}).get("handler_name")),
        }
        label = TELEMETRY_TAGS.get(tag)
        (telemetry if label else data_streams).append(
            {**entry, "kind": label} if label else entry
        )

    if detected in ("GoPro", "Insta360", "DJI"):
        suggestion = {
            "camera_model": "OPENCV_FISHEYE", "mapper": "global",
            "reason": f"{detected} wide-angle / action-cam footage",
        }
    else:
        suggestion = {
            "camera_model": "OPENCV", "mapper": "incremental",
            "reason": "standard lens assumed; use OPENCV_FISHEYE for iPhone 0.5x / wide-angle",
        }

    duration, width, height = video_dimensions(meta)
    return {
        "source": video.name,
        "container": fmt.get("format_name"),
        "duration_sec": round(duration, 3),
        "creation_time": ftags.get("creation_time"),
        "video": {
            "codec": vstream.get("codec_name"),
            "width": width,
            "height": height,
            "fps": _fps(vstream),
            "pix_fmt": vstream.get("pix_fmt"),
        },
        "camera": {"make": make, "model": model, "firmware": firmware, "detected": detected},
        "telemetry_streams": telemetry,
        "data_streams": data_streams,
        "suggested_colmap": suggestion,
    }


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



# ---------------------------------------------------------------------------
# Fast blur scoring: one decode pass, no intermediate JPEGs, cached.
# Laplacian-only at reduced resolution. Benchmarked ~120x faster than the
# full-res SIFT path, with picks that are marginally *sharper* (the legacy
# sort key led on SIFT keypoint count, which measures texture density rather
# than focus). Use --legacy-sift to restore the old behaviour.
# ---------------------------------------------------------------------------

CACHE_VERSION = 1


def _score_cache_key(video: Path, candidate_step: int, score_scale: int) -> str:
    st = video.stat()
    raw = f"{CACHE_VERSION}|{video.resolve()}|{st.st_size}|{int(st.st_mtime)}|{candidate_step}|{score_scale}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _score_with_pyav(video: Path, candidate_step: int, score_scale: int) -> tuple[list[int], list[float]]:
    import av
    import cv2

    container = av.open(str(video))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    stream.thread_count = 0
    indices: list[int] = []
    scores: list[float] = []
    try:
        for n, frame in enumerate(container.decode(video=0)):
            if candidate_step > 1 and n % candidate_step:
                continue
            gray = frame.to_ndarray(format="gray")
            if score_scale > 1:
                h, w = gray.shape
                gray = cv2.resize(gray, (w // score_scale, h // score_scale),
                                  interpolation=cv2.INTER_AREA)
            indices.append(n)
            scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    finally:
        container.close()
    return indices, scores


def _score_with_ffmpeg_pipe(
    video: Path, candidate_step: int, score_scale: int, width: int, height: int,
) -> tuple[list[int], list[float]]:
    """Score by piping downscaled grayscale frames straight out of ffmpeg.

    Needs no extra Python packages — ffmpeg does the decode and the downscale.
    """
    import cv2
    import numpy as np

    w, h = max(width // score_scale, 16), max(height // score_scale, 16)
    vf = f"scale={w}:{h},format=gray"
    if candidate_step > 1:
        vf = f"select='not(mod(n\\,{candidate_step}))'," + vf
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-map", "0:v:0",
        "-vf", vf, "-vsync", "vfr",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    frame_bytes = w * h
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 8)
    indices: list[int] = []
    scores: list[float] = []
    i = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            gray = np.frombuffer(buf, np.uint8).reshape(h, w)
            indices.append(i * candidate_step)
            scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()
    return indices, scores


def score_video(
    video: Path,
    width: int,
    height: int,
    candidate_step: int = 1,
    score_scale: int = 4,
    cache_path: Path | None = None,
) -> tuple[list[int], list[float]]:
    """Score every candidate frame's sharpness in a single decode pass."""
    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("--select sharp requires OpenCV Python (cv2)") from exc

    key = _score_cache_key(video, candidate_step, score_scale)
    if cache_path and cache_path.exists():
        try:
            blob = json.loads(cache_path.read_text())
            if blob.get("key") == key:
                print(f"Scoring   : cache hit ({len(blob['scores'])} frames) - skipping decode")
                return blob["indices"], blob["scores"]
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    try:
        import av  # noqa: F401
        backend = "PyAV"
    except ImportError:
        backend = "ffmpeg-pipe"

    print(f"Scoring   : Laplacian at 1/{score_scale} resolution via {backend}...")
    started = time.perf_counter()
    if backend == "PyAV":
        indices, scores = _score_with_pyav(video, candidate_step, score_scale)
    else:
        indices, scores = _score_with_ffmpeg_pipe(video, candidate_step, score_scale, width, height)
    elapsed = time.perf_counter() - started
    if scores:
        print(f"Scoring   : {len(scores)} frames in {elapsed:.1f}s "
              f"({elapsed / len(scores) * 1000:.2f} ms/frame)")

    if cache_path and scores:
        try:
            cache_path.write_text(json.dumps({"key": key, "indices": indices, "scores": scores}))
        except OSError:
            pass
    return indices, scores


def choose_sharp_indices(
    indices: list[int], scores: list[float], target_count: int, min_sharpness: float,
) -> tuple[list[int], int]:
    """Pick the sharpest frame per time bucket (same bucketing as the legacy path)."""
    n = len(indices)
    selected: list[int] = []
    weak_buckets = 0
    bucket_count = min(target_count, n)
    for bucket in range(bucket_count):
        start = bucket * n // bucket_count
        end = (bucket + 1) * n // bucket_count
        rng = list(range(start, end)) or [min(start, n - 1)]
        strong = [i for i in rng if scores[i] >= min_sharpness]
        if not strong:
            weak_buckets += 1
            strong = rng
        selected.append(indices[max(strong, key=lambda i: scores[i])])
    return selected, weak_buckets


def extract_selected_frames(
    video: Path, selected: list[int], images_dir: Path, max_width: int | None,
) -> list[Path]:
    """Write only the chosen frames at full resolution, in one ffmpeg pass."""
    clear_frames(images_dir)
    terms = "+".join(f"eq(n\\,{n})" for n in sorted(selected))
    vf = f"select='{terms}'"
    if max_width:
        vf += f",scale='min(iw,{max_width})':-2"
    started = time.perf_counter()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(vf)
        script = Path(fh.name)
    try:
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video), "-map", "0:v:0",
            "-filter_script:v", str(script),
            "-vsync", "vfr", "-q:v", "2", "-start_number", "1",
            str(images_dir / "frame_%04d.jpg"),
        ])
    finally:
        script.unlink(missing_ok=True)
    written = sorted(images_dir.glob("frame_*.jpg"))
    print(f"Extract   : {len(written)} full-res frames in {time.perf_counter() - started:.1f}s")
    return written


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
        help="Frame selection mode: even = uniform sampling; sharp = sharpest (least-blurred) frame per time bucket (default: even)",
    )
    parser.add_argument(
        "--candidate-step", type=int, default=1, metavar="N",
        help="For --select sharp, score every N-th decoded frame (default: 1)",
    )
    parser.add_argument(
        "--min-sharpness", type=float, default=0.0,
        help="For --select sharp, prefer frames with Laplacian variance at least this value (default: 0)",
    )
    parser.add_argument(
        "--score-scale", type=int, default=4, metavar="N",
        help="For --select sharp, score at 1/N resolution (default: 4; 1 = full res)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="For --select sharp, ignore and overwrite the cached sharpness scores",
    )
    parser.add_argument(
        "--legacy-sift", action="store_true",
        help="For --select sharp, use the original full-res SIFT+Laplacian scorer (~120x slower)",
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
    if args.score_scale <= 0:
        print("Error: --score-scale must be positive", file=sys.stderr)
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

    meta = probe(video)
    duration, width, height = video_dimensions(meta)
    print(f"Source    : {video.name}  ({width}x{height}, {duration:.1f}s)")
    print(f"Workspace : {workspace}")

    if args.select == "even":
        fps = args.frames / duration
        print(f"Target    : ~{args.frames} even frames at {fps:.3f} fps -> {images_dir}")
        frames = extract_even(video, images_dir, args.frames, args.max_width, duration)
    else:
        print(f"Target    : {args.frames} sharp frames across full video -> {images_dir}")
        if args.legacy_sift:
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
        else:
            cache_path = None if args.no_cache else workspace / ".sharpness_cache.json"
            try:
                indices, scores = score_video(
                    video, width, height,
                    candidate_step=args.candidate_step,
                    score_scale=args.score_scale,
                    cache_path=cache_path,
                )
            except RuntimeError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            if not scores:
                print("Error: no candidate frames scored", file=sys.stderr)
                return 1
            print(f"Candidates: {len(scores)} split into {min(args.frames, len(scores))} time buckets")
            selected, weak_buckets = choose_sharp_indices(
                indices, scores, args.frames, args.min_sharpness
            )
            frames = extract_selected_frames(video, selected, images_dir, args.max_width)
        if weak_buckets and args.min_sharpness > 0:
            print(f"Warning   : {weak_buckets} buckets had no frame above --min-sharpness; kept their best frame")

    metadata = describe_capture(video, meta)
    metadata["extraction"] = {
        "frames_written": len(frames),
        "frames_requested": args.frames,
        "select": args.select,
        "max_width": args.max_width,
    }
    meta_path = workspace / "capture_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nExtracted {len(frames)} frames to {images_dir}/")
    print(f"Metadata  : {meta_path.name}  (camera: {metadata['camera']['detected']})")
    if metadata["telemetry_streams"]:
        print(f"Telemetry : {', '.join(t['kind'] for t in metadata['telemetry_streams'])}")
    sug = metadata["suggested_colmap"]
    hint = f"--camera-model {sug['camera_model']}"
    if sug["mapper"] != "incremental":
        hint += f" --mapper {sug['mapper']}"
    print(f"COLMAP    : suggested {hint}  ({sug['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
