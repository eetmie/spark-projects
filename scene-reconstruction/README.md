# Scene Reconstruction Pipeline

General **video → PLY / USD** pipeline: Video -> COLMAP -> 3DGRUT Gaussian splat -> SuperSplat cleanup/compression -> Isaac Sim NuRec USDZ on DGX Spark.

Any video works — iPhone/Android, GoPro or other action cams, drones, or a plain `.mp4`/`.mov`. Frame extraction is codec-agnostic (ffmpeg); telemetry/data streams (GoPro GPMD, CAMM, etc.) are recorded to `capture_metadata.json` and otherwise ignored by the solve. For wide-angle footage (GoPro, iPhone 0.5x) use the `OPENCV_FISHEYE` camera model in COLMAP — the extractor prints the suggested model per clip. See step 1.

## Workspace Layout

The normal pipeline only creates the folders and files that are used by the next step:

```text
my_scene/
├── video.MOV        <- source video (iPhone, GoPro, any .mp4/.mov)
├── images/          <- ffmpeg-extracted frames for COLMAP and 3DGRUT
├── capture_metadata.json  <- provenance: codec/res/fps, camera, telemetry streams
├── database.db      <- COLMAP database
├── sparse/0/        <- COLMAP sparse reconstruction and camera parameters
├── models/          <- 3DGRUT checkpoints
├── raw.ply          <- direct 3DGRUT export
├── cleaned.ply      <- SuperSplat-trimmed/compressed export, SH degree 3
└── scene.usdz       <- Isaac Sim / Omniverse NuRec bundle
```

## Quick Start

Run commands from this directory:

```bash
cd /home/masi-pgx/spark-projects/scene-reconstruction
```

### 1. Extract Frames

Put the video in a scene folder, then extract frames (works for iPhone `.MOV`, GoPro `.MP4`, or any video):

```bash
mkdir -p /path/to/my_scene
cp /path/to/video.MOV /path/to/my_scene/
python tools/extract_video_frames.py /path/to/my_scene/video.MOV
```

This creates `/path/to/my_scene/images/`. Useful options:

```bash
python tools/extract_video_frames.py /path/to/my_scene/video.MOV --frames 150 --max-width 1920
python tools/extract_video_frames.py /path/to/my_scene/video.MOV --frames 200 --select sharp
python tools/extract_video_frames.py /path/to/my_scene/video.MOV --frames 200 --select sharp --candidate-step 2
python tools/extract_video_frames.py /path/to/my_scene/video.MOV --clear
```

`--select sharp` scores candidates with SIFT keypoint count plus Laplacian sharpness, then keeps the best frame from each time bucket so the final set covers the full video instead of only the sharpest few seconds.

Extraction also writes `capture_metadata.json` next to `images/` — a camera-agnostic provenance record (container/codec/resolution/fps, camera identity, and any telemetry stream: GoPro GPMD, Android/Insta360 CAMM, Apple `mebx`, Sony `rtmd`). It prints a suggested COLMAP camera model based on what it detects, e.g. GoPro/action-cam footage → `OPENCV_FISHEYE --mapper global`, standard lenses → `OPENCV`. The telemetry is recorded for provenance; it is not yet fed into the SfM solve.

### 2. COLMAP GUI

```bash
python tools/colmap.py /path/to/my_scene
```

Inside COLMAP:

```text
File > New project
  Database : /data/database.db
  Images   : /data/images

Processing > Feature extraction
  Camera model : OPENCV          <- iPhone 1x / normal phone video
                 OPENCV_FISHEYE  <- iPhone 0.5x / fisheye / action cam
  Single camera for all images: enabled

Processing > Sequential matching
  Loop detection: disabled unless the video revisits the same area

Reconstruction > Start reconstruction
File > Save project
File > Quit
```

COLMAP writes the database and `sparse/0/`. Do not run undistortion for the normal pipeline; 3DGRUT reads `images/` and the COLMAP camera model directly.

Headless COLMAP is still available when you do not want the GUI:

```bash
python tools/colmap.py /path/to/my_scene --headless
python tools/colmap.py /path/to/my_scene --headless --camera-model OPENCV_FISHEYE
```

### 3. Train 3DGRUT

```bash
python tools/train.py /path/to/my_scene
```

Common options:

```bash
python tools/train.py /path/to/my_scene --iterations 30000
python tools/train.py /path/to/my_scene --viser
python tools/train.py /path/to/my_scene --method 3dgut
```

Default training uses 3DGRT, which is the preferred renderer on DGX Spark. 3DGUT is a faster rasterization fallback.

Checkpoints land in `models/<experiment>/ours_<iteration>/ckpt_<iteration>.pt`. The `ours_<iteration>` folder name is upstream 3DGRUT convention, inherited from the original Inria 3D Gaussian Splatting eval layout (each viewpoint is saved as `ours_<iter>/` next to `gt/` for benchmark tables) — it is not per-scene and can be ignored. Model weights (VGG16/LPIPS) are cached under `~/.cache/torch` and mounted into the container, so they download once, not every run.

### 4. Export Raw PLY

```bash
python tools/export_scene.py /path/to/my_scene
```

This exports:

```text
/path/to/my_scene/raw.ply
```

### 5. Clean and Compress in SuperSplat

Open `raw.ply` in SuperSplat, trim/clean the scene, then export the compressed PLY as:

```text
/path/to/my_scene/cleaned.ply
```

Keep spherical harmonics at degree 3 when exporting. Lower SH export settings can lose view-dependent color detail and may not match the USDZ conversion assumptions.

### 6. Convert to USDZ

Requires `pxr` from Isaac Sim or Kit, plus `msgpack`. The helper auto-reruns with Isaac Sim `python.sh` when it can find it.

```bash
python tools/usd_convert.py /path/to/my_scene/cleaned.ply /path/to/my_scene/scene.usdz --align
```

Without `--align` the USDZ ships raw COLMAP world coordinates: an arbitrary origin
near the camera centroid, Y-down axes, and an arbitrary scale — so the scene arrives
in Isaac Sim tilted roughly 90 degrees despite the stage declaring `upAxis = Z`.

`--align` rebuilds the frame from the COLMAP cameras in `sparse/0/` (found automatically
next to the PLY) and writes it into the volume's transform op:

- origin at the first image's camera centre
- `+Z` from the mean camera up, so the ground is level
- `+X` the first camera's viewing direction flattened to horizontal

`--align-mode first-pose` takes origin *and* axes wholly from the first camera instead
(inherits that frame's tilt); `--align-mode translate` only re-origins and leaves the
COLMAP axes alone.

COLMAP scale stays arbitrary — pass `--scale S` to get the scene into metres, since the
stage declares `metersPerUnit = 1.0`.

Import `scene.usdz` in Isaac Sim with `File > Import`.

## Environment Notes

- Docker image: `3dgrut:spark-cuda130-v2` by default (3DGRUT 2.0.0).
- Containers inherit the host timezone, so run directories and logs are stamped
  in local time rather than UTC.
- Requires NVIDIA Docker and `xhost +local:docker` for COLMAP GUI.
- Video frames do not carry useful per-frame EXIF exposure, so training disables EXIF loading by default.
- For large scenes, reduce `--frames`, `--max-width`, or `--iterations` first.

See `pipeline_commands.txt` for the same flow as a compact checklist.
