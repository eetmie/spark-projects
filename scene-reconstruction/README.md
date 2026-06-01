# Scene Reconstruction Pipeline

Smartphone video -> COLMAP -> 3DGRUT Gaussian splat -> SuperSplat cleanup/compression -> Isaac Sim NuRec USDZ on DGX Spark.

## Workspace Layout

The normal pipeline only creates the folders and files that are used by the next step:

```text
my_scene/
├── video.MOV        <- source video copied from iPhone
├── images/          <- ffmpeg-extracted frames for COLMAP and 3DGRUT
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

Put the iPhone video in a scene folder, then extract frames:

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
python tools/usd_convert.py /path/to/my_scene/cleaned.ply /path/to/my_scene/scene.usdz
```

Import `scene.usdz` in Isaac Sim with `File > Import`.

## Environment Notes

- Docker image: `3dgrut:spark-cuda130` by default.
- Requires NVIDIA Docker and `xhost +local:docker` for COLMAP GUI.
- Video frames do not carry useful per-frame EXIF exposure, so training disables EXIF loading by default.
- For large scenes, reduce `--frames`, `--max-width`, or `--iterations` first.

See `pipeline_commands.txt` for the same flow as a compact checklist.
