# Scene Reconstruction Pipeline

Smartphone or wide-angle video → 3DGRUT Gaussian splat → Isaac Sim NuRec USDZ on DGX Spark.

## Renderers

3DGRUT ships two renderer backends. **3DGRT is the default** and the right choice on DGX Spark.

| | 3DGRT (default) | 3DGUT |
|---|---|---|
| Rendering | Full ray tracing (OptiX) | Rasterization (unscented transform) |
| Quality | Best — reflections, shadows, secondary rays | Good — primary rays only |
| Speed | Slower (needs RT hardware) | Faster |
| Distorted cameras | Native (OPENCV_FISHEYE etc.) | Native (OPENCV_FISHEYE etc.) |
| DGX Spark (GB10) | Recommended — dedicated RT cores | Fallback if speed matters |

Both renderers handle distorted cameras natively — no image undistortion required.

## Workspace Layout

```text
my_scene/
├── images/          ← extracted images used by COLMAP and 3DGRUT
├── database.db      ← COLMAP database
├── sparse/0/        ← COLMAP sparse reconstruction + camera parameters
├── vocab_tree.bin   ← optional loop-detection vocabulary tree
├── models/          ← 3DGRUT checkpoints
└── output/          ← timestamped PLY and USDZ exports
```

## Quick Start

### 1. Extract Images

```bash
python tools/extract_video_frames.py /path/to/my_scene/video.MOV
# Creates images/, models/, output/ automatically.
# Options: --frames 150  --max-width 1920  (default: native video resolution)
```

### 2. COLMAP reconstruction

Open the GUI:

```bash
python tools/colmap.py /path/to/my_scene
```

Inside COLMAP:

```
File > New project
  Database : /data/database.db
  Images   : /data/images

Processing > Feature extraction
  Camera model : OPENCV          ← phone / DSLR / standard wide-angle
                 OPENCV_FISHEYE  ← fisheye / action cam (GoPro etc.)
  Single camera for all images: enabled

Processing > Sequential matching
  Loop detection: disabled   ← requires a vocab_tree.bin file; leave off for video

Reconstruction > Start reconstruction

File > Save project > Quit
```

Or run headless (GUI + reconstruction in one command):

```bash
python tools/colmap.py /path/to/my_scene --headless
python tools/colmap.py /path/to/my_scene --headless --camera-model OPENCV_FISHEYE
```

### 3. Optional Undistortion

3DGRUT reads `images/` and the COLMAP camera parameters from `sparse/0/` directly.
No linking step is needed. Optionally, undistort to pinhole if you want a simplified
camera model, at the cost of losing some field of view:

```bash
python tools/colmap.py /path/to/my_scene --undistort-only
```

### 4. Train 3DGRUT

```bash
python tools/train.py /path/to/my_scene
```

Common options:

```bash
python tools/train.py /path/to/my_scene --iterations 30000          # quick test
python tools/train.py /path/to/my_scene --mcmc                      # MCMC densification
python tools/train.py /path/to/my_scene --method 3dgut              # rasterization backend
python tools/train.py /path/to/my_scene --viser                     # live viewer → localhost:8080
python tools/train.py /path/to/my_scene --experiment living_room    # named experiment
```

### 5. Export PLY

```bash
python tools/export_scene.py /path/to/my_scene
python tools/export_scene.py /path/to/my_scene --npz   # also export raw Gaussian arrays
```

### 6. Convert to USDZ (Isaac Sim / Omniverse)

Requires `pxr` from Isaac Sim or Kit, plus `pip install msgpack`:

```bash
python tools/usd_convert.py /path/to/my_scene/output/TIMESTAMP/scene.ply
python tools/usd_convert.py scene.ply scene.usdz --extract-sidecars
```

Import `scene.usdz` in Isaac Sim with `File > Import`.

## Environment Notes

- Docker image: `3dgrut:spark-cuda130` (CUDA 13.0, COLMAP 4.0.4 GPU SIFT, SM 12.1)
- Requires NVIDIA Docker (`--runtime=nvidia`) and `xhost +local:docker` for GUI.
- `dataset.load_exif=false` is set automatically — video-extracted frames don't have per-frame EXIF exposure.
- Isaac Sim / Kit provides host `pxr`; `usd-core` has no aarch64 PyPI wheel.
- For large scenes on 128 GB unified memory: reduce `--frames`, `--max-width`, or `--iterations` if you run out of memory.

## Container

All wrappers default to `3dgrut:spark-cuda130`. Override with `--image`:

```bash
python tools/train.py /path/to/my_scene --image your-image-tag
```

See `pipeline_commands.txt` for the same flow in checklist format.
