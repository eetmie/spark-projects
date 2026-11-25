# scene-reconstruction

Videos -> metric 3DGS -> Isaac Sim USDZ on the DGX Spark (GB10), written as a runbook an
agent can follow unattended. Just use [Spirula Studio](https://github.com/harry7557558/spirula-studio)
(not sponsored haha) and NVIDIA's [`usd-convert-gsplat`](https://pypi.org/project/usd-convert-gsplat/).

## Setup
- Spirula Studio built with the two GB10 fixes in [`patches/`](patches/) (`git apply` in the
  spirula-studio tree): `gb10-queue-idle-sync.patch` (otherwise depth/dense freezes; enabled by
  `SS_NN_QUEUE_IDLE_SYNC=1`) and `ffmpeg-n-version.patch` (ffmpeg builds printing `n4.4`).
- Converter: `uv venv -p 3.12 .venv && uv pip install -p .venv 'usd-convert-gsplat[usd]==0.2.1'`

```bash
export SS_NN_QUEUE_IDLE_SYNC=1 SS_VK_DEVICE=NVIDIA      # every spirula command on GB10
```
Run each step with `< /dev/null > logs/<step>.txt 2>&1`; long steps detached, progress as
timestamped lines in `logs/queue.status`.

## Pipeline
Times are for 4 iPhone 4K clips -> 1066 frames, 1M splats (PSNR ~30).

| # | Step | Command | GB10 |
|---|------|---------|------|
| 1 | Frames per clip | `spirula sam extract videos/$v.mp4 -o images/$v --skip 10 --blur-reject 8 --skipped-dir skipped/$v --report reports/$v.csv -q 95` | min |
| 2 | SfM, all clips | `spirula sfm auto images -o . --quality high --data-type video --sequence <v>... --prefilter-sequential` | 4 min |
| 3 | Dense seed (RoMa v2) | `spirula dense . --preset high --stride 2 --accept-license roma=yes` | 69 min |
| 4 | Normals (Marigold v2) | `spirula geometry . --model marigold-v2` | 5.8 s/img |
| 5 | Train | `spirula train 3dgs --data . --primitive 3dgs --num-iterations 40000 --cap-max 1000000 --sh-degree 3 --output-dir-name <run>` | 53 min |
| 6 | AprilTags -> metres | `spirula marker detect . --decimate 1` then `python3 scale_ply.py . in.ply metric.ply <tag_edge_m>` | 2 min |
| 7 | USDZ | `usd-convert-gsplat -i metric.ply -o scene.usdz --up-axis Z [--half-precision]` | min |

Check before moving on: SfM registers ~100% of frames at <~1.5 px; tag edges agree <2%;
camera-path extent in metres is plausible; open the USDZ with `pxr` (Z-up, splat count, finite).

## Decisions
- **3DGS, not 3DGUT**: Isaac's `ParticleField3DGaussianSplat` is perspective-only; 3DGUT assets felt heavy.
- **Size** only shrinks via splat count, SH degree, `--half-precision` (~half) and cropping.
  SuperSplat/SOG/SPZ codecs don't carry into USDZ and the converter rejects compressed PLY:
  crop in SuperSplat, export a **plain** PLY, then convert. 1M splats SH3: 248 MB float, 130 MB half.
- `--cache-images cpu` gives no speedup on GB10; keep `disk`.

## Gotchas
- `spirula train` never exits (the viewer waits): stop it by PID on the `Training complete` log line.
- `sam extract --adaptive` left 34 s gaps on iPhone clips; use a fixed `--skip`.
- `--blur-reject` keeps up to 2 consecutive rejects (neighbour rule); move those by hand from
  `reports/*.csv` for strict filtering.
- Dense at stride 1 on 4K filled the disk (130-160 GB pair cache + sort). Need >200 GB free or
  `--stride 2`; reruns reuse `dense/.work/cache/pairs-v4`.
- Marigold also fetches `moge2-vitb-normal.onnx`; if HF DNS flakes, curl it into
  `~/.cache/spirula-studio/models/`.
- `marker align` needs 3 tags and scales only by tag distance. `scale_ply.py` uses the tag size
  instead and keeps the SfM frame (Z-up, tag plane ~z=0). Tag size = black-square edge (what
  AprilTag detects), not the whole printed tag.
