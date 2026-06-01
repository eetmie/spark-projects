#!/usr/bin/env python3
"""Train a 3DGRUT scene from a COLMAP sparse reconstruction or standard dataset.

3DGRUT is NVIDIA's combined Gaussian framework. Both renderers support distorted
cameras (OPENCV_FISHEYE) natively — no image undistortion required for either.

    3dgrt  (default)  Full ray tracing via OptiX. Supports distorted cameras, rolling
                      shutter, reflections, shadows. Best quality; needs RT hardware.
                      Use this on DGX Spark (GB10 has dedicated ray-tracing cores).

    3dgut             Rasterization via unscented transform. Also supports distorted
                      cameras. Faster than 3dgrt but no secondary ray effects.

Dataset types (--dataset-type):
    colmap         (default)  COLMAP sparse reconstruction — workspace contains images/ + sparse/
    nerf_synthetic            NeRF Synthetic (Blender) dataset — workspace IS the scene directory
    scannetpp                 ScanNet++ scene directory — workspace IS the scene directory

Usage:
    python tools/train.py /path/to/scene
    python tools/train.py /path/to/scene --method 3dgut
    python tools/train.py /path/to/scene --mcmc
    python tools/train.py /path/to/scene --iterations 30000 --experiment my_scene
    python tools/train.py /path/to/scene --viser              # live viewer at http://localhost:8080
    python tools/train.py /path/to/scene --downsample 2       # half-res (recommended for >4MP photos)
    python tools/train.py /path/to/scene --load-exif          # enable EXIF exposure for real photos
    python tools/train.py /path/to/scene --background white   # better for outdoor/sky scenes
    python tools/train.py /path/to/scene --resume models/scene/ckpt_last.pt
    python tools/train.py /path/to/nerf_synthetic/lego --dataset-type nerf_synthetic
    python tools/train.py /path/to/scannetpp/scene0000 --dataset-type scannetpp
    python tools/train.py /path/to/scene --wandb               # log to W&B project "3dgrt"
    python tools/train.py /path/to/scene --wandb my_project    # log to named W&B project
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# Config names in apps/ keyed by (dataset_type, method, mcmc).
# nerf_synthetic and scannetpp have no MCMC config variants upstream.
_CONFIGS: dict[tuple[str, str, bool], str] = {
    ("colmap",         "3dgrt", False): "apps/colmap_3dgrt",
    ("colmap",         "3dgrt", True):  "apps/colmap_3dgrt_mcmc",
    ("colmap",         "3dgut", False): "apps/colmap_3dgut",
    ("colmap",         "3dgut", True):  "apps/colmap_3dgut_mcmc",
    ("nerf_synthetic", "3dgrt", False): "apps/nerf_synthetic_3dgrt",
    ("nerf_synthetic", "3dgut", False): "apps/nerf_synthetic_3dgut",
    ("scannetpp",      "3dgrt", False): "apps/scannetpp_3dgrt",
    ("scannetpp",      "3dgut", False): "apps/scannetpp_3dgut",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "workspace",
        help="Workspace root directory. For colmap: contains images/ and sparse/. "
             "For nerf_synthetic/scannetpp: this IS the scene directory.",
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset-type", choices=["colmap", "nerf_synthetic", "scannetpp"], default="colmap",
        help="Dataset format (default: colmap). "
             "nerf_synthetic = Blender NeRF Synthetic scene directory. "
             "scannetpp = ScanNet++ scene directory.",
    )
    parser.add_argument(
        "--colmap-dir", default=".",
        help="COLMAP dataset directory relative to workspace (default: workspace root). "
             "colmap dataset type only.",
    )

    # ── Renderer ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--method", choices=["3dgrt", "3dgut"], default="3dgrt",
        help="Renderer backend: 3dgrt = full ray tracing via OptiX (default); "
             "3dgut = rasterization via unscented transform (faster, no secondary rays)",
    )
    parser.add_argument(
        "--mcmc", action="store_true",
        help="Use MCMC densification strategy instead of standard GS. "
             "Supported for colmap dataset type only.",
    )

    # ── Training schedule ─────────────────────────────────────────────────────
    parser.add_argument(
        "--iterations", type=int, default=60000,
        help="Training iterations (default: 60000; use 30000 for a quick test)",
    )
    parser.add_argument(
        "--experiment", default="scene",
        help="Experiment name — models are saved to models/<experiment>/ (default: scene)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for models (default: <workspace>/models)",
    )
    parser.add_argument(
        "--resume", default="",
        help="Resume training from a checkpoint (.pt file, path relative to workspace or absolute)",
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--downsample", type=int, default=1, metavar="N",
        help="Downsample images by factor N before training "
             "(e.g. 2 = half-res; recommended for photos above ~4MP to save GPU memory; "
             "matches upstream dataset.downsample_factor)",
    )
    parser.add_argument(
        "--load-exif", action="store_true",
        help="Load EXIF exposure metadata from JPEG images. Enable for real photos. "
             "Disabled by default because video-extracted frames lack per-frame EXIF exposure.",
    )
    parser.add_argument(
        "--test-split-interval", type=int, default=None, metavar="N",
        help="Hold out every N-th image for evaluation (default: use config value, typically 8). "
             "Set to 0 to train on all images with no held-out test set.",
    )

    # ── Scene / model ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--background", choices=["black", "white", "random"], default="black",
        help="Background colour for unbounded scenes (white helps with outdoor/sky; default: black)",
    )

    # ── Viewer / monitoring ───────────────────────────────────────────────────
    parser.add_argument(
        "--viser", action="store_true",
        help="Enable live Viser viewer at http://localhost:8080 during training",
    )
    parser.add_argument(
        "--val-frequency", type=int, default=None, metavar="N",
        help="Validate every N iterations and log PSNR. "
             "Default: colmap configs disable validation (999999). "
             "Use e.g. 5000 for periodic quality measurements.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="DataLoader worker count (default: use config value, typically 24)",
    )

    parser.add_argument(
        "--wandb", nargs="?", const="3dgrt", default=None, metavar="PROJECT",
        help="Log metrics to Weights & Biases. Optional project name "
             "(default project name: 3dgrt). Requires wandb installed and logged in.",
    )

    # ── Container ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--image", default="3dgrut:spark-cuda130",
        help="Docker image (default: 3dgrut:spark-cuda130)",
    )
    args = parser.parse_args()

    # ── Validate workspace ────────────────────────────────────────────────────
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        print(f"Error: workspace not found: {workspace}", file=sys.stderr)
        return 1
    if args.downsample <= 0:
        print("Error: --downsample must be a positive integer", file=sys.stderr)
        return 1

    # MCMC is only available for colmap configs
    if args.mcmc and args.dataset_type != "colmap":
        print(
            f"Warning: --mcmc has no config variant for --dataset-type {args.dataset_type!r}; "
            "ignoring --mcmc",
            file=sys.stderr,
        )
        args.mcmc = False

    # Dataset path inside the container
    if args.dataset_type == "colmap":
        colmap_path = (workspace / args.colmap_dir).resolve()
        if not (colmap_path / "images").is_dir():
            print(
                f"Error: {colmap_path}/images/ not found.\n"
                "Run tools/extract_video_frames.py first, or point --colmap-dir at a COLMAP dataset.",
                file=sys.stderr,
            )
            return 1
        sparse0 = colmap_path / "sparse" / "0"
        nested_sparse0 = sparse0 / "0"
        if not sparse0.is_dir():
            print(
                f"Error: {sparse0}/ not found.\n"
                "Run COLMAP reconstruction first.",
                file=sys.stderr,
            )
            return 1
        if not (sparse0 / "images.bin").exists() and not (nested_sparse0 / "images.bin").exists():
            print(
                f"Error: no COLMAP model found in {sparse0}/ or {nested_sparse0}/.\n"
                "Run COLMAP reconstruction first.",
                file=sys.stderr,
            )
            return 1
        dataset_container = f"/data/{colmap_path.relative_to(workspace)}"
    else:
        # For nerf_synthetic / scannetpp the workspace IS the scene directory.
        dataset_container = "/data"

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else workspace / "models"
    try:
        out_container = f"/data/{out_dir.relative_to(workspace)}"
    except ValueError:
        print(f"Error: --out-dir must be inside the workspace ({workspace})", file=sys.stderr)
        return 1

    # --resume: translate to container path
    resume_container = ""
    if args.resume:
        resume_path = Path(args.resume).expanduser()
        if not resume_path.is_absolute():
            resume_path = workspace / resume_path
        resume_path = resume_path.resolve()
        try:
            resume_container = f"/data/{resume_path.relative_to(workspace)}"
        except ValueError:
            print(
                f"Error: --resume path must be inside the workspace ({workspace})",
                file=sys.stderr,
            )
            return 1

    # ── Build Hydra overrides ─────────────────────────────────────────────────
    config = _CONFIGS[(args.dataset_type, args.method, args.mcmc)]
    iters = args.iterations

    overrides: list[str] = [
        f"path={dataset_container}",
        f"out_dir={out_container}",
        f"experiment_name={args.experiment}",
        f"n_iterations={iters}",
        f"scheduler.positions.max_steps={iters}",
        f'checkpoint.iterations="[{max(iters // 8, 1000)},{iters // 2},{iters}]"',
    ]

    if args.viser:
        overrides.append("with_viser_gui=True")

    if args.val_frequency is not None:
        overrides.append(f"val_frequency={args.val_frequency}")

    if args.workers is not None:
        overrides.append(f"num_workers={args.workers}")

    if args.resume:
        overrides.append(f"resume={resume_container}")

    if args.wandb is not None:
        overrides += [f"use_wandb=true", f"wandb_project={args.wandb}"]

    # Dataset overrides
    if not args.load_exif:
        overrides.append("dataset.load_exif=false")
    if args.downsample != 1:
        overrides.append(f"dataset.downsample_factor={args.downsample}")
    if args.test_split_interval is not None:
        if args.test_split_interval == 0:
            # test_split_interval=0 is not a valid config value upstream; use a large number instead
            overrides.append("dataset.test_split_interval=999999")
        else:
            overrides.append(f"dataset.test_split_interval={args.test_split_interval}")

    # Model overrides
    if args.background != "black":
        overrides.append(f"model.background.color={args.background}")

    # Standard GS: extend densification window to 25k
    # (MCMC uses its own internal schedule — don't override)
    if not args.mcmc:
        end = min(25000, iters)
        overrides += [
            f"strategy.densify.end_iteration={end}",
            f"strategy.prune.end_iteration={end}",
            f"strategy.reset_density.end_iteration={end}",
        ]

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"Workspace  : {workspace}  →  /data")
    print(f"Dataset    : {dataset_container}  ({args.dataset_type})")
    print(f"Config     : {config}.yaml")
    print(f"Method     : {args.method}{' + MCMC' if args.mcmc else ''}")
    print(f"Iterations : {iters}")
    print(f"Output     : {out_container}/{args.experiment}/")
    if resume_container:
        print(f"Resume     : {resume_container}")
    if args.wandb is not None:
        print(f"W&B        : {args.wandb}")
    print()

    override_str = " \\\n  ".join(overrides)
    prepare_downsample = ""
    if args.dataset_type == "colmap" and args.downsample != 1:
        prepare_downsample = f"""
python - <<'PY'
from pathlib import Path
from PIL import Image

factor = {args.downsample}
dataset = Path({dataset_container!r})
src_dir = dataset / "images"
dst_dir = dataset / f"images_{{factor}}"
suffixes = {{".jpg", ".jpeg", ".png", ".tif", ".tiff"}}

if not src_dir.is_dir():
    raise SystemExit(f"Missing source image directory: {{src_dir}}")

sources = sorted(path for path in src_dir.rglob("*") if path.suffix.lower() in suffixes)
if not sources:
    raise SystemExit(f"No images found in {{src_dir}}")

refreshed = 0
for src in sources:
    rel = src.relative_to(src_dir)
    dst = dst_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        continue

    with Image.open(src) as image:
        width = max(1, image.width // factor)
        height = max(1, image.height // factor)
        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        if resized.mode == "RGBA" and dst.suffix.lower() in {{".jpg", ".jpeg"}}:
            resized = resized.convert("RGB")
        resized.save(dst, quality=95)

    refreshed += 1

print(f"Downsample : {{src_dir}} -> {{dst_dir}} (factor {{factor}}, {{len(sources)}} images, {{refreshed}} refreshed)")
PY
"""
    inner = f"""\
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh && conda activate 3dgrut
export UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX
bash /workspace/scripts/install_slangc.sh
cd /workspace
{prepare_downsample}python train.py --config-name {config}.yaml \\
  {override_str}
"""

    torch_cache = Path.home() / ".cache" / "torch"
    torch_cache.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--net=host", "--ipc=host",
        "-v", f"{workspace}:/data",
        "-v", f"{torch_cache}:/root/.cache/torch",
        "--runtime=nvidia",
        args.image,
        "bash", "-lc", inner,
    ]

    if sys.stdout.isatty():
        cmd.insert(3, "-it")

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
