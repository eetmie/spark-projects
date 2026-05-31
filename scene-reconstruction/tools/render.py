#!/usr/bin/env python3
"""Render all training/test views from a 3DGRUT checkpoint and compute metrics.

Runs upstream render.py inside the Docker container. Outputs rendered images
and PSNR/SSIM/LPIPS metrics to the output directory.

Usage:
    python tools/render.py /path/to/scene
    python tools/render.py /path/to/scene --checkpoint models/scene/ckpt_last.pt
    python tools/render.py /path/to/scene --out output/my_eval
    python tools/render.py /path/to/scene --no-gt       # skip saving ground-truth images
    python tools/render.py /path/to/scene --no-metrics  # skip PSNR/SSIM/LPIPS computation
    python tools/render.py /path/to/scene --path /other/dataset  # override dataset path
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def find_latest_checkpoint(models_dir: Path) -> Path | None:
    candidates = list(models_dir.rglob("ckpt_last.pt"))
    candidates += list(models_dir.rglob("ckpt_*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_checkpoint(workspace: Path, checkpoint_arg: str) -> Path:
    raw = Path(checkpoint_arg).expanduser()
    ckpt = raw if raw.is_absolute() else workspace / raw
    ckpt = ckpt.resolve()
    if ckpt.is_dir():
        candidates = list(ckpt.glob("ckpt_last.pt")) + list(ckpt.glob("ckpt_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"no ckpt_last.pt or ckpt_*.pt found in {ckpt}")
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return ckpt


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "workspace",
        help="Workspace root directory (must contain a models/ subdirectory with checkpoints)",
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to a .pt checkpoint (workspace-relative or absolute, or a directory). "
             "Default: most recent checkpoint found under models/.",
    )
    parser.add_argument(
        "--path",
        help="Dataset path override (workspace-relative or absolute). "
             "Default: the path stored inside the checkpoint. "
             "Use this when the dataset has moved since training.",
    )
    parser.add_argument(
        "--out",
        help="Output directory for rendered images and metrics "
             "(default: <workspace>/output/<timestamp>/renders)",
    )
    parser.add_argument(
        "--no-gt", action="store_true",
        help="Do not save ground-truth images alongside renders",
    )
    parser.add_argument(
        "--no-metrics", action="store_true",
        help="Skip PSNR/SSIM/LPIPS metric computation",
    )
    parser.add_argument(
        "--image", default="3dgrut:spark-cuda130",
        help="Docker image (default: 3dgrut:spark-cuda130)",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        print(f"Error: workspace not found: {workspace}", file=sys.stderr)
        return 1

    # Resolve checkpoint
    if args.checkpoint:
        try:
            ckpt = resolve_checkpoint(workspace, args.checkpoint)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    else:
        ckpt = find_latest_checkpoint(workspace / "models")

    if not ckpt or not ckpt.exists():
        print(
            f"Error: no checkpoint found in {workspace}/models/\n"
            "Run training first, or pass --checkpoint path/to/ckpt.pt",
            file=sys.stderr,
        )
        return 1

    try:
        ckpt_rel = ckpt.relative_to(workspace)
    except ValueError:
        print(f"Error: checkpoint must be inside workspace ({workspace})", file=sys.stderr)
        return 1

    # Output directory
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_host = (
        Path(args.out).expanduser().resolve()
        if args.out
        else workspace / "output" / stamp / "renders"
    )
    out_host.mkdir(parents=True, exist_ok=True)
    try:
        out_rel = out_host.relative_to(workspace)
    except ValueError:
        print(f"Error: --out must be inside workspace ({workspace})", file=sys.stderr)
        return 1

    ckpt_container = f"/data/{ckpt_rel}"
    out_container = f"/data/{out_rel}"

    # Optional dataset path override
    path_arg = ""
    if args.path:
        data_path = Path(args.path).expanduser()
        if not data_path.is_absolute():
            data_path = workspace / data_path
        data_path = data_path.resolve()
        try:
            path_container = f"/data/{data_path.relative_to(workspace)}"
        except ValueError:
            print(f"Error: --path must be inside workspace ({workspace})", file=sys.stderr)
            return 1
        path_arg = f" \\\n  --path {path_container}"

    flags = ""
    if args.no_gt:
        flags += " \\\n  --no-save-gt"
    if args.no_metrics:
        flags += " \\\n  --no-compute-extra-metrics"

    print(f"Workspace  : {workspace}  →  /data")
    print(f"Checkpoint : {ckpt}")
    print(f"Output     : {out_host}/")
    print()

    inner = f"""\
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh && conda activate 3dgrut
export UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX
bash /workspace/scripts/install_slangc.sh
cd /workspace
python render.py \\
  --checkpoint {ckpt_container} \\
  --out-dir {out_container}{path_arg}{flags}
"""

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--net=host", "--ipc=host",
        "-v", f"{workspace}:/data",
        "--runtime=nvidia",
        args.image,
        "bash", "-lc", inner,
    ]

    if sys.stdout.isatty():
        cmd.insert(3, "-it")

    rc = subprocess.run(cmd).returncode
    if rc == 0:
        print(f"\nRenders : {out_host}/")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
