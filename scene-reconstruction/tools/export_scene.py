#!/usr/bin/env python3
"""Export a 3DGRUT checkpoint to a Gaussian Splat PLY file.

Finds the most recent checkpoint under models/ by default. Exports raw.ply
to the workspace root for cleanup/compression in SuperSplat.

Usage:
    python tools/export_scene.py /path/to/scene
    python tools/export_scene.py /path/to/scene --iteration 15000
    python tools/export_scene.py /path/to/scene --checkpoint models/scene/data-.../ours_15000
    python tools/export_scene.py /path/to/scene --checkpoint models/scene/data-.../ours_15000/ckpt_15000.pt
    python tools/export_scene.py /path/to/scene --out /path/to/scene/exports
    python tools/export_scene.py /path/to/scene --name living_room_raw.ply
    python tools/export_scene.py /path/to/scene --npz
    python tools/export_scene.py /path/to/scene --image 3dgrut:spark-cuda130
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_latest_checkpoint(models_dir: Path) -> Path | None:
    candidates = list(models_dir.rglob("ckpt_last.pt"))
    candidates += list(models_dir.rglob("ckpt_*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_iteration_checkpoint(models_dir: Path, iteration: int) -> Path | None:
    candidates = list(models_dir.rglob(f"ckpt_{iteration}.pt"))
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
        help="Workspace root directory",
    )
    parser.add_argument(
        "--checkpoint",
        help="Workspace-relative or absolute path to a .pt checkpoint, or a directory containing one.",
    )
    parser.add_argument(
        "--iteration", type=int,
        help="Export ckpt_<iteration>.pt from models/ (for example: --iteration 15000).",
    )
    parser.add_argument(
        "--out",
        help="Output directory (default: workspace root)",
    )
    parser.add_argument(
        "--name", default="raw.ply",
        help="Output PLY filename (default: raw.ply)",
    )
    parser.add_argument(
        "--image", default="3dgrut:spark-cuda130",
        help="Docker image (default: 3dgrut:spark-cuda130)",
    )
    parser.add_argument(
        "--npz", action="store_true",
        help="Also export raw Gaussian arrays to scene_gaussians.npz",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        print(f"Error: workspace not found: {workspace}", file=sys.stderr)
        return 1

    if args.checkpoint and args.iteration is not None:
        print("Error: use either --checkpoint or --iteration, not both", file=sys.stderr)
        return 1

    if args.checkpoint:
        try:
            ckpt = resolve_checkpoint(workspace, args.checkpoint)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    elif args.iteration is not None:
        ckpt = find_iteration_checkpoint(workspace / "models", args.iteration)
    else:
        ckpt = find_latest_checkpoint(workspace / "models")

    if not ckpt or not ckpt.exists():
        print(
            f"Error: no checkpoint found in {workspace}/models/\n"
            "Run training first, or pass --iteration 15000 or --checkpoint path/to/checkpoint.pt",
            file=sys.stderr,
        )
        return 1
    if ckpt.is_dir():
        print(f"Error: checkpoint resolved to a directory, not a .pt file: {ckpt}", file=sys.stderr)
        return 1
    if ckpt.suffix != ".pt":
        print(f"Error: checkpoint must be a .pt file: {ckpt}", file=sys.stderr)
        return 1

    try:
        ckpt_rel = ckpt.relative_to(workspace)
    except ValueError:
        print(f"Error: checkpoint must be inside workspace ({workspace})", file=sys.stderr)
        return 1

    if Path(args.name).name != args.name:
        print("Error: --name must be a filename, not a path", file=sys.stderr)
        return 1

    out_host = Path(args.out).expanduser().resolve() if args.out else workspace
    out_host.mkdir(parents=True, exist_ok=True)

    try:
        out_rel = out_host.relative_to(workspace)
    except ValueError:
        print(f"Error: --out must be inside workspace ({workspace})", file=sys.stderr)
        return 1

    ckpt_container = f"/data/{ckpt_rel}"
    out_container = f"/data/{out_rel}"
    ply_name = args.name

    print(f"Checkpoint : {ckpt}")
    print(f"Output     : {out_host / ply_name}")
    if args.npz:
        print(f"NPZ output : {out_host}/scene_gaussians.npz")
    print()

    export_npz = "1" if args.npz else "0"
    inner = f"""\
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh && conda activate 3dgrut
export UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX
bash /workspace/scripts/install_slangc.sh
mkdir -p {out_container}
cd /workspace
python - <<'PY'
import os
from pathlib import Path
import numpy as np
import torch
from threedgrut.export import PLYExporter
from threedgrut.export.accessor import GaussianExportAccessor
from threedgrut.model.model import MixtureOfGaussians

ckpt_path = "{ckpt_container}"
out_path  = Path("{out_container}")
export_npz = {export_npz}
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
conf  = checkpoint["config"]
model = MixtureOfGaussians(conf, scene_extent=checkpoint.get("scene_extent"))
model.init_from_checkpoint(checkpoint, setup_optimizer=False)
model.eval()
ply_name = "{ply_name}"
PLYExporter().export(model, out_path / ply_name, conf=conf)
print(f"Exported: {{out_path}}/{{ply_name}}")
if export_npz:
    accessor = GaussianExportAccessor(model, conf)
    attrs = accessor.get_attributes(preactivation=True)
    np.savez(
        out_path / "scene_gaussians.npz",
        positions=attrs.positions,
        rotations=attrs.rotations,
        scales=attrs.scales,
        densities=attrs.densities,
        albedo=attrs.albedo,
        specular=attrs.specular,
        sh_degree=np.array(accessor.get_max_sh_degree(), dtype=np.int64),
        num_gaussians=np.array(attrs.num_gaussians, dtype=np.int64),
        preactivation=np.array(True, dtype=np.bool_),
    )
    print(f"Exported: {{out_path}}/scene_gaussians.npz")
PY
"""

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{workspace}:/data",
        "--runtime=nvidia",
        args.image,
        "bash", "-lc", inner,
    ]

    result = subprocess.run(cmd)
    if result.returncode == 0:
        ply = out_host / ply_name
        print(f"\nPLY: {ply}")
        if args.npz:
            print(f"NPZ: {out_host / 'scene_gaussians.npz'}")
        cleaned = out_host / "cleaned.ply"
        print("Next: clean/compress raw.ply in SuperSplat, export cleaned.ply with SH degree 3")
        print(f"Then: python tools/usd_convert.py {cleaned} {out_host / 'scene.usdz'}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
