#!/usr/bin/env python3
"""Launch the interactive 3DGRUT ray-tracing playground.

Opens an OpenGL/OptiX viewer with real-time ray-tracing effects: reflections,
refractions, shadows. Accepts a 3DGRUT checkpoint (.pt), a Gaussian PLY file,
or an INGP checkpoint.

Requires X11 forwarding. The tool runs xhost +local:docker automatically.

Usage:
    python tools/playground.py /path/to/scene
    python tools/playground.py /path/to/scene --checkpoint models/scene/ckpt_last.pt
    python tools/playground.py /path/to/scene --ply output/20240101_120000/scene.ply
    python tools/playground.py /path/to/scene --ingp path/to/model.ingp
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


def resolve_path(workspace: Path, path_arg: str) -> Path:
    p = Path(path_arg).expanduser()
    if not p.is_absolute():
        p = workspace / p
    return p.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "workspace",
        help="Workspace root directory",
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--checkpoint",
        help="3DGRUT .pt checkpoint (workspace-relative or absolute). "
             "Default: most recent checkpoint found under models/.",
    )
    source.add_argument(
        "--ply",
        help="Gaussian PLY file to visualise (workspace-relative or absolute). "
             "Useful for inspecting an exported scene without the original checkpoint.",
    )
    source.add_argument(
        "--ingp",
        help="INGP checkpoint file (workspace-relative or absolute).",
    )

    parser.add_argument(
        "--config", default="apps/colmap_3dgrt.yaml", metavar="CONFIG",
        help="Config name used when loading .ply or .ingp files "
             "(default: apps/colmap_3dgrt.yaml). Ignored for .pt checkpoints "
             "which carry their own config.",
    )
    parser.add_argument(
        "--mesh-assets",
        help="Directory containing .obj/.glb mesh assets to place in the scene. "
             "Default: bundled assets inside the container.",
    )
    parser.add_argument(
        "--envmap-assets",
        help="Directory containing .hdr environment maps for mesh lighting. "
             "Default: bundled assets inside the container.",
    )
    parser.add_argument(
        "--buffer-mode", choices=["device2device", "host2device"], default="device2device",
        help="CUDA-to-OpenGL transfer mode (default: device2device — recommended on DGX Spark).",
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

    # Resolve the scene object to visualise
    if args.ply:
        obj_host = resolve_path(workspace, args.ply)
        is_checkpoint = False
    elif args.ingp:
        obj_host = resolve_path(workspace, args.ingp)
        is_checkpoint = False
    elif args.checkpoint:
        obj_host = resolve_path(workspace, args.checkpoint)
        is_checkpoint = True
    else:
        obj_host = find_latest_checkpoint(workspace / "models")
        is_checkpoint = True

    if not obj_host or not obj_host.exists():
        print(
            f"Error: no checkpoint found in {workspace}/models/\n"
            "Run training first, or pass --checkpoint / --ply / --ingp.",
            file=sys.stderr,
        )
        return 1

    try:
        obj_rel = obj_host.relative_to(workspace)
    except ValueError:
        print(f"Error: scene object path must be inside workspace ({workspace})", file=sys.stderr)
        return 1

    obj_container = f"/data/{obj_rel}"

    # Build optional CLI flags for playground.py
    extra: list[str] = []
    if not is_checkpoint:
        extra += ["--default_gs_config", args.config]
    if args.buffer_mode != "device2device":
        extra += ["--buffer_mode", args.buffer_mode]
    for flag, attr in [("--mesh_assets", args.mesh_assets), ("--envmap_assets", args.envmap_assets)]:
        if attr:
            p = resolve_path(workspace, attr)
            try:
                extra += [flag, f"/data/{p.relative_to(workspace)}"]
            except ValueError:
                print(f"Error: {flag} path must be inside workspace ({workspace})", file=sys.stderr)
                return 1

    extra_str = (" \\\n  " + " \\\n  ".join(extra)) if extra else ""

    # X11 forwarding
    subprocess.run(["xhost", "+local:docker"], capture_output=True)
    display = (
        subprocess.run(["bash", "-c", "echo $DISPLAY"], capture_output=True, text=True)
        .stdout.strip() or ":0"
    )

    print(f"Workspace  : {workspace}  →  /data")
    print(f"Scene      : {obj_container}")
    print(f"Image      : {args.image}")
    print()

    inner = f"""\
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh && conda activate 3dgrut
export UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX
bash /workspace/scripts/install_slangc.sh
cd /workspace
python playground.py \\
  --gs_object {obj_container}{extra_str}
"""

    cmd = [
        "docker", "run", "--rm", "-it", "--gpus", "all",
        "--net=host", "--ipc=host",
        "-e", f"DISPLAY={display}",
        "-e", "QT_X11_NO_MITSHM=1",
        "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw",
        "-v", f"{workspace}:/data",
        "--runtime=nvidia",
        args.image,
        "bash", "-lc", inner,
    ]

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
