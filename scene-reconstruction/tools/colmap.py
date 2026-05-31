#!/usr/bin/env python3
"""Run COLMAP inside the 3dgrut Docker container.

Default (GUI mode): opens the COLMAP GUI with X11 forwarding.
The workspace is mounted as /data — use /data/... paths inside the GUI.

  --headless: runs the full pipeline automatically without a GUI.

--- STANDARD CAMERA (phone, DSLR) ---
    python tools/colmap.py /scene --headless
    python tools/train.py  /scene

--- WIDE-ANGLE / FISHEYE (iPhone 0.5x, GoPro, action cam) ---
COLMAP 4.0 ships GLOMAP (--mapper global), which is more robust than the
incremental mapper when SIFT matching degrades at the image edges due to
wide-angle distortion. Try in order until reconstruction succeeds:

  1. Global mapper + fisheye model (best starting point for iPhone 0.5x):
    python tools/colmap.py /scene --headless --camera-model OPENCV_FISHEYE --mapper global

  2. Global mapper + SIMPLE_RADIAL_FISHEYE (fewer parameters, more stable):
    python tools/colmap.py /scene --headless --camera-model SIMPLE_RADIAL_FISHEYE --mapper global

  3. Add exhaustive matching if sequential matching misses many frame pairs:
    python tools/colmap.py /scene --headless --camera-model OPENCV_FISHEYE --mapper global --matcher exhaustive

  The --view-graph-calibrator flag runs an extra COLMAP 4.0 intrinsics
  calibration pass before the global mapper — recommended for fisheye:
    python tools/colmap.py /scene --headless --camera-model OPENCV_FISHEYE --mapper global --view-graph-calibrator

--- LOOP CLOSURE (revisit same locations) ---
    python tools/colmap.py /scene --headless --loop-detection

For GUI workflow, fetch the pretrained tree first:
    python tools/colmap.py /scene --fetch-vocab-tree
    → in GUI, set Sequential matching → Loop detection ON → /data/vocab_tree.bin

After GUI reconstruction, train directly from the workspace. Optional pinhole output:
    python tools/colmap.py /scene --undistort-only

Usage:
    python tools/colmap.py /path/to/scene
    python tools/colmap.py /path/to/scene --headless
    python tools/colmap.py /path/to/scene --headless --camera-model OPENCV_FISHEYE
    python tools/colmap.py /path/to/scene --headless --camera-model OPENCV_FISHEYE --mapper global
    python tools/colmap.py /path/to/scene --headless --loop-detection
    python tools/colmap.py /path/to/scene --headless --matcher exhaustive

COLMAP GUI sequential matching settings:
    Loop detection : ON  (if you revisit the same locations)
      Vocab tree   : /data/vocab_tree.bin  ← run --fetch-vocab-tree first
    Loop detection : OFF (simple linear walkthrough)
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

FISHEYE_MODELS = {"OPENCV_FISHEYE", "RADIAL_FISHEYE", "SIMPLE_RADIAL_FISHEYE"}
VOCAB_TREE_NAME = "vocab_tree.bin"
VOCAB_TREE_URL = (
    "https://github.com/colmap/colmap/releases/download/3.11.1/"
    "vocab_tree_faiss_flickr100K_words256K.bin"
)
VOCAB_TREE_SHA256 = "96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "workspace",
        help="Workspace root directory (must be specified explicitly)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run full pipeline automatically (no GUI).",
    )
    parser.add_argument(
        "--undistort-only", action="store_true",
        help="Only run image_undistorter on an existing sparse/0/ reconstruction.",
    )
    parser.add_argument(
        "--loop-detection", action="store_true",
        help="Enable vocabulary tree loop detection during sequential matching. "
             "Use when your video revisits the same locations. "
             "Downloads a pretrained vocab_tree.bin if not present.",
    )
    parser.add_argument(
        "--fetch-vocab-tree", action="store_true",
        help="Download COLMAP's pretrained FAISS vocab_tree.bin and exit. "
             "Outputs <workspace>/vocab_tree.bin.",
    )
    parser.add_argument(
        "--build-vocab-tree", action="store_true",
        help="Advanced: build vocab_tree.bin from features already in this COLMAP database. "
             "Usually slower and unnecessary; prefer --fetch-vocab-tree.",
    )
    parser.add_argument(
        "--vocab-tree", default=None,
        help="Path to vocab_tree.bin (default: <workspace>/vocab_tree.bin).",
    )
    parser.add_argument(
        "--camera-model", default="OPENCV",
        help="COLMAP camera model (default: OPENCV). "
             "OPENCV: phone / standard wide-angle (Brown-Conrady distortion). "
             "OPENCV_FISHEYE: fisheye / action cam (Kannala-Brandt, iPhone 0.5x). "
             "SIMPLE_RADIAL_FISHEYE: fisheye with fewer params — more stable when OPENCV_FISHEYE diverges. "
             "SIMPLE_RADIAL: minimal radial model. PINHOLE: no distortion.",
    )
    parser.add_argument(
        "--matcher", default="sequential",
        choices=["sequential", "exhaustive", "vocab_tree"],
        help="Feature matching strategy (default: sequential — correct for video). "
             "exhaustive: unordered photo sets (much slower, O(n²)).",
    )
    parser.add_argument(
        "--mapper", default="incremental",
        choices=["incremental", "global"],
        help="SfM mapper (default: incremental). "
             "global: GLOMAP global mapper (COLMAP 4.0+) — estimates all poses simultaneously "
             "instead of registering cameras one-by-one. More robust for wide-angle footage "
             "where incremental registration fails due to edge distortion. "
             "Recommended for iPhone 0.5x / fisheye cameras.",
    )
    parser.add_argument(
        "--view-graph-calibrator", action="store_true",
        help="Run COLMAP 4.0 view_graph_calibrator before the global mapper to refine "
             "camera intrinsics from two-view geometries. Recommended when using "
             "--mapper global with fisheye or wide-angle cameras.",
    )
    parser.add_argument(
        "--mapper-only", action="store_true",
        help="Skip feature extraction and matching — run only the mapper on an existing "
             "database. Use this after completing feature extraction + matching in the "
             "COLMAP GUI when you want to run the global mapper instead of the GUI's "
             "built-in incremental reconstruction:\n"
             "  1. GUI: File → New project, feature extraction, matching, save project\n"
             "  2. python tools/colmap.py /scene --mapper-only --mapper global",
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

    if args.build_vocab_tree:
        return _build_vocab_tree(workspace, args)
    if args.fetch_vocab_tree:
        return 0 if _fetch_vocab_tree(workspace, args) else 1
    if args.mapper_only:
        return _run_mapper_only(workspace, args)
    if args.headless or args.undistort_only:
        return _run_headless(workspace, args)
    return _run_gui(workspace, args)


def _build_vocab_tree(workspace: Path, args) -> int:
    dest = Path(args.vocab_tree) if args.vocab_tree else workspace / VOCAB_TREE_NAME
    db   = workspace / "database.db"
    if not db.exists():
        print(f"Error: {db} not found. Run feature extraction first.", file=sys.stderr)
        return 1
    if dest.exists():
        print(f"Already exists: {dest} — delete it first to rebuild.")
        return 0

    db_container   = "/data/database.db"
    dest_container = f"/data/{dest.relative_to(workspace)}"

    print(f"Building custom vocab tree from extracted features → {dest}")
    print(f"(This may take a few minutes for large frame sets)")

    inner = f"""\
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh && conda activate 3dgrut
colmap vocab_tree_builder \
  --database_path {db_container} \
  --vocab_tree_path {dest_container} \
  --num_visual_words 65536
"""
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{workspace}:/data",
        "--runtime=nvidia",
        args.image,
        "bash", "-lc", inner,
    ]
    rc = subprocess.run(cmd).returncode
    if rc == 0:
        print(f"\nVocab tree ready: {dest}")
        print(f"In COLMAP GUI → Sequential matching → Loop detection: ON")
        print(f"  Vocab tree path: /data/{dest.relative_to(workspace)}")
    return rc


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_vocab_tree(workspace: Path, args) -> Path | None:
    dest = Path(args.vocab_tree).expanduser().resolve() if args.vocab_tree else workspace / VOCAB_TREE_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        actual = _hash_file(dest)
        if actual == VOCAB_TREE_SHA256:
            print(f"Vocab tree already exists: {dest}")
            return dest
        print(f"Existing vocab tree has unexpected SHA256: {dest}", file=sys.stderr)
        print(f"  expected: {VOCAB_TREE_SHA256}", file=sys.stderr)
        print(f"  actual  : {actual}", file=sys.stderr)
        print("Delete or move it, then rerun --fetch-vocab-tree.", file=sys.stderr)
        return None

    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    print("Fetching pretrained COLMAP vocab tree:")
    print(f"  URL : {VOCAB_TREE_URL}")
    print(f"  To  : {dest}")
    try:
        with urllib.request.urlopen(VOCAB_TREE_URL) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        print(f"Error: failed to download vocab tree: {exc}", file=sys.stderr)
        return None

    actual = _hash_file(tmp)
    if actual != VOCAB_TREE_SHA256:
        tmp.unlink()
        print("Error: downloaded vocab tree failed SHA256 check.", file=sys.stderr)
        print(f"  expected: {VOCAB_TREE_SHA256}", file=sys.stderr)
        print(f"  actual  : {actual}", file=sys.stderr)
        return None

    tmp.replace(dest)
    print(f"Vocab tree ready: {dest}")
    print("In COLMAP GUI → Sequential matching → Loop detection: ON")
    print(f"  Vocab tree path: /data/{dest.relative_to(workspace)}")
    return dest


def _ensure_vocab_tree(workspace: Path, args) -> Path | None:
    dest = Path(args.vocab_tree).expanduser().resolve() if args.vocab_tree else workspace / VOCAB_TREE_NAME
    if not dest.exists():
        print("Vocab tree not found — fetching pretrained COLMAP tree...")
        return _fetch_vocab_tree(workspace, args)
    return dest


def _run_gui(workspace: Path, args) -> int:
    subprocess.run(["xhost", "+local:docker"], capture_output=True)
    display = subprocess.run(
        ["bash", "-c", "echo $DISPLAY"], capture_output=True, text=True
    ).stdout.strip() or ":0"

    if args.loop_detection:
        vtree = _ensure_vocab_tree(workspace, args)
        if vtree is None:
            return 1
        vtree_rel = vtree.relative_to(workspace)
        print(f"Vocab tree ready at /data/{vtree_rel}")
        print(f"In COLMAP GUI → Sequential matching → Loop detection: ON, path: /data/{vtree_rel}")
        print()

    inner = """\
source /opt/conda/etc/profile.d/conda.sh && conda activate 3dgrut
PROJECT=/data/sparse/project.ini
if [ -f "$PROJECT" ]; then
  colmap gui --project_path "$PROJECT"
else
  colmap gui
fi"""

    print(f"Workspace : {workspace}  →  /data inside container")
    print(f"Image     : {args.image}")
    print()

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


def _run_mapper_only(workspace: Path, args) -> int:
    """Run just the mapper on an existing COLMAP database (features + matches already done)."""
    db = workspace / "database.db"
    if not db.exists():
        print(
            f"Error: {db} not found.\n"
            "Run feature extraction and matching first (use the COLMAP GUI or --headless "
            "without --mapper-only), then re-run with --mapper-only.",
            file=sys.stderr,
        )
        return 1

    db_container     = "/data/database.db"
    images_container = "/data/images"
    sparse_container = "/data/sparse"

    fisheye = args.camera_model in FISHEYE_MODELS

    view_graph_step = ""
    if args.view_graph_calibrator:
        view_graph_step = (
            f"# View-graph calibrator — refine intrinsics before global mapper\n"
            f"colmap view_graph_calibrator \\\n"
            f"  --database_path {db_container}\n\n"
        )

    if args.mapper == "global":
        mapper_cmd = (
            f"{view_graph_step}"
            f"colmap global_mapper \\\n"
            f"  --database_path {db_container} \\\n"
            f"  --image_path {images_container} \\\n"
            f"  --output_path {sparse_container}"
        )
    else:
        mapper_cmd = (
            f"colmap mapper \\\n"
            f"  --database_path {db_container} \\\n"
            f"  --image_path {images_container} \\\n"
            f"  --output_path {sparse_container}"
        )

    undistort = "" if fisheye else f"\n\n{_undistort(f'{sparse_container}/0', images_container)}"

    print(f"Workspace : {workspace}  →  /data")
    print(f"Mapper    : {args.mapper}{' + view_graph_calibrator' if args.view_graph_calibrator else ''}")
    print(f"Database  : {db}")
    print()

    inner = f"""\
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh && conda activate 3dgrut
mkdir -p {sparse_container}
{mapper_cmd}{undistort}
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

    return subprocess.run(cmd).returncode


def _run_headless(workspace: Path, args) -> int:
    db      = "/data/database.db"
    images  = "/data/images"
    sparse  = "/data/sparse"
    sparse0 = f"{sparse}/0"
    fisheye = args.camera_model in FISHEYE_MODELS

    loop_flags = "--SequentialMatching.loop_detection 0"
    if args.loop_detection:
        vtree = _ensure_vocab_tree(workspace, args)
        if vtree is None:
            return 1
        vtree_container = f"/data/{vtree.relative_to(workspace)}"
        loop_flags = (
            f"--SequentialMatching.loop_detection 1 \\\n"
            f"  --SequentialMatching.vocab_tree_path {vtree_container}"
        )

    if args.undistort_only:
        steps = _undistort(sparse0, images)
        print(f"Workspace : {workspace}  →  /data")
        print("Running   : image_undistorter only")
    else:
        # Step 3: mapper command
        if args.mapper == "global":
            # GLOMAP global mapper (COLMAP 4.0+): estimates all poses simultaneously.
            # --view-graph-calibrator refines intrinsics from two-view geometries first,
            # which is strongly recommended for fisheye/wide-angle cameras.
            view_graph_step = ""
            if args.view_graph_calibrator:
                view_graph_step = (
                    f"# 3a. View-graph calibrator — refine intrinsics before global mapper\n"
                    f"colmap view_graph_calibrator \\\n"
                    f"  --database_path {db}\n\n"
                )
            mapper_step = (
                f"{view_graph_step}"
                f"# {'3b' if args.view_graph_calibrator else '3'}. GLOMAP global mapper\n"
                f"mkdir -p {sparse}\n"
                f"colmap global_mapper \\\n"
                f"  --database_path {db} \\\n"
                f"  --image_path {images} \\\n"
                f"  --output_path {sparse}\n\n"
            )
        else:
            mapper_step = (
                f"# 3. Incremental mapper\n"
                f"mkdir -p {sparse}\n"
                f"colmap mapper \\\n"
                f"  --database_path {db} \\\n"
                f"  --image_path {images} \\\n"
                f"  --output_path {sparse}\n\n"
            )

        steps = (
            f"# 1. Feature extraction — GPU SIFT\n"
            f"colmap feature_extractor \\\n"
            f"  --database_path {db} \\\n"
            f"  --image_path {images} \\\n"
            f"  --ImageReader.camera_model {args.camera_model} \\\n"
            f"  --ImageReader.single_camera 1 \\\n"
            f"  --SiftExtraction.use_gpu 1\n\n"
            f"# 2. {args.matcher.capitalize()} matching\n"
            f"colmap {args.matcher}_matcher \\\n"
            f"  --database_path {db} \\\n"
            f"  --SiftMatching.use_gpu 1 \\\n"
            f"  {loop_flags}\n\n"
            f"{mapper_step}"
        )
        if not fisheye:
            steps += _undistort(sparse0, images)

        print(f"Workspace     : {workspace}  →  /data")
        print(f"Camera model  : {args.camera_model}")
        print(f"Matcher       : {args.matcher}")
        print(f"Mapper        : {args.mapper}{' + view_graph_calibrator' if getattr(args, 'view_graph_calibrator', False) else ''}")
        print(f"Loop detect   : {'ON' if args.loop_detection else 'OFF'}")
        print()

    inner = f"""\
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh && conda activate 3dgrut
mkdir -p /data/images /data/sparse
{steps}
"""

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--net=host", "--ipc=host",
        "-v", f"{workspace}:/data",
        "--runtime=nvidia",
        args.image,
        "bash", "-lc", inner,
    ]
    rc = subprocess.run(cmd).returncode

    return rc


def _undistort(sparse0: str, images: str) -> str:
    return (
        f"# Undistort — converts to pinhole, writes real JPEGs to /data/images/\n"
        f"colmap image_undistorter \\\n"
        f"  --image_path {images} \\\n"
        f"  --input_path {sparse0} \\\n"
        f"  --output_path /data \\\n"
        f"  --output_type COLMAP\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
