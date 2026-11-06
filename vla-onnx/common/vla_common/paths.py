"""Path resolution for the vla-onnx pipeline — the Python side of `paths.sh`.

Everything resolves from this file's installed location or from the environment, so
nothing here hardcodes a user, a home directory, or a tree layout.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else default


# vla_common lives in vla-onnx/common/vla_common/, so vla-onnx is three parents up.
VLA_ONNX = _env_path("VLA_ONNX", Path(__file__).resolve().parents[2])
SPARK_PROJECTS = VLA_ONNX.parent

# The recorded excavator datasets — source data, not derived, not in git.
VLA_DATASETS = _env_path("VLA_DATASETS", Path.home() / "Desktop")

# Finished bundles, staged for `ship_bundle.sh`.
VLA_BUNDLES = _env_path("VLA_BUNDLES", Path.home() / "bundles")


def playbook(name: str) -> Path:
    """Root of one model playbook, e.g. playbook("xvla")."""
    return VLA_ONNX / name


def dataset(name: str) -> Path:
    """A recorded dataset by name, e.g. dataset("masi_digging_dry_2")."""
    return VLA_DATASETS / name


# The pipeline environment, mirroring paths.sh.
VENV_LEROBOT061 = _env_path("VENV_LEROBOT061", VLA_ONNX / ".venv-lerobot061")


def checkpoint(playbook_name: str, sweep: str, run: str, step: int | str = "last") -> Path:
    """One checkpoint's `pretrained_model` dir.

    Encodes the layout smolvla and xvla both already write:

        <playbook>/outputs/<sweep>/<run>/checkpoints/<NNNNNN>/pretrained_model/

    The six-digit zero padding is load-bearing, not cosmetic -- it is what makes
    `sorted()` over the checkpoint dirs a numeric sort, which eval_curve relies on to
    walk a run in step order. It was being spelled out with a bare `printf "%06d"` in
    three separate places; a fourth would eventually have disagreed.

    `step="last"` returns the symlink LeRobot maintains, which is the only name that
    does not need to know how far the run got.
    """
    base = playbook(playbook_name) / "outputs" / sweep / run / "checkpoints"
    return base / (step if step == "last" else f"{int(step):06d}") / "pretrained_model"
