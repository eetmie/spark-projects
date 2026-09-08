#!/usr/bin/env python3
"""Fast, non-training validation for the local X-VLA fine-tuning stack."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import shutil
import sys
from pathlib import Path
from vla_common.paths import dataset, playbook

ROOT = playbook("xvla")
EXPECTED_REVISION = "cdb7964e4fe842935d671bfab5a5ebe00a96648c"
EXPECTED_WEIGHT_BYTES = 3_519_073_692



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # No run->dataset table here any more. It was the fourth copy of the same mapping
    # and drifted from the others silently; the dataset is now named the way
    # run_training.sh names it, and the view is resolved by the same code.
    parser.add_argument("--dataset", default=None,
                        help="recording to check the contract of (default: env checks only)")
    parser.add_argument("--cameras", default=None, help="e.g. cam1 or cam1,cam2")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/xvla-base-excavator")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--save-freq", type=int, default=5_000)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")
    print(f"ok  {message}")


def shape(features: dict, key: str) -> tuple[int, ...]:
    require(key in features, f"dataset feature {key} exists")
    return tuple(features[key]["shape"])


def main() -> None:
    args = parse_args()
    require(sys.version_info[:2] == (3, 12), f"Python 3.12 ({sys.version.split()[0]})")

    versions = {name: importlib.metadata.version(name) for name in ("torch", "lerobot", "transformers")}
    require(versions["lerobot"] == "0.5.1", f"LeRobot 0.5.1 ({versions['lerobot']})")
    require(versions["torch"].startswith("2.12.0+cu130"),
            f"CUDA 13 PyTorch 2.12 ({versions['torch']})")
    require(versions["transformers"] == "5.3.0", f"Transformers 5.3.0 ({versions['transformers']})")

    import torch
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    require(torch.cuda.is_available(), "CUDA is available")
    require(torch.cuda.is_bf16_supported(), "GPU supports bfloat16 training")
    require(XVLAConfig is not None and XVLAPolicy is not None, "X-VLA policy imports")

    train_cli = Path(sys.executable).with_name("lerobot-train")
    require(train_cli.is_file(), f"trainer exists at {train_cli}")

    checkpoint = args.checkpoint.resolve()
    weights = checkpoint / "model.safetensors"
    config_path = checkpoint / "config.json"
    require(weights.is_file(), f"checkpoint weights exist at {weights}")
    require(weights.stat().st_size == EXPECTED_WEIGHT_BYTES,
            f"checkpoint size is {EXPECTED_WEIGHT_BYTES} bytes")
    require(config_path.is_file(), "checkpoint config exists")
    config = json.loads(config_path.read_text())
    require(config.get("type") == "xvla", "checkpoint policy type is xvla")
    require(config.get("input_features") == {}, "derived checkpoint lets the dataset define inputs")
    require(config.get("output_features") == {}, "derived checkpoint lets the dataset define actions")
    revision_path = checkpoint / "REVISION"
    if revision_path.exists():
        require(revision_path.read_text().strip() == EXPECTED_REVISION,
                f"base model revision is {EXPECTED_REVISION}")
    else:
        print("warn checkpoint predates REVISION marker; Hub checksum was verified separately")

    if args.dataset:
        from vla_common.dataset import split as split_mod
        from vla_common.dataset import view as view_mod

        src = Path(args.dataset)
        if not src.is_dir():
            src = dataset(args.dataset)
        cams = args.cameras.split(",") if args.cameras else None
        # Resolve exactly as run_training.sh will, so a green preflight means the run
        # will see the same dataset root -- including a rebuilt view if the source grew.
        dataset_root = view_mod.resolve(src, cams)
        info_path = dataset_root / "meta/info.json"
        require(info_path.is_file(), f"dataset metadata exists at {dataset_root}")
        info = json.loads(info_path.read_text())
        features = info["features"]
        require(shape(features, "observation.state") == (3,), "state contract is 3-D")
        require(shape(features, "action") == (4,), "action contract is 4-D")
        for camera in view_mod.source_cameras(dataset_root):
            require(features[camera]["dtype"] == "video",
                    f"camera contract {camera} is video")
        n_eps, val_eps, train_eps = split_mod.split(dataset_root)
        require(len(train_eps) > 0,
                f"dataset has {n_eps} episodes ({len(train_eps)} train, {len(val_eps)} held out)")
    else:
        print("skip dataset contract (pass --dataset NAME to check one)")

    free = shutil.disk_usage(ROOT).free
    # The measured FP32 probe checkpoint was 6.01 GB. BF16 halves model and Adam state,
    # but retain the conservative FP32 figure so a long run never fills the workstation.
    checkpoint_bytes = 6_100_000_000
    saves = math.ceil(args.steps / args.save_freq) if args.save_freq > 0 else 1
    estimate = saves * checkpoint_bytes
    require(free >= estimate + 10 * 1024**3,
            f"disk headroom: {free / 1024**3:.1f} GiB free, "
            f"conservative checkpoint budget {estimate / 1024**3:.1f} GiB")

    print(f"\nREADY: {args.dataset or 'env only'}, {args.steps} steps, save every {args.save_freq}")


if __name__ == "__main__":
    main()
