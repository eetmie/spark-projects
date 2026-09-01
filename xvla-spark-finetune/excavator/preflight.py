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

ROOT = Path(__file__).resolve().parents[1]
SPARK = ROOT.parent
SMOL = SPARK / "smolvla-spark-finetune"
EXPECTED_REVISION = "cdb7964e4fe842935d671bfab5a5ebe00a96648c"
EXPECTED_WEIGHT_BYTES = 3_519_073_692

RUNS = {
    "ir": (SMOL / "datasets/masi_digging_ir", ("observation.images.cam1",)),
    "both": (Path("/home/masi-pgx/Desktop/masi_digging"),
             ("observation.images.cam1", "observation.images.cam2")),
    "clean_ir": (SMOL / "datasets/masi_digging_clean_ir", ("observation.images.cam1",)),
    "clean_both": (SMOL / "datasets/masi_digging_clean",
                   ("observation.images.cam1", "observation.images.cam2")),
    "dry_ir": (SMOL / "datasets/masi_digging_dry_ir", ("observation.images.cam1",)),
    "dry2_ir": (SMOL / "datasets/masi_digging_dry2_ir", ("observation.images.cam1",)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=RUNS, default="clean_ir")
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

    dataset_root, cameras = RUNS[args.run]
    info_path = dataset_root / "meta/info.json"
    require(info_path.is_file(), f"{args.run} dataset metadata exists at {dataset_root}")
    info = json.loads(info_path.read_text())
    features = info["features"]
    require(shape(features, "observation.state") == (3,), "state contract is 3-D")
    require(shape(features, "action") == (4,), "action contract is 4-D")
    for camera in cameras:
        require(camera in features and features[camera]["dtype"] == "video",
                f"camera contract {camera} is video")
    n_episodes = int(info["total_episodes"])
    val_episodes = list(range(5, n_episodes, 10))
    require(n_episodes > len(val_episodes),
            f"dataset has {n_episodes} episodes ({n_episodes - len(val_episodes)} train by default)")

    free = shutil.disk_usage(ROOT).free
    # The measured FP32 probe checkpoint was 6.01 GB. BF16 halves model and Adam state,
    # but retain the conservative FP32 figure so a long run never fills the workstation.
    checkpoint_bytes = 6_100_000_000
    saves = math.ceil(args.steps / args.save_freq) if args.save_freq > 0 else 1
    estimate = saves * checkpoint_bytes
    require(free >= estimate + 10 * 1024**3,
            f"disk headroom: {free / 1024**3:.1f} GiB free, "
            f"conservative checkpoint budget {estimate / 1024**3:.1f} GiB")

    print(f"\nREADY: {args.run}, {args.steps} steps, save every {args.save_freq}")


if __name__ == "__main__":
    main()
