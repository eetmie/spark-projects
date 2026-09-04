#!/usr/bin/env python3
"""Validate the pinned EVO1 Spark environment and optionally instantiate the policy."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = ROOT / "models" / "InternVL3-1B-hf"
EXPECTED_REVISION = "014c0583a0d4bedf29fbe2dbff4f865eb998e171"
EXPECTED_VLM_PARAMETERS = 938_193_024
EXPECTED_POLICY_PARAMETERS = 775_198_640
EXPECTED_STAGE1_TRAINABLE_PARAMETERS = 122_029_360
EXPECTED_POLICY_STATE_ELEMENTS = 776_139_440

EXPECTED_VERSIONS = {
    "lerobot": "0.6.1",
    "torch": "2.11.0+cu130",
    "torchvision": "0.26.0+cu130",
    "torchcodec": "0.11.1+cu130",
    "transformers": "5.5.4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="also load the truncated InternVL3 backbone and freshly initialized action head on CUDA",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")
    print(f"ok  {message}")


def safetensors_element_count(path: Path) -> int:
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        require(len(raw_size) == 8, f"safetensors header prefix exists in {path}")
        (header_size,) = struct.unpack("<Q", raw_size)
        header = json.loads(handle.read(header_size))

    total = 0
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        count = 1
        for dimension in metadata["shape"]:
            count *= dimension
        total += count
    return total


def load_policy(base: Path) -> None:
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.evo1.configuration_evo1 import Evo1Config
    from lerobot.policies.evo1.modeling_evo1 import Evo1Policy
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    config = Evo1Config(
        device="cuda",
        training_stage="stage1",
        vlm_model_name=str(base),
        use_flash_attn=False,
        input_features={
            OBS_IMAGES + ".image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 448, 448)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(24,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(24,))},
    )
    policy = Evo1Policy(config)
    total = sum(parameter.numel() for parameter in policy.parameters())
    trainable = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
    state_elements = sum(tensor.numel() for tensor in policy.state_dict().values())
    require(total == EXPECTED_POLICY_PARAMETERS, f"EVO1 policy has {total:,} parameters")
    require(
        trainable == EXPECTED_STAGE1_TRAINABLE_PARAMETERS,
        f"stage-1 action head has {trainable:,} trainable parameters",
    )
    require(
        state_elements == EXPECTED_POLICY_STATE_ELEMENTS,
        f"policy state has {state_elements:,} serialized tensor elements",
    )
    require(not policy.model.embedder.training, "frozen InternVL3 embedder remains in eval mode")
    del policy
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    base = args.base.resolve()

    require(sys.version_info[:2] == (3, 12), f"Python 3.12 ({sys.version.split()[0]})")
    for package, expected in EXPECTED_VERSIONS.items():
        actual = importlib.metadata.version(package)
        require(actual == expected, f"{package} {expected} ({actual})")

    import torch
    from lerobot.policies.evo1.configuration_evo1 import Evo1Config
    from lerobot.policies.evo1.modeling_evo1 import Evo1Policy

    require(Evo1Config is not None and Evo1Policy is not None, "LeRobot EVO1 policy imports")
    require(torch.cuda.is_available(), "CUDA is available")
    require(torch.cuda.is_bf16_supported(), "GB10 supports bfloat16")
    print(f"    CUDA device: {torch.cuda.get_device_name(0)}")

    revision_path = base / "REVISION"
    require(revision_path.is_file(), f"base provenance marker exists at {revision_path}")
    require(
        revision_path.read_text().strip() == EXPECTED_REVISION,
        f"InternVL3 revision is {EXPECTED_REVISION}",
    )

    config_path = base / "config.json"
    weights_path = base / "model.safetensors"
    require(config_path.is_file(), f"base config exists at {config_path}")
    require(weights_path.is_file(), f"base weights exist at {weights_path}")
    config = json.loads(config_path.read_text())
    require(config.get("model_type") == "internvl", "base uses native Transformers InternVL")
    require(config.get("architectures") == ["InternVLForConditionalGeneration"], "base architecture matches")
    require(
        safetensors_element_count(weights_path) == EXPECTED_VLM_PARAMETERS,
        f"base checkpoint has {EXPECTED_VLM_PARAMETERS:,} parameters",
    )

    if args.load_model:
        load_policy(base)

    print("\nREADY: EVO1 host-side environment and base checkpoint")


if __name__ == "__main__":
    main()
