"""Excavator pi0.5 TrainConfigs, registered into OpenPI at runtime (no OpenPI fork).

Dataset: kaivuriprokkis native recorder export, converted to LeRobot v2.1 with
`vla_common.dataset.to_v21`. Keys: observation.images.{cam1,cam2} (IR, gray repeated
to 3 channels), observation.state = [lift, tilt, scoop] deg, action = [slew, lift, tilt,
scoop(, trackL, trackR)], task -> prompt.

`discrete_state_input=True` (decided 2026-10-16): pi0.5 has no continuous state input,
so the joint angles reach the model only as quantile-normalized 256-bin integers in the
prompt ("Task: ..., State: 131 87 200;\\nAction: "). `False` would drop them entirely.
The exporter and the C++ runtime still implement only the fixed-prompt pi05_libero
contract and must build this prompt per step.
"""

import dataclasses
import pathlib

import numpy as np
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.policies.libero_policy import _parse_image
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders

PI05_BASE = "gs://openpi-assets/checkpoints/pi05_base/params"


@dataclasses.dataclass(frozen=True)
class ExcavatorInputs(_transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        cam1 = _parse_image(data["observation/cam1"])
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": cam1,
                "left_wrist_0_rgb": _parse_image(data["observation/cam2"]),
                "right_wrist_0_rgb": np.zeros_like(cam1),
            },
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.False_},
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class ExcavatorOutputs(_transforms.DataTransformFn):
    action_dim: int = 4

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}


@dataclasses.dataclass(frozen=True)
class ExcavatorDataConfig(_config.DataConfigFactory):
    action_dim: int = 4

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/cam1": "observation.images.cam1",
                        "observation/cam2": "observation.images.cam2",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[ExcavatorInputs(model_type=model_config.model_type)],
            outputs=[ExcavatorOutputs(action_dim=self.action_dim)],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )


CONFIGS = [
    _config.TrainConfig(
        name="pi05_excavator",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50, discrete_state_input=True),
        data=ExcavatorDataConfig(
            repo_id="masi_native_test_v21",
            base_config=_config.DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # EMA adds a full fp32 parameter copy (~13 GB) on device and in every checkpoint save;
        # it does not fit next to AdamW + the save's host copy in the Spark's 121 GB.
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader(PI05_BASE),
        num_train_steps=30_000,
        keep_period=None,
        wandb_enabled=False,
    ),
]


def register() -> None:
    for cfg in CONFIGS:
        if cfg.name not in _config._CONFIGS_DICT:
            _config._CONFIGS.append(cfg)
            _config._CONFIGS_DICT[cfg.name] = cfg
