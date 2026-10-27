"""Excavator modality config for GR00T N1.7, registered as NEW_EMBODIMENT.

Loaded by launch_finetune.py through `--modality-config-path`. Keys match the
meta/modality.json that prepare_dataset.py writes:

- video: cam1, cam2 (the two IR views, gray repeated to RGB by the recorder export)
- state: joints = observation.state as recorded (joint angles, degrees)
- action: valves = direct normalized valve commands in [-1, 1]. ABSOLUTE and NON_EEF: a valve
  command is not a joint target, so GR00T's relative-action rebasing must not apply.
- language: the episode's task text (task_index -> meta/tasks.jsonl)

Horizon 40 rows = N1.7's base horizon (1.33 s at 30 FPS). Video history is off by default
(current frames only, NVIDIA's custom-embodiment example). GROOT_VIDEO_DELTAS="-30,0" turns on
one past slot 1 s back, the base model's layout; the board's capture tool supports both.
"""

import os

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

ACTION_HORIZON = int(os.environ.get("GROOT_ACTION_HORIZON", "40"))
VIDEO_DELTAS = [int(x) for x in os.environ.get("GROOT_VIDEO_DELTAS", "0").split(",")]

excavator_config = {
    "video": ModalityConfig(delta_indices=VIDEO_DELTAS, modality_keys=["cam1", "cam2"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["joints"]),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["valves"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(excavator_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
