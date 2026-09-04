#!/usr/bin/env python
"""Run a fine-tuned excavator SmolVLA checkpoint.

Import ExcavatorPolicy for a robot loop, or run this file directly for a demo
against a held-out episode with latency numbers.

    from excavator.infer import ExcavatorPolicy
    pol = ExcavatorPolicy("~/smolvla/outputs/excavator/A/checkpoints/006000/pretrained_model")
    pol.reset()                                  # once per episode
    while running:
        action = pol.select_action(image, state) # -> (4,) float32
        send_to_machine(action)

WHAT GOES IN
    image : RGB frame from cam1. Either HWC uint8 (0-255) or CHW float (0-1).
            Any resolution -- the policy resizes to 512x512 with padding itself.
    state : (4,) joint angles in DEGREES, order [slew, lift, tilt, scoop].
            Pass them RAW; the checkpoint's preprocessor normalizes internally
            using the training statistics. Do not pre-normalize.

WHAT COMES OUT
    action : (4,) joystick RATE commands in [-1, 1], same joint order. These are
             velocity commands, not position targets -- corr(action, measured
             joint velocity) was 0.71-0.88 on the training data.

CONTROL RATE MATTERS
    A checkpoint trained at 30 fps expects to be stepped at 30 Hz; the 10 fps and
    6 fps runs expect 10 Hz and 6 Hz. Stepping a 10 fps model at 30 Hz executes
    each rate command for a third of its intended duration and the machine moves
    a third as far. `policy_fps` below reports the rate the checkpoint expects.

    select_action() returns one action per call and internally replans every
    n_action_steps calls, so just call it once per control tick at policy_fps.
"""

import argparse
import json
import time
from pathlib import Path
from vla_common.paths import dataset, playbook

import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

JOINTS = ["slew", "lift", "tilt", "scoop"]
TASK = "scoop blocks and dump it to the left"
FPS_BY_REPO = {"local/masi_kaivuri_juusto": 30, "local/masi_kaivuri_10fps": 10, "local/masi_kaivuri_6fps": 6}
DEFAULT_CKPT = "outputs/excavator/A/checkpoints/006000/pretrained_model"


class ExcavatorPolicy:
    """Thin wrapper putting the LeRobot processor pipeline behind a numpy interface."""

    def __init__(self, checkpoint: str | Path, device: str = "cuda", task: str = TASK):
        self.checkpoint = Path(checkpoint).expanduser()
        self.device = device
        self.task = task
        self.policy = SmolVLAPolicy.from_pretrained(self.checkpoint)
        self.policy.eval().to(device)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config, pretrained_path=str(self.checkpoint)
        )
        cfg = json.loads((self.checkpoint / "train_config.json").read_text())
        self.policy_fps = FPS_BY_REPO.get(cfg["dataset"]["repo_id"])
        self.chunk_size = self.policy.config.chunk_size
        self.n_action_steps = self.policy.config.n_action_steps
        self.image_key = next(k for k in self.policy.config.input_features if k.startswith("observation.images"))

    def reset(self):
        """Clear the action queue. Call once at the start of each episode."""
        self.policy.reset()

    def _batch(self, image, state):
        img = torch.as_tensor(np.ascontiguousarray(image))
        if img.ndim != 3:
            raise ValueError(f"image must be 3-dimensional, got shape {tuple(img.shape)}")
        if img.shape[-1] == 3:  # HWC -> CHW
            img = img.permute(2, 0, 1)
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        st = torch.as_tensor(np.asarray(state, dtype=np.float32)).reshape(-1)
        if st.numel() != len(JOINTS):
            raise ValueError(f"state must have {len(JOINTS)} elements, got {st.numel()}")
        return {
            "observation.state": st.unsqueeze(0),
            self.image_key: img.unsqueeze(0),
            "task": [self.task],
        }

    @torch.no_grad()
    def predict_chunk(self, image, state) -> np.ndarray:
        """Full action chunk, shape (chunk_size, 4), covering chunk_size/policy_fps seconds."""
        batch = self.preprocessor(self._batch(image, state))
        return self.postprocessor(self.policy.predict_action_chunk(batch)).float().cpu().numpy()[0]

    @torch.no_grad()
    def select_action(self, image, state) -> np.ndarray:
        """One action, shape (4,). Replans internally every n_action_steps calls."""
        batch = self.preprocessor(self._batch(image, state))
        return self.postprocessor(self.policy.select_action(batch)).float().cpu().numpy().reshape(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--episode", type=int, default=3, help="held-out episode to demo on")
    p.add_argument("--steps", type=int, default=60, help="control ticks to run")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import pandas as pd

    src = dataset("masi_kaivuri_juusto")
    pol = ExcavatorPolicy(args.checkpoint, device=args.device)
    print(f"checkpoint      : {pol.checkpoint}")
    print(f"expects         : {pol.policy_fps} Hz control rate")
    print(f"chunk_size      : {pol.chunk_size} actions = {pol.chunk_size / pol.policy_fps:.2f} s")
    print(f"replans every   : {pol.n_action_steps} ticks")
    print(f"image key       : {pol.image_key}\n")

    ds = LeRobotDataset(repo_id="local/src", root=src, video_backend="torchcodec")
    eps = pd.read_parquet(src / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    row = eps[eps["episode_index"] == args.episode].iloc[0]
    start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
    gt = np.stack(pd.read_parquet(src / "data" / "chunk-000" / "file-000.parquet")["action"].to_numpy())

    # The 30 fps recording is subsampled to whatever rate this checkpoint expects.
    stride = 30 // pol.policy_fps
    pol.reset()
    latencies, preds, truth = [], [], []
    for i in range(args.steps):
        idx = start + i * stride
        if idx >= stop:
            break
        item = ds[idx]
        img = item[pol.image_key]
        state = item["observation.state"].numpy()
        t0 = time.perf_counter()
        action = pol.select_action(img, state)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
        preds.append(action)
        truth.append(gt[idx])

    preds, truth = np.array(preds), np.array(truth)
    lat = np.array(latencies)
    replan = lat > np.median(lat) * 3  # the ticks that actually ran the model

    print(f"ran {len(preds)} control ticks on held-out episode {args.episode}\n")
    print("latency (ms):")
    print(f"  replanning ticks ({replan.sum():>3}) : mean {lat[replan].mean():7.1f}  max {lat[replan].max():7.1f}")
    if (~replan).any():
        print(f"  queued ticks     ({(~replan).sum():>3}) : mean {lat[~replan].mean():7.1f}  max {lat[~replan].max():7.1f}")
    print(f"  budget at {pol.policy_fps} Hz          : {1000 / pol.policy_fps:.1f} ms/tick")
    print(f"\nmean |predicted - actual| per joint (rate command units):")
    for j, name in enumerate(JOINTS):
        print(f"  {name:>5}: {np.abs(preds[:, j] - truth[:, j]).mean():.4f}")
    print(f"\nfirst 3 predicted actions:\n{np.array2string(preds[:3], precision=3, suppress_small=True)}")
    print(f"ground truth:\n{np.array2string(truth[:3], precision=3, suppress_small=True)}")


if __name__ == "__main__":
    main()
