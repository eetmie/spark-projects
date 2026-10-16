"""Load a fine-tuned checkpoint as an OpenPI policy and run it on dataset frames.

    python excavator/check_policy.py checkpoints/pi05_excavator/<exp>/<step> [--frames 0 2000 4000]

Prints the predicted chunk shape, timing and the error against the recorded action chunk.
"""

import argparse
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spark_env  # noqa: E402, F401
import config  # noqa: E402

config.register()

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("--config", default="pi05_excavator")
p.add_argument("--frames", type=int, nargs="+", default=[0, 2000, 4000])
args = p.parse_args()

cfg = _config.get_config(args.config)
data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
horizon = cfg.model.action_horizon
ds = LeRobotDataset(data_cfg.repo_id, delta_timestamps={"action": [t / 30 for t in range(horizon)]})
policy = policy_config.create_trained_policy(cfg, args.checkpoint)


def hwc(img):
    return (np.asarray(img).transpose(1, 2, 0) * 255).astype(np.uint8)


for i in args.frames:
    item = ds[i]
    obs = {
        "observation/cam1": hwc(item["observation.images.cam1"]),
        "observation/cam2": hwc(item["observation.images.cam2"]),
        "observation/state": np.asarray(item["observation.state"]),
        "prompt": item["task"],
    }
    t = time.perf_counter()
    out = policy.infer(obs)
    dt = time.perf_counter() - t
    pred, gt = out["actions"], np.asarray(item["action"])
    mae = np.abs(pred - gt).mean(axis=0)
    print(f"frame {i} ({item['task']}): chunk {pred.shape}, {dt * 1000:.0f} ms, "
          f"finite={np.isfinite(pred).all()}, MAE per dim {np.round(mae, 3)}")
