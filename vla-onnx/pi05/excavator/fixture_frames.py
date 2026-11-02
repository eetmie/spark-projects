"""Pick export fixture frames from the training dataset (runs in the OpenPI venv, which has lerobot).

    python excavator/fixture_frames.py --out <work>/frames.npz [--count 5]

One npz per call: frame_<i>/<policy input key> arrays in the excavator policy's input format
(observation/cam1, observation/cam2 uint8 HxWx3, observation/state) plus prompt and the source
dataset index. Frames are spread evenly over the dataset, so every task is covered when present.
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spark_env  # noqa: E402, F401
import config  # noqa: E402

config.register()

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from openpi.training import config as _config  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--out", type=pathlib.Path, required=True)
p.add_argument("--config", default="pi05_excavator")
p.add_argument("--count", type=int, default=5)
args = p.parse_args()

cfg = _config.get_config(args.config)
ds = LeRobotDataset(cfg.data.create(cfg.assets_dirs, cfg.model).repo_id)
indices = np.linspace(0, len(ds) - 1, args.count).round().astype(int)
arrays = {}
for i, index in enumerate(indices):
    item = ds[int(index)]
    for key in ("cam1", "cam2"):
        img = np.asarray(item[f"observation.images.{key}"])
        arrays[f"frame_{i}/observation/{key}"] = (img.transpose(1, 2, 0) * 255).round().astype(np.uint8)
    arrays[f"frame_{i}/observation/state"] = np.asarray(item["observation.state"], np.float32)
    arrays[f"frame_{i}/prompt"] = np.asarray(item["task"])
    arrays[f"frame_{i}/index"] = np.asarray(index)
    print(f"fixture {i}: dataset index {index}, task {item['task']!r}, state {arrays[f'frame_{i}/observation/state']}")
tasks = [ds.meta.tasks[i] for i in sorted(ds.meta.tasks)]  # task_index order (the D-pad order)
print("tasks:", tasks)

# Recorder provenance (native/session-*/provenance/manifest.json): every session must agree on the
# cameras and controller profile the state and images depend on.
# A merged set (digging_complete) has no native/ of its own: KAIVURI_PROVENANCE_FROM names the
# native sets it was merged from (comma-separated, names under HF_LEROBOT_HOME or paths), and their
# sessions must agree in the same way.
roots = [pathlib.Path(ds.root)]
if not list((roots[0] / "native").glob("session-*/provenance/manifest.json")) and os.environ.get("KAIVURI_PROVENANCE_FROM"):
    home = pathlib.Path(os.environ["HF_LEROBOT_HOME"])
    roots = [home / r if not pathlib.Path(r).is_absolute() else pathlib.Path(r)
             for r in os.environ["KAIVURI_PROVENANCE_FROM"].split(",") if r]
    print(f"provenance from {[str(r) for r in roots]} (KAIVURI_PROVENANCE_FROM)")
sessions = sorted(m for r in roots for m in (r / "native").glob("session-*/provenance/manifest.json"))
if roots[0] != pathlib.Path(ds.root) and not sessions:
    sys.exit(f"KAIVURI_PROVENANCE_FROM: no native/session-*/provenance/manifest.json under {roots}")
provenance = None
for path in sessions:
    m = json.loads(path.read_text())
    current = {"camera_order": m["camera_order"], "camera_rotations_deg": m["camera_rotations_deg"],
               "profile_files": {k: v for k, v in m["files"].items() if "profiles/" in k},
               "revision": m["revision"]}
    if provenance is None:
        provenance = current
    elif {k: v for k, v in current.items() if k != "revision"} != {k: v for k, v in provenance.items() if k != "revision"}:
        sys.exit(f"{path}: cameras/profile differ from {sessions[0]}")
info = json.loads((pathlib.Path(ds.root) / "meta" / "info.json").read_text())
features = {k: info["features"][k].get("names") for k in ("observation.state", "action")}
np.savez_compressed(args.out, count=np.asarray(args.count), tasks=np.asarray(tasks), fps=np.asarray(ds.fps),
                    provenance=np.asarray(json.dumps(provenance)), features=np.asarray(json.dumps(features)),
                    **arrays)
