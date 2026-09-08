"""Reshape and describe a LeRobot v3 dataset — model-agnostic, so all three playbooks
share them.

These operate on the parquet + meta layout directly rather than through
`LeRobotDataset`, which is why they are safe in `vla_common`: the dataset class API
moved between lerobot 0.5.1 and 0.6.1 (16 -> 21 `__init__` parameters) and the two
pipeline environments pin different versions.

    split.py           derive the train / held-out episode split from meta/info.json
    view.py            resolve (and freshen) a camera-subset view for a training run
    camera_variant.py  build a camera-subset view without copying video
    retask.py          rename the instruction string, in place
    trim_variant.py    trim dead air off every episode

`split.py` and `view.py` are stdlib-only and are what `run_training.sh` calls; the other
three need pandas and are the tools that actually rewrite a dataset.
"""
