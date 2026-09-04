"""Reshape a LeRobot v3 dataset — model-agnostic, so all three playbooks share them.

These operate on the parquet + meta layout directly rather than through
`LeRobotDataset`, which is why they are safe in `vla_common`: the dataset class API
moved between lerobot 0.5.1 and 0.6.1 (16 -> 21 `__init__` parameters) and the two
pipeline environments pin different versions.

`resample_dataset.py` is the exception and stays model-side: it constructs
`LeRobotDataset(...)` and `LeRobotDataset.create(...)`, so it must run in the venv
whose lerobot it was written against.

    retask.py          rename the instruction string, in place
    camera_variant.py  build a camera-subset view without copying video
    trim_variant.py    trim dead air off every episode
"""
