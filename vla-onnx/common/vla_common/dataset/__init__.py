"""Reshape and describe a LeRobot v3 dataset — model-agnostic, so all three playbooks
share them.

These operate on the parquet + meta layout directly rather than through
`LeRobotDataset`, whose API moves between lerobot releases (16 -> 21 `__init__`
parameters from 0.5.1 to 0.6.1).

    split.py           derive the train / held-out episode split from meta/info.json
    view.py            resolve (and freshen) a camera-subset view for a training run
    camera_variant.py  build a camera-subset view without copying video
    state_variant.py   drop channels from observation.state (a sensor the policy must not see)
    retask.py          rename the instruction string, in place
    trim_variant.py    trim dead air off every episode
    transcode.py       re-encode videos as H.264 (AV1 recordings do not decode here)
    to_v21.py          write a lossless, verified LeRobot v2.1 copy (for openpi and GR00T)
    from_native.py     raw kaivuri native sessions -> simplified LeRobot v2.1 (openpi, GR00T)

`split.py` and `view.py` are stdlib-only and are what `run_training.sh` calls; the other
four need pandas and are the tools that actually rewrite a dataset. `to_v21` and
`from_native` need pyarrow and PyAV (`av`) instead; a lerobot 0.6.1 environment has both.
"""
