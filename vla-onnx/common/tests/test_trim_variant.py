"""vla_common.dataset.trim_variant --bounds command on a synthetic two-episode v3 dataset."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd

FPS = 30
CAM = "observation.images.front"


def make_dataset(root):
    # ep 0: 300 frames, commands at 200..219 -> keeps 50..249.
    # ep 1: 120 frames, commands at 30..39, under 5 s in -> lead-in clamps to frame 0, keeps 0..69.
    lengths, active = [300, 120], [(200, 219), (30, 39)]
    rows, eps, cursor, t0 = [], [], 0, 0.0
    for e, (n, (a, b)) in enumerate(zip(lengths, active)):
        act = np.zeros((n, 2), np.float32)
        act[a:b + 1, 0] = 50.0
        act[a:b + 1, 1] = -20.0
        for i in range(n):
            rows.append({"action": act[i], "observation.state": np.zeros(2, np.float32),
                         "timestamp": np.float32(i / FPS), "frame_index": i, "episode_index": e,
                         "index": cursor + i, "task_index": e})
        eps.append({"episode_index": e, "length": n, "dataset_from_index": cursor,
                    "dataset_to_index": cursor + n, f"videos/{CAM}/from_timestamp": t0,
                    f"videos/{CAM}/to_timestamp": t0 + n / FPS, "tasks": [f"task {e}"]})
        cursor += n
        t0 += n / FPS
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "videos").mkdir()
    pd.DataFrame(rows).to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    pd.DataFrame(eps).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    pd.DataFrame({"task_index": [0, 1]}, index=["task 0", "task 1"]).rename_axis("task") \
        .to_parquet(root / "meta/tasks.parquet")
    info = {"fps": FPS, "total_episodes": 2, "total_frames": cursor, "splits": {"train": "0:2"},
            "features": {"action": {"dtype": "float32", "shape": [2], "names": ["boom", "bucket"]},
                         CAM: {"dtype": "video", "shape": [8, 8, 3]}}}
    (root / "meta/info.json").write_text(json.dumps(info))
    stats = {k: {"mean": [0.0, 0.0]} for k in ("action", "observation.state")}
    stats[CAM] = {"mean": [[[0.5]], [[0.5]], [[0.5]]]}
    (root / "meta/stats.json").write_text(json.dumps(stats))


def test_command_bounds_margins_and_scale(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_dataset(src)
    subprocess.run([sys.executable, "-m", "vla_common.dataset.trim_variant", "--src", str(src),
                    "--dst", str(dst), "--bounds", "command", "--before", "5", "--after", "1",
                    "--action-scale", "0.01", "--min-frames", "1", "--apply"],
                   check=True, capture_output=True)

    data = pd.read_parquet(dst / "data/chunk-000/file-000.parquet")
    eps = pd.read_parquet(dst / "meta/episodes/chunk-000/file-000.parquet")
    # ep 0 keeps 200-150 .. 219+30 = 50..249; ep 1 clamps the lead-in at 0: 0..69.
    assert eps["length"].tolist() == [200, 70]
    assert eps[f"videos/{CAM}/from_timestamp"].tolist() == [50 / FPS, 300 / FPS + 0.0]
    assert np.allclose(eps[f"videos/{CAM}/to_timestamp"], [250 / FPS, 300 / FPS + 70 / FPS])
    act = np.stack(data["action"].to_numpy())
    assert np.allclose(act.max(axis=0), [0.5, 0.0]) and np.allclose(act.min(axis=0), [0.0, -0.2])
    # The tail is zeros, so end-of-episode padding repeats "stop", not a command.
    last = data.groupby("episode_index").tail(1)
    assert not np.stack(last["action"].to_numpy()).any()
    assert data["index"].tolist() == list(range(270))

    info = json.loads((dst / "meta/info.json").read_text())
    assert info["features"]["action"]["names"] == ["boom*0.01", "bucket*0.01"]
    assert info["total_frames"] == 270
    stats = json.loads((dst / "meta/stats.json").read_text())
    assert np.isclose(stats["action"]["max"][0], 0.5)
