"""vla_common.dataset.from_native on a synthetic two-run native session."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

H, W, FPS = 48, 64, 30
TASKS = ("scoop the rocks", "scoop the sand")


def write_take(path: Path, n: int, seed: int, armed=True):
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True)
    frames = {k: rng.integers(0, 256, (n, H, W), dtype=np.uint8) for k in ("cam1", "cam2")}
    for k, f in frames.items():
        f.tofile(path / f"{k}.y8")
    t0 = 10**12 + seed * 10**10
    with (path / "samples.jsonl").open("w") as out:
        for i in range(n):
            t = t0 + round(i * 1e9 / FPS)
            out.write(json.dumps({"sample_ns": t + 50_000, "requested_action": [0] * 6}) + "\n")  # status-only row
            status = {"state": "armed" if armed or i < n - 1 else "disarmed", "fk_valid": True, "imu_healthy": True,
                      "angles_rad": [0.1, 0.2 + i * 0.01, -0.3, 1.0], "applied_action": [0.5, -0.25, 0, 1, 0, 0],
                      "imu_host_monotonic_ns": t - 1_000_000, "applied_host_monotonic_ns": t - 5_000_000}
            cams = {k: {"sequence": 100 + i, "host_monotonic_ns": t - (3 + 4 * c) * 1_000_000, "width": W,
                        "height": H, "stride": W, "pixel_bytes": W * H} for c, k in enumerate(("cam1", "cam2"))}
            out.write(json.dumps({"sample_ns": t, "frame_index": i, "status": status, **cams}) + "\n")
    (path / "episode.json").write_text(json.dumps({"episode_number": seed, "frames": n, "depth": []}))
    return frames


def write_session(path: Path, task: str, takes: list[tuple[int, int]], armed=True):
    path.mkdir(parents=True, exist_ok=True)
    (path / "session.json").write_text(json.dumps({"format": "kaivuri-native-episode-v1", "task": task, "fps": FPS,
                                                   "enable_tracks": False, "revision": "abc"}))
    (path / "provenance").mkdir()
    (path / "provenance/manifest.json").write_text(json.dumps({"camera_order": ["a:2", "b:1"], "revision": "abc"}))
    return [write_take(path / f"episode-{1000 + s}.ready", n, s, armed) for n, s in takes]


def run(*args):
    return subprocess.run([sys.executable, "-m", "vla_common.dataset.from_native", *map(str, args)],
                          capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("native")
    session = tmp / "session"
    frames = write_session(session, TASKS[0], [(8, 1), (6, 2)])
    frames += write_session(session / "recordings" / "run-000002-x", TASKS[1], [(7, 3)])
    out = run(session, "--dst", tmp / "out", "--crf", "0", "--jobs", "2")
    assert out.returncode == 0, out.stderr
    return tmp / "out", frames


def test_layout_and_meta(converted):
    root, _ = converted
    info = json.loads((root / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.1" and info["total_episodes"] == 3 and info["total_frames"] == 21
    assert info["features"]["observation.state"]["names"] == ["lift", "tilt", "scoop"]
    assert info["features"]["action"]["names"] == ["slew", "lift", "tilt", "scoop"]
    tasks = [json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines()]
    assert [t["task"] for t in tasks] == list(TASKS)
    eps = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    assert [(e["length"], e["tasks"]) for e in eps] == [(8, [TASKS[0]]), (6, [TASKS[0]]), (7, [TASKS[1]])]
    stats = json.loads((root / "meta/episodes_stats.jsonl").read_text().splitlines()[0])["stats"]
    assert np.asarray(stats["observation.images.cam1"]["mean"]).shape == (3, 1, 1)
    # The pi0.5 export (fixture_frames.py) reads cameras/profile from this layout.
    assert len(sorted((root / "native").glob("session-*/provenance/manifest.json"))) == 2
    assert (root / "native/session-001/session.json").exists()
    for e in range(3):
        for cam in ("cam1", "cam2"):
            assert (root / f"videos/chunk-000/observation.images.{cam}/episode_{e:06d}.mp4").exists()


def test_parquet_values(converted):
    root, _ = converted
    t = pq.read_table(root / "data/chunk-000/episode_000001.parquet").to_pydict()
    assert t["index"] == list(range(8, 14)) and t["frame_index"] == list(range(6))
    assert set(t["task_index"]) == {0} and set(t["episode_index"]) == {1}
    np.testing.assert_allclose(t["observation.state"][2], np.rad2deg([0.22, -0.3, 1.0]), rtol=1e-6)
    assert t["action"][0] == [0.5, -0.25, 0.0, 1.0]
    np.testing.assert_allclose(t["timestamp"], np.arange(6) / FPS, rtol=1e-6)
    np.testing.assert_allclose(t["clock.cam1.age_s"], 0.003, atol=1e-6)
    np.testing.assert_allclose(t["clock.cam2.age_s"], 0.007, atol=1e-6)
    np.testing.assert_allclose(t["clock.state_age_s"], 0.001, atol=1e-6)
    np.testing.assert_allclose(t["clock.action_age_s"], 0.005, atol=1e-6)
    hf = json.loads(pq.read_schema(root / "data/chunk-000/episode_000001.parquet").metadata[b"huggingface"])
    assert hf["info"]["features"]["observation.state"]["_type"] == "Sequence"
    assert pq.read_table(root / "data/chunk-000/episode_000002.parquet")["task_index"].to_pylist() == [1] * 7


def test_video_close_to_raw(converted):
    import av

    root, frames = converted
    with av.open(str(root / "videos/chunk-000/observation.images.cam2/episode_000002.mp4")) as c:
        dec = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
    assert dec.shape == (7, H, W, 3)
    # CRF 0 on yuv420p: only the RGB -> limited-range YUV rounding is left.
    assert np.abs(dec[..., 0].astype(int) - frames[2]["cam2"]).max() <= 2
    src = json.loads((root / "meta/native_source.json").read_text())
    assert [e["frames"] for e in src["episodes"]] == [8, 6, 7]
    assert math.isclose(src["timing"][0]["cam2.age_ms"]["p50"], 7.0, abs_tol=0.01)


def test_refuses_bad_take(tmp_path):
    write_session(tmp_path / "s", TASKS[0], [(5, 1)], armed=False)
    out = run(tmp_path / "s", "--dst", tmp_path / "out")
    assert out.returncode != 0 and "invalid FK/controller state" in out.stderr


def test_refuses_existing_dst(tmp_path):
    write_session(tmp_path / "s", TASKS[0], [(5, 1)])
    (tmp_path / "out").mkdir()
    assert "exists" in run(tmp_path / "s", "--dst", tmp_path / "out").stderr
