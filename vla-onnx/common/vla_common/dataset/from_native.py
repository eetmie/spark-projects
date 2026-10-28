#!/usr/bin/env python
"""Raw kaivuri native sessions -> simplified LeRobot v2.1 training set, in one pass.

    python -m vla_common.dataset.from_native SESSION [SESSION ...] --dst DIR [--crf 18] [--jobs 8] [--v3]

`--v3` also writes DIR_v3, a LeRobot v3.0 copy for the LeRobot policies and lerobot-dataset-viz
(`to_v3`: LeRobot's own converter, stream copy, ~12 s for 100 takes).

Reads the board recorder's committed takes (`episode-*.ready/{samples.jsonl,cam1.y8,cam2.y8}`,
format kaivuri-native-episode-v1); a session's `recordings/run-*` resumes are included
after it, each run with its own task. Writes only what the policy reads:

    observation.images.cam1, cam2   IR Y8 repeated to RGB, H.264
    observation.state               [lift, tilt, scoop] deg (no slew: cameras only)
    action                          applied [slew, lift, tilt, scoop(, trackL, trackR)]
    task                            session.json "task"
    clock.{cam1,cam2}.age_s         camera frame host arrival -> recorder sample
    clock.state_age_s               IMU read (joint angles) -> recorder sample
    clock.action_age_s              valve command applied -> recorder sample

The clock.* columns are not policy inputs (neither trainer reads them); they record how old
each input was when the frame was paired, so inference can be checked against the same
timing. Per-run p50/p99 go to meta/native_source.json "timing".

The board's `export_episodes.py` writes the same state/action/video plus ~60 clock, FK, IMU
and diagnostics columns, as LeRobot v3 that then needs `to_v21`. This skips both. Per-episode
provenance (source path, samples.jsonl sha256, achieved rate, video error) goes to
meta/native_source.json; each run's session.json and provenance/ to native/session-NNN/
(the board exporter's layout, which the pi0.5 export reads for camera and profile checks).

VIDEO. Encoded as LeRobot's recorder does (libx264, yuv420p, GOP 2, RGB input) so it decodes
the same way in OpenPI's pinned LeRobot (PyAV) and GR00T (torchcodec). Lossy on purpose:
the robot feeds raw frames to the policy, as the Python stack did with CRF 30 data. CRF 18
is ~6% of the Y8 size at mean |error| ~2 grey levels (CRF 30: ~1.6%, ~3.2). Bit-exact
lossless needs libx264rgb (~125% of Y8; gray and yuv444 inputs come back up to 20 levels
off in both loaders) and decodes ~5x slower in GR00T's loader; `--lossless` writes that.
Every frame is decoded back and compared with the raw frame.

CHECKS (from the board exporter): contiguous frame indices, armed + FK valid + IMU healthy on
every frame, finite applied actions within [-1, 1], frame count and Y8 sizes match, capture
rate within 3% of the declared FPS and no gap over 1.5 frames.

Needs numpy, pyarrow and PyAV with libx264 (the lerobot 0.6.1 venv has them).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FORMAT = "kaivuri-native-episode-v1"
CAMERAS = ("cam1", "cam2")
JOINTS = {"slew": 0, "lift": 1, "tilt": 2, "scoop": 3}
ACTIONS = ("slew", "lift", "tilt", "scoop", "trackL", "trackR")
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
CHUNK = 1000
GOP = 2
AGE_COLUMNS = ("clock.cam1.age_s", "clock.cam2.age_s", "clock.state_age_s", "clock.action_age_s")


# ---------------------------------------------------------------- sessions and checks


def recording_runs(session: Path) -> list[Path]:
    """The session, then its recordings/run-* resumes in order (native_sessions.recording_paths)."""
    if not (session / "session.json").is_file():
        raise SystemExit(f"{session}: missing session.json")
    runs = [session]
    for run in sorted((session / "recordings").glob("run-*")):
        if (run / "session.json").is_file():
            runs += recording_runs(run)
        elif run.is_dir() and any(run.glob("episode-*.ready")):
            raise SystemExit(f"{run}: saved episodes have no session.json")
    return [r for r in runs if any(r.glob("episode-*.ready"))]


def frame_rows(episode: Path):
    with (episode / "samples.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if "frame_index" in row:
                yield row


def read_episode(episode: Path, fps: int, n_actions: int, joints: list[int], tolerance=0.03):
    """Validated per-frame state/action arrays and frame geometry of one take."""
    summary = json.loads((episode / "episode.json").read_text())
    if summary.get("depth"):
        raise SystemExit(f"{episode}: depth takes are not handled here (use the board exporter)")
    state, action, stamps, ages, shape = [], [], [], [], None
    for i, row in enumerate(frame_rows(episode)):
        s = row["status"]
        if row["frame_index"] != i:
            raise SystemExit(f"{episode}: non-contiguous frame indices")
        if not (s["fk_valid"] and s["imu_healthy"] and s["state"] == "armed"):
            raise SystemExit(f"{episode}: frame {i}: invalid FK/controller state")
        a = s["applied_action"]
        if len(a) != 6 or not all(math.isfinite(x) and abs(x) <= 1 for x in a):
            raise SystemExit(f"{episode}: frame {i}: invalid applied action")
        shapes = []
        for key in CAMERAS:
            m = row[key]
            if m["stride"] != m["width"] or m["pixel_bytes"] != m["width"] * m["height"]:
                raise SystemExit(f"{episode}: invalid packed Y8 geometry")
            shapes.append((m["height"], m["width"]))
        if stamps and any(row[k]["sequence"] <= stamps[-1][k][0] or row[k]["host_monotonic_ns"] <= stamps[-1][k][1]
                          for k in CAMERAS):
            raise SystemExit(f"{episode}: duplicate/restarted camera stream")
        if shape and shapes != shape:
            raise SystemExit(f"{episode}: camera resolution changed")
        shape = shapes
        state.append([s["angles_rad"][j] for j in joints])
        action.append(a[:n_actions])
        stamps.append({k: (row[k]["sequence"], row[k]["host_monotonic_ns"]) for k in CAMERAS})
        t = row["sample_ns"]
        ages.append([t - row[k]["host_monotonic_ns"] for k in CAMERAS]
                    + [t - s["imu_host_monotonic_ns"], t - s["applied_host_monotonic_ns"]])
    n = len(stamps)
    if n < 2 or n != summary["frames"]:
        raise SystemExit(f"{episode}: {n} frame rows, episode.json says {summary['frames']}")
    for key, (h, w) in zip(CAMERAS, shape):
        if (episode / f"{key}.y8").stat().st_size != n * h * w:
            raise SystemExit(f"{episode}: truncated/extra pixels in {key}.y8")
    t = np.array([s["cam1"][1] for s in stamps], dtype=np.int64)
    hz = (n - 1) / ((t[-1] - t[0]) / 1e9)
    gap = np.diff(t).max() / 1e9
    if abs(hz / fps - 1) > tolerance or gap > 1.5 / fps:
        raise SystemExit(f"{episode}: {hz:.3f} Hz for declared {fps} FPS, max gap {gap * 1e3:.1f} ms")
    return {
        "state": np.rad2deg(np.asarray(state, dtype=np.float64)).astype(np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "ages": (np.asarray(ages, dtype=np.float64) / 1e9).astype(np.float32),
        "shapes": shape,
        "frames": n,
        "hz": hz,
        "episode_number": summary.get("episode_number"),
    }


# ---------------------------------------------------------------- stats and parquet


def numeric_stats(x: np.ndarray) -> dict:
    x = x.reshape(len(x), -1).astype(np.float64)
    out = {"min": x.min(0), "max": x.max(0), "mean": x.mean(0), "std": x.std(0), "count": np.array([len(x)])}
    out.update({f"q{round(q * 100):02d}": np.quantile(x, q, axis=0) for q in QUANTILES})
    return {k: v.tolist() for k, v in out.items()}


def image_stats(y8: Path, n: int, h: int, w: int) -> dict:
    """LeRobot-shaped (3,1,1) stats in [0, 1] from up to 100 frames at 1/4 resolution."""
    frames = np.memmap(y8, np.uint8, "r", shape=(n, h, w))
    idx = np.unique(np.linspace(0, n - 1, min(n, 100)).round().astype(int))
    px = (frames[idx, ::4, ::4].astype(np.float64) / 255.0).reshape(-1)
    vals = {"min": px.min(), "max": px.max(), "mean": px.mean(), "std": px.std()}
    vals.update({f"q{round(q * 100):02d}": np.quantile(px, q) for q in QUANTILES})
    out = {k: [[[float(v)]]] * 3 for k, v in vals.items()}
    out["count"] = [int(px.size)]
    return out


def hf_feature(dtype: str, length: int | None) -> dict:
    value = {"dtype": dtype, "_type": "Value"}
    # datasets 3.x `Sequence` spelling: openpi's lock cannot read datasets 4's `List` (see to_v21).
    return {"feature": value, "length": length, "_type": "Sequence"} if length else value


def write_parquet(path: Path, cols: dict[str, np.ndarray]) -> None:
    arrays, fields, hf = [], [], {}
    for name, x in cols.items():
        if x.ndim == 2:
            typ = pa.from_numpy_dtype(x.dtype)
            arrays.append(pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1), typ), x.shape[1]))
            fields.append(pa.field(name, pa.list_(typ, x.shape[1])))
            hf[name] = hf_feature(str(x.dtype), x.shape[1])
        else:
            arrays.append(pa.array(x))
            fields.append(pa.field(name, pa.from_numpy_dtype(x.dtype)))
            hf[name] = hf_feature(str(x.dtype), None)
    meta = {b"huggingface": json.dumps({"info": {"features": hf}}).encode()}
    pq.write_table(pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=meta)), path)


# ---------------------------------------------------------------- video


def encode_video(y8: Path, out: Path, n: int, h: int, w: int, fps: int, crf: int, lossless: bool) -> dict:
    """Encode Y8 frames as LeRobot's recorder does, decode them back, return the error vs raw."""
    import av

    frames = np.memmap(y8, np.uint8, "r", shape=(n, h, w))
    out.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(out), "w") as c:
        if lossless:
            s = c.add_stream("libx264rgb", rate=fps)
            s.pix_fmt, s.options = "rgb24", {"qp": "0", "preset": "ultrafast", "g": str(GOP)}
        else:
            s = c.add_stream("libx264", rate=fps)
            s.pix_fmt, s.options = "yuv420p", {"crf": str(crf), "g": str(GOP)}
        s.width, s.height = w, h
        for i in range(n):
            frame = av.VideoFrame.from_ndarray(np.repeat(frames[i][..., None], 3, axis=2), format="rgb24")
            frame.pts = i
            for p in s.encode(frame):
                c.mux(p)
        for p in s.encode():
            c.mux(p)

    abs_sum, sq_sum, worst, count, tb_ok = 0.0, 0.0, 0, 0, True
    with av.open(str(out)) as c:
        st = c.streams.video[0]
        for i, fr in enumerate(c.decode(st)):
            if i >= n:
                count += 1
                break
            tb_ok &= fr.pts is not None and round(float(fr.pts * st.time_base) * fps) == i
            rgb = fr.to_ndarray(format="rgb24")
            if rgb.shape != (h, w, 3) or not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])):
                raise SystemExit(f"{out}: frame {i} is not gray {rgb.shape}")
            d = rgb[..., 0].astype(np.int16) - frames[i]
            abs_sum += np.abs(d).sum()
            sq_sum += (d.astype(np.int64) ** 2).sum()
            worst = max(worst, int(np.abs(d).max()))
            count += 1
    if count != n or not tb_ok:
        raise SystemExit(f"{out}: decoded {count} frames for {n}, timestamps ok={tb_ok}")
    if lossless and worst:
        raise SystemExit(f"{out}: lossless video differs from raw by {worst}")
    mse = sq_sum / (n * h * w)
    return {"mean_abs": abs_sum / (n * h * w), "max_abs": worst,
            "psnr_db": float("inf") if mse == 0 else 10 * math.log10(255**2 / mse),
            "bytes": out.stat().st_size}


# ---------------------------------------------------------------- one episode


def write_episode(job: dict) -> dict:
    ep, dst, index = Path(job["episode"]), Path(job["dst"]), job["index"]
    data = read_episode(ep, job["fps"], job["n_actions"], job["joints"])
    n, (h, w) = data["frames"], data["shapes"][0]
    if data["shapes"] != [tuple(job["shape"])] * len(CAMERAS):
        raise SystemExit(f"{ep}: camera shape {data['shapes']}, dataset is {job['shape']}")
    chunk = index // CHUNK
    frame_index = np.arange(n, dtype=np.int64)
    cols = {
        "observation.state": data["state"],
        "action": data["action"],
        "timestamp": (frame_index / job["fps"]).astype(np.float32),
        "frame_index": frame_index,
        "episode_index": np.full(n, index, np.int64),
        "index": job["first_index"] + frame_index,
        "task_index": np.full(n, job["task_index"], np.int64),
    }
    cols.update({name: data["ages"][:, i] for i, name in enumerate(AGE_COLUMNS)})
    path = dst / f"data/chunk-{chunk:03d}/episode_{index:06d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(path, cols)
    stats = {k: numeric_stats(v) for k, v in cols.items()}
    video = {}
    for key in CAMERAS:
        vkey = f"observation.images.{key}"
        out = dst / f"videos/chunk-{chunk:03d}/{vkey}/episode_{index:06d}.mp4"
        video[key] = encode_video(ep / f"{key}.y8", out, n, h, w, job["fps"], job["crf"], job["lossless"])
        stats[vkey] = image_stats(ep / f"{key}.y8", n, h, w)
    sha = hashlib.sha256((ep / "samples.jsonl").read_bytes()).hexdigest()
    return {"index": index, "length": n, "stats": stats,
            "ages": data["ages"],
            "source": {"episode_index": index, "source": str(ep), "run": job["run"],
                       "episode_number": data["episode_number"], "frames": n, "achieved_hz": data["hz"],
                       "samples_sha256": sha, "video": video}}


# ---------------------------------------------------------------- dataset


def features(shape, joints: list[str], n_actions: int, fps: int, crf: int, lossless: bool) -> dict:
    h, w = shape
    vec = lambda n, names: {"dtype": "float32", "shape": [n], "names": names}  # noqa: E731
    scalar = lambda dt: {"dtype": dt, "shape": [1], "names": None}  # noqa: E731
    video_info = {"video.height": h, "video.width": w,
                  "video.codec": "h264", "video.pix_fmt": "gbrp" if lossless else "yuv420p",
                  "video.fps": fps, "video.channels": 3, "has_audio": False, "video.g": GOP,
                  "video.crf": None if lossless else crf, "video.lossless": lossless, "is_depth_map": False}
    feats = {"observation.state": vec(len(joints), joints),
             "action": vec(n_actions, list(ACTIONS[:n_actions]))}
    for key in CAMERAS:
        feats[f"observation.images.{key}"] = {"dtype": "video", "shape": [h, w, 3],
                                              "names": ["height", "width", "channels"], "info": video_info}
    feats.update(timestamp=scalar("float32"), frame_index=scalar("int64"), episode_index=scalar("int64"),
                 index=scalar("int64"), task_index=scalar("int64"))
    feats.update({name: scalar("float32") for name in AGE_COLUMNS})
    return feats


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("sessions", type=Path, nargs="+", help="native session dirs, in episode order")
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--state-joints", default="lift,tilt,scoop")
    p.add_argument("--action-dims", type=int, default=4, help="4 valves, 6 with tracks")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--lossless", action="store_true", help="bit-exact libx264rgb (~125%% of Y8)")
    p.add_argument("--robot-type", default="masi_excavator_native")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--v3", action="store_true", help="also write DST_v3 (LeRobot v3.0 copy)")
    a = p.parse_args()
    joints = a.state_joints.split(",")
    if len(set(joints)) != len(joints) or any(j not in JOINTS for j in joints):
        p.error("--state-joints must be a unique subset of slew,lift,tilt,scoop")
    if a.dst.exists():
        raise SystemExit(f"{a.dst} exists; choose a new destination")

    runs = [r for s in a.sessions for r in recording_runs(s.resolve())]
    sessions = [json.loads((r / "session.json").read_text()) for r in runs]
    fps = sessions[0]["fps"]
    for run, s in zip(runs, sessions):
        if s["format"] != FORMAT:
            raise SystemExit(f"{run}: format {s['format']}")
        if s["fps"] != fps:
            raise SystemExit(f"{run}: fps {s['fps']} != {fps}")
        if a.action_dims > (6 if s["enable_tracks"] else 4):
            raise SystemExit(f"{run}: --action-dims {a.action_dims} but tracks are off")
    tasks = list(dict.fromkeys(s["task"] for s in sessions))

    jobs, total, shape = [], 0, None
    for r, (run, s) in enumerate(zip(runs, sessions)):
        for ep in sorted(run.glob("episode-*.ready"), key=lambda e: int(e.name.split("-")[1].split(".")[0])):
            rows = frame_rows(ep)
            first = next(rows)
            shp = (first["cam1"]["height"], first["cam1"]["width"])
            shape = shape or shp
            n = json.loads((ep / "episode.json").read_text())["frames"]
            jobs.append({"episode": str(ep), "dst": str(a.dst), "index": len(jobs), "first_index": total,
                         "fps": fps, "n_actions": a.action_dims, "joints": [JOINTS[j] for j in joints],
                         "shape": shape, "task_index": tasks.index(s["task"]), "run": r,
                         "crf": a.crf, "lossless": a.lossless})
            total += n
    print(f"{len(runs)} runs, {len(jobs)} episodes, {total} frames, {shape[1]}x{shape[0]} @ {fps} FPS, tasks {tasks}")

    a.dst.mkdir(parents=True)
    results = []
    with cf.ProcessPoolExecutor(a.jobs) as pool:
        for res in pool.map(write_episode, jobs):
            v = res["source"]["video"]
            print(f"episode {res['index']:3d}: {res['length']:5d} frames, "
                  + ", ".join(f"{k} {x['mean_abs']:.2f}/{x['max_abs']} ({x['bytes'] / 2**20:.1f} MB)" for k, x in v.items()),
                  flush=True)
            results.append(res)
    if sum(r["length"] for r in results) != total:
        raise SystemExit("frame total changed while converting (episode.json vs samples)")

    meta = a.dst / "meta"
    meta.mkdir()
    info = {"codebase_version": "v2.1", "robot_type": a.robot_type, "fps": fps,
            "total_episodes": len(results), "total_frames": total, "total_tasks": len(tasks),
            "total_videos": len(results) * len(CAMERAS), "total_chunks": (len(results) - 1) // CHUNK + 1,
            "chunks_size": CHUNK, "splits": {"train": f"0:{len(results)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features(shape, joints, a.action_dims, fps, a.crf, a.lossless)}
    (meta / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    with (meta / "tasks.jsonl").open("w") as f:
        for i, t in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")
    with (meta / "episodes.jsonl").open("w") as f, (meta / "episodes_stats.jsonl").open("w") as g:
        for r, job in zip(results, jobs):
            f.write(json.dumps({"episode_index": r["index"], "tasks": [tasks[job["task_index"]]], "length": r["length"]}) + "\n")
            g.write(json.dumps({"episode_index": r["index"], "stats": r["stats"]}) + "\n")
    timing = []
    for i, run in enumerate(runs):
        ages = np.concatenate([r["ages"] for r, j in zip(results, jobs) if j["run"] == i]) * 1e3
        timing.append({"run": i, **{name.removeprefix("clock.").replace("_s", "_ms"):
                       {"p50": round(float(np.median(ages[:, c])), 2), "p99": round(float(np.percentile(ages[:, c], 99)), 2),
                        "max": round(float(ages[:, c].max()), 2)} for c, name in enumerate(AGE_COLUMNS)}})
        print("timing", json.dumps(timing[-1]))
    video_err = [x["mean_abs"] for r in results for x in r["source"]["video"].values()]
    (meta / "native_source.json").write_text(json.dumps({
        "format": FORMAT, "tool": "vla_common.dataset.from_native",
        "runs": [{"run": i, "path": str(run), "task": s["task"], "revision": s.get("revision")}
                 for i, (run, s) in enumerate(zip(runs, sessions))],
        "timing": timing,
        "state": {"joints": joints, "unit": "deg", "source": "status.angles_rad"},
        "action": {"names": list(ACTIONS[:a.action_dims]), "source": "status.applied_action"},
        "video": {"lossless": a.lossless, "crf": None if a.lossless else a.crf,
                  "mean_abs_error_max": max(video_err), "mean_abs_error_mean": float(np.mean(video_err))},
        "episodes": [r["source"] for r in results]}, indent=1) + "\n")
    for i, run in enumerate(runs):
        target = a.dst / "native" / f"session-{i:03d}"
        target.mkdir(parents=True)
        shutil.copy2(run / "session.json", target / "session.json")
        if (run / "provenance").is_dir():
            shutil.copytree(run / "provenance", target / "provenance")
    size = sum(f.stat().st_size for f in a.dst.rglob("*") if f.is_file())
    print(f"wrote {a.dst}: {len(results)} episodes, {total} frames, {size / 2**30:.2f} GB, "
          f"video mean |err| {np.mean(video_err):.2f} (worst episode {max(video_err):.2f})")
    if a.v3:
        from vla_common.dataset.to_v3 import to_v3
        to_v3(a.dst)


if __name__ == "__main__":
    main()
