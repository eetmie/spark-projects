#!/usr/bin/env python
"""Open raw kaivuri native takes in rerun: everything the recorder logged, not just what trains.

    python -m vla_common.dataset.rerun_native SESSION [EP ...]     # EP = dataset episode index
    python -m vla_common.dataset.rerun_native SESSION --all
    python -m vla_common.dataset.rerun_native path/to/episode-*.ready

Episodes are numbered as `from_native` numbers them (session, then recordings/run-* resumes),
so EP 37 here is episode 37 of the converted dataset.

Per take:
    cam1, cam2            raw Y8, lossless, at each frame row's time
    status/...            every field that changes, at the full sample rate (~100 Hz)
    cam1/meta, cam2/meta  exposure, gain, ...
    timing/...            ms: input ages vs the sample (what from_native's clock.* columns hold),
                          frame/sample intervals, cam1-cam2 skew, dropped frames
    arm                   FK joint chain + bucket tip, base frame, with the tip's trail
    constants             fields that never change in the take (text panel)

Needs rerun-sdk 0.26 + numpy (the lerobot 0.5.1 venv has both, and its viewer binary).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import rerun as rr

from vla_common.dataset.from_native import ACTIONS, CAMERAS, recording_runs

# counters, absolute clocks and geometry: used for timing/ below, not plotted raw
SKIP = {"sample_ns", "episode_ns", "host_monotonic_ns", "imu_host_monotonic_ns", "applied_host_monotonic_ns",
        "imu_device_us", "sensor_timestamp_us", "frame_timestamp_us", "sdk_timestamp_ms", "sequence",
        "requested_sequence", "command_sequence", "applied_command_sequence", "imu_sequence", "ticks",
        "frame_counter", "frame_index"}


def episodes_of(src: Path) -> list[tuple[Path, str]]:
    if src.name.endswith(".ready"):
        task = json.loads((src.parent / "session.json").read_text()).get("task", "") \
            if (src.parent / "session.json").is_file() else ""
        return [(src, task)]
    out = []
    for run in recording_runs(src.resolve()):
        task = json.loads((run / "session.json").read_text())["task"]
        eps = sorted(run.glob("episode-*.ready"), key=lambda e: int(e.name.split("-")[1].split(".")[0]))
        out += [(e, task) for e in eps]
    return out


def names(key: str, n: int, joints: list[str]) -> list[str]:
    if key.endswith("wxyz"):
        return list("wxyz")
    if key.endswith("xz_m"):
        return ["x", "z"]
    if "xyz" in key or (n == 3 and key.startswith("imu_")):
        return list("xyz")
    if key.endswith("_action"):
        return list(ACTIONS[:n])
    if n == len(joints):
        return joints
    return [str(i) for i in range(n)]


def flatten(prefix: str, key: str, v, joints, out: dict):
    if isinstance(v, dict):
        for k, x in v.items():
            if k not in SKIP:
                flatten(f"{prefix}/{k}", k, x, joints, out)
    elif isinstance(v, list):
        if v and isinstance(v[0], list):
            for name, x in zip(names("", len(v), joints), v):
                for sub, y in zip(names(key, len(x), joints), x):
                    out[f"{prefix}/{name}/{sub}"] = y
        elif v and not isinstance(v[0], str):
            for sub, y in zip(names(key, len(v), joints), v):
                out[f"{prefix}/{sub}"] = y
        else:
            out[prefix] = json.dumps(v)
    else:
        out[prefix] = v


def log_episode(rec: rr.RecordingStream, ep: Path, task: str):
    rows = [json.loads(l) for l in (ep / "samples.jsonl").open()]
    summary = json.loads((ep / "episode.json").read_text())
    joints = rows[0]["status"].get("joint_names", ["slew", "boom", "arm", "bucket"])
    t = np.array([r["episode_ns"] for r in rows], dtype=np.int64) / 1e9
    frames = [i for i, r in enumerate(rows) if "frame_index" in r]
    tf = t[frames]

    # every leaf of status / requested_action / teleop_lateness, per sample
    flat = []
    for r in rows:
        d = {}
        flatten("status", "status", r["status"], joints, d)
        flatten("requested_action", "requested_action", r["requested_action"], joints, d)
        d["teleop_lateness_ms"] = r["teleop_lateness_ms"]
        flat.append(d)
    keys = list(dict.fromkeys(k for d in flat for k in d))
    constants, changing = {}, []
    for k in keys:
        vals = [d.get(k) for d in flat]
        if all(v == vals[0] for v in vals):
            constants[k] = vals[0]
        elif all(isinstance(v, (int, float)) for v in vals):  # bools are ints
            rec.send_columns(k, indexes=[rr.TimeColumn("time", duration=t)],
                             columns=rr.Scalars.columns(scalars=np.asarray(vals, dtype=np.float64)))
            changing.append(k)
        else:  # strings that change (state, reason): log each change
            last = object()
            for ti, v in zip(t, vals):
                if v != last:
                    rec.set_time("time", duration=float(ti))
                    rec.log(k, rr.TextLog(f"{k} = {v}"))
                    last = v

    # camera metadata + timing
    sample_ns = np.array([r["sample_ns"] for r in rows], dtype=np.int64)
    ms = lambda x: np.asarray(x, dtype=np.float64) / 1e6
    timing = {
        "timing/sample_interval_ms": (t[1:], np.diff(sample_ns) / 1e6),
        "timing/imu_age_ms": (t, ms(sample_ns - [r["status"]["imu_host_monotonic_ns"] for r in rows])),
        "timing/action_age_ms": (t, ms(sample_ns - [r["status"]["applied_host_monotonic_ns"] for r in rows])),
    }
    host = {}
    for cam in CAMERAS:
        meta = [rows[i][cam] for i in frames]
        host[cam] = np.array([m["host_monotonic_ns"] for m in meta], dtype=np.int64)
        fc = np.array([m["frame_counter"] for m in meta], dtype=np.int64)
        timing[f"timing/{cam}_age_ms"] = (tf, ms(sample_ns[frames] - host[cam]))
        timing[f"timing/{cam}_interval_ms"] = (tf[1:], np.diff(host[cam]) / 1e6)
        timing[f"timing/{cam}_dropped_frames"] = (tf[1:], np.diff(fc) - 1.0)
        for k in ("actual_exposure_us", "gain_level", "valid_metadata"):
            vals = np.array([m[k] for m in meta], dtype=np.float64)
            if (vals != vals[0]).any():
                rec.send_columns(f"{cam}/meta/{k}", indexes=[rr.TimeColumn("time", duration=tf)],
                                 columns=rr.Scalars.columns(scalars=vals))
            else:
                constants[f"{cam}/meta/{k}"] = vals[0].item()
    timing["timing/cam1_minus_cam2_ms"] = (tf, ms(host["cam1"] - host["cam2"]))
    for k, (tt, v) in timing.items():
        rec.send_columns(k, indexes=[rr.TimeColumn("time", duration=tt)], columns=rr.Scalars.columns(scalars=v))

    # raw Y8 frames, straight from disk
    for cam in CAMERAS:
        m = rows[frames[0]][cam]
        h, w = m["height"], m["width"]
        y8 = np.memmap(ep / f"{cam}.y8", dtype=np.uint8, mode="r").reshape(len(frames), h * w)
        rec.log(cam, rr.Image.from_fields(format=rr.components.ImageFormat(
            width=w, height=h, color_model="L", channel_datatype="U8")), static=True)
        rec.send_columns(cam, indexes=[rr.TimeColumn("time", duration=tf)], columns=rr.Image.columns(buffer=y8))

    # arm from FK (base frame), at frame rate
    rec.log("arm", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    tips = np.array([rows[i]["status"]["fk_xyz_m"] for i in frames])
    rec.log("arm/tip_trail", rr.Points3D(tips, radii=0.002, colors=[120, 120, 120]), static=True)
    for i, tip in zip(frames, tips):
        s = rows[i]["status"]
        chain = np.vstack([[0, 0, 0], s["fk_joint_xyz_m"], tip])
        rec.set_time("time", duration=float(t[i]))
        rec.log("arm/links", rr.LineStrips3D([chain], radii=0.006, colors=[230, 140, 30]))
        rec.log("arm/joints", rr.Points3D(chain[1:-1], radii=0.012, labels=joints))
        rec.log("arm/tip", rr.Points3D([tip], radii=0.015, colors=[220, 40, 40]))

    rec.log("constants", rr.TextDocument(
        f"task: {task}\nsource: {ep}\nepisode.json: {json.dumps(summary)}\n\n"
        + "\n".join(f"{k}: {v}" for k, v in constants.items())), static=True)
    rec.flush()
    hz = (len(frames) - 1) / (tf[-1] - tf[0])
    print(f"{ep.name}: {len(rows)} samples, {len(frames)} frames @ {hz:.2f} Hz, "
          f"{len(changing)} changing fields, {len(constants)} constant — {task}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src", type=Path, help="native session dir, or one episode-*.ready dir")
    p.add_argument("episodes", nargs="*", type=int, help="dataset episode indices (default 0)")
    p.add_argument("--all", action="store_true")
    a = p.parse_args()

    os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ['PATH']}"  # rerun viewer binary
    eps = episodes_of(a.src)
    chosen = range(len(eps)) if a.all else (a.episodes or [0])
    for n, i in enumerate(chosen):
        ep, task = eps[i]
        rec = rr.RecordingStream("kaivuri_native", recording_id=f"{a.src.name}/{ep.name}")
        rec.spawn() if n == 0 else rec.connect_grpc()
        rec.send_recording_name(f"ep {i:03d} · {task}")
        log_episode(rec, ep, task)


if __name__ == "__main__":
    main()
