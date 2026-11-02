"""Audit a LeRobot dataset (v2.1 or v3.0) for recording faults before anything trains on it.

    python -m vla_common.dataset.audit DATASET [DATASET ...] [--no-video] [--workers N]
    python -m vla_common.dataset.audit --require DATASET      # the training gate (exit 2 = refuse)

Why this exists: on the weekend of 2026-10-31 two large fine-tunes (pi0.5, X-VLA) drove badly and
a weekend went into the models. The cause was hardware. A loose IMU connector froze the lift
reading at -45.06 deg for 74 of native run 4's 107 episodes (and all of sand-depth-002), with fresh
timestamps, so nothing downstream noticed. Every conversion (from_native, to_v21, to_v3, merge,
subset_v21) now ends with this audit, and the trainers refuse a dataset that failed it or was
never audited.

Writes `meta/audit.json` (status, per-episode issues, the bad-episode list, a fingerprint of the
data it saw) and prints a report. Status per episode and for the dataset: PASS, WARN, FAIL.

FAIL (do not train on the episode):
  frozen joint     a state value identical for >= 1 s while its joint is commanded, or one value
                   for the whole episode while commanded. -45.06 deg lift is the known IMU fault.
  bad numbers      non-finite state/action, or an action outside [-1, 1]
  timestamp gap    a step longer than 1.5 frames
  video mismatch   the episode's video has more than 1 frame more/less than its rows
  camera stall     >= 10 near-identical consecutive frames (a frozen camera, ~0.33 s)
WARN (look at it):
  stale camera     a stored camera age over 1.5 frame periods (clock.*age* columns)
  repeated frames  3-9 near-identical consecutive frames
  dark frames      frames with mean brightness < 3 (lens cap, exposure, dead IR emitter)
  short episode    under 3 s (often a session cut at an arbitrary point)
  idle edges       over 1 s of zero action at the start or end
  weak response    a joint whose velocity barely follows its command (corr < 0.3 over >= 2 s
                   commanded): a dying sensor, a swapped joint, or a mapping error
  video offset     image motion vs the slew command lags > 2 frames off the dataset median
                   (the camera stream and the data drifted apart)

The thresholds are module constants. What they rest on (2026-11-02 measurements): healthy IMU
angles never repeat exactly while moving (3000+ distinct values per episode); stored camera ages
stay under one frame (~30 ms at 30 fps); cabin-image motion follows the slew command 7-8 frames
later with corr ~0.95 in every healthy recording; isolated near-identical frames (~1 %) are still
scenes, not stalls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 1
FROZEN_S = 1.0             # identical joint value while commanded, seconds
COMMANDED = 0.2            # |action| above this counts as commanding the joint
KNOWN_STUCK_LIFT_DEG = -45.06
GAP_FRAMES = 1.5
STALE_FRAMES = 1.5
REPEAT_DIFF = 0.15         # mean |diff| (0-255) at 160x120 below which a frame repeats the last
REPEAT_WARN, REPEAT_FAIL = 3, 10
DARK_MEAN = 3.0
SHORT_S = 3.0
IDLE_S = 1.0
IDLE_ACTION = 0.05
WEAK_CORR, WEAK_MIN_S = 0.3, 2.0
OFFSET_FRAMES = 2
MAX_LAG = 15

RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


# --- loading --------------------------------------------------------------------------------

def _parquets(d: Path) -> list[Path]:
    return sorted(d.rglob("*.parquet"))


def load(root: Path):
    """info, a per-episode metadata frame, the data rows, and the video keys."""
    info = json.loads((root / "meta" / "info.json").read_text())
    feats = info["features"]
    want = [c for c in ("episode_index", "frame_index", "timestamp", "observation.state", "action") if c in feats]
    want += [k for k in feats if k.startswith("clock.") and "age" in k]
    data = pd.concat([pd.read_parquet(f, columns=want) for f in _parquets(root / "data")], ignore_index=True)
    if info["codebase_version"].startswith("v3"):
        eps = pd.concat([pd.read_parquet(f) for f in _parquets(root / "meta" / "episodes")], ignore_index=True)
    else:
        eps = pd.DataFrame([json.loads(l) for l in (root / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()])
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    videos = [k for k, v in feats.items() if v.get("dtype") == "video"]
    return info, eps, data, videos


def fingerprint(root: Path) -> str:
    """Names and sizes of the data files and the episode table: cheap, and it changes with the data."""
    h = hashlib.sha256()
    for f in _parquets(root / "data") + [root / "meta" / "info.json"]:
        h.update(f"{f.relative_to(root)}:{f.stat().st_size}".encode())
    return h.hexdigest()[:16]


def video_jobs(root: Path, info: dict, eps: pd.DataFrame, videos: list[str]) -> list[dict]:
    jobs = []
    v3 = info["codebase_version"].startswith("v3")
    for _, r in eps.iterrows():
        e = int(r["episode_index"])
        for key in videos:
            if v3:
                path = root / info["video_path"].format(video_key=key, chunk_index=int(r[f"videos/{key}/chunk_index"]),
                                                        file_index=int(r[f"videos/{key}/file_index"]))
                t0, t1 = float(r[f"videos/{key}/from_timestamp"]), float(r[f"videos/{key}/to_timestamp"])
            else:
                path = root / info["video_path"].format(episode_chunk=e // int(info.get("chunks_size", 1000)),
                                                        video_key=key, episode_index=e)
                t0, t1 = 0.0, None
            jobs.append({"episode": e, "key": key, "path": str(path), "t0": t0, "t1": t1})
    return jobs


def decode(job: dict) -> dict:
    """One episode of one camera at 160x120 gray: frame count, repeats, darkness, motion series."""
    cmd = ["ffmpeg", "-loglevel", "error", "-nostdin"]
    if job["t0"]:
        cmd += ["-ss", f"{job['t0']:.4f}"]
    cmd += ["-i", job["path"]]
    if job["t1"] is not None:
        cmd += ["-t", f"{job['t1'] - job['t0']:.4f}"]
    cmd += ["-vf", "scale=160:120,format=gray", "-f", "rawvideo", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return {**job, "error": (p.stderr.decode(errors="replace").strip().splitlines() or ["no frames"])[-1]}
    fr = np.frombuffer(p.stdout, np.uint8).reshape(-1, 120, 160).astype(np.int16)
    motion = np.abs(np.diff(fr, axis=0)).mean((1, 2)) if len(fr) > 1 else np.zeros(0)
    rep = motion < REPEAT_DIFF
    longest = cur = 0
    for x in rep:
        cur = cur + 1 if x else 0
        longest = max(longest, cur)
    return {**job, "frames": int(len(fr)), "repeats": int(rep.sum()), "longest_repeat": int(longest + (longest > 0)),
            "dark": int((fr.mean((1, 2)) < DARK_MEAN).sum()), "motion": motion.astype(np.float32)}


# --- checks ---------------------------------------------------------------------------------

def best_lag(cmd: np.ndarray, resp: np.ndarray) -> tuple[int, float]:
    """Lag (frames, resp after cmd) with the highest correlation, and that correlation."""
    best, bc = 0, -1.0
    for lag in range(0, MAX_LAG + 1):
        a, b = cmd[: len(cmd) - lag], resp[lag:]
        if len(a) < 30 or a.std() < 1e-6 or b.std() < 1e-6:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if c > bc:
            best, bc = lag, c
    return best, bc


def longest_frozen(x: np.ndarray, commanded: np.ndarray) -> int:
    same = np.r_[False, x[1:] == x[:-1]] & commanded
    best = cur = 0
    for s in same:
        cur = cur + 1 if s else 0
        best = max(best, cur)
    return best


class Episode:
    def __init__(self, index: int):
        self.index, self.issues = index, []

    def add(self, level: str, what: str):
        self.issues.append({"level": level, "issue": what})

    @property
    def status(self) -> str:
        return max((i["level"] for i in self.issues), key=RANK.get, default="PASS")


def audit(root: Path, video: bool = True, write: bool = True, workers: int | None = None, quiet: bool = False) -> dict:
    root = Path(root).expanduser().resolve()
    t_start = time.time()
    info, eps, data, videos = load(root)
    fps = float(info["fps"])
    feats = info["features"]
    snames = list((feats.get("observation.state") or {}).get("names") or [])
    anames = list((feats.get("action") or {}).get("names") or [])
    joint_cmd = {j: anames.index(n) for j, n in enumerate(snames) if n in anames}  # state joint -> action column
    sdim = ((feats.get("observation.state") or {}).get("shape") or [0])[0]
    adim = ((feats.get("action") or {}).get("shape") or [0])[0]
    if not joint_cmd and sdim == 3 and adim >= 4:
        # Unnamed excavator layout (GR00T views): state [lift, tilt, scoop], action [slew, lift, tilt, scoop, ...].
        snames, anames = ["lift", "tilt", "scoop"], ["slew", "lift", "tilt", "scoop"] + [f"a{i}" for i in range(4, adim)]
        joint_cmd = {0: 1, 1: 2, 2: 3}
    slew = anames.index("slew") if "slew" in anames else None

    S = np.stack(data["observation.state"].to_numpy()).astype(np.float64) if "observation.state" in data else None
    A = np.stack(data["action"].to_numpy()).astype(np.float64) if "action" in data else None
    ep_of = data["episode_index"].to_numpy()
    order = np.argsort(ep_of, kind="stable")
    bounds, start = {}, 0
    sorted_eps = ep_of[order]
    for e in np.unique(sorted_eps):
        n = int((sorted_eps == e).sum())
        bounds[int(e)] = order[start:start + n]
        start += n

    out = {int(e): Episode(int(e)) for e in eps["episode_index"]}
    lags = {j: [] for j in joint_cmd}
    per_ep_slew = {}
    for e, rows in bounds.items():
        ep = out.setdefault(e, Episode(e))
        n = len(rows)
        if "frame_index" in data:
            rows = rows[np.argsort(data["frame_index"].to_numpy()[rows], kind="stable")]
        if n / fps < SHORT_S:
            ep.add("WARN", f"short episode ({n / fps:.1f} s)")
        ts = data["timestamp"].to_numpy()[rows].astype(np.float64)
        if n > 1:
            dt = np.diff(ts)
            gaps = int((dt > GAP_FRAMES / fps).sum()) + int((dt <= 0).sum())
            if gaps:
                ep.add("FAIL", f"timestamp gap/backstep x{gaps} (largest step {dt.max() * fps:.1f} frames)")
        for c in [c for c in data.columns if c.startswith("clock.") and "age" in c]:
            age = data[c].to_numpy()[rows].astype(np.float64)
            age = age[np.isfinite(age)]
            if len(age) and age.max() > STALE_FRAMES / fps:
                ep.add("WARN", f"stale {c}: max {age.max() * 1000:.0f} ms, {int((age > STALE_FRAMES / fps).sum())} rows over {STALE_FRAMES} frames")
        if S is not None:
            s = S[rows]
            if not np.isfinite(s).all():
                ep.add("FAIL", "non-finite state")
        if A is not None:
            a = A[rows]
            if not np.isfinite(a).all():
                ep.add("FAIL", "non-finite action")
            elif np.abs(a).max() > 1.0001:
                ep.add("FAIL", f"action outside [-1, 1] (max |a| {np.abs(a).max():.3f})")
            act = np.abs(a[:, :4] if a.shape[1] >= 4 else a).max(1) > IDLE_ACTION
            idx = np.flatnonzero(act)
            lead = (idx[0] if len(idx) else n) / fps
            trail = (n - 1 - idx[-1] if len(idx) else n) / fps
            if lead > IDLE_S or trail > IDLE_S:
                ep.add("WARN", f"idle edges (start {lead:.1f} s, end {trail:.1f} s)")
        if S is not None and A is not None:
            for j, k in joint_cmd.items():
                x, cmd = S[rows, j], A[rows, k]
                commanded = np.abs(cmd) > COMMANDED
                name = snames[j]
                run = longest_frozen(x, commanded)
                whole = len(np.unique(x)) <= 2 and commanded.sum() / fps >= FROZEN_S
                if run / fps >= FROZEN_S or whole:
                    known = " (the known IMU fault value)" if name == "lift" and np.any(np.abs(x - KNOWN_STUCK_LIFT_DEG) < 0.05) else ""
                    what = "for the whole episode" if whole else f"for {run / fps:.1f} s"
                    ep.add("FAIL", f"frozen {name}: {x[0] if whole else np.median(x):.2f} unchanged while commanded {what}{known}")
                    continue
                if commanded.sum() / fps >= WEAK_MIN_S and n > 60:
                    lag, c = best_lag(cmd, np.gradient(x) * fps)
                    lags[j].append((lag, c))
                    if c < WEAK_CORR:
                        ep.add("WARN", f"weak response: {name} velocity barely follows its command (corr {c:.2f})")
        if slew is not None and A is not None:
            per_ep_slew[e] = np.abs(A[rows, slew])

    # Video: per episode and camera, in parallel.
    vid = {}
    if video and videos:
        jobs = video_jobs(root, info, eps, videos)
        with ProcessPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 2)) as pool:
            for r in pool.map(decode, jobs, chunksize=4):
                vid[(r["episode"], r["key"])] = r
        cam_corr = {k: [] for k in videos}
        for (e, key), r in vid.items():
            ep = out[e]
            cam = key.rsplit(".", 1)[-1]
            if "error" in r:
                ep.add("FAIL", f"{cam} video unreadable: {r['error']}")
                continue
            rows = len(bounds.get(e, []))
            if abs(r["frames"] - rows) > 1:
                ep.add("FAIL", f"{cam} video has {r['frames']} frames for {rows} rows")
            if r["longest_repeat"] >= REPEAT_FAIL:
                ep.add("FAIL", f"{cam} stalled: {r['longest_repeat']} identical frames in a row")
            elif r["longest_repeat"] >= REPEAT_WARN:
                ep.add("WARN", f"{cam}: {r['longest_repeat']} identical frames in a row")
            if r["dark"]:
                ep.add("WARN", f"{cam}: {r['dark']} dark frames")
            if e in per_ep_slew and per_ep_slew[e].std() > 0.05 and len(r["motion"]) > 60:
                m = r["motion"]
                k = min(len(m), len(per_ep_slew[e]) - 1)
                lag, c = best_lag(per_ep_slew[e][1:k + 1], m[:k])
                r["slew_lag"], r["slew_corr"] = lag, c
                cam_corr[key].append(c)
        # The camera whose image follows slew (the cabin view) carries the timing check.
        timed = max(cam_corr, key=lambda k: np.median(cam_corr[k]) if cam_corr[k] else -1, default=None)
        if timed and cam_corr[timed] and np.median(cam_corr[timed]) > 0.5:
            good = [r["slew_lag"] for (e, k), r in vid.items() if k == timed and r.get("slew_corr", 0) > 0.5]
            med = float(np.median(good)) if good else None
            for (e, k), r in vid.items():
                if k == timed and med is not None and r.get("slew_corr", 0) > 0.5 and abs(r["slew_lag"] - med) > OFFSET_FRAMES:
                    out[e].add("WARN", f"video offset: {k.rsplit('.', 1)[-1]} image motion {r['slew_lag']} frames after slew, dataset median {med:.0f}")
            video_timing = {"camera": timed, "median_lag_frames": med, "median_corr": float(np.median(cam_corr[timed]))}
        else:
            video_timing = {"camera": None, "note": "no camera follows the slew command (corr <= 0.5); timing not checked"}
    else:
        video_timing = {"note": "video not checked" if videos else "no video features"}

    statuses = {e: ep.status for e, ep in out.items()}
    bad = sorted(e for e, s in statuses.items() if s == "FAIL")
    warn = sorted(e for e, s in statuses.items() if s == "WARN")
    status = "FAIL" if bad else ("WARN" if warn else "PASS")
    counts = {}
    for ep in out.values():
        for i in ep.issues:
            key = (i["level"], i["issue"].split(":")[0].split(" (")[0])
            counts[key] = counts.get(key, 0) + 1
    report = {
        "version": VERSION,
        "tool": "vla_common.dataset.audit",
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": status,
        "video_checked": bool(video and videos),
        "fingerprint": fingerprint(root),
        "summary": {
            "episodes": len(out), "frames": int(len(data)), "fps": fps, "minutes": round(len(data) / fps / 60, 1),
            "fail_episodes": len(bad), "warn_episodes": len(warn),
            "fail_frames": int(sum(len(bounds.get(e, [])) for e in bad)),
            "issue_counts": {f"{lvl} {what}": n for (lvl, what), n in sorted(counts.items(), key=lambda x: (-RANK[x[0][0]], -x[1]))},
            "command_to_motion_lag_frames": {snames[j]: {"median": float(np.median([l for l, _ in v])), "median_corr": round(float(np.median([c for _, c in v])), 2)}
                                             for j, v in lags.items() if v},
            "video_timing": video_timing,
        },
        "bad_episodes": bad,
        "warn_episodes": warn,
        "episodes": {str(e): {"status": ep.status, "issues": ep.issues} for e, ep in sorted(out.items()) if ep.issues},
        "thresholds": {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float)) and k != "VERSION"},
        "seconds": round(time.time() - t_start, 1),
    }
    if write:
        (root / "meta" / "audit.json").write_text(json.dumps(report, indent=1) + "\n")
    if not quiet:
        print_report(root, report)
    return report


def ranges(xs: list[int]) -> str:
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(f"{xs[i]}" if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ", ".join(out)


def print_report(root: Path, r: dict) -> None:
    s = r["summary"]
    mark = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL  <-- do not train on the bad episodes"}[r["status"]]
    print(f"\n=== audit {root.name}: {mark}")
    print(f"    {s['episodes']} episodes, {s['frames']} frames, {s['minutes']} min @ {s['fps']:g} fps"
          f"{'' if r['video_checked'] else '   (video NOT checked)'}   [{r['seconds']} s]")
    if r["bad_episodes"]:
        print(f"    FAIL {len(r['bad_episodes'])} episodes ({s['fail_frames']} frames): {ranges(r['bad_episodes'])}")
    if r["warn_episodes"]:
        print(f"    WARN {len(r['warn_episodes'])} episodes: {ranges(r['warn_episodes'])}")
    for what, n in s["issue_counts"].items():
        print(f"      {n:4d} x {what}")
    lag = s["command_to_motion_lag_frames"]
    if lag:
        print("    command -> joint motion: " + ", ".join(f"{k} {v['median']:.0f} frames (corr {v['median_corr']})" for k, v in lag.items()))
    vt = s["video_timing"]
    if vt.get("camera"):
        print(f"    video timing: {vt['camera'].rsplit('.', 1)[-1]} motion follows slew by {vt['median_lag_frames']:.0f} frames (corr {vt['median_corr']:.2f})")
    elif vt.get("note"):
        print(f"    video timing: {vt['note']}")
    shown = 0
    for e, ep in r["episodes"].items():
        if ep["status"] != "FAIL" or shown >= 8:
            continue
        print(f"      episode {e}: " + "; ".join(i["issue"] for i in ep["issues"] if i["level"] == "FAIL"))
        shown += 1
    if len(r["bad_episodes"]) > shown:
        print(f"      ... all issues in {root / 'meta' / 'audit.json'}")


def audit_after(root: Path) -> dict | None:
    """The audit every conversion ends with. Never undoes the conversion; a FAIL is reported, and
    the trainers refuse the dataset until its bad episodes are dropped. VLA_SKIP_AUDIT=1 skips it."""
    if os.environ.get("VLA_SKIP_AUDIT") == "1":
        print(f"!! audit skipped (VLA_SKIP_AUDIT=1): {root} cannot be trained on until it is audited", file=sys.stderr)
        return None
    try:
        return audit(root)
    except Exception as e:  # the conversion is done; a broken audit must not lose it
        print(f"!! audit of {root} failed to run ({type(e).__name__}: {e}); run it by hand:\n"
              f"   python -m vla_common.dataset.audit {root}", file=sys.stderr)
        return None


# --- the training gate ----------------------------------------------------------------------

def find_report(root: Path) -> tuple[Path, Path] | None:
    """(dataset dir, audit.json) for ROOT or, for a view whose data/ is a symlink, its source."""
    root = Path(root).expanduser().resolve()
    for d in (root, (root / "data").resolve().parent):
        if (d / "meta" / "audit.json").exists():
            return d, d / "meta" / "audit.json"
    return None


def require(root: Path, allow_fail: bool = False) -> dict:
    """Refuse (SystemExit 2) unless ROOT has a current audit that did not FAIL.

    Override with allow_fail or VLA_ALLOW_UNAUDITED=1, for a deliberate exception only.
    """
    root = Path(root).expanduser().resolve()
    override = allow_fail or os.environ.get("VLA_ALLOW_UNAUDITED") == "1"
    found = find_report(root)
    why, r, current = None, {}, False
    if found is None:
        why = f"{root} has never been audited."
    else:
        d, path = found
        r = json.loads(path.read_text())
        current = r.get("fingerprint") == fingerprint(d)
        if not current:
            why = f"{d.name}'s audit ({r.get('audited_at')}) is stale: the data changed since."
        elif r["status"] == "FAIL":
            why = (f"{d.name} FAILED its audit ({r['audited_at']}): {len(r['bad_episodes'])} bad episodes "
                   f"({ranges(r['bad_episodes'])}). See {path}.")
    if why is None:
        print(f"audit ok: {found[0].name} {r['status']} ({r['audited_at']}"
              f"{'' if r.get('video_checked') else ', video not checked'})", file=sys.stderr)
        return r
    msg = (f"!! {why}\n   Run: python -m vla_common.dataset.audit {root}\n"
           f"   A FAIL means recording faults (frozen sensor, stalled camera, gaps). Train on a copy without\n"
           f"   the bad episodes, or set VLA_ALLOW_UNAUDITED=1 to train anyway (a deliberate exception).")
    if override:
        print(msg.replace("!!", "!! (overridden)"), file=sys.stderr)
        return r if current else {}
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="+", type=Path)
    p.add_argument("--no-video", action="store_true", help="skip decoding (structure, timing and sensors only)")
    p.add_argument("--workers", type=int, default=None, help="video decode processes (default min(8, CPUs))")
    p.add_argument("--no-write", action="store_true", help="do not write meta/audit.json")
    p.add_argument("--require", action="store_true", help="check an existing audit instead; exit 2 to refuse")
    p.add_argument("--allow-fail", action="store_true", help="with --require: warn instead of refusing")
    a = p.parse_args()
    if a.require:
        for d in a.datasets:
            require(d, a.allow_fail)
        return
    worst = "PASS"
    for d in a.datasets:
        r = audit(d, video=not a.no_video, write=not a.no_write, workers=a.workers)
        worst = max(worst, r["status"], key=RANK.get)
    sys.exit(2 if worst == "FAIL" else 0)


if __name__ == "__main__":
    main()
