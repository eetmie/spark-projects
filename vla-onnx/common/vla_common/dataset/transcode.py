#!/usr/bin/env python
"""Re-encode a LeRobot v3 dataset's videos as H.264, keeping every frame's timestamp.

    python -m vla_common.dataset.transcode --src DIR --dst DIR [--crf 18] [--jobs 4]

WHY. Newer LeRobot recorders write AV1. The torchcodec wheels here link the system FFmpeg,
which has no software AV1 decoder ("Could not push packet to decoder: Function not
implemented"), so training dies on its first batch. PyAV bundles libdav1d, so it can
decode them, and H.264 decodes in every loader the pipeline uses.

Writes a full copy: meta/ and data/ are copied unchanged, each video is re-encoded the way
from_native encodes (libx264, yuv420p, GOP 2), and info.json's video.codec/pix_fmt are
updated. The source is never touched. Lossy: the report gives PSNR against the source.

CHECKS, per file: the same frame count, every timestamp within 1e-4 s of the source
(LeRobot's default tolerance_s), and the source must be constant-rate at the dataset fps.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np

from vla_common.dataset.from_native import GOP

TOL_S = 1e-4


def transcode_file(src: Path, dst: Path, fps: int, crf: int, threads: int) -> dict:
    import av

    dst.parent.mkdir(parents=True, exist_ok=True)
    times = []
    with av.open(str(src)) as ic, av.open(str(dst), "w") as oc:
        ist = ic.streams.video[0]
        ist.thread_type = "AUTO"
        ost = oc.add_stream("libx264", rate=fps)
        ost.width, ost.height, ost.pix_fmt = ist.width, ist.height, "yuv420p"
        ost.options = {"crf": str(crf), "g": str(GOP), "threads": str(threads)}
        for fr in ic.decode(ist):
            t = float(fr.pts * ist.time_base)
            i = round(t * fps)
            if abs(i / fps - t) > TOL_S or (times and i <= round(times[-1] * fps)):
                raise SystemExit(f"{src}: frame at {t:.6f}s is not on the {fps} fps grid")
            times.append(t)
            out = fr.reformat(format="yuv420p")
            out.pts, out.time_base = i, Fraction(1, fps)  # reformat keeps the source's time_base
            for p in ost.encode(out):
                oc.mux(p)
        for p in ost.encode():
            oc.mux(p)

    # Decode both in lockstep (strict zip: same count), timestamps and luma error against the source.
    sq, n = 0.0, 0
    with av.open(str(src)) as a, av.open(str(dst)) as b:
        sa, sb = a.streams.video[0], b.streams.video[0]
        sa.thread_type = sb.thread_type = "AUTO"
        for fa, fb in zip(a.decode(sa), b.decode(sb), strict=True):
            tb = float(fb.pts * sb.time_base)
            if abs(tb - times[n]) > TOL_S:
                raise SystemExit(f"{dst}: frame {n} at {tb:.6f}s, source {times[n]:.6f}s")
            ya = fa.to_ndarray(format="gray").astype(np.int16)
            yb = fb.to_ndarray(format="gray").astype(np.int16)
            sq += float(((ya - yb) ** 2).mean())
            n += 1
    mse = sq / max(n, 1)
    return {"file": str(dst), "frames": n,
            "psnr_db": float("inf") if mse == 0 else 10 * math.log10(255**2 / mse)}


def transcode(src: Path, dst: Path, crf: int = 18, jobs: int = 4, force: bool = False) -> list[dict]:
    src, dst = Path(src).resolve(), Path(dst).resolve()
    info = json.loads((src / "meta/info.json").read_text())
    if dst.exists():
        if not force:
            raise SystemExit(f"{dst} exists (use --force)")
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("videos", ".cache"))

    videos = sorted((src / "videos").rglob("*.mp4"))
    if not videos:
        raise SystemExit(f"{src} has no videos/*.mp4")
    threads = max(1, (os.cpu_count() or 4) // jobs)
    with cf.ProcessPoolExecutor(jobs) as ex:
        futs = [ex.submit(transcode_file, v, dst / v.relative_to(src), info["fps"], crf, threads)
                for v in videos]
        reports = [f.result() for f in futs]

    for ft in info["features"].values():
        if ft.get("dtype") == "video":
            vi = ft.setdefault("info", {})
            vi.pop("video.preset", None)  # the source encoder's, meaningless for libx264
            vi.update({"video.codec": "h264", "video.pix_fmt": "yuv420p", "video.g": GOP, "video.crf": crf})
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    reports = transcode(a.src, a.dst, a.crf, a.jobs, a.force)
    worst = min(r["psnr_db"] for r in reports)
    print(f"{len(reports)} videos, {sum(r['frames'] for r in reports)} frames, "
          f"timestamps match, worst PSNR {worst:.1f} dB -> {a.dst}")


if __name__ == "__main__":
    main()
