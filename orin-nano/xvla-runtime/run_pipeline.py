#!/usr/bin/env python3
"""Run the X-VLA split pipeline and report latency + memory.

Doubles as the stress test: `--duration-s 1800` runs the loop for half an hour and prints
a periodic memory line, which is the thing that actually needs watching on this board.
Resident memory matters as much as the build peak -- a dozen TRT sessions each hold their
own weights and CUDA context, and unified memory means that all comes out of the same
7.4 GB.

    python run_pipeline.py --duration-s 30 --show-actions
    python run_pipeline.py --duration-s 1800 --report-every 60     # stress
"""

from __future__ import annotations

import argparse
import logging
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from xvla_runtime.split_ort import XVLASplitPolicy, prebuild_engines  # noqa: E402

LOG = logging.getLogger("run_pipeline")


def meminfo_available_gb() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", type=Path, default=Path("exports/split"))
    ap.add_argument("--cache-dir", default=None,
                    help="TRT engine cache; defaults to <split-dir>/trt_cache")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer dir; defaults to models/tokenizer when present. "
                         "Falls back to the hub id, which needs network -- not something "
                         "to discover on a robot with no connectivity.")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "realsense"])
    ap.add_argument("--instruction", default="pick up the rock and place it in the bucket")
    ap.add_argument("--duration-s", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=None,
                    help="override num_denoising_steps (the main latency lever)")
    ap.add_argument("--report-every", type=float, default=0.0,
                    help="seconds between memory/latency lines; 0 = only at the end")
    ap.add_argument("--prebuild", action="store_true",
                    help="build every engine in its own subprocess first (do this on a "
                         "cold cache -- building inside the runtime process risks OOM)")
    ap.add_argument("--show-actions", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    if args.prebuild:
        LOG.info("prebuilding engines (one subprocess per graph) ...")
        t0 = time.time()
        prebuild_engines(args.split_dir, args.cache_dir, args.precision)
        LOG.info("prebuild done in %.0f s", time.time() - t0)

    base_avail = meminfo_available_gb()
    LOG.info("available before load: %.2f GB", base_avail)

    tokenizer = args.tokenizer
    if tokenizer is None:
        local = Path(__file__).resolve().parent / "models" / "tokenizer"
        tokenizer = str(local) if local.is_dir() else None
        if tokenizer is None:
            LOG.warning("no local models/tokenizer -- falling back to the hub, which "
                        "needs network. Save one with AutoTokenizer.save_pretrained.")

    t0 = time.time()
    policy = XVLASplitPolicy(
        args.split_dir, cache_dir=args.cache_dir, precision=args.precision,
        tokenizer_dir=tokenizer, num_denoising_steps=args.steps,
    )
    load_s = time.time() - t0
    after_load = meminfo_available_gb()
    n_sessions = (
        len(policy.vision) + len(policy.text_encoder) + 1 + len(policy.denoise)
    )
    LOG.info("loaded %d sessions in %.0f s | available %.2f GB (load cost %.2f GB) | rss %.2f GB",
             n_sessions, load_s, after_load, base_avail - after_load, rss_gb())

    if args.source == "realsense":
        sys.exit("realsense source not wired here yet -- see kaivuriprokkis/lerobot_vla "
                 "for the D435i reader; this stage is model-only")

    rng = np.random.default_rng(0)
    images = [
        rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        for _ in range(policy.valid_views)
    ]
    state = rng.standard_normal(policy.state_dim).astype(np.float32)

    LOG.info("warming up ...")
    action = policy.sample_actions(images, args.instruction, state)

    chunk_ms: list[float] = []
    stage: dict[str, list[float]] = {}
    min_avail = meminfo_available_gb()
    start = time.time()
    last_report = start

    while time.time() - start < args.duration_s:
        t = time.time()
        action = policy.sample_actions(images, args.instruction, state)
        chunk_ms.append((time.time() - t) * 1000)
        for k, v in policy.last_timings.items():
            if k.endswith("_ms"):
                stage.setdefault(k, []).append(v)
        min_avail = min(min_avail, meminfo_available_gb())

        if args.report_every and time.time() - last_report >= args.report_every:
            last_report = time.time()
            LOG.info("n=%d  chunk avg %.0f ms p95 %.0f ms | available %.2f GB "
                     "(min %.2f) | rss %.2f GB",
                     len(chunk_ms), float(np.mean(chunk_ms)), pct(chunk_ms, 95),
                     meminfo_available_gb(), min_avail, rss_gb())

    if args.show_actions:
        print(f"\naction chunk {action.shape}:\n{np.array2string(action[:3], precision=3)}")

    if not chunk_ms:
        print(f"\nno timed chunks (--duration-s {args.duration_s}); "
              f"memory: available min {min_avail:.2f} GB, peak rss {rss_gb():.2f} GB")
        return

    print(f"\n{len(chunk_ms)} chunks over {time.time() - start:.0f} s")
    print(f"  chunk       avg {np.mean(chunk_ms):7.1f} ms   p95 {pct(chunk_ms, 95):7.1f} ms   "
          f"min {min(chunk_ms):7.1f} ms")
    for k in sorted(stage):
        v = stage[k]
        print(f"  {k:12s}avg {np.mean(v):7.1f} ms   p95 {pct(v, 95):7.1f} ms")
    print(f"  replan rate {1000 / np.mean(chunk_ms):.2f} Hz "
          f"({policy.chunk_size} actions per chunk)")
    print(f"  memory: available min {min_avail:.2f} GB, peak rss {rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
