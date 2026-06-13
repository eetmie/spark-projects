#!/usr/bin/env python3
"""SmolVLA model pipeline on Jetson Orin Nano (JetPack 7.2).

    RealSense D435i RGB  ->  SmolVLA (ONNX Runtime + TensorRT EP)  ->  action chunk

This is the model pipeline ONLY — it never sends commands to a robot. It reads
the latest RGB frame, runs one SmolVLA inference, and reports the action-chunk
shape, a preview, and latency. Mapping actions onto a real robot's command space
(normalization, limits, safety) is intentionally out of scope here.

Backends:
  ort   ONNX Runtime + TensorRT EP (the deployment path)  --onnx-path model.onnx
        --precision fp16 (default, accelerated on Orin) | bf16 (experimental)
  mock  zero actions, no model     (camera + loop plumbing check)

Sources:
  --source realsense   the D435i RGB stream (default)
  --source synthetic   a generated frame (run with no camera / no ONNX)

Examples
--------
    # 1. Plumbing only — no model, just confirm the camera + loop:
    python run_pipeline.py --backend mock --source realsense --duration-s 5

    # 2. ORT/TRT-EP, synthetic frames (builds + caches the engine, no camera needed):
    python run_pipeline.py --backend ort --onnx-path exports/smolvla.onnx \
        --model-id lerobot/smolvla_base --source synthetic --duration-s 20 --show-actions

    # 3. ORT/TRT-EP + real D435i RGB:
    python run_pipeline.py --backend ort --onnx-path exports/smolvla.onnx \
        --model-id lerobot/smolvla_base --source realsense --duration-s 30 --show-actions
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from smolvla_runtime.camera import CameraConfig, RealSenseRGB, SyntheticRGB

LOG = logging.getLogger("run_pipeline")


def percentile(values, pct):
    return float(np.percentile(values, pct)) if values else 0.0


def parse_state(value):
    if not value or not value.strip():
        return None
    return np.asarray([float(p) for p in value.split(",")], dtype=np.float32)


def build_backend(args):
    if args.backend == "mock":
        from smolvla_runtime.backends.mock import MockBackend
        return MockBackend()
    if args.backend == "ort":
        if not args.onnx_path:
            raise SystemExit("--onnx-path is required for --backend ort.")
        from smolvla_runtime.backends.ort import ORTBackend
        return ORTBackend(args.onnx_path, model_id=args.model_id,
                          engine_cache_dir=args.engine_cache_dir,
                          precision=args.precision, fixed_noise=args.fixed_noise)
    raise SystemExit(f"unknown backend {args.backend}")


def make_source(args):
    cfg = CameraConfig(width=args.width, height=args.height, fps=args.fps)
    return SyntheticRGB(cfg) if args.source == "synthetic" else RealSenseRGB(cfg)


def summarize_actions(actions, n=8):
    arr = np.asarray(actions, dtype=np.float32)
    if arr.size == 0:
        return "[]"
    first = arr.reshape(-1, arr.shape[-1])[0]
    head = ", ".join(f"{v:+.3f}" for v in first[:n])
    more = " ..." if first.size > n else ""
    return f"[{head}{more}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("mock", "ort"), default="mock")
    ap.add_argument("--source", choices=("realsense", "synthetic"), default="realsense")
    ap.add_argument("--onnx-path", help="FP32 ONNX (--backend ort).")
    ap.add_argument("--precision", choices=("fp16", "bf16"), default="fp16",
                    help="TRT-EP reduced precision. fp16 = Orin deploy default; "
                         "bf16 = experimental (not hardware-accelerated on compute 8.7).")
    ap.add_argument("--engine-cache-dir", default="/tmp/smolvla_trt_cache")
    ap.add_argument("--model-id", default="lerobot/smolvla_base",
                    help="HF id or local dir for the tokenizer.")
    ap.add_argument("--instruction", default="pick up the object")
    ap.add_argument("--state", default=None, help="Comma-separated robot state vector.")
    ap.add_argument("--fixed-noise", action="store_true",
                    help="Reuse one noise draw (reproducible, cleaner latency numbers).")
    ap.add_argument("--duration-s", type=float, default=30.0)
    ap.add_argument("--policy-hz", type=float, default=0.0,
                    help="0 = run flat out (benchmark); >0 = throttle the loop to this rate.")
    ap.add_argument("--print-every-s", type=float, default=2.0)
    ap.add_argument("--show-actions", action="store_true")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    backend = build_backend(args)
    LOG.info("Backend: %s", backend.description)

    source = make_source(args)
    source.start()
    if not source.wait_for_first_frame(timeout_s=5.0):
        LOG.error("No frame from source within 5 s. Is the D435i connected? "
                  "(librealsense RGB must be installed — see ../realsense-rgb.)")
        source.stop()
        return 1

    state = parse_state(args.state)
    period = 1.0 / args.policy_hz if args.policy_hz > 0 else 0.0

    lat, loops = [], []
    count = 0
    last_report = time.perf_counter()
    end = time.perf_counter() + args.duration_s
    last_actions = None

    try:
        while time.perf_counter() < end:
            loop_t0 = time.perf_counter()
            frame, host_ts, _ = source.latest()
            if frame is None:
                time.sleep(0.005)
                continue
            img_age_ms = (time.perf_counter() - host_ts) * 1000.0

            result = backend.predict(frame, args.instruction, state)
            last_actions = result.actions

            loop_ms = (time.perf_counter() - loop_t0) * 1000.0
            lat.append(result.latency_ms)
            loops.append(loop_ms)
            count += 1

            if time.perf_counter() - last_report >= args.print_every_s:
                elapsed = time.perf_counter() - last_report
                LOG.info(
                    "loop_hz=%.1f infer_ms(avg/p95)=%.1f/%.1f loop_ms(avg/p95)=%.1f/%.1f "
                    "img_age_ms=%.1f action_shape=%s",
                    count / max(elapsed, 1e-9),
                    float(np.mean(lat)), percentile(lat, 95),
                    float(np.mean(loops)), percentile(loops, 95),
                    img_age_ms, tuple(np.asarray(last_actions).shape),
                )
                if args.show_actions and last_actions is not None:
                    LOG.info("  action[0] = %s", summarize_actions(last_actions))
                lat.clear(); loops.clear(); count = 0
                last_report = time.perf_counter()

            if period:
                sleep = period - (time.perf_counter() - loop_t0)
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        LOG.info("Interrupted.")
    finally:
        source.stop()

    if last_actions is not None:
        LOG.info("Final action chunk shape: %s", tuple(np.asarray(last_actions).shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
