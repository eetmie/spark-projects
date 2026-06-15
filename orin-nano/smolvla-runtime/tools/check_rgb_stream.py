#!/usr/bin/env python3
"""Verify the D435i color stream is *actually* stable at the requested rate.

The VLA only consumes RGB, so this is the one camera property that matters before
trusting the runtime: does the color stream hold its nominal fps with no dropped
frames on the RSUSB/no-depth path? The classic failure (see camera.py) is the
sensor silently halving fps in dim light when auto_exposure_priority is left on.

Unlike the threaded reader in camera.py (which keeps only the latest frame), this
reads *every* frame straight off the pipeline so it can detect drops from gaps in
the hardware frame number, and reports delivered fps + inter-frame jitter.

    python tools/check_rgb_stream.py                 # 640x480@30 for 10s, pin on
    python tools/check_rgb_stream.py --no-pin        # A/B: auto_exposure_priority on
    python tools/check_rgb_stream.py --width 848 --height 480 --duration-s 20

512x512 is NOT a native D435i color mode; we capture a native 4:3 mode and show a
center-crop + resize to the --target square the model wants (default 512).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the runtime's import shim + resize so this matches the real pipeline.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smolvla_runtime.camera import import_pyrealsense2  # noqa: E402
from smolvla_runtime.preprocess import resize_rgb  # noqa: E402


def center_square_crop(rgb: np.ndarray) -> np.ndarray:
    """Crop the largest centered square — avoids the anamorphic squash a straight
    4:3 -> 1:1 resize would do. Use this if the model was trained on square crops."""
    h, w = rgb.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return rgb[y0:y0 + s, x0:x0 + s]


def list_color_profiles(rs, device) -> None:
    print("Supported COLOR profiles (native sensor modes):")
    modes = set()
    for sensor in device.query_sensors():
        for p in sensor.get_stream_profiles():
            if p.stream_type() != rs.stream.color:
                continue
            v = p.as_video_stream_profile()
            modes.add((v.width(), v.height(), p.fps(), p.format().name))
    for w, h, fps, fmt in sorted(modes, reverse=True):
        print(f"  {w}x{h} @ {fps:>3} Hz  {fmt}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration-s", type=float, default=10.0)
    ap.add_argument("--target", type=int, default=512, help="Square size the model wants.")
    ap.add_argument("--no-pin", action="store_true",
                    help="Leave auto_exposure_priority ON (the unstable default) to A/B it.")
    args = ap.parse_args()

    rs = import_pyrealsense2()
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        print("ERROR: no RealSense device found.")
        return 1
    dev = devs[0]
    print(f"Device: {dev.get_info(rs.camera_info.name)}  "
          f"FW {dev.get_info(rs.camera_info.firmware_version)}  "
          f"USB {dev.get_info(rs.camera_info.usb_type_descriptor)}")
    list_color_profiles(rs, dev)
    print()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)

    pinned = False
    if not args.no_pin:
        for sensor in profile.get_device().query_sensors():
            if sensor.supports(rs.option.auto_exposure_priority):
                sensor.set_option(rs.option.auto_exposure_priority, 0)
                pinned = True
    print(f"Streaming {args.width}x{args.height}@{args.fps} BGR8 for {args.duration_s:.0f}s "
          f"(auto_exposure_priority {'OFF -> rate pinned' if pinned else 'ON -> may sag'}) ...")

    arrivals: list[float] = []   # host arrival times (perf_counter)
    frame_nums: list[int] = []
    # Warm up: the first second of AE settling isn't representative.
    warmup_until = time.perf_counter() + 1.0
    end = time.perf_counter() + 1.0 + args.duration_s
    try:
        while time.perf_counter() < end:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except Exception as exc:  # noqa: BLE001
                print(f"  wait_for_frames failed: {exc}")
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            if time.perf_counter() < warmup_until:
                continue
            arrivals.append(time.perf_counter())
            frame_nums.append(color.get_frame_number())
    finally:
        pipeline.stop()

    if len(arrivals) < 3:
        print("ERROR: too few frames received — stream did not start.")
        return 1

    # One 512-square sample to prove the model-facing path works end to end.
    # (re-open briefly is overkill; reuse shape from the last config instead)
    sample = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    square = resize_rgb(center_square_crop(sample), args.target)

    span = arrivals[-1] - arrivals[0]
    n = len(arrivals)
    delivered_fps = (n - 1) / span if span > 0 else 0.0
    dt = np.diff(arrivals)
    inst_fps = 1.0 / dt
    nominal_dt = 1.0 / args.fps
    # Dropped frames = gaps in the hardware frame counter beyond the expected +1.
    fn = np.array(frame_nums)
    steps = np.diff(fn)
    dropped = int(np.clip(steps - 1, 0, None).sum())
    within = float(np.mean(np.abs(dt - nominal_dt) <= 0.20 * nominal_dt) * 100.0)

    print("\n--- results --------------------------------------------------")
    print(f"frames received        {n}  over {span:.1f}s")
    print(f"delivered fps          {delivered_fps:5.2f}   (nominal {args.fps})")
    print(f"instantaneous fps      p50 {np.percentile(inst_fps,50):5.2f}  "
          f"p5 {np.percentile(inst_fps,5):5.2f}  p95 {np.percentile(inst_fps,95):5.2f}")
    print(f"inter-frame gap        max {dt.max()*1000:5.1f}ms  "
          f"(nominal {nominal_dt*1000:.1f}ms)")
    print(f"intervals within +/-20% {within:4.1f}%")
    print(f"dropped frames         {dropped}")
    print(f"model-facing frame     {sample.shape[1]}x{sample.shape[0]} "
          f"-> crop -> {square.shape[1]}x{square.shape[0]} (target {args.target})")

    stable = (delivered_fps >= 0.97 * args.fps and dropped == 0 and within >= 95.0)
    print(f"\nVERDICT: {'STABLE ✅' if stable else 'UNSTABLE ⚠️'} "
          f"at {args.width}x{args.height}@{args.fps}")
    if not stable and not pinned:
        print("  -> retry without --no-pin (auto_exposure_priority off) before concluding.")
    return 0 if stable else 2


if __name__ == "__main__":
    raise SystemExit(main())
