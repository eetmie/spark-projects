"""Frame sources for the runtime: a threaded RealSense D435i RGB reader and a
synthetic source for running the pipeline before a camera (or an ONNX) is around.

The D435i reader is deliberately a background thread that always holds the *latest*
frame — inference should never block on the camera, and a control loop wants the
freshest frame, not a queued backlog. Adapted from the prior smolvla bench project.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOG = logging.getLogger("smolvla_runtime.camera")


def import_pyrealsense2():
    """Import pyrealsense2, including the common Jetson system-install paths.

    The librealsense source build installs the Python binding under
    /usr/lib/python3/dist-packages, which a venv won't see by default.
    """
    try:
        import pyrealsense2 as rs
        return rs
    except ModuleNotFoundError:
        pass
    py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for path in (
        Path("/usr/lib/python3/dist-packages"),
        Path(f"/usr/lib/{py_tag}/dist-packages"),
        Path(f"/usr/local/lib/{py_tag}/dist-packages"),
    ):
        if path.exists() and str(path) not in sys.path:
            sys.path.append(str(path))
    import pyrealsense2 as rs
    return rs


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    lock_auto_exposure: bool = False
    # Keep auto-exposure adaptive but forbid it from stretching exposure past the
    # frame budget. Without this the D435i silently halves the color frame rate in
    # dim light (the "30fps that's really 15fps" trap on the RSUSB/no-depth path).
    pin_frame_rate: bool = True


class RealSenseRGB:
    """Background-threaded latest-frame RGB reader for the D435i."""

    def __init__(self, cfg: CameraConfig | None = None):
        self._rs = import_pyrealsense2()
        rs = self._rs
        self._cfg = cfg or CameraConfig()
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(
            rs.stream.color, self._cfg.width, self._cfg.height,
            rs.format.bgr8, self._cfg.fps,
        )
        self._latest_rgb: np.ndarray | None = None
        self._latest_host_ts = 0.0
        self._latest_camera_ts_ms = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        profile = self._pipeline.start(self._config)
        if self._cfg.pin_frame_rate:
            self._try_pin_frame_rate(profile)
        if self._cfg.lock_auto_exposure:
            self._try_lock_auto_exposure(profile)
        self._thread = threading.Thread(target=self._run, name="d435i-rgb", daemon=True)
        self._thread.start()
        LOG.info("D435i RGB stream started: %dx%d @ %d Hz",
                 self._cfg.width, self._cfg.height, self._cfg.fps)

    def _try_pin_frame_rate(self, profile) -> None:
        # auto_exposure_priority = 0 -> constant frame rate; AE adjusts gain/exposure
        # within the 1/fps budget instead of lengthening exposure (which drops fps).
        try:
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(self._rs.option.auto_exposure_priority):
                    sensor.set_option(self._rs.option.auto_exposure_priority, 0)
                    LOG.info("D435i auto-exposure priority off (frame rate pinned to %d Hz)",
                             self._cfg.fps)
                    return
        except Exception as exc:
            LOG.warning("Could not pin frame rate (auto_exposure_priority): %s", exc)

    def _try_lock_auto_exposure(self, profile) -> None:
        try:
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(self._rs.option.enable_auto_exposure):
                    sensor.set_option(self._rs.option.enable_auto_exposure, 0)
                    LOG.info("D435i auto-exposure disabled")
                    return
        except Exception as exc:
            LOG.warning("Could not disable auto-exposure: %s", exc)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            except Exception as exc:
                LOG.warning("wait_for_frames failed: %s", exc)
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            bgr = np.asanyarray(color.get_data())
            rgb = bgr[:, :, ::-1].copy()
            with self._lock:
                self._latest_rgb = rgb
                self._latest_host_ts = time.perf_counter()
                self._latest_camera_ts_ms = float(color.get_timestamp())

    def latest(self) -> tuple[np.ndarray | None, float, float]:
        with self._lock:
            if self._latest_rgb is None:
                return None, 0.0, 0.0
            return self._latest_rgb.copy(), self._latest_host_ts, self._latest_camera_ts_ms

    def wait_for_first_frame(self, timeout_s: float = 5.0) -> bool:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            frame, _, _ = self.latest()
            if frame is not None:
                return True
            time.sleep(0.02)
        return False

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self._pipeline.stop()
        except Exception:
            pass


class SyntheticRGB:
    """Deterministic moving gradient — lets the full pipeline run with no camera."""

    def __init__(self, cfg: CameraConfig | None = None):
        self._cfg = cfg or CameraConfig()
        self._step = 0

    def start(self) -> None:
        LOG.info("Synthetic RGB source: %dx%d", self._cfg.width, self._cfg.height)

    def latest(self) -> tuple[np.ndarray, float, float]:
        w, h = self._cfg.width, self._cfg.height
        self._step += 1
        x = np.linspace(0, 255, w, dtype=np.uint8)
        y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        r = np.broadcast_to(x, (h, w))
        g = np.broadcast_to(y, (h, w))
        b = np.full((h, w), (self._step * 7) % 255, dtype=np.uint8)
        rgb = np.stack([r, g, b], axis=2)
        now = time.perf_counter()
        return rgb, now, now * 1000.0

    def wait_for_first_frame(self, timeout_s: float = 5.0) -> bool:
        return True

    def stop(self) -> None:
        pass
