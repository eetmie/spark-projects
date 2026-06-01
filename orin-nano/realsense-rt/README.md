# RealSense D435i on Jetson Orin Nano — PREEMPT_RT kernel

Reproducible setup for an Intel RealSense **D435i** on a Jetson **Orin Nano** running the
**PREEMPT_RT** kernel (`5.15.148-rt-tegra`, L4T R36.4.4 / JetPack 6).

The RT kernel is intentional — the downstream control stack runs a 100 Hz IK/control loop and
needs bounded latency. The catch: JetsonHacks' prebuilt RealSense kernel modules are compiled for
the **stock** `5.15.148-tegra` kernel and will not load on `-rt-tegra`. So the modules have to be
built from source against the RT kernel. That is what this playbook automates.

> **Status: done and live-verified on this device.** The D435i enumerates over the kernel-UVC
> backend (not the userspace RSUSB fallback), the RGB stream is stutter-free, and the IMU
> `hid_sensor_*` modules load at boot. This folder is the *recipe* so it can be reproduced on a
> reflashed board or a second unit.

## What "working" looks like here

- librealsense **2.57.7** built from source with `FORCE_RSUSB_BACKEND=OFF` (kernel UVC backend),
  `BUILD_WITH_CUDA=ON`, `BUILD_WITH_OPENMP=ON`, `CMAKE_CUDA_ARCHITECTURES=87` (Orin = sm_87).
- Six RealSense kernel modules built against the RT kernel source and installed under
  `/lib/modules/5.15.148-rt-tegra/kernel/...`:
  `uvcvideo.ko`, `hid-sensor-hub.ko`, `hid-sensor-accel-3d.ko`, `hid-sensor-gyro-3d.ko`,
  `hid-sensor-iio-common.ko`, `hid-sensor-trigger.ko`.
- The three HID sensor modules added to `/etc/modules` for boot persistence.

## Prerequisites (external sources, not vendored here)

These are large upstream trees — clone them on the device, don't commit them:

| What | Where it goes | Source |
|---|---|---|
| librealsense source | `~/librealsense` | https://github.com/IntelRealSense/librealsense |
| JetsonHacks RealSense patches | `~/Desktop/jetson-orin-librealsense-main` | https://github.com/jetsonhacks/jetson-orin-librealsense |
| Kernel-source fetcher | `~/jetson-orin-kernel-builder` | https://github.com/jetsonhacks/jetson-orin-kernel-builder |

The install script clones the kernel builder itself if it is missing; the other two it expects to
already be present (override with the `LIBREALSENSE_SRC` / `JETHACKS_RS_SRC` env vars).

## Reproduce

Boot into the RT kernel first (`uname -r` must contain `rt`), plug in the D435i, then:

```bash
./install-realsense-rt.sh
```

Roughly 1.5–2 h end to end (kernel source download + CUDA rebuild dominate). The script is
idempotent on the slow parts: it skips the kernel-source download if the tree is already present.

What it does, in order: fetch RT kernel source → verify `CONFIG_PREEMPT_RT=y` → apply the two
JetsonHacks RealSense patches → set the three HID modules to `m` → build & install the six modules
→ persist them in `/etc/modules` → rebuild librealsense with CUDA/OpenMP and the kernel UVC backend
→ `modprobe` the HID modules → enumerate the device.

## Validate (after reboot, camera plugged in)

```bash
./tests/validate.sh
```

Checks module load, device enumeration, IMU IIO devices, and runs a 30 s `cyclictest` RT-latency
sweep (target: max well under 1 ms for the 100 Hz loop). See `notes/validation-checklist.txt` for
the manual `realsense-viewer` stream-under-load check.

## Files

- `install-realsense-rt.sh` — the reproducible installer.
- `tests/validate.sh` — post-reboot validation.
- `udev/99-realsense-libusb.rules` — the udev rules in use on this device (kept newer than the
  JetsonHacks copy; `install-udev.sh` was intentionally **not** run so as not to downgrade them).
- `notes/install-log-2026-06-01.md` — the actual install log from when this was first done.
- `notes/validation-checklist.txt` — manual validation steps.

## Gotchas worth remembering

- **Do not switch kernels to use prebuilt modules.** The whole point is to keep PREEMPT_RT. Build
  the modules against it.
- **Third HID patch is already merged** into 5.15.148 — only two of the three JetsonHacks patches
  apply. The script applies exactly those two.
- **A GB10-built artifact is the wrong artifact.** Kernel modules and TensorRT engines are both
  hardware/kernel specific — build them on this board.
