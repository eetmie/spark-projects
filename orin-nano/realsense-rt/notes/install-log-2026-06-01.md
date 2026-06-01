REALSENSE D435i ON JETSON ORIN NANO — RT KERNEL INSTALL LOG
2026-06-01

WHAT WAS DONE
-------------
Goal: install RealSense D435i properly on RT kernel (5.15.148-rt-tegra),
      keeping PREEMPT_RT for 100Hz IK/control stack.

Previous state:
  - librealsense v2.57.7 built with FORCE_RSUSB_BACKEND=true (userspace USB)
  - No CUDA, no OpenMP
  - RGB stream stuttering under load
  - IMU not working (HID modules not present)

Steps completed:
  1. Cloned JetsonHacks jetson-orin-kernel-builder
  2. Ran get_kernel_sources.sh — auto-detected RT kernel, copied /proc/config.gz
     as .config (LOCALVERSION=-rt-tegra, CONFIG_PREEMPT_RT=y confirmed)
  3. Applied 2 RealSense patches from jetson-orin-librealsense-main/build/:
       - realsense-metadata-focal-hwe-5.15.patch
       - realsense-camera-formats-focal-hwe-5.15.patch
     (3rd HID patch already merged in 5.15.148 — skipped)
  4. Set HID modules via scripts/config + make olddefconfig:
       CONFIG_HID_SENSOR_HUB=m
       CONFIG_HID_SENSOR_ACCEL_3D=m
       CONFIG_HID_SENSOR_GYRO_3D=m
  5. Built 6 kernel modules (make -j$(nproc) M=... modules)
  6. Created missing dir: /lib/modules/5.15.148-rt-tegra/kernel/drivers/iio/common/hid-sensors/
  7. Copied all 6 .ko files into /lib/modules/5.15.148-rt-tegra/kernel/...
  8. Skipped install-udev.sh — existing rules (96 lines) newer than JetsonHacks (91 lines)
  9. sudo depmod -a
  10. Rebuilt librealsense from ~/librealsense with:
        FORCE_RSUSB_BACKEND=OFF
        BUILD_WITH_CUDA=ON
        BUILD_WITH_OPENMP=ON
        CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
        CMAKE_CUDA_ARCHITECTURES=87
        CMAKE_BUILD_TYPE=Release
  11. sudo modprobe hid-sensor-hub hid-sensor-accel-3d hid-sensor-gyro-3d
  12. Added all three to /etc/modules for boot persistence

Result:
  - RGB stream: no stutter
  - IMU: working in realsense-viewer
  - Backend: kernel UVC (not RSUSB)
  - CUDA + OpenMP: enabled
  - Verified clean: no old library conflicts, librealsense2.so.2.57.7 dated today

See ~/Desktop/realsense-install-rt.sh for a reusable install script.
