#!/bin/bash
# RealSense D435i install for Jetson Orin Nano with PREEMPT_RT kernel
#
# Run this while booted into the RT kernel.
# Requires: librealsense source already cloned to LIBREALSENSE_SRC,
#           jetson-orin-librealsense repo already cloned to JETHACKS_RS_SRC.
#
# Time: ~1.5-2 hours (kernel source download + CUDA build dominate)

set -e

# ── config ────────────────────────────────────────────────────────────────────
LIBREALSENSE_SRC="${LIBREALSENSE_SRC:-$HOME/librealsense}"
JETHACKS_RS_SRC="${JETHACKS_RS_SRC:-$HOME/Desktop/jetson-orin-librealsense-main}"
KERNEL_BUILDER_DIR="$HOME/jetson-orin-kernel-builder"
KERNEL_SRC="/usr/src/kernel/kernel-jammy-src"
KVER="$(uname -r)"
CUDA_ARCH=87   # all Jetson Orin (Nano/NX/AGX) = sm_87; Xavier would be 72, desktop Ampere 86
# ─────────────────────────────────────────────────────────────────────────────

if [[ "$KVER" != *"rt"* ]]; then
    echo "ERROR: not running on RT kernel (uname -r = $KVER). Boot into RT kernel first."
    exit 1
fi

echo "==> Kernel: $KVER"
echo "==> CUDA arch: sm_$CUDA_ARCH"
echo "==> librealsense src: $LIBREALSENSE_SRC"
echo "==> JetsonHacks RS src: $JETHACKS_RS_SRC"
echo ""

# ── 1. kernel source ──────────────────────────────────────────────────────────
if [ ! -f "$KERNEL_SRC/Makefile" ]; then
    if [ ! -d "$KERNEL_BUILDER_DIR/.git" ]; then
        echo "==> Cloning JetsonHacks kernel builder..."
        git clone https://github.com/jetsonhacks/jetson-orin-kernel-builder "$KERNEL_BUILDER_DIR"
    else
        echo "==> Kernel builder already present at $KERNEL_BUILDER_DIR, skipping clone."
    fi
    echo "==> Downloading kernel source (~2 GB, takes a few minutes)..."
    cd "$KERNEL_BUILDER_DIR"
    ./scripts/get_kernel_sources.sh --force-replace
else
    echo "==> Kernel source already present at $KERNEL_SRC, skipping download."
fi

# ── 2. verify RT config ───────────────────────────────────────────────────────
echo "==> Checking kernel config..."
grep -q "CONFIG_PREEMPT_RT=y" "$KERNEL_SRC/.config" || {
    echo "ERROR: CONFIG_PREEMPT_RT not set in kernel config."
    echo "       Make sure get_kernel_sources.sh was run while booted on RT kernel."
    exit 1
}
echo "    CONFIG_PREEMPT_RT=y confirmed."

# Guard against a vermagic mismatch: if the source's LOCALVERSION doesn't match
# the running kernel's suffix, every module builds fine but modprobe silently
# refuses them at the end. Catch it now instead of after a 30-min build.
SRC_LOCALVERSION="$(sed -n 's/^CONFIG_LOCALVERSION="\(.*\)"/\1/p' "$KERNEL_SRC/.config")"
RUN_SUFFIX="-${KVER#*-}"   # 5.15.148-rt-tegra -> -rt-tegra
if [ "$SRC_LOCALVERSION" != "$RUN_SUFFIX" ]; then
    echo "WARNING: kernel-source CONFIG_LOCALVERSION='$SRC_LOCALVERSION' != running suffix '$RUN_SUFFIX'."
    echo "         Built modules' vermagic may not match $KVER and modprobe will reject them."
    echo "         Re-run get_kernel_sources.sh while booted on the RT kernel, or fix CONFIG_LOCALVERSION."
    read -r -p "         Continue anyway? [y/N] " ans
    [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 1
fi

# ── 3. apply RealSense patches ────────────────────────────────────────────────
echo "==> Applying RealSense patches..."
cd "$JETHACKS_RS_SRC/build"
./patch-for-realsense.sh

# ── 4. configure HID modules ──────────────────────────────────────────────────
echo "==> Configuring HID sensor modules..."
cd "$KERNEL_SRC"
sudo bash scripts/config --set-val CONFIG_HID_SENSOR_HUB m
sudo bash scripts/config --set-val CONFIG_HID_SENSOR_ACCEL_3D m
sudo bash scripts/config --set-val CONFIG_HID_SENSOR_GYRO_3D m
sudo make olddefconfig

# ── 5. build kernel modules ───────────────────────────────────────────────────
echo "==> Building kernel modules (~15 min)..."
sudo make -j"$(nproc)" prepare
sudo make -j"$(nproc)" modules_prepare
sudo make -j"$(nproc)" M=drivers/media/usb/uvc modules
sudo make -j"$(nproc)" M=drivers/hid modules
sudo make -j"$(nproc)" M=drivers/iio/accel modules
sudo make -j"$(nproc)" M=drivers/iio/gyro modules
sudo make -j"$(nproc)" M=drivers/iio/common/hid-sensors modules

# ── 6. install modules ────────────────────────────────────────────────────────
echo "==> Installing kernel modules to /lib/modules/$KVER/..."
BASE=/lib/modules/$KVER/kernel

sudo mkdir -p "$BASE/drivers/media/usb/uvc"
sudo mkdir -p "$BASE/drivers/hid"
sudo mkdir -p "$BASE/drivers/iio/accel"
sudo mkdir -p "$BASE/drivers/iio/gyro"
sudo mkdir -p "$BASE/drivers/iio/common/hid-sensors"

sudo cp drivers/media/usb/uvc/uvcvideo.ko              "$BASE/drivers/media/usb/uvc/"
sudo cp drivers/hid/hid-sensor-hub.ko                  "$BASE/drivers/hid/"
sudo cp drivers/iio/accel/hid-sensor-accel-3d.ko       "$BASE/drivers/iio/accel/"
sudo cp drivers/iio/gyro/hid-sensor-gyro-3d.ko         "$BASE/drivers/iio/gyro/"
sudo cp drivers/iio/common/hid-sensors/hid-sensor-iio-common.ko "$BASE/drivers/iio/common/hid-sensors/"
sudo cp drivers/iio/common/hid-sensors/hid-sensor-trigger.ko    "$BASE/drivers/iio/common/hid-sensors/"

sudo depmod -a

# ── 7. boot persistence ───────────────────────────────────────────────────────
echo "==> Adding HID modules to /etc/modules..."
for mod in hid-sensor-hub hid-sensor-accel-3d hid-sensor-gyro-3d; do
    grep -qxF "$mod" /etc/modules || echo "$mod" | sudo tee -a /etc/modules
done

# ── 8. rebuild librealsense ───────────────────────────────────────────────────
echo "==> Rebuilding librealsense (CUDA build, ~30-45 min)..."
mkdir -p "$LIBREALSENSE_SRC/build"
cd "$LIBREALSENSE_SRC/build"
sudo cmake .. \
    -DFORCE_RSUSB_BACKEND=OFF \
    -DBUILD_WITH_CUDA=ON \
    -DBUILD_WITH_OPENMP=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH \
    -DCMAKE_BUILD_TYPE=Release
# CUDA compiles are memory-heavy; full -j can OOM the 8 GB Nano. Cap the job count
# (override with BUILD_JOBS=N). Falls back to all cores if free RAM looks ample.
BUILD_JOBS="${BUILD_JOBS:-$(( $(nproc) > 4 ? 4 : $(nproc) ))}"
echo "    Building with -j$BUILD_JOBS (set BUILD_JOBS to override)."
sudo make -j"$BUILD_JOBS"
sudo make install

# ── 9. load modules now (without reboot) ─────────────────────────────────────
echo "==> Loading HID modules..."
sudo modprobe hid-sensor-hub
sudo modprobe hid-sensor-accel-3d
sudo modprobe hid-sensor-gyro-3d

echo ""
echo "==> Done. Testing device enumeration..."
rs-enumerate-devices 2>&1 | grep -E "Name|Serial|Motion|ERROR|Firmware" || \
    echo "    (no matching enumeration output — check the camera is plugged in)"
echo ""
echo "All done. Reboot to confirm modules load automatically, then run realsense-viewer."
