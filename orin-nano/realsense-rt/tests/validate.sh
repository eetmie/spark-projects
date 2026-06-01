#!/bin/bash
# RealSense D435i + RT-kernel validation.
# Run AFTER reboot, with the camera plugged in. Non-destructive (read-only checks
# except cyclictest, which needs sudo). Derived from realsense-tests.txt.

set -u
KVER="$(uname -r)"
fail=0

echo "==> Kernel: $KVER"
[[ "$KVER" == *rt* ]] || { echo "  WARN: not on an RT kernel"; }

echo "==> 1. Modules loaded automatically (no manual modprobe)?"
lsmod | grep -qE "uvcvideo"      && echo "  ok  uvcvideo"            || { echo "  MISS uvcvideo";            fail=1; }
lsmod | grep -qE "hid_sensor_hub|hid_sensor_iio_common" && echo "  ok  hid-sensor stack" || { echo "  MISS hid-sensor stack"; fail=1; }
lsmod | grep -qE "hid_sensor_accel_3d" && echo "  ok  accel_3d"      || { echo "  MISS hid_sensor_accel_3d"; fail=1; }
lsmod | grep -qE "hid_sensor_gyro_3d"  && echo "  ok  gyro_3d"       || { echo "  MISS hid_sensor_gyro_3d";  fail=1; }

echo "==> 2. Device enumeration"
if command -v rs-enumerate-devices >/dev/null; then
    timeout 25 rs-enumerate-devices 2>&1 | grep -E "Name|Serial Number|Firmware Version|Motion Module" \
        && echo "  ok  device responded" \
        || { echo "  FAIL no device / enumeration error"; fail=1; }
else
    echo "  FAIL rs-enumerate-devices not in PATH (librealsense not installed?)"; fail=1
fi

echo "==> 3. IMU IIO devices"
if ls /sys/bus/iio/devices/ >/dev/null 2>&1; then
    cat /sys/bus/iio/devices/iio:device*/name 2>/dev/null | sed 's/^/    /'
    cat /sys/bus/iio/devices/iio:device*/name 2>/dev/null | grep -q accel_3d && echo "  ok  accel_3d present" || { echo "  MISS accel_3d"; fail=1; }
    cat /sys/bus/iio/devices/iio:device*/name 2>/dev/null | grep -q gyro_3d  && echo "  ok  gyro_3d present"  || { echo "  MISS gyro_3d";  fail=1; }
else
    echo "  FAIL no IIO devices"; fail=1
fi

echo "==> 4. RT latency (30 s cyclictest — needs sudo; target max < 1 ms)"
if command -v cyclictest >/dev/null; then
    sudo cyclictest --mlockall --smp --priority=80 --interval=1000 --distance=0 \
        --duration=30s --quiet | tail -5
else
    echo "  SKIP cyclictest not installed (apt install rt-tests)"
fi

echo
if [[ "$fail" == "0" ]]; then
    echo "==> PASS (also do the manual realsense-viewer stream-under-load check; see notes/)"
else
    echo "==> FAILURES above — investigate before trusting the stack"
fi
exit "$fail"
