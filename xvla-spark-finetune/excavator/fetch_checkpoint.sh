#!/usr/bin/env bash
# Fetch lerobot/xvla-base (3.52 GB) into models/xvla-base/.
#
# Why curl and not huggingface_hub: on this box `snapshot_download` stalled twice partway
# through the safetensors — the process stayed alive, the .incomplete file stopped growing
# (dead at 245 and 257 MB of 3519), and no exception was ever raised, so the built-in retry
# never fired. curl with `--speed-limit/--speed-time` treats a stall as an error, which lets
# `--retry` actually restart it, and `-C -` resumes instead of starting over. Measured ~9 MB/s
# against ~4 MB/s for the python client before it hung.
#
# Re-running is safe and cheap: finished files resume to a no-op.

set -uo pipefail

DST=${1:-/home/masi-pgx/spark-projects/xvla-spark-finetune/models/xvla-base}
BASE=https://huggingface.co/lerobot/xvla-base/resolve/main

mkdir -p "$DST"
for f in config.json policy_preprocessor.json policy_postprocessor.json model.safetensors; do
    echo "=== $f ==="
    curl -L --fail --retry 20 --retry-delay 5 --retry-all-errors \
         --speed-limit 102400 --speed-time 30 \
         -C - -o "$DST/$f" "$BASE/$f" 2>&1 | tail -2
    echo "  -> $(stat -c %s "$DST/$f" 2>/dev/null || echo MISSING) bytes"
done

echo
echo "expected: model.safetensors ~3519 MB, config.json ~5463 B"
ls -la "$DST"
