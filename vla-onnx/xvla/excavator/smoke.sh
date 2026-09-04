#!/usr/bin/env bash
# Run a disposable training smoke test without writing a multi-gigabyte checkpoint.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
RUN=${1:-clean_ir}
STAMP=$(date +%Y%m%d-%H%M%S)
SMOKE_OUT=$ROOT/outputs/smoke/$STAMP

STEPS=${STEPS:-2} BATCH=${BATCH:-2} WORKERS=${WORKERS:-2} LOG_FREQ=1 \
SAVE_FREQ=100000 SAVE_CHECKPOINT=false OUT="$SMOKE_OUT" \
    bash "$HERE/run_digging.sh" "$RUN"

LOG=$SMOKE_OUT/logs/$RUN.log
if ! grep -q "End of training" "$LOG"; then
    echo "smoke test did not reach End of training: $LOG" >&2
    exit 1
fi

echo "smoke test passed: $LOG"
