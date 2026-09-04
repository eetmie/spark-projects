#!/usr/bin/env bash
# Start the first real X-VLA fine-tune once the SmolVLA side is completely off the GPU.
#
#   setsid nohup bash chain_xvla_dry2.sh > outputs/digging_dry2/chain.log 2>&1 &
#
# WAITS ON A PID, not a log marker. The SmolVLA export chain has four exit paths (three
# failures and one success) and only one of them prints "done"; keying on that marker
# would leave this run un-launched forever if the export failed for an unrelated reason.
# The X-VLA fine-tune does not depend on that export succeeding -- only on the GPU being
# free -- so the condition is simply "that process is gone". Not pgrep: a pattern loose
# enough to match the chain also matches agent shells that merely mention it, which is
# the trap documented at length in the SmolVLA queue script.
SMOLVLA_CHAIN_PID=2097647

set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$ROOT/outputs/digging_dry2

echo "=== xvla dry2 chain start $(date) ==="
if [ "$SMOLVLA_CHAIN_PID" != "0" ]; then
    echo "waiting for the SmolVLA export chain (pid $SMOLVLA_CHAIN_PID) to exit ..."
    while [ -d "/proc/$SMOLVLA_CHAIN_PID" ]; do sleep 60; done
fi
# That chain exits the moment the export finishes; give the GPU a breath to settle.
sleep 60
echo "[$(date +%T)] GPU free"

# Gate on a 2-step smoke before committing ~5 h. smoke.sh writes to a disposable dir with
# SAVE_CHECKPOINT=false, so it cannot leave behind the 250-step probe checkpoint that
# silently no-opped a resume once before.
echo "=== smoke (2 steps) $(date) ==="
if ! bash "$ROOT/excavator/smoke.sh" dry2_ir; then
    echo "!! smoke test FAILED — not launching the 10000-step run. See outputs/smoke/."
    exit 1
fi

echo "=== real run: dry2_ir, 10000 steps, chunk 30, bf16 $(date) ==="
OUT="$OUT" STEPS=10000 CHUNK=30 SAVE_FREQ=2500 WORKERS=10 \
    bash "$ROOT/excavator/run_digging.sh" dry2_ir
echo "run rc=$?"
echo "=== xvla dry2 chain done $(date) ==="
