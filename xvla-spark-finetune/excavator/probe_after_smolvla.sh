#!/usr/bin/env bash
# Wait for the SmolVLA digging queue to release the GPU, then measure X-VLA's step cost.
#
# The point is NOT to train anything — it is to answer "what does an X-VLA overnight run
# actually cost on this box" with a measured number instead of a guess, so the real run can
# be sized in the morning. 250 steps, no checkpoints.
#
# It matters that this runs AFTER the SmolVLA queue rather than beside it: two trainings
# on one GB10 do not go twice as fast, they contend, and the SmolVLA numbers are the
# deliverable tonight. It also matters that the measurement is taken heat-soaked — the
# SmolVLA probe read 0.507 s/step on a cold GPU and the real run settled at 0.55-0.63 once
# the board reached ~79 C and clocked down to 2457 of 3003 MHz. Running straight after an
# 8-hour job means this probe reads the honest number by construction.
#
#   setsid nohup bash probe_after_smolvla.sh > outputs/probe_after.log 2>&1 &
#
# Match on the script PATH, never on a pattern that also appears in this script's own
# command line — `pgrep -f` happily matches the waiting shell itself and loops forever.

set -uo pipefail

ROOT=/home/masi-pgx/spark-projects/xvla-spark-finetune
# The queue was launched as `bash excavator/queue_digging.sh` from the SmolVLA project, so
# its command line carries the RELATIVE path — matching on the absolute one finds nothing
# and the probe would start immediately, on top of a running job.
SMOLVLA_QUEUE="excavator/queue_digging.sh"
PROBE_LOG=$ROOT/outputs/digging/logs/ir.log

echo "[$(date +%T)] waiting for the SmolVLA queue to finish ..."
while pgrep -f "$SMOLVLA_QUEUE" > /dev/null 2>&1; do sleep 60; done
echo "[$(date +%T)] SmolVLA queue gone"

# Do not start on top of a still-draining trainer.
while pgrep -f "lerobot-train" > /dev/null 2>&1; do sleep 30; done
sleep 30
echo "[$(date +%T)] GPU clear, starting X-VLA probe"

cd "$ROOT" || exit 1
STEPS=250 SAVE_FREQ=100000 bash excavator/run_digging.sh ir
rc=$?

echo "[$(date +%T)] probe finished rc=$rc"
echo "--- measured step cost (updt_s = optimizer step, data_s = dataloader wait) ---"
grep -oE "step:[0-9]+ .*updt_s:[0-9.]+ data_s:[0-9.]+" "$PROBE_LOG" 2>/dev/null | tail -5
grep -E "Traceback|Error|OutOfMemory|CUDA out of memory" "$PROBE_LOG" 2>/dev/null | head -5

echo
echo "To size a real run: steps x updt_s / 3600 = hours. SmolVLA IR was 0.55-0.63 s/step."
