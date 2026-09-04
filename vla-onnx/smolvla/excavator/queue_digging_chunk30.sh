#!/usr/bin/env bash
# Follow-on: add a chunk-30 cleaned-IR run so the chunk sweep is 12 / 30 / 50, then
# re-score everything in one five-way table.
#
#   setsid nohup bash queue_digging_chunk30.sh > outputs/digging_clean30/queue.log 2>&1 &
#
# Waits for queue_digging_clean.sh to finish COMPLETELY (its own eval included) rather
# than just for the GPU, so the two queues never interleave. That means the four-way
# results land first and this adds the fifth on top.
#
# Uses OUT + CHUNK overrides on the existing clean_ir target instead of adding a new one:
# run_digging.sh is executing right now for the chunk-12 run, and inserting bytes into a
# script bash is partway through corrupts its read offset. Nothing here edits it.
#
# Match on the OTHER script's path -- this file's own name must not appear in the
# pattern, or pgrep -f matches the waiting shell itself and loops forever.

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../paths.sh"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$ROOT/outputs/digging_clean30
VENV=$VENV_LEROBOT051/bin
CLEAN=$ROOT/outputs/digging_clean
BASE=$ROOT/outputs/digging189/ir

export VAL_EPISODES="5 15 25 35 45 55 65 75 87 97 107 117 127 137 147 157 167 177"

mkdir -p "$OUT"
echo "=== chunk-30 follow-on start $(date) ==="
echo "waiting for the four-run queue (training + its eval) to finish ..."
while pgrep -f "queue_digging_clean.sh" > /dev/null 2>&1; do sleep 120; done
while pgrep -f "lerobot-train" > /dev/null 2>&1; do sleep 60; done
sleep 30
echo "[$(date +%T)] GPU clear, starting clean_ir at chunk 30"

OUT=$OUT STEPS=50000 SAVE_FREQ=2500 CHUNK=30 \
    bash "$ROOT/excavator/run_digging.sh" clean_ir
echo "[$(date +%T)] === chunk-30 run done (rc=$?) ==="

cd "$ROOT/excavator" || exit 1

# Five-way. 0.3 s is the only horizon every run covers: chunk 12 spans 0.4 s, chunk 30
# spans 1.0 s, chunk 50 spans 1.67 s. At 0.5 s the chunk-12 run drops out; at 1.5 s both
# chunk-12 and chunk-30 drop out. They vanish silently rather than erroring, so read the
# 0.3 s table for anything involving chunk size.
echo "=== eval_compare, five-way $(date) ==="
"$VENV/python" eval_compare.py --preset digging_clean --horizons 0.3 0.5 1.5 \
    --extra-runs "full_ir=$BASE" "clean_ir30=$OUT/clean_ir" \
    > "$OUT/RESULTS5.txt" 2>&1
echo "rc=$?"

echo "=== eval_curve @0.3s for the chunk-30 run $(date) ==="
"$VENV/python" eval_curve.py --preset digging_clean --out-dir "$OUT" --horizon 0.3 \
    > "$OUT/CURVE30.txt" 2>&1
echo "rc=$?"

for f in RESULTS5.txt CURVE30.txt; do
    [ -f "$OUT/$f" ] && cp "$OUT/$f" "$VLA_DATASETS/digging-$f"
done

echo "=== chunk-30 follow-on done $(date) ==="
tail -50 "$OUT/RESULTS5.txt" 2>/dev/null
