#!/usr/bin/env bash
# Queue the two CLEANED-dataset runs behind the full-dataset IR run already training,
# then score all three against each other.
#
#   setsid nohup bash queue_digging_clean.sh > outputs/digging_clean/queue.log 2>&1 &
#
# The plan, and why it is in this order:
#
#   1. digging189/ir      (already running)  full dataset, IR only     -- the baseline
#   2. digging_clean/ir                      cleaned dataset, IR only  -- tests CLEANING
#   3. digging_clean/both                    cleaned dataset, 2 cams   -- tests CAMERAS
#
# That is two clean one-variable comparisons instead of one. (2) vs (1) changes only the
# dataset; (3) vs (2) changes only the camera count, and does it on the better dataset.
# The earlier plan trained the 2-camera model on the untrimmed data, which would have
# answered the camera question on data we now know carries ~9% dead air and a block of
# dirty episodes.
#
# The cleaned dataset dropped episodes 83-90, which RENUMBERS everything after them, so
# "every 10th from 5" would hold out different recordings than the baseline did and the
# comparison would be meaningless. VAL_EPISODES below is the mapped set: the same
# recordings digging189 held out, less source ep 85 which fell inside the dropped block.
# Mapping is x if x < 83 else x-8, verified by exact frame match.
#
# Match on "lerobot-train", never on a pattern that also appears in this script's own
# command line -- pgrep -f happily matches the waiting shell itself and loops forever.

set -uo pipefail

ROOT=/home/masi-pgx/spark-projects/smolvla-spark-finetune
OUT=$ROOT/outputs/digging_clean
VENV=$ROOT/.venv/bin
BASE=$ROOT/outputs/digging189/ir      # the full-dataset IR baseline, for the 3-way table

export VAL_EPISODES="5 15 25 35 45 55 65 75 87 97 107 117 127 137 147 157 167 177"

mkdir -p "$OUT"
echo "=== queue start $(date) ==="
echo "held out (cleaned ids): $VAL_EPISODES"
echo "waiting for the full-dataset IR run to release the GPU ..."
while pgrep -f "lerobot-train" > /dev/null 2>&1; do sleep 60; done
sleep 30
echo "[$(date +%T)] GPU clear"

for run in clean_ir clean_both; do
    echo "[$(date +%T)] === starting $run ==="
    OUT=$OUT STEPS=50000 SAVE_FREQ=2500 bash "$ROOT/excavator/run_digging.sh" "$run"
    echo "[$(date +%T)] === $run done (rc=$?) ==="
done

# Score even if a run failed -- a single finished model is still worth reading.
cd "$ROOT/excavator" || exit 1

# 0.5 s is the DEPLOYMENT horizon: run_inference plays whatever fits in one inference
# (~12 steps = 0.4 s at 30 fps), not the whole 50-step chunk. 1.5 s is kept because every
# earlier number in this project is at 1.5 s. No 4.0 s -- chunk 50 covers only 1.67 s, so
# a 4.0 s horizon silently drops every model and prints baselines alone.
echo "=== eval_compare: cleaned ir vs cleaned both vs full-dataset ir $(date) ==="
"$VENV/python" eval_compare.py --preset digging_clean --horizons 0.5 1.5 \
    --extra-runs "full_ir=$BASE" > "$OUT/RESULTS.txt" 2>&1
echo "rc=$?"

echo "=== eval_curve @0.5s (every checkpoint) $(date) ==="
"$VENV/python" eval_curve.py --preset digging_clean --horizon 0.5 > "$OUT/CURVE.txt" 2>&1
echo "rc=$?"

echo "=== eval_curve @1.5s $(date) ==="
"$VENV/python" eval_curve.py --preset digging_clean --horizon 1.5 > "$OUT/CURVE_1p5.txt" 2>&1
echo "rc=$?"

for f in RESULTS.txt CURVE.txt CURVE_1p5.txt curve.png; do
    [ -f "$OUT/$f" ] && cp "$OUT/$f" "/home/masi-pgx/Desktop/digging-clean-$f"
done

echo "=== queue done $(date) ==="
tail -45 "$OUT/RESULTS.txt" 2>/dev/null
