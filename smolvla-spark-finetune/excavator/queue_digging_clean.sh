#!/usr/bin/env bash
# Queue the cleaned-dataset runs behind the full-dataset IR run already training, then
# score everything against everything.
#
#   setsid nohup bash queue_digging_clean.sh > outputs/digging_clean/queue.log 2>&1 &
#
#   1. digging189/ir       (already running)  full data,    IR, chunk 50  -- the baseline
#   2. digging_clean/ir                       cleaned data, IR, chunk 50  -- tests CLEANING
#   3. digging_clean/both                     cleaned data, 2 cams, ch 50 -- tests CAMERAS
#   4. digging_clean/clean_ir12               cleaned data, IR, chunk 12  -- tests CHUNK
#
# Four runs, three one-variable comparisons: (2)vs(1) dataset, (3)vs(2) cameras,
# (4)vs(2) chunk size. Runs 1-3 stay at chunk 50 deliberately -- the baseline is already
# training at 50 and cannot change, so moving 2 or 3 would confound the dataset and
# camera questions with a chunk change.
#
# WHY CHUNK 12 IS WORTH A RUN. run_inference hands the controller the whole chunk and
# replans as soon as the next inference lands, so what actually EXECUTES is
# ~infer_s * fps. Measured on the machine over 5 minutes: never more than 9 steps. At
# chunk 50 that means 82% of every predicted chunk is discarded while the training loss
# is spread uniformly over all 50 -- the steps that drive valves get under a fifth of the
# gradient. Chunk 12 concentrates all of it on 0.4 s.
#
# The risk is latency margin, and it is real: 0.4 s of buffered plan against ~0.3 s
# inference is only 1.3x. One slow inference and the setpoint scheduler falls into
# hold(0.25 s) -> decay(0.25 s). That is a DEPLOYMENT concern, not a scoring one -- this
# run answers whether the accuracy is there; whether to drive with it is a separate call.
#
# SCORING HORIZONS. 0.3 s is what the machine actually executes (9 steps) and is the only
# horizon every run can be scored at: chunk 12 covers 0.4 s, so a 0.5 s horizon silently
# DROPS it from the table -- the same failure that emptied the 4.0 s tables earlier.
# 0.5 s and 1.5 s are kept for continuity with every earlier number in this project; the
# chunk-12 run simply will not appear in those two.
#
# The cleaned dataset dropped episodes 83-90, which RENUMBERS everything after them, so
# "every 10th from 5" would hold out different recordings than the baseline and the
# comparison would be meaningless. VAL_EPISODES below is the mapped set (x if x < 83 else
# x-8): the same recordings digging189 held out, less source ep 85 which fell inside the
# dropped block.
#
# Match on "lerobot-train", never on a pattern that also appears in this script's own
# command line -- pgrep -f happily matches the waiting shell itself and loops forever.

set -uo pipefail

ROOT=/home/masi-pgx/spark-projects/smolvla-spark-finetune
OUT=$ROOT/outputs/digging_clean
VENV=$ROOT/.venv/bin
BASE=$ROOT/outputs/digging189/ir      # full-dataset IR baseline, pulled into the table

export VAL_EPISODES="5 15 25 35 45 55 65 75 87 97 107 117 127 137 147 157 167 177"

mkdir -p "$OUT"
echo "=== queue start $(date) ==="
echo "held out (cleaned ids): $VAL_EPISODES"
echo "waiting for the full-dataset IR run to release the GPU ..."
while pgrep -f "lerobot-train" > /dev/null 2>&1; do sleep 60; done
sleep 30
echo "[$(date +%T)] GPU clear"

run_one() {   # name, chunk
    echo "[$(date +%T)] === starting $1 (chunk $2) ==="
    OUT=$OUT STEPS=50000 SAVE_FREQ=2500 CHUNK=$2 \
        bash "$ROOT/excavator/run_digging.sh" "$1"
    echo "[$(date +%T)] === $1 done (rc=$?) ==="
}

run_one clean_ir   50
run_one clean_both 50
run_one clean_ir12 12

# Score even if a run failed -- a finished model is still worth reading.
cd "$ROOT/excavator" || exit 1

echo "=== eval_compare (0.3s = what the machine executes; 0.5/1.5 for continuity) $(date) ==="
"$VENV/python" eval_compare.py --preset digging_clean --horizons 0.3 0.5 1.5 \
    --extra-runs "full_ir=$BASE" > "$OUT/RESULTS.txt" 2>&1
echo "rc=$?"

echo "=== eval_curve @0.3s (every checkpoint, all four runs) $(date) ==="
"$VENV/python" eval_curve.py --preset digging_clean --horizon 0.3 > "$OUT/CURVE.txt" 2>&1
echo "rc=$?"

echo "=== eval_curve @0.5s (chunk-50 runs only; chunk 12 cannot cover it) $(date) ==="
"$VENV/python" eval_curve.py --preset digging_clean --horizon 0.5 \
    --runs clean_ir clean_both > "$OUT/CURVE_0p5.txt" 2>&1
echo "rc=$?"

for f in RESULTS.txt CURVE.txt CURVE_0p5.txt curve.png; do
    [ -f "$OUT/$f" ] && cp "$OUT/$f" "/home/masi-pgx/Desktop/digging-clean-$f"
done

echo "=== queue done $(date) ==="
tail -60 "$OUT/RESULTS.txt" 2>/dev/null
