#!/usr/bin/env bash
# Overnight driver: train both digging runs, then score them while nobody is watching.
#
# Scoring is the part that is easy to forget and expensive to redo, because the
# useful checkpoint is usually NOT the last one — on the previous sweep the best
# held-out checkpoint was step 17500-20000 of 30000. eval_curve scores every saved
# checkpoint so the morning starts with a curve instead of a guess.
#
#   nohup bash queue_digging.sh > outputs/digging/queue.log 2>&1 &
#
# Everything is resumable: re-running skips finished training and redoes the eval.

set -uo pipefail

ROOT=/home/masi-pgx/spark-projects/smolvla-spark-finetune
HERE=$ROOT/excavator
VENV=$ROOT/.venv/bin
OUT=$ROOT/outputs/digging

mkdir -p "$OUT"

echo "=== queue start $(date) ==="
bash "$HERE/run_digging.sh" ir both
rc=$?
echo "=== training done (rc=$rc) $(date) ==="

# Score even if one run failed — a single finished model is still worth reading.
cd "$HERE" || exit 1

echo "=== eval_compare (last checkpoint) $(date) ==="
"$VENV/python" eval_compare.py --preset digging --horizons 1.5 4.0 \
    > "$OUT/RESULTS.txt" 2>&1
echo "rc=$?"

echo "=== eval_curve (every checkpoint) $(date) ==="
"$VENV/python" eval_curve.py --preset digging --horizon 1.5 \
    > "$OUT/CURVE.txt" 2>&1
echo "rc=$?"

# Drop the readable artefacts somewhere the user will actually see them.
for f in RESULTS.txt CURVE.txt curve.png; do
    [ -f "$OUT/$f" ] && cp "$OUT/$f" "/home/masi-pgx/Desktop/digging-$f"
done

echo "=== queue done $(date) ==="
tail -30 "$OUT/RESULTS.txt" 2>/dev/null
