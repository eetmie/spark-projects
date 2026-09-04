#!/usr/bin/env bash
# Score the dry-sand chunk-10 run as soon as it releases the GPU.
#
#   setsid nohup bash queue_digging_dry.sh > outputs/digging_dry/queue.log 2>&1 &
#
# HORIZONS. Chunk 10 @30fps covers 0.333 s, and eval_compare SILENTLY SKIPS any run
# whose chunk is shorter than the horizon (`if covered + 1e-6 < h: continue`) -- ask for
# 0.5 s or 1.5 s here and the run vanishes from the table, leaving only the baselines and
# no error. So this scores two horizons and no more:
#
#   0.17 s  what the machine now actually executes -- inference settled at ~5 chunk
#           steps, per the 2026-08-28 measurement. This is the number that matters.
#   0.30 s  continuity with every earlier digging table (the old 9-step figure), and
#           the longest horizon chunk 10 can cover at all.
#
# The 0.5/1.5 s columns of the five-way sweep have no counterpart here and cannot get
# one without retraining at a longer chunk.
#
# NOT COMPARABLE ONE-VARIABLE to the digging_clean sweep: different sand condition,
# different recording session, and a different task string ("move sand to container"
# vs "move the sand to the container"). Read it as a standalone number against the
# zero-action baseline, not as chunk 10 beating chunk 12.
#
# HOW THIS WAITS, AND WHY NOT pgrep. The obvious `while pgrep -f "lerobot-train"` is
# broken here and cost this run its overnight scoring: pgrep -f matches ANY process with
# that string in its argv, and an interactive agent session tailing the run leaves a
# trail of shells whose command lines contain it. Training exited 01:57:27 and the queue
# was still "waiting for the GPU" minutes later, with nothing wrong and nothing logged.
# Excluding this script's own name -- the guard the other queue scripts carry -- does not
# help, because the matches are not this script.
#
# So wait on the LAUNCHER'"'"'S OWN MARKER instead. run_digging.sh prints "digging sweep done"
# to launch.log after its last run returns, whether that run passed or failed. It is a
# fact about this training run, not a guess about the process table, and no unrelated
# command line can forge it. The pgrep pass that follows is a belt-and-braces check with
# the agent/harness shells filtered out.

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../paths.sh"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$ROOT/outputs/digging_dry
VENV=$VENV_LEROBOT051/bin
echo "=== dry scoring queue start $(date) ==="
echo "waiting for the chunk-10 run to release the GPU ..."
while ! grep -q "digging sweep done" "$OUT/launch.log" 2>/dev/null; do sleep 60; done
while pgrep -af "lerobot-train" 2>/dev/null \
      | grep -vE "shell-snapshots|queue_digging_dry|pgrep" | grep -q .; do sleep 60; done
sleep 30
echo "[$(date +%T)] GPU clear"

cd "$ROOT/excavator" || exit 1

echo "=== eval_compare @0.17s (executed) + 0.3s (continuity) $(date) ==="
"$VENV/python" eval_compare.py --preset digging_dry --horizons 0.17 0.3 \
    > "$OUT/RESULTS.txt" 2>&1
echo "rc=$?"

echo "=== eval_curve @0.17s, every checkpoint $(date) ==="
"$VENV/python" eval_curve.py --preset digging_dry --out-dir "$OUT" --horizon 0.17 \
    > "$OUT/CURVE.txt" 2>&1
echo "rc=$?"

for f in RESULTS.txt CURVE.txt curve.png; do
    [ -f "$OUT/$f" ] && cp "$OUT/$f" "$VLA_DATASETS/digging-dry-$f"
done

echo "=== dry scoring queue done $(date) ==="
tail -40 "$OUT/RESULTS.txt" 2>/dev/null
