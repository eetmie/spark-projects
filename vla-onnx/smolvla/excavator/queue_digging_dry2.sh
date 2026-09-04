#!/usr/bin/env bash
# Score the two-task dry_2 run as soon as it releases the GPU.
#
#   setsid nohup bash queue_digging_dry2.sh > outputs/digging_dry2/queue.log 2>&1 &
#
# HORIZONS. Chunk 10 @30fps covers 0.333 s, and eval_compare SILENTLY SKIPS any run whose
# chunk is shorter than the horizon (`if covered + 1e-6 < h: continue`) -- ask for 0.5 s or
# 1.5 s here and the run vanishes from the table, leaving only the baselines and no error.
# So, as with the first dry run, two horizons and no more:
#
#   0.17 s  what the machine actually executes -- inference settled at ~5 chunk steps
#           (2026-08-28 measurement). This is the number that matters.
#   0.30 s  continuity with every earlier digging table, and the longest horizon
#           chunk 10 can cover at all.
#
# TWO TASKS, THREE TABLES. This is the first recording carrying two instructions
# ("move sand to container", eps 0-62; "move rock to container", eps 63-77). The combined
# table blends them, and rock is only 87 of 409 eval points -- a model that learned sand
# and ignored rock would still look respectable there. So the sand and rock halves are
# also scored on their own. They use the same run, the same checkpoint and the same
# held-out episodes; only the episode subset differs, so the three tables decompose.
# Rock rests on 2 held-out episodes: a smoke test, not a tight estimate.
#
# COMPARABILITY. Not one-variable against digging_dry (62 eps, sand only) and not against
# the digging_clean sweep either: different session, different material, an added task.
# Read it against its own zero-action baseline.
#
# HOW THIS WAITS, AND WHY NOT pgrep. The obvious `while pgrep -f "lerobot-train"` is broken
# here and cost the first dry run its overnight scoring: pgrep -f matches ANY process with
# that string in its argv, and an interactive agent session tailing the run leaves a trail
# of shells whose command lines contain it. Training exited and the queue was still
# "waiting for the GPU" minutes later, with nothing wrong and nothing logged. So wait on
# the LAUNCHER'S OWN MARKER: run_digging.sh prints "digging sweep done" to launch.log after
# its last run returns, pass or fail. That is a fact about this run, not a guess about the
# process table. The pgrep pass after it is belt-and-braces, with agent shells filtered out.

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../paths.sh"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$ROOT/outputs/digging_dry2
VENV=$VENV_LEROBOT051/bin
echo "=== dry2 scoring queue start $(date) ==="
echo "waiting for the chunk-10 two-task run to release the GPU ..."
while ! grep -q "digging sweep done" "$OUT/launch.log" 2>/dev/null; do sleep 60; done
while pgrep -af "lerobot-train" 2>/dev/null \
      | grep -vE "shell-snapshots|queue_digging_dry2|pgrep" | grep -q .; do sleep 60; done
sleep 30
echo "[$(date +%T)] GPU clear"

cd "$ROOT/excavator" || exit 1

echo "=== eval_compare, both tasks @0.17s + 0.3s $(date) ==="
"$VENV/python" eval_compare.py --preset digging_dry2 --horizons 0.17 0.3 \
    --json-out "$OUT/comparison.json" > "$OUT/RESULTS.txt" 2>&1
echo "rc=$?"

for half in sand rock; do
    echo "=== eval_compare, $half only @0.17s + 0.3s $(date) ==="
    "$VENV/python" eval_compare.py --preset "digging_dry2_$half" --horizons 0.17 0.3 \
        --json-out "$OUT/comparison-$half.json" > "$OUT/RESULTS-$half.txt" 2>&1
    echo "rc=$?"
done

echo "=== eval_curve @0.17s, every checkpoint $(date) ==="
"$VENV/python" eval_curve.py --preset digging_dry2 --out-dir "$OUT" --horizon 0.17 \
    > "$OUT/CURVE.txt" 2>&1
echo "rc=$?"

for f in RESULTS.txt RESULTS-sand.txt RESULTS-rock.txt CURVE.txt curve.png; do
    [ -f "$OUT/$f" ] && cp "$OUT/$f" "$VLA_DATASETS/digging-dry2-$f"
done

echo "=== dry2 scoring queue done $(date) ==="
echo; echo "########## COMBINED ##########"; tail -32 "$OUT/RESULTS.txt" 2>/dev/null
echo; echo "########## SAND ##########";     tail -14 "$OUT/RESULTS-sand.txt" 2>/dev/null
echo; echo "########## ROCK ##########";     tail -14 "$OUT/RESULTS-rock.txt" 2>/dev/null
