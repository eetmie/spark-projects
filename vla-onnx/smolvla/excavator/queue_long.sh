#!/usr/bin/env bash
# Waits for the 8k sweep (and its comparison) to finish, then runs the 30k sweep
# and scores every checkpoint. Chained so the GPU is never shared between two
# training runs, which would roughly halve both.

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../paths.sh"

PROJ=$ROOT
SHORT=$PROJ/outputs/excavator
LONG=$PROJ/outputs/excavator_long
DESKTOP=$VLA_DATASETS

mkdir -p "$LONG"
QLOG=$LONG/logs/queue.log
mkdir -p "$(dirname "$QLOG")"

{
  echo "queue started $(date)"

  # 1. the 8k sweep itself
  while ! grep -q "^sweep done" "$SHORT/logs/sweep.log" 2>/dev/null; do sleep 60; done
  echo "8k sweep finished $(date)"

  # 2. its comparison — wait for the watcher PROCESS to exit, not for RESULTS.txt
  # to exist (the shell creates that file at pipeline start, so a file check
  # passes immediately and the sweeps overlap on the GPU). Bounded, so a hung
  # watcher cannot block the night.
  waited=0
  while pgrep -f watch_and_compare.sh >/dev/null && [ $waited -lt 2400 ]; do sleep 60; waited=$((waited + 60)); done
  [ -s "$SHORT/RESULTS.txt" ] && echo "8k comparison done $(date)" || echo "8k comparison missing/empty, proceeding anyway $(date)"

  # 3. the long sweep
  echo "starting 30k sweep $(date)"
  bash "$PROJ/excavator/run_long.sh"

  # 4. score it: headline table at the final checkpoint, then the full curve
  echo "scoring $(date)"
  "$VENV_LEROBOT051/bin/python" "$PROJ/excavator/eval_compare.py" --out-dir "$LONG" > "$LONG/RESULTS.txt" 2>&1
  "$VENV_LEROBOT051/bin/python" "$PROJ/excavator/eval_curve.py" --out-dir "$LONG" > "$LONG/CURVE.txt" 2>&1

  cp "$LONG/RESULTS.txt" "$DESKTOP/smolvla-excavator-LONG-RESULTS.txt" 2>/dev/null
  cp "$LONG/CURVE.txt" "$DESKTOP/smolvla-excavator-LONG-CURVE.txt" 2>/dev/null
  cp "$LONG/curve.png" "$DESKTOP/smolvla-excavator-LONG-curve.png" 2>/dev/null
  echo "queue done $(date)"
} >> "$QLOG" 2>&1
