#!/usr/bin/env bash
# Waits for run_experiments.sh to finish, then runs the comparison and leaves the
# table where it is easy to find. Launched alongside the sweep so results are ready
# without anyone having to babysit it.

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../paths.sh"

PROJ=$ROOT
OUT=$PROJ/outputs/excavator
SWEEP_LOG=$OUT/logs/sweep.log
RESULT=$OUT/RESULTS.txt
DESKTOP_COPY=$VLA_DATASETS/smolvla-excavator-RESULTS.txt

while ! grep -q "^sweep done" "$SWEEP_LOG" 2>/dev/null; do
  sleep 60
done

{
  echo "SmolVLA excavator sweep — comparison"
  echo "generated $(date)"
  echo
  sed -n '1,200p' "$SWEEP_LOG"
  echo
  echo "final training loss per run:"
  for r in A B C D; do
    log=$OUT/logs/$r.log
    [ -f "$log" ] || continue
    # LeRobot abbreviates step counts over 999 as "8K", so the regex must allow K/M.
    last=$(tr '\r' '\n' < "$log" | grep -oE "step:[0-9.]+[KM]? .*" | tail -1)
    echo "  $r: ${last:-<no steps logged>}"
  done
  echo
  echo "========================================================================"
  "$VENV_LEROBOT051/bin/python" "$PROJ/excavator/eval_compare.py" 2>&1 \
    | grep -viE "^(warning|.*torch_dtype|Loading weights|Reducing the number)"

  echo
  echo "trajectory plots:"
  for ep in 3 11; do
    "$VENV_LEROBOT051/bin/python" "$PROJ/excavator/plot_predictions.py" --episode "$ep" 2>&1 \
      | grep -viE "^(warning|.*torch_dtype|Loading weights|Reducing the number)"
  done
} > "$RESULT" 2>&1

cp "$RESULT" "$DESKTOP_COPY"
cp "$OUT"/plots/*.png $VLA_DATASETS/ 2>/dev/null
echo "comparison written to $RESULT and $DESKTOP_COPY"
