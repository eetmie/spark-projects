#!/usr/bin/env bash
# Train one X-VLA run, score its last checkpoint, then score every saved checkpoint.
# The recommended first architecture comparison is cleaned IR, chunk 30.

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
SMOL=$(cd "$ROOT/../smolvla-spark-finetune" && pwd)
VENV=$SMOL/.venv/bin
RUN=${1:-clean_ir}

STEPS=${STEPS:-30000}
CHUNK=${CHUNK:-}
SAVE_FREQ=${SAVE_FREQ:-5000}

case "$RUN" in
  ir)
    PRESET=digging189
    HORIZON=0.5
    CHUNK=${CHUNK:-50}
    REFERENCE=$SMOL/outputs/digging189/ir
    ;;
  both)
    PRESET=digging189
    HORIZON=0.5
    CHUNK=${CHUNK:-50}
    REFERENCE=$SMOL/outputs/digging189/both
    ;;
  clean_ir)
    PRESET=digging_clean
    HORIZON=0.5
    CHUNK=${CHUNK:-30}
    case "$CHUNK" in
      12) REFERENCE=$SMOL/outputs/digging_clean/clean_ir12 ;;
      30) REFERENCE=$SMOL/outputs/digging_clean30/clean_ir ;;
      50) REFERENCE=$SMOL/outputs/digging_clean/clean_ir ;;
      *)  echo "no matched SmolVLA clean_ir reference for chunk $CHUNK" >&2; exit 2 ;;
    esac
    ;;
  clean_both)
    PRESET=digging_clean
    HORIZON=0.5
    CHUNK=${CHUNK:-50}
    if [ "$CHUNK" != 50 ]; then
      echo "no matched SmolVLA clean_both reference for chunk $CHUNK" >&2
      exit 2
    fi
    REFERENCE=$SMOL/outputs/digging_clean/clean_both
    ;;
  dry_ir)
    PRESET=digging_dry
    HORIZON=0.3
    CHUNK=${CHUNK:-10}
    if [ "$CHUNK" != 10 ]; then
      echo "no matched SmolVLA dry_ir reference for chunk $CHUNK" >&2
      exit 2
    fi
    REFERENCE=$SMOL/outputs/digging_dry/dry_ir
    ;;
  *) echo "unknown run '$RUN'" >&2; exit 2 ;;
esac

case "$RUN" in
  clean_ir|clean_both)
    # Same source recordings as the SmolVLA cleaned-data runs after 83-90 were
    # dropped and all later episodes renumbered.
    export VAL_EPISODES="5 15 25 35 45 55 65 75 87 97 107 117 127 137 147 157 167 177"
    ;;
esac

if [ ! -d "$REFERENCE/checkpoints/last" ]; then
    echo "matched SmolVLA reference is missing: $REFERENCE" >&2
    exit 1
fi

OUT=${OUT:-$ROOT/outputs/${RUN}_chunk${CHUNK}}
mkdir -p "$OUT"
echo "=== X-VLA queue start $(date) ==="
"$VENV/python" "$HERE/preflight.py" --run "$RUN" --steps "$STEPS" --save-freq "$SAVE_FREQ" || exit 1

OUT="$OUT" STEPS="$STEPS" CHUNK="$CHUNK" SAVE_FREQ="$SAVE_FREQ" \
    bash "$HERE/run_digging.sh" "$RUN"
rc=$?
echo "=== training done (rc=$rc) $(date) ==="

if [ ! -d "$OUT/$RUN/checkpoints/last" ]; then
    echo "no checkpoint to evaluate under $OUT/$RUN" >&2
    exit "$rc"
fi

cd "$SMOL/excavator" || exit 1

echo "=== held-out last-checkpoint comparison $(date) ==="
"$VENV/python" eval_compare.py --preset "$PRESET" --runs \
    --horizons "$HORIZON" \
    --extra-runs "smolvla_reference=$REFERENCE" "xvla_${RUN}=$OUT/$RUN" \
    --json-out "$OUT/comparison.json" > "$OUT/RESULTS.txt" 2>&1
echo "rc=$?"

echo "=== held-out checkpoint curve $(date) ==="
"$VENV/python" eval_curve.py --preset "$PRESET" --out-dir "$OUT" \
    --runs "$RUN" --horizon "$HORIZON" > "$OUT/CURVE.txt" 2>&1
echo "rc=$?"

echo "=== X-VLA queue done $(date) ==="
tail -40 "$OUT/RESULTS.txt" 2>/dev/null
