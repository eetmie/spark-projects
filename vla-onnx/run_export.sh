#!/usr/bin/env bash
# Export a trained run as a split ONNX bundle, and prove it against the torch checkpoint.
#
#   ./run_export.sh --model smolvla --run smolvla/outputs/<sweep>/<run>
#   ./run_export.sh --model xvla --run xvla/outputs/<sweep>/<run> --step 17500
#   ./run_export.sh --model smolvla --run <dir> --step last --dest ~/Desktop/my-bundle
#
# Generalises the per-recording chain scripts (chain_export_dry2.sh and friends), which
# each hardcoded one sweep, one run name and one destination.
#
# THE BEST CHECKPOINT IS READ FROM THE EVAL, NEVER ASSUMED. `--step best` (the default)
# takes the minimum held-out disp_err from the run's curve.json. "Take the last one" is
# wrong on this project's own history: the kaivuri sweep peaked at 20000 of 30000 and the
# first dry run at 25000 of 50000, while the digging sweep did peak at its last step. If
# curve.json is missing or unreadable this EXPORTS NOTHING and says so -- a bundle
# silently built from the wrong checkpoint is worse than no bundle.
#
# THE EXPORT VENV IS NOT THE TRAINING VENV FOR X-VLA. It trains in .venv-lerobot051 and
# exports in .venv-lerobot061 (see the environments table in README.md). Getting this
# backwards produces an import error deep in the exporter rather than at the top.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/paths.sh"

die() { echo "!! $*" >&2; exit 2; }

MODEL="" RUN="" STEP=best DEST="" VIEWS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --run)   RUN=$2; shift 2 ;;
        --step)  STEP=$2; shift 2 ;;
        --dest)  DEST=$2; shift 2 ;;
        --views) VIEWS=$2; shift 2 ;;
        -h|--help)
            sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            cat <<'USAGE'

  --model M    smolvla | xvla | evo1                                   (required)
  --run DIR    a run directory holding checkpoints/                    (required)
  --step S     best (default, from curve.json) | last | a step number
  --dest DIR   bundle destination (default: $VLA_DATASETS/<model>-<sweep>-<run>-<step>)
  --views N    how many camera views the bundle is sized for (default: the checkpoint's)
USAGE
            exit 0 ;;
        *) die "unknown flag: $1" ;;
    esac
done
[ -n "$MODEL" ] || die "--model is required"
[ -n "$RUN" ]   || die "--run is required"

case "$MODEL" in
    # X-VLA and EVO1 export in the 0.6.1 stack; only SmolVLA exports where it trained.
    smolvla) VENV=$VENV_LEROBOT051; CKPT_FLAG=--model-id;   PARITY=parity_split_vs_torch.py ;;
    xvla)    VENV=$VENV_LEROBOT061; CKPT_FLAG=--checkpoint; PARITY=parity.py ;;
    evo1)    VENV=$VENV_LEROBOT061; CKPT_FLAG=--checkpoint; PARITY="" ;;
    *) die "unknown --model $MODEL (smolvla | xvla | evo1)" ;;
esac
PLAYBOOK=$VLA_ONNX/$MODEL
PY=$VENV/bin/python
[ -x "$PY" ] || die "no python in $VENV"

RUN=${RUN%/}
[ -d "$RUN/checkpoints" ] || die "$RUN has no checkpoints/ -- is that a run directory?"

# --- which step ----------------------------------------------------------------------
case "$STEP" in
    best)
        CURVE=$(dirname "$RUN")/curve.json
        [ -f "$CURVE" ] || die "no $CURVE -- score the sweep first:
     $PY $VLA_ONNX/eval/curve.py --sweep $(dirname "$RUN")
   Exporting nothing: the best checkpoint is read from the eval, never assumed."
        STEP=$("$PY" - "$CURVE" "$(basename "$RUN")" <<'PY'
import json, sys
curve = json.loads(open(sys.argv[1]).read())["curves"]
pts = curve.get(sys.argv[2])
if not pts:
    print(f"ERR curve.json has no points for {sys.argv[2]!r} "
          f"(has {sorted(curve)})", file=sys.stderr); sys.exit(1)
best = min(pts, key=lambda r: r["disp_err"])
print(best["step"])
print(f"best checkpoint: step {best['step']}  disp_err {best['disp_err']:.4f}  "
      f"(last: {pts[-1]['step']} {pts[-1]['disp_err']:.4f})", file=sys.stderr)
PY
        ) || die "could not read the best checkpoint from $CURVE -- exporting nothing."
        ;;
esac

if [ "$STEP" = last ]; then
    CKPT=$RUN/checkpoints/last/pretrained_model
    STEP=$(grep -oE "[0-9]+" "$RUN/checkpoints/last/training_state/training_step.json" 2>/dev/null | head -1)
else
    printf -v PADDED "%06d" "$STEP"
    CKPT=$RUN/checkpoints/$PADDED/pretrained_model
fi
[ -d "$CKPT" ] || die "$CKPT missing -- exporting nothing."

: "${DEST:=$VLA_DATASETS/$MODEL-$(basename "$(dirname "$RUN")")-$(basename "$RUN")-$STEP}"

echo "=== export $MODEL step $STEP -> $DEST  $(date) ==="
EXPORT_ARGS=("$CKPT_FLAG" "$CKPT" --out-dir "$DEST")
[ -n "$VIEWS" ] && EXPORT_ARGS+=(--views "$VIEWS")
"$PY" "$PLAYBOOK/export_split_onnx.py" "${EXPORT_ARGS[@]}"
rc=$?; [ $rc -eq 0 ] || die "export failed (rc=$rc), stopping before parity."

if [ -n "$PARITY" ] && [ -f "$PLAYBOOK/$PARITY" ]; then
    echo "=== parity vs the torch checkpoint  $(date) ==="
    PARITY_ARGS=(--split-dir "$DEST" "$CKPT_FLAG" "$CKPT")
    # X-VLA's parity caches its PyTorch reference in ONE shared npz by default and refuses
    # when that cache was emitted for a different bundle -- correctly, but it used to scroll
    # past inside `| tee` and the export still "succeeded" unverified. Keep the reference
    # with the bundle, and let a refusal or a FAIL stop the script.
    [ "$MODEL" = xvla ] && PARITY_ARGS+=(--reference "$DEST/parity_reference.npz")
    "$PY" "$PLAYBOOK/$PARITY" "${PARITY_ARGS[@]}" 2>&1 | tee "$DEST/PARITY.txt"
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] || die "parity did not pass (rc=$rc), see $DEST/PARITY.txt"
else
    echo "!! no parity script for $MODEL -- bundle is UNVERIFIED against torch" >&2
fi

echo "=== bundle ==="
ls -la "$DEST"
echo "  ship it:  $SPARK_PROJECTS/ship_bundle.sh $DEST orin"
