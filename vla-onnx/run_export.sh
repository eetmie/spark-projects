#!/usr/bin/env bash
# Export a trained run as a split ONNX bundle, and prove it against the torch checkpoint.
#
#   ./run_export.sh --run xvla/outputs/<sweep>/<run> --step 17500
#   ./run_export.sh --run xvla/outputs/<sweep>/<run> --step last --dest ~/bundles/my-bundle
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
# WHAT SHIPS IS WHAT THE ORIN RUNTIME LOADS. Since 2026-10-29 the robot runs X-VLA on native
# TensorRT (kaivuriprokkis-python lerobot_vla/runtime/xvla_native.py), which wants mixed-FP16
# graphs and refuses a bundle without a stock reference fixture. So, in order:
#   1. FP32 split export -> DEST.fp32 (work dir), parity vs the torch checkpoint
#   2. reference.py: fixture.npz + bundle.json "fixture" (the board's load gate)
#   3. fp16_weights.py: mixed FP16 -> DEST, parity again against the same reference
# Any failure stops the script; DEST is only complete when it says "=== bundle ===".
#
# One venv: .venv-lerobot061 trains, scores and exports (README.md, environments).
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/paths.sh"

die() { echo "!! $*" >&2; exit 2; }

MODEL=xvla RUN="" STEP=best DEST="" VIEWS=""
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

  --model xvla accepted for old command lines; X-VLA is the only model
  --run DIR    a run directory holding checkpoints/                    (required)
  --step S     best (default, from curve.json) | last | a step number
  --dest DIR   bundle destination (default: $VLA_DATASETS/<model>-<sweep>-<run>-<step>)
  --views N    how many camera views the bundle is sized for (default: the checkpoint's)
USAGE
            exit 0 ;;
        *) die "unknown flag: $1" ;;
    esac
done
[ "$MODEL" = xvla ] || die "--model $MODEL: only X-VLA is left here (git tag archive/lerobot-playbooks-2026-10-29 has the rest)"
[ -n "$RUN" ]   || die "--run is required"

VENV=$VENV_LEROBOT061; CKPT_FLAG=--checkpoint; PARITY=parity.py
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
   Trained with --holdout none? There is nothing to score: pass --step last or a number.
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

DEST=${DEST%/}
: "${DEST:=$VLA_DATASETS/$MODEL-$(basename "$(dirname "$RUN")")-$(basename "$RUN")-$STEP}"
# The FP32 export is a work dir, the FP16 bundle is what ships. The Orin runtime builds FP16
# engines either way, but from FP32 graphs it built them ~4.5x slower to run (1725 vs 382 ms
# a chunk, dual-camera run 2026-11-02) and twice the size. Both must be new: a half-written
# bundle from an earlier attempt is not something to export over.
WORK=$DEST.fp32
[ ! -e "$DEST" ] && [ ! -e "$WORK" ] || die "$DEST or $WORK exists -- pick a new --dest or move them away."

echo "=== export $MODEL step $STEP -> $WORK (fp32)  $(date) ==="
EXPORT_ARGS=("$CKPT_FLAG" "$CKPT" --out-dir "$WORK")
[ -n "$VIEWS" ] && EXPORT_ARGS+=(--views "$VIEWS")
"$PY" "$PLAYBOOK/export_split_onnx.py" "${EXPORT_ARGS[@]}"
rc=$?; [ $rc -eq 0 ] || die "export failed (rc=$rc), stopping before parity."

# X-VLA's parity caches its PyTorch reference in ONE shared npz by default and refuses when
# that cache was emitted for a different bundle -- correctly, but it used to scroll past
# inside `| tee` and the export still "succeeded" unverified. Keep the reference with the
# work dir, and let a refusal or a FAIL stop the script.
REFERENCE=$WORK/parity_reference.npz
parity() {  # BUNDLE LABEL: the bundle's graphs vs the torch checkpoint, PARITY.txt in the bundle
    echo "=== parity ($2) vs the torch checkpoint  $(date) ==="
    "$PY" "$PLAYBOOK/$PARITY" --split-dir "$1" "$CKPT_FLAG" "$CKPT" --reference "$REFERENCE" 2>&1 | tee "$1/PARITY.txt"
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] || die "$2 parity did not pass (rc=$rc), see $1/PARITY.txt"
}
parity "$WORK" fp32

# The board's load gate: a stock-LeRobot chunk on seeded inputs, stored as fixture.npz. Without
# it the Orin runtime refuses the bundle. Written into the work dir so the FP16 step copies it.
echo "=== reference fixture  $(date) ==="
"$PY" "$PLAYBOOK/reference.py" --checkpoint "$CKPT" --bundle "$WORK" \
    || die "reference fixture failed -- the robot would refuse this bundle."

echo "=== mixed FP16 -> $DEST  $(date) ==="
"$PY" "$PLAYBOOK/fp16_weights.py" --split-dir "$WORK" --out-dir "$DEST" \
    || die "FP16 conversion failed."
parity "$DEST" fp16

echo "=== bundle ==="
ls -la "$DEST"
echo "  fp32 work dir (not shipped, delete when done): $WORK"
echo "  ship it:  $SPARK_PROJECTS/ship_bundle.sh $DEST <ssh-host>     (e.g. joel@pikkumasi2tb)"
