#!/usr/bin/env bash
# Export a trained run with jetson-orin-nano-vla's exporter (mixed-FP16 split ONNX bundle
# with the stock-policy fixture the Jetson checks before it runs).
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

PY=$VENV_LEROBOT061/bin/python   # reads curve.json
[ -x "$PY" ] || die "no python in $VENV_LEROBOT061"

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

: "${DEST:=$VLA_DATASETS/$MODEL-$(basename "$(dirname "$RUN")")-$(basename "$RUN")-$STEP}"

echo "=== export $MODEL step $STEP -> $DEST  $(date) ==="
# jetson-orin-nano-vla's exporter: mixed-FP16 graphs, the stock-policy fixture and the
# manifest. The Jetson's ./run.sh checks the engines against that fixture before anything runs.
[ -e "$DEST" ] && die "$DEST already exists -- remove it or pass another --dest."
[ -x "$JETSON_VLA/export.sh" ] || die "no $JETSON_VLA/export.sh -- clone github.com/eetmie/jetson-orin-nano-vla or set JETSON_VLA"
"$JETSON_VLA/export.sh" "$CKPT" --out "$DEST" ${VIEWS:+--views "$VIEWS"} || die "export failed"

echo "  dry run on the board:  ./run.sh $DEST   (jetson-orin-nano-vla)"
echo "  ship it:  $SPARK_PROJECTS/ship_bundle.sh $DEST orin"
