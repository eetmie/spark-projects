#!/usr/bin/env bash
# Fine-tuned OpenPI pi0.5 checkpoint step -> kaivuri-inference bundle (README.md).
#
#   ./export.sh CHECKPOINT_STEP_DIR OUT [--config NAME] [--converted DIR] [-- EXPORT_FLAGS...]
#
#   --config NAME     excavator TrainConfig (default pi05_excavator, vla-onnx/pi05/excavator/config.py)
#   --converted DIR   reuse an earlier PyTorch conversion of this step (model.safetensors + assets/)
#   -- ...            passed to container/export.py (--row-time, --rtc-rows N, --rtc-elapsed N, ...)
#
# Host steps run in the OpenPI venv (dataset access); conversion and export run in the pinned
# container built from container/Dockerfile. OUT and OUT.work must be new.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PI05=$(cd "$HERE/../pi05" && pwd)
usage() { sed -n '2,11p' "$0" | sed 's/^# \?//'; exit "${1:-2}"; }
[[ $# -ge 2 ]] || usage
[[ $1 == -h || $1 == --help ]] && usage 0
STEP=$(cd "$1" && pwd)
OUT=$(realpath -m "$2")
shift 2
CONFIG=pi05_excavator CONVERTED="" EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG=$2; shift 2 ;;
        --converted) CONVERTED=$(cd "$2" && pwd); shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "unknown flag: $1" >&2; usage ;;
    esac
done
WORK="$OUT.work"
[[ ! -e "$OUT" && ! -e "$WORK" ]] || { echo "OUT and OUT.work must be new" >&2; exit 1; }
[[ -d "$STEP/params" && -d "$STEP/assets" ]] || { echo "$STEP is not an OpenPI checkpoint step (params/ + assets/)" >&2; exit 1; }
VENV_PY="$PI05/third_party/openpi/.venv/bin/python"
[[ -x "$VENV_PY" ]] || { echo "no OpenPI venv at $VENV_PY (vla-onnx/pi05 README)" >&2; exit 1; }
REVISION=$(git -C "$HERE" rev-parse HEAD)$(git -C "$HERE" diff --quiet HEAD -- . || echo "+dirty")
mkdir -p "$WORK"
log() { echo "=== $*  $(date +%T)"; }

log "fixture frames (training dataset)"
"$VENV_PY" "$PI05/excavator/fixture_frames.py" --config "$CONFIG" --out "$WORK/frames.npz" | tee "$WORK/frames.log"
log "action-chunk statistics (RTC)"
"$VENV_PY" "$HERE/host/action_stats.py" --step "$STEP" --config "$CONFIG" --out "$WORK/action_stats.npz" | tee "$WORK/action_stats.log"

log "container"
docker build -q --iidfile "$WORK/image-id" "$HERE/container"
RUN=(docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp
     -e OPENPI_DATA_HOME=/work/cache -e HF_HOME=/work/hf-cache -e PI05_CONFIG_DIR=/cfg -e PI05_CONFIG="$CONFIG"
     -e PI05_EXPORT_DIR=/pi05 -e KP_EXPORTER_REVISION="$REVISION" -e PI05_SOURCE_STEP="$STEP"
     -e PYTHONPATH=/opt/openpi/src:/opt/openpi/packages/openpi-client/src
     -v "$HERE:/kp:ro" -v "$PI05/export:/pi05:ro" -v "$PI05/excavator:/cfg:ro" -v "$STEP:/step:ro"
     -v "$WORK:/work" -v "$(dirname "$OUT"):/output")
if [[ -n "$CONVERTED" ]]; then
    RUN+=(-v "$CONVERTED:/converted:ro"); CHECKPOINT=/converted
    log "reusing conversion $CONVERTED"
else
    log "JAX -> PyTorch conversion"
    "${RUN[@]}" "$(cat "$WORK/image-id")" \
        python /pi05/convert.py --config_name "$CONFIG" --checkpoint_dir /step --output_path /work/checkpoint
    cp -r "$STEP/assets" "$WORK/checkpoint/assets"
    CHECKPOINT=/work/checkpoint
fi
log "export"
"${RUN[@]}" "$(cat "$WORK/image-id")" python /kp/container/export.py --checkpoint "$CHECKPOINT" \
    --frames /work/frames.npz --action-stats /work/action_stats.npz --out "/output/$(basename "$OUT")" \
    --work /work/export --config "$CONFIG" "${EXTRA[@]}" 2>&1 | tee "$WORK/export.log"
log "done: $OUT"
echo "ship:  rsync -a $OUT/ joel@pikkumasi2tb:bundles/$(basename "$OUT")/   (README.md, On the board)"
