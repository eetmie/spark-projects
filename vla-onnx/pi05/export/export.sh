#!/usr/bin/env bash
# Fine-tuned OpenPI checkpoint step dir -> compact FP16 split bundle (BUNDLE.md).
#   export/export.sh checkpoints/pi05_excavator/<exp>/<step> OUT [CONFIG]
# Fixture frames come from the training dataset (OpenPI venv); conversion and the four export
# stages run in jetson-orin-nano-vla's pinned openpi container (export/pi05/Dockerfile).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PI05=$(cd "$HERE/.." && pwd)
STEP=$(cd "${1:?usage: export.sh CHECKPOINT_STEP_DIR OUT [CONFIG]}" && pwd)
OUT=$(realpath -m "${2:?usage: export.sh CHECKPOINT_STEP_DIR OUT [CONFIG]}")
CONFIG=${3:-pi05_excavator}
WORK="$OUT.work"
JETSON_VLA=${JETSON_VLA:-$HOME/jetson-orin-nano-vla}
[[ ! -e "$OUT" && ! -e "$WORK" ]] || { echo "OUT and OUT.work must be new"; exit 1; }
[[ -d "$STEP/params" && -d "$STEP/assets" ]] || { echo "$STEP is not an OpenPI checkpoint step (params/ + assets/)"; exit 1; }
mkdir -p "$WORK"

"$PI05/third_party/openpi/.venv/bin/python" "$PI05/excavator/fixture_frames.py" --config "$CONFIG" --out "$WORK/frames.npz"
docker build -q --iidfile "$WORK/image-id" -f "$JETSON_VLA/export/pi05/Dockerfile" "$JETSON_VLA"
IMAGE=$(cat "$WORK/image-id")
RUN=(docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp
     -e OPENPI_DATA_HOME=/work/cache -e HF_HOME=/work/hf-cache -e PI05_CONFIG_DIR=/cfg -e PI05_CONFIG="$CONFIG"
     -e PYTHONPATH=/opt/openpi/src:/opt/openpi/packages/openpi-client/src:/jvla
     -v "$JETSON_VLA/export/vla_common:/jvla/vla_common:ro"
     -v "$PI05/excavator:/cfg:ro" -v "$HERE:/exp:ro" -v "$STEP:/step:ro" -v "$WORK:/work"
     -v "$(dirname "$OUT"):$(dirname "$OUT")" "$IMAGE")
"${RUN[@]}" python /exp/convert.py --config_name "$CONFIG" --checkpoint_dir /step --output_path /work/checkpoint
cp -r "$STEP/assets" "$WORK/checkpoint/assets"
"${RUN[@]}" python /exp/export.py --checkpoint /work/checkpoint --frames /work/frames.npz \
  --out "$OUT" --work-dir /work/stages
