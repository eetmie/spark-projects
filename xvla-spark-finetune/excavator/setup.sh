#!/usr/bin/env bash
# Idempotently install/validate the X-VLA fine-tuning stack on this DGX Spark.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
SMOL=$(cd "$ROOT/../smolvla-spark-finetune" && pwd)
VENV_ROOT=$SMOL/.venv
VENV=$VENV_ROOT/bin
RUN=${1:-clean_ir}

if [ ! -x "$VENV/python" ]; then
    echo "creating the shared, comparison-pinned LeRobot environment at $VENV_ROOT"
    python3 -m venv "$VENV_ROOT"
    "$VENV/python" -m pip install --upgrade pip
    "$VENV/python" -m pip install -r "$SMOL/requirements.txt" \
        --extra-index-url https://download.pytorch.org/whl/cu130
fi

if [ ! -f "$ROOT/models/xvla-base/model.safetensors" ]; then
    bash "$HERE/fetch_checkpoint.sh"
fi

if [ ! -f "$ROOT/models/xvla-base-excavator/model.safetensors" ]; then
    "$VENV/python" "$HERE/prepare_checkpoint.py"
fi

"$VENV/python" "$HERE/preflight.py" --run "$RUN"
echo
echo "X-VLA fine-tuning stack is ready. Smoke test: bash $HERE/smoke.sh $RUN"
