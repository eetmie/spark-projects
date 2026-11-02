#!/usr/bin/env bash
# Idempotently install/validate the X-VLA fine-tuning stack on this DGX Spark.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../paths.sh"

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
VENV=$VENV_LEROBOT061/bin

# One environment: lerobot 0.6.1 / torch 2.11 trains, scores and exports (2026-11-02). The
# old 0.5.1 / torch 2.12 training venv (requirements.txt) is retired; 0.6.1 needs torch < 2.12.
if [ ! -x "$VENV_LEROBOT061/bin/python" ]; then
    echo "creating the X-VLA environment at $VENV_LEROBOT061"
    python3 -m venv "$VENV_LEROBOT061"
    "$VENV_LEROBOT061/bin/python" -m pip install --upgrade pip "setuptools<82" wheel
    "$VENV_LEROBOT061/bin/python" -m pip install -r "$ROOT/requirements-export.txt" \
        --extra-index-url https://download.pytorch.org/whl/cu130
fi

if [ ! -f "$ROOT/models/xvla-base/model.safetensors" ]; then
    bash "$HERE/fetch_checkpoint.sh"
fi

if [ ! -f "$ROOT/models/xvla-base-excavator/model.safetensors" ]; then
    "$VENV/python" "$HERE/prepare_checkpoint.py"
fi

"$VENV/python" "$HERE/preflight.py" "$@"
echo
echo "X-VLA fine-tuning stack is ready. Smoke test:"
echo "  $VLA_ONNX/run_training.sh --dataset <recording> --cameras cam1 --smoke"
