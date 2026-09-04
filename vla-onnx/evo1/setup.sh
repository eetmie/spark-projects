#!/usr/bin/env bash
# Idempotent EVO1 host-side environment and base-checkpoint setup for DGX Spark.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../paths.sh"

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV_ROOT=${VENV_ROOT:-$VENV_LEROBOT061}
VENV_BIN="$VENV_ROOT/bin"
PYTHON_BIN=${PYTHON_BIN:-python3}

if [ ! -x "$VENV_BIN/python" ]; then
    echo "creating EVO1 environment at $VENV_ROOT"
    "$PYTHON_BIN" -m venv "$VENV_ROOT"
fi

"$VENV_BIN/python" -m pip install --upgrade pip "setuptools<82" wheel
"$VENV_BIN/python" -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    -r "$HERE/requirements.txt"

HF_BIN="$VENV_BIN/hf" bash "$HERE/download_base.sh"
"$VENV_BIN/python" "$HERE/preflight.py" --load-model

echo
echo "EVO1 Spark setup is ready."
echo "Activate with: source $VENV_BIN/activate"
