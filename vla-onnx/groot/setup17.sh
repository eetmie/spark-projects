#!/usr/bin/env bash
# GR00T N1.7 export environment on DGX Spark. Own venv: N1.7's Qwen3-VL backbone needs
# transformers 4.57.3, which cannot share .venv-groot (N1.6, 4.51.3).
#
# The model code is NVIDIA Isaac-GR00T at the n1.7-release tag, fetched as source into
# $GROOT_SRC and installed without its dependency list (flash-attn, deepspeed, triton
# pins are training deps) and without its python==3.10 marker (the export runs on 3.12).
#
# The tokenizer/processor files come from nvidia/Cosmos-Reason2-2B, a gated HF repo:
# accept its terms on huggingface.co first, or pass --vlm-files to reference17.py.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../paths.sh"

VENV_ROOT=${VENV_ROOT:-$HERE/.venv-groot17}
GROOT_SRC=${GROOT_SRC:-$HOME/Isaac-GR00T-n1.7}
GROOT_REF=23ace64f17aa5015259b8609d371eb61a357c776   # n1.7-release

if [ ! -f "$GROOT_SRC/.source-commit" ] || [ "$(cat "$GROOT_SRC/.source-commit")" != "$GROOT_REF" ]; then
    echo "fetching Isaac-GR00T $GROOT_REF into $GROOT_SRC"
    mkdir -p "$GROOT_SRC"
    curl -sfL "https://codeload.github.com/NVIDIA/Isaac-GR00T/tar.gz/$GROOT_REF" \
        | tar -xz -C "$GROOT_SRC" --strip-components=1 --exclude='*/demo_data' --exclude='*.ipynb'
    echo "$GROOT_REF" > "$GROOT_SRC/.source-commit"
fi

[ -x "$VENV_ROOT/bin/python" ] || python3.12 -m venv "$VENV_ROOT"
"$VENV_ROOT/bin/python" -m pip install -q --upgrade pip "setuptools<82" wheel
"$VENV_ROOT/bin/python" -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    -r "$HERE/requirements17.txt"
"$VENV_ROOT/bin/python" -m pip install --no-deps --ignore-requires-python -e "$GROOT_SRC"
"$VENV_ROOT/bin/hf" download nvidia/GR00T-N1.7-3B >/dev/null

echo "GR00T N1.7 export environment ready: $VENV_ROOT"
