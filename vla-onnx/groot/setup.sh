#!/usr/bin/env bash
# GR00T N1.6 export environment on DGX Spark. Own venv: transformers 4.51.3 cannot
# share either lerobot venv.
#
# The model code is NVIDIA Isaac-GR00T at the n1.6.1-release tag, fetched as source
# into $GROOT_SRC and installed without its dependency list (flash-attn, deepspeed and
# torchcodec are training/dataset deps the export does not touch).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../paths.sh"

VENV_ROOT=${VENV_ROOT:-$HERE/.venv-groot}
GROOT_SRC=${GROOT_SRC:-$HOME/Isaac-GR00T-n1.6}
GROOT_REF=5dc80c4afd726b34faad1d8f7e007a13b34e4c88   # n1.6.1-release

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
    -r "$HERE/requirements.txt"
"$VENV_ROOT/bin/python" -m pip install --no-deps -e "$GROOT_SRC"
"$VENV_ROOT/bin/hf" download nvidia/GR00T-N1.6-3B >/dev/null

echo "GR00T export environment ready: $VENV_ROOT"
