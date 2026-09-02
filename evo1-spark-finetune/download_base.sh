#!/usr/bin/env bash
# Download the native Hugging Face InternVL3 checkpoint used to initialize EVO1.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_ID=OpenGVLab/InternVL3-1B-hf
# Exact base revision used by the verified LeRobot EVO1 LIBERO training recipe.
REVISION=014c0583a0d4bedf29fbe2dbff4f865eb998e171
DEST=${1:-"$HERE/models/InternVL3-1B-hf"}
HF_BIN=${HF_BIN:-hf}

if ! command -v "$HF_BIN" >/dev/null 2>&1; then
    echo "missing Hugging Face CLI: $HF_BIN" >&2
    echo "run ./setup.sh, or install it from https://hf.co/cli/install.sh" >&2
    exit 1
fi

mkdir -p "$DEST"
"$HF_BIN" download "$MODEL_ID" \
    --revision "$REVISION" \
    --local-dir "$DEST"

printf '%s\n' "$REVISION" > "$DEST/REVISION"

# REVISION is our local provenance marker and is intentionally an extra file.
"$HF_BIN" cache verify "$MODEL_ID" \
    --revision "$REVISION" \
    --local-dir "$DEST" \
    --format agent

echo
echo "EVO1 VLM base ready: $DEST"
echo "revision: $REVISION"
