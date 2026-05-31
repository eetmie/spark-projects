#!/usr/bin/env bash
# Convert a π0.5 JAX checkpoint to PyTorch safetensors. Run INSIDE the container.
set -euo pipefail
CKPT_NAME="${1:-pi05_libero}"
IN="/workspace/checkpoints/${CKPT_NAME}"
OUT="/workspace/checkpoints/${CKPT_NAME}_pytorch"

echo "Converting $IN -> $OUT"
# Use the baked-in openpi at /opt/openpi (import path is stable regardless of cwd).
python /opt/openpi/examples/convert_jax_model_to_pytorch.py \
  --config_name "${CKPT_NAME}" \
  --checkpoint_dir "$IN" \
  --output_path "$OUT"

# The convert script's asset copy looks in the wrong dir; copy norm_stats ourselves
# so the PyTorch checkpoint is self-contained (benchmark loads them from here).
if [ -d "$IN/assets" ] && [ ! -d "$OUT/assets" ]; then
  cp -r "$IN/assets" "$OUT/assets"
  echo "Copied assets into $OUT/assets"
fi

echo "Done. Output:"
ls -lh "$OUT" | head
