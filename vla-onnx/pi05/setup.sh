#!/usr/bin/env bash
# OpenPI venv for vla-onnx/pi05 (excavator fine-tune + export) on the DGX Spark.
#
#   ./setup.sh      # clone OpenPI into third_party/openpi (if missing) and build its .venv
#
# The venv the digging640 runs trained in. Differs from OpenPI's own uv.lock in:
#   lerobot 0cf8648   from the GitHub tarball (a git fetch of the lerobot repo crawls)
#   av 13.1.0         lock: 17.0.0
#   nvidia-*-cu12     CUDA 12.9 libraries, lock: 12.6
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPENPI=$HERE/third_party/openpi
OPENPI_REV=c23745b5ad24e98f66967ea795a07b2588ed6c79
LEROBOT_REV=0cf864870cf29f4738d3ade893e6fd13fbd7cdb5

[ -d "$OPENPI/.git" ] || git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI"
git -C "$OPENPI" checkout --detach "$OPENPI_REV"
cd "$OPENPI"

GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-package lerobot
PIN=(
  "lerobot @ https://github.com/huggingface/lerobot/archive/$LEROBOT_REV.tar.gz"
  av==13.1.0
  nvidia-cublas-cu12==12.9.0.13 nvidia-cuda-cupti-cu12==12.9.19 nvidia-cuda-runtime-cu12==12.9.37
  nvidia-cudnn-cu12==9.10.1.4 nvidia-cufft-cu12==11.4.0.6 nvidia-cusolver-cu12==11.7.4.40
  nvidia-cusparse-cu12==12.5.9.5 nvidia-nccl-cu12==2.26.5 nvidia-nvjitlink-cu12==12.9.41
)
uv pip install --python .venv/bin/python --no-deps "${PIN[@]}"
.venv/bin/python -c "import jax, openpi, lerobot; print('jax', jax.__version__, jax.devices())"
