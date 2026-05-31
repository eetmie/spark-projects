#!/usr/bin/env bash
# Verify the baked-in openpi environment (the install now lives in the image;
# see docker/Dockerfile). Run INSIDE the container.
set -euo pipefail
python - <<'PY'
import torch, openpi
from openpi.models_pytorch.transformers_replace.models.siglip import check
print("openpi import OK")
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
print("transformers_replace ok:", check.check_whether_transformers_replace_is_installed_correctly())
PY
