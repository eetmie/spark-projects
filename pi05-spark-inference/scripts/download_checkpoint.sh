#!/usr/bin/env bash
# Download the π0.5 LIBERO JAX checkpoint from the public openpi GCS bucket.
# Run INSIDE the container (gsutil is installed there).
set -euo pipefail
CKPT_NAME="${1:-pi05_libero}"
DEST="/workspace/checkpoints/${CKPT_NAME}"
SRC="gs://openpi-assets/checkpoints/${CKPT_NAME}"

mkdir -p "$DEST"
echo "Downloading $SRC -> $DEST"
# -m parallel, -n no-clobber. Public bucket (anonymous).
# check_hashes=never: the container's crcmod lacks the C extension, which otherwise
# makes gsutil refuse composite-object downloads. We trust gs:// transport here.
gsutil -o "GSUtil:check_hashes=never" -o "Boto:https_validate_certificates=True" \
  -m cp -r -n "$SRC/*" "$DEST/"

echo "Done. Contents:"
ls -R "$DEST" | head -40
