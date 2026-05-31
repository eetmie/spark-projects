#!/usr/bin/env bash
# Launch the inference container with the repo + persistent caches mounted.
# Usage: ./docker/run.sh [command...]   (no args = interactive bash)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Persistent cache on the host so checkpoints/engines survive container restarts.
CACHE_DIR="${PI05_CACHE_DIR:-$REPO_ROOT/.cache}"
mkdir -p "$CACHE_DIR/openpi" "$CACHE_DIR/hf"

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$REPO_ROOT":/workspace \
  -v "$CACHE_DIR":/cache \
  -w /workspace \
  pi05-spark:latest "$@"
