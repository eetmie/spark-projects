#!/usr/bin/env bash
# Build the GB10 inference image.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# Build context = repo root (Dockerfile COPYs third_party/openpi). .dockerignore
# keeps the 11 GB checkpoint and caches out of the context.
docker build -t pi05-spark:latest -f docker/Dockerfile .
echo "Built pi05-spark:latest"
