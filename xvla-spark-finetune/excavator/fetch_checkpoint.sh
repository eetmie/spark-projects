#!/usr/bin/env bash
# Fetch lerobot/xvla-base (3.52 GB) into models/xvla-base/.
#
# Why curl and not huggingface_hub: on this box `snapshot_download` stalled twice partway
# through the safetensors — the process stayed alive, the .incomplete file stopped growing
# (dead at 245 and 257 MB of 3519), and no exception was ever raised, so the built-in retry
# never fired. curl with `--speed-limit/--speed-time` treats a stall as an error, which lets
# `--retry` actually restart it, and `-C -` resumes instead of starting over. Measured ~9 MB/s
# against ~4 MB/s for the python client before it hung.
#
# Re-running is safe and cheap: finished files resume to a no-op.

set -euo pipefail

DST=${1:-/home/masi-pgx/spark-projects/xvla-spark-finetune/models/xvla-base}
# Pin the model used by the validated 879.7 M parameter runtime contract. A moving
# `main` here would make an old training result impossible to reproduce after an
# upstream checkpoint update.
REV=${REV:-cdb7964e4fe842935d671bfab5a5ebe00a96648c}
BASE=https://huggingface.co/lerobot/xvla-base/resolve/$REV

mkdir -p "$DST"
for f in config.json policy_preprocessor.json policy_postprocessor.json model.safetensors; do
    echo "=== $f ==="
    curl -L --fail --retry 20 --retry-delay 5 --retry-all-errors \
         --speed-limit 102400 --speed-time 30 \
         -C - -o "$DST/$f" "$BASE/$f" 2>&1 | tail -2
    echo "  -> $(stat -c %s "$DST/$f" 2>/dev/null || echo MISSING) bytes"
done

printf '%s\n' "$REV" > "$DST/REVISION"

# `hf cache verify` also works for a curl-populated local directory. The repository
# README and .gitattributes are intentionally omitted, so missing-file warnings are
# expected; every downloaded training artifact must still checksum correctly.
if command -v hf >/dev/null 2>&1; then
    hf cache verify lerobot/xvla-base --local-dir "$DST" --revision "$REV" --format agent
fi

echo
echo "revision: $REV"
echo "expected: model.safetensors 3519073692 B, config.json ~5463 B"
ls -la "$DST"
