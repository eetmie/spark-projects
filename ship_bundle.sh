#!/usr/bin/env bash
# Ship an export bundle from the Spark to the Orin, and verify it on arrival.
#
#   ./ship_bundle.sh ~/bundles/xvla-base-split orin
#   ./ship_bundle.sh vla-onnx/xvla/exports/split orin xvla-base-split
#
# The rules this encodes, all of them learned the hard way:
#
#   * TensorRT engines are hardware- and version-specific and are NEVER copied. The
#     ONNX is the portable artefact; the Orin builds its own engines and caches them.
#     Anything that looks like an engine or an engine cache is excluded below.
#   * A bundle without MANIFEST.sha256 is refused. The graphs travel as .onnx +
#     .onnx.data pairs, and a truncated external-data file fails much later at
#     engine-build time with an error that names neither the file nor the cause.
#   * A bundle whose export_info.json has a null fps/task, or whose stats.json has no
#     usable normalization, still LOADS on the robot and drives it wrong. That is a
#     loud warning here, and a hard stop unless --allow-base says you meant it.
set -euo pipefail

ALLOW_BASE=0
args=()
for a in "$@"; do
    case "$a" in
        --allow-base) ALLOW_BASE=1 ;;
        *) args+=("$a") ;;
    esac
done
set -- "${args[@]}"

BUNDLE="${1:?usage: ship_bundle.sh <bundle-dir> <ssh-host> [remote-name] [--allow-base]}"
HOST="${2:?usage: ship_bundle.sh <bundle-dir> <ssh-host> [remote-name] [--allow-base]}"
NAME="${3:-$(basename "$BUNDLE")}"
# Relative to the remote home on purpose: rsync does NOT expand $HOME in a remote
# path (it quotes it), so an absolute-looking "$HOME/bundles" arrives literally and
# rsync cheerfully makes a directory called '$HOME'. A relative path is resolved by
# the remote shell against the login home, which is what we actually want.
REMOTE_ROOT="${REMOTE_ROOT:-bundles}"

BUNDLE="${BUNDLE%/}"
[ -d "$BUNDLE" ] || { echo "!! no such bundle: $BUNDLE" >&2; exit 1; }

if [ ! -f "$BUNDLE/MANIFEST.sha256" ]; then
    echo "!! $BUNDLE has no MANIFEST.sha256 — it predates the metadata exporter." >&2
    echo "   Re-export it so the transfer can be verified on arrival:" >&2
    echo "     python export_split_onnx.py --model-id <ckpt> --out-dir <dir>" >&2
    exit 1
fi

# --- what is actually in this bundle -------------------------------------------------
INFO="$BUNDLE/export_info.json"
if [ -f "$INFO" ]; then
    python3 - "$INFO" "$BUNDLE" "$ALLOW_BASE" <<'PY'
import json, sys
from pathlib import Path
info = json.loads(Path(sys.argv[1]).read_text())
bundle, allow_base = Path(sys.argv[2]), sys.argv[3] == "1"
print(f"   model      : {info.get('model_id')}")
print(f"   exported   : {info.get('exported_at')}  (exporter {info.get('exporter_sha')})")
print(f"   shapes     : chunk={info.get('chunk_size')} steps={info.get('num_steps')} "
      f"state={info.get('state_dim')}/{info.get('max_state_dim')} "
      f"action={info.get('action_dim')}/{info.get('max_action_dim')}")
print(f"   fps / task : {info.get('fps')} / {info.get('task')!r}")
print(f"   cameras    : {info.get('cameras')} in {info.get('n_cam_slots')} slot(s)")
if info.get("state_blind"):
    print("   STATE-BLIND: the runtime must feed observation.state ZEROS.")

problems = []
if info.get("fps") is None:
    problems.append("fps is null — the runtime has to guess the control rate")
if info.get("task") is None:
    problems.append("task is null — run_inference refuses to start without --task")
stats_f = bundle / "stats.json"
stats = json.loads(stats_f.read_text()) if stats_f.exists() else {}
usable = [k for k in ("observation.state", "action") if k in stats]
if len(usable) < 2:
    problems.append(f"stats.json has {len(usable)}/2 usable keys — IDENTITY normalization")
if problems:
    print("\n!! this bundle is not robot-ready:")
    for p in problems:
        print(f"     - {p}")
    if not allow_base:
        print("\n   Base-weight bundles are fine for latency benchmarking; pass --allow-base")
        print("   to ship it anyway. Never drive a machine with one.")
        sys.exit(2)
    print("\n   --allow-base given: shipping as a benchmark artefact.")
PY
elif [ -f "$BUNDLE/bundle.json" ]; then
    # The current X-VLA exporter (vla-onnx/run_export.sh) writes bundle.json. The Orin's native
    # TensorRT runtime (kaivuriprokkis-python, since 2026-10-29) refuses a bundle without the
    # stock reference fixture, and from FP32 graphs it builds engines ~4.5x slower to run.
    python3 - "$BUNDLE" "$ALLOW_BASE" <<'PY'
import json, sys
from pathlib import Path
bundle, allow_base = Path(sys.argv[1]), sys.argv[2] == "1"
b = json.loads((bundle / "bundle.json").read_text())
fx = b.get("fixture") or {}
print(f"   checkpoint : {b.get('checkpoint')}")
print(f"   shapes     : chunk={b.get('chunk_size')} steps={b.get('num_denoising_steps')} "
      f"state={b.get('real_state_dim')}/{b.get('max_state_dim')} "
      f"action={b.get('real_action_dim')}/{b.get('max_action_dim')}  views={b.get('valid_views')}")
print(f"   fps / tasks: {b.get('fps')} / {b.get('tasks')}")
print(f"   fixture    : {fx.get('file')} ({fx.get('source')})")
problems = []
if not b.get("fps"):
    problems.append("fps is missing -- the runtime has to guess the control rate")
if not b.get("tasks"):
    problems.append("no tasks recorded -- run_inference needs --task")
if not (fx.get("file") and (bundle / fx["file"]).exists()):
    problems.append("no reference fixture -- the Orin runtime refuses it (vla-onnx/xvla/reference.py)")
if not list((bundle / "processor").glob("*normalizer*.safetensors")):
    problems.append("no processor normalizer stats -- IDENTITY normalization")
# FP16 weights are ~2 bytes a parameter; FP32 ~4.
fp32 = [g["name"] for g in b.get("graphs", []) if g.get("params") and g["size_mb"] * 1e6 > 3 * g["params"]]
if fp32:
    problems.append(f"FP32 graphs ({', '.join(fp32[:4])}{'...' if len(fp32) > 4 else ''}) -- "
                    "run vla-onnx/xvla/fp16_weights.py (run_export.sh does it)")
if problems:
    print("\n!! this bundle is not robot-ready:")
    for p in problems:
        print(f"     - {p}")
    if not allow_base:
        print("\n   Pass --allow-base to ship it anyway as a benchmark artefact. Never drive with one.")
        sys.exit(2)
    print("\n   --allow-base given: shipping as a benchmark artefact.")
PY
else
    echo "!! neither export_info.json nor bundle.json — cannot tell what this bundle is." >&2
    [ "$ALLOW_BASE" = 1 ] || exit 1
fi

DEST="$REMOTE_ROOT/$NAME"
echo
echo ">> $BUNDLE  ->  $HOST:$DEST  ($(du -sh "$BUNDLE" | cut -f1))"

ssh "$HOST" "mkdir -p $DEST"
rsync -avh --partial --progress \
      --exclude 'trt_cache/' --exclude 'trt_native_cache/' --exclude 'ctx/' \
      --exclude '*.engine' --exclude '*.plan' --exclude '*.profile' \
      --exclude '__pycache__/' --exclude '_meta_*.json' \
      "$BUNDLE/" "$HOST:$DEST/"

echo
echo ">> verifying on $HOST ..."
ssh "$HOST" "cd $DEST && sha256sum -c MANIFEST.sha256 --quiet && echo '   MANIFEST OK — all files match'"

echo
echo "Bundle is on the board. Engines build on the first load and are cached in the bundle:"
if [ -f "$BUNDLE/bundle.json" ] && [ ! -f "$BUNDLE/export_info.json" ]; then
    echo "  cd ~/kaivuriprokkis-python && .venv/bin/python -m lerobot_vla.run_inference --split-dir ~/$DEST --synthetic"
else
    echo "  python -m bench ort-split --bundle $DEST --precision fp16 --projectors gpu --iobinding"
fi
