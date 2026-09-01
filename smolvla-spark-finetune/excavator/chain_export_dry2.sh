#!/usr/bin/env bash
# Wait for the dry2 run AND its scoring to finish, then export the best checkpoint as a
# split ONNX bundle straight to the Desktop, and prove parity against the torch policy.
#
#   setsid nohup bash chain_export_dry2.sh > outputs/digging_dry2/chain.log 2>&1 &
#
# WAITS ON THE SCORING QUEUE'S MARKER, not on training. queue_digging_dry2.sh already
# waits for training and then runs eval_compare (x3) + eval_curve; its final line is
# "dry2 scoring queue done". Waiting on that gets both the GPU being free and the
# curve.json this script needs, in one condition. Not pgrep: see the long note in
# queue_digging_dry2.sh about pgrep -f matching agent shells and hanging forever.
#
# THE BEST CHECKPOINT IS READ FROM THE EVAL, NEVER ASSUMED. "Take the last one" is wrong
# on this project's own history: the kaivuri sweep peaked at 20000 of 30000 and the first
# dry run peaked at 25000 of 50000, while the digging sweep did peak at its last step.
# So if curve.json is missing or unreadable this script EXPORTS NOTHING and says so --
# a bundle silently built from the wrong checkpoint is worse than no bundle.

set -uo pipefail

ROOT=/home/masi-pgx/spark-projects/smolvla-spark-finetune
OUT=$ROOT/outputs/digging_dry2
VENV=$ROOT/.venv/bin
RUN=dry2_ir

echo "=== dry2 export chain start $(date) ==="
echo "waiting for the scoring queue to finish ..."
while ! grep -q "dry2 scoring queue done" "$OUT/queue.log" 2>/dev/null; do sleep 60; done
echo "[$(date +%T)] scoring done"

BEST=$("$VENV/python" - <<'PY'
import json, pathlib, sys
p = pathlib.Path("/home/masi-pgx/spark-projects/smolvla-spark-finetune/outputs/digging_dry2/curve.json")
try:
    pts = json.loads(p.read_text())["curves"]["dry2_ir"]
except Exception as exc:
    print(f"ERR {exc}", file=sys.stderr); sys.exit(1)
if not pts:
    print("ERR curve.json has no points for dry2_ir", file=sys.stderr); sys.exit(1)
best = min(pts, key=lambda r: r["disp_err"])
print(f"{best['step']} {best['disp_err']:.4f} {pts[-1]['step']} {pts[-1]['disp_err']:.4f}")
PY
)
if [ -z "$BEST" ]; then
    echo "!! could not read the best checkpoint from $OUT/curve.json — EXPORTING NOTHING."
    echo "   Check $OUT/CURVE.txt and $OUT/queue.log; eval_curve probably failed."
    exit 1
fi
set -- $BEST
STEP=$1; DISP=$2; LAST_STEP=$3; LAST_DISP=$4
printf -v CKSTEP "%06d" "$STEP"
CKPT=$OUT/$RUN/checkpoints/$CKSTEP/pretrained_model
DEST=/home/masi-pgx/Desktop/smolvla-digging-dry2-ir10-$STEP

echo "[$(date +%T)] best checkpoint: step $STEP  disp_err $DISP   (last: $LAST_STEP $LAST_DISP)"
[ -d "$CKPT" ] || { echo "!! $CKPT missing — exporting nothing."; exit 1; }

echo "=== export -> $DEST $(date) ==="
"$VENV/python" "$ROOT/export_split_onnx.py" --model-id "$CKPT" --out-dir "$DEST"
rc=$?; echo "export rc=$rc"
[ $rc -eq 0 ] || { echo "!! export failed, stopping before parity."; exit 1; }

echo "=== parity vs the torch checkpoint $(date) ==="
"$VENV/python" "$ROOT/parity_split_vs_torch.py" --split-dir "$DEST" --model-id "$CKPT" \
    2>&1 | tee "$DEST/PARITY.txt"
echo "parity rc=$?"

echo "=== bundle ==="
ls -la "$DEST"
"$VENV/python" -c "
import json;i=json.load(open('$DEST/export_info.json'))
print('fps',i['fps'],'chunk',i['chunk_size'],'cams',i['cameras'])
print('task ',repr(i.get('task')))
print('tasks',i.get('tasks'))
"
echo "=== dry2 export chain done $(date) ==="
