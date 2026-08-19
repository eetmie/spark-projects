#!/usr/bin/env bash
# Two SmolVLA fine-tunes on the masi_digging dataset (82 eps, 41765 frames @30fps,
# task "move the sand to the container"), differing ONLY in how many cameras the
# policy sees:
#
#   ir    observation.images.cam1                      (D435i infrared left imager)
#   both  observation.images.cam1 + .cam2              (IR + RGB colour imager)
#
# Everything else is run A's settled recipe (batch 32, chunk 50, lr 1e-4,
# frozen vision encoder, expert-only) so the camera count is the only variable.
# The `ir` run reads datasets/masi_digging_ir, a metadata-only view built by
# make_camera_variant.py whose data/ and cam1 videos are symlinks into the source
# — both runs therefore train on byte-identical cam1 frames.
#
# observation.state is 3-dim [lift, tilt, scoop]: slew IMU feedback is left out
# because its yaw origin drifts across sessions (kaivuriprokkis lerobot_vla/README).
# Actions stay 4-dim [slew, lift, tilt, scoop] — slew is still commanded, the task
# needs the rotation, it just is not fed back.
#
#   bash run_digging.sh            # both runs, sequentially
#   bash run_digging.sh ir         # just one
#   STEPS=8000 bash run_digging.sh # shorter budget
#
# Re-running resumes any run that has a checkpoint and is short of STEPS.

set -uo pipefail

ROOT=/home/masi-pgx/spark-projects/smolvla-spark-finetune   # was ~/smolvla, which is gone
VENV=$ROOT/.venv/bin
OUT=$ROOT/outputs/digging
LOGS=$OUT/logs

STEPS=${STEPS:-25000}
BATCH=${BATCH:-32}
CHUNK=${CHUNK:-50}
LR=${LR:-1e-4}
SEED=${SEED:-1000}
SAVE_FREQ=${SAVE_FREQ:-2500}
WORKERS=${WORKERS:-10}

DS_BOTH=/home/masi-pgx/Desktop/masi_digging
DS_IR=$ROOT/datasets/masi_digging_ir

# Held out for eval_compare/eval_curve: every 10th episode from 5, 8 of 82 (9.8%).
# Spread across the session so the split is not a single stretch of the recording.
VAL_EPISODES="5 15 25 35 45 55 65 75"
TRAIN_EPS=$(python3 -c "
val={$(echo $VAL_EPISODES | tr ' ' ',')}
print('['+','.join(str(e) for e in range(82) if e not in val)+']')")

mkdir -p "$LOGS"

config_for() {
  case "$1" in
    ir)   echo "$DS_IR|local/masi_digging_ir"    ;;
    both) echo "$DS_BOTH|local/masi_digging_both" ;;
    *) return 1 ;;
  esac
}

train_one() {
  local name=$1
  local cfg; cfg=$(config_for "$name") || { echo "unknown run '$name'"; return 1; }
  local root=${cfg%%|*}; local repo=${cfg##*|}

  local dir=$OUT/$name
  local log=$LOGS/$name.log

  if [ -d "$dir/checkpoints/last" ]; then
    local done_steps
    done_steps=$(grep -oE "[0-9]+" "$dir/checkpoints/last/training_state/training_step.json" 2>/dev/null | head -1)
    if [ "${done_steps:-0}" -ge "$STEPS" ]; then
      echo "[$(date +%H:%M:%S)] run $name already complete ($done_steps steps), skipping"
      return 0
    fi
    echo "[$(date +%H:%M:%S)] run $name resuming from step ${done_steps:-?}"
    WANDB_MODE=disabled "$VENV/lerobot-train" \
      --config_path="$dir/checkpoints/last/pretrained_model/train_config.json" \
      --resume=true >> "$log" 2>&1
  else
    echo "[$(date +%H:%M:%S)] === run $name: $repo chunk=$CHUNK steps=$STEPS ==="
    WANDB_MODE=disabled "$VENV/lerobot-train" \
      --dataset.repo_id="$repo" \
      --dataset.root="$root" \
      --dataset.video_backend=torchcodec \
      --dataset.episodes="$TRAIN_EPS" \
      --policy.type=smolvla \
      --policy.pretrained_path=lerobot/smolvla_base \
      --policy.load_vlm_weights=true \
      --policy.chunk_size="$CHUNK" \
      --policy.n_action_steps="$CHUNK" \
      --policy.device=cuda \
      --policy.use_amp=true \
      --policy.push_to_hub=false \
      --policy.repo_id="local/smolvla-digging-$name" \
      --output_dir="$dir" \
      --job_name="digging_$name" \
      --seed=$SEED \
      --steps=$STEPS \
      --batch_size=$BATCH \
      --num_workers=$WORKERS \
      --log_freq=250 \
      --save_freq=$SAVE_FREQ \
      --eval_freq=0 \
      --optimizer.lr=$LR \
      > "$log" 2>&1
  fi

  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] run $name finished ok"
  else
    echo "[$(date +%H:%M:%S)] run $name FAILED (rc=$rc), see $log"
  fi
  return $rc
}

RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(ir both)

echo "digging sweep start $(date)"
echo "held-out episodes: $VAL_EPISODES"
echo "steps=$STEPS batch=$BATCH chunk=$CHUNK lr=$LR workers=$WORKERS"
for r in "${RUNS[@]}"; do train_one "$r"; done
echo "digging sweep done $(date)"
