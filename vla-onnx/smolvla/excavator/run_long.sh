#!/usr/bin/env bash
# Long version of the A/B/C/D sweep: 30000 steps instead of 8000.
#
# Why 30000 steps rather than "250 epochs": the four variants have very different
# dataset sizes (18108 / 6043 / 3634 frames), so a fixed epoch count would hand
# each run a wildly different compute budget (21.6 h for A vs 4.3 h for C) and
# destroy the equal-compute basis that makes them comparable. Equal steps keeps
# the comparison honest and still reaches 264 epochs on the 6 fps set, 159 on the
# 10 fps set, 53 on native. It also matches SmolVLA's own scheduler_decay_steps
# default, so the LR schedule runs as designed instead of being auto-compressed.
#
# Checkpoints every 2500 steps (12 per run) so eval_curve.py can find where
# held-out performance actually peaks — with 27 training episodes the last step
# is not guaranteed to be the best one.
#
#   bash run_long.sh          # all four
#   bash run_long.sh A D      # a subset

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../paths.sh"

PROJ=$ROOT
VENV=$VENV_LEROBOT051/bin
OUT=$PROJ/outputs/excavator_long
LOGS=$OUT/logs

STEPS=30000
BATCH=32
LR=1e-4
SEED=1000
SAVE_FREQ=2500
WORKERS=8

VAL_EPISODES="3 11 19 27"
TRAIN_EPS=$(python3 -c "
val={$(echo $VAL_EPISODES | tr ' ' ',')}
print('['+','.join(str(e) for e in range(31) if e not in val)+']')")

DS_30=$VLA_DATASETS/masi_kaivuri_juusto
DS_10=$PROJ/datasets/masi_kaivuri_10fps
DS_06=$PROJ/datasets/masi_kaivuri_6fps

mkdir -p "$LOGS"

config_for() {
  case "$1" in
    A) echo "$DS_30|local/masi_kaivuri_juusto|50"  ;;
    B) echo "$DS_10|local/masi_kaivuri_10fps|50"   ;;
    C) echo "$DS_06|local/masi_kaivuri_6fps|50"    ;;
    D) echo "$DS_30|local/masi_kaivuri_juusto|150" ;;
    *) return 1 ;;
  esac
}

train_one() {
  local name=$1
  local cfg; cfg=$(config_for "$name") || { echo "unknown run '$name'"; return 1; }
  local root=${cfg%%|*}; local rest=${cfg#*|}
  local repo=${rest%%|*}; local chunk=${rest##*|}

  local dir=$OUT/$name
  local log=$LOGS/$name.log

  # Resume rather than restart if a previous attempt got partway.
  local resume=""
  if [ -d "$dir/checkpoints/last" ]; then
    local done_steps
    done_steps=$(cat "$dir/checkpoints/last/training_state/training_step.json" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    if [ "${done_steps:-0}" -ge "$STEPS" ]; then
      echo "[$(date +%H:%M:%S)] run $name already complete ($done_steps steps), skipping"
      return 0
    fi
    echo "[$(date +%H:%M:%S)] run $name resuming from step ${done_steps:-?}"
    resume="--resume=true"
  fi

  echo "[$(date +%H:%M:%S)] === run $name: $repo chunk=$chunk steps=$STEPS ==="
  if [ -n "$resume" ]; then
    WANDB_MODE=disabled "$VENV/lerobot-train" \
      --config_path="$dir/checkpoints/last/pretrained_model/train_config.json" \
      --resume=true >> "$log" 2>&1
  else
    WANDB_MODE=disabled "$VENV/lerobot-train" \
      --dataset.repo_id="$repo" \
      --dataset.root="$root" \
      --dataset.video_backend=torchcodec \
      --dataset.episodes="$TRAIN_EPS" \
      --policy.type=smolvla \
      --policy.pretrained_path=lerobot/smolvla_base \
      --policy.load_vlm_weights=true \
      --policy.chunk_size="$chunk" \
      --policy.n_action_steps="$chunk" \
      --policy.device=cuda \
      --policy.use_amp=true \
      --policy.push_to_hub=false \
      --policy.repo_id="local/smolvla-excav-long-$name" \
      --output_dir="$dir" \
      --job_name="excav_long_$name" \
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
[ ${#RUNS[@]} -eq 0 ] && RUNS=(A B C D)

echo "long sweep start $(date)"
echo "held-out episodes: $VAL_EPISODES"
for r in "${RUNS[@]}"; do train_one "$r"; done
echo "long sweep done $(date)"
