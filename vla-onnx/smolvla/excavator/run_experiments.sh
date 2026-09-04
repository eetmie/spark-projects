#!/usr/bin/env bash
# Sequential SmolVLA fine-tune sweep on the masi_excavator dataset.
#
# The axis under test is temporal: the data is 30 fps but the machine is slow,
# so consecutive frames are nearly redundant. Runs B and C decimate the data;
# run D keeps 30 fps but stretches the action chunk to the same wall-clock
# horizon as B, which separates "less visual redundancy" from "longer horizon".
#
# Everything else (seed, lr, batch size, optimizer steps, held-out episodes) is
# held fixed so the runs are comparable.
#
#   bash run_experiments.sh            # run all four
#   bash run_experiments.sh A D        # run a subset

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../paths.sh"

PROJ=$ROOT
VENV=$VENV_LEROBOT051/bin
OUT=$PROJ/outputs/excavator
LOGS=$OUT/logs

STEPS=8000
BATCH=32
LR=1e-4
SEED=1000
SAVE_FREQ=2000
WORKERS=8

# 4 of 31 episodes held out for the post-hoc comparison, spread across the session.
VAL_EPISODES="3 11 19 27"
TRAIN_EPS=$(python3 -c "
val={$(echo $VAL_EPISODES | tr ' ' ',')}
print('['+','.join(str(e) for e in range(31) if e not in val)+']')")

DS_30=$VLA_DATASETS/masi_kaivuri_juusto
DS_10=$PROJ/datasets/masi_kaivuri_10fps
DS_06=$PROJ/datasets/masi_kaivuri_6fps

mkdir -p "$LOGS"

# name | dataset root | repo_id | chunk_size
config_for() {
  case "$1" in
    A) echo "$DS_30|local/masi_kaivuri_juusto|50"  ;;  # 30 fps, 1.67 s horizon  (baseline)
    B) echo "$DS_10|local/masi_kaivuri_10fps|50"   ;;  # 10 fps, 5.00 s horizon  (decimated)
    C) echo "$DS_06|local/masi_kaivuri_6fps|50"    ;;  #  6 fps, 8.33 s horizon  (decimated harder)
    D) echo "$DS_30|local/masi_kaivuri_juusto|150" ;;  # 30 fps, 5.00 s horizon  (horizon control)
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

  if [ -f "$dir/checkpoints/last/pretrained_model/model.safetensors" ]; then
    echo "[$(date +%H:%M:%S)] run $name already has a checkpoint, skipping"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] === run $name: $repo chunk=$chunk steps=$STEPS ==="
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
    --policy.repo_id="local/smolvla-excav-$name" \
    --output_dir="$dir" \
    --job_name="excav_$name" \
    --seed=$SEED \
    --steps=$STEPS \
    --batch_size=$BATCH \
    --num_workers=$WORKERS \
    --log_freq=100 \
    --save_freq=$SAVE_FREQ \
    --eval_freq=0 \
    --optimizer.lr=$LR \
    > "$log" 2>&1

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

echo "sweep start $(date)"
echo "held-out episodes: $VAL_EPISODES"
for r in "${RUNS[@]}"; do train_one "$r"; done
echo "sweep done $(date)"
