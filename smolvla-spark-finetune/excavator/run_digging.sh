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
OUT=${OUT:-$ROOT/outputs/digging}   # overridable so a new dataset gets a fresh dir
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
# Trimmed of episode-boundary dead air and with the dirty 83-90 block skipped;
# built by make_trim_variant.py + make_camera_variant.py. See NOTES-dataset-trim.md.
DS_CLEAN=$ROOT/datasets/masi_digging_clean
DS_CLEAN_IR=$ROOT/datasets/masi_digging_clean_ir

# Held out for eval_compare/eval_curve, spread across the session so the split is not a
# single stretch of the recording.
# Episode count is READ FROM THE DATASET, never hardcoded. It used to be a literal
# range(82); when masi_digging grew to 189 episodes that would have silently trained on
# episodes 0-81 only and thrown away every new one, with no error and a plausible-looking
# loss curve. Held-out = every 10th episode from 5 (~10%), spread across the recording.
# Derived PER RUN from that run's own dataset, because variants differ in episode
# count: masi_digging has 189, masi_digging_clean has 181 (the dirty 83-90 block was
# dropped). Deriving once from DS_BOTH would have given a cleaned run the wrong split
# silently.
#
# VAL_EPISODES can be overridden. Dropping episodes RENUMBERS the survivors, so holding
# out "every 10th from 5" on a cleaned dataset scores different recordings than the same
# rule on the source. To compare a cleaned run against an earlier sweep, pass the mapped
# indices explicitly, e.g. for masi_digging_clean (source 83-90 removed):
#   VAL_EPISODES="5 15 25 35 45 55 65 75 87 97 107 117 127 137 147 157 167 177"
# which is the same 19 recordings the 189-episode sweep held out, less source ep 85.
split_for() {
  local root=$1 n val train
  n=$(python3 -c "import json;print(json.load(open('$root/meta/info.json'))['total_episodes'])")
  val=${VAL_EPISODES:-$(python3 -c "print(' '.join(str(e) for e in range(5,$n,10)))")}
  train=$(python3 -c "
val={$(echo $val | tr ' ' ',')}
print('['+','.join(str(e) for e in range($n) if e not in val)+']')")
  echo "$n|$val|$train"
}

mkdir -p "$LOGS"

config_for() {
  case "$1" in
    ir)         echo "$DS_IR|local/masi_digging_ir"                ;;
    both)       echo "$DS_BOTH|local/masi_digging_both"            ;;
    clean_ir)   echo "$DS_CLEAN_IR|local/masi_digging_clean_ir"    ;;
    # Same dataset as clean_ir -- a separate name only so an aggressive-chunk run gets
    # its own output dir instead of colliding with the chunk-50 one. Pass CHUNK=12.
    clean_ir12) echo "$DS_CLEAN_IR|local/masi_digging_clean_ir"    ;;
    clean_both) echo "$DS_CLEAN|local/masi_digging_clean_both"     ;;
    *) return 1 ;;
  esac
}

train_one() {
  local name=$1
  local cfg; cfg=$(config_for "$name") || { echo "unknown run '$name'"; return 1; }
  local root=${cfg%%|*}; local repo=${cfg##*|}

  local dir=$OUT/$name
  local log=$LOGS/$name.log

  local sp; sp=$(split_for "$root")
  local n_eps=${sp%%|*}; local _rest=${sp#*|}
  local val_eps=${_rest%%|*}; local train_eps=${_rest##*|}

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
    echo "[$(date +%H:%M:%S)]     $n_eps episodes, held out: $val_eps"
    WANDB_MODE=disabled "$VENV/lerobot-train" \
      --dataset.repo_id="$repo" \
      --dataset.root="$root" \
      --dataset.video_backend=torchcodec \
      --dataset.episodes="$train_eps" \
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
echo "held-out override: ${VAL_EPISODES:-<derived per run: every 10th from 5>}"
echo "steps=$STEPS batch=$BATCH chunk=$CHUNK lr=$LR workers=$WORKERS"
for r in "${RUNS[@]}"; do train_one "$r"; done
echo "digging sweep done $(date)"
