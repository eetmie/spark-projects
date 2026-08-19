#!/usr/bin/env bash
# X-VLA-0.9B fine-tune on masi_digging, set up to be scored against the SmolVLA runs
# in ../../smolvla-spark-finetune/outputs/digging/ on exactly the same footing.
#
#   ir    observation.images.cam1               (D435i infrared left imager)
#   both  observation.images.cam1 + .cam2       (IR + RGB colour imager)
#
#   bash run_digging.sh              # both runs, sequentially
#   bash run_digging.sh ir           # just the IR one — answers the SmolVLA-vs-X-VLA
#                                    # question at half the cost
#   STEPS=2000 bash run_digging.sh ir
#
# This is a SEPARATE pipeline from the SmolVLA one on purpose (the SmolVLA side is still
# moving). It deliberately shares two things, because sharing them is what makes the
# comparison mean anything:
#
#   * the SAME venv — lerobot 0.5.1, torch 2.12.0+cu130. Identical stack, no version
#     variable smuggled into the result. (The Orin xvla-runtime README says 0.5.1 has no
#     xvla policy; that is not true of THIS install — XVLAConfig/XVLAPolicy import fine.
#     If the Orin later runs lerobot 0.6.1, check the config contract before deploying.)
#   * the SAME datasets and held-out episodes as run_digging.sh on the SmolVLA side,
#     including the metadata-only cam1 view built by make_camera_variant.py.
#
# ---------------------------------------------------------------------------------
# Choices that are NOT X-VLA's defaults, and why
#
#  --policy.action_mode=auto
#      Default is "ee6d": a 20-dim end-effector arm space that the excavator is not.
#      "auto" reads the real action dim from the dataset (4 = slew/lift/tilt/scoop),
#      pads to max_action_dim=20 so the pretrained head still loads, computes the loss
#      on the first 4 dims only, and trims the output back to 4. Without this the
#      policy is trained against an action space the robot does not have.
#
#  --policy.normalization_mapping STATE=MEAN_STD  (ACTION left at the checkpoint's MEAN_STD)
#      Read the CHECKPOINT, not the class defaults — they disagree. XVLAConfig's dataclass
#      defaults are IDENTITY for all three, but lerobot/xvla-base actually ships
#      {STATE: IDENTITY, ACTION: MEAN_STD, VISUAL: IDENTITY}.
#      Only STATE needs changing. X-VLA does NOT normalize proprio internally
#      (_prepare_state only pads to max_state_dim), and our state is joint angles in
#      DEGREES — [lift, tilt, scoop] spanning roughly -114..134 — so IDENTITY would push
#      raw two- and three-digit values straight into the proprio projection. That default
#      suits pre-normalized pose data, not this.
#      ACTION stays MEAN_STD because that is what the pretrained action head was trained
#      with, and it is also what SmolVLA uses — so after this override the two
#      architectures normalize identically and the comparison has one less confound.
#
#  --policy.chunk_size=50 / n_action_steps=50
#      X-VLA defaults to 32, which covers only 1.07 s at 30 fps — less than the 1.5 s
#      scoring horizon, so eval_compare could not score it at all. 50 matches SmolVLA
#      exactly (1.67 s), which is the whole point. NOTE for deployment: the Orin
#      xvla-runtime numbers (390 ms/chunk, 2.56 Hz) were measured at 30 actions; the
#      denoise loop re-runs all 24 blocks over the full sequence 10 times with no KV
#      cache, so a 50-action chunk will be materially slower than that. Re-measure
#      before promising a replan rate.
#
#  --policy.freeze_vision_encoder=true --policy.freeze_language_encoder=true
#      The config's own docstring says "By default, VLM encoders are frozen and only
#      policy transformer + soft prompts train", but the literal defaults are False.
#      Freezing both matches the documented intent AND mirrors the SmolVLA runs
#      (freeze_vision_encoder=True, train_expert_only=True), so both architectures are
#      adapting a comparable slice of themselves on 74 episodes.
#
#  domain_id
#      Left at the default 0 — _get_domain_id falls back to zeros when the batch has no
#      domain_id and domain_feature_key is unset. Matches the Orin export (--domain-id 0).
#
#  --policy.path (NOT --policy.type=xvla --policy.pretrained_path=...)
#      This one is not cosmetic. With `--policy.type`, draccus builds XVLAConfig from its
#      DEFAULTS plus CLI overrides, and `florence_config` defaults to `{}` — so
#      get_florence_config() dies with "vision_config is required" before training starts.
#      `--policy.path` takes the branch in TrainPipelineConfig.validate() that calls
#      PreTrainedConfig.from_pretrained(path, cli_overrides=...), which loads the
#      checkpoint's real config (Florence-2 vision/text configs included) and then applies
#      the overrides below on top. SmolVLA tolerates the --policy.type form because its
#      config carries no such nested block; X-VLA does not.
# ---------------------------------------------------------------------------------

set -uo pipefail

ROOT=/home/masi-pgx/spark-projects/xvla-spark-finetune
SMOLVLA=/home/masi-pgx/spark-projects/smolvla-spark-finetune
VENV=$SMOLVLA/.venv/bin                      # shared on purpose — see above
OUT=${OUT:-$ROOT/outputs/digging}   # overridable so a smoke test cannot pollute the real run
LOGS=$OUT/logs

STEPS=${STEPS:-20000}
BATCH=${BATCH:-32}
CHUNK=${CHUNK:-50}
SEED=${SEED:-1000}
SAVE_FREQ=${SAVE_FREQ:-2500}
WORKERS=${WORKERS:-10}
NORM=${NORM:-'{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'}

# Local checkpoint dir rather than the `lerobot/xvla-base` repo id: huggingface_hub's
# downloader stalled twice mid-file on this box (dead at 245-257 MB of 3519 with the
# process still alive), so the 3.5 GB safetensors is fetched once with curl -C - and
# reused. Mirrors ../orin-nano/xvla-runtime/models/xvla-base/. Fetch with:
#   bash excavator/fetch_checkpoint.sh
CKPT=${CKPT:-$ROOT/models/xvla-base-excavator}

DS_BOTH=/home/masi-pgx/Desktop/masi_digging
DS_IR=$SMOLVLA/datasets/masi_digging_ir

# Identical split to the SmolVLA sweep — the two are only comparable if the held-out
# episodes are the same ones.
VAL_EPISODES="5 15 25 35 45 55 65 75"
TRAIN_EPS=$(python3 -c "
val={$(echo $VAL_EPISODES | tr ' ' ',')}
print('['+','.join(str(e) for e in range(82) if e not in val)+']')")

mkdir -p "$LOGS"

if [ ! -f "$CKPT/model.safetensors" ] || [ ! -f "$CKPT/config.json" ]; then
  echo "missing X-VLA checkpoint at $CKPT — run: bash excavator/fetch_checkpoint.sh && python excavator/prepare_checkpoint.py"
  exit 1
fi

config_for() {
  case "$1" in
    ir)   echo "$DS_IR|local/masi_digging_ir"     ;;
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
    echo "[$(date +%H:%M:%S)] === xvla run $name: $repo chunk=$CHUNK steps=$STEPS ==="
    WANDB_MODE=disabled "$VENV/lerobot-train" \
      --dataset.repo_id="$repo" \
      --dataset.root="$root" \
      --dataset.video_backend=torchcodec \
      --dataset.episodes="$TRAIN_EPS" \
      --policy.path="$CKPT" \
      --policy.action_mode=auto \
      --policy.normalization_mapping="$NORM" \
      --policy.chunk_size="$CHUNK" \
      --policy.n_action_steps="$CHUNK" \
      --policy.freeze_vision_encoder=true \
      --policy.freeze_language_encoder=true \
      --policy.device=cuda \
      --policy.push_to_hub=false \
      --policy.repo_id="local/xvla-digging-$name" \
      --output_dir="$dir" \
      --job_name="xvla_digging_$name" \
      --seed=$SEED \
      --steps=$STEPS \
      --batch_size=$BATCH \
      --num_workers=$WORKERS \
      --log_freq=250 \
      --save_freq=$SAVE_FREQ \
      --eval_freq=0 \
      > "$log" 2>&1
  fi

  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] xvla run $name finished ok"
  else
    echo "[$(date +%H:%M:%S)] xvla run $name FAILED (rc=$rc), see $log"
  fi
  return $rc
}

RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(ir both)

echo "xvla digging sweep start $(date)"
echo "held-out episodes: $VAL_EPISODES"
echo "steps=$STEPS batch=$BATCH chunk=$CHUNK workers=$WORKERS"
echo "norm=$NORM"
for r in "${RUNS[@]}"; do train_one "$r"; done
echo "xvla digging sweep done $(date)"
