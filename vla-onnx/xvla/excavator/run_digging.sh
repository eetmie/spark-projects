#!/usr/bin/env bash
# X-VLA-0.9B fine-tune on the excavator datasets, set up to be scored against the
# SmolVLA runs on exactly the same footing.
#
#   ir    observation.images.cam1               (D435i infrared left imager)
#   both  observation.images.cam1 + .cam2       (IR + RGB colour imager)
#
#   bash run_digging.sh              # full-data IR and both-camera runs, sequentially
#   bash run_digging.sh clean_ir     # cleaned IR data (recommended first real run)
#   CHUNK=30 bash run_digging.sh clean_ir
#   CHUNK=20 NSTEPS=10 bash run_digging.sh dry2_ir   # predict 20, execute 10
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
#  --policy.chunk_size / n_action_steps (CHUNK / NSTEPS, default 50/50)
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

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../paths.sh"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SMOLVLA=$VLA_ONNX/smolvla                    # shared 0.5.1 env, on purpose
VENV=$VENV_LEROBOT051/bin
OUT=${OUT:-$ROOT/outputs/digging}   # overridable so a smoke test cannot pollute the real run
LOGS=$OUT/logs

STEPS=${STEPS:-20000}
BATCH=${BATCH:-32}
CHUNK=${CHUNK:-50}
# How many of the predicted actions are actually executed before replanning. Defaults to
# CHUNK (predict N, execute N), which is what every run up to 2026-09-01 did. Setting it
# lower decouples the two: the policy still learns a CHUNK-long trajectory -- more context
# to be consistent with -- while the robot commits to only the first NSTEPS of it and
# replans, so a stale plan is never executed to its end. CHUNK=20 NSTEPS=10 at 30 fps =
# a 0.67 s prediction of which 0.33 s is used.
NSTEPS=${NSTEPS:-$CHUNK}

# FULL_FT=1 trains the VLM too, which is what X-VLA's own docs prescribe:
#   "the Vision-Language Model (VLM) must be trained with only 1/10 of the base learning
#    rate, while all other components use the full LR. This LR ratio is crucial for
#    achieving strong and stable finetuning performance."
# That 1/10 is already implemented -- XVLAAdamWConfig puts anything matching "vlm" in a
# param group at lr*0.1 -- so full finetuning needs only the two freeze flags off.
#
# Every run before 2026-09-02 froze both encoders instead. That came from
# configuration_xvla.py's own docstring ("By default, VLM encoders are frozen and only
# policy transformer + soft prompts train"), which contradicts BOTH the documentation and
# its own literal defaults (False). Freezing also made the lr*0.1 group -- the mechanism
# the docs call crucial -- collect nothing but `model.transformer.vlm_proj`, a policy-side
# Linear caught by the substring filter, which therefore trained at 1/10 the LR of the
# transformer it belongs to. Unfreezing fixes that as a side effect.
#
# Cost: learnable params go 311M -> 879M and the backward pass adds DaViT + BART.
FULL_FT=${FULL_FT:-0}
if [ "$FULL_FT" = "1" ]; then
  FREEZE_VISION=false; FREEZE_LANGUAGE=false
else
  FREEZE_VISION=true;  FREEZE_LANGUAGE=true
fi

# Soft-prompt warmup. The docs: "Completely matching the official reported performance may
# require an additional warm-up LR schedule for soft-prompts, which can bring minor
# improvements." Setting this starts the soft-prompt group at lr*SP_WARMUP and lets the
# existing warmup schedule bring it up. Empty = the default (no separate warmup).
SP_WARMUP=${SP_WARMUP:-}
DTYPE=${DTYPE:-bfloat16}
SEED=${SEED:-1000}
SAVE_FREQ=${SAVE_FREQ:-2500}
WORKERS=${WORKERS:-10}
LOG_FREQ=${LOG_FREQ:-250}
SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-true}
NORM=${NORM:-'{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'}

# scheduler_decay_steps. XVLAConfig defaults this to 30000, and
# CosineDecayWithWarmupSchedulerConfig.build() only ever scales it DOWN:
#   if num_training_steps < self.num_decay_steps: ... actual_decay_steps = num_training_steps
# Below 30000 that auto-fit is exactly what we want -- it compresses warmup AND decay so the
# cosine lands on the final step (this is why the 10k run annealed all the way to 2.5e-7).
# Above 30000 nothing scales: the LR reaches the 2.5e-6 floor at step 30000 and every step
# after that trains at the floor. A 50000-step run would spend its last 20000 steps -- about
# 11 hours at batch 64 -- going nowhere, and would report a perfectly healthy loss while
# doing it. So when STEPS exceeds the built-in horizon, match the decay to the run length.
# Set DECAY explicitly to override (e.g. DECAY=30000 to deliberately reproduce the old shape).
DECAY=${DECAY:-}
if [ -z "$DECAY" ] && [ "$STEPS" -gt 30000 ]; then
  DECAY=$STEPS
fi

# Local checkpoint dir rather than the `lerobot/xvla-base` repo id: huggingface_hub's
# downloader stalled twice mid-file on this box (dead at 245-257 MB of 3519 with the
# process still alive), so the 3.5 GB safetensors is fetched once with curl -C - and
# reused. Fetch with:
#   bash excavator/fetch_checkpoint.sh
CKPT=${CKPT:-$ROOT/models/xvla-base-excavator}

DS_BOTH=$VLA_DATASETS/masi_digging
DS_IR=$SMOLVLA/datasets/masi_digging_ir
DS_CLEAN=$SMOLVLA/datasets/masi_digging_clean
DS_CLEAN_IR=$SMOLVLA/datasets/masi_digging_clean_ir
DS_DRY_IR=$SMOLVLA/datasets/masi_digging_dry_ir
# 2026-08-31 session, 78 eps / 65655 frames: the first TWO-TASK recording (eps 0-62
# "move sand to container", 63-77 "move rock to container"). Nothing here needs to know
# that -- the instruction is a per-frame column -- but the eval and the export bundle do;
# see the SmolVLA side's digging_dry2 presets and export_split_onnx's `tasks` list.
DS_DRY2_IR=$SMOLVLA/datasets/masi_digging_dry2_ir

# Episode count is read from each dataset. masi_digging grew from 82 to 189 episodes;
# the old literal range(82) silently discarded every appended recording. By default we
# hold out every tenth episode from 5, matching the current SmolVLA runner. For cleaned
# data, queue_digging.sh exports the remapped held-out ids so both architectures score
# the same source recordings after episodes 83-90 were dropped and survivors renumbered.
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

if [ ! -f "$CKPT/model.safetensors" ] || [ ! -f "$CKPT/config.json" ]; then
  echo "missing X-VLA checkpoint at $CKPT — run: bash excavator/fetch_checkpoint.sh && python excavator/prepare_checkpoint.py"
  exit 1
fi

config_for() {
  case "$1" in
    ir)         echo "$DS_IR|local/masi_digging_ir"             ;;
    both)       echo "$DS_BOTH|local/masi_digging_both"         ;;
    clean_ir)   echo "$DS_CLEAN_IR|local/masi_digging_clean_ir" ;;
    clean_both) echo "$DS_CLEAN|local/masi_digging_clean_both"  ;;
    dry_ir)     echo "$DS_DRY_IR|local/masi_digging_dry_ir"     ;;
    dry2_ir)    echo "$DS_DRY2_IR|local/masi_digging_dry2_ir"   ;;
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
    # TrainPipelineConfig normally gives the saved checkpoint config precedence over
    # the current command line. Explicitly override the run-length controls: otherwise
    # the 250-step throughput probe resumes with steps=250 and exits immediately when
    # the user asks for a 20k run. Architecture/data settings stay checkpoint-owned --
    # including scheduler_decay_steps, so DECAY has NO effect here. A resume cannot be used
    # to extend a run past its original decay horizon; start a fresh run instead.
    WANDB_MODE=disabled "$VENV/lerobot-train" \
      --config_path="$dir/checkpoints/last/pretrained_model/train_config.json" \
      --resume=true \
      --steps="$STEPS" \
      --batch_size="$BATCH" \
      --num_workers="$WORKERS" \
      --log_freq="$LOG_FREQ" \
      --save_freq="$SAVE_FREQ" \
      --save_checkpoint="$SAVE_CHECKPOINT" >> "$log" 2>&1
  else
    local -a sched=()
    if [ -n "$DECAY" ]; then
      sched+=(--policy.scheduler_decay_steps="$DECAY")
    fi
    if [ -n "$SP_WARMUP" ]; then
      sched+=(--policy.optimizer_soft_prompt_warmup_lr_scale="$SP_WARMUP")
    fi
    echo "[$(date +%H:%M:%S)] === xvla run $name: $repo chunk=$CHUNK/$NSTEPS steps=$STEPS ==="
    echo "[$(date +%H:%M:%S)]     $n_eps episodes, held out: $val_eps"
    echo "[$(date +%H:%M:%S)]     decay_steps=${DECAY:-30000 (policy default, auto-fit to STEPS)}"
    echo "[$(date +%H:%M:%S)]     full_finetune=$FULL_FT (freeze vision=$FREEZE_VISION language=$FREEZE_LANGUAGE) soft_prompt_warmup=${SP_WARMUP:-none}"
    WANDB_MODE=disabled "$VENV/lerobot-train" \
      --dataset.repo_id="$repo" \
      --dataset.root="$root" \
      --dataset.video_backend=torchcodec \
      --dataset.episodes="$train_eps" \
      --policy.path="$CKPT" \
      --policy.action_mode=auto \
      --policy.normalization_mapping="$NORM" \
      --policy.chunk_size="$CHUNK" \
      --policy.n_action_steps="$NSTEPS" \
      "${sched[@]}" \
      --policy.dtype="$DTYPE" \
      --policy.freeze_vision_encoder=$FREEZE_VISION \
      --policy.freeze_language_encoder=$FREEZE_LANGUAGE \
      --policy.device=cuda \
      --policy.push_to_hub=false \
      --policy.repo_id="local/xvla-digging-$name" \
      --output_dir="$dir" \
      --job_name="xvla_digging_$name" \
      --seed=$SEED \
      --steps=$STEPS \
      --batch_size=$BATCH \
      --num_workers=$WORKERS \
      --log_freq=$LOG_FREQ \
      --save_freq=$SAVE_FREQ \
      --save_checkpoint=$SAVE_CHECKPOINT \
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
echo "held-out override: ${VAL_EPISODES:-<derived per run: every 10th from 5>}"
echo "steps=$STEPS batch=$BATCH chunk=$CHUNK n_action_steps=$NSTEPS dtype=$DTYPE workers=$WORKERS full_ft=$FULL_FT"
echo "norm=$NORM"
for r in "${RUNS[@]}"; do train_one "$r"; done
echo "xvla digging sweep done $(date)"
