#!/usr/bin/env bash
# Fine-tune a VLA on this machine. One entry point, three models.
#
#   ./run_training.sh --model smolvla --dataset masi_digging_dry_2 --cameras cam1 \
#                     --train-mode expert --chunk 10 --steps 30000
#
#   ./run_training.sh --model xvla --dataset masi_digging_dry_2 --cameras cam1 \
#                     --chunk 30 --execute 10 --train-mode full --steps 20000
#
#   ./run_training.sh --model smolvla --dataset X --cameras cam1 --dry-run
#
# This replaces 15 scripts: two `run_digging.sh`, seven `queue_*.sh`, two chain scripts
# and four that had been dead since the tree was renamed (`PROJ=$ROOT` under `set -u`,
# and paths.sh exports VLA_ONNX, never ROOT). What they encoded that was worth keeping
# is below, as behaviour rather than as prose.
#
# WHAT USED TO BE INVISIBLE, AND NOW IS NOT
#
#   * The training mode. Every SmolVLA excavator run trained 99,880,992 of 450,046,176
#     parameters because `freeze_vision_encoder` / `train_expert_only` / `train_state_proj`
#     all default to true in the checkpoint's config and no script ever passed them. The
#     X-VLA 20k full-finetune that beat the 10k frozen run was produced by a hand-typed
#     FULL_FT=1 and survives only as the directory string someone chose. `--train-mode` is
#     now explicit, and it goes into the run name, so the artefact says what it is.
#
#   * The camera set. It used to be whichever pre-built dataset view a case statement
#     happened to point at. `--cameras` builds or refreshes that view (see
#     vla_common.dataset.view), which also closes the trap where a grown recording left
#     the view describing the old episode count.
#
#   * The learning rate. `--optimizer.lr` was a NO-OP: TrainPipelineConfig.validate()
#     replaces the parsed optimizer with policy.get_optimizer_preset() whenever
#     use_policy_training_preset is true, which is the default. Every run's --optimizer.lr
#     was discarded. It happened to equal the policy default so nothing was harmed, but
#     LR=3e-5 would have silently trained at 1e-4. --lr maps to --policy.optimizer_lr.
#
# WAITING FOR THE GPU. `--after-pid` blocks on /proc/<pid>. Do not reach for `pgrep -f`:
# it matches any process whose argv contains the pattern, including the agent shell that
# is tailing the run, so a queue can sit "waiting for the GPU" forever with nothing wrong
# and nothing logged. That cost one overnight scoring run. Wait on a fact, not a guess.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/paths.sh"

usage() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    cat <<'USAGE'

Passed straight to lerobot-train -- same names it uses, --flag=value also accepted.
The short forms are aliases.

  --steps N                              (default 30000)
  --seed N                               (default 1000)
  --batch_size N          / --batch      (default 32)
  --num_workers N         / --workers    (default 10)
  --save_freq N           / --save-freq  (default 2500)
  --log_freq N            / --log-freq   (default 250)
  --policy.chunk_size N   / --chunk      (default 50)
  --policy.n_action_steps N / --execute  (default: = chunk_size)
  --policy.optimizer_lr X / --lr         (default: per model)
  --policy.load_vlm_weights BOOL / --no-vlm-weights   smolvla (default true)
  -- ARGS...              everything after this goes to lerobot-train verbatim; a flag
                          this script already sets is refused, naming the owning flag

This script's own -- they compute something, so they are NOT lerobot spellings:

  --model M            smolvla | xvla | evo1                            (required)
  --dataset NAME       a recording under $VLA_DATASETS, or a path       (required)
  --cameras a,b        cameras to train on; builds the dataset view (default: all)
  --train-mode M       expert | frozen | full | lora   (default: per model)
  --holdout SPEC       every10@5 | "5 15 25" | none    (-> --dataset.episodes)
  --init-from PATH     start from this checkpoint instead of the base weights
  --name S             run dir name (default: <cams>-c<chunk>[x<exec>]-<mode>)
  --sweep S            sweep dir name (default: the dataset name)
  --out DIR            the SWEEP dir, <playbook>/outputs/<sweep>. Not lerobot's
                       --output_dir, which is the run dir inside it.
  --after-pid N        wait for /proc/N to disappear, then start
  --if-stale MODE      camera view staleness: rebuild | reuse | refuse | force
  --smoke              2 steps, tiny batch, no checkpoint, disposable output dir
  --dry-run            print the resolved lerobot-train command and exit
  --force              start fresh over an existing run dir
  --resume-anyway      resume even if this invocation disagrees with the checkpoint
USAGE
}

die() { echo "!! $*" >&2; exit 2; }

MODEL="" DATASET="" CAMERAS="" MODE="" NAME="" SWEEP="" OUT=""
CHUNK=50 EXECUTE="" STEPS=30000 BATCH=32 LR="" SEED=1000
SAVE_FREQ=2500 WORKERS=10 LOG_FREQ=250 HOLDOUT="every10@5"
AFTER_PID="" IF_STALE=rebuild INIT_FROM="" VLM_WEIGHTS=true
DRY_RUN=0 SMOKE=0 FORCE=0 RESUME_ANYWAY=0
PASSTHROUGH=()

# lerobot-train spells its flags --flag=value; accept that everywhere, and split it
# into --flag value before dispatch. Everything after a bare `--` is passthrough and is
# left exactly as typed.
ARGS=() ; seen_ddash=0
for a in "$@"; do
    if [ "$seen_ddash" = 1 ]; then ARGS+=("$a"); continue; fi
    case "$a" in
        --) seen_ddash=1; ARGS+=("$a") ;;
        --*=*) ARGS+=("${a%%=*}" "${a#*=}") ;;
        *) ARGS+=("$a") ;;
    esac
done
set -- ${ARGS+"${ARGS[@]}"}

# Where a flag is a straight pass to lerobot-train, it is SPELLED THE WAY LEROBOT SPELLS
# IT, so there is nothing new to learn and `--batch_size=32` works here exactly as it
# does there. The short forms are aliases, not the real names.
#
# Flags that are NOT a straight pass keep their own names on purpose -- they compute
# something, and giving them lerobot's spelling would promise passthrough semantics that
# do not hold. `--out` is the clearest case: it is the SWEEP directory, while lerobot's
# --output_dir is the RUN directory inside it. Same word, different thing, different flag.
while [ $# -gt 0 ]; do
    case "$1" in
        # --- computed: no lerobot equivalent -----------------------------------------
        --model)         MODEL=$2; shift 2 ;;
        --dataset)       DATASET=$2; shift 2 ;;
        --cameras)       CAMERAS=$2; shift 2 ;;
        --train-mode)    MODE=$2; shift 2 ;;
        --holdout)       HOLDOUT=$2; shift 2 ;;
        --name)          NAME=$2; shift 2 ;;
        --sweep)         SWEEP=$2; shift 2 ;;
        --out)           OUT=$2; shift 2 ;;
        --init-from)     INIT_FROM=$2; shift 2 ;;
        --after-pid)     AFTER_PID=$2; shift 2 ;;
        --if-stale)      IF_STALE=$2; shift 2 ;;
        --smoke)         SMOKE=1; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --force)         FORCE=1; shift ;;
        --resume-anyway) RESUME_ANYWAY=1; shift ;;
        # --- lerobot's own spelling, plus a short alias ------------------------------
        --steps)                      STEPS=$2; shift 2 ;;
        --seed)                       SEED=$2; shift 2 ;;
        --batch_size|--batch)         BATCH=$2; shift 2 ;;
        --num_workers|--workers)      WORKERS=$2; shift 2 ;;
        --save_freq|--save-freq)      SAVE_FREQ=$2; shift 2 ;;
        --log_freq|--log-freq)        LOG_FREQ=$2; shift 2 ;;
        --policy.chunk_size|--chunk)  CHUNK=$2; shift 2 ;;
        --policy.n_action_steps|--execute) EXECUTE=$2; shift 2 ;;
        --policy.optimizer_lr|--lr)   LR=$2; shift 2 ;;
        --policy.load_vlm_weights)    VLM_WEIGHTS=$2; shift 2 ;;
        --no-vlm-weights)             VLM_WEIGHTS=false; shift ;;
        --)              shift; PASSTHROUGH=("$@"); break ;;
        -h|--help)       usage; exit 0 ;;
        *) die "unknown flag: $1  (--help for the list)" ;;
    esac
done

[ -n "$MODEL" ]   || { usage; die "--model is required"; }
[ -n "$DATASET" ] || { usage; die "--dataset is required"; }

# --- model dispatch ------------------------------------------------------------------
# The three arms below are the whole reason this is one file with a case statement and
# not one shared invocation: the models spell "which weights" three incompatible ways,
# and collapsing them breaks at startup with an error that names nothing relevant.
case "$MODEL" in
    smolvla) VENV=$VENV_LEROBOT051; PLAYBOOK=$VLA_ONNX/smolvla; : "${MODE:=expert}"; : "${LR:=1e-4}" ;;
    xvla)    VENV=$VENV_LEROBOT051; PLAYBOOK=$VLA_ONNX/xvla;    : "${MODE:=frozen}"; : "${LR:=1e-4}" ;;
    evo1)    VENV=$VENV_LEROBOT061; PLAYBOOK=$VLA_ONNX/evo1;    : "${MODE:=expert}"; : "${LR:=1e-5}" ;;
    *) die "unknown --model $MODEL (smolvla | xvla | evo1)" ;;
esac
PY=$VENV/bin/python
[ -x "$PY" ] || die "no python in $VENV -- run the playbook's setup.sh first"

# --- dataset + camera view -----------------------------------------------------------
SRC=$DATASET
[ -d "$SRC" ] || SRC=$VLA_DATASETS/$DATASET
[ -d "$SRC" ] || die "no such recording: $DATASET (looked in $VLA_DATASETS)"

VIEW_ARGS=(--src "$SRC" --if-stale "$IF_STALE")
[ -n "$CAMERAS" ] && VIEW_ARGS+=(--cameras "$CAMERAS")
ROOT=$("$PY" -m vla_common.dataset.view "${VIEW_ARGS[@]}") || exit 2
REPO_ID=local/$(basename "$ROOT")

# --- split ---------------------------------------------------------------------------
SPLIT_ARGS=(--root "$ROOT")
case "$HOLDOUT" in
    every10@5|"") ;;
    none) SPLIT_ARGS=() ;;
    *) SPLIT_ARGS+=(--val "$HOLDOUT") ;;
esac
if [ "$HOLDOUT" = "none" ]; then
    TRAIN_EPS=""
else
    TRAIN_EPS=$("$PY" -m vla_common.dataset.split "${SPLIT_ARGS[@]}" --emit train) || exit 2
    HELD_OUT=$("$PY" -m vla_common.dataset.split "${SPLIT_ARGS[@]}" --emit val)
fi

# --- naming --------------------------------------------------------------------------
: "${EXECUTE:=$CHUNK}"
CAM_TAG=$(echo "${CAMERAS:-all}" | tr ',' '+')
CHUNK_TAG=c$CHUNK; [ "$EXECUTE" != "$CHUNK" ] && CHUNK_TAG=c${CHUNK}x${EXECUTE}
: "${NAME:=$CAM_TAG-$CHUNK_TAG-$MODE}"
: "${SWEEP:=$(basename "$SRC")}"
: "${OUT:=$PLAYBOOK/outputs/$SWEEP}"
if [ "$SMOKE" = 1 ]; then
    OUT=$PLAYBOOK/outputs/smoke/$(date +%Y%m%d-%H%M%S)
    STEPS=2 BATCH=2 WORKERS=2 LOG_FREQ=1
fi
DIR=$OUT/$NAME
LOG=$OUT/logs/$NAME.log

# --- train-mode -> per-model flags ---------------------------------------------------
# A mode a model does not have is an error naming the modes it does have. Silently
# falling back is how you end up with two runs called "full" that trained differently.
MODE_FLAGS=()
case "$MODEL:$MODE" in
    smolvla:expert|smolvla:frozen)
        # The checkpoint's own defaults: freeze_vision_encoder, train_expert_only,
        # train_state_proj all true. Passed explicitly so the run records what it did.
        MODE_FLAGS=(--policy.freeze_vision_encoder=true --policy.train_expert_only=true
                    --policy.train_state_proj=true) ;;
    smolvla:full)
        MODE_FLAGS=(--policy.freeze_vision_encoder=false --policy.train_expert_only=false
                    --policy.train_state_proj=true) ;;
    smolvla:lora)
        # LoRA is triggered by the top-level `peft` config being set, NOT by
        # --policy.use_peft, which means "pretrained_path is an adapter dir" and would
        # raise on a fresh fine-tune.
        MODE_FLAGS=(--peft.method_type=LORA --peft.r=8) ;;
    xvla:frozen|xvla:full)
        if [ "$MODE" = full ]; then FV=false; FL=false; else FV=true; FL=true; fi
        MODE_FLAGS=(--policy.freeze_vision_encoder=$FV --policy.freeze_language_encoder=$FL) ;;
    xvla:expert)
        die "--train-mode expert does not exist for xvla.
   X-VLA has no action-head-only mode: freezing both encoders still trains the full
   policy transformer and the soft prompts. Use --train-mode frozen (311M trainable)
   for the nearest equivalent, or --train-mode full (879M)." ;;
    xvla:lora|evo1:lora)
        die "--train-mode lora is only wired for smolvla.
   LoRA needs target modules and only SmolVLA defines _get_default_peft_targets.
   For $MODEL you must choose them and pass --peft.target_modules yourself." ;;
    evo1:expert|evo1:frozen) MODE_FLAGS=(--policy.training_stage=stage1) ;;
    evo1:full)               MODE_FLAGS=(--policy.training_stage=stage2) ;;
    *) die "unknown --train-mode $MODE for $MODEL (expert | frozen | full | lora)" ;;
esac

# --- which weights, and the model's own dialect --------------------------------------
MODEL_FLAGS=()
case "$MODEL" in
    smolvla)
        # load_vlm_weights=true initialises the VLM from the pretrained SmolVLA weights;
        # false trains the action expert from scratch against a randomly initialised one.
        MODEL_FLAGS=(--policy.type=smolvla
                     --policy.pretrained_path="${INIT_FROM:-lerobot/smolvla_base}"
                     --policy.load_vlm_weights=$VLM_WEIGHTS
                     --policy.use_amp=true) ;;
    xvla)
        # NEVER --policy.type here. draccus would build XVLAConfig from defaults with
        # florence_config={}, and get_florence_config() dies "vision_config is required"
        # before training starts, naming nothing that points at this line.
        CKPT=${INIT_FROM:-$PLAYBOOK/models/xvla-base-excavator}
        [ -f "$CKPT/model.safetensors" ] || die "no prepared X-VLA checkpoint at $CKPT
   Run xvla/excavator/fetch_checkpoint.sh, then prepare_checkpoint.py."
        MODEL_FLAGS=(--policy.path="$CKPT"
                     --policy.action_mode=auto
                     --policy.dtype=bfloat16) ;;
    evo1)
        BASE=$PLAYBOOK/models/InternVL3-1B-hf
        [ -d "$BASE" ] || die "no InternVL3 base at $BASE -- run evo1/download_base.sh"
        # EVO1 has never been trained on this data. Its exporter expects
        # observation.images.image{,2,3} while the excavator records cam1, max_views is
        # baked into the export and must be decided before training, and its default
        # normalization is MIN_MAX where the other two land on MEAN_STD. Refuse rather
        # than emit a command that has never run.
        die "--model evo1 has no trained excavator path yet.
   Three things are missing, and none of them are this script's to guess:
     1. a feature-contract adapter: the exporter wants observation.images.image*,
        this recording has $(basename "$SRC")'s cam keys
     2. --policy.max_views must be pinned before training (it is baked into the
        export; a 1-view bundle cannot serve a 2-view runtime)
     3. its normalization default is MIN_MAX, not the MEAN_STD the other two use,
        so an EVO1 run left at defaults is not one-variable against them
   Build those in evo1/, then this arm is a few lines." ;;
esac

# --- assemble ------------------------------------------------------------------------
CMD=("$VENV/bin/lerobot-train"
     --dataset.repo_id="$REPO_ID"
     --dataset.root="$ROOT"
     --dataset.video_backend=torchcodec
     --policy.chunk_size="$CHUNK"
     --policy.n_action_steps="$EXECUTE"
     --policy.device=cuda
     --policy.push_to_hub=false
     --policy.repo_id="local/$MODEL-$SWEEP-$NAME"
     --policy.optimizer_lr="$LR"
     --output_dir="$DIR"
     --job_name="${SWEEP}_${NAME}"
     --seed="$SEED"
     --steps="$STEPS"
     --batch_size="$BATCH"
     --num_workers="$WORKERS"
     --log_freq="$LOG_FREQ"
     --save_freq="$SAVE_FREQ"
     --eval_freq=0)
[ -n "$TRAIN_EPS" ] && CMD+=(--dataset.episodes="$TRAIN_EPS")
[ "$SMOKE" = 1 ] && CMD+=(--save_checkpoint=false)
CMD+=("${MODEL_FLAGS[@]}" "${MODE_FLAGS[@]}")

# Anything after `--` goes to lerobot-train untouched -- the escape hatch for the long
# tail of policy knobs that will never all have their own flag here. It may NOT shadow a
# flag this script derives: a passthrough --policy.chunk_size=30 alongside --chunk 10
# would train one thing and name the run another, which is the exact invisibility this
# script exists to remove. The check is against what was actually assembled, so it stays
# correct as flags are added.
for extra in ${PASSTHROUGH+"${PASSTHROUGH[@]}"}; do
    case "$extra" in
        --*) key=${extra%%=*}
             for own in "${CMD[@]}"; do
                 [ "${own%%=*}" = "$key" ] || continue
                 die "$key is set by this script ($own).
   Passing it again after -- would train one thing and name the run another.
   Use the script's own flag, or --name to say what the run is."
             done ;;
    esac
done
CMD+=(${PASSTHROUGH+"${PASSTHROUGH[@]}"})

summary() {
    echo "model      $MODEL   mode $MODE   lr $LR"
    echo "dataset    $(basename "$SRC")  ->  $ROOT"
    [ -n "$TRAIN_EPS" ] && echo "split      $(echo "$HELD_OUT" | wc -w) held out: $HELD_OUT"
    echo "chunk      $CHUNK predicted, $EXECUTE executed"
    echo "budget     $STEPS steps, batch $BATCH, save every $SAVE_FREQ"
    echo "output     $DIR"
    [ -n "$INIT_FROM" ] && echo "init from  $INIT_FROM"
    [ ${#PASSTHROUGH[@]} -gt 0 ] && echo "passthrough ${PASSTHROUGH[*]}"
    return 0
}

if [ "$DRY_RUN" = 1 ]; then
    summary; echo
    echo 'WANDB_MODE=disabled \\'
    # Quote only what needs it. %q would escape every bracket and comma in the episode
    # list into an unreadable wall, and that list is the one argument worth eyeballing.
    for a in "${CMD[@]}"; do
        case "$a" in
            *[[:space:]\'\"]*) printf '  %q \\\n' "$a" ;;
            *)                  printf '  %s \\\n' "$a" ;;
        esac
    done | sed '$ s/ \\$//'
    exit 0
fi

# --- wait, then run ------------------------------------------------------------------
if [ -n "$AFTER_PID" ]; then
    echo "[$(date +%T)] waiting for pid $AFTER_PID to exit ..."
    while [ -d "/proc/$AFTER_PID" ]; do sleep 30; done
    echo "[$(date +%T)] pid $AFTER_PID gone"
    sleep 10
fi

mkdir -p "$OUT/logs"

if [ -d "$DIR/checkpoints/last" ] && [ "$FORCE" = 0 ]; then
    DONE=$(grep -oE "[0-9]+" "$DIR/checkpoints/last/training_state/training_step.json" 2>/dev/null | head -1)
    if [ "${DONE:-0}" -ge "$STEPS" ]; then
        echo "$NAME already at ${DONE} steps (>= $STEPS), nothing to do"; exit 0
    fi
    # A resume cannot change the architecture, the data or the LR schedule -- those live
    # in the checkpoint. Say which, rather than accepting the flags and ignoring them.
    if [ "$RESUME_ANYWAY" = 0 ]; then
        "$PY" - "$DIR/checkpoints/last/pretrained_model/train_config.json" \
               "$ROOT" "$CHUNK" "$EXECUTE" "$LR" <<'PY' || exit 2
import json, sys
cfg = json.loads(open(sys.argv[1]).read())
root, chunk, execute, lr = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
want = {"dataset.root": (cfg["dataset"]["root"], root),
        "policy.chunk_size": (cfg["policy"]["chunk_size"], chunk),
        "policy.n_action_steps": (cfg["policy"]["n_action_steps"], execute),
        "policy.optimizer_lr": (float(cfg["policy"]["optimizer_lr"]), lr)}
bad = {k: v for k, v in want.items() if str(v[0]) != str(v[1])}
if bad:
    print("!! resume cannot honour these -- they are owned by the checkpoint:", file=sys.stderr)
    for k, (saved, req) in bad.items():
        print(f"     {k:<24} saved {saved}   requested {req}", file=sys.stderr)
    print("   Start a fresh run (--name ...) or pass --resume-anyway.", file=sys.stderr)
    sys.exit(2)
PY
    fi
    echo "[$(date +%T)] resuming $NAME from step ${DONE:-?} -> $STEPS"
    # Re-pass the budget. A bare --config_path --resume=true keeps the OLD step count,
    # which is how resuming a 250-step smoke probe silently stayed 250 steps.
    WANDB_MODE=disabled "$VENV/bin/lerobot-train" \
        --config_path="$DIR/checkpoints/last/pretrained_model/train_config.json" \
        --resume=true --steps="$STEPS" --batch_size="$BATCH" --num_workers="$WORKERS" \
        --log_freq="$LOG_FREQ" --save_freq="$SAVE_FREQ" >> "$LOG" 2>&1
else
    [ "$FORCE" = 1 ] && rm -rf "$DIR"
    summary
    echo "[$(date +%T)] === $MODEL / $SWEEP / $NAME ==="
    WANDB_MODE=disabled "${CMD[@]}" > "$LOG" 2>&1
fi

rc=$?
if [ $rc -eq 0 ]; then
    echo "[$(date +%T)] $NAME finished ok -> $DIR"
    echo "  score it:  $PY $VLA_ONNX/eval/compare.py --sweep $OUT"
else
    echo "[$(date +%T)] $NAME FAILED (rc=$rc), see $LOG" >&2
    tail -20 "$LOG" >&2
fi
exit $rc
