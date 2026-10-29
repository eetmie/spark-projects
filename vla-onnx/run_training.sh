#!/usr/bin/env bash
# Fine-tune X-VLA on this machine.
#
#   ./run_training.sh --dataset masi_dual_camera --cameras cam1,cam2 --chunk 30 --execute 10 --steps 20000
#   ./run_training.sh --dataset <recording> --cameras cam1 --holdout none
#   ./run_training.sh --dataset <recording> --cameras cam1 --dry-run
#
# This used to drive SmolVLA and EVO1 as well; those playbooks are gone (git tag
# archive/lerobot-playbooks-2026-10-29 has them). What it encodes that was worth keeping:
#
#   * The training mode is explicit and goes into the run name. `full` (879 M) is the
#     default and what drives the excavator; `frozen` (311 M) scored 14-22 % worse.
#
#   * The camera set is explicit. `--cameras` builds or refreshes a dataset view (see
#     vla_common.dataset.view), which closes the trap where a grown recording left the
#     view describing the old episode count.
#
#   * The learning rate. `--optimizer.lr` is a NO-OP: TrainPipelineConfig.validate()
#     replaces the parsed optimizer with policy.get_optimizer_preset() whenever
#     use_policy_training_preset is true, which is the default. --lr maps to
#     --policy.optimizer_lr.
#
# WAITING FOR THE GPU. `--after-pid` blocks on /proc/<pid>. Do not reach for `pgrep -f`:
# it matches any process whose argv contains the pattern, including the shell that is
# tailing the run, so a queue can sit "waiting for the GPU" forever with nothing logged.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/paths.sh"

usage() {
    sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
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
  --policy.optimizer_lr X / --lr         (default 1e-4)
  -- ARGS...              everything after this goes to lerobot-train verbatim; a flag
                          this script already sets is refused, naming the owning flag

This script's own -- they compute something, so they are NOT lerobot spellings:

  --dataset NAME       a recording under $VLA_DATASETS, or a path       (required)
  --model xvla         accepted for old command lines; X-VLA is the only model
  --cameras a,b        cameras to train on; builds the dataset view (default: all)
  --train-mode M       full | frozen                                    (default: full)
  --holdout SPEC       every10@5 | "5 15 25" | none    (-> --dataset.episodes)
  --init-from PATH     start from this checkpoint instead of the base weights
  --name S             run dir name (default: <cams>-c<chunk>[x<exec>]-<mode>)
  --sweep S            sweep dir name (default: the dataset name)
  --out DIR            the SWEEP dir, xvla/outputs/<sweep>. Not lerobot's
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

MODEL=xvla DATASET="" CAMERAS="" MODE=full NAME="" SWEEP="" OUT=""
CHUNK=50 EXECUTE="" STEPS=30000 BATCH=32 LR=1e-4 SEED=1000
SAVE_FREQ=2500 WORKERS=10 LOG_FREQ=250 HOLDOUT="every10@5"
AFTER_PID="" IF_STALE=rebuild INIT_FROM=""
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
# IT, so `--batch_size=32` works here exactly as it does there. The short forms are
# aliases. Flags that compute something keep their own names on purpose: `--out` is the
# SWEEP directory while lerobot's --output_dir is the RUN directory inside it.
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
        --)              shift; PASSTHROUGH=("$@"); break ;;
        -h|--help)       usage; exit 0 ;;
        *) die "unknown flag: $1  (--help for the list)" ;;
    esac
done

[ -n "$DATASET" ] || { usage; die "--dataset is required"; }
[ "$MODEL" = xvla ] || die "--model $MODEL: only X-VLA is left here.
   SmolVLA and EVO1 were removed; git checkout archive/lerobot-playbooks-2026-10-29 has them."

VENV=$VENV_LEROBOT051
PLAYBOOK=$VLA_ONNX/xvla
PY=$VENV/bin/python
[ -x "$PY" ] || die "no python in $VENV -- run xvla/excavator/setup.sh first"

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

# --- train-mode ----------------------------------------------------------------------
# X-VLA has no action-head-only mode: freezing both encoders still trains the full policy
# transformer and the soft prompts. A mode that does not exist is an error, never a
# silent fallback -- that is how two runs both called "full" end up training differently.
case "$MODE" in
    full)   FV=false; FL=false ;;
    frozen) FV=true;  FL=true ;;
    *) die "unknown --train-mode $MODE (full | frozen)" ;;
esac
MODE_FLAGS=(--policy.freeze_vision_encoder=$FV --policy.freeze_language_encoder=$FL)

# --- which weights, and the model's own dialect --------------------------------------
# NEVER --policy.type here. draccus would build XVLAConfig from defaults with
# florence_config={}, and get_florence_config() dies "vision_config is required" before
# training starts, naming nothing that points at this line.
CKPT=${INIT_FROM:-$PLAYBOOK/models/xvla-base-excavator}
[ -f "$CKPT/model.safetensors" ] || die "no prepared X-VLA checkpoint at $CKPT
   Run xvla/excavator/fetch_checkpoint.sh, then prepare_checkpoint.py."
# STATE=MEAN_STD is not optional: xvla-base ships STATE=IDENTITY and X-VLA does not
# normalise proprio itself, so without it the joint angles go in as raw degrees.
MODEL_FLAGS=(--policy.path="$CKPT"
             --policy.action_mode=auto
             --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'
             --policy.dtype=bfloat16)

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
# scheduler_decay_steps defaults to 30000 and the scheduler only scales it DOWN to fit a
# shorter run. Past 30000 the LR sits at its floor for every extra step, so stretch the
# cosine to the run instead.
[ "$STEPS" -gt 30000 ] && CMD+=(--policy.scheduler_decay_steps="$STEPS")
[ "$SMOKE" = 1 ] && CMD+=(--save_checkpoint=false)
CMD+=("${MODEL_FLAGS[@]}" "${MODE_FLAGS[@]}")

# Anything after `--` goes to lerobot-train untouched -- the escape hatch for the long
# tail of policy knobs that will never all have their own flag here. It may NOT shadow a
# flag this script derives: a passthrough --policy.chunk_size=30 alongside --chunk 10
# would train one thing and name the run another. The check is against what was actually
# assembled, so it stays correct as flags are added.
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
    echo "  score it:  $PY $VLA_ONNX/eval/compare.py --sweep $OUT   (needs held-out episodes)"
else
    echo "[$(date +%T)] $NAME FAILED (rc=$rc), see $LOG" >&2
    tail -20 "$LOG" >&2
fi
exit $rc
