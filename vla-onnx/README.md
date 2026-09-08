# vla-onnx

One pipeline, three models. Fine-tune a LeRobot VLA on the DGX Spark, cut it into split
ONNX graphs that a Jetson Orin Nano can actually build TensorRT engines for, verify the
export against PyTorch, and ship a bundle.

```
checkpoint ──► fine-tune ──► split export ──► mixed FP16 ──► manifest ──► parity ──► bundle
               (Spark)       (Spark)          (Spark)        (Spark)      (Spark)     └─► ~/bundles
```

Running the bundle is somebody else's job: base-model fit and benchmarks live in
[jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla), the robot lives in
`kaivuriprokkis`. This side stops at a parity-checked bundle.

| playbook | model | state |
|---|---|---|
| [`smolvla/`](smolvla/) | SmolVLA 450 M | fine-tuned, deployed |
| [`xvla/`](xvla/) | X-VLA 0.9 B | fine-tuned, deployed |
| [`evo1/`](evo1/) | EVO1 775 M | export + Orin bootstrap proven; **no trained checkpoint yet** |

## Running a fine-tune

One entry point for all three models. Everything you vary is a flag; nothing is an
environment variable and nothing is a bare word matched inside a case statement.

```bash
./run_training.sh --model smolvla --dataset masi_digging_dry_2 --cameras cam1 \
                  --train-mode expert --chunk 10 --steps 30000

./run_training.sh --model xvla --dataset masi_digging_dry_2 --cameras cam1 \
                  --chunk 30 --execute 10 --train-mode full --steps 20000

./run_training.sh --model smolvla --dataset <recording> --cameras cam1 --dry-run
```

`--dry-run` prints the resolved `lerobot-train` command and exits; `--smoke` runs two
steps into a disposable directory; `--help` lists every flag. `--after-pid N` waits for a
running job to exit before starting (never `pgrep -f`, which matches the shell that is
tailing the log).

**`--train-mode` is explicit because it used to be invisible.** Every SmolVLA excavator
run trained 99,880,992 of 450,046,176 parameters — the checkpoint's config defaults — and
no script said so. The mode now goes into the run directory name, so an artefact says what
it is rather than what someone happened to type.

| mode | smolvla | xvla | evo1 |
|---|---|---|---|
| `expert` | vision + VLM frozen, action expert only | *refused* — X-VLA has no such mode | `training_stage=stage1` |
| `frozen` | = `expert` | both encoders frozen (311 M) | = `stage1` |
| `full` | nothing frozen (~450 M) | nothing frozen (879 M) | `training_stage=stage2` |
| `lora` | `--peft.method_type=LORA` | *refused* | *refused* |

A mode a model does not have is an error naming the modes it does have, never a silent
fallback. `--model evo1` currently refuses outright: it has no trained excavator path, and
the message says which three pieces are missing.

**Where the weights start.** `--init-from <checkpoint>` replaces the model's base weights
(SmolVLA's `lerobot/smolvla_base`, X-VLA's prepared `models/xvla-base-excavator`) — that is
how you continue from a fine-tune rather than from the base. `--no-vlm-weights` is the
SmolVLA-only case of training the action expert against a randomly initialised VLM.

**The escape hatch.** Everything after `--` is appended to `lerobot-train` verbatim:

```bash
./run_training.sh --model smolvla --dataset <rec> --cameras cam1 -- --policy.num_steps=5
```

There is a long tail of policy knobs that will never all deserve their own flag, and this
is where they go. One rule: a passthrough flag may **not** shadow one the script derives.
`-- --policy.chunk_size=30` alongside `--chunk 10` is refused, because it would train one
thing and name the run directory another — the invisibility this script exists to remove.
The check is against the command actually assembled, so it stays correct as flags are added.

## Scoring a run

```bash
smolvla/excavator/eval_compare.py --sweep smolvla/outputs/<sweep> --horizons 0.17 0.3
smolvla/excavator/eval_compare.py --ckpt <checkpoint> --only-task "move rock to container"
smolvla/excavator/eval_curve.py   --sweep smolvla/outputs/<sweep> --horizon 0.17
```

The source recording, the held-out episodes and the per-episode instructions are read from
the checkpoint's own `train_config.json` — held-out is the complement of
`dataset.episodes`, so it cannot disagree with what the run actually trained on. There is
no preset table any more; see [`smolvla/notes/retired-presets.md`](smolvla/notes/retired-presets.md)
for the one it replaced and what each field of it cost.

## What is shared, and what is deliberately not

[`common/`](common/) (`vla_common`) holds what all three do *identically*: hash a bundle,
validate a traced graph, record provenance, convert weights to mixed FP16, read a
safetensors header, reshape a LeRobot dataset.

What is **not** shared is how a model gets cut into graphs. `_build_wrappers()` is ~200
lines in each playbook and about 5% similar between them, because the cut follows the
architecture — SmolVLA's VLM + expert prefill/decode, X-VLA's DaViT + BART + denoiser,
EVO1's InternVL. That is the actual content of an exporter, and merging it would mean
one file with three unrelated halves. The pipeline is shared; the models are not.

The FP16 recipe in `vla_common/fp16_weights.py` is a hard-won constant, not a default:
a blanket cast overflowed SmolVLA's vision tower here (cosine 0.805), so
`LayerNormalization` and `Softmax` stay FP32.

## Views

All three exporters take `--views N` — how many camera views the bundle is sized for.
Nothing in the pipeline assumes a camera count, a camera name, or infrared: camera keys
are discovered from the checkpoint's `observation.images.*`, and `vla_common` contains no
camera assumptions at all. Camera names appear only where you type them, on
`run_training.sh --cameras`.

`--views` is baked into the graphs, not read at runtime, so **a bundle exported at 1 view
cannot serve a 2-view runtime.** For SmolVLA it sets a static prefix length (1 view = 113
tokens, 2 = 177); for X-VLA the static batch of the vision engine; for EVO1 the vision
graph's static view count. The old per-model spellings — `--cam-slots`, `--valid-views`,
`--max-views` — still work as aliases.

Training on a camera subset needs a *view* of the recording, because `lerobot-train` has
no camera flag: every `observation.images.*` in `meta/info.json` becomes a policy input.
`run_training.sh --cameras` builds and freshens that view for you; the pieces are:

```bash
python -m vla_common.dataset.view --src <recording> --cameras cam1   # resolve, build if needed
python -m vla_common.dataset.camera_variant --src <rec> --dst <view> --keep cam1   # the builder
```

Views live in `datasets/` (shared, not owned by one playbook) and are named from what they
are — `masi_digging_dry_2__cam1`. A view symlinks `data/` and the kept videos but **copies**
`meta/`, so a recording that grows leaves the view describing the old episode count while
new frames appear through the symlink. `view.py` stamps each view with the source
fingerprint and compares: a grown source is rebuilt automatically, a *shrunk* one is
refused, because dropping episodes renumbers the survivors and makes every split derived
from an older checkpoint a lie.

## Two environments, and why they cannot be one

| env | lerobot | torch | used by |
|---|---|---|---|
| `.venv-lerobot051` | 0.5.1 | 2.12.0+cu130 | smolvla fine-tune, **xvla fine-tune** |
| `.venv-lerobot061` | 0.6.1 | 2.11.0+cu130 | evo1 everything, **xvla export** |

lerobot 0.6.1 requires `torch>=2.7,<2.12`; the 0.5.1 stack pins 2.12.0. They are mutually
exclusive, so this is two venvs on purpose and not a cleanup that was missed.

SmolVLA and X-VLA fine-tune in the *same* venv deliberately — an identical stack is what
makes "X-VLA is ~20% behind SmolVLA on this data" a statement about the models rather
than about two dependency trees.

`vla_common` is installed into both (`pip install -e common`), which is why it carries no
torch or lerobot pin and imports nothing version-sensitive at module scope.

## Paths

Nothing hardcodes a home directory. Shell scripts source [`paths.sh`](paths.sh); Python
imports `vla_common.paths`. Both resolve from their own location and honour overrides:

| variable | default | what it is |
|---|---|---|
| `VLA_ONNX` | this directory | pipeline root |
| `VLA_DATASETS` | `~/Desktop` | recorded excavator datasets — **source data, the only copy** |
| `VLA_BUNDLES` | `~/bundles` | finished bundles, staged for `ship_bundle.sh` |
| `VENV_LEROBOT051` / `VENV_LEROBOT061` | as above | the two environments |

This matters because the tree has already been renamed twice, and each time a crop of
absolute paths went stale silently — a script would still run, and write where nobody
was reading.

## Not in git

`models/`, `outputs/`, `exports/`, `datasets/`, `.venv-*`. Checkpoints and exports are
regenerable from a recipe plus a checkpoint; the recorded datasets are not regenerable at
all and live outside the repo under `VLA_DATASETS`.
