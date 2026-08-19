# xvla-spark-finetune — X-VLA-0.9B fine-tune on the DGX Spark (GB10)

The missing half of the X-VLA experiment. [`../orin-nano/xvla-runtime/`](../orin-nano/xvla-runtime/)
answered *can this model run on an 8 GB Orin* (yes — 12 split TRT engines, 390 ms/chunk,
5.71 GB peak RSS, parity cosine 1.000000) and ended on the sentence that defines this
project:

> `xvla-base` cannot drive the excavator as-is (it is a 20-dim `ee6d` arm policy), so a
> deployable X-VLA needs a Spark fine-tune first, exactly as SmolVLA did.

This is that fine-tune. The goal is not just a working X-VLA — it is a **fair number
against SmolVLA on the same excavator data**, so "2x the parameters and 2.6x the memory"
can be weighed against whatever accuracy it actually buys.

Kept separate from `../smolvla-spark-finetune/` deliberately: that side is still moving,
and neither experiment should be able to break the other.

## What makes the comparison fair

| shared | why it matters |
|---|---|
| the same venv (`../smolvla-spark-finetune/.venv`) | lerobot 0.5.1, torch 2.12.0+cu130 for both — no version variable smuggled into the result |
| the same dataset `masi_digging` | 82 eps, 41765 frames @30 fps, task "move the sand to the container" |
| the same cam1-only view | `datasets/masi_digging_ir`, metadata-only, **symlinked** frames — byte-identical images |
| the same held-out episodes | `5 15 25 35 45 55 65 75` |
| the same chunk length (50) | the xvla-base checkpoint ships 30 = 1.0 s, less than the 1.5 s scoring horizon |
| the same frozen slice | VLM encoders frozen, action head/policy transformer trains |
| the same metric | `eval_compare.py` disp_err, integrated command error over a fixed wall-clock horizon |

The eval harness lives on the SmolVLA side but is **architecture-agnostic**: it loads
whatever `policy.type` a checkpoint records via `get_policy_class`, and feeds each model
only the cameras its own config lists. Runs in different sweep dirs are joined with
`--extra-runs`:

```bash
cd ../smolvla-spark-finetune/excavator
../.venv/bin/python eval_compare.py --preset digging --horizons 1.5 \
    --extra-runs xvla_ir=../../xvla-spark-finetune/outputs/digging/ir \
                 xvla_both=../../xvla-spark-finetune/outputs/digging/both
```

That prints SmolVLA and X-VLA rows in one table with a `policy` column.

## Run

```bash
bash excavator/fetch_checkpoint.sh                  # once — 3.52 GB into models/xvla-base/
python excavator/prepare_checkpoint.py              # once — derive xvla-base-excavator/
bash excavator/run_digging.sh ir          # IR only — the comparison at half the cost
bash excavator/run_digging.sh             # ir then both
```

The checkpoint is fetched with curl, not `huggingface_hub`: `snapshot_download` stalled
twice partway through the safetensors here — process alive, file not growing, no exception,
so its own retry never fired. `curl --speed-limit/--speed-time` turns a stall into an error
that `--retry` can act on, and `-C -` resumes. ~9 MB/s vs ~4 MB/s before it hung.

**Measure throughput before committing to a step budget.** X-VLA is 879.7 M params against
SmolVLA's 450 M, and its hot loop has no KV cache. On GB10 the SmolVLA runs measured
0.507 s/step cold and **0.630 s/step heat-soaked** (batch 32, chunk 50, frozen vision) —
the GPU settles around 79 °C and 2457/3003 MHz under sustained load, so always size an
overnight run from the heat-soaked figure, not a cold probe. Do the same for X-VLA:

```bash
STEPS=250 SAVE_FREQ=100000 bash excavator/run_digging.sh ir
grep -oE "updt_s:[0-9.]+" outputs/digging/logs/ir.log | tail
```

## The stock checkpoint cannot train on this dataset unmodified

`lerobot/xvla-base`'s config.json declares the input contract of the robots it was
pretrained on — `observation.images.image / image2 / image3` and `observation.state (8,)`.
`make_policy` only fills `input_features` from the dataset **when the config leaves it
empty**, so with `--policy.path` that stock contract survives and training dies with:

```
ValueError: Feature mismatch between dataset/environment and policy config.
- Missing features: ['observation.images.image', 'observation.images.image2', 'observation.images.image3']
- Extra features: ['observation.images.cam1']
```

The error suggests `--rename_map`, and that does make it start — but it is the **wrong
fix**: `make_policy` skips `validate_visual_features_consistency` entirely when a rename_map
is present, and the normalizer is then built with `features=policy.config.input_features`
(state shape `(8,)`) against `stats=dataset.meta.stats` (state shape `(3,)`). Mismatched
shapes, no error. `--policy.input_features={}` does not work either — draccus *merges* the
CLI dict into the one from config.json rather than replacing it.

`excavator/prepare_checkpoint.py` does the honest thing: derive a checkpoint dir with
`input_features` emptied (weights symlinked, not copied), so the dataset defines the contract.
Nothing about the weights depends on the emptied keys — `dim_proprio = max_state_dim` (20)
and `_prepare_state` pads our 3-dim state to it, so the declared `(8,)` only ever fed the
normalizer. `num_image_views` stays 3 and `_prepare_images` zero-pads our 1-2 real views to
3 with `mask=False`; `forward_vlm` encodes valid views only, so the padding is free — the
same shape as the Orin runtime's `--valid-views 1`.

## Settings that are not X-VLA's defaults

All four are argued in the header of `excavator/run_digging.sh`. Short version:

- **`action_mode=auto`** — the default `ee6d` is a 20-dim arm space the excavator does not
  have. `auto` detects real_dim=4 from the dataset, pads to 20 so the pretrained head
  loads, and computes loss on the real 4 only.
- **`STATE=MEAN_STD`** — read the *checkpoint*, not the class defaults: XVLAConfig's
  dataclass says IDENTITY for all three, but `lerobot/xvla-base` actually ships
  `{STATE: IDENTITY, ACTION: MEAN_STD, VISUAL: IDENTITY}`. Only STATE needs changing —
  X-VLA does not normalize proprio internally and our state is joint angles in degrees
  (−114…134). **ACTION stays MEAN_STD**: that is what the pretrained head was trained with
  *and* what SmolVLA uses, so after the override both architectures normalize identically.
- **`chunk_size=50`** — the base checkpoint ships 30 (which is where the Orin's "30 actions
  per chunk" comes from); 50 matches SmolVLA and clears the 1.5 s scoring horizon.
- **VLM encoders frozen** — matches the config's own documented intent (its literal
  defaults disagree with its docstring) and the SmolVLA recipe.

`domain_id` stays 0, matching `../orin-nano/xvla-runtime`'s `--domain-id 0`.

## Deployment note

`num_image_views` auto-resolves to the camera count and only sizes a runtime buffer — no
parameter shapes depend on it, so 1- and 2-camera fine-tunes both load the 3-view base
checkpoint cleanly. But two deployment gaps are open:

1. **A 2-camera model has no deploy path on either architecture.** SmolVLA's exporter and
   Orin runtime are single-camera end to end (see the SmolVLA project notes); X-VLA's
   split export was made with `--valid-views 1`.
2. **The Orin latency numbers were measured at 30 actions/chunk.** Training at 50 changes
   the denoise sequence length, and X-VLA cannot KV-cache across denoising steps, so
   re-measure before quoting a replan rate.

## Status

- 2026-08-19 — pipeline written and config-validated end to end against `masi_digging`:
  `action_mode=auto` resolves real_dim 4 → model dim 20, both cameras detected, state 3-dim,
  `normalization_mapping` override parses, `XVLAPolicy` loads through `get_policy_class`.
  Config load verified through the real `--policy.path` code path: action_mode ee6d→auto,
  chunk 30→50, freeze (False,False)→(True,True), Florence2Config builds.
  `lerobot/xvla-base` (3.52 GB) fetched to `models/xvla-base/`.
  Smoke-tested end to end (2 steps, batch 2): weights load, **879.48 M total / 310.98 M
  learnable** (frozen encoders confirmed), 74-episode split, checkpoint written. The saved
  train_config records action_mode=auto, chunk 50/50, STATE+ACTION MEAN_STD, VISUAL IDENTITY,
  input_features `[observation.state, observation.images.cam1]`, action shape `[4]`.
  **Not yet trained for real** — the GPU was busy with the SmolVLA digging sweep. A 250-step
  throughput probe is armed to run automatically when that queue finishes
  (`excavator/probe_after_smolvla.sh`, logs to `outputs/probe_after.log`).
