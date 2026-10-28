# Training for real-time chunking: pi0.5 (OpenPI) and GR00T N1.7 (Isaac-GR00T)

Training side of the board's RTC work: kaivuriprokkis `feat/flow-rtc-scaffold` (19239b0,
`notes/FLOW_MATCHING_RTC.md`), `feat/groot-inference-scaffold` (1889ce8) and
`feat/pi05-rtc-plan` (cecd713, `RTC_PLAN.md` Mode C / P5). Both models train with each
project's native trainer: OpenPI JAX for pi0.5 and NVIDIA's `launch_finetune.py` for GR00T.
Neither is forked. The excavator configs and patches are applied on top at run time.

| file | what |
|---|---|
| `../pi05/excavator/rtc_model.py` | `Pi0RTC` + `Pi0RTCConfig`: per-row time, clean prefix, postfix loss, prefix sampler |
| `../pi05/excavator/config.py` | `pi05_excavator_rtc` next to `pi05_excavator` (same data, optimizer, weights) |
| `../groot/excavator/rtc_patch.py` | the same for GR00T N1.7, source-hash pinned to 23ace64 |
| `../groot/excavator/modality.py` | excavator `NEW_EMBODIMENT` modality config |
| `../groot/excavator/prepare_dataset.py` | GR00T view of a v2.1 dataset (symlinks + `modality.json`) |
| `../groot/excavator/finetune.py` | `launch_finetune.py` with the excavator defaults and `--rtc-max-delay` |
| `../common/vla_common/dataset/from_native.py` | raw native sessions -> simplified LeRobot v2.1 for both trainers |
| `../common/vla_common/dataset/to_v3.py` | v2.1 -> LeRobot v3.0 copy (`from_native --v3`), for LeRobot policies and `lerobot-dataset-viz` |
| `../common/vla_common/dataset/rerun_native.py` | raw native takes in rerun: cameras, every status field, timing, FK arm |

## Data

The board records `kaivuri-native-episode-v1` sessions (Y8 frames + `samples.jsonl`).
`from_native` writes the training set directly: cam1 + cam2, state `[lift, tilt, scoop]` deg
(slew from the cameras only), the 4 applied valve commands, the session task, and per-frame
input ages (`clock.*age_s`, not policy inputs). It runs the board exporter's take checks.

```bash
# on the Jetson (the board's link is a slow hotspot; move only the compressed set)
~/kaivuriprokkis/.venv-lerobot061/bin/python from_native.py \
  ~/data_collection/native_sessions/digging_native_640 --dst ~/spark-tools/digging_native_640_simple --jobs 4
rsync -a joel@pikkumasi2tb:spark-tools/digging_native_640_simple/ vla-onnx/pi05/datasets/digging_native_640_simple/
```

- Video is H.264 CRF 18 (yuv420p, GOP 2, as LeRobot records). The robot feeds raw frames,
  as the Python stack did with CRF 30 data, so inference needs no converter. Error vs raw:
  CRF 18 mean 2.0 grey levels (p99 6), 6% of Y8; CRF 30 mean 3.2 (p99 13), 1.6%.
  Encoding the live frames on the Orin to match would cost 6-10 ms per camera (no NVENC).
  `--lossless` is bit-exact (libx264rgb) at ~125% of Y8 and decodes ~5x slower in GR00T.
- Input ages on digging_native_640 (p50 / p99 ms, relative to the recorder's pairing):
  cam1 5.7 / 28.7, cam2 15.7 / 26.2, joint angles 1.35 / 6.4, applied action 5.3 / 5.5.
  The cameras are not synchronized; each is the latest frame. Inference should pair the
  latest frames with a fresh state the same way.
- `digging_native_640_simple`: 100 takes (50 rocks, 50 sand), 74,610 frames at 30 FPS,
  2.6 GB (raw 44 GB). The raw session is archived under `~/Desktop/kaivurin_datasetit/`.

## Training-time RTC

Black, Ren, Equi, Levine, *Training-Time Action Conditioning for Efficient Real-Time Chunking*,
[arXiv 2512.05964](https://arxiv.org/abs/2512.05964), Algorithm 1:

- Per training sample, draw a delay `d`. Rows `< d` are the clean ground-truth actions, and
  each of them gets the "clean" flow time. Rows `>= d` are noised as usual. The loss covers
  rows `>= d` only.
- At inference, write the committed rows of the previous chunk into rows `< d` before every
  denoising step, with clean time. There is no VJP and no extra forward pass. The sampler
  interface matches inference-time RTC.
- For adaLN-style time conditioning, scale/shift/gate vary per token. This adds no parameters,
  so base checkpoints load unchanged.
- Paper settings: real robot `d ~ U[0, 10)` at 50 Hz, 5 denoising steps, π0.6 fine-tuned 8k
  steps from a trained model; simulation weights `d` exponentially. It beats inference-time
  RTC at delays ≥ 2 and is marginally worse at 0–1 (less supervision on the first rows).

PI ran this on π0.6, which is closed. Public OpenPI has no RTC of either kind; its last commit is
215abfb (2026-08-24), and we pin c23745b. Isaac-GR00T has NVIDIA's inference-only "frozen rows +
velocity-strength ramp" at both the export pin 23ace64 and main 944fe70; neither has a training
side. Main adds training fixes (resume, percentile stats, a Beta-sampler dtype fix). `dit.py` and
`embodiment_conditioned_mlp.py` are byte-identical between 23ace64 and main. In `gr00t_n1d7.py`
only the Beta construction differs, so the patch's hashed functions are the same on both.

| | pi0.5 (OpenPI JAX) | GR00T N1.7 |
|---|---|---|
| flow time | noise 1 → actions 0, clean = **0** | noise 0 → actions 1, clean = **1** (bucket 1000; training never draws it) |
| where time enters | AdaRMS in each action-expert layer + final norm (`gemma.RMSNorm`, `[B, emb]`) | action encoder sinusoid; AdaLayerNorm in every DiT block + DiT output modulation (`[B, D]`) |
| per-row change | `RowRMSNorm` accepts `[B, H, emb]`; `embed_suffix` takes `[B, H]` time | encoder takes `[B, H]`; DiT takes per-token time `[B, 1+H]` (state token keeps the sample's t) |
| loss mask | returned `[B, H]` scaled so `train.py`'s mean is the mean over kept rows | multiplied into GR00T's padding `action_mask` |
| default `d_max` | 20 rows: obs→plan p99 479 ms at 30 FPS ≈ 15 rows, plus margin | 12 rows: board N1.7 p99 ≈ 284 ms (3-view base) + observation age ≈ 10 rows |
| inference export | exporter `--row-time` templates + `conditioning_clean_fp16.npy` exist; runtime P4 not built | **not exported**: needs per-token time in the action-encoder and DiT engines |

In both models a `[B]` time takes the stock code path, so plain configs, plain checkpoints and the
existing exports are unaffected. `d` is drawn from `{0..d_max}` inclusive, uniform by default.
Choose `d_max` from the runtime's largest committed prefix: the board's `flow_rtc` commit tick,
`ceil((latency + observation age) × fps)`. It is tied to the dataset FPS. Re-check it when FPS,
latency or pacing change.

## pi0.5

```bash
cd vla-onnx/pi05
# dataset: $HF_LEROBOT_HOME/$KAIVURI_DATASET (pi05/datasets/digging_native_640_simple)
third_party/openpi/.venv/bin/python excavator/run.py norm-stats --config-name pi05_excavator_rtc
third_party/openpi/.venv/bin/python excavator/run.py train pi05_excavator_rtc --exp-name NAME
#   from a plain excavator checkpoint: --weight-loader.params-path checkpoints/pi05_excavator/EXP/STEP/params
JAX_PLATFORMS=cpu third_party/openpi/.venv/bin/python -m pytest excavator/tests -q   # 8 tests, ~50 s
```

`Pi0RTCConfig.create()` swaps `gemma.RMSNorm` for `RowRMSNorm` (same parameters; a 2-D cond is
bit-exact with stock). `Pi0RTC.sample_actions(..., action_prefix=[B,H,32], delay=[B])` is the
reference sampler. With `delay=None` it is OpenPI's. The exporter's `rtc.py` "clean" mode is
the board-side oracle of the same loop.

## GR00T N1.7

```bash
cd vla-onnx/groot/excavator
PY=../.venv-groot17/bin/python          # setup17.sh's export venv runs the trainer (SDPA, no flash-attn)
$PY prepare_dataset.py ../../pi05/datasets/digging_native_640_simple ../datasets/digging_native_640_gr00t
$PY finetune.py --rtc-max-delay 12 -- \
  --base-model-path nvidia/GR00T-N1.7-3B --dataset-path ../datasets/digging_native_640_gr00t \
  --output-dir runs/NAME --num-gpus 1 --global-batch-size 32 --max-steps 20000 --save-steps 2000
PYTHONPATH=<pytest> $PY -m pytest tests -q                                            # 15 tests, ~5 s
```

- Do **not** run NVIDIA's `scripts/deployment/spark/install_deps.sh` on the Spark host: it uses
  `sudo apt-get`. `.venv-groot17` already runs the trainer; their Spark Dockerfile is the
  fallback.
- The stock backbone downloads Cosmos-Reason2-2B's 2B weights only to overwrite them.
  `finetune.py` builds the VLM from its config instead (as `groot17_split.load_policy` does).
  It also resolves both gated repos to their cached HF snapshots, so no login is needed. The
  Spark's stored HF token had expired on 2026-10-26.
- GR00T writes `meta/stats.json` into the dataset when it lacks q01/q99. LeRobot's stats lack
  them, so it would overwrite the OpenPI dataset's stats. `prepare_dataset.py` gives GR00T its
  own `meta/` and symlinks the data and videos.
- Valve actions are `ABSOLUTE` / `NON_EEF`. The launcher forces `use_relative_action=True`, but
  that only affects `RELATIVE` keys.
- `state_dropout_prob`: the launcher default is 0.2 (the model config says 0.8). The excavator
  models lean on joint angles; 0.2 is a starting point to A/B, not a decision.
- Training-time RTC needs `--tune-diffusion-model` (default on). The backbone stays frozen
  (`tune_llm`/`tune_visual` off) as in the board plan.
- `rtc_training.json` in the output dir marks an RTC checkpoint. NVIDIA's own inference RTC
  still works on any checkpoint (`action_input["action"]` + `rtc_*` options).

## Verified (2026-10-26, Spark)

- pi0.5 CPU tests (dummy Gemma, randomized weights):
  - `RowRMSNorm` with a 2-D cond is bit-exact with stock;
  - parameters are identical to `Pi0Config`;
  - the `delay=0` per-row sampler matches stock within 2e-5;
  - prefix rows come back exactly, and free rows depend on the prefix;
  - the loss is zero on rows `< d`, and its mean is over the kept rows.
- pi0.5 full model on the GB10: `pi05_excavator_rtc` from pi05_base, 5 steps at batch 4, loss
  0.018–0.041, checkpoint saved. Sampling that checkpoint (bf16 params, fake observation):
  - with `delay=15` the prefix rows come back exactly;
  - the per-row path with `delay=0` is within 0.005 of stock (on a ±0.81 range; bf16
    per-row vs broadcast matmul);
  - JAX sampling takes 303 ms vs 281 ms stock.
- GR00T CPU tests (tiny action head, both DiT variants):
  - the source hashes match;
  - a different revision is refused;
  - the patched stock loss is within 1e-6 and the stock sampler is bit-exact;
  - the RTC mask zeroes exactly rows `< d`, and the loss is differentiable;
  - eval mode has no prefix;
  - prefix rows come back exactly, and free rows depend on the prefix;
  - per-sample delays work.

- GR00T N1.7 full model on the GB10: `finetune.py --rtc-max-delay 12`, 5 steps at batch 4, on the
  smoke set's GR00T view. Results:
  - 75 s including data warm-up; train loss 1.18;
  - the checkpoint and `rtc_training.json` were saved;
  - the GR00T stats went into the view's own `meta/`, and the OpenPI dataset is untouched;
  - all 493 backbone tensors are byte-equal to the base model, so the config-built VLM got
    every weight; only the action head changed;
  - `lm_head` is not saved, which is harmless because the LLM is cut at layer 12.

## Verified (2026-10-27, digging_native_640)

- `from_native` on the Jetson: 100 takes pass the checks, 2.58 GB, video mean |error| 1.98
  (worst take 2.09), ~35 min with `--jobs 4`. The Spark copy is checksum-identical.
- OpenPI's pinned LeRobot loads it (two 3x480x640 views, state 3, 50x4 action chunk, task).
- GR00T N1.7 with `--rtc-max-delay 12` on the full view: 2.42 s/step at batch 32 on the GB10
  (20k steps ~13.5 h), loss 1.25 -> 1.21 over 20 steps. Stats went into the view's `meta/`.
- pi0.5 `pi05_excavator_rtc` on the full set from pi05_base: 16 s/step at batch 32, GPU-bound
  (96%; 16 data workers change nothing), same as plain pi0.5 (16.2 s/step), so RTC is free.
  Loss 0.03-0.04 by step 5. 30k steps would take ~5.5 days; 10k ~45 h.
- GR00T `--dataloader-num-workers 8` costs ~9-10 GB RAM per worker (71 GB) and is slower
  (2.87 s/step) than the default 2 (2.4-2.6 s/step); keep 2.
- Initial runs (2026-10-27, `groot/runs/queue_digging640.sh`, detached, one GPU job at a time
  under memguard): GR00T RTC `d_max` 12, 20k steps -> `groot/runs/digging640_rtc12`; then pi0.5
  RTC `d_max` 20, 10k steps (decay 10k) -> `pi05/checkpoints/pi05_excavator_rtc/digging640_rtc20`.
  An RTC checkpoint also runs on the existing scalar-time export at delay 0.
- 5 CPU tests for `from_native` (`vla-onnx/common/tests`, synthetic two-run session).
- No validation split: both initial runs train on all 100 takes. Hold out whole takes (5 rock
  + 5 sand) when checkpoints need an offline comparison.

## What trains, and the memory budget (one Spark, 121 GB unified)

| | trained | frozen | GPU / total RAM |
|---|---|---|---|
| GR00T N1.7 (NVIDIA default) | projectors + DiT, 1.62 B (51.5 %) | vision, LLM | 38 GB / ~74 GB with 2 workers |
| pi0.5 (OpenPI default) | everything | nothing | 72 GB with EMA (59 GB without) / ~81 GB |

- **Full GR00T fine-tune** (`--tune-visual --tune-llm`): estimated, not run. +1.5 B params x 12 B
  (grad + AdamW) ~ +18 GB, plus 10-20 GB activations: ~65-75 GB GPU, ~100-110 GB total, against
  memguard's 8 GB floor. If tight: batch 16 x grad-accum 2, 1 worker, gradient checkpointing.
  Expect ~3-4 s/step. X-VLA lesson: head-only drove badly on the machine and full fine-tune won,
  so try this if the head-only GR00T drives poorly.
- **EMA** (OpenPI default, decay 0.99) keeps a smoothed fp32 copy of the weights, and that copy is
  what gets saved and deployed: usually a little better and smoother. **On by default since
  2026-10-28** (`--ema-decay None` turns it off). Measured at batch 32, `pi05_excavator_rtc`:

  | allocator, cap | EMA | device | free while training / lowest in a save | s/step |
  |---|---|---|---|---|
  | BFC 85 % | 0.99 | 106 GB | 8 GB: memguard kill at step 0 | - |
  | BFC 72 % | 0.99 | 90 GB (whole cap) | 20 / 4 GB: kill in the save | 18.4 |
  | BFC 65 % | 0.99 | - | OOM: 21.9 GB step scratch does not fit | - |
  | **cuda_async 72 %** | **0.99** | **72 GB** | **40 / 14 GB, saves ok** | **17.5** |
  | cuda_async 72 % | off | 60 GB | - / 28 GB | 17.5 |

  BFC keeps every freed block and needs the step's scratch contiguous; `spark_env.py` now selects
  `cuda_async`. EMA costs no step time (same step-20 loss with and without). Saves block ~1 s and
  finish in 10-20 s on the internal NVMe. The step-20 checkpoint is finite and moved 2.4e-7/element
  from pi05_base (live weights would be ~1.2e-5, EMA ~8e-7), so the saved params are the EMA copy.
  Full resumable saves (`OPENPI_SAVE_OPT_STATE=1`, ~53 GB host copy) are still too big.
- **Two Sparks** (ConnectX-7 link) give room for both, but only with sharding; plain DDP keeps a
  full copy per box. GR00T already trains through DeepSpeed (ZeRO-2 default, ZeRO-3 available);
  multi-node is `torchrun --nnodes 2`, untested. OpenPI has `fsdp_devices` and a
  `process_count`-aware loader, but nothing calls `jax.distributed.initialize`, so multi-host
  JAX needs a small patch, untested.

## GR00T digging640_rtc12 on the board (2026-10-28)

The single-frame checkpoint (20k steps) runs on pikkumasi2tb at delay 0 on the existing scalar-time
export, from live cameras and IMUs to drive plans. Valves and pump were off at the mains.

```bash
cd vla-onnx/groot/excavator; PY=~/spark-projects/vla-onnx/groot/.venv-groot17/bin/python
$PY open_loop.py --checkpoint ../runs/digging640_rtc12 --dataset ../datasets/digging_native_640_gr00t \
  --episodes 0 25 49 50 75 99 --out OUT                       # chunks vs recorded valves
$PY physical_fixture.py --checkpoint ../runs/digging640_rtc12 --dataset ../datasets/digging_native_640_gr00t \
  --samples 50:0 50:240 51:600 75:120 99:300 60:450 --task "scoop the sand and dump it into the container" \
  --dtype float32 --out fixture_physical_sand.npz              # and --dtype bfloat16 for the envelope
```

- **Export.** jetson-orin-nano-vla 6e2d857 `export/groot` (its K/V-gather + AdaLN-modulation
  layout, newer than this repo's copy) with `reference17.py`'s `processor/` lookup, run in
  `.venv-groot17`: `reference17.py --checkpoint runs/digging640_rtc12 --embodiment new_embodiment
  --task ...`, `export_split_onnx17.py`, `vla_common.half_boundary` as `export.sh` does. 5.3 GB,
  30 graphs, 2 views x 1 frame, prompt 147 tokens. The baked prompt is one task (one bundle per task).
  The frozen vision/LLM graphs differ from the base bundle only slightly: rsync against a hardlinked
  copy of the base bundle sent 2.6 of 5.7 GB.
- **Open loop** (training takes, so not generalization): mean |chunk - recorded| 0.04-0.10 per
  valve vs 0.16-0.47 for zero/mean chunks. The error falls from 0.10 at row 0 to 0.063 at row 12,
  then flattens: rows `< d` carry no loss, so row `i` learns only when `d <= i` (`d_max` 12). At
  delay 0 the first ~12 rows are the weakest; the board starts playing at row 13-23 anyway.
- **Normalization.** GR00T min-max normalizes to q01/q99 and clips. For valves this caps the
  model's output at q99: lift up +0.547, slew +0.790/-0.894 (1% of frames clipped each). The X-VLA
  runtime uses LeRobot MEAN_STD with no clip. Consider `use_percentiles` off for valve actions.
- **bf16.** `Gr00tPolicy` casts the model and inputs to bfloat16. Its chunks differ from FP32 by
  up to 0.089 valve units. `physical_fixture.py --dtype float32` is the exact reference.
- **Board parity** (`kaivuriprokkis` `groot_live.py parity`, TRT 10.16 host chain): raw IR frames +
  degrees -> valves vs FP32 Gr00tPolicy on 6 real frames, cosine 0.99954-1.0, mean |d| 0.0002-0.0097.
  Each sample is closer to FP32 than bf16 Gr00tPolicy is (the gate). The bundle fixture passes
  (chunk 0.09% of range, preprocessing exact). The FP32 split fed the board's inputs differs
  from FP32 Gr00tPolicy by 0.0002-0.002 mean, so the prompt, image transform and boundary match.
- **Live** (`groot_live.py run --start-services --live-pwm --enable-slew --drive`, 120 s): 388
  chunks back to back, inference p50 302 ms, observation -> plan p50 333 / p99 344 ms. Plans:
  t0 = observation, renewed every 100 ms. The controller applied the interpolated planned row
  within 0.0006 mean / 0.014 max (matched by `applied_command_sequence`). No faults.
- The engines are native TensorRT on the host chain. The AOT kernels and the CUDA-graph device
  chain assume the base 3-view, 2-frame profile.

### Runtime prompts (2026-10-28)

```bash
git -C ~/jetson-orin-nano-vla archive 6e2d857 export | tar -x -C groot/third_party/jetson-orin-nano-vla-6e2d857
cd groot/excavator && $PY export_bundle.py --checkpoint ../runs/digging640_rtc12 \
  --dataset ../datasets/digging_native_640_gr00t --out BUNDLE        # ~15 min, mostly fixtures
```

- The processor's prompt is a fixed 135-token prefix (chat start + both image slots), then
  `tokenize(formalize(text))`, then `<|im_end|>\n`. That reproduces its ids exactly (12 strings,
  including unicode, empty and 81-token ones; `tokenizers` alone matches `transformers`).
- The baked MRoPE table is bit-identical for every text length, because the images precede the
  text and the pads continue its positions. So the LLM graphs are prompt-free. Only the gathered
  K/V rows baked a prompt; `--full-kv` makes the masks runtime inputs.
- `--seq-len 256` (2 images need 128 slots, not the base 448) gives 119 text tokens and
  248 ms on the board, vs 297 ms for the baked 448 bundle (`--full-kv` at 448: 320 ms).
- bundle.json `prompt`: prefix/suffix ids, `tokenizer.json`, capacity, the dataset's tasks, and
  per-prompt FP32 + bf16 physical fixtures (each trained task, plus "dig a hole"). bundle.json
  `kaivuri`: fps, names and the normalization contract, so the board needs no `--contract`.
- Board (`groot_live.py parity --bundle B`): every prompt, tokenized on the board, passes vs
  FP32 Gr00tPolicy inside the bf16 envelope. Live, D-pad right/left cycles the list and a typed
  line is appended to it. A switch drops the in-flight chunk; the controller holds until a chunk
  observed after the switch.
- A re-export of the same checkpoint rsyncs in ~13 s against a hardlinked copy of the previous
  bundle (same weights, small graph diffs).

### RTC export and board runtime (2026-10-28)

The board had been running the RTC12 checkpoint with the stock sampler: fresh noise, no prefix,
every chunk replacing the last. The live log reconstructed handover jumps up to 1.30 valve units
(84 of 797 above 0.5). Now the trained conditioning runs on the board as well.

- **Graphs** (`rtc_export.py`, picked by `export_bundle.py` when the checkpoint has
  `rtc_training.json`): every `dit_k` takes `rtc_mask` [1,H,1]; dit_0 and the last chunk also
  take `prefix` [1,H,D]. Committed rows are written into the iterate before each step and after
  the last. They get the clean time (bucket 1000) in the action encoder and in every AdaLN; the
  state token and the free rows keep the step time. The clean-time modulations are baked
  constants, so the time/mod graphs and the runtime's step constants are unchanged. The hold
  sits in the FP32 `/euler/` nodes. Only the 8 DiT graphs change (1.7 GB); rsync against the
  previous bundle sends 8 MB.
- **Tests** (`tests/test_rtc_export.py`, tiny head, CPU): the split chunks match rtc_patch's
  sampler at delays 0/1/4/9 (2e-5), committed rows come back bit-exact, and a zero mask
  reproduces the stock export.
- **Fixtures**: `physical_fixture.py --rtc-delays` runs Gr00tPolicy with rtc_patch's sampler. The
  prefix is the recorded valves of the same frames, normalized and clipped. The bundle carries
  FP32 + bf16 RTC fixtures for delays 4/8/10/12. FP32 committed rows equal clip(recorded) within
  6e-8. The step from row d-1 to row d is 0.03-0.05, against 0.02-0.03 in the recording.
- **Runtime** (`groot_live.py`, kaivuriprokkis): d = ceil((recent max observation -> result
  latency + 40 ms) x fps), capped at the trained 12 and at the rows the running plan still
  covers. The prefix is the running plan as the controller interpolates it, at the new chunk's
  row times, converted back to model units. `--rtc off` runs the stock sampler on the same
  engines for A/B.
- **Board** (bundle `~/bundles/groot17-dig640-rtc`): parity PASSES on every prompt and on all
  four RTC delays, with committed rows bit-exact on TensorRT. 251 ms p50 vs 248 without RTC.
  Live, 60 s each, same engines and static scene (valves and pump cut), handover jump measured
  at the first send of each plan:

  | | RTC on (d = 11) | `--rtc off` |
  |---|---|---|
  | handover jump p50 / p99 / max | 0.001 / 0.089 / 0.107 | 0.137 / 0.689 / 0.753 |
  | handovers > 0.5 | 0 / 228 | 9 / 227 |
  | handover row p50 / max | 9.3 / 10.4 | 9.4 / 10.3 |
  | inference p50 | 253.5 ms | 255.5 ms |

### Normalization and full fine-tune comparisons (2026-10-28)

Each changes one thing against A0 = `digging640_rtc12` (head-only, q01/q99 action bounds):

| run | change | s/step (batch 32) | 20k steps |
|---|---|---|---|
| A1 `digging640_rtc12_fixedbounds` | action bounds fixed at the valves' -1/+1 | 2.59 | ~14 h |
| F0 `digging640_rtc12_fullft` | `--tune-llm --tune-visual` (3.14 B trainable) | 4.8 | ~27 h |

- A1: `prepare_dataset.py --action-bounds fixed` writes a complete `meta/stats.json`, so GR00T
  keeps it. These are GR00T's own statistics, except action min/max/q01/q99 = -1/+1. Min-max
  normalization is then the identity: no clipping, and neutral stays 0. The empirical stats are
  kept as `stats_empirical.json`. A 10-step smoke checkpoint carries -1/+1 in
  `processor/statistics.json`, with state stats identical to A0.
- F0 fits on one Spark: ~98 GB used, at least 23 GB available (memguard floor 8).
- `runs/queue_norm_fullft.sh` (`ONLY=A1|F0`) trains and exports each to `~/bundles/groot17-NAME`.
  `runs/to_board.sh NAME` rsyncs it against the RTC bundle and runs board parity. Both are local
  (runs/ is ignored).

## Not done

1. **Delay distribution.** `uniform(0..12)` trains every d equally, but the board commits ~10-12
   rows. A re-train could weight d around the measured latency, or raise d_max to ~15 for slack.
2. **pi0.5 exporter references with a prefix.** The exporter's stock references use OpenPI's
   PyTorch model. Mode C fixtures need the same per-row change there, or JAX references.
3. **Correlated noise.** This is optional and only matters for Mode A inpainting
   (`host/action_stats.py` has Σ). Both trainers sample i.i.d. noise.
4. **The experiment.** Train plain and RTC on the same data (`digging_native_640_simple` first)
   (RTC from scratch, or as the paper does, as a short fine-tune of the plain checkpoint).
   Compare with the board's join metrics at matched delays. Then drive.
5. **GR00T EMA (later, low priority).** N1.7 and the HF Trainer have none. Plan: a
   `TrainerCallback` in `groot/excavator/finetune.py` (flag `--ema-decay`, off by default) keeps an
   fp32 copy of the trainable params, updates it in `on_step_end` with `torch._foreach_lerp_`
   (decay 0.99-0.999, after the optimizer step), and in `on_save` writes it as the checkpoint's
   `model-*.safetensors` (live weights kept beside it for a live-vs-EMA comparison), so the exporter
   needs no change; record it in `rtc_training.json`. Frozen params are unaffected. Cost ~6.5 GB
   head-only, ~12.6 GB full fine-tune; DeepSpeed ZeRO-2 shards only optimizer state, so the copy is
   whole per GPU. Test: a short run whose checkpoint differs from the base by the EMA-expected
   amount (the pi0.5 check), then load and export it.
