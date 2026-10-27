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

## Not done

0. **GR00T RTC sampling on the full model.** It has been run through the tiny-head tests only.
   The training path ran at full size.
1. **GR00T export of an RTC checkpoint.** The split exporter traces the action encoder and DiT
   with a scalar timestep. Per-token time means an encoder input `[B, H]` and per-token adaLN in
   every DiT engine. The board's `flow_rtc` CUDA adapter covers NVIDIA's strength ramp only.
2. **pi0.5 exporter references with a prefix.** The exporter's stock references use OpenPI's
   PyTorch model. Mode C fixtures need the same per-row change there, or JAX references.
3. **Correlated noise.** This is optional and only matters for Mode A inpainting
   (`host/action_stats.py` has Σ). Both trainers sample i.i.d. noise.
4. **The experiment.** Train plain and RTC on the same data (`digging_native_640_simple` first)
   (RTC from scratch, or as the paper does, as a short fine-tune of the plain checkpoint).
   Compare with the board's join metrics at matched delays. Then drive.
