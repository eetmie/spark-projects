# SmolVLA GB10 -> Jetson Notes

## Objective

Fine-tune SmolVLA on this GB10 machine, export ONNX here, then run inference on Jetson Orin Nano through TensorRT.

The important boundary is:

- GB10 machine: PyTorch, LeRobot training, ONNX export, ONNX structural validation.
- Jetson Orin Nano: ONNX Runtime + TensorRT EP inference (FP16); the TRT engine is auto-built + cached from the ONNX on first run.

The Orin Nano builds + caches its TensorRT engine from a valid ONNX via ONNX Runtime's TensorRT EP — the first build is slow and memory-sensitive (needs the swap + MAXN_SUPER from `orin-nano/system/`). It is not the right place to export PyTorch/LeRobot checkpoints to ONNX.

## NEXT BIG TASK: split-graph export (2026-06-17)

The on-Orin work proved the **monolithic** ONNX (`export_valid_onnx.py`, the unrolled `sample_actions`
loop) **cannot TRT-build on the Orin Nano's 8 GB** — TRT imports all 450M weights as FP32 working
copies at once (~6 GB floor, node-count-independent). FP16 weights and `--num-steps 5` do NOT fix it
(both still OOM). Settled fix = **export per-component split graphs**; each carries only its weight
slice → builds in ≤60 s, runs in ms (validated on-device with `ainekko/smolvla_base_onnx`).
Full matrix + validation: `orin-nano/smolvla-runtime/notes/findings.md`.

**What to build here on GB10** — mirror `ainekko/smolvla_base_onnx` (9 graphs) and the inference loop
in `github.com/aifoundry-org/ETARS` (`notebooks/smolVLA_export.ipynb`, `src/lerobot/policies/smolvla/
smolvlm_with_expert_onnx.py`, `modeling_smolvla_ort.py`):

- `smolvlm_vision` (image → features, ×1), `smolvlm_text` (tokens → features, ×1)
- `smolvlm_expert_prefill` (vision+text+state → conditioning + **KV cache out**, ×1)
- `smolvlm_expert_decode` (x_t, t, **KV in** → v_t, ONE flow-matching step, run **×N** in Python)
- `state_projector`, `time_in/out_projector`, `action_in/out_projector`

Keep the current `export_valid_onnx.py` monolith only as the **FP32 parity gold** (run parity on a big
box, not the Orin). Export FP32; the Orin's TRT-EP lowers to FP16 per engine (each builds fine, weights
are split). Ship the 9 graphs + `tokenizer/` + normalization stats as the deploy bundle. On the Orin,
the prefill→decode loop gets wired into `orin-nano/smolvla-runtime/backends/ort.py`.

Projected end-to-end on Orin: (vision 33 ms + text + prefill 16.5 ms) once + decode 11.4 ms ×N →
~5–6 Hz at 10 steps, ~9 Hz at 5 steps, full num_steps quality.

## Current Environment

- Python: 3.12.3 in `.venv`
- GPU: NVIDIA GB10, compute capability 12.1
- PyTorch: `2.12.0+cu130`
- LeRobot: `0.5.1`
- Transformers: `5.3.0`
- PEFT: `0.19.1`
- TorchCodec: installed and used for smoke-test dataset video decode
- ONNX: `1.21.0`
- ONNX Runtime: `1.26.0`, CPU providers only on this machine

PyTorch CUDA is working. A dummy SmolVLA CUDA forward produced a valid action chunk on GB10.

Deploy target (Orin Nano, JetPack 7.2): Python 3.12, CUDA 13.2, TensorRT 10.16, `onnxruntime-gpu
1.24.0` from `pypi.jetson-ai-lab.io/sbsa/cu130`. ONNX opset 17 here loads on that ORT 1.24 (verified).
FP16 deploy (BF16 N/A on compute 8.7).

## Directory Layout

- `export_valid_onnx.py`: conservative ONNX export script. Produces FP32 ONNX (the Orin's ORT TensorRT-EP lowers it to FP16 on device) plus a deploy bundle: `exports/tokenizer/` + normalization stats.
- `exports/`: valid ONNX exports + the deploy bundle (tokenizer/, normalization stats) for the Orin.
- `datasets/lerobot_svla_so101_pickplace`: original downloaded official SO-101-ish smoke-test dataset. Videos are AV1; keep as source copy.
- `datasets/lerobot_svla_so101_pickplace_h264`: H.264 working copy used only for local stack smoke tests.
- `outputs/`: LeRobot training outputs/checkpoints.
- `archive/failed_export_2026-05-31`: old failed export attempt and TorchScript fallback. Do not deploy.

## Initial Tests Completed

### 1. PyTorch SmolVLA Forward

Command:

```bash
python export_onnx.py --runs 1 --skip-export
```

This was run before archiving the old script. Result:

- Model loaded on CUDA GB10.
- First warmup action chunk: about 667 ms.
- One measured 10-step action chunk: about 97 ms.
- Output shape: `[1, 50, 6]` from the high-level policy call.

### 2. Baseline ONNX Export

Command:

```bash
python export_valid_onnx.py --output exports/smolvla_base_fp32_valid.onnx
```

Result:

- ONNX checker passed.
- CPU ONNX Runtime session creation passed.
- Output file: `exports/smolvla_base_fp32_valid.onnx`
- Size: about 1.5 GB.

ONNX interface:

- `image0`: float, `[batch, 3, 512, 512]`
- `img_mask0`: bool, `[batch]`
- `lang_tokens`: int64, `[batch, 48]`
- `lang_masks`: bool, `[batch, 48]`
- `state`: float, `[batch, 32]`
- `noise`: float, `[batch, 50, 32]`
- output `actions`: float, `[batch, 50, 32]`

Note: this graph includes padded action/state dimensions of 32. The smoke-test SO-101 action/state uses 6 dims; real deployments must map the relevant model outputs to the target robot action space explicitly.

### 2b. PyTorch-vs-ONNX Parity Check

Command:

```bash
python parity_check_onnx.py \
  --onnx exports/smolvla_base_fp32_valid.onnx \
  --out exports/smolvla_base_fp32_valid.parity.json
```

Result on GB10 with PyTorch reference and CPU ONNX Runtime:

- PyTorch shape: `[1, 50, 32]`
- ONNX shape: `[1, 50, 32]`
- max_abs_diff: `2.6226043701171875e-06`
- mean_abs_diff: `1.4934558123513852e-07`
- cosine: `0.9999999999981618`
- passed `1e-4` and `1e-3` thresholds

This validates PyTorch -> ONNX numerics here. TensorRT parity still needs to be checked on the Jetson after engine build.

### 3. Smoke-Test SO-101 Dataset Load

Downloaded official small dataset for smoke testing only:

- HF dataset: `lerobot/svla_so101_pickplace`
- 50 episodes
- 11,939 frames
- task: `pink lego brick into the transparent box`
- state/action: 6D SO arm joints/gripper
- cameras: `observation.images.up`, `observation.images.side`

The original videos are AV1. Local torchvision/torchcodec could not decode those directly, so a H.264 copy was made:

- `datasets/lerobot_svla_so101_pickplace_h264`

LeRobot dataset item loading passed with:

```python
LeRobotDataset(
    repo_id="lerobot/svla_so101_pickplace",
    root="datasets/lerobot_svla_so101_pickplace_h264",
    video_backend="torchcodec",
)
```

### 4. One-Step Fine-Tune Smoke Test

Command:

```bash
WANDB_MODE=disabled .venv/bin/lerobot-train \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --dataset.root=datasets/lerobot_svla_so101_pickplace_h264 \
  --dataset.video_backend=torchcodec \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.load_vlm_weights=true \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --policy.repo_id=local/smolvla-so101-smoke-loadvlm \
  --output_dir=outputs/smolvla_so101_smoke_loadvlm \
  --steps=1 \
  --batch_size=1 \
  --optimizer.lr=1e-4 \
  --peft.method_type=LORA \
  --peft.r=8
```

Result:

- Dataset loaded.
- SmolVLA loaded with VLM weights.
- LoRA wrapping succeeded.
- One CUDA training step completed.
- Checkpoint written to:
  `outputs/smolvla_so101_smoke_loadvlm/checkpoints/000001/pretrained_model/adapter_model.safetensors`

## Dataset and Robot Configuration Warning

The included SO-101 dataset is only a smoke-test fixture for dependency, video decode, training-loop, and checkpoint-write validation. For useful behavior, train on demonstrations from your own robot/task setup. Match these pieces before trusting actions on hardware:

- camera keys and camera count
- image preprocessing and resize policy
- state vector order, units, and scaling
- action vector order, units, and limits
- normalization statistics
- control frequency and action horizon
- postprocessing from padded SmolVLA output dims to robot commands

## Real Fine-Tune Starting Point

Do not use the bundled SO-101 smoke-test dataset for real training claims. Replace `--dataset.repo_id`, `--dataset.root`, camera names/configuration, robot normalization, state/action mapping, and postprocessing with your own robot dataset and configuration. Start small before a longer run:

```bash
WANDB_MODE=disabled .venv/bin/lerobot-train \
  --dataset.repo_id=YOUR_DATASET_REPO_OR_LOCAL_ID \
  --dataset.root=/path/to/your/lerobot_dataset \
  --dataset.video_backend=torchcodec \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.load_vlm_weights=true \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --policy.repo_id=local/smolvla-your-robot-test \
  --output_dir=outputs/smolvla_your_robot_test \
  --steps=100 \
  --batch_size=1 \
  --optimizer.lr=1e-4 \
  --peft.method_type=LORA \
  --peft.r=8
```

Then try longer runs after confirming loss/logs look sane.

## Jetson deploy path (ONNX Runtime + TensorRT EP)

The Orin side was rebuilt for JetPack 7.2 (2026-06-13). It does **NOT** use `trtexec` / a pure
`.engine` build any more — that monolithic build OOM'd on the Orin's 8 GB shared memory. Instead it
runs the ONNX through **ONNX Runtime's TensorRT execution provider** at **FP16**: ORT partitions the
graph, builds + caches a TensorRT engine per supported subgraph on first run, and falls back to the
CUDA EP for the rest. One inference path, no separate engine-build step. (Why FP16 not BF16: Orin =
compute 8.7 has no fast BF16; `platform_has_fast_bf16 = n/a`. FP16 is the accelerated dtype, with
layernorm/sensitive ops kept FP32.)

Recommended path:

1. Export ONNX on GB10 with `export_valid_onnx.py` — this now also writes a **deploy bundle** next to
   the ONNX: `tokenizer/` (vocab-exact, from the checkpoint's processor) + the normalization stats
   (`policy_preprocessor*` / `policy_postprocessor*`).
2. Copy the whole bundle to the Orin's `orin-nano/smolvla-runtime/exports/`: the `*.onnx`
   (+ `*.onnx.data` sidecar if present), the `tokenizer/` dir, and the normalization stats.
3. On the Orin (one-time): `pip install onnxruntime-gpu` from the **`sbsa/cu130`** Jetson AI Lab
   index (there is no `jp7` index; CUDA-13 aarch64 wheels live under `sbsa`). Verified:
   `onnxruntime-gpu 1.24.0` with TensorRT + CUDA + CPU EPs.
4. Run: `python run_pipeline.py --backend ort --onnx-path exports/<name>.onnx
   --model-id exports/tokenizer --source realsense`. First run builds + caches the engine (minutes,
   needs the 16 GB swap + MAXN_SUPER from `orin-nano/system/`); later runs load from cache.
5. Gate before trusting actions: `python parity.py --onnx exports/<name>.onnx
   --model-id exports/tokenizer` (FP16 vs FP32-CPU, expect cosine ≥ 0.997, all finite). If it FAILS
   (non-finite / cosine drop from the vision tower's `inf` mask constants), re-export here with
   `--fp16-safe-masks` and repeat.

Do not export PyTorch/LeRobot to ONNX on the Orin Nano — do that on GB10. The cached TensorRT engine
is built on the target Orin and is not portable from the GB10.

> Note: the exported graph is `model.sample_actions` only — it does NOT normalize `state` or
> un-normalize `actions` (those live in the bundled `policy_preprocessor`/`postprocessor`). The Orin
> "model pipeline" stage runs raw; applying the normalization stats is required before driving a real
> robot.

## Archived Failed Artifacts

Moved to `archive/failed_export_2026-05-31/`:

- `smolvla_base.onnx`: invalid. ORT rejected it because a Conv input remained bfloat16.
- `smolvla_base.pt`: TorchScript fallback, not useful for the current ONNX->TRT plan.
- `export_onnx.py`: old experimental exporter kept only for reference.
