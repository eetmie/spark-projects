# SmolVLA GB10 -> Jetson Notes

## Objective

Fine-tune SmolVLA on this GB10 machine, export ONNX here, then run inference on Jetson Orin Nano through TensorRT.

The important boundary is:

- GB10 machine: PyTorch, LeRobot training, ONNX export, ONNX structural validation.
- Jetson Orin Nano: TensorRT engine build from ONNX, then TensorRT inference.

The Orin Nano can usually build a TensorRT engine from a valid ONNX file, although first build may be slow and memory-sensitive. It is not the right place to export PyTorch/LeRobot checkpoints to ONNX.

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

## Directory Layout

- `export_valid_onnx.py`: conservative ONNX export script. Produces FP32 ONNX so Jetson TensorRT can build FP16.
- `exports/`: valid ONNX exports and future TensorRT artifacts.
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

## Jetson TensorRT Plan

Recommended path:

1. Export ONNX on GB10.
2. Copy `exports/*.onnx` plus tokenizer/preprocess/postprocess code to Jetson.
3. On Jetson, build a TensorRT engine from ONNX with FP16 enabled.
4. Run inference with a TensorRT runner.

Likely Jetson command shape for TensorRT 10 / recent JetPack:

```bash
trtexec \
  --onnx=smolvla_base_fp32_valid.onnx \
  --saveEngine=smolvla_base_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:4096
```

Older TensorRT examples may use `--workspace=4096` instead of `--memPoolSize=workspace:4096`.

Expect this to be the next hard part. The Orin Nano has limited shared memory, and this monolithic graph is about 1.5 GB as FP32 ONNX. ONNX-to-TensorRT build on the Orin Nano is plausible, but not guaranteed: unsupported ops, tactic memory, or builder OOM may stop it. If full monolithic TensorRT build fails, the fallback is to split the graph, reduce exported scope/precision, or build a smaller runtime around subgraphs.

Do not export PyTorch/LeRobot to ONNX on the Orin Nano. Do that on GB10. TensorRT engines should be built on the target Jetson or a very similar Jetson/TensorRT/CUDA stack; a GB10-built engine is not the right artifact for Orin.

## Archived Failed Artifacts

Moved to `archive/failed_export_2026-05-31/`:

- `smolvla_base.onnx`: invalid. ORT rejected it because a Conv input remained bfloat16.
- `smolvla_base.pt`: TorchScript fallback, not useful for the current ONNX->TRT plan.
- `export_onnx.py`: old experimental exporter kept only for reference.
