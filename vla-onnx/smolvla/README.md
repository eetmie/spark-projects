# SmolVLA GB10 Fine-Tune and Export Workspace

This workspace is for one narrow path:

1. Fine-tune SmolVLA on this NVIDIA GB10 machine with LeRobot.
2. Export a valid ONNX model on this machine.
3. Copy the ONNX to Jetson Orin Nano.
4. Build and run a TensorRT engine on the Jetson.

Do not use the archived failed export artifacts for deployment.


## Install

This was tested on Python 3.12.3 on NVIDIA GB10 / aarch64 / CUDA 13.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

For an exact reproduction of this machine's working venv, use:

```bash
python -m pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

The lock file is intentionally platform-specific. It is for GB10/aarch64/CUDA 13, not a generic x86 CUDA environment.

## Smoke-Test Commands

Activate the environment:

```bash
source .venv/bin/activate
```

Run the one-step training smoke test using the included SO-101 sample dataset. This only verifies the stack; replace the dataset and robot configuration for real work:

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

## Real Training Data

The included SO-101 data is only a smoke-test fixture. For actual fine-tuning, use your own LeRobot-format dataset and the robot configuration that matches your hardware: camera names, state/action dimensions, normalization statistics, control rate, and action postprocessing.

## ONNX Export

Export the baseline ONNX:

```bash
python export_valid_onnx.py --output exports/smolvla_base_fp32_valid.onnx
```

## Jetson Boundary

Export ONNX here on GB10. On Jetson Orin Nano, build TensorRT from ONNX. Do not plan to run PyTorch-to-ONNX export on the Orin Nano.
