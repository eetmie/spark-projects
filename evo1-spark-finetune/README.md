# EVO1 on DGX Spark — host setup and split-ONNX exploration

This playbook starts an EVO1 path without disturbing the older SmolVLA/X-VLA
environments. The bootstrap path now covers Spark export through Orin parity:

1. pin the first clean released LeRobot integration;
2. download and verify the correct native InternVL3 base;
3. prove the policy can be instantiated on the GB10;
4. size an Orin-safe split-ONNX profile from real checkpoint headers.

See [the split profile](notes/split_onnx_profile.md) for the graph boundaries and
measured Orin result. This is still an infrastructure-only bundle: its action head is
randomly initialized and must never control a robot.

## Why LeRobot 0.6.1

EVO1 entered LeRobot in 0.6.0. This workspace pins **0.6.1**, whose patch fixes two
relevant integration edges: checkpoint-loaded state/action normalization stats are
re-padded to EVO1's fixed widths, and FlashAttention availability is detected through
Transformers rather than by import alone.

LeRobot 0.6.1 declares `torch>=2.7,<2.12` and
`torchvision>=0.22,<0.27`. On this CUDA 13 aarch64 machine the matching pair is
`torch==2.11.0+cu130` and `torchvision==0.26.0+cu130`. Do not reuse the SmolVLA
LeRobot 0.5.1 / torch 2.12 environment.

## What “base” means for EVO1

There is no generic pretrained LeRobot EVO1 robot-policy checkpoint. For a new task,
`policy.type=evo1` loads **`OpenGVLab/InternVL3-1B-hf`** as its VLM and initializes
a fresh flow-matching action head. Stage 1 freezes the VLM and trains that head; Stage 2
fine-tunes both.

The downloader pins InternVL3 revision
`014c0583a0d4bedf29fbe2dbff4f865eb998e171`, the exact base used by LeRobot's verified
EVO1 LIBERO recipe. The `-hf` suffix matters: EVO1 expects the native Transformers
InternVL implementation.

The older `MINT-SJTU/Evo1_*` artifacts are author-format DeepSpeed checkpoints, not
LeRobot `PreTrainedPolicy` directories. The current known LeRobot-format reference
checkpoint is `zuoxingdong/evo1_libero`, but it is LIBERO-specific and is used here
only as an architecture/profile reference.

For a pretrained transfer initializer, the current recommendation is
`MINT-SJTU/Evo1_RoboTwin2_clean`; its broader 50-task training is a better fit than the
6-joint SO100 example unless the target robot is actually SO100/SO101. See the
[checkpoint candidate assessment](notes/checkpoint_candidates.md), including a strict
LeRobot 0.6.1 conversion proof.

## Setup

```bash
cd /home/masi-pgx/spark-projects/evo1-spark-finetune
bash setup.sh
source .venv/bin/activate
```

`setup.sh` is idempotent. It creates a private Python 3.12 environment, installs the
pinned CUDA 13 stack, downloads the 1.9 GB InternVL base to
`models/InternVL3-1B-hf/`, verifies it against the Hub revision, imports EVO1, and
instantiates the 775.20 M-parameter policy on CUDA. Its state dict has 776.14 M tensor
elements because it also serializes two fixed sinusoidal buffers. FlashAttention is
intentionally not installed for this first pass; LeRobot's eager-attention fallback is the
safer ONNX export reference.

For a fast validation after setup:

```bash
.venv/bin/python preflight.py
```

To repeat the full CUDA model-load check:

```bash
.venv/bin/python preflight.py --load-model
```

## Checkpoint profiling

The profiler reads only a safetensors JSON header, never tensor payloads:

```bash
.venv/bin/python tools/inspect_checkpoint.py \
  /path/to/evo1-checkpoint/model.safetensors --detail
```

It supports both raw InternVL3 and LeRobot EVO1 key layouts. For a complete trained EVO1
checkpoint it prints component sizes and the proposed ten-TRT-engine plus CPU-embedding
profile under the existing 100 M parameter Orin build budget.

Original MINT DeepSpeed checkpoints can be audited and converted to LeRobot's native-HF
state-dictionary namespace with `tools/convert_mint_checkpoint.py`. This converts weights
only; the target feature contract and normalization processors must still be packaged and
validated before deployment.

## Training handoff

Once the target dataset/camera contract is chosen, Stage 1 starts from the local VLM:

```bash
.venv/bin/lerobot-train \
  --dataset.repo_id=YOUR_ORG/YOUR_DATASET \
  --policy.type=evo1 \
  --policy.training_stage=stage1 \
  --policy.vlm_model_name=models/InternVL3-1B-hf \
  --policy.device=cuda \
  --policy.use_flash_attn=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.max_state_dim=24 \
  --policy.max_action_dim=24 \
  --batch_size=4 \
  --steps=5000 \
  --output_dir=outputs/evo1-stage1
```

Those state/action maxima are EVO1's padded internal widths, not a declaration that the
robot has 24 controls. LeRobot 0.6.1 pads dataset statistics and tensors, then crops
actions back to the dataset's real output dimension. Camera count and
`policy.max_views` must be decided from the actual training/deployment contract before
training because they affect prompt positions and the final ONNX sequence shape.

## Export and Orin bootstrap

```bash
# Export the 11 fixed-shape FP32 graphs, then emit the native CUDA fixture.
.venv/bin/python export_split_onnx.py
.venv/bin/python emit_reference.py

# Keep graph I/O FP32, convert TRT graph weights to mixed FP16, and preserve
# LayerNormalization/Softmax in FP32. The 544 MB token table remains FP32/CPU.
.venv/bin/python fp16_weights.py
```

The result is a checksummed 1.827 GB bundle: ten TensorRT graphs plus one CPU token
embedding graph. Copy it with [`../orin-nano/evo1-runtime/`](../orin-nano/evo1-runtime/)
to the Jetson, build each engine in an isolated subprocess, and run the included fixture.

## Current status

- Repository updated from `origin/main`: already current on 2026-09-02.
- LeRobot 0.6.1, exact InternVL revision, CUDA stack, and tokenizer are pinned.
- All 11 ONNX graphs pass the checker; Spark FP32 split action parity is effectively
  exact (cosine 0.999999999999, max abs 3.47e-6) against the native CUDA fixture.
- All ten FP16 TensorRT engines build on the Orin Nano Super; cache size is 1.3 GB.
- Full Orin parity passes at cosine 0.999606 vision, 0.999546 valid fused tokens, and
  0.999991 final action.
- Device-resident action I/O binding reduces a cached 32-step chunk from 390.61 ms to
  294.85 ms (24.5%) at 4.77 GB peak RSS with identical output. See the
  [Orin performance sweep](../orin-nano/evo1-runtime/notes/performance.md).
- The pinned RoboTwin checkpoint converts to all 728 LeRobot 0.6.1 tensors with exact
  key/shape coverage and passes a strict serialized load.
- Remaining deployment blocker: train/obtain a real LeRobot EVO1 policy checkpoint. The
  selected checkpoint still needs processors, native-to-converted output parity, and a
  target-embodiment task-quality gate. The random-head bootstrap remains `deployable: false`.
