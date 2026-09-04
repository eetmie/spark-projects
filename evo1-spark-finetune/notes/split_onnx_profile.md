# EVO1 split-ONNX profile — validated bootstrap

Status: **exported, parity-checked, and TensorRT-tested on Orin Nano Super (2026-09-02)**.
The tested bundle uses the real InternVL base and a deterministic random action head, so
it validates the deployment shape but is explicitly not a robot policy.

## Inputs pinned for this design

- LeRobot `0.6.1` (`7e241bd630a3719a56157a497ce5d08f244784f1`). EVO1 first ships in
  0.6.0, but 0.6.1 fixes checkpoint-loaded EVO1 normalization stats and FlashAttention
  detection, so there is no reason to start on 0.6.0.
- `OpenGVLab/InternVL3-1B-hf` revision
  `014c0583a0d4bedf29fbe2dbff4f865eb998e171`. The `-hf` model is mandatory: LeRobot's
  implementation uses the native Transformers InternVL layout and no remote modeling code.
- Serialized tensor layout measured from the verified LeRobot-format checkpoint
  `zuoxingdong/evo1_libero` revision
  `515921f4a2c1d3f3ad523721eafa26fdf2af315b` by reading its safetensors header.
- The splits started from the **100 M tensor-element / 0.40 GB FP32 build budget**
  measured by `build_probe.py` in jetson-orin-nano-vla (`bench/tools/`), then every proposed graph
  was actually built on the target Orin. The worst measured build still left 2.995 GB
  system memory available.

The `MINT-SJTU/Evo1_*` Hub files are original-author DeepSpeed checkpoints
(`mp_rank_00_model_states.pt` plus `checkpoint.json`). They are not LeRobot
`PreTrainedPolicy` directories. A fresh LeRobot EVO1 run initializes the VLM from the
pinned InternVL3 checkpoint and initializes a new action head; there is no generic
robot-policy base checkpoint to download.

## Exact released-policy serialized layout

| component | tensor elements | FP32 build data |
|---|---:|---:|
| InternVL3 vision tower, 24 layers | 304,012,288 | 1.216 GB |
| multimodal projector | 4,482,816 | 0.018 GB |
| token embedding | 135,899,904 | 0.544 GB |
| retained language layers, 14 x 14,912,384 | 208,773,376 | 0.835 GB |
| action/state input encoders | 2,618,624 | 0.010 GB |
| action transformer, 8 x 9,645,440 | 77,163,520 | 0.309 GB |
| action time encoding, pool, norm, and MLP output | 43,188,016 | 0.173 GB |
| **total** | **776,139,440** | **3.105 GB** |

The released policy has 775,198,640 parameters. Its state dict contains 776,139,440
elements because two fixed sinusoidal buffers add 940,800 serialized values: 896,000
for diffusion time and 44,800 for action positions. Those constants still matter to
ONNX/TensorRT sizing, so the tables and profiler deliberately count state-dict elements.

`OpenGVLab/InternVL3-1B-hf` itself has 938,193,024 parameters. LeRobot keeps the first
14 language layers instead of all 24, and action inference does not need the language
model's vocabulary output head. That is why the deployed policy is smaller than the raw
VLM even after adding the 122.03 M trainable action head plus its fixed buffers.

Reproduce the table against any complete or header-only checkpoint file:

```bash
.venv/bin/python tools/inspect_checkpoint.py /path/to/model.safetensors --detail
```

## Validated graph profile

| backend / frequency | graph | tensor elements | FP32 build data |
|---|---|---:|---:|
| TRT cold x1 | vision embeddings + layers 0-6 | 89,841,664 | 0.359 GB |
| TRT cold x1 | vision layers 7-13 | 88,187,904 | 0.353 GB |
| TRT cold x1 | vision layers 14-20 | 88,187,904 | 0.353 GB |
| TRT cold x1 | vision layers 21-23 + multimodal projector | 42,277,632 | 0.169 GB |
| ORT CPU x1 | token embedding lookup | 135,899,904 | 0.544 GB |
| TRT cold x1 | language layers 0-5 | 89,474,304 | 0.358 GB |
| TRT cold x1 | language layers 6-11 | 89,474,304 | 0.358 GB |
| TRT cold x1 | language layers 12-13 + final norm | 29,825,664 | 0.119 GB |
| TRT cold x1 | state encoder + K/V projections for 8 action blocks | 13,803,392 | 0.055 GB |
| TRT hot x32 | action encoder + Q/out/FFN for blocks 0-7 | 66,874,752 | 0.267 GB |
| TRT hot x32 | action norm + sequence pool + MLP output | 42,292,016 | 0.169 GB |

That is **ten TensorRT engines plus one CPU/ORT embedding lookup**. Every TRT
engine is below the existing 100 M budget. The 135.90 M token table is deliberately not
a TRT engine: by itself it is already 0.544 GB in the FP32 representation TensorRT uses
while building. The validated bundle keeps the table FP32 on CPU and transfers the gathered token
embeddings once per observation. If that transfer matters, try an ORT CUDA Gather after
the safe baseline works.

### Why the action K/V split matters

EVO1's flow head runs 32 Euler steps. Each of its eight transformer blocks uses action
tokens as queries and the same image/language/state context as keys and values. PyTorch's
`nn.MultiheadAttention` repeats all three projections every step, but K and V are loop
invariant. The cold action-context graph can apply the K/V two-thirds of each packed
`in_proj_weight` once. The hot graph retains the Q third, attention, output projection,
and FFN. This is an exact factorization in eval mode (`dropout=0`), not an approximation.

The Euler loop stays in Python/runtime code:

```text
action ~ Uniform(-1, 1)
for i in range(32):
    t = i / 32
    velocity = action_step(action, time_embedding[t], cached_context_kv)
    action = (action + velocity / 32) * action_mask
```

Do not unroll 32 copies into ONNX, and do not reuse X-VLA's interpolation rule. EVO1
integrates from noise at `t=0` toward actions at `t=1` using `x += dt * velocity`.

## Export contract used

1. Keep tokenization, prompt construction, uint8-compatible bicubic resize, ImageNet
   normalization, action sampling, and the Euler loop outside ONNX.
2. Export fixed batch 1, fixed 448 x 448 images, and a checkpoint-specific fixed language
   sequence length. The final profile must use the fine-tuned checkpoint's camera count,
   `max_views`, `max_text_length`, state/action dimensions, and chunk size.
3. With fewer real cameras than `max_views`, run vision only for valid views, scatter their
   projected tokens into the expected placeholder positions, and preserve the exact
   attention mask/positions for absent views. Simply deleting absent placeholders changes
   later text positions and is not parity-safe.
4. Force eager/math attention for the PyTorch export reference; FlashAttention is a
   training/runtime optimization, not part of the portable ONNX contract.
5. First produce FP32 external-data ONNX and compare every graph boundary with PyTorch.
   Only after split parity passes, make FP16-weight copies for Orin TensorRT builds.
6. Bake `num_categories=1`, the deployment action mask, and static dimensions where the
   final fine-tune permits it. Keep the initial noise tensor as an explicit input so parity
   uses exactly the same sample.
7. Defer Real-Time Chunking. Establish ordinary 32-step inference parity first; RTC changes
   the denoising call contract and needs its own correctness test.

## Measured feasibility call

**Proceed to a trained checkpoint.** EVO1 is a better structural candidate than X-VLA for the 8 GB build limit:
the 24 vision and 14 language layers divide cleanly below 100 M tensor elements, and the
action head also divides into two small hot graphs after loop-invariant K/V projections
are hoisted. The token embedding is the only known build-budget exception and has a
straightforward non-TRT fallback.

The FP32 ONNX split is structurally exact against native CUDA: final-action cosine is
0.999999999999 with max absolute error 3.47e-6. The bootstrap runtime also fits: the
ten-engine cache is 1.3 GB, all sessions plus the CPU embedding table peak at 4.754 GB RSS, and a 32-step chunk takes 0.56 s (~1.78 Hz).
Final-action cosine against native LeRobot 0.6.1 FP32 CUDA is 0.999991. The model remains
non-deployable only because this bootstrap action head is random; repeat the exact export
and parity gates after training a task checkpoint.
