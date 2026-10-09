# GR00T N1.6 / N1.7 split export

`nvidia/GR00T-N1.6-3B` cut into 26 graphs for the 8 GB Orin Nano Super, run there on the
TensorRT runtime alone. Benchmarked through
[jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla), which vendors
`groot_split.py`, `reference.py`, `export_split_onnx.py` (export/groot/) and the runtime
(`bench/vendor/groot_trt.py`, from `pipeline.py` + `trt_runtime.py`).

```bash
./setup.sh                                   # .venv-groot + Isaac-GR00T n1.6.1-release source
.venv-groot/bin/python reference.py --out work/ref.npz            # stock PyTorch FP32 fixture
.venv-groot/bin/python check_split.py --ref work/ref.npz           # split == stock, in PyTorch
.venv-groot/bin/python export_split_onnx.py --ref work/ref.npz --out work/groot-n16-split
.venv-groot/bin/python parity.py --bundle work/groot-n16-split --ref work/ref.npz   # ORT CPU
```

## Measured (2026-10-09, Orin Nano Super, JetPack 7.2.1, 3 views, 4 steps)

26 engines, ~105 M params each; every build peaked at 3.3 GB RSS with >= 3.9 GB free.
Loaded: 5.44 GB system used, 5.56 GB peak, swap untouched. 335.7 ms p50 (2.98 Hz).
TensorRT FP16 vs stock PyTorch FP32: cosine 0.999999, full-chunk max 0.26 % of range.

## Why pure TensorRT

ORT's TensorRT EP keeps the ONNX initializers in host memory beside the engines; on
unified memory both count (~5.5 bytes/param for X-VLA). Engines alone hold each weight
once in FP16, and all contexts share one 17 MB scratch buffer.

## FP16 traps

- **Qwen3 massive activation.** Layer 2 writes ~16150 into the sink token's residual;
  RMSNorm's mean(x^2) is then ~1.3e5 and overflows FP16. ORT's `float16` converter
  inserts FP32->FP16->FP32 Cast round trips *between* blocked nodes, so op_block_list
  alone does not save it. `strip_fp16_roundtrips` removes those pairs; the RMSNorm nodes
  are also blocked by name. Before: llm_7 cos 0.567, actions cos 0.992.
- **Time sinusoids** are their own FP32 graph (`time`).
- ORT-CPU FP16 reproduces TensorRT's FP16 error exactly, so it is a valid Spark-side
  proxy for this class of bug; `work/`-style per-graph bisection finds the stage.
- SigLIP2's vendored eager attention ignores per-image `seq_len_list` (only flash-attn
  honours it); the reference patches it to per-image attention.

## GR00T N1.7

`nvidia/GR00T-N1.7-3B` (backbone Cosmos-Reason2-2B = Qwen3-VL) cut into 26 graphs the same
way: vision x4 (Qwen3-VL ViT; DeepStack mergers ride in their layer's chunk), llm x8,
cond x2 (vlln + the new 4-layer VL self-attention + state encoder), time, dit x11.
~2.51 B deployed params. Own venv (transformers 4.57.3) and source checkout.

```bash
./setup17.sh                                  # .venv-groot17 + Isaac-GR00T n1.7-release source
.venv-groot17/bin/python reference17.py --out work/ref17.npz     # stock PyTorch FP32 fixture
.venv-groot17/bin/python check_split17.py --ref work/ref17.npz
.venv-groot17/bin/python export_split_onnx17.py --ref work/ref17.npz --out work/groot-n17-split
.venv-groot17/bin/python parity17.py --bundle work/groot-n17-split --ref work/ref17.npz
```

Tokenizer/processor files come from `nvidia/Cosmos-Reason2-2B`, a gated repo; until access
is granted `--vlm-files Qwen/Qwen3-VL-2B-Instruct` builds the same architecture (all
checkpoint keys load) but the chat template is not verified to be Cosmos's.

Contract: embodiment `xdof_relative_eef_relative_joint` (3 cameras x 2 frames: now and 30
frames earlier; robocasa is not an N1.7 pretrained embodiment), 256x256 per image (the
letterbox/crop eval transform already lands on a multiple of 32, so Qwen's resize is a
no-op), 64 tokens per image, prompt 412 tokens padded to 448, chunk 40 x 132, 4 steps.

What differs from N1.6, each found by bisecting against stock:

- **backbone_features is pre-norm.** Stock takes `hidden_states[-1]` of
  Qwen3VLForConditionalGeneration, which (transformers 4.57.3) is the last decoder layer's
  output BEFORE the final RMSNorm, ~1.5e4 massive activation included. vlln (LayerNorm,
  FP32) consumes it directly.
- **`get_action` overwrites `backbone_features` in place** with the post-vlln/VL-self-attn
  tensor, so the reference's saved features are the conditioning, not the LLM output.
- **DeepStack:** ViT layers 5/11/17 each feed a merger whose output is added to the
  hidden state after LLM layers 0/1/2 at image positions. The host scatters them into
  zero [1,S,2048] buffers; the graphs just add.
- **VL self-attention** has no mask in stock (batch 1, no pads); the split passes a key
  bias that hides the right-padding, which keeps real tokens exact.
- The ViT's eager path already attends per image (unlike N1.6's SigLIP2 patch), and
  each image is independent, so a frame's vision output can be cached across calls.

Measured on the Spark: split vs stock FP32 action max 1.4e-6; ORT CPU mixed FP16 vs
stock: cos 0.9999999, full chunk max 0.041 % of range.
