# GR00T N1.7: excavator fine-tune and split export

`excavator/` is the pipeline: NVIDIA's N1.7 trainer on the Spark with the excavator modality
config, a GR00T view of the dataset, optional training-time RTC, and `export_bundle.py` for
the board (graphs from the pinned jetson-orin-nano-vla exporter in `third_party/`). See
[../kaivuriprokkis/RTC_TRAINING.md](../kaivuriprokkis/RTC_TRAINING.md).

The files here are the split export it grew from: `nvidia/GR00T-N1.7-3B` (backbone
Cosmos-Reason2-2B = Qwen3-VL) cut into 26 graphs for the 8 GB Orin Nano Super: vision x4
(Qwen3-VL ViT; DeepStack mergers ride in their layer's chunk), llm x8, cond x2 (vlln + the
4-layer VL self-attention + state encoder), time, dit x11. ~2.51 B deployed params.
`groot_split.py`, `export_split_onnx.py` and `parity.py` hold the pieces the N1.7 files share.
N1.6 was removed 2026-10-28; its published exporter and benchmark are in
[jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla).

```bash
./setup17.sh                                  # .venv-groot17 + Isaac-GR00T n1.7-release source
.venv-groot17/bin/python reference17.py --out work/ref17.npz     # stock PyTorch FP32 fixture
.venv-groot17/bin/python check_split17.py --ref work/ref17.npz   # split == stock, in PyTorch
.venv-groot17/bin/python export_split_onnx17.py --ref work/ref17.npz --out work/groot-n17-split
.venv-groot17/bin/python parity17.py --bundle work/groot-n17-split --ref work/ref17.npz   # ORT CPU
```

Tokenizer/processor files come from `nvidia/Cosmos-Reason2-2B`, a gated repo (accept its
terms on huggingface.co). Its config, tokenizer, chat template and processor files are
byte-identical to `Qwen/Qwen3-VL-2B-Instruct`'s (checked 2026-10-09); the published bundle
was exported from the Cosmos files.

Base contract: embodiment `xdof_relative_eef_relative_joint` (3 cameras x 2 frames: now and
30 frames earlier), 256x256 per image, 64 tokens per image, prompt 412 tokens padded to
448, chunk 40 x 132, 4 steps.

Measured on the Spark: split vs stock FP32 action max 1.4e-6; ORT CPU mixed FP16 vs
stock: cos 0.9999999, full chunk max 0.041 % of range.

## Why pure TensorRT

ORT's TensorRT EP keeps the ONNX initializers in host memory beside the engines; on
unified memory both count (~5.5 bytes/param for X-VLA). Engines alone hold each weight
once in FP16, and all contexts share one scratch buffer.

## Traps, each found by bisecting against stock

- **Qwen3 massive activation.** A ~1.5e4 value rides the sink token's residual; RMSNorm's
  mean(x^2) then overflows FP16. ORT's `float16` converter inserts FP32->FP16->FP32 Cast
  round trips *between* blocked nodes, so op_block_list alone does not save it.
  `strip_fp16_roundtrips` removes those pairs; the RMSNorm nodes are also blocked by name.
- **Time sinusoids** are their own FP32 graph (`time`).
- ORT-CPU FP16 reproduces TensorRT's FP16 error exactly, so it is a valid Spark-side
  proxy for this class of bug; per-graph bisection finds the stage.
- **backbone_features is pre-norm.** Stock takes `hidden_states[-1]` of
  Qwen3VLForConditionalGeneration, which (transformers 4.57.3) is the last decoder layer's
  output BEFORE the final RMSNorm, massive activation included. vlln (LayerNorm, FP32)
  consumes it directly.
- **`get_action` overwrites `backbone_features` in place** with the post-vlln/VL-self-attn
  tensor, so the reference's saved features are the conditioning, not the LLM output.
- **DeepStack:** ViT layers 5/11/17 each feed a merger whose output is added to the
  hidden state after LLM layers 0/1/2 at image positions. The host scatters them into
  zero [1,S,2048] buffers; the graphs just add.
- **VL self-attention** has no mask in stock (batch 1, no pads); the split passes a key
  bias that hides the right-padding, which keeps real tokens exact.
- The ViT attends per image and each image is independent, so a frame's vision output
  can be cached across calls.
