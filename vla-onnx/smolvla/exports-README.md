# Exports

The `exports/` dir holds **only the deploy bundle to copy to the Orin Nano** — nothing else.
The whole dir is gitignored (it's regenerable from `export_valid_onnx.py`). Spark-local receipts
(this file, `precision_sweep_spark.json`) live one level up so `exports/` stays copy-only.

## The deploy bundle to move to the Orin Nano (everything in `exports/`)

`export_valid_onnx.py` writes a self-contained **deploy bundle** into `exports/`, not just the ONNX:

- `smolvla_base_fp16_static.onnx` (~0.8 GB, not tracked) — **DEPLOY THIS to the Orin Nano.**
  Mixed-precision FP16 copy of the FP32 graph (weights → FP16; LayerNormalization + Softmax + graph
  IO kept FP32; the `inf` attention-mask sentinels clamped to ±1e4 first so FP16 can't NaN). Why a
  pre-cast FP16 file at all: the 1.5 GB FP32 graph **can't be TRT-built within the Orin Nano's 8 GB**
  unified memory — the resident FP32 weights during the build OOM the GPU allocator. Halving the graph
  upstream lets the on-device build complete. The *deployed* engine is FP16 either way (the Orin's
  TRT-EP lowers FP32→FP16 at build time); this just moves the halving before the build. Spark-local
  FP16-vs-FP32 parity: **cosine 0.9999972**, max-abs-diff 0.0087, all-finite. Produced by
  `onnxruntime.transformers.float16` (not `onnxconverter-common`, whose 1.16.0 release crashes on this
  graph), then topologically re-sorted so `onnx.checker` passes.
- `smolvla_base_fp32_static.onnx` (~1.5 GB, not tracked) — **parity gold, NOT for the Orin Nano build**
  (it OOMs the 8 GB build; see above). FP32 SmolVLA export, **num_steps=10**, static graph (the
  boolean-mask `NonZero` was rewritten to `torch.where`, so no device→host sync stall /
  data-dependent-shape fragility on the Orin). Bit-identical in output to the original `*_valid.onnx`
  export, and the source the FP16 file is converted from. On a *bigger* Jetson (not the 8 GB Nano) this
  FP32 graph runs directly through **ONNX Runtime's TensorRT EP at FP16** — ORT partitions the graph,
  builds + caches a TRT engine per subgraph, keeps layernorm / sensitive ops FP32, falls back to the
  CUDA EP for the rest. No `trtexec`, no pure `.engine`. (FP16 not BF16: Orin is compute 8.7 —
  `platform_has_fast_bf16 = n/a`. The old "build in BF16, FP16 is broken" guidance was a *blanket-FP16*
  `trtexec` engine on Blackwell; the partitioned TRT-EP path keeps the overflowing vision-tower ops in
  FP32, so FP16 is the right Orin dtype.)
- `tokenizer/` — vocab-exact tokenizer saved from the checkpoint processor. Point the Orin runtime at
  it with `--model-id exports/tokenizer` (self-contained, no network, no backbone guessing).
- `*preprocessor*` / `*postprocessor*` — normalization stats, needed to normalize `state` in and
  un-normalize padded actions out before driving a real robot (the graph is `sample_actions` only).
- `*.onnx.sha256` (one per ONNX) — verify after transfer: `sha256sum -c <file>.sha256`.

To regenerate, or to make a fewer-steps variant for more reactive runs (run from this dir):

    # FP32 gold + FP16 deploy file in one pass (--fp16-weights writes the FP16 sibling)
    python export_valid_onnx.py --output exports/smolvla_base_fp32_static.onnx --fp16-weights   # num_steps=10
    python export_valid_onnx.py --output exports/smolvla_base_fp32_static_s5.onnx --num-steps 5 --fp16-weights

The FP16 sibling is named by swapping `fp32`→`fp16` in the output filename. If you already have the
FP32 gold and only need to (re)build the FP16 file, call `export_fp16_weights(src, dst)` from
`export_valid_onnx` directly — it converts from the existing FP32 ONNX and leaves it byte-identical.

## Spark-local receipt (NOT copied to the Orin — kept beside this README, one level up)

- `precision_sweep_spark.json` — TRT-engine precision sweep run on the Spark as a cheap proxy. It
  shows that a **blanket-FP16** `trtexec` engine collapses to cosine 0.805 (the SmolVLM vision tower
  has `inf`/out-of-range attention constants that overflow FP16). **This is why the Orin path does
  NOT use blanket FP16:** ORT's TensorRT EP partitions the graph and keeps the overflowing
  vision-tower / layernorm ops in FP32 while lowering the rest to FP16, so on-Orin FP16 parity holds
  (≥ 0.997). BF16 was near-lossless here (0.9974) but has no fast path on the Orin (compute 8.7), so
  it's not the deploy dtype. If on-Orin FP16 parity ever fails, re-export with `--fp16-safe-masks` to
  clamp those `inf` sentinels to a finite value.
