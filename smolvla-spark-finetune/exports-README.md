# Exports

The `exports/` dir holds **only the deploy bundle to copy to the Orin Nano** — nothing else.
The whole dir is gitignored (it's regenerable from `export_valid_onnx.py`). Spark-local receipts
(this file, `precision_sweep_spark.json`) live one level up so `exports/` stays copy-only.

## The deploy bundle to move to the Orin Nano (everything in `exports/`)

`export_valid_onnx.py` writes a self-contained **deploy bundle** into `exports/`, not just the ONNX:

- `smolvla_base_fp32_static.onnx` (~1.5 GB, not tracked) — FP32 SmolVLA export, **num_steps=10**,
  static graph (the boolean-mask `NonZero` was rewritten to `torch.where`, so no device→host sync
  stall / data-dependent-shape fragility on the Orin). Bit-identical in output to the original
  `*_valid.onnx` export. On the Orin (JetPack 7.2) it runs through **ONNX Runtime's TensorRT EP at
  FP16** — ORT partitions the graph, builds + caches a TRT engine per subgraph, keeps layernorm /
  sensitive ops FP32, and falls back to the CUDA EP for the rest. No `trtexec`, no pure `.engine`.
  (FP16 not BF16: Orin is compute 8.7 — `platform_has_fast_bf16 = n/a`. The old "build in BF16, FP16
  is broken" guidance was a *blanket-FP16* `trtexec` engine on Blackwell; the partitioned TRT-EP path
  keeps the overflowing vision-tower ops in FP32, so FP16 is the right Orin dtype.)
- `tokenizer/` — vocab-exact tokenizer saved from the checkpoint processor. Point the Orin runtime at
  it with `--model-id exports/tokenizer` (self-contained, no network, no backbone guessing).
- `*preprocessor*` / `*postprocessor*` — normalization stats, needed to normalize `state` in and
  un-normalize padded actions out before driving a real robot (the graph is `sample_actions` only).
- `smolvla_base_fp32_static.onnx.sha256` — verify after transfer: `sha256sum -c <file>.sha256`.

To regenerate, or to make a fewer-steps variant for more reactive runs (run from this dir):

    python export_valid_onnx.py --output exports/smolvla_base_fp32_static.onnx       # num_steps=10
    python export_valid_onnx.py --output exports/smolvla_base_fp32_static_s5.onnx --num-steps 5

## Spark-local receipt (NOT copied to the Orin — kept beside this README, one level up)

- `precision_sweep_spark.json` — TRT-engine precision sweep run on the Spark as a cheap proxy. It
  shows that a **blanket-FP16** `trtexec` engine collapses to cosine 0.805 (the SmolVLM vision tower
  has `inf`/out-of-range attention constants that overflow FP16). **This is why the Orin path does
  NOT use blanket FP16:** ORT's TensorRT EP partitions the graph and keeps the overflowing
  vision-tower / layernorm ops in FP32 while lowering the rest to FP16, so on-Orin FP16 parity holds
  (≥ 0.997). BF16 was near-lossless here (0.9974) but has no fast path on the Orin (compute 8.7), so
  it's not the deploy dtype. If on-Orin FP16 parity ever fails, re-export with `--fp16-safe-masks` to
  clamp those `inf` sentinels to a finite value.
