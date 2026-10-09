# GR00T N1.6 split export

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
