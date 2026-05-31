# Findings — π0.5 inference on GB10

Running log of results and gotchas. Newest first.

## Environment (2026-05-31)
- GPU: NVIDIA GB10 (Blackwell, sm_121), 119 GiB unified memory
- aarch64 / Ubuntu 24.04 / driver 580.142 / CUDA 13.0
- Container: `nvcr.io/nvidia/pytorch:25.09-py3`
- Reference target (Jetson Thor, same Blackwell family):
  PyTorch BF16 ~163 ms → TRT FP8+NVFP4 ~94 ms (1.73×), cosine sim 0.9956

## Phase 1 — PyTorch BF16 baseline
- [x] image built (openpi baked in; torch intact, CPU jax)
- [x] checkpoint downloaded (pi05_libero, 12 GB, gs://openpi-assets)
- [x] converted to PyTorch (6.8 GB model.safetensors)
- [x] baseline latency measured ✅

### Results

| date | config | precision | steps | mean ms | p90 ms | Hz | notes |
|------|--------|-----------|-------|---------|--------|----|-------|
| 2026-05-31 | pi05_libero | BF16 (eager-ish) | default(~10) | **202.8** | 234.0 | **4.9** | min 172, stdev 25; action chunk [10,7]; load 42s |
| 2026-05-31 | pi05_libero | BF16 (eager-ish) | 10 | 200.5 | 229.0 | 5.0 | confirms default ≈ 10 denoise steps |
| 2026-05-31 | pi05_libero | BF16 (eager-ish) | 5 | 163.9 | 191.5 | 6.1 | ~37 ms faster than 10 steps |

**Step-scaling read:** 5→10 steps = 164→200 ms, i.e. ~7 ms per extra denoise step on
top of a ~130 ms fixed cost (vision encode + prefill). So latency is dominated by the
fixed forward, not the flow-matching loop — Phase 2 (TRT FP8+NVFP4) attacks that fixed
cost and is the real lever.

**Headline:** π0.5 runs on GB10 at ~203 ms / **~4.9 Hz** in PyTorch BF16.
But note: `torch._inductor: Not enough SMs to use max_autotune_gemm mode` — so
`pytorch_compile_mode='max-autotune'` does NOT fully engage on GB10; this is
effectively an eager/partially-compiled baseline. Reference: Jetson Thor PyTorch
BF16 ~163 ms. The big win is Phase 2 (TRT FP8+NVFP4 → Thor saw ~94 ms, 1.73×).

Since the action chunk is 10 steps, one inference yields 10 actions → effective
control rate can be higher than 4.9 Hz if actions are executed open-loop within a chunk.

## Phase 2 — TensorRT FP8 + NVFP4 ✅ (2026-06-01)

Followed the Jetson Thor recipe (`phase2/openpi_on_thor/`), which is **API-compatible
with our openpi main clone** (c23745b5) — all internals it needs exist, no re-clone to
175f89c3 required. Pipeline: convert → `pytorch_to_onnx.py` (FP8+NVFP4, attn QDQ) →
`build_engine.sh` (trtexec) → `pi05_inference.py`.

| backend | end-to-end | model-only | rate | cosine sim | notes |
|---------|-----------|-----------|------|-----------|-------|
| PyTorch BF16 | 200.8 ± 22.0 ms | 198.3 ms | 5.0 Hz | — | matches Phase 1 baseline |
| **TRT FP8+NVFP4** | **94.7 ± 10.9 ms** | **92.2 ms** | **~10.6 Hz** | **0.99719** | **2.12× faster** |

- trtexec's own profiling (synthetic, no transfers): mean 88.0 ms, p90 91.7, p99 98.2.
- Engine 2.9 GB (vs 6.8 GB BF16 safetensors). Engine build: 323 s (single-core CPU
  bound — trtexec tactic selection; GPU ~idle during build).
- **Accuracy held with DUMMY calibration** (HF_HUB_OFFLINE → CalibrationDataset falls
  back to dummy): cosine 0.997 vs PyTorch, matching Thor's 0.9956. Real LIBERO-dataset
  calibration is a possible follow-up but evidently not needed for fidelity here.
- The exported ONNX **unrolls the full sample_actions** (prefix forward + 10 denoise
  steps), so these latencies are complete action-chunk generation, not a single step.

**GB10 vs Jetson Thor:** GB10 ~94.7 ms end-to-end vs Thor ~94 ms — essentially identical
on the same Blackwell-family TRT path. (Our BF16 baseline was slower than Thor's, so our
*speedup* is bigger: 2.12× vs Thor's 1.73×.)

### Phase 2 gotchas
- `build_engine.sh` hardcoded `/usr/src/tensorrt/bin/trtexec`; nvcr image has it at
  `/opt/tensorrt/bin` → patched the script to auto-detect.
- `ACTION_HORIZON` must be **10** for pi05_libero (script default is 15).
- lerobot git-pin needs import-time deps not in --no-deps: `datasets deepdiff jsonlines
  draccus termcolor zarr av gymnasium`.
- gemma ONNX patch (`patches/apply_gemma_fixes.py`) must be applied to installed
  transformers (RMSNorm extra_repr guard + explicit attn reshape dim for FP4 block quant).

## Gotchas log
- **gsutil CRC32c**: container's crcmod lacks the C ext → gsutil refuses composite
  downloads. Fix: `-o GSUtil:check_hashes=never` (in download_checkpoint.sh).
- **torch pin**: openpi pins `torch==2.7.1`; must `pip install --no-deps` to keep the
  base image's Blackwell torch (`2.9.0a0...nv25.09`). CPU `jax==0.5.3` is enough for
  the PyTorch inference path (jax used only for tree utils + orbax load).
- **lerobot**: `policy_config` → `training.checkpoints` → `data_loader` imports
  `lerobot` (git-pinned, torch-pinning, training-only). Inference doesn't need it →
  benchmark builds the `Policy` object directly, bypassing that import.
- **tied embedding**: `save_model` dedups `language_model.embed_tokens.weight` (shares
  storage with `paligemma.lm_head.weight`). openpi's `load_pytorch` uses strict load
  and trips on it → load with `strict=False` (the tie auto-fills embed_tokens).
- **assets not copied**: convert script copies assets from `checkpoint_dir.parent/assets`
  (wrong here); norm_stats live in the JAX ckpt at `assets/physical-intelligence/libero`.
  We copy them into the PyTorch ckpt dir (convert_to_pytorch.sh now does this).
