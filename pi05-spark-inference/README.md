# pi05-spark-inference

Benchmarking **π0.5 (openpi)** Vision-Language-Action inference on an **NVIDIA DGX Spark (GB10, Blackwell, aarch64)**.

Goal: measure π0.5 inference latency on this machine and drive it toward real-time
control rates, mirroring the proven [Jetson Thor recipe](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/)
(Thor is the same Blackwell family as GB10).

## Hardware / software (this box)

| | |
|---|---|
| GPU | NVIDIA GB10 (Grace-Blackwell, sm_121) |
| Unified memory | 119 GiB |
| Arch / OS | aarch64 / Ubuntu 24.04 |
| Driver / CUDA | 580.142 / CUDA 13.0 |
| Container base | `nvcr.io/nvidia/pytorch:25.09-py3` |

## Plan (two phases)

1. **Phase 1 — PyTorch BF16 baseline.** Convert the π0.5 LIBERO JAX checkpoint to
   PyTorch, run `policy.infer()` in a loop, report latency percentiles. This is the
   1.0× reference (Thor reports ~163 ms here).
2. **Phase 2 — TensorRT FP8 + NVFP4.** Export PyTorch → ONNX (NVIDIA ModelOpt,
   QDQ nodes) → TRT engine (`trtexec --stronglyTyped`). Thor reports ~94 ms (1.73×).
   Note: pure FP16 is unsupported (Gemma attention dynamic range over the denoising loop).

## Layout

```
docker/     Dockerfile + build/run helpers (GPU container)
scripts/    setup, checkpoint download, JAX->PyTorch conversion
bench/      benchmark_pytorch.py — latency harness
notes/      RUNBOOK.md (step-by-step) + findings.md (results log)
checkpoints/  (gitignored) downloaded + converted weights
results/    benchmark JSON output
third_party/openpi  upstream openpi (cloned)
```

## Quick start

See [notes/RUNBOOK.md](notes/RUNBOOK.md). TL;DR:

```bash
./docker/build.sh                 # build the image
./docker/run.sh                   # drop into the container (repo + caches mounted)
# --- inside the container ---
./scripts/setup_in_container.sh   # pip install openpi + deps (once per build)
./scripts/download_checkpoint.sh  # fetch pi05_libero (JAX) from gs://openpi-assets
./scripts/convert_to_pytorch.sh   # JAX -> PyTorch safetensors
python bench/benchmark_pytorch.py --checkpoint checkpoints/pi05_libero_pytorch
```

## Status

See `notes/findings.md` for the running results log.
