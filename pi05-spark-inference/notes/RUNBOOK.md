# Runbook — π0.5 inference on DGX Spark (GB10)

## Phase 1 — PyTorch BF16 baseline

```bash
# 1. Build the image (thin; heavy install happens in-container)
./docker/build.sh

# 2. Enter the container (repo + ./.cache mounted, all GPUs)
./docker/run.sh

# --- inside the container ---
# 3. Install openpi for PyTorch inference (keeps container torch, CPU jax)
./scripts/setup_in_container.sh

# 4. Download the π0.5 LIBERO JAX checkpoint (public GCS bucket)
./scripts/download_checkpoint.sh pi05_libero

# 5. Convert JAX -> PyTorch safetensors
./scripts/convert_to_pytorch.sh pi05_libero

# 6. Benchmark
python bench/benchmark_pytorch.py \
    --config pi05_libero \
    --checkpoint checkpoints/pi05_libero_pytorch \
    --warmup 5 --iters 50

# Optional: sweep denoising steps (latency scales ~linearly with steps)
for s in 5 10 20; do
  python bench/benchmark_pytorch.py --config pi05_libero \
    --checkpoint checkpoints/pi05_libero_pytorch --num-steps $s --iters 30
done
```

Results land in `results/*.json` and should be logged in `findings.md`.

## Known risk points (and the chosen mitigations)

| Risk | Mitigation |
|---|---|
| `jax[cuda12]==0.5.3` clashes with CUDA 13 container | install **CPU jax** only; torch (container) does GPU work |
| openpi pins `torch==2.7.1`, would clobber Blackwell torch | `pip install --no-deps`; hand-pick other deps |
| PyTorch path needs patched transformers | `setup_in_container.sh` overlays `transformers_replace/*` |
| `infer()` calls `jax.tree.map` even on PyTorch path | CPU jax is enough; no GPU JAX plugin needed |

## Phase 2 — TensorRT FP8 + NVFP4 ✅ DONE

Uses the Jetson Thor scripts in `phase2/openpi_on_thor/` (fetched via jetson-ai-lab
download.sh). Image: `pi05-spark-trt:latest` (`docker/Dockerfile.phase2`).

```bash
# 0. Build the Phase 2 image (extends pi05-spark:latest with onnx/trt/lerobot stack)
docker build -t pi05-spark-trt:latest -f docker/Dockerfile.phase2 .

# Common docker run prefix (PYTHONPATH exposes openpi_on_thor; HF offline => dummy calib)
DR='docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v '"$PWD"':/workspace -v '"$PWD"'/.cache:/cache -w /workspace \
    -e PYTHONPATH=/workspace/phase2 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    pi05-spark-trt:latest'

# 1. PyTorch -> ONNX (FP8 + NVFP4 + attention QDQ). ~6 min, writes onnx/model_fp8_nvfp4.{onnx,data}
$DR python phase2/openpi_on_thor/pytorch_to_onnx.py \
    --checkpoint_dir /workspace/checkpoints/pi05_libero_pytorch \
    --output_path  /workspace/checkpoints/pi05_libero_pytorch \
    --config_name pi05_libero --num_steps 10 --precision fp8 \
    --enable_llm_nvfp4 --quantize_attention_matmul --num_calibration_samples 8

# 2. ONNX -> TRT engine (~5 min, single-core CPU bound). ACTION_HORIZON=10 for pi05_libero!
$DR bash -c 'ACTION_HORIZON=10 \
    ONNX_PATH=/workspace/checkpoints/pi05_libero_pytorch/onnx/model_fp8_nvfp4.onnx \
    bash phase2/openpi_on_thor/build_engine.sh'

# 3. Benchmark TRT vs PyTorch (synthetic LIBERO example + cosine similarity)
$DR python phase2/openpi_on_thor/pi05_inference.py --inference-mode compare \
    --config-name pi05_libero --checkpoint-dir /workspace/checkpoints/pi05_libero_pytorch \
    --engine-path /workspace/checkpoints/pi05_libero_pytorch/onnx/model_fp8_nvfp4.engine \
    --num-warmup 5 --num-test-runs 30
```

**Result (2026-06-01):** TRT FP8+NVFP4 **94.7 ms / ~10.6 Hz**, **2.12× over BF16**,
**cosine 0.997** vs PyTorch. See `findings.md`. Notes: pure FP16 unsupported (Gemma
dynamic range); dummy calibration (HF offline) was sufficient for fidelity; for a faster
re-build add `--builderOptimizationLevel=2` + a `--timingCacheFile` to build_engine.sh.
