# spark-projects

Personal playbooks for ML/robotics experiments on an **NVIDIA DGX Spark (GB10, Grace-Blackwell, aarch64)**.
Each subdirectory is a self-contained playbook: code, Dockerfiles, scripts, and notes.
Large artifacts (checkpoints, datasets, ONNX/engines, venvs) are **gitignored** — these
repos hold the *recipe*, not the data.

> Environment across all playbooks: NVIDIA GB10, Ubuntu 24.04, aarch64, CUDA 13, driver 580.

## Playbooks

### [`pi05-spark-inference/`](pi05-spark-inference/) — π0.5 VLA inference on GB10
openpi π0.5 (Physical Intelligence) Vision-Language-Action inference, BF16 baseline → TensorRT FP8+NVFP4.

| backend | latency | rate | fidelity |
|---|---|---|---|
| PyTorch BF16 | ~203 ms | ~4.9 Hz | — |
| **TensorRT FP8+NVFP4** | **~94.7 ms** | **~10.6 Hz** | **cosine 0.997 vs PyTorch** |

**2.12× speedup**, essentially lossless — matches the Jetson Thor reference (~94 ms).
Full pipeline: JAX→PyTorch convert → ONNX (ModelOpt) → trtexec engine → benchmark.
See `pi05-spark-inference/notes/findings.md` and `RUNBOOK.md`.

### [`smolvla-spark-finetune/`](smolvla-spark-finetune/) — SmolVLA fine-tune + ONNX export on GB10
Fine-tune SmolVLA with LeRobot on GB10 and export a valid ONNX (parity-checked). Actual
inference/TensorRT runs downstream on Jetson Orin Nano, not on the Spark. Verified on GB10:
SmolVLA CUDA forward + 1-step LoRA smoke test + ONNX export with PyTorch-vs-ONNX parity
(max_abs_diff ~2.6e-6, cosine ~1.0). See `smolvla-spark-finetune/STATUS.md`.

### [`scene-reconstruction/`](scene-reconstruction/) — video → Gaussian splat → Isaac Sim
Smartphone/wide-angle video → COLMAP → 3DGRUT Gaussian splat → Isaac Sim NuRec USDZ on
DGX Spark. 3DGRT ray-tracing backend (uses GB10 RT cores). See `scene-reconstruction/README.md`.

### [`orin-nano/`](orin-nano/) — Jetson Orin Nano deploy side (RealSense + SmolVLA TensorRT)
The deploy counterpart to the Spark playbooks, on a Jetson **Orin Nano** (JetPack 6, PREEMPT_RT
kernel). Two parts: `realsense-rt/` builds the RealSense D435i kernel modules against the RT kernel
(reproducible, live-verified), and `smolvla-runtime/` runs the SmolVLA ONNX exported by
`smolvla-spark-finetune/` through a **pure TensorRT engine** (camera → model → action chunk; no
robot control yet). Engines are built on-device — never copied from the Spark.

## Notes
- `pi05-spark-inference/phase2/openpi_on_thor/` contains scripts adapted from NVIDIA /
  Jetson AI Lab's "OpenPi π0.5 on Jetson Thor" tutorial, kept here for reproducibility.
- Each playbook documents the exact aarch64/Blackwell dependency gotchas it hit — that's
  the main value, since stock install instructions rarely work cleanly on GB10.
