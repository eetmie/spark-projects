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

### [`evo1-spark-finetune/`](evo1-spark-finetune/) — EVO1 Spark export + Orin bootstrap
Pinned LeRobot 0.6.1 / InternVL3-1B-hf setup, 11-graph split exporter, native fixture,
and mixed-FP16 conversion. The non-deployable random-head bootstrap builds as ten TRT
engines plus a CPU embedding graph on Orin Nano Super: final-action cosine 0.999991,
~0.56 s per 32-step chunk, and 4.75 GB peak RSS.

### [`xvla-spark-finetune/`](xvla-spark-finetune/) — X-VLA-0.9B fine-tune + split export
Full fine-tune of X-VLA on the excavator data, plus the budget-driven split exporter,
parity guard and bundle contract. Full-finetune beats a frozen-encoder run by ~14 %.
The 8 GB fit measurements that motivated the split are in `notes/fit-on-8gb.md`.

### [`scene-reconstruction/`](scene-reconstruction/) — video → Gaussian splat → Isaac Sim
(basically deprecated since Spirula Studio exists. I only use the usd_export from here!)
Smartphone video -> COLMAP -> 3DGRUT raw Gaussian splat -> SuperSplat cleanup/compression -> Isaac Sim NuRec USDZ on DGX Spark. See `scene-reconstruction/README.md`.

## Where the deploy side lives

This repo is the **Spark side**: fine-tune, export, and validate the export. It stops at a
parity-checked ONNX bundle.

| repo | owns |
|---|---|
| **spark-projects** (here) | fine-tuning, ONNX export, export parity, bundles |
| [**jetson-orin-nano-vla**](https://github.com/eetmie/jetson-orin-nano-vla) | does a **base** model fit on the 8 GB Orin, and what it costs — board prep, TRT build probes, the benchmark |
| **kaivuriprokkis** | the robot — fine-tuned runtime, camera, control |

TensorRT engines are built on-device and never copied; the ONNX bundle is the portable
artifact. Bundles ship with `ship_bundle.sh <dir> orin <name>` and land in `~/bundles/`.

## Notes
- `pi05-spark-inference/phase2/openpi_on_thor/` contains scripts adapted from NVIDIA /
  Jetson AI Lab's "OpenPi π0.5 on Jetson Thor" tutorial, kept here for reproducibility.
- Each playbook documents the exact aarch64/Blackwell dependency gotchas it hit — that's
  the main value, since stock install instructions rarely work cleanly on GB10.
