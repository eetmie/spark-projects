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

### [`vla-onnx/`](vla-onnx/) — LeRobot VLA → split ONNX → Orin Nano
One pipeline, three models: **SmolVLA 450 M**, **X-VLA 0.9 B**, **EVO1 775 M**. Fine-tune on
GB10, cut into split ONNX graphs the 8 GB Orin can actually build engines for. Verify against
PyTorch baseline.

Two environments on purpose — lerobot 0.5.1/torch 2.12 for the SmolVLA↔X-VLA comparison,
0.6.1/torch 2.11 for EVO1 and the X-VLA export. They are mutually exclusive pins.
See [`vla-onnx/README.md`](vla-onnx/README.md).

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
