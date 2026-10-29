# spark-projects

Personal playbooks for ML/robotics experiments on an **NVIDIA DGX Spark (GB10, Grace-Blackwell, aarch64)**.
Each subdirectory is a self-contained playbook: code, Dockerfiles, scripts, and notes.
Large artifacts (checkpoints, datasets, ONNX/engines, venvs) are **gitignored** — these
repos hold the *recipe*, not the data.

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

### [`vla-onnx/`](vla-onnx/) — excavator VLAs: datasets → fine-tune → robot bundles
Active. Native recordings → LeRobot datasets (`common/vla_common/dataset`), GR00T N1.7 and
pi0.5 (OpenPI) fine-tunes with training-time RTC, bundles for kaivuriprokkis on the Orin.
Start at [`vla-onnx/kaivuriprokkis/RTC_TRAINING.md`](vla-onnx/kaivuriprokkis/RTC_TRAINING.md).
`xvla/` is the LeRobot path kept for the Python-side backup stack. SmolVLA and EVO1 were
removed (tag `archive/lerobot-playbooks-2026-10-29`); Orin benchmarks live in
[jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla).


### [`scene-reconstruction/`](scene-reconstruction/) — Gaussian PLY → USDZ for Isaac Sim
Just use Spirula Studio (not sponsored haha). Only the PLY → USD converter is left.


## Notes
- `pi05-spark-inference/phase2/openpi_on_thor/` contains scripts adapted from NVIDIA /
  Jetson AI Lab's "OpenPi π0.5 on Jetson Thor" tutorial, kept here for reproducibility.
