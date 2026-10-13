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

### [`vla-onnx/`](vla-onnx/) — LeRobot VLA → split ONNX → Orin Nano
Old project, go see jetson-orin-nano-vla


### [`scene-reconstruction/`](scene-reconstruction/) — video → Gaussian splat → Isaac Sim
Old project, just use Spirula studio (not sponsored). Just add vkQueueWaitIdle() after the normal timeline wait in Stream.cpp (spirula/spirula-studio/src/nn/vk/Stream.cpp:313). Enabled by SS_NN_QUEUE_IDLE_SYNC=1

Otherwise depth generation freezes.


## Notes
- `pi05-spark-inference/phase2/openpi_on_thor/` contains scripts adapted from NVIDIA /
  Jetson AI Lab's "OpenPi π0.5 on Jetson Thor" tutorial, kept here for reproducibility.
