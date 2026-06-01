# orin-nano

Jetson **Orin Nano** (8 GB, L4T R36.4.4 / JetPack 6, kernel `5.15.148-rt-tegra` PREEMPT_RT) —
the *deploy* side of the Spark playbooks. Where the DGX Spark trains and exports, this board runs
the model and (eventually) the robot.

> Top-level on purpose: this is general Orin Nano infrastructure, not tied to one model. The
> RealSense + RT-kernel setup underpins anything camera-driven on this board.

## Subfolders

### [`realsense-rt/`](realsense-rt/) — RealSense D435i on the RT kernel
Reproducible build of the six RealSense kernel modules against the PREEMPT_RT kernel + a CUDA
librealsense, so the D435i runs on the kernel-UVC backend (not the userspace RSUSB fallback) while
keeping PREEMPT_RT for the 100 Hz control loop. **Done and live-verified**; this is the recipe.

### [`smolvla-runtime/`](smolvla-runtime/) — SmolVLA model pipeline (pure TensorRT)
`RealSense RGB → SmolVLA (pure TensorRT engine) → action chunk`. The compatibility layer that runs
the SmolVLA ONNX exported on the Spark ([`../smolvla-spark-finetune/`](../smolvla-spark-finetune/))
through a native TensorRT engine. No robot control yet — model pipeline only. Plumbing + real
camera path verified; engine path awaits the Spark ONNX.

## The boundary

```
DGX Spark (GB10)                    Orin Nano (this board)
────────────────                    ──────────────────────
fine-tune SmolVLA (LeRobot)         build TensorRT engine from ONNX
export + parity-check ONNX     ──>  run pure-TRT inference (camera → actions)
                                    (later) map actions → robot control
```

A TensorRT engine is hardware/version specific — it is **built here, never copied from the Spark**.
The ONNX is the portable artifact; the engine is a local cache.
