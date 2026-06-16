# orin-nano

Jetson **Orin Nano Super** (8 GB, **JetPack 7.2** / L4T R39.2.0, Ubuntu 24.04, Python 3.12,
CUDA 13.2, TensorRT 10.16) — the *deploy* side of the Spark playbooks. Where the DGX Spark trains
and exports, this board runs the model and (eventually) the robot.

> Top-level on purpose: general Orin Nano infrastructure, not tied to one model.

## Subfolders

### [`system/`](system/) — performance + memory setup
MAXN_SUPER power mode + pinned clocks (one-shot script and a persistent systemd unit), and a 16 GB
swapfile so the TensorRT engine build fits in the 8 GB unified memory. Also the note on getting
`onnxruntime-gpu` for CUDA 13.

### [`realsense-rgb/`](realsense-rgb/) — D435i RGB, no kernel patches
librealsense built with the userspace USB backend (`FORCE_RSUSB_BACKEND=ON`) for the D435i **color**
stream only. Runs on the stock JetPack 7.2 kernel — no PREEMPT_RT, no UVC metadata patches, no
HID-sensor modules. (Depth + onboard IMU + an RT control loop are deferred; they're what needed the
kernel work before.)

### [`smolvla-runtime/`](smolvla-runtime/) — SmolVLA model pipeline (ORT + TensorRT EP)
`RealSense RGB → SmolVLA (ONNX Runtime + TensorRT EP) → action chunk`. Runs the SmolVLA export from
the Spark ([`../smolvla-spark-finetune/`](../smolvla-spark-finetune/)) through ORT's TensorRT EP
(FP16, engine-cached). No robot control yet — model pipeline only. **Note:** the *monolithic* ONNX
won't TRT-build on 8 GB (all 450M weights at once); the deploy path is **per-component split engines**
(vision/text/prefill/decode + Python denoise loop) — validated on-device, see `smolvla-runtime/notes/findings.md`.

## The boundary

```
DGX Spark (GB10)                    Orin Nano (this board)
────────────────                    ──────────────────────
fine-tune SmolVLA (LeRobot)         ORT/TensorRT-EP builds + caches the engine from ONNX
export + parity-check FP32 ONNX ──> run inference (camera → actions), FP16
                                    (later) map actions → kaivuriprokkis control
```

The cached TensorRT engine is hardware/version specific — it is **built here, never copied from the
Spark**. The FP32 ONNX is the portable artifact; the engine cache is local.

## Bring-up order

1. `system/` — `./power-max.sh`, `sudo ./setup-swap.sh`, install `jetson-perf.service`.
2. `realsense-rgb/` — `./install-realsense-rgb.sh`, plug in the D435i, verify.
3. `smolvla-runtime/` — venv (`--system-site-packages`) + `onnxruntime-gpu`, drop in the Spark ONNX,
   run.
