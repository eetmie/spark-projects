# Moved → `../../orin-nano/smolvla-runtime/`

The Jetson Orin Nano deploy half of this playbook now lives at the repo top level, alongside the
RealSense/RT-kernel setup it depends on:

- **[`../../orin-nano/smolvla-runtime/`](../../orin-nano/smolvla-runtime/)** — the camera→model→
  actions pipeline via ONNX Runtime + the TensorRT execution provider (FP16, engine auto-cached).
- **[`../../orin-nano/realsense-rgb/`](../../orin-nano/realsense-rgb/)** — RealSense D435i RGB
  (librealsense RSUSB build, no kernel patches) for the real-camera path.

This `smolvla-spark-finetune/` playbook keeps the **Spark side**: fine-tune SmolVLA and export a
parity-checked ONNX (+ tokenizer + normalization bundle). The boundary is unchanged — export on the
Spark; the Orin runs it through ORT's TensorRT EP (no separate trtexec engine build).
