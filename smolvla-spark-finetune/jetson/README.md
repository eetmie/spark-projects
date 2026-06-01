# Moved → `../../orin-nano/smolvla-runtime/`

The Jetson Orin Nano deploy half of this playbook now lives at the repo top level, alongside the
RealSense/RT-kernel setup it depends on:

- **[`../../orin-nano/smolvla-runtime/`](../../orin-nano/smolvla-runtime/)** — ONNX → TensorRT
  engine + the camera→model→actions pipeline (pure TensorRT).
- **[`../../orin-nano/realsense-rt/`](../../orin-nano/realsense-rt/)** — RealSense D435i on the RT
  kernel (a prerequisite for the real-camera path).

This `smolvla-spark-finetune/` playbook keeps the **Spark side**: fine-tune SmolVLA and export a
parity-checked ONNX. The boundary is unchanged — export ONNX on the Spark, build the TensorRT engine
on the Orin Nano.
