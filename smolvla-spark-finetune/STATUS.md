# Status

Experimental but locally verified on NVIDIA GB10.

## Verified

- PyTorch CUDA works on NVIDIA GB10 with `torch==2.12.0+cu130`.
- LeRobot `0.5.1` imports and runs SmolVLA.
- SmolVLA CUDA forward works on GB10.
- Official `lerobot/svla_so101_pickplace` dataset was downloaded as a smoke-test fixture only.
- AV1 videos were transcoded to a local H.264 working copy for reliable LeRobot loading.
- A 1-step SmolVLA LoRA fine-tune smoke test completed on GB10 using the fixture dataset.
- Baseline SmolVLA ONNX export completed.
- ONNX checker passed.
- CPU ONNX Runtime session creation passed.
- PyTorch-vs-ONNX parity passed with `max_abs_diff ~= 2.62e-6` and cosine effectively `1.0`.

## Not Yet Verified

- Exporting a fine-tuned LoRA checkpoint after a real training run.
- Merging LoRA weights into a self-contained checkpoint for deployment export.
- TensorRT engine build on Jetson Orin Nano.
- TensorRT-vs-ONNX parity on Jetson.
- Real robot preprocessing/postprocessing loop.
- Useful task-specific behavior from the fixture dataset.
- Real SO-101 / SO-100 hardware inference.

## Current Intended Workflow

1. Fine-tune SmolVLA on GB10 using LeRobot.
2. Export ONNX on GB10.
3. Run PyTorch-vs-ONNX parity on GB10.
4. Copy ONNX and runtime code to Jetson Orin Nano.
5. Build TensorRT engine on the Jetson.
6. Run TensorRT parity and then robot integration.
