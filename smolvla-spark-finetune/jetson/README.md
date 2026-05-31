# SmolVLA on Jetson Orin Nano (deploy half)

Downstream of the GB10 fine-tune/export ([`../`](../)). Takes the ONNX exported on the
Spark and runs it on **Jetson Orin Nano** via TensorRT.

Scope here: ONNX → TensorRT engine build + TRT-vs-ONNX parity on the Jetson.
(Robot control stack is kept in a separate repo.)

## Workflow
1. Copy the parity-checked ONNX from GB10 (`../exports/…`) to the Jetson.
2. Build the TensorRT engine on the Jetson (`trtexec`).
3. Run TRT-vs-ONNX parity to confirm the engine matches.

See [`notes/findings.md`](notes/findings.md).
