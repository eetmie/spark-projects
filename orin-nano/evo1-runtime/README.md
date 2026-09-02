# EVO1 initial Orin Nano runtime

This is the first hardware-validation stage for EVO1, following the same split-ONNX,
one-engine-per-subprocess, native-reference-parity workflow used for SmolVLA and X-VLA.
It targets one RGB camera, a fixed 320-token prompt, a 50x24 action chunk, and EVO1's
32-step Euler flow loop.

**Safety boundary:** the current bundle combines the exact InternVL3 base weights at
revision `014c0583a0d4bedf29fbe2dbff4f865eb998e171` with a deterministic but randomly
initialized EVO1 action head. It is marked `deployable: false`; these outputs validate
the export/runtime infrastructure and must never control a robot.

The eleven-graph layout keeps the 544 MB token table FP32 on CPU and builds ten mixed-FP16
TensorRT engines:

- vision: four chunks (7 + 7 + 7 + 3 InternVL blocks and projector)
- language: three chunks (6 + 6 + 2 Qwen2 blocks)
- action context: state projection plus the eight blocks' cached K/V tensors
- action hot path: one eight-block action step and one output head, each invoked 32 times

On the Jetson, place the verified bundle at `bundle/`, then run:

```bash
# Uses the Jetson ORT wheel with TensorRT/CUDA providers.
python prebuild.py --bundle bundle --cache trt_cache --precision fp16

# Compares every stage with a native LeRobot 0.6.1 FP32-CUDA fixture emitted on Spark.
python run_fixture.py --bundle bundle --cache trt_cache --precision fp16
```

Engine builds are deliberately isolated. Each graph gets its own child process so the
TensorRT builder returns its unified memory before the next graph starts. ORT graph
fusions are disabled, matching the X-VLA workaround for FP16 `com.microsoft.Gelu`
fallback failures. The token-embedding graph always uses CPU EP; every other graph uses
TensorRT, CUDA fallback, then CPU fallback during the parity run.

This directory has no live camera or actuator entry point. Adding those belongs after a
trained LeRobot EVO1 policy checkpoint replaces the bootstrap action head and passes the
same Spark-to-Orin parity gate.

## Initial Orin Nano Super result (2026-09-02)

All ten TensorRT engines build and execute on the 8 GB board. The largest build-time
unified-memory dip left 2.995 GB available; the completed cache is 1.3 GB, and engine
construction consumed only 1.8 MiB of swap. With every engine plus the CPU token table
resident, the process peaks at 4.759 GB RSS and leaves 2.5 GB system memory available.

One deterministic chunk takes 553.6 ms (1.81 Hz): 167.6 ms vision, 74.9 ms language plus
CPU token lookup, 16.7 ms action context, and 294.3 ms for the 32 action-step/output
pairs. Parity against native LeRobot 0.6.1 FP32 CUDA is:

| boundary | cosine | max abs |
|---|---:|---:|
| vision | 0.999606 | 1.0863 |
| valid fused tokens | 0.999546 | 18.8793 |
| final action | 0.999991 | 0.00747 |

The parity gate excludes the 51 padded language query positions because
`action_context` masks them as keys; their full-tensor cosine is still emitted as a
diagnostic (`0.998448`). The final action passes the 0.999 threshold by a wide margin.
