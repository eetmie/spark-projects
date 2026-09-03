# EVO1 Orin Nano Super performance sweep

Measured 2026-09-03 on `joel@192.168.0.94` with the nondeployable deterministic EVO1
bootstrap bundle. Every row uses the same mixed-FP16 TensorRT graphs and fixed fixture.
Each benchmark has 3 warmups and 20 timed chunks. Raw reports are preserved in
[`perf/`](perf/).

These timings validate execution mechanics only. The bootstrap action head is random;
similarity to its native 32-step output is not evidence of task success.

## Exact 32-step optimizations

| variant | mean ms | p95 ms | Hz | peak RSS GB | action cosine | change vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline: CPU embedding, CUDA fallback | 390.61 | 392.92 | 2.56 | 4.737 | 0.999991 | — |
| remove CUDA fallback | 389.33 | 391.75 | 2.57 | 4.723 | 0.999991 | 0.3% faster |
| CUDA embedding, no fallback | 388.96 | 390.88 | 2.57 | 5.199 | 0.999991 | 0.4% faster, +0.462 GB |
| fused action graph, no fallback | 380.65 | 383.11 | 2.63 | 4.937 | 0.999991 | 2.6% faster, +0.200 GB |
| **device-resident split action, no fallback** | **294.85** | **297.68** | **3.39** | **4.770** | **0.999991** | **24.5% faster, +0.033 GB** |
| fused + device-resident, no fallback | 287.12 | 291.57 | 3.48 | 4.941 | 0.999991 | 26.5% faster, +0.204 GB |

The recommended profile keeps the existing split action engines, the 544 MB token table
on CPU, and uses I/O binding to upload the action K/V cache once. It then keeps the
`action_step` hidden output on CUDA for `action_output`, returning only each velocity to
the host for the FP32 Euler update. Its action output is bit-for-bit the same array as the
ordinary split path in this run: cosine `0.999990951569` and max absolute error
`0.00747218` versus native FP32.

The fused graph is retained as an optional experiment. Fusion adds only another 2.6%
over the recommended device-resident split profile while increasing peak RSS by about
171 MB relative to it. Moving token embedding to CUDA saves roughly 3 ms in that stage
but loses about 0.48 GB system headroom, so it is rejected.

Run the recommended benchmark after the normal engine prebuild:

```bash
TRT_DROP_CUDA_EP=1 python benchmark.py \
  --bundle bundle \
  --cache trt_cache \
  --precision fp16 \
  --embedding-device cpu \
  --device-resident-action \
  --steps 32 \
  --output results/device-action-no-cuda-cpu-embed-32.json
```

## Solver-step sweep

Changing solver steps changes the policy algorithm, so this is not a free optimization.
The correct value must come from the trained checkpoint recipe and then pass real task
evaluation.

| steps | ordinary split mean ms / Hz | device-resident mean ms / Hz | action cosine vs native 32-step |
|---:|---:|---:|---:|
| 10 | 226.83 / 4.41 | 196.20 / 5.10 | 0.999980 |
| 16 | 268.76 / 3.72 | not measured | 0.999992 |
| 32 | 389.33 / 2.57 | 294.85 / 3.39 | 0.999991 |
| 50 | 525.55 / 1.90 | 371.74 / 2.69 | 0.999988 |

The released RoboTwin checkpoint requires 50 solver steps; its relevant infrastructure
estimate is therefore 371.74 ms (2.69 Hz) with device-resident action execution. The
published SO100 configuration uses 10 steps, whose infrastructure estimate is 196.20 ms
(5.10 Hz). Those estimates still exclude live camera capture, preprocessing, action
smoothing, transport, and robot I/O.

The device-resident optimization becomes more valuable as the loop grows: it saves
24.3% at 32 steps and 29.3% at 50 steps. At 50 steps, its measured mean stage times are
111.51 ms vision, 27.32 ms language plus CPU embedding, 6.30 ms action context, 4.19 ms
one-time cache upload, and 220.25 ms for the action loop.
