# SmolVLA runtime — Jetson Orin Nano (pure TensorRT)

The model pipeline, end to end, on the Orin Nano:

```
RealSense D435i RGB  ->  SmolVLA (pure TensorRT engine)  ->  action chunk
```

**No robot control here.** This stage proves the *model* runs fast enough on the Nano: one camera
feed in, action chunks out. Mapping actions onto a real robot (normalization, limits, watchdogs,
safety) is deliberately a separate, later stage.

Upstream of this is [`../../smolvla-spark-finetune/`](../../smolvla-spark-finetune/) — fine-tune +
ONNX export happen on the **DGX Spark**. This box only ever sees the ONNX, and builds a TensorRT
engine from it locally.

## Why pure TensorRT

You asked for the native tensor runtime for performance, so the primary path runs the serialized
`.engine` directly via the TensorRT 10.x API + `cuda-python` device buffers — no ONNX Runtime in
the hot loop. The trade-off is that a pure-TRT build is all-or-nothing: an unsupported op aborts the
build with no detail. So an **ORT TensorRT-EP backend is kept as a diagnostic** — it partitions the
graph and falls back to CUDA for unsupported subgraphs, telling you exactly what to fix. See
[`notes/findings.md`](notes/findings.md) for the full catch list.

## Layout

```
smolvla-runtime/
  build_engine.py            ONNX -> .engine via trtexec (FP16, workspace cap, layer-precision pins)
  run_pipeline.py            the runner: source -> backend -> action chunk + latency report
  requirements.txt
  smolvla_runtime/
    camera.py                threaded D435i latest-frame reader + a synthetic source
    preprocess.py            image resize/normalize, tokenize, state pad, noise (the only model glue)
    io_spec.py               maps logical roles -> engine tensor names (survives image vs image0)
    backends/
      trt_engine.py          PRIMARY: pure TensorRT runtime
      ort_trt.py             diagnostic/fallback: ONNX Runtime + TensorRT EP
      mock.py                zero actions, no model (plumbing check)
  notes/findings.md
```

## Setup (JetPack 6, Python 3.10, aarch64)

`tensorrt` and `pyrealsense2` come from the system (JetPack + the librealsense source build), so the
venv must see them:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import tensorrt, pyrealsense2; from cuda.bindings import runtime; print('runtime deps ok')"
```

The RealSense kernel modules must be loaded first — see [`../realsense-rt/`](../realsense-rt/).

## Run

```bash
# 0. Plumbing only (no model, no camera hardware beyond the source):
python run_pipeline.py --backend mock --source synthetic --duration-s 5

# 0b. Real D435i feed, still no model — confirms the camera path:
python run_pipeline.py --backend mock --source realsense --duration-s 8

# 1. Build the engine from the Spark-exported ONNX (slow, one-time, ~20-60 min).
#    Use BF16 — FP16 is broken for SmolVLA (cosine 0.805, vision-tower overflow);
#    BF16 is near-lossless (0.9974). --static-batch supplies the required profile.
python build_engine.py --onnx exports/smolvla.onnx --engine exports/smolvla.engine \
    --precision bf16 --static-batch

# 2. Parity-check the engine vs the FP32 ONNX BEFORE trusting any action.
#    Reference runs FP32 on CPU (true FP32); engine and reference run sequentially
#    so peak RAM stays inside 8 GB. BF16 should pass at ~0.997; FP16 fails at ~0.805.
python parity.py --onnx exports/smolvla.onnx --engine exports/smolvla.engine \
    --model-id lerobot/smolvla_base --num-samples 3

# 3. Pure TRT engine on synthetic frames (validate engine before the camera):
python run_pipeline.py --backend trt --engine-path exports/smolvla.engine \
    --model-id lerobot/smolvla_base --source synthetic --duration-s 20 --show-actions

# 4. Pure TRT engine + real D435i — the actual pipeline:
python run_pipeline.py --backend trt --engine-path exports/smolvla.engine \
    --model-id lerobot/smolvla_base --source realsense --duration-s 30 --show-actions

# If the engine build aborts on an unsupported op, find it with the ORT diagnostic:
python run_pipeline.py --backend ort --onnx-path exports/smolvla.onnx \
    --model-id lerobot/smolvla_base --source synthetic --duration-s 20 --log-level INFO
```

## Reading the output

| field | meaning |
|---|---|
| `infer_ms(avg/p95)` | inference latency only (H2D + execute + D2H + sync) |
| `loop_ms(avg/p95)` | full camera→action loop time |
| `img_age_ms` | how stale the frame was when inference started |
| `action_shape` | raw SmolVLA output, e.g. `(50, 32)` — padded; map to the robot's dims later |

**Go / no-go** (10-step denoising, targeting a 10 Hz loop): `p95 < 100 ms` is viable; `100–200 ms`
is supervisory-only; `>200 ms` or OOM → fewer denoising steps, smaller image, or a distilled policy.

## Current status

- Venv: **set up** (`.venv`, `--system-site-packages`). `tensorrt` 10.3, `pyrealsense2`,
  `cuda-python` 12.9 (`cuda.bindings.runtime` → `cudaGetDeviceCount` sees the GPU), `transformers`,
  `pillow` all import. The pure-TRT backend imports clean and fails only at "no engine yet".
- Plumbing + real D435i capture path: **verified live** (mock backend, ~30 Hz fresh frames).
- Pure-TRT + ORT backends: **written, not yet exercised** — they need the SmolVLA ONNX from the
  Spark, which is not on this box yet. The moment `exports/*.onnx` lands, steps 1–3 above run it.
