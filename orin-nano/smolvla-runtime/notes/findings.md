# Findings — SmolVLA runtime on Jetson Orin Nano

Running log. Newest first. (Merges the earlier `smolvla-spark-finetune/jetson/notes`.)

## Decision: pure TensorRT engine as the primary path

Chose the serialized `.engine` + TensorRT 10.x API + `cuda-python` over the ORT TensorRT-EP path
for latency. ORT-TRT-EP is retained as a **diagnostic/fallback** backend because pure TRT has one
bad failure mode (below).

### Catches accepted, with mitigations

1. **Unsupported op = hard build abort, no detail.** The monolithic graph bundles the VLM (vision
   encoder + language model) and the flow-matching denoising loop; iterative loops can export as
   `Loop`/`Scan`/`If`, plus boolean-mask `Where`/`Gather`. Pure `trtexec` aborts on any op it can't
   build. → *Mitigation:* run `run_pipeline.py --backend ort` to see which subgraph ORT falls back
   to CUDA on; fix the export (static shapes, small fixed `num_steps`) or add a plugin.
2. **Builder OOM on 8 GB shared RAM.** Building from a ~1.5 GB FP32 ONNX needs weights + workspace +
   tactic memory at once. → FP16 build, `--workspace-mib 2048`, `--opt-level 2`, build headless,
   add zram/swap.
3. **FP16 overflow** (NaN in softmax/layernorm). → parity-check vs ONNX; pin sensitive layers with
   `--layer-precisions '*softmax*:fp32,*norm*:fp32'`.
4. **Engine is non-portable.** Locked to sm_87 + TRT 10.3 + CUDA 12.6. Build on THIS board; a
   JetPack/TRT upgrade invalidates it → rebuild. ONNX is the portable source of truth.
5. **Static everything.** batch=1, image size, lang length, `num_steps` all baked in — change one →
   re-export on the Spark.
6. **Pre/post stays in Python** (tokenize, resize/normalize, pad→robot map). Cheap; not accelerated.

## Stack confirmed installed (live, 2026-06-01)

TensorRT is purely userspace — no kernel changes beyond the RealSense work (see `../realsense-rt/`).

- TensorRT **10.3.0.30** + CUDA 12.5 target (`libnvinfer10`, `python3-libnvinfer`, `libnvinfer-bin`)
- CUDA 12.6 toolkit, `nvcc` present
- `trtexec` at `/usr/src/tensorrt/bin/trtexec`
- L4T R36.4.4 (JetPack 6), kernel `5.15.148-rt-tegra`, Python 3.10.12
- 8 GB shared RAM (~5 GB free), 79 GB disk free
- `import tensorrt` → 10.3.0; `pyrealsense2` imports from system; D435i enumerates over kernel UVC
- `onnxruntime-gpu` NOT in system Python — install in venv from the Jetson AI Lab index (only place
  with an aarch64 GPU wheel): `onnxruntime-gpu==1.24.0 --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126`

## Pipeline plumbing — verified

`run_pipeline.py` runs end to end with the **real D435i** (mock backend): threaded latest-frame
reader delivers fresh frames (`img_age_ms` 1–30 ms at 640×480@30). Synthetic source works with no
camera. Remaining gap: the SmolVLA engine itself — needs the ONNX from the Spark.

## Which ONNX to deploy

Produced on the Spark (`../../smolvla-spark-finetune/`). Two things matter here:

- Bake **`num_steps`** small (4 = fast prototype, 10 = better quality) and **static shapes**.
- Transfer `model.onnx` **and** `model.onnx.data` if present (large ONNX splits weights into a
  sidecar) — both files in the same dir.
- ONNX interface seen from the Spark export (`export_valid_onnx.py`): inputs `image0[ B,3,512,512]`,
  `img_mask0[B]`, `lang_tokens[B,48]`, `lang_masks[B,48]`, `state[B,32]`, `noise[B,50,32]`; output
  `actions[B,50,32]`. The older bench export used `image`/`image_mask`. `io_spec.py` resolves either
  by name + dtype/rank so the runtime doesn't care which.
- Output dims are **padded to 32**; the SO-101 smoke task is 6D. Real deployments must map the
  relevant output dims to the target robot action space explicitly.

## Expected performance (Orin Nano 8 GB)

| Path | Estimated latency |
|---|---|
| ORT CUDA EP (TRT not used) | ~108 ms |
| ORT TRT EP FP16 | ~40–80 ms |
| **Pure TRT engine FP16 (target)** | **≤ pure-TRT ORT, lower overhead** |

Target: p95 < 100 ms at 10 denoising steps → 10 Hz loop. Measure before wiring anything.

## Next steps in order

1. Export ONNX on the Spark with small `num_steps` + static shapes.
2. Copy `model.onnx` (+ `.data`) into `smolvla-runtime/exports/`.
3. `pip`-set up the venv (`--system-site-packages`), confirm `import tensorrt, cuda.cudart`.
4. `build_engine.py` → `.engine` (be patient / watch for OOM).
5. Parity-check engine vs ONNX (TODO: add `parity.py`; reuse the Spark `parity_check_onnx.py` shape).
6. `run_pipeline.py --backend trt --source synthetic` → first run, sanity.
7. `--source realsense` → real benchmark. Record p95.
8. If pure build fails on an op: `--backend ort` to locate it.

## Parity (TRT vs ONNX)

_(none yet — run after the first engine build)_
