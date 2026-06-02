# Findings — SmolVLA runtime on Jetson Orin Nano

Running log. Newest first. (Merges the earlier `smolvla-spark-finetune/jetson/notes`.)

## Static ONNX (no NonZero) + num_steps variants — validated on Spark (2026-06-02)

Re-exported with the `torch.where` fix (see `smolvla-spark-finetune/export_valid_onnx.py`) to kill
the data-dependent `NonZero` (device→host sync stall + DDS fragility on Orin TRT 10.3). Added a
`--num-steps` flag to bake fewer denoise steps. **`num_steps` is the flow-matching ODE step count
(`dt=-1/num_steps; x_t += dt·v_t`), NOT the action chunk (50) and NOT the control dt** — it's
unrolled into the graph, so changing it needs a re-export. Validated on the Spark (BF16, opt-0,
warmed):

| ONNX (in spark-finetune/exports) | num_steps | nodes | NonZero | BF16 cosine | infer median (Blackwell) |
|----------------------------------|-----------|-------|---------|-------------|--------------------------|
| `smolvla_base_fp32_static.onnx`     | 10 | 108,695 | 0 | 0.9974 | 94.7 ms |
| `smolvla_base_fp32_static_s5.onnx`  |  5 |  60,480 | 0 | 0.9985 | 63.7 ms |

- `torch.where` rewrite is **bit-identical** to the original (cosine 1.0, max_abs 0 vs `*_valid.onnx`).
- num_steps 10→5: **~33% faster** inference, ~half the graph (build 248→112 s), fidelity-vs-own-FP32
  even slightly better. Cost is coarser denoising — a *task-quality* question to judge on the robot,
  not a numerics one. (3-step would be the next probe if 5 isn't enough.)
- These are Blackwell/opt-0 absolutes; Orin will be slower. The ~33% step speedup should roughly
  transfer; measure on-device.
- **Move to the Orin:** `*_static.onnx` for the apples-to-apples vs the PyTorch demo (both 10-step);
  `*_static_s5.onnx` for reactive real runs. Both have `.sha256` receipts.

## "10 Hz" is the control rate, NOT inference rate — reframes the target (2026-06-02)

Reference: `~/Desktop/isaacsim_vla_ws-robot-so101_new_calib` (github.com/MyLovelyAxe/isaacsim_vla_ws)
— a ROS2 + Isaac Sim SO-101 sim2real demo running SmolVLA in **PyTorch inside a container**
(`smolvla_pytorch27_container`) on an Orin Nano 8 GB. It is the natural baseline to benchmark our
TRT path against.

Its "10 Hz" is **not** SmolVLA inference speed. Confirmed in the code: `send_observation*.py`
throttles observations to 10 Hz, `safety_rules.py` records at `DT=0.1` (10 Hz), joint states stream
at 200 Hz. SmolVLA emits a **50-action chunk per observation, executed open-loop**, and a learned
**safety estimator** decides when to pull a fresh chunk. So one inference covers up to ~50 control
steps (~5 s at 10 Hz) — a 200–500 ms inference is hidden behind chunking.

Consequences for our work:
- The old "p95 < 100 ms = go" bar was wrong. Plain PyTorch already gives 10 Hz *control* via
  chunking. We are not unlocking a rate PyTorch couldn't reach.
- The real value of pure-TRT BF16 is **lower inference latency → fresher re-planning/reactivity
  (safety estimator can request chunks sooner) + lower memory & power**, not a Hz threshold.
- **Tomorrow's decisive test:** head-to-head *inference latency* (and peak RAM) of the PyTorch
  container vs our TRT-BF16 engine on the *same* Orin. That is the concrete "is native worth it"
  number. Our ONNX bakes num_steps=10 — matching the demo's default, so it's a fair comparison.

## USE BF16, NOT FP16 — precision sweep on the Spark (2026-06-02)

De-risked the whole build on the DGX Spark (GB10/Blackwell, TRT 10.13) before touching the
Nano — same TRT 10.x family, so build-time op support and FP16/BF16 numerics transfer; only the
absolute latencies don't (Blackwell ≫ Orin). Built engines from `smolvla_base_fp32_valid.onnx`
and compared each against the **FP32 ONNX (ORT CPU = true FP32)** on identical seeded inputs:

| precision | cosine | max_abs | infer (Blackwell, opt0) | verdict |
|-----------|--------|---------|-------------------------|---------|
| fp32 (tf32) | 0.999997 | 1.3e-3 | 165 ms | correct, slowest |
| fp16        | **0.805** | 3.1e-1 | 43 ms | **BROKEN — wrong signs** |
| fp16+bf16   | 0.805 | 3.1e-1 | 44 ms | BROKEN (TRT picks fp16 for the hot layers) |
| **bf16**    | **0.9974** | 6.9e-2 | 104 ms | **near-lossless — RECOMMENDED** |

Why FP16 breaks: the SmolVLM **vision tower** has 730 constants that overflow FP16's exponent
range, incl. literal `inf` attention-mask values in *every* `vision_model/.../self_attn` layer
(→ clipped to ±65504, softmax/layernorm then diverge). BF16 shares FP32's exponent range, so no
overflow, while still using tensor cores. **Per-layer FP32 pinning does NOT save FP16**: TRT's
myelin fusion collapses the softmax/norm nodes into unnamed `__myl_*` supernodes, and
`trtexec --layerPrecisions` only supports a global `*:` default (not substring globs) — so name
pins matched nothing (engine came out byte-identical to plain FP16). BF16 sidesteps all of it.

Also confirmed on the Spark: **the pure-TRT build succeeds** — the vision-tower masked-indexing
ops (`NonZero` ×2, `GatherND`, `ScatterND` ×543) are accepted by TRT 10.x, no hard abort. Build
took ~4 min at opt-level 0 on Blackwell for a 108k-node graph; budget much longer + OOM-watch on
the 8 GB Nano. Raw numbers: `smolvla-spark-finetune/exports/precision_sweep_spark.json`.

→ `build_engine.py --precision bf16 --static-batch` is the recipe; `parity.py` threshold is 0.997.
Open question for the Nano: BF16 ~104 ms on *Blackwell* → Orin will be slower; hitting 10 Hz may
need fewer denoise steps (re-export), independent of precision.

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

NOTE: superseded by the Spark precision sweep at the top — **build BF16, not FP16**. The earlier
FP16 latency guesses below are moot since FP16 is numerically broken for this model. Use them only
as a rough ORT-vs-pure-TRT shape; the real Orin numbers must be measured on-device.

| Path | Estimated latency |
|---|---|
| ORT CUDA EP (TRT not used) | ~108 ms |
| ORT TRT EP | ~40–80 ms |
| Pure TRT engine BF16 | measure on-device (Blackwell did 104 ms; Orin will be slower) |

Target: p95 < 100 ms at 10 denoising steps → 10 Hz loop. Measure before wiring anything. If BF16
can't hit it on the Orin, the lever is **fewer denoise steps** (re-export), not precision.

## Next steps in order

1. Export ONNX on the Spark with small `num_steps` + static shapes.
2. Copy `model.onnx` (+ `.data`) into `smolvla-runtime/exports/`.
3. `pip`-set up the venv (`--system-site-packages`), confirm `import tensorrt, cuda.cudart`.
4. `build_engine.py` → `.engine` (be patient / watch for OOM).
5. Parity-check engine vs ONNX: `parity.py` (FP16 engine vs **FP32 ONNX on CPU EP** — true
   FP32; CUDA EP would be TF32-tainted). Runs ref then engine sequentially to stay under 8 GB;
   identical seeded inputs (same noise per sample); PASS = action cosine ≥ threshold + no NaN/Inf.
6. `run_pipeline.py --backend trt --source synthetic` → first run, sanity.
7. `--source realsense` → real benchmark. Record p95.
8. If pure build fails on an op: `--backend ort` to locate it.

## Parity (TRT vs ONNX)

Harness ready: `parity.py` (FP16 `.engine` vs FP32 ONNX). Reference on CPU EP for *true* FP32
(ORT's CUDA EP uses TF32 on Ampere → not a clean gold). Identical seeded inputs to both, same
noise draw per sample. Reports per-output cosine / max_abs / mean_abs / max_rel + NaN/Inf flag;
PASS when worst action-chunk cosine ≥ `--cos-threshold` (default 0.999) and all outputs finite.
On FAIL it prints the FP32-pin rebuild command. Run after the first engine build:

```bash
python parity.py --onnx exports/smolvla.onnx --engine exports/smolvla.engine --num-samples 3
```

_(no numbers yet — needs the Spark ONNX + a built engine on the box.)_
