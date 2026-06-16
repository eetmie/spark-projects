# Findings — SmolVLA runtime on Jetson Orin Nano

Running log. Newest first. (Merges the earlier `smolvla-spark-finetune/jetson/notes`.)

## First-build OOM is a NODE-COUNT wall, not weight dtype (2026-06-16)

Tried to build the TRT engine on-device from the **full 10-step** export
(`smolvla_base_fp{32,16}_static.onnx`, **108,695 nodes**). It does **not** build on the 8 GB
Orin Nano — FP16 or FP32, GUI or headless, default or aggressive TRT knobs. Matrix (synthetic
`run_pipeline`, cold engine cache each run):

| graph        | precision | desktop  | TRT build knobs              | result                          |
|--------------|-----------|----------|------------------------------|---------------------------------|
| 10-step 108k | FP32      | GUI up   | defaults                     | hard OOM (NvMap err 12) ~15 min |
| 10-step 108k | FP32      | GUI up   | opt1, ws256                  | hard OOM ~25 min                |
| 10-step 108k | FP32      | GUI up   | opt1, ws256, no-CUDA-EP      | hard OOM ~16 min                |
| 10-step 108k | FP16      | GUI up   | defaults                     | thrash, no engine in ~50 min    |
| 10-step 108k | FP16      | headless | defaults                     | hard OOM (NvMap err 12) ~44 min |
| 10-step 108k | FP16      | headless | opt1, ws256, no-CUDA-EP      | build FAILED (Err 10) ~85 min → CPU fallback |

The last run is the most informative: after ~85 min of tactic-skips it hit
`IBuilder::buildSerializedNetwork: Error Code 10: Could not find any implementation for node`
— TRT couldn't find *any* tactic small enough for the ~60 MB free, so the engine build **failed
outright** (not merely slow). ORT then fell back to **CPU-only** and ran the whole model there,
emitting a **finite, plausible** action chunk: `action[0]=[+0.014,-0.115,-0.076,-0.005,+0.096,
-0.470,...]`, shape `(50,32)`. No engine was cached. **Silver lining: that finite CPU-fallback
output validates the `--fp16-weights` conversion is numerically sound end-to-end** (LayerNorm/
Softmax-in-FP32 block list held; no NaN/inf) — the only missing piece is a *buildable* engine.

**Diagnosis: build-memory peak scales with graph NODE COUNT, not weight dtype.** TRT's per-node
tactic exploration over 108k nodes drives host RSS to ~7.4 GB *regardless of FP16/FP32* — the
803 MB FP16 file hit the same 7.4 GB peak as the 1.58 GB FP32. Physical RAM fills, so GPU NvMap
allocations (non-swappable) fail → hard OOM, or TRT exhausts all tactics and the build errors out
(Err 10). So `--fp16-weights` does **not** help the *build* — it only halves the deployed/loaded
footprint. (It's still the right deploy artifact; just not what unblocks the build.)

- **Headless barely mattered here:** stopping `gdm3` freed only ~110 MB (idle GNOME is light); the
  Jetson AI Lab "~800 MB" assumes a fuller desktop. Set `default.target=multi-user.target` anyway.
- **Build-peak env knobs** (`backends/ort.py`: `TRT_OPT_LEVEL`, `TRT_WORKSPACE_MB`,
  `TRT_DROP_CUDA_EP`) turn the *hard* OOM into a *survivable thrash*, but don't make a 108k-node
  build finish in reasonable time.

**CORRECTION (later same day): reduced num_steps does NOT fix it — weights are the floor.**
Built the 5-step export (`smolvla_base_fp16_static_s5.onnx`, **61,370 nodes**, 43% fewer than 108k)
headless, both with defaults and with the ws256/no-CUDA knobs. Both thrashed/OOM'd the same way,
peaking **~6.7 GB** — only ~0.7 GB below the 10-step's 7.4 GB despite far fewer nodes. So the build
peak is **dominated by a node-count-INDEPENDENT floor (~6 GB): TRT imports the weights as FP32
working copies (~1.6 GB) + CUDA/ORT runtime (~1–2 GB) + scratch.** That's also why FP16 and FP32
peaked identically (TRT builds in FP32 regardless of the file's dtype). `num_steps` only trims the
thin node-scaling layer on top — not enough to fit. (Note: the NvMap `error 12` lines are NOT fatal;
TRT logs them and keeps skipping tactics — the build survives but crawls, and at best limp-completes
via CPU fallback, which is not a deployable engine.)

**VALIDATED on-device (2026-06-17): the split builds.** Ran `ainekko/smolvla_base_onnx` (9 base-weight
split graphs) through `tools/build_probe.py` (TRT-EP build of a single ONNX with dummy static inputs).
The three heavy transformer graphs each built + cached a clean FP16 TRT engine in ≤60 s, no thrash, no
swap spike, no OOM — where the monolith OOM'd for 85+ min:
  - `smolvlm_expert_prefill` (644 MB) → 320 MB engine, ~60 s
  - `smolvlm_vision` (393 MB) → ~60 s
  - `smolvlm_expert_decode` (399 MB) → ~43 s
  - `smolvlm_text` (189 MB) + 5 projectors + state → run on the EP stack in 1–3 s (text falls to CUDA-EP,
    fine for a once-per-inference encoder). 3 real TRT engines, 690 MB cached total.
Inference also validated (`build_probe.py --runs 30`, FP16 TRT, finite outputs): vision 33.1 ms,
expert_prefill 16.5 ms, expert_decode 11.4 ms (text 0.1 ms is a dummy seq-len-1 input, not real).
Projected full loop = (vision + text + prefill) once + decode ×N: ~52 ms fixed + ~12 ms/step →
**~170 ms @ 10 steps (~6 Hz), ~110 ms @ 5 steps (~9 Hz)** — full num_steps quality, loop in Python.
(Per-engine dummy-input numbers; real end-to-end measured once the loop is wired.)
Confirms the diagnosis: the wall was building all 450M weights at once, not node count or precision.
Per-component, each weight slice builds in ~a minute. Deploy path = re-export OUR fine-tuned weights in
this split layout + Python denoise loop (prefill ×1 → decode ×N) in `backends/ort.py`.

**Real fix = SPLIT the model into per-component engines (not fewer steps).** Each split graph carries
only its slice of the 450M weights (vision / text / expert-prefill / expert-decode / projectors), so
each builds with a few hundred MB of weights → ~2–3 GB peak each → fits easily on 8 GB. The denoise
loop runs in Python (prefill ×1, decode ×N). Reference: HF `ainekko/smolvla_base_onnx` (9 graphs) +
github.com/aifoundry-org/ETARS. See [[smolvla-orin-split-engine-deploy]] in memory. Replicate the
split for our fine-tuned weights in `export_valid_onnx.py`; the full/s5 monolith is a dead end on 8 GB.

## JetPack 7.2 migration — FP16 deploy, single ORT/TRT-EP backend (2026-06-13)

Board reflashed to **JetPack 7.2**: L4T R39.2.0, Ubuntu 24.04, Python 3.12.3, CUDA 13.2, TensorRT
**10.16.2**. Two earlier decisions below (pure-TRT primary; "BF16 recommended") are **superseded**:

- **Precision is FP16 on the Orin, not BF16.** On-device `tools/probe_precision.py` (TRT 10.16,
  compute 8.7): `platform_has_fast_fp16=True`, `platform_has_fast_int8=True`, **`platform_has_fast_bf16
  = n/a`**. BF16 has no hardware fast path on Orin. The "FP16 is broken (cos 0.805)" sweep below was a
  *blanket-FP16* engine that forced the whole vision tower to FP16 → exponent overflow. The deploy
  path now is **FP32 ONNX → ORT TensorRT-EP with `trt_fp16_enable` + `trt_layer_norm_fp32_fallback`
  + CUDA-EP fallback**, so TRT lowers only what's safe to FP16 and the sensitive norms / fallback ops
  stay FP32. BF16 (`trt_bf16_enable`) is kept as a *gated experiment* — keep only if logs show real
  BF16 tactics here AND it beats FP16 on latency + parity. Reference dtype stays PyTorch-BF16 on the
  Spark.
- **Single backend; pure-TRT path deleted.** The monolithic `trtexec` engine build OOM'd on 8 GB and
  was the all-or-nothing path. Removed `build_engine.py` + `backends/trt_engine.py` (and `cuda-python`).
  ORT's TRT-EP builds engines per-subgraph (lower memory peak) and caches them — it is now the only
  inference path. `run_pipeline.py --backend ort` / `parity.py` both run it.
- **Build memory fix:** 8 GB is *unified*. Grow swap to 16 GB on the NVMe (`system/setup-swap.sh`) +
  MAXN_SUPER + pinned clocks (`system/`) so the first engine build fits. Prefer the 5-step ONNX
  (`*_static_s5.onnx`, ~60k nodes) for the first build to shrink it further.
- **Camera:** RGB-only via librealsense RSUSB build (`realsense-rgb/`) — no kernel patches on 7.2.
- **onnxruntime-gpu for CUDA 13 — RESOLVED.** No `jp7` index exists on Jetson AI Lab; the CUDA-13
  aarch64 wheels are under the **`sbsa`** index. `onnxruntime-gpu==1.24.0` from
  `https://pypi.jetson-ai-lab.io/sbsa/cu130` installs (cp312) and reports
  `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`. No source build
  needed. (numpy resolves to 1.26.4.) Benign DRM-probe warning on import; CUDA EP works regardless.

--- everything below predates the JetPack 7.2 migration (Spark-era / JetPack 6) ---

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

## [SUPERSEDED on Orin — see top] USE BF16, NOT FP16 — precision sweep on the Spark (2026-06-02)

> Superseded for *Orin deployment*: this sweep forced a blanket-FP16 engine on Blackwell. On the
> Orin (compute 8.7) BF16 isn't hardware-accelerated, and the partitioned ORT/TRT-EP FP16 path keeps
> the overflowing vision-tower ops in FP32. Still useful for *why* blanket-FP16 overflows.


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
the 8 GB Nano. Raw numbers: `smolvla-spark-finetune/precision_sweep_spark.json`.

→ `build_engine.py --precision bf16 --static-batch` is the recipe; `parity.py` threshold is 0.997.
Open question for the Nano: BF16 ~104 ms on *Blackwell* → Orin will be slower; hitting 10 Hz may
need fewer denoise steps (re-export), independent of precision.

## [SUPERSEDED on Orin — see top] Decision: pure TensorRT engine as the primary path

> Superseded: the pure-TRT engine build OOM'd on the 8 GB Orin and is deleted. ORT/TRT-EP (which was
> the "diagnostic" here) is now the single backend. The catch-list below is still an accurate map of
> SmolVLA's TRT-unfriendly ops.


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
