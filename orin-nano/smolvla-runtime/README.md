# SmolVLA runtime — Jetson Orin Nano (ONNX Runtime + TensorRT EP)

The model pipeline, end to end, on the Orin Nano (JetPack 7.2):

```
RealSense D435i RGB  ->  SmolVLA (ONNX Runtime + TensorRT EP)  ->  action chunk
```

**No robot control here.** This stage proves the *model* runs fast enough on the Nano: one camera
feed in, action chunks out. Mapping actions onto a real robot (normalization, limits, watchdogs,
safety) is deliberately a separate, later stage.

Upstream is [`../../smolvla-spark-finetune/`](../../smolvla-spark-finetune/) — fine-tune + FP32 ONNX
export happen on the **DGX Spark**. This box only ever sees the ONNX; ORT's TensorRT EP builds and
caches a TensorRT engine from it locally.

## Deploy path: SPLIT engines (the monolith does NOT build on 8 GB)

The single-graph SmolVLA export (`sample_actions`, denoise loop unrolled) **cannot TRT-build on this
8 GB board** — not at FP16, not at `--num-steps 5`, not headless. The build OOMs because TRT imports
all 450M weights as FP32 working copies at once (~6 GB floor, node-count-independent — `num_steps`
barely moves it). See [`notes/findings.md`](notes/findings.md) for the full matrix.

**The fix is to split the model into per-component engines** and run the denoise loop in Python:
vision + text + expert-**prefill** (KV cache) run **once**, then expert-**decode** runs **×N** steps
(`x_t += dt·v_t`). Each engine carries only its weight slice, so each builds in ≤60 s and runs in ms.
Validated on-device against the reference base-weight split (HF `ainekko/smolvla_base_onnx`, loop in
`github.com/aifoundry-org/ETARS`) with [`tools/build_probe.py`](tools/build_probe.py): 3 heavy engines
built (690 MB), inference vision 33 ms / prefill 16.5 ms / decode 11.4 ms → **~5–9 Hz end-to-end**.

**To deploy our fine-tuned policy:** re-export OUR weights in this split layout on the Spark (see
`../../smolvla-spark-finetune/`), then wire the prefill→decode loop into `backends/ort.py`. The
monolithic path below still documents the ORT/TRT-EP mechanics each split engine uses.

## Why ORT + TensorRT EP (one backend)

ONNX Runtime's TensorRT execution provider partitions the graph, builds a TensorRT engine per
supported subgraph (cached to disk on first run), and hands anything TRT can't take to the CUDA EP.
That gets near-TensorRT speed while sidestepping the two things that sank a single monolithic
`trtexec` build on this board: the all-or-nothing abort on one unsupported op, and the build-time
memory peak that OOM'd 8 GB. So there is exactly one inference path — no separate engine-build step.

## Precision: FP16 (BF16 is experimental)

`tools/probe_precision.py` on this board reports `platform_has_fast_fp16 = True` but
`platform_has_fast_bf16 = n/a` — **BF16 is not hardware-accelerated on the Orin (compute 8.7)**.
SmolVLA's reference dtype is BF16, but that reference lives on the Spark (PyTorch); the Orin deploy
path is **FP16** via the TRT EP, with `trt_layer_norm_fp32_fallback` keeping the precision-sensitive
norms in FP32 and the CUDA EP catching the rest at FP32. (The old "FP16 is broken, cos 0.805" result
was a *blanket-FP16* engine on Blackwell that forced the vision tower to overflow — not this
partitioned path. `parity.py` is the on-device guard.) BF16 stays a gated experiment: keep it only
if it builds real BF16 tactics here *and* beats FP16. See [`notes/findings.md`](notes/findings.md).

## Layout

```
smolvla-runtime/
  run_pipeline.py            the runner: source -> backend -> action chunk + latency report
  parity.py                  ORT/TRT-EP (fp16) vs FP32 ONNX (CPU) — the precision guard
  tools/probe_precision.py   what TRT can accelerate on this board (fp16/bf16/int8)
  requirements.txt
  smolvla_runtime/
    camera.py                threaded D435i latest-frame reader + a synthetic source
    preprocess.py            image resize/normalize, tokenize, state pad, noise (the only model glue)
    io_spec.py               maps logical roles -> tensor names (survives image vs image0)
    backends/
      ort.py                 the deployment backend: ONNX Runtime + TensorRT EP (fp16 default)
      mock.py                zero actions, no model (camera + loop plumbing check)
  notes/findings.md
```

## Setup (JetPack 7.2, Python 3.12, aarch64)

Host prep first: [`../system/`](../system/) (power, swap) and [`../realsense-rgb/`](../realsense-rgb/)
(camera). Then the venv — `tensorrt` and `pyrealsense2` come from the system, so it must see them:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt   # see requirements.txt for the onnxruntime-gpu / CUDA 13 index
python -c "import tensorrt, pyrealsense2; print(tensorrt.__version__, 'deps ok')"
python -c "import onnxruntime as o; print(o.get_available_providers())"  # expect Tensorrt + CUDA EP
```

## Run

```bash
# 0. Plumbing only (no model) — confirm the real D435i feed + the loop:
python run_pipeline.py --backend mock --source realsense --duration-s 8

# 1. Confirm what TRT can accelerate on this board (fp16 yes, bf16 n/a):
python tools/probe_precision.py

# 2. First inference: synthetic frames, builds + caches the engine (slow first time, minutes).
#    No camera needed. Watch `tegrastats` in another shell for the memory peak.
python run_pipeline.py --backend ort --onnx-path exports/smolvla.onnx \
    --model-id lerobot/smolvla_base --source synthetic --duration-s 20 --show-actions

# 3. Parity-check vs the FP32 ONNX BEFORE trusting any action (fp16 should pass at >= 0.997):
python parity.py --onnx exports/smolvla.onnx --num-samples 3

# 4. The actual pipeline: ORT/TRT-EP + real D435i RGB:
python run_pipeline.py --backend ort --onnx-path exports/smolvla.onnx \
    --model-id lerobot/smolvla_base --source realsense --duration-s 30 --show-actions

# (experiment) compare the BF16 mode — keep only if it builds BF16 tactics AND beats fp16:
python parity.py --onnx exports/smolvla.onnx --precision bf16 --num-samples 3
```

The TRT-EP engine cache lives in `--engine-cache-dir` (default `/tmp/smolvla_trt_cache`); the first
build is one-time, later runs load from it in ~seconds. A different `--precision` builds a separate
cached engine.

## Reading the output

| field | meaning |
|---|---|
| `infer_ms(avg/p95)` | inference latency only |
| `loop_ms(avg/p95)` | full camera→action loop time |
| `img_age_ms` | how stale the frame was when inference started |
| `action_shape` | raw SmolVLA output, e.g. `(50, 32)` — padded; map to the robot's dims later |

**On the target:** "10 Hz" is the *control* rate, not the inference rate — SmolVLA emits a 50-action
chunk per observation, executed open-loop, so a 200–500 ms inference is hidden behind chunking (see
findings). Lower latency buys fresher re-planning + lower power, not a Hz threshold. If latency is
the bottleneck, the lever is **fewer denoising steps** (re-export the ONNX on the Spark), independent
of precision.

## Validate the runtime without the Spark ONNX

There is **no published ONNX of SmolVLA** (HF `lerobot/smolvla_base` is PyTorch only) — the
deployable ONNX is exported on the Spark by `../../smolvla-spark-finetune/export_valid_onnx.py`. To
exercise the *whole runtime* on the board before that lands, generate a shape-correct stand-in:

```bash
python tools/make_synthetic_onnx.py --out exports/synthetic_smolvla.onnx
python run_pipeline.py --backend ort --onnx-path exports/synthetic_smolvla.onnx \
    --model-id HuggingFaceTB/SmolLM2-135M-Instruct --source synthetic --duration-s 8 --show-actions
python parity.py --onnx exports/synthetic_smolvla.onnx --model-id HuggingFaceTB/SmolLM2-135M-Instruct
```

This proves provider registration, the TRT-EP engine build + cache, io_spec, preprocess, and the
loop. It is **not** the model — outputs are meaningless; it only validates the plumbing.

## Tokenizer note (matters for real inference)

`--model-id` selects the tokenizer (we don't load any torch weights). `lerobot/smolvla_base` ships
**no tokenizer files**, so a mismatched tokenizer = wrong tokens = wrong actions. The Spark export
(`smolvla-spark-finetune/export_valid_onnx.py`) now saves the **vocab-exact** tokenizer beside the
ONNX as `exports/tokenizer/` — point `--model-id exports/tokenizer` at that bundle (authoritative,
self-contained, no network). For a quick plumbing test before the real bundle arrives, any backbone
tokenizer works (e.g. `HuggingFaceTB/SmolLM2-135M-Instruct`; the SmolLM2 tokenizer needs
`sentencepiece`, already in requirements) — outputs are meaningless then anyway.

## Status

Migrated to JetPack 7.2 and **baselined on-device (2026-06-13)**: MAXN_SUPER + pinned clocks, 16 GB
swap, `onnxruntime-gpu 1.24.0` (CUDA 13, from the `sbsa/cu130` index) with **TensorRT + CUDA + CPU
EPs active**. Full ORT/TRT-EP path verified through a synthetic ONNX — engine builds + caches,
`run_pipeline` runs (~110 Hz loop on the trivial graph), `parity.py` PASSES (cos 1.0). Remaining for
real inference: the Spark SmolVLA ONNX (+ matching tokenizer) and the D435i RGB camera.
