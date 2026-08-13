# X-VLA runtime — Jetson Orin Nano 8 GB (ONNX Runtime + TensorRT EP)

Can [X-VLA-0.9B](https://huggingface.co/lerobot/xvla-base) run on the same 8 GB board that
runs SmolVLA? It is roughly **2x the parameters** (879.7 M vs 450 M), and the SmolVLA
deploy already needed split engines to build at all.

The wider point is comparison: **what does a given VLA actually cost on cheap edge
hardware, in memory and in re-planning rate**, so models can be judged side by side on the
same board rather than on a datasheet. Hence the numbers here are measured rather than
estimated, negative results are kept rather than dropped, and the tooling
(`tools/build_probe.py`, `tools/memory_probe.py`, `parity.py`, `run_pipeline.py`) is
written to be pointed at the *next* model too. The comparison so far:

| | SmolVLA (450 M) | X-VLA (879.7 M) |
|---|---:|---:|
| engines | 9 | 12 |
| chunk latency | 210–240 ms | 390 ms |
| actions per chunk | 50 | 30 |
| replan rate | ~4.5 Hz | 2.56 Hz |
| peak RSS | — | 5.71 GB |
| KV cache across denoising steps | yes | **impossible** (see below) |

Short answer so far: **the memory side works, but only with a finer split than SmolVLA
needed.** The numbers behind that are in [`notes/split_design.md`](notes/split_design.md).

This project is deliberately separate from
[`../smolvla-runtime/`](../smolvla-runtime/) so neither experiment can destabilise the
other. Switching between the two at the robot is sketched in
[`notes/model_switching.md`](notes/model_switching.md) — including the part that matters
most: **`xvla-base` cannot drive the excavator as-is** (it is a 20-dim `ee6d` arm policy),
so a deployable X-VLA needs a Spark fine-tune first, exactly as SmolVLA did.

## What the constraint actually is

Not runtime — **the TensorRT build**. TRT imports weights as FP32 working copies
regardless of the ONNX dtype, so the build peak tracks the weight slice a single engine
carries. Measured on this board with `tools/build_probe.py`:

    build peak RSS  ~=  3.18 GB  +  5.63 x (FP32 weight GB)

which leaves room for about **0.40 GB of FP32 weights (~100 M params) per engine**. All
three of X-VLA's heavy components exceed that on their own, so each is split:

| component | FP32 | engines |
|---|---:|---:|
| DaViT vision tower + projector | 1.44 GB | 4 |
| BART encoder + token embedding | 0.83 GB | 3 |
| policy transformer (24 blocks) | 1.21 GB | 4 |
| conditioning projections | 0.01 GB | 1 |

## Why the split falls where it does

`XVLAModel.generate_actions` runs the VLM **once** and the policy transformer
**`num_denoising_steps` (10) times** over the same conditioning, which is the natural
cold-path/hot-path seam. Two things are worth knowing before touching this code:

- **No KV cache is possible.** Unlike SmolVLA's prefill/decode, X-VLA's policy transformer
  is a bidirectional encoder over one concatenated sequence, so the conditioning tokens
  attend *to* the action tokens and change on every step. All 24 blocks re-run over all
  262 tokens, 10 times. That is the latency floor, and the only real lever on it is
  `num_denoising_steps`.
- **The loop is not Euler integration.** It re-forms `x_t` by interpolating between a
  fixed noise draw and the current action estimate, and the transformer predicts the clean
  action directly. Porting SmolVLA's `x_t += dt * v_t` here produces plausible-looking
  garbage.

What *is* hoisted out of the loop: the conditioning projections, their positional
embedding slice, and the soft prompts, none of which depend on `x_t` or `t`. That is
exact, not an approximation.

## Layout

```
xvla-runtime/
  run_pipeline.py            runner + stress test: latency, staged timings, memory
  parity.py                  split engines vs the PyTorch reference — the correctness guard
  xvla_runtime/
    split_ort.py             providers, preprocessing, engine prebuild, the denoising loop
  tools/
    inspect_checkpoint.py    per-component parameter accounting from the safetensors header
    build_probe.py           measures the TRT build-memory curve on this board
    export_split_onnx.py     the split exporter (budget-driven)
  models/xvla-base/          the checkpoint (3.5 GB, not in git)
  exports/split/             exported graphs + bundle.json
  notes/split_design.md      architecture, measurements, engine layout
```

## Setup (JetPack 7.2, Python 3.12, aarch64)

Same host prep as smolvla-runtime ([`../system/`](../system/)). The venv needs
`--system-site-packages` for the JetPack `tensorrt`:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "lerobot[xvla]==0.6.1"          # 0.5.1 has no xvla policy
pip install onnx onnxruntime-gpu==1.24.0 --extra-index-url https://pypi.jetson-ai-lab.io/sbsa/cu130
pip install --ignore-installed "numpy==2.2.6" "scipy>=1.14"
```

That last line is not optional. With `--system-site-packages`, the JetPack `scipy` (built
against numpy 1.x) shadows through and makes `import lerobot` fail on
`cannot import name 'Inf' from 'numpy'`; pip "resolves" it by downgrading numpy, which
then breaks lerobot's `numpy>=2` pin. Installing both *into* the venv settles it. Same
class of trap as the pandas shadowing in `kaivuriprokkis/.venv-lerobot`.

## Run

```bash
# 0. Where does the memory actually go? (no checkpoint needed)
python tools/build_probe.py --blocks 4 8 12

# 1. Parameter accounting straight from the checkpoint header
python tools/inspect_checkpoint.py models/xvla-base/model.safetensors --detail

# 2. Export the split graphs (one subprocess per graph family)
python tools/export_split_onnx.py --checkpoint models/xvla-base --domain-id 0 --valid-views 1

# 3. Build every engine, one subprocess each — do this before the first run
python -c "from xvla_runtime.split_ort import prebuild_engines; \
           prebuild_engines('exports/split', 'exports/split/trt_cache', 'fp16')"

# 4. Parity vs the PyTorch reference BEFORE trusting any action
python parity.py --split-dir exports/split --checkpoint models/xvla-base

# 5. Latency + memory, then the long stress run
python run_pipeline.py --duration-s 30 --show-actions
python run_pipeline.py --duration-s 1800 --report-every 60
```

The engine cache defaults to `exports/split/trt_cache`, not `/tmp`: twelve engines are a
long build to repeat, and `/tmp` is cleared on reboot.

`--valid-views` should be the number of **real** cameras. `num_image_views` is 3 for the
base checkpoint, but padded views are zeroed by the runtime (matching `forward_vlm`, which
scatters valid views into a zero buffer) and never need a forward pass — so one camera
means a batch-1 vision engine and a third of the vision cost.

Engine builds **must** be one subprocess per graph (`--prebuild` does this). Two TRT
builders resident in one process was enough to OOM 8 GB during the SmolVLA work.

## Status — it runs, and it matches the reference (2026-08-13)

X-VLA-0.9B works on this board as 12 split engines.

- **Memory**: all 12 engines build one-subprocess-each with swap untouched (16.38 of
  16.78 GB free throughout). A full rebuild against a warm timing cache is 123 s.
- **Parity**: `action` cosine **1.000000** vs the PyTorch reference (max abs diff 6.5e-4),
  `cond_tokens` 0.999997. The split is faithful, not approximately faithful.
- **Latency**: warm steady state is **397.8 ms avg / 402.3 ms p95** per 30-action chunk
  (denoise 334.6, vision 52.8, text 8.6, cond 1.0) = **2.51 Hz replan**. Over a 15-minute,
  2261-chunk run the p95 sits 1% above the mean and never drifts: no thermal throttling,
  no leak. For comparison SmolVLA does a 50-action chunk in 210–240 ms; X-VLA's hot loop
  is intrinsically ~10x the work and cannot be KV-cached.
- **The new tight constraint is resident memory, not the build**: the stress run held
  **6.73 GB RSS with the available floor at 0.18 GB**. That is the whole board, and the
  real deployment shape puts the camera reader and the 100 Hz controller in the *same*
  process (see `kaivuriprokkis/lerobot_vla/run_inference.py`).

**Memory was chased down and is now understood** (`tools/memory_probe.py`, and the tables in
`notes/split_design.md`). It decomposes as ~2.5 GB fixed CUDA/TRT context plus ~3.5 GB of
engine/runtime allocation, and it is stubborn: FP16 weights in the ONNX buy 0.34 GB, ORT
arena off 0.16 GB, dropping the CUDA EP 0.14 GB, and they do not stack. The guess that ORT
was holding the weights twice was **wrong** — per-engine marginal cost is ~1.0–1.2x, so the
weights are resident once.

**Use `exports/split_fp16`.** Its own 300 s stress run: 769 chunks, **390.0 ms avg /
397.9 ms p95, 2.56 Hz replan, peak RSS 5.71 GB, available floor 1.47 GB** — flat
throughout, and unlike the FP32 build it never touches swap. Parity is still cosine
1.000000. Budget ~5.7 GB resident, leaving **~1.5 GB** for the camera reader and control
loop. Further headroom needs structural changes (fewer views — already 1 of 3, a smaller
backbone, or splitting policy and controller into two processes), not more numerical
tuning.

Other levers: `num_denoising_steps` (10 → 5 would put a chunk near 240 ms) at unmeasured
cost to action quality, IOBinding to keep the denoise chain on device, and a Spark
fine-tune before any of this can drive the excavator.
