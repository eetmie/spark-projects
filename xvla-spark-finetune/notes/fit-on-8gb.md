# Does X-VLA-0.9B fit on an 8 GB Orin Nano? — the measurement record

> Was `orin-nano/xvla-runtime/README.md` before that folder was dissolved. The export
> tooling it describes now lives at this playbook's root; the on-device benchmark and the
> board probes live in [jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla);
> the robot runtime is `kaivuriprokkis/lerobot_vla/`. Kept for the numbers, which stand.

Can [X-VLA-0.9B](https://huggingface.co/lerobot/xvla-base) run on the same 8 GB board that
runs SmolVLA? It is roughly **2x the parameters** (879.7 M vs 450 M), and the SmolVLA
deploy already needed split engines to build at all.

The wider point is comparison: **what does a given VLA actually cost on cheap edge
hardware, in memory and in re-planning rate**, so models can be judged side by side on the
same board rather than on a datasheet. Hence the numbers here are measured rather than
estimated, negative results are kept rather than dropped, and the tooling
(`build_probe.py`, `memory_probe.py`, `parity.py`) is
written to be pointed at the *next* model too. The comparison so far:

Both measured on this board, 1 real camera, 10 denoising steps, FP16 TRT engines:

| | SmolVLA 450 M (`excav_A20k`) | X-VLA 879.7 M (`split_fp16`) |
|---|---:|---:|
| engines | 9 | 12 |
| chunk latency | 210 ms | 390 ms |
| actions per chunk | 50 | 30 |
| motion per chunk @30 fps | 1.67 s (13% duty) | 1.0 s (39% duty) |
| replan rate | 4.8 Hz | 2.56 Hz |
| **peak RSS** | **2.21 GB** | **5.71 GB** |
| free RAM left of 7.4 GB | 4.86 GB | 1.47 GB |
| bytes/param resident | ~4.4 | ~6.9 |
| KV cache across denoising steps | yes | **impossible** (see below) |

The memory column is the striking one: **X-VLA is 2x the parameters but 2.6x the resident
memory**, and it is the difference between 4.9 GB of headroom for the rest of the robot and
1.5 GB. Cost per parameter is also worse (~6.9 vs ~4.4 bytes), which points at TRT
per-engine activation memory rather than weights — 12 engines instead of 9, a 262-token
sequence through 24 blocks at hidden 1024, and DaViT feature maps.

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
carries. Measured on the board with `build_probe.py`:

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

## Where this code lives now

```
xvla-spark-finetune/            (this playbook — export + validate on the Spark)
  export_split_onnx.py          the split exporter (budget-driven)
  parity.py                     split graphs vs the PyTorch reference — the correctness guard
  split_ort.py                  the reference runtime parity scores against
  bundle_contract.py            stats/processor contract + bundle verification
  fp16_weights.py               FP32 -> mixed-FP16 bundle conversion
  tools/inspect_checkpoint.py   per-component parameter accounting from the safetensors header
  notes/split_design.md         architecture, measurements, engine layout

jetson-orin-nano-vla/           (base-model fit + benchmark, on the board)
  bench/tools/build_probe.py    the TRT build-memory curve measured below
  bench/tools/memory_probe.py   resident-memory decomposition
  bench/vendor/xvla_split_ort.py + bench/backends/ort_split_xvla.py

kaivuriprokkis/lerobot_vla/     (the robot)
  vendor/xvla_split_ort.py, xvla_split.py
```

## Setup (JetPack 7.2, Python 3.12, aarch64)

Same host prep as the benchmark repo (`jetson-orin-nano-vla/scripts/`). The venv needs
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
python bench/tools/build_probe.py --blocks 4 8 12    # in jetson-orin-nano-vla, on the board

# 1. Parameter accounting straight from the checkpoint header
python tools/inspect_checkpoint.py models/xvla-base/model.safetensors --detail

# 2. Export the split graphs (one subprocess per graph family)
python export_split_onnx.py --checkpoint models/xvla-base --domain-id 0 --valid-views 1

# 3. Build every engine, one subprocess each — do this before the first run
python -c "from split_ort import prebuild_engines; \
           prebuild_engines('exports/split', 'exports/split/trt_cache', 'fp16')"

# 4. Parity vs the PyTorch reference BEFORE trusting any action
python parity.py --split-dir exports/split --checkpoint models/xvla-base

# 5. Latency + memory on the board — now `python -m bench` in jetson-orin-nano-vla
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

**Memory was chased down and is now understood** (`memory_probe.py`, and the tables in
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
