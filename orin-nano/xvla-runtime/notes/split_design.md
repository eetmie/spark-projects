# X-VLA split design for the 8 GB Orin Nano

## The shape of the problem

X-VLA-0.9B is 879.7 M parameters — 3.52 GB as FP32, near 2x SmolVLA's 450 M. From the
`lerobot/xvla-base` checkpoint header (`tools/inspect_checkpoint.py`):

| component | params | FP32 |
|---|---:|---:|
| `vlm.vision_tower` (DaViT) | 360.6 M | 1.44 GB |
| — conv-embeds + stages 0–1 | 32.7 M | 0.13 GB |
| — stage 2 (9 blocks @ dim 1024) | 227.1 M | 0.91 GB |
| — stage 3 (1 block @ dim 2048) | 100.8 M | 0.40 GB |
| `vlm.language_model` (BART encoder + embed, **no decoder**) | 207.9 M | 0.83 GB |
| `transformer.blocks` (24 × hidden 1024) | 302.3 M | 1.21 GB |
| everything else (projections, soft prompts, pos_emb) | 8.9 M | 0.04 GB |

The checkpoint ships in the *vendored* Florence-2 layout (`vlm.language_model.model.*`,
`vlm.image_projection`), which `XVLAPolicy.from_pretrained` remaps to the native
`transformers` layout on load. It contains no text decoder — `XVLAModel.__init__` deletes
it — so the 207.9 M is encoder + token embedding only.

Runtime is not the binding constraint: 879.7 M FP16 weights = 1.76 GB resident, plus CUDA
context and activations, lands around 3–3.5 GB against ~5.4–6.5 GB available. **The TRT
build is the wall**, exactly as with SmolVLA, because TRT imports weights as FP32 working
copies regardless of the file dtype. So the split is chosen by build peak per engine, and
`tools/build_probe.py` measures that curve directly rather than trusting extrapolation.

## Where the model wants to be cut

`XVLAModel.generate_actions` has a much cleaner seam than SmolVLA's prefill/decode:

```python
enc = self.forward_vlm(input_ids, image_input, image_mask)   # ONCE
for i in range(steps, 0, -1):                                # 10x
    action = self.transformer(domain_id=..., action_with_noise=x_t,
                              proprio=..., t=t, **enc)
```

The VLM runs once per observation; the policy transformer runs `num_denoising_steps=10`
times over the *same* `enc`. That maps onto two engine groups: a cold path (vision + text,
1x) and a hot path (policy transformer, 10x).

### No KV cache is possible here — and that is the cost driver

SmolVLA cached the prefill KV and ran a small per-step decode. X-VLA cannot: the policy
transformer is a **bidirectional encoder over one concatenated sequence**, so the
conditioning tokens attend *to* the action tokens and their representations change on every
denoising step. Nothing about the conditioning is reusable across steps beyond its
projection. All 24 blocks re-run over the full sequence, 10 times.

Sequence layout (`SoftPromptedTransformer.forward`), for our config:

| segment | tokens | source |
|---|---:|---|
| action | 30 | `chunk_size` |
| `vlm_features` | 100 | 50 image tokens (view 0) + 50 language tokens |
| `aux_visual_inputs` | 100 | views 1–2, 50 tokens each |
| soft prompts | 32 | `len_soft_prompts` |
| **total** | **262** | must stay ≤ `max_len_seq` 512 |

Language is 50 tokens, from the checkpoint's `policy_preprocessor.json`
(`tokenizer_processor.max_length: 50`) — *not* the 1024 in `config.json`, which the
preprocessor overrides. At 1024 the sequence would be 1204 and blow the 512 `pos_emb`
limit, so the preprocessor value is the authoritative one.

Cost: 24 layers × 12.6 M MAC/token × 262 tokens ≈ 79 G MAC ≈ 158 GFLOP per denoising step,
×10 steps ≈ 1.6 TFLOP per action chunk. This is why X-VLA will not match SmolVLA's
210–240 ms; the hot loop is intrinsically ~10x more work.

### The free win: hoist the loop-invariant conditioning

Inside the per-step forward, three things depend only on `enc` and `domain_id`, never on
`x_t` or `t`:

- `vlm_proj(vlm_features)` and `aux_visual_proj(aux_visual_inputs)` — 200 tokens through a
  1024→1024 linear
- the `pos_emb` slice added to those 200 conditioning positions
- `soft_prompt_hub(domain_id)` — the 32 prompt tokens

Recomputing them 10x is pure waste. Hoisting them into the cold path is *exactly*
equivalent, not an approximation: positions are fixed, so `pos_emb[:, 30:230]` can be
folded into the precomputed conditioning block while the per-step graph keeps
`pos_emb[:, :30]` for the action segment. That leaves the hot graph with only the action
encoder, the 24 blocks, and the decoder.

`domain_id` is baked at export time. It selects the `DomainAwareLinear` weights for
`action_encoder`/`action_decoder` and the soft prompts via an `nn.Embedding` gather; with a
fixed deployment domain those fold to constants and the 30-domain tables never enter the
engine. Changing domain means re-exporting, which is the right trade for a deployed policy.

## Measured: the build-memory curve (2026-08-13)

`tools/build_probe.py`, FP16, seq_len 262, `TRT_OPT_LEVEL=2`, `TRT_WORKSPACE_MB=512`,
CUDA EP dropped, export and build in separate processes:

| blocks | params | FP32 weights | build peak RSS | system consumed | headroom left | build | infer |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 50.4 M | 0.20 GB | 4.32 GB | 3.62 GB | 2.88 GB | 35 s | 7.0 ms |
| 8 | 100.8 M | 0.40 GB | 5.42 GB | 4.19 GB | 1.69 GB | 38 s | 11.4 ms |
| 12 | 151.2 M | 0.60 GB | 6.59 GB | 5.49 GB | **0.47 GB** | 43 s | 14.7 ms |

    build peak RSS  ~=  3.18 GB  +  5.63 x (FP32 weight GB)

The ~3.2 GB intercept is the node-count-independent floor the SmolVLA work already
identified — CUDA/TRT context plus ORT's copies — and the 5.6x slope is TRT holding
several FP32 working copies of the weights while it explores tactics.

Both numbers are for an FP32 ONNX built with `trt_fp16_enable`, which is the only
configuration measured here. The SmolVLA work found that an FP16 ONNX peaks the same
because TRT builds in FP32 regardless of file dtype; that was not re-tested for X-VLA, and
it is worth re-testing, because an FP16 export could still halve the *resident* footprint
even if it leaves the build peak alone — and resident memory turned out to be the binding
constraint (below).

**The 12-block build finished with 0.47 GB to spare.** It "passed", but that is not a
margin worth deploying against, and it sets the real ceiling:

- **budget ≈ 0.40 GB FP32 (~100 M params) per engine**, which built with 1.69 GB of slack
- 0.60 GB per engine is the edge of OOM
- one monolithic 24-block denoiser (1.21 GB) projects to ~10 GB — never going to build,
  which is the same wall SmolVLA hit, just further up

Latency scales at ~0.96 ms per block plus ~3.1 ms fixed per engine call, so splitting is
not free: each extra engine adds its own per-call overhead, paid on every denoising step.

## Engine layout

Sized to the 0.40 GB budget above. Cold path, once per observation:

1. `vision` — DaViT + pos/temporal embeds + projection: `[V,3,224,224] → [V,50,1024]`.
   1.44 GB total, so it splits at stage boundaries: stage 2 (0.91 GB) needs 2–3 pieces,
   stage 3 (0.40 GB) is one, and the conv-embeds plus stages 0–1 (0.13 GB) are one.
2. `text_encoder` — token embedding + 12 BART layers over `[50 image ; 50 text]`
   → `vlm_features [1,100,1024]`. The 0.21 GB embedding is a Gather and does not need
   TRT; the 12 layers (0.60 GB) split 2×6 at 0.30 GB each.
3. `cond` — the hoisted projections above → `cond_tokens [1,200,1024]`. ~2 M params.

Hot path, 10x per observation:

4. `denoise` — action encoder, 24 blocks over 262 tokens, norm + action decoder.
   **4 engines × 6 blocks (0.30 GB each)**. 3×8 at 0.40 GB sits exactly on the budget and
   would save ~3 ms/step; 4×6 buys real margin for ~35 ms/step total. Each boundary costs
   one `[1,262,1024]` round-trip per step — ~0.5 MB in FP16, negligible against 158
   GFLOP, and removable with IOBinding.

Projected hot path: ~35 ms/step × 10 steps ≈ 350 ms, plus the cold path. The lever if that
is too slow is `num_denoising_steps`, not the split — 5 steps would halve it, and the
action-quality cost of that needs measuring against the 10-step reference.

Every engine is built in its own subprocess. Two resident TRT builders OOM'd 8 GB during
the SmolVLA work, and that finding is not model-specific.

## Export pitfalls (both cost a full build cycle to find)

**1. `BartEncoder.forward` cannot be traced.** It calls `create_bidirectional_mask`
unconditionally, which reaches `sdpa_mask` and does
`q_length.shape[0]` on something that is a tuple under tracing → `IndexError: tuple index
out of range`. Passing `attention_mask=None` does *not* skip the call. The fix is to
reproduce the four-line preamble (embed_positions, layernorm_embedding, then the layers)
and pass `attention_mask=None` into the layers directly. That is safe here only because
`forward_vlm` builds an all-ones mask — it does not mask language padding — so full
attention is the reference behaviour, not a simplification of it.

**2. The TorchScript exporter silently emits unloadable graphs.** `torch.onnx.export`
with `dynamo=False` turned the denoiser's `torch.ones_like(x) ; x[..., idx] = 0` and
`.expand(-1, n, -1)` into `Add`/`Expand` nodes whose inputs were **empty strings**. The
`.onnx` file writes without complaint and fails minutes later at engine build with
`input 0 is marked single but has an empty string`. Fixes: precompute the gripper masks
as constant buffers (shapes are static anyway) and use `repeat` with explicit counts
instead of `expand(-1, ...)`.

Because that failure surfaces so far from its cause, `dump()` now runs
`onnx.checker.check_model` **and** an explicit empty-input scan on every graph, so a bad
export fails immediately with the graph name.

**3. ORT's own fusions break the FP16 graphs.** After the FP16 weight cast, building
`vision_0` failed with
`NOT_IMPLEMENTED: Failed to find kernel for com.microsoft.Gelu(1) ... implemented only for
tensor(float), node has tensor(float16)`. Neither the FP32 nor the FP16 ONNX contains a
Gelu node — every op in both files is default-domain. ORT's graph optimizer *creates* the
fused `com.microsoft.Gelu` at session load, before provider partitioning; TRT cannot take
a contrib-domain op, so it falls back, and the CPU EP has no float16 kernel for it. Fix is
`graph_optimization_level = ORT_DISABLE_ALL` (`make_session_options`), which is the right
setting with the TRT EP anyway — ORT fusions produce nodes TRT then has to refuse.

## Measured: the actual build (2026-08-13)

12 engines, `exports/split`, FP16, one subprocess each. **Swap stayed at 16.38 GB free of
16.78 GB throughout** — nothing thrashed, which is the result that matters: the split
turns a build that cannot happen at all into one that never touches swap.

Build times are dominated by a cold shared timing cache, not by engine size:

| engine | FP32 weights | build |
|---|---:|---:|
| vision_0 | 0.36 GB | 166 s |
| vision_1 | 0.30 GB | 18 s |
| vision_2 | 0.38 GB | 24 s |
| vision_3 | 0.41 GB | 65 s |
| text_encoder_0 | 0.38 GB | 25 s |

The first engine pays CUDA/TRT init plus an empty timing cache; every engine after it
reuses `trt_timing_cache` from the same directory and lands in tens of seconds. So the
one-time cost of a 12-engine split is minutes, not the hours a naive per-engine
extrapolation from the first build would suggest. The cache lives in
`exports/split/trt_cache` rather than `/tmp` precisely so this is paid once per export,
not once per reboot. A full rebuild against a warm cache is **123 s for all 12 engines**.

## Measured: parity and latency (2026-08-13)

`parity.py`, FP16 engines vs the FP32 PyTorch reference on CPU, same seeded inputs and the
same `x1` noise draw:

| tensor | cosine | max abs diff |
|---|---:|---:|
| `cond_tokens` | 0.999997 | 0.565 |
| `action` | 1.000000 | 0.00065 |

So the whole rearrangement — 12 engines, the hoisted conditioning, the baked domain
weights, the reconstructed BART preamble, `attention_mask=None` standing in for the
all-ones mask — reproduces the reference. (`cond_tokens` has a larger absolute delta
because those activations are large; the relative error is 0.006, and it does not
propagate: the action output matches to 6e-4.)

Cold-path/hot-path timings for one 30-action chunk, 1 real camera, 10 denoising steps:

| stage | runs | ms |
|---|---:|---:|
| vision (4 engines) | 1x | 69 |
| text encoder (3 engines) | 1x | 9 |
| cond | 1x | 1 |
| **denoise (4 engines)** | **10x** | **411** |
| total | | ~490 |

≈2.0 Hz replan with 30 actions per chunk. The denoising loop is 84% of the time, exactly
as the no-KV-cache analysis predicted, so `num_denoising_steps` is the only lever that
matters: 5 steps would put the chunk near 290 ms. Whether the action quality survives
that is an open question and needs measuring against the 10-step output.

## Measured: 15-minute stress run (2026-08-13)

`run_pipeline.py --duration-s 900`, synthetic frames, 10 steps, 2261 chunks:

| | avg | p95 | min |
|---|---:|---:|---:|
| chunk | 397.8 ms | 402.3 ms | 377.0 ms |
| — denoise | 334.6 ms | 338.4 ms | |
| — vision | 52.8 ms | 54.4 ms | |
| — text | 8.6 ms | 9.0 ms | |
| — cond | 1.0 ms | 1.1 ms | |

**2.51 Hz replan, and completely flat**: p95 is 1% above the mean and the per-120 s lines
never move off 398 ms across the whole run. No thermal throttling, no latency drift, no
memory growth. Warm steady state is faster than the 490 ms seen on the first (cold)
inference in `parity.py`.

**Resident memory is the binding constraint now, not the build.** The run held
**6.73 GB RSS with the system available floor at 0.18 GB**, dipping ~1.3 GB into swap.
Nothing is left for a camera pipeline and a control loop, which is the actual deployment
shape (`kaivuriprokkis/lerobot_vla/run_inference.py` runs the policy, the RealSense reader
and the 100 Hz controller in one process). Reducing this is tracked below and in
`model_switching.md`.

### Where the 6.7 GB actually goes (2026-08-13)

The first guess was double counting — ORT holding the parsed ONNX proto *and* the TRT
engine, i.e. the weights twice. **`tools/memory_probe.py` says that is wrong**, which is
the reason to measure a memory problem rather than reason about one. Loading the 12
sessions one at a time, baseline config:

| engine | FP32 weights | marginal RSS | ratio |
|---|---:|---:|---:|
| vision_0 | 0.36 GB | **2.56 GB** | 7.15x |
| vision_1 | 0.30 GB | 0.37 GB | 1.24x |
| vision_2 | 0.38 GB | 0.46 GB | 1.22x |
| vision_3 | 0.41 GB | 0.41 GB | 1.01x |
| text_encoder_0 | 0.38 GB | 0.40 GB | 1.06x |
| text_encoder_1 | 0.35 GB | 0.74 GB | 2.09x |
| text_encoder_2 | 0.10 GB | 0.05 GB | 0.47x |
| denoise_0..3 | 0.30 GB each | 0.24–0.46 GB | 0.80–1.53x |
| **all 12** | **3.50 GB** | **6.41 GB** | 1.83x |

The 1.83x aggregate is misleading: `vision_0` carries the one-time **CUDA/TRT context
init (~2.2 GB)** because it is simply the first session created. Every session after it
costs ~1.0–1.2x its own weights, so the weights are resident **once**, not twice, and
EPContext will not recover much (it should still cut load time and the transient peak
during loading, so it stays on by default).

The real decomposition is therefore:

    6.4 GB  ≈  2.2 GB fixed CUDA/TRT context  +  3.5 GB FP32 weights  +  ~0.7 GB slack

ORT knobs are noise against that — `enable_cpu_mem_arena=False` saves 0.16 GB, dropping
the CUDA EP saves 0.14 GB, and combining them saves nothing extra:

| config | RSS | vs baseline |
|---|---:|---:|
| baseline | 6.46 GB | — |
| no_arena | 6.30 GB | −0.16 GB |
| no_cuda_ep | 6.32 GB | −0.14 GB |
| no_arena_no_pattern | 6.30 GB | −0.16 GB |
| no_cuda_ep_no_arena | 6.32 GB | −0.14 GB |

The context is fixed cost, so the term worth attacking is the 3.5 GB of weights. The
obvious lever is **FP16 weights in the ONNX** (`tools/fp16_weights.py`), since the
smolvla-runtime findings measured that this does not help the build but does "halve the
deployed/loaded footprint".

Precision recipe is inherited rather than reinvented: mixed FP16 with
`LayerNormalization` and `Softmax` blocked to FP32 and `keep_io_types=True`. A *blanket*
FP16 cast is what overflowed SmolVLA's vision tower (cos 0.805). BF16 is not an option —
on Orin (compute 8.7) `platform_has_fast_bf16` is n/a, no hardware fast path.

### FP16 weights: correct and slightly faster, but it did NOT free the memory

The conversion itself is clean — every graph halved, 3503 MB → 1753 MB of ONNX, 0.50x
across the board — and it is numerically sound: parity still passes at **action cos
1.000000** (max|d| 7.4e-4), and it is a little quicker (denoise 366 ms vs 411 ms cold,
vision 55 vs 69 ms).

But resident memory barely moved:

| build | sessions | RSS | available after |
|---|---:|---:|---:|
| FP32 graphs | 6.41 GB | 6.46 GB | 0.29 GB |
| FP16 graphs | 6.03 GB | 6.07 GB | 0.56 GB |

**0.39 GB, not the ~1.75 GB the halved weights predicted.** So TRT is not storing the
engine at the file's dtype: `trt_fp16_enable` already let it choose FP16 per layer when
building from the FP32 graph, and whatever it keeps in FP32 it keeps in FP32 either way.
Halving the file halves what ORT parses, not what the engine holds — and ORT evidently
frees the parsed initializers once the engine is live (consistent with the ~1.0-1.2x
marginal cost measured above).

That run changed two things at once — dtype *and* `ORT_DISABLE_ALL`, which the FP16 build
needed to dodge the `com.microsoft.Gelu` failure. Rebuilding the FP32 graphs with the same
session options (`exports/split/trt_cache_nofuse`) isolates the dtype:

| graphs | ORT fusions | RSS | available after |
|---|---|---:|---:|
| FP32 | on | 6.46 GB | 0.29 GB |
| FP32 | off | 6.41 GB | 0.29 GB |
| FP16 | off | **6.07 GB** | **0.56 GB** |

So FP16 weights are worth **~0.34 GB**, and the fusion setting is worth almost nothing for
RSS (6.46 vs 6.41) even though it changes the on-disk engine cache a lot. Halving 3.5 GB of
weights in the file bought a tenth of that in memory.

### FP16 stress run, 300 s (2026-08-13)

`run_pipeline.py --split-dir exports/split_fp16 --duration-s 300`, 769 chunks:

| | avg | p95 | min |
|---|---:|---:|---:|
| chunk | 390.0 ms | 397.9 ms | 376.1 ms |
| — denoise | 327.3 ms | 334.6 ms | |
| — vision | 52.3 ms | 53.8 ms | |
| — text | 8.6 ms | 8.9 ms | |
| — cond | 1.1 ms | 1.1 ms | |

**2.56 Hz replan, peak RSS 5.71 GB, available floor 1.47 GB**, and utterly flat — RSS did
not move off 5.71 GB across the whole run and the p95 stayed within 2% of the mean.

Against the FP32 15-minute run (397.8 ms, 6.73 GB, 0.18 GB free) the FP16 build is
**~8 ms faster per chunk and leaves 1.29 GB more headroom** — and unlike the FP32 run it
never touches swap. This is the build to deploy.

### The floor, and what it means

    ~2.5 GB   fixed CUDA/TRT context   (visible as vision_0's 2.57 GB marginal cost)
    ~3.5 GB   engine + runtime allocations, largely insensitive to ONNX dtype
    --------
    ~6.0 GB   practical floor for X-VLA-0.9B on this board

Nothing available at the ONNX or ORT level moves this much: FP16 weights −0.34 GB, ORT
arena off −0.16 GB, CUDA EP dropped −0.14 GB, and they do not stack meaningfully. **Use the
FP16 build** (`exports/split_fp16`) — it is the lowest measured RSS, the fastest (denoise
366 ms vs 411 ms), and parity-clean — but budget on ~6.1 GB resident, leaving **~1.3 GB**
for the camera reader and control loop.

If more headroom is ever needed, the remaining levers are structural, not numerical: fewer
camera views (already at 1 of 3), a smaller backbone, or running the policy in a separate
process from the controller and paying an IPC hop. `num_denoising_steps` buys latency, not
memory.
