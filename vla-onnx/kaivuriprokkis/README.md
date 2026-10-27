# vla-onnx/kaivuriprokkis — pi0.5 exporter for the excavator

Fine-tuned OpenPI pi0.5 checkpoint → the bundle `kaivuri-inference` runs on the Orin Nano
(kaivuriprokkis `native_stack`, policy `pi05_openpi_state`). This is the project's own
pipeline. `jetson-orin-nano-vla` stays the public reference repo: files taken from it
are vendored here with their source named, and that repo is not changed for this one.

```bash
./export.sh ../pi05/checkpoints/pi05_excavator/<exp>/<step> runs/<name>
./export.sh <step> runs/<name> --converted <earlier conversion>        # skip JAX -> PyTorch
./export.sh <step> runs/<name> -- --row-time --rtc-rows 15             # flags for container/export.py
python host/compare.py runs/<name> <other bundle of the same checkpoint>
python -m unittest discover -s tests                                   # RTC oracle, CPU only
```

`export.sh` needs the OpenPI venv and the dataset from `vla-onnx/pi05` (its README), plus
Docker with the NVIDIA toolkit. On the Spark (GB10) the smoke9 export takes 3.5 minutes with
`--converted`.

## Steps

| step | where | what |
|---|---|---|
| `pi05/excavator/fixture_frames.py` | OpenPI venv | 5 real dataset frames, task order, camera/profile provenance |
| `host/action_stats.py` | OpenPI venv | mean and covariance of normalized 50-row action windows (RTC) |
| `pi05/export/convert.py` | container | JAX → PyTorch (skipped with `--converted`) |
| `container/export.py` | container | everything below, one model load |

`container/export.py`:

1. **Stock references.** The OpenPI policy in **FP32** on CUDA (its own transforms,
   normalization, tokenizer and Euler loop) on the real frames, noise seeded `1000 + i`.
   The FP16 split must stay within cosine 0.999 of it. The older `vla-onnx/pi05/export`
   stages used OpenPI's mixed BF16 policy instead. This exporter follows jetson-orin-nano-vla's
   custom-checkpoint exporter, because BF16 alone drifts from FP32 more than the FP16 split does.
2. **FP16 split model** in the compact layout of `vla-onnx/pi05/BUNDLE.md`:
   - two valid cameras, then the prompt's real tokens, then a zero tail;
   - an FP32 language residual stream;
   - a K/V-only last language layer.

   The exporter checks that cached and live AdaRMS modulation agree bit-exactly, and that
   the board-composed state prompt equals OpenPI's tokenization on every fixture.
3. **Templates and weights.** These cover:
   - one ONNX template per layer kind, with per-layer FP16 weights for refit;
   - the conditioning table, RoPE tables, embedding and norm stats;
   - `MANIFEST.sha256` over every file.
4. **RTC artifacts** (below).

The result loads in today's runtime unchanged.

**Checked 2026-10-23 on smoke9.** These results come from `host/compare.py` against the
earlier `vla-onnx/pi05` bundle:

- weights, templates, tables, fixture inputs and FP16 references are byte-identical;
- only the stock references differ, because they are FP32 now; FP16 vs FP32 stock is
  cosine ≥ 0.99985 and max 0.029.

The template sha256s are unchanged, so the board's existing engine cache applies.

On pikkumasi2tb (kaivuriprokkis main a631ff5, TRT 10.16):

- `--validate-only` passes at the board gate; worst cosine 0.99997, max 0.0132.
- A 120 s `--drive --allow-live --enable-slew --rate-hz 2` run used real IMUs and both IR
  cameras, with pump and servos unpowered. It made 240/240 plans and 956 renewals with
  0 overruns. Inference p50/p99 was 437.3/438.6 ms and camera-to-plan p50 452.6 ms.

## RTC (chunk-to-chunk consistency)

The work plan is kaivuriprokkis `native_stack/notes/RTC_PLAN.md` (branch `feat/pi05-rtc-plan`).
Training side (training-time RTC for pi0.5 and GR00T N1.7): [RTC_TRAINING.md](RTC_TRAINING.md).
This exporter covers its P5 export side, plus the CPU oracle of P1 in Python.

- `rtc.py`: a NumPy reference of both modes, which the C++ host library and device op must
  match.
  - **Inpaint (Mode A).** A line-by-line port of the BEHAVIOR-1K / LeHome step loop. Anchor
    rows are held on the flow path, `(1-t)·x0 + t·z`, while `t > t_ip`. A correlation-based
    correction is pushed to the free rows, `C = Σ_UO Σ_OO⁻¹`. Σ is the correlation matrix,
    shrunk to `β·Σ + (1-β)·I` with β = 0.5. The push is skipped for a step when its max |corr|
    exceeds 1.
  - **Clean (Mode C).** Prefix rows are fixed to the anchor and get the t = 0 style.
  - It also holds the OpenPI float32 time grid (bit-equal to `conditioning_manifest.json`),
    anchor alignment and join metrics.
- `kaivuri.rtc` in `bundle.json`, plus these files:

| file | |
|---|---|
| `rtc/action_covariance_f64.npy`, `rtc/action_mean_f64.npy` | `[H·D, H·D]` / `[H·D]` of normalized training-action windows, row-major `(row, dim)`; `H`, `D`, fps and the norm-stats hash are recorded next to them |
| `conditioning_clean_fp16.npy` | `[37, 3072]` AdaRMS style at t = 0 for Mode C prefix rows |
| `rtc/rtc_fixture_<i>.npz` | parity reference for the runtime's Mode A (and C) |

  Each RTC fixture holds: the previous chunk, `elapsed`, `rows`, the noise `z`, the anchor, `C`,
  the per-step `alpha`/`beta`/`gate`, and the FP16 split chunk for each mode. For each mode the
  oracle blends on the host between the FP16 steps. The fixture also records which steps were
  corrected. The "previous chunk" is the fixture's own FP16 output with another noise seed. That
  is a stand-in until replayed recordings exist.

  smoke9 measurements (d = 15, elapsed 15, switch at row 14, t_ip 0.3). The switch jump is in
  normalized units:

  | fixture | off | inpaint | median row-to-row step |
  |---|---|---|---|
  | 0 | 0.014 | 0.014 | 0.008 |
  | 1 | 0.42 | 0.23 | 0.003 |
  | 2 | 0.76 | 0.42 | 0.003 |
  | 3 | 1.05 | 0.34 | 0.016 |
  | 4 | 1.01 | 0.39 | 0.007 |

  Inpainting removes 45–68 % of the jump in fixtures 1–4. The rest comes from the last three
  ungated steps (t ≤ t_ip), where the model moves the anchored rows freely. A smoke model on
  8 episodes says little about a real fine-tune.
- `--row-time` traces the `action` and `action_output` templates with modulation
  `[1, H, 3072]` instead of `[1, 1, 3072]`. The exporter checks, bit-exactly, that these give the
  same output as the broadcast form when every row carries the same style. Such a bundle needs
  the runtime's P4 first, so the flag is off by default. It only helps a model fine-tuned with
  per-row time.
- `noise: "iid"`: OpenPI trains with i.i.d. noise. Correlated noise (`eps = L·z`) would be a
  training-side change. If it is adopted, `host/action_stats.py` is where Σ comes from.

## On the board (pikkumasi2tb)

From `~/kaivuriprokkis-export-test/native_stack` (a clone of kaivuriprokkis main):

```bash
python3 tools/build_pi05_cache.py --bundle ~/bundles/B --out ~/bundles/B.cache      # ~3 min, new templates only
~/venvs/openpi-lerobot/bin/python -B tools/prepare_pi05.py --bundle ~/bundles/B --out ~/bundles/B-fixtures
./build/kaivuri-inference --bundle ~/bundles/B --cache ~/bundles/B.cache --fixtures ~/bundles/B-fixtures \
  --out results/B-validate.json --validate-only --parity-cosine 0.9998 --parity-max-abs 0.02
```

A bundle whose templates are byte-identical to an existing one can reuse that cache, because
the runtime checks each template's sha256.
