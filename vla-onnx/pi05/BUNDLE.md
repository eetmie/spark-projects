# pi05_openpi_state bundle (fine-tuned OpenPI pi0.5, joint angles in the prompt)

Written by `vla-onnx/pi05` for `kaivuri-inference`. Same split layout as the LIBERO compact
bundle (`jetson-orin-nano-vla/export/pi05`: templates, per-layer FP16 weights, embedding,
RoPE, 10-step conditioning, fixtures), plus what the per-step state prompt needs.

## bundle.json

| Field | Value |
|---|---|
| `schema` | `pi05-split/1` |
| `policy` | `pi05_openpi_state` |
| `layout` | `compact`: 2 valid cameras (512 image tokens), then the prompt's real tokens, zero tail |
| `prefix_tokens` | `512 + prompt_capacity` |
| `prompt_capacity` | longest task head + 4 tokens per state value + tail |
| `action_horizon` / `action_dim` / `robot_action_dim` | 50 / 32 / 4 (6 with tracks) |
| `state_dim` | 3: `[lift, tilt, scoop]` in degrees |
| `num_steps` | 10 |
| `image_keys` | `base_0_rgb` = cam1, `left_wrist_0_rgb` = cam2, `right_wrist_0_rgb` masked |
| `tokenizer` | `{"sep": id("▁"), "neg": id("▁-"), "pieces": {"0".."9": id}, "tail": ids(";\nAction: ")}` |
| `tasks` | `{task: {"head": ids("<bos>Task: <task>, State:")}}` |
| `kaivuri` | runtime contract from `kaivuriprokkis/native_stack/notes/PI05_FINETUNE.md` (fps, task order, names, cameras) |
| `language_residual` | `float32`: pi05_base-derived weights overflow FP16 there. `vision_tail` outputs FP32, `language` (00–16) takes/returns FP32 hidden, the last layer is `language_kv` (K/V only). Prompt rows: `embed_fp16 * sqrt(2048)` rounded to FP16, then FP32. Same layout as jetson-orin-nano-vla's `pi05_lerobot` |

Run on native TensorRT attention (one plan per kind, `build_pi05_cache.py`); the AOT attention
kernel is LIBERO's 10 rows / 521 prefix only.

## Per-step prompt

```
norm = (state - q01) / (q99 - q01 + 1e-6) * 2 - 1        # norm_stats.json state, float64
bin  = digitize(norm, linspace(-1, 1, 257)[:-1]) - 1     # -1 .. 255; -1 below -1
ids  = head + for each bin: [neg if bin < 0 else sep] + pieces[d] for d in str(|bin|) + tail
```

Exporter verifies `compose(head, bins) == sentencepiece("<bos>Task: T, State: b0 b1 b2;\nAction: ")`
for every bin value and random vectors, per task.

## Actions

`robot = (x[:, :4] + 1) / 2 * (q99 - q01 + 1e-6) + q01` with `norm_stats.json` actions.
Order `[slew, lift, tilt, scoop]`, already in the controller's `[-1, 1]` command units.

## Fixtures

`fixtures/fixture_<i>.npz` from real dataset frames: raw uint8 cam1/cam2, raw state, task,
noise `(50, 32)`, token ids/mask, FP16 split reference and mixed-precision OpenPI reference.
