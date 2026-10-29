# vla-onnx

Excavator VLAs on the DGX Spark: native recordings → LeRobot datasets → fine-tune → split
ONNX bundles for the Orin Nano. Orin fit and benchmarks of the base models live in
[jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla); the robot is
`kaivuriprokkis`. This side stops at a parity-checked bundle.

| project | model | runtime on the robot | state |
|---|---|---|---|
| [`xvla/`](xvla/) | X-VLA 0.9 B (LeRobot) | `kaivuriprokkis` branch `python`, the backup stack | live; fine-tune with `run_training.sh`, export with `run_export.sh` |
| [`groot/`](groot/) | GR00T N1.7 3 B | `kaivuriprokkis` native C++ | live; training-time RTC, `groot/excavator/` |
| [`pi05/`](pi05/) | π0.5 3.3 B (OpenPI) | `kaivuriprokkis` native C++ | live; training-time RTC |
| [`kaivuriprokkis/`](kaivuriprokkis/) | π0.5 | native C++ | the π0.5 robot-bundle exporter, plus the RTC training docs shared by GR00T and π0.5. Start at [`RTC_TRAINING.md`](kaivuriprokkis/RTC_TRAINING.md) |

SmolVLA and EVO1 were removed on 2026-10-29. `git checkout archive/lerobot-playbooks-2026-10-29`
has them.

```
recording ─► dataset view ─► fine-tune ─► split export ─► mixed FP16 ─► parity ─► bundle
 (robot)     (vla_common)     (Spark)       (Spark)        (Spark)      (Spark)   └─► ~/bundles
```

## X-VLA

```bash
bash xvla/excavator/setup.sh        # both venvs, base checkpoint, preflight
./run_training.sh --dataset <recording> --cameras cam1,cam2 --chunk 30 --execute 10 --steps 20000
./run_training.sh --dataset <recording> --cameras cam1,cam2 --holdout none      # small dataset: train on all
./run_export.sh  --run xvla/outputs/<sweep>/<run> --step last
./ship_bundle.sh ~/bundles/<bundle> orin
```

Details, the settings that are not X-VLA's defaults, and scoring: [`xvla/README.md`](xvla/README.md).
`eval/` scores held-out episodes (`compare.py`, `curve.py`) and is model-agnostic.

## Shared: `common/` (`vla_common`)

Bundle and dataset plumbing that every project uses: hash a bundle, validate a traced graph,
record provenance, convert weights to mixed FP16, read a safetensors header, reshape a
LeRobot dataset. What is **not** shared is how a model gets cut into graphs; that follows the
architecture and stays in each project.

Bundle tools (from the Orin optimization work in
[jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla/blob/main/docs/07-optimization-playbook.md)):

| | |
|---|---|
| `python -m vla_common.fp16_mixed --bundle B --graphs ... [--half-io REGEX]` | mixed FP16 for a strongly typed TensorRT build, every RMSNorm found by shape and kept FP32; `--half-io` keeps chosen engine-boundary tensors FP16 |
| `python -m vla_common.boundary_report B` | FP32 tensors crossing engines that round-trip FP16→FP32→FP16 and would be exact in FP16 |
| `python -m vla_common.half_boundary --bundle B --names REGEX` | switches those tensors to FP16 in an existing bundle; refuses anything not exact |
| `python -m vla_common.fp32_accumulate --bundle B --graphs ...` | every FP16 MatMul as Cast(FP32) → MatMul → Cast(FP16), which TensorRT runs FP32-accumulating (on the Orin: +1–3 % latency, 30–45 % lower action error) |

The FP16 recipe in `vla_common/fp16_weights.py` is a hard-won constant, not a default: a
blanket cast overflowed a vision tower here (cosine 0.805), so `LayerNormalization` and
`Softmax` stay FP32.

## Datasets: views and variants

`lerobot-train` has no camera flag: every `observation.images.*` in `meta/info.json` becomes
a policy input. `run_training.sh --cameras` therefore builds a **view** of the recording
(symlinked `data/` and the kept videos, copied `meta/`) and refreshes it when the source
grows:

```bash
python -m vla_common.dataset.view --src <recording> --cameras cam1        # resolve, build if needed
python -m vla_common.dataset.camera_variant --src <rec> --dst <view> --keep cam1   # the builder
```

Views live in `datasets/` and are named from what they are (`masi_digging_sand_mix__cam1`).
A grown source is rebuilt automatically; a *shrunk* one is refused, because dropping
episodes renumbers the survivors and invalidates every split derived from an older
checkpoint. Asking for all cameras uses the recording itself, with no view.

`observation.state` goes to the policy whole, so a channel that should not be an input has to
leave the dataset. That produces a **recording**, not a view, and belongs next to its source:

```bash
python -m vla_common.dataset.state_variant --src <rec> --dst <rec>_nogyro --drop slew
```

Slew has no tare, so its yaw means something different each session; the masi recordings
already ship `[lift, tilt, scoop]`. `action` and `feedback.*` keep every channel.

## Environments

| env | lerobot | torch | used by |
|---|---|---|---|
| `.venv-lerobot051` | 0.5.1 | 2.12.0+cu130 | X-VLA fine-tune |
| `.venv-lerobot061` | 0.6.1 | 2.11.0+cu130 | X-VLA export |

lerobot 0.6.1 requires `torch>=2.7,<2.12` while the 0.5.1 stack pins 2.12.0, so these are two
venvs on purpose. `xvla/excavator/setup.sh` builds both from `xvla/requirements.txt` and
`xvla/requirements-export.txt`. `vla_common` is installed into both (`pip install -e common`),
which is why it carries no torch or lerobot pin. GR00T and π0.5 have their own venvs
(`groot/.venv-groot17`, `pi05/third_party/openpi/.venv`).

## Paths

Nothing hardcodes a home directory. Shell scripts source [`paths.sh`](paths.sh); Python
imports `vla_common.paths`. Both resolve from their own location and honour overrides:

| variable | default | what it is |
|---|---|---|
| `VLA_ONNX` | this directory | pipeline root |
| `VLA_DATASETS` | `~/Desktop` | recorded excavator datasets — **source data, the only copy** |
| `VLA_BUNDLES` | `~/bundles` | finished bundles, staged for `ship_bundle.sh` |
| `VENV_LEROBOT051` / `VENV_LEROBOT061` | as above | the two environments |

A recording elsewhere (for example `~/datasets/masi_dual_camera`) is passed as a full path.
