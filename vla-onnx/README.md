# vla-onnx

One pipeline, three models. Fine-tune a LeRobot VLA on the DGX Spark, cut it into split
ONNX graphs that a Jetson Orin Nano can actually build TensorRT engines for, verify the
export against PyTorch, and ship a bundle.

```
checkpoint ──► fine-tune ──► split export ──► mixed FP16 ──► manifest ──► parity ──► bundle
               (Spark)       (Spark)          (Spark)        (Spark)      (Spark)     └─► ~/bundles
```

Running the bundle is somebody else's job: base-model fit and benchmarks live in
[jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla), the robot lives in
`kaivuriprokkis`. This side stops at a parity-checked bundle.

| playbook | model | state |
|---|---|---|
| [`smolvla/`](smolvla/) | SmolVLA 450 M | fine-tuned, deployed, driven the machine |
| [`xvla/`](xvla/) | X-VLA 0.9 B | fine-tuned, exported, benchmarked |
| [`evo1/`](evo1/) | EVO1 775 M | export + Orin bootstrap proven; **no trained checkpoint yet** |

## What is shared, and what is deliberately not

[`common/`](common/) (`vla_common`) holds what all three do *identically*: hash a bundle,
validate a traced graph, record provenance, convert weights to mixed FP16, read a
safetensors header, reshape a LeRobot dataset.

What is **not** shared is how a model gets cut into graphs. `_build_wrappers()` is ~200
lines in each playbook and about 5% similar between them, because the cut follows the
architecture — SmolVLA's VLM + expert prefill/decode, X-VLA's DaViT + BART + denoiser,
EVO1's InternVL. That is the actual content of an exporter, and merging it would mean
one file with three unrelated halves. The pipeline is shared; the models are not.

The FP16 recipe in `vla_common/fp16_weights.py` is a hard-won constant, not a default:
a blanket cast overflowed SmolVLA's vision tower here (cosine 0.805), so
`LayerNormalization` and `Softmax` stay FP32.

## Two environments, and why they cannot be one

| env | lerobot | torch | used by |
|---|---|---|---|
| `.venv-lerobot051` | 0.5.1 | 2.12.0+cu130 | smolvla fine-tune, **xvla fine-tune** |
| `.venv-lerobot061` | 0.6.1 | 2.11.0+cu130 | evo1 everything, **xvla export** |

lerobot 0.6.1 requires `torch>=2.7,<2.12`; the 0.5.1 stack pins 2.12.0. They are mutually
exclusive, so this is two venvs on purpose and not a cleanup that was missed.

SmolVLA and X-VLA fine-tune in the *same* venv deliberately — an identical stack is what
makes "X-VLA is ~20% behind SmolVLA on this data" a statement about the models rather
than about two dependency trees.

`vla_common` is installed into both (`pip install -e common`), which is why it carries no
torch or lerobot pin and imports nothing version-sensitive at module scope.

## Paths

Nothing hardcodes a home directory. Shell scripts source [`paths.sh`](paths.sh); Python
imports `vla_common.paths`. Both resolve from their own location and honour overrides:

| variable | default | what it is |
|---|---|---|
| `VLA_ONNX` | this directory | pipeline root |
| `VLA_DATASETS` | `~/Desktop` | recorded excavator datasets — **source data, the only copy** |
| `VLA_BUNDLES` | `~/bundles` | finished bundles, staged for `ship_bundle.sh` |
| `VENV_LEROBOT051` / `VENV_LEROBOT061` | as above | the two environments |

This matters because the tree has already been renamed twice, and each time a crop of
absolute paths went stale silently — a script would still run, and write where nobody
was reading.

## Not in git

`models/`, `outputs/`, `exports/`, `datasets/`, `.venv-*`. Checkpoints and exports are
regenerable from a recipe plus a checkpoint; the recorded datasets are not regenerable at
all and live outside the repo under `VLA_DATASETS`.
