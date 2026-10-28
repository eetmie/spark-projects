# Native OpenPI pi0.5 training and EMA on DGX Spark

Research date: 2026-10-28. Repository baseline: `f908e4e`; OpenPI pin recorded in
[RTC_TRAINING.md](RTC_TRAINING.md): `c23745b`.

Preferred hardware is one DGX Spark. The fallback is two A100 40 GB GPUs. Keep the
native OpenPI JAX trainer and full fine-tuning for the first EMA experiment.
**Full FP32 EMA with the existing params-only saver is worth testing on Spark
before changing to LoRA or moving machines.**

**Result (2026-10-28): it fits at batch 32 with JAX's `cuda_async` allocator** (72 GB device,
>= 14 GB free through saves, 17.5 s/step, no step-time cost). The default BFC allocator could not
save. EMA is now on by default. Measurements: [RTC_TRAINING.md](RTC_TRAINING.md), "What trains,
and the memory budget".

## Public training reports

| Source | Relevant evidence | Limits |
|---|---|---|
| [Robokan OpenPI Spark guide](https://github.com/Robokan/openpi/blob/main/DGX_SPARK_TRAINING.md) | Dedicated pi0.5/OpenArm recipe using native JAX, LoRA, batch 2, and EMA off. Reports about 100–105 GB during training and checkpoint-save memory pressure. | A community recipe, without a controlled policy evaluation. Its conclusion that full fine-tuning cannot fit applies to its configuration; this repository already records full training on Spark. |
| [BEHAVIOR-1K issue 1910](https://github.com/StanfordVL/BEHAVIOR-1K/issues/1910) | On 2025-11-18, a Spark owner explicitly reported training models including pi05 through LeRobot. | Firsthand report without training configuration, memory measurements, or resulting policy evaluation. |
| [Kabilankb GB10 training report](https://medium.com/@kabilankb2003/fine-tuned-a-3-5b-parameter-robot-brain-on-an-edge-device-in-under-an-hour-a5d29b349e20) | Reports LeRobot pi0 fine-tuning at about 9 s/step and 20% simulation success after 400 steps. | This is pi0, not pi0.5. Only the public preview was accessible; its OpenPI speed comparison is not a matched benchmark. |
| [NVIDIA forum Spark report](https://forums.developer.nvidia.com/t/questions-about-openpi-on-jetson-thor-tutorial/368766) | pi0.5 inference works on Spark using the Thor tutorial. | Inference only; it does not establish training feasibility. |

No independently reproducible public example combining **Spark + full pi0.5
fine-tuning + EMA + evaluated robot performance** was found in this search.
That is a limit of the available evidence, not proof nobody has done it.

This project's own [recorded Spark results](RTC_TRAINING.md) are more useful for
the immediate decision: full-model RTC training saved a batch-4 smoke checkpoint,
and the full dataset ran at about 16 s/step at batch 32 with EMA off. That is about
44.4 hours for 10k steps or 5.6 days for 30k, before additional overhead. These are
existing repository measurements and timing projections, not new EMA benchmarks.

## Native JAX memory requirements

[OpenPI's published estimates](https://github.com/Physical-Intelligence/openpi/blob/c23745b/README.md)
are more than 22.5 GB for LoRA and more than 70 GB for full fine-tuning. They are
general estimates rather than guarantees for a particular batch, camera count,
or action horizon. In particular, the large LIBERO training preset is not a
small-GPU starting point: its batch is 256, versus 32 in the excavator config.

Native JAX uses FP32 trainable weights and gradients with mostly BF16 computation.
For approximately 3.35 billion parameters, tensor-size arithmetic gives:

| Allocation | Decimal GB | GiB |
|---|---:|---:|
| FP32 live weights | 13.4 | 12.5 |
| Adam first and second moments | 26.8 | 25.0 |
| Persistent weights and Adam, EMA off | **40.2** | **37.5** |
| Additional FP32 EMA copy | **13.4** | **12.5** |
| Persistent weights, Adam, and EMA | **53.6** | **49.9** |
| FP32 gradients when materialized | another 13.4 | another 12.5 |

These are estimates of tensor storage, not measured GPU peaks. Activations,
compiler workspace, temporary arrays, allocator reservations, data loading, and
checkpoint staging are additional. XLA can reuse or fuse buffers, so the table
does not imply every transient allocation exists simultaneously. Compare all
measurements in consistent units; existing notes use approximate GB labels.

Sources: [native training state and EMA update](https://github.com/Physical-Intelligence/openpi/blob/c23745b/scripts/train.py),
[optimizer](https://github.com/Physical-Intelligence/openpi/blob/c23745b/src/openpi/training/optimizer.py).

## EMA and the existing params-only saver

[`config.py`](../pi05/excavator/config.py) explicitly sets `ema_decay=None`.
[`run.py`](../pi05/excavator/run.py) already selects the weights through
OpenPI's `_split_params(state)`. With EMA enabled, that function selects
`state.ema_params` for the inference checkpoint.

**Enabling EMA adds a training-memory copy, but does not add a second saved model
to the params-only checkpoint.** Approximate tensor payloads are:

| Checkpoint contents | EMA off | EMA on |
|---|---:|---:|
| Params-only inference weights | 13.4 GB live weights | 13.4 GB EMA weights |
| Full resumable state, including inference params | 40.2 GB | 53.6 GB |

Metadata and serialization overhead are excluded. Full state must retain live
weights, optimizer moments, and EMA history. Params-only mode cannot perform a
true `--resume`; loading those weights starts a new optimizer and EMA history.

The runner estimates about 80 GB for training without EMA. If that measurement
and its units remain comparable, adding EMA suggests about 93 GB for training,
then about 107 GB while staging one model for a params-only save, before other
overhead. This is a rough budget hypothesis. Initialization, compilation,
allocator retention, and save overlap can still make it fail. A lower batch can
reduce activation memory, but does not shrink weights, Adam, or EMA.

The current config comment saying EMA cannot fit should therefore be read as a
result for the earlier memory/save setup, not a universal limit. Source:
[OpenPI checkpoint split and restore](https://github.com/Physical-Intelligence/openpi/blob/c23745b/src/openpi/training/checkpoints.py).

## Orbax staging is the next checkpoint improvement

The runner sets `save_concurrent_gb=8`. Orbax documents this as a limit on
concurrent disk writes; it does **not** bound the whole device-to-host staging
copy. Setting it to 8 does not guarantee an 8 GB checkpoint-memory peak.

Newer Orbax versions expose `save_device_host_concurrent_gb`. This also limits
device-to-host transfers, waiting for array writes before staging more data.
It may reduce the save peak enough to make full resumable EMA checkpoints
practical, at the cost of more blocking during saves. This is an experiment,
not an established Spark result.

OpenPI's pin uses `orbax-checkpoint==0.11.13`; that handler does not have the newer
argument. Check the installed version/signature on Spark before choosing a
backport or a tested dependency upgrade. Do not assume latest Orbax is compatible
with the pinned JAX/Flax stack. Preserve the existing streaming limit and leave
room for the largest array, which the runner identifies as about 4.8 GB FP32.

Sources: [current Orbax handler API](https://orbax.readthedocs.io/en/latest/api_reference/checkpoint.handlers.html),
[OpenPI dependency pin](https://github.com/Physical-Intelligence/openpi/blob/c23745b/pyproject.toml),
[Orbax 0.11.13 handler](https://github.com/google/orbax/blob/v0.11.13/checkpoint/orbax/checkpoint/_src/handlers/pytree_checkpoint_handler.py).

## Spark and the two A100s

| Hardware | Starting configuration | Memory implications |
|---|---|---|
| One Spark | Full native JAX fine-tune, FP32 EMA, batch 8, two workers, params-only saves | Preferred first experiment. CPU, GPU, workers, and checkpoint staging compete for one physical memory pool. |
| Two A100 40 GB GPUs in one host | Native JAX with `--fsdp-devices 2`, global batch 8–16 | Large weights, Adam arrays, and EMA arrays are sharded. Persistent state with EMA is approximately 26.8 GB per GPU, before gradients, activations, collectives, and replicated arrays. |

With native `fsdp_devices=1`, two visible GPUs replicate training state instead
of dividing it. Set it explicitly to 2. The implementation replicates vectors
and small arrays, and shards eligible large arrays. Two 40 GB cards are therefore
a plausible fallback, not a guaranteed equivalent of one unrestricted 80 GB GPU.
The estimate assumes both GPUs are visible to one JAX process on the same host;
separate hosts need distributed initialization work. Interconnect performance
also affects training speed.

Sources: [native sharding implementation](https://github.com/Physical-Intelligence/openpi/blob/c23745b/src/openpi/training/sharding.py),
[FSDP configuration](https://github.com/Physical-Intelligence/openpi/blob/c23745b/src/openpi/training/config.py).

## First Spark experiment

Run this when the current GPU job has finished. From the repository root:

```bash
cd vla-onnx/pi05

# Needed once if this config and dataset do not already have normalization stats.
third_party/openpi/.venv/bin/python excavator/run.py norm-stats \
  --config-name pi05_excavator_rtc

OPENPI_SAVE_OPT_STATE=0 OPENPI_SAVE_CONCURRENT_GB=8 \
third_party/openpi/.venv/bin/python excavator/run.py train \
  pi05_excavator_rtc \
  --exp-name ema099-b8-probe \
  --ema-decay 0.99 \
  --batch-size 8 \
  --num-workers 2 \
  --num-train-steps 100 \
  --save-interval 50
```

Use a fresh experiment name if the directory already exists. This is a memory
and checkpoint probe, not a policy-quality run; its 100 steps do not complete
the existing 1,000-step learning-rate warmup. The same flags can probe the plain
`pi05_excavator` config after computing its own stats.

1. Record installed JAX, jaxlib, Flax, Orbax, CUDA, and driver versions, plus
   config, batch size, worker count, and dataset.
2. Record free system memory, GPU allocated/reserved memory where available,
   and step time during initialization, compilation, steady training, both
   saves, and after save completion. Use `/proc/meminfo` or `free -b` for system
   headroom. Do not blindly add process RSS and GPU figures: the physical pool
   is shared and some mappings may overlap.
3. Wait for checkpoint writes to finish and load a saved checkpoint with the
   same RTC config. Check finite outputs, action shapes, and normalization
   assets. Confirm the saved weights correspond to EMA; loss alone cannot
   establish this.
4. If stable with comfortable system headroom, repeat at batch 16, then 32.
   Keep the horizon and camera configuration fixed during this comparison.
5. Investigate bounded Orbax staging separately if optimizer-state resume is
   required. Validate save, process restart, restore, and another optimizer/EMA
   update before trusting a long resumable run.

For a quality comparison, hold out whole takes, including both rock and sand,
and compare live versus EMA weights at matched training exposure. The present
params-only saver exports EMA alone when enabled, so retaining both versions for
that comparison needs an explicit additional export. Reducing batch changes
examples processed per optimizer step and EMA's smoothing span in examples.

## Other findings that affect implementation

- **Keep native JAX for this work.** Upstream OpenPI's PyTorch implementation
  lists EMA and FSDP as unsupported. Its BF16 memory/performance reports are
  not direct measurements of native JAX training. See the
  [upstream support matrix](https://github.com/Physical-Intelligence/openpi/blob/c23745b/README.md).
- **The frozen-parameter EMA optimization is useful only if weights are frozen.**
  [OpenPI PR 1055](https://github.com/Physical-Intelligence/openpi/pull/1055) was
  open and unmerged on the research date. It averages trainable parameters only
  and reconstructs the full inference tree on save. This can help LoRA or
  frozen-backbone training, but leaves full fine-tuning unchanged.
- **Native Gemma already rematerializes transformer blocks.** Do not assume a
  PyTorch `gradient_checkpointing` flag is a missing switch in this JAX trainer.
  See [Gemma's existing `nn.remat`](https://github.com/Physical-Intelligence/openpi/blob/c23745b/src/openpi/models/gemma.py).
- **CPU offload does not create a second memory pool on Spark.** Moving an EMA
  copy to CPU still consumes the shared physical memory. A BF16 EMA is a
  numerical-method change, not just a storage optimization.
- **EMA decay is an experimental choice.** `TrainConfig` defaults to 0.99;
  the stock pi05 LIBERO preset uses 0.999. Start with 0.99, keep the accumulation
  in FP32, and evaluate it. Smoothed weights do not guarantee smoother robot
  commands. See [upstream configs](https://github.com/Physical-Intelligence/openpi/blob/c23745b/src/openpi/training/config.py).
- **Keep the existing Spark allocator settings as the first baseline.**
  [`spark_env.py`](../pi05/excavator/spark_env.py) disables Triton GEMMs and
  preallocation. A memory fraction is not an overall system-memory limit;
  [JAX's allocator documentation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)
  explains the allocation controls. Increasing the fraction cannot reserve
  headroom for CPU workers or checkpoint copies.

The immediate decision is to test FP32 EMA with params-only saves on Spark.
If it fits, checkpoint resumability can be addressed independently. If it does
not fit at a useful batch, try the two A100s with native FSDP before changing the
training objective or which parameters are trained.
