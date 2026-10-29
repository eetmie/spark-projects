# vla-onnx/xvla — X-VLA-0.9B fine-tune on the DGX Spark (GB10)

Fine-tune `lerobot/xvla-base` on an excavator recording, then export it as a split-ONNX
bundle for the Orin Nano. Fit on the 8 GB Orin is settled in
[`notes/fit-on-8gb.md`](notes/fit-on-8gb.md): 12 split TRT engines, 390 ms/chunk, 5.71 GB
peak RSS. The stock checkpoint cannot drive an excavator (it is a 20-dim `ee6d` arm
policy), hence this fine-tune.

Training goes through `../run_training.sh`, export through `../run_export.sh`; this
directory holds the X-VLA-specific parts (checkpoint prep, exporter, parity).

## Run

```bash
bash excavator/setup.sh                      # idempotent: venv, checkpoint, preflight

../run_training.sh --dataset <recording> --cameras cam1,cam2 --smoke   # 2 steps

../run_training.sh --dataset <recording> --cameras cam1,cam2 \
    --chunk 30 --execute 10 --steps 20000
```

`<recording>` is a name under `$VLA_DATASETS` (default `~/Desktop`) or a full path.
`--train-mode` defaults to `full` (879 M trainable); `frozen` (311 M) scored 14-22 % worse
held-out on two recordings and drove the machine badly. `--dry-run` prints the resolved
`lerobot-train` command.

### Small dataset: train on everything

`--holdout` defaults to `every10@5` (episodes 5, 15, 25, … held out). Pass `--holdout none`
to train on all episodes:

```bash
../run_training.sh --dataset <recording> --cameras cam1,cam2 \
    --chunk 30 --execute 10 --steps 20000 --holdout none
```

Without held-out episodes there is no curve, so there is no "best" step:
`eval/curve.py` and `eval/compare.py` have nothing to score, and `run_export.sh`'s default
`--step best` exports nothing (by design). Pick the step yourself:

```bash
../run_export.sh --run xvla/outputs/<sweep>/<run> --step last      # or --step 17500
```

The export parity gate compares the bundle against the torch checkpoint and does not need
held-out data. For reference, the bundle that drove well on the machine was step 17500 of a
20000-step full fine-tune.

### Two cameras

Camera keys and their order come from the checkpoint's `observation.images.*`; the bundle
records them and the Orin runtime refuses a mismatch. The exporter sizes the vision engine
to the checkpoint's camera count (`--views` overrides), so a 2-camera fine-tune exports at 2
views. A 1-view bundle cannot serve a 2-view runtime, and the reverse. No 2-camera X-VLA
bundle has been exported or run on the Orin yet; treat the first one as the test.

## The stock checkpoint cannot train on this data unmodified

`lerobot/xvla-base`'s config declares the robots it was pretrained on:
`observation.images.image{,2,3}` and `observation.state (8,)`. `make_policy` only takes
`input_features` from the dataset when the config leaves them empty, so training dies with
a feature mismatch. `--rename_map` suppresses the error but is the wrong fix: it also skips
the consistency check and builds the normalizer with state shape `(8,)` against dataset
stats of shape `(3,)`, silently.

`excavator/prepare_checkpoint.py` derives `models/xvla-base-excavator` from the base: same
weights (symlinked), `input_features` emptied, so the dataset defines the contract. Nothing
in the weights depends on the emptied keys: state is padded to `max_state_dim` (20), and
`num_image_views` stays 3 with unused views zero-padded and masked.

## Settings that are not X-VLA's defaults

All argued in the `xvla` arm of `../run_training.sh`:

- **`action_mode=auto`** — the default `ee6d` is a 20-dim arm space. `auto` reads the
  dataset's real action dim, pads to 20 so the pretrained head loads, and computes loss on
  the real dims only.
- **`STATE=MEAN_STD`** — `xvla-base` ships `STATE=IDENTITY` and X-VLA does not normalize
  proprio itself, so joint angles in degrees would go in raw. ACTION stays `MEAN_STD`, which
  is what the pretrained head was trained with.
- **`dtype=bfloat16`** — the model card's setting; native on GB10.
- **`chunk_size=30`** — what the base checkpoint and the Orin runtime use. Chunk length is
  part of the deployment contract (it sets the denoise sequence length and X-VLA cannot
  KV-cache across denoise steps), so re-measure latency before quoting a replan rate at any
  other value.
- **`domain_id=0`** — matches the exporter's `--domain-id 0`.

## Throughput

Size a run from a measurement, not from this table. On GB10, X-VLA bf16 at batch 32 /
chunk 30 measured 1.22-1.26 s/step on a single camera; the GPU heat-soaks to ~79 °C and
~2457/3003 MHz under sustained load, so use a heat-soaked figure for overnight runs. More
cameras mean more vision tokens; probe your own configuration:

```bash
../run_training.sh --dataset <recording> --cameras cam1,cam2 \
    --chunk 30 --steps 250 --holdout none --out outputs/probe
grep -oE "updt_s:[0-9.]+" outputs/probe/logs/*.log | tail
```

Epochs per run: `steps × batch ÷ frames`. 20000 × 32 on the 61,777-frame dual-camera
recording is about 10 epochs.

## Scoring (needs held-out episodes)

```bash
../.venv-lerobot051/bin/python ../eval/compare.py --horizons 0.5 --ckpt xvla=<run>/checkpoints/last
../.venv-lerobot051/bin/python ../eval/curve.py --sweep outputs/<sweep>
```

The eval harness is model-agnostic and reads the split from each checkpoint's own
`train_config.json`. In a table with more than one architecture `--n-draws` defaults to 4:
X-VLA gains ~18.5 % from averaging flow-matching draws and SmolVLA ~0 %, so scoring at 1
draw would charge X-VLA for sampler variance SmolVLA does not have.

## Base checkpoint

Pinned to Hub revision `cdb7964e4fe842935d671bfab5a5ebe00a96648c`, verified with
`hf cache verify`, and fetched with curl rather than `snapshot_download`, which stalled
silently twice partway through the safetensors (process alive, file not growing, no
exception). `curl --speed-limit/--speed-time` turns a stall into an error that `--retry`
can act on.

## Environments

Trains in `.venv-lerobot051` (lerobot 0.5.1), exports in `.venv-lerobot061` (0.6.1);
`run_export.sh` picks the right one. They cannot be merged — see `../README.md`.

## Status

- Full fine-tune is the deployed recipe; X-VLA drives the excavator on single-camera data
  and on the new-IMU recordings.
- Bundle contract (`bundle_contract.py`) is vendored byte-for-byte into
  `kaivuriprokkis-python` (`runtime/vendor/xvla_bundle_contract.py`, SHA recorded in its header).
  Re-sync it there when this file changes.
- Open: a 2-camera export and Orin run (above).
