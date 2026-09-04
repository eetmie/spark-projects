# EVO1 checkpoint candidates

Assessment date: 2026-09-03. Hub metadata and weights were read at the immutable
revisions shown below.

## Recommendation

Use [`MINT-SJTU/Evo1_RoboTwin2_clean`](https://huggingface.co/MINT-SJTU/Evo1_RoboTwin2_clean)
as the first pretrained **initializer** for this project. It is the broader checkpoint:
one policy trained on 50 clean RoboTwin 2.0 bimanual manipulation tasks, using 14-D
absolute ALOHA joint actions. The authors report 65.7% aggregate success with 50 flow
steps, a 37-action execution horizon, and Gaussian action smoothing with kernel 9.

Use [`MINT-SJTU/Evo1_SO100`](https://huggingface.co/MINT-SJTU/Evo1_SO100) instead only
when the target is specifically an SO100/SO101-like 6-joint arm with the published
front/side two-camera observation contract. That embodiment match is stronger than
RoboTwin's task diversity for an actual SO100 deployment, but it is a narrower starting
point for a custom robot.

Neither repository is a drop-in `Evo1Policy.from_pretrained()` ID. Both contain an
original MINT/DeepSpeed `mp_rank_00_model_states.pt`, not a LeRobot 0.6.1 policy package
with `model.safetensors` and serialized pre/postprocessors.

| candidate | immutable revision | trained contract | released inference recipe |
|---|---|---|---|
| RoboTwin2 clean | `ce8c583724706fbf7a03c17237761c65bf6813a7` | 50 tasks, bimanual ALOHA, actual state/action width 14, padded width 24 | 50 solver steps, execute 37, Gaussian smoothing 9 |
| SO100 | `cfc2ce796f8f89ae270ad8f15ce48e850024b7ef` | SO100/SO101, actual state/action width 6, padded width 24, two RGB cameras | 10 solver steps, chunk/action horizon 50 |

## LeRobot 0.6.1 compatibility proof

The RoboTwin weight file was downloaded at the pinned revision and loaded with
`torch.load(..., weights_only=True, mmap=True)`. Its `checkpoint["module"]` contains
632 tensors and 776,139,440 elements. The 116 action-head tensors already have exactly
the LeRobot 0.6.1 names and shapes.

The remaining difference is the migration from the original remote-code InternVL3
module to Transformers' native InternVL module:

- `language_model.model.*` becomes `language_model.*`;
- `vision_model` becomes `vision_tower`, with deterministic submodule renames;
- each of 24 packed vision `qkv` weights and biases splits into separate Q/K/V tensors;
- the old `mlp1` projector becomes `multi_modal_projector`.

[`tools/convert_mint_checkpoint.py`](../tools/convert_mint_checkpoint.py) implements
only these fail-closed transformations. On the official RoboTwin checkpoint it produces
728 tensors with all 776,139,440 elements preserved. Comparison with an instantiated
LeRobot 0.6.1 `Evo1Policy` found 728/728 keys, no missing or unexpected keys, and no
shape mismatches. A serialized 1,554,263,024-byte safetensors file also passed a strict
`policy.load_state_dict(..., strict=True)` load.

Dry-run the conversion audit:

```bash
.venv/bin/python tools/convert_mint_checkpoint.py \
  models/candidates/Evo1_RoboTwin2_clean/mp_rank_00_model_states.pt
```

Create the ignored local weight file:

```bash
.venv/bin/python tools/convert_mint_checkpoint.py \
  models/candidates/Evo1_RoboTwin2_clean/mp_rank_00_model_states.pt \
  --output models/candidates/Evo1_RoboTwin2_clean/lerobot-0.6.1/model.safetensors
```

The conversion is structural proof, not deployment parity. Before using the trained
checkpoint, the next gate is to package the selected camera/action features and its
normalization statistics as LeRobot 0.6.1 processors, then compare original-MINT and
converted-LeRobot outputs on the same observation. RoboTwin's per-task normalization
keys also need an explicit target-task choice or replacement statistics from target
data. Only then should the trained policy replace the nondeployable random-head export.

Primary references: the
[`evo1-flash` source and training recipes](https://github.com/MINT-SJTU/Evo-1/tree/evo1-flash),
the [RoboTwin evaluation recipe](https://github.com/MINT-SJTU/Evo-1/blob/evo1-flash/RoboTwin_evaluation/README.md),
and the [SO100 conversion instructions](https://github.com/MINT-SJTU/Evo-1/tree/evo1-flash#-5-inference-in-lerobot-so100so101).
