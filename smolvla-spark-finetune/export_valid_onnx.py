"""
Export LeRobot SmolVLA to a validity-first ONNX graph.

This script is intentionally conservative for the GB10 -> Jetson Orin Nano
workflow:
  - load the official LeRobot SmolVLA policy
  - force export math/weights to float32 so ONNX Runtime accepts the graph
  - keep dynamic batch, fixed image/token/action shapes
  - validate the resulting ONNX with onnx.checker and ORT CPU session creation
  - SAVE THE TOKENIZER next to the ONNX, and bundle the normalization stats, so the
    Orin gets a vocab-exact, self-contained deploy bundle (no "guess the backbone").

On Jetson the ONNX runs through ONNX Runtime's TensorRT execution provider with
**FP16** (the Orin builds + caches the TensorRT engine itself per-subgraph — no
trtexec, no separate .engine build). FP16 (not bfloat16) is the right Orin dtype:
Orin (compute 8.7) has no fast BF16, and Orin TensorRT dislikes bfloat16 inputs
around Conv/MatMul. The graph stays FP32 here; the TRT-EP lowers what's safe to FP16
and keeps layernorm/sensitive ops in FP32.

If on-Orin FP16 parity ever fails (non-finite / cosine drop from the vision tower's
`inf` attention-mask constants), re-export with --fp16-safe-masks to clamp those
sentinels to a finite value. Leave it OFF until proven necessary.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import torch


def clamp_inf_constants_for_fp16(onnx_path: str, finite: float = 1.0e4) -> int:
    """Replace inf / sentinel-huge constants in the ONNX with a FP16-safe finite.

    The SmolVLM vision self-attn bakes additive attention-mask sentinels of
    `torch.finfo(dtype).min` (~-3.4e38) / `-inf`. In FP32/BF16 that's fine; under the
    Orin's FP16 TensorRT-EP they overflow (clip to +/-65504) and softmax/layernorm can
    diverge. We only touch clearly-sentinel magnitudes (|x| >= 1e30 or non-finite) so
    real weights are never altered. Returns the number of tensors changed.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)  # loads external data alongside if present

    def fix(tensor) -> bool:
        arr = numpy_helper.to_array(tensor)
        if not np.issubdtype(arr.dtype, np.floating):
            return False
        bad = ~np.isfinite(arr) | (np.abs(arr) >= 1.0e30)
        if not bad.any():
            return False
        out = arr.copy()
        out[bad & (arr < 0)] = -finite
        out[bad & (arr >= 0)] = finite
        new = numpy_helper.from_array(out.astype(arr.dtype), tensor.name)
        tensor.CopyFrom(new)
        return True

    changed = 0
    for init in model.graph.initializer:
        changed += int(fix(init))
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    changed += int(fix(attr.t))
    if changed:
        # Save with external data: this graph is ~1.5 GB and would blow protobuf's
        # 2 GB single-file limit if serialized inline. Produces <name>.onnx + .data.
        onnx.save(
            model, onnx_path,
            save_as_external_data=True, all_tensors_to_one_file=True,
            location=Path(onnx_path).name + ".data", size_threshold=1024,
        )
    return changed


def patch_smolvla_for_legacy_onnx_export() -> None:
    """Patch a few LeRobot/Transformers paths that trip legacy ONNX export."""
    import lerobot.policies.smolvla.modeling_smolvla as smolvla_module
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMVisionEmbeddings

    def sinusoidal_f32(time, dimension, min_period, max_period, device="cpu"):
        fraction = torch.linspace(
            0.0, 1.0, dimension // 2, dtype=torch.float32, device=device
        )
        period = min_period * (max_period / min_period) ** fraction
        scaling = 1.0 / period * 2 * math.pi
        sin_in = scaling[None, :] * time.float()[:, None]
        return torch.cat([torch.sin(sin_in), torch.cos(sin_in)], dim=1)

    smolvla_module.create_sinusoidal_pos_embedding = sinusoidal_f32

    def make_att_2d_masks_fixed(pad_masks, att_masks):
        att_int = att_masks.to(torch.int32) if att_masks.dtype == torch.bool else att_masks
        cumsum = torch.cumsum(att_int, dim=1)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks

    smolvla_module.make_att_2d_masks = make_att_2d_masks_fixed

    def vis_emb_forward_fixed(self, pixel_values, patch_attention_mask=None, tgt_sizes=None):
        batch_size, _, max_im_h, max_im_w = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        max_nb_patches_h = max_im_h // self.patch_size
        max_nb_patches_w = max_im_w // self.patch_size
        boundaries = torch.arange(
            1 / self.num_patches_per_side,
            1.0,
            1 / self.num_patches_per_side,
            device=pixel_values.device,
        )
        position_ids = torch.full(
            size=(batch_size, max_nb_patches_h * max_nb_patches_w),
            fill_value=0,
            dtype=torch.int64,
            device=pixel_values.device,
        )
        nb_patches_h = patch_attention_mask[:, :, 0].sum(dim=1)
        nb_patches_w = patch_attention_mask[:, 0, :].sum(dim=1)
        step_h = 1.0 / nb_patches_h
        step_w = 1.0 / nb_patches_w
        max_patches_h = patch_attention_mask.size(1)
        max_patches_w = patch_attention_mask.size(2)
        h_indices = torch.arange(max_patches_h, device=position_ids.device, dtype=torch.float32)
        w_indices = torch.arange(max_patches_w, device=position_ids.device, dtype=torch.float32)
        fractional_coords_h = torch.clamp(h_indices[None, :] * step_h[:, None], max=(1.0 - 1e-6))
        fractional_coords_w = torch.clamp(w_indices[None, :] * step_w[:, None], max=(1.0 - 1e-6))
        fractional_coords_h = fractional_coords_h.to(pixel_values.dtype)
        fractional_coords_w = fractional_coords_w.to(pixel_values.dtype)
        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
        pos_ids = bucket_coords_h[:, :, None] * self.num_patches_per_side + bucket_coords_w[:, None, :]
        pos_ids = pos_ids.reshape(batch_size, -1).to(torch.int64)
        flat_mask = patch_attention_mask.view(batch_size, -1)
        # Boolean-mask assignment (position_ids[flat_mask] = pos_ids[flat_mask])
        # exports to NonZero + GatherND/ScatterND, i.e. data-dependent shapes: TRT
        # inserts a device->host copy + sync per inference (latency stall) and the
        # DDS path is the fragile one on Jetson/older TRT. torch.where is equivalent
        # here (position_ids starts all-zeros, mask is all-true for a full image)
        # and exports to a static `Where`. See orin-nano notes/findings.md.
        position_ids = torch.where(flat_mask, pos_ids, position_ids)
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings

    SmolVLMVisionEmbeddings.forward = vis_emb_forward_fixed

    original_cumsum = torch.cumsum

    def cumsum_no_bool(input, *args, **kwargs):
        if input.dtype == torch.bool:
            input = input.to(torch.int32)
        return original_cumsum(input, *args, **kwargs)

    torch.cumsum = cumsum_no_bool


class SmolVLASampleActionsWrapper(torch.nn.Module):
    def __init__(self, model, n_cams: int):
        super().__init__()
        self.model = model
        self.n_cams = n_cams

    def forward(self, *args) -> torch.Tensor:
        imgs = list(args[: self.n_cams])
        masks = list(args[self.n_cams : 2 * self.n_cams])
        lang_tokens, lang_masks, state, noise = args[2 * self.n_cams :]
        return self.model.sample_actions(
            images=imgs,
            img_masks=masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            noise=noise,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="lerobot/smolvla_base")
    parser.add_argument("--output", default="exports/smolvla_base_fp32_valid.onnx")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--task", default="Pick up the red cube and place it in the bin.")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Flow-matching denoising steps baked into the graph (default: "
                             "policy config = 10). Fewer = faster inference, slightly coarser "
                             "actions. This is the ODE integration count, NOT the action chunk "
                             "size or the control dt.")
    parser.add_argument("--fp16-safe-masks", action="store_true",
                        help="Clamp inf/sentinel attention-mask constants to a finite value so the "
                             "Orin's FP16 TensorRT-EP path stays numerically safe. Leave OFF unless "
                             "on-Orin FP16 parity.py fails (non-finite / cosine drop).")
    args = parser.parse_args()

    patch_smolvla_for_legacy_onnx_export()

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    device = torch.device(args.device)
    output = Path(args.output)

    print(f"Loading {args.model_id} on {device} ...")
    policy = SmolVLAPolicy.from_pretrained(args.model_id)
    policy.eval()
    policy.to(device)
    policy.float()
    for param in policy.parameters():
        param.requires_grad_(False)

    if args.num_steps is not None:
        # num_steps is read inside sample_actions as self.config.num_steps; set it on
        # whichever config objects exist so the baked (unrolled) denoise loop uses it.
        for cfg in (getattr(policy, "config", None), getattr(policy.model, "config", None)):
            if cfg is not None and hasattr(cfg, "num_steps"):
                cfg.num_steps = args.num_steps
        print(f"num_steps set to {args.num_steps}")

    img_keys = list(policy.config.image_features.keys())
    active_img_key = img_keys[0]
    state_dim = policy.config.input_features[OBS_STATE].shape[0]
    image_h, image_w = policy.config.resize_imgs_with_padding

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    enc = tokenizer(
        args.task,
        return_tensors="pt",
        padding="max_length",
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
    )

    batch = {
        active_img_key: torch.rand(1, 1, 3, image_h, image_w, dtype=torch.float32, device=device),
        OBS_STATE: torch.zeros(1, 1, state_dim, dtype=torch.float32, device=device),
        OBS_LANGUAGE_TOKENS: enc["input_ids"].to(device),
        OBS_LANGUAGE_ATTENTION_MASK: enc["attention_mask"].bool().to(device),
    }

    with torch.no_grad():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch).float()
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        noise = torch.randn(
            1,
            policy.config.chunk_size,
            policy.config.max_action_dim,
            dtype=torch.float32,
            device=device,
        )
        pt_out = policy.model.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            noise=noise,
        )
        print(f"PyTorch output shape: {tuple(pt_out.shape)}")

    wrapper = SmolVLASampleActionsWrapper(policy.model, len(images)).eval()
    example_inputs = tuple(images) + tuple(img_masks) + (lang_tokens, lang_masks, state, noise)
    input_names = (
        [f"image{i}" for i in range(len(images))]
        + [f"img_mask{i}" for i in range(len(images))]
        + ["lang_tokens", "lang_masks", "state", "noise"]
    )
    dynamic_axes = {name: {0: "batch"} for name in input_names}
    dynamic_axes["actions"] = {0: "batch"}

    print(f"Exporting ONNX to {output} ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_inputs,
            str(output),
            input_names=input_names,
            output_names=["actions"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=False,
            dynamo=False,
        )

    if args.fp16_safe_masks:
        n = clamp_inf_constants_for_fp16(str(output))
        print(f"--fp16-safe-masks: clamped {n} inf/sentinel constant tensor(s) to +/-1e4")

    import onnx

    model = onnx.load(str(output), load_external_data=False)
    onnx.checker.check_model(model)
    print(f"ONNX checker OK: {output} ({output.stat().st_size / 1e6:.1f} MB)")

    # --- deploy bundle: tokenizer + normalization stats next to the ONNX ----------
    # The Orin loads the tokenizer with --model-id <dir>/tokenizer (vocab-exact, no
    # network, no backbone guessing). The normalization stats are needed later when the
    # Orin maps padded model actions onto real robot commands (un-normalize) and feeds
    # normalized state in. We ship them together so the bundle is self-contained.
    bundle = output.parent
    tok_dir = bundle / "tokenizer"
    tokenizer.save_pretrained(tok_dir)
    print(f"Saved tokenizer -> {tok_dir}  (use on Orin: --model-id {tok_dir})")

    src = Path(args.model_id)
    if not src.is_dir():
        try:
            from huggingface_hub import snapshot_download
            src = Path(snapshot_download(
                args.model_id,
                allow_patterns=["*preprocessor*", "*postprocessor*", "config.json"],
            ))
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: could not resolve normalization stats for {args.model_id}: {exc}")
            src = None
    copied = []
    if src is not None:
        for f in sorted(src.glob("*")):
            if any(k in f.name for k in ("preprocessor", "postprocessor")):
                shutil.copy2(f, bundle / f.name)
                copied.append(f.name)
    print(f"Bundled normalization stats: {copied or 'NONE FOUND — un-normalization stats missing'}")

    import onnxruntime as ort

    print("Creating CPU ORT session ...")
    sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    print("ORT providers:", sess.get_providers())
    print("Inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("Outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])


if __name__ == "__main__":
    main()
