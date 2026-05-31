"""
PyTorch-vs-ONNX parity check for the exported SmolVLA graph.

This validates the ONNX graph on the GB10 machine using CPU ONNX Runtime and
PyTorch CUDA/CPU for the reference output. TensorRT parity must still be checked
on the Jetson after building the engine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from export_valid_onnx import SmolVLASampleActionsWrapper, patch_smolvla_for_legacy_onnx_export


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    if denom == 0:
        return 1.0 if np.linalg.norm(af - bf) == 0 else 0.0
    return float(np.dot(af, bf) / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="lerobot/smolvla_base")
    parser.add_argument("--onnx", default="exports/smolvla_base_fp32_valid.onnx")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task", default="Pick up the red cube and place it in the bin.")
    parser.add_argument("--out", default="exports/smolvla_base_fp32_valid.parity.json")
    args = parser.parse_args()

    patch_smolvla_for_legacy_onnx_export()

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
    import onnxruntime as ort

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading PyTorch policy {args.model_id} on {device} ...")
    policy = SmolVLAPolicy.from_pretrained(args.model_id)
    policy.eval().to(device).float()
    for param in policy.parameters():
        param.requires_grad_(False)

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
        wrapper = SmolVLASampleActionsWrapper(policy.model, len(images)).eval()
        torch_out = wrapper(*(tuple(images) + tuple(img_masks) + (lang_tokens, lang_masks, state, noise)))
        if device.type == "cuda":
            torch.cuda.synchronize()
        torch_np = torch_out.detach().cpu().numpy().astype(np.float32)

    print(f"Creating ORT CPU session for {args.onnx} ...")
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    feed = {}
    for i, (img, mask) in enumerate(zip(images, img_masks)):
        feed[f"image{i}"] = img.detach().cpu().numpy().astype(np.float32)
        feed[f"img_mask{i}"] = mask.detach().cpu().numpy()
    feed["lang_tokens"] = lang_tokens.detach().cpu().numpy()
    feed["lang_masks"] = lang_masks.detach().cpu().numpy()
    feed["state"] = state.detach().cpu().numpy().astype(np.float32)
    feed["noise"] = noise.detach().cpu().numpy().astype(np.float32)

    print("Running ORT CPU inference ...")
    ort_np = sess.run(None, feed)[0].astype(np.float32)

    abs_diff = np.abs(torch_np - ort_np)
    result = {
        "onnx": args.onnx,
        "model_id": args.model_id,
        "seed": args.seed,
        "shape_torch": list(torch_np.shape),
        "shape_onnx": list(ort_np.shape),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "cosine": cosine(torch_np, ort_np),
        "passed_1e_4": bool(abs_diff.max() < 1e-4),
        "passed_1e_3": bool(abs_diff.max() < 1e-3),
    }
    print(json.dumps(result, indent=2))
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
