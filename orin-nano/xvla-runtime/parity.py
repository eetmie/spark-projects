#!/usr/bin/env python3
"""Parity guard: split ORT/TRT engines vs the PyTorch X-VLA reference.

Nothing downstream should be trusted until this passes. The split rearranges real
computation -- the conditioning projections are hoisted out of the denoising loop, the
domain-conditioned weights are baked to constants, and the graph is cut into a dozen
engines -- so "it runs and the numbers look like actions" is not evidence of correctness.

Runs in TWO PROCESSES, and that is not incidental: the PyTorch reference is 3.5 GB on CPU
and the dozen TRT sessions are several GB more, so holding both would OOM this board. The
reference phase writes its tensors to an .npz, exits, and the compare phase loads only the
engines.

Both paths must see the same inputs, including the noise -- `generate_actions` draws its
own `x1` internally, so the reference phase saves the draw and the split path is given it.

    python parity.py --split-dir exports/split --checkpoint models/xvla-base
    python parity.py --refresh-reference        # re-run the reference phase
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from xvla_runtime.bundle_contract import (normalize_vector, tree_sha256,
                                          unnormalize_vector, verify_bundle)

LOG = logging.getLogger("parity")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else float("nan")


def report(name: str, ref: np.ndarray, got: np.ndarray, threshold: float) -> bool:
    if ref.shape != got.shape:
        print(f"  {name:14s} SHAPE MISMATCH ref{list(ref.shape)} vs got{list(got.shape)}")
        return False
    cos = cosine(ref, got)
    max_abs = float(np.abs(ref - got).max())
    rel = max_abs / (float(np.abs(ref).max()) + 1e-9)
    ok = cos >= threshold
    print(f"  {name:14s} cos {cos:.6f}  max|d| {max_abs:.5f}  rel {rel:.5f}  "
          f"{'ok' if ok else 'FAIL'}")
    return ok


# ======================================================================================
# phase 1 -- PyTorch reference, run alone in its own process
# ======================================================================================


# The reference npz is a cache: emitting it costs ~3.5 GB of PyTorch and a couple of
# minutes, so it is written once and reused. Reuse is only valid for a bundle that
# would have produced the SAME reference, and nothing used to check that -- a reference
# emitted for a 1-view bundle was silently loaded against a 3-view one. That case
# happened to die on a shape mismatch inside ORT, which was luck: a bundle differing
# only in something shape-preserving (a different domain_id, a different seed, a
# different instruction) would have compared clean numbers against the wrong reference
# and printed a PASS that meant nothing. Stamp the inputs into the file and refuse to
# reuse it when they differ.
REFERENCE_SIGNATURE_KEYS = (
    "valid_views", "chunk_size", "lang_len", "max_state_dim", "max_action_dim",
    "real_state_dim", "real_action_dim", "domain_id", "num_denoising_steps",
)


def _reference_signature(bundle: dict, args) -> dict:
    sig = {k: bundle.get(k) for k in REFERENCE_SIGNATURE_KEYS}
    if args.steps is not None:            # --steps overrides the bundle's own count
        sig["num_denoising_steps"] = args.steps
    sig["checkpoint_tree_sha256"] = (bundle.get("checkpoint") or {}).get("tree_sha256")
    sig["processor_artifacts"] = (bundle.get("processor_contract") or {}).get("artifacts")
    sig["tokenizer_tree_sha256"] = (bundle.get("tokenizer") or {}).get("tree_sha256")
    sig["seed"] = args.seed
    sig["instruction"] = args.instruction
    return sig


def emit_reference(args) -> None:
    import torch

    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    from xvla_runtime.split_ort import preprocess_image

    bundle = verify_bundle(args.split_dir, verify_manifest=True)
    expected_checkpoint_sha = (bundle.get("checkpoint") or {}).get("tree_sha256")
    if tree_sha256(args.checkpoint) != expected_checkpoint_sha:
        raise ValueError("PyTorch checkpoint does not match bundle identity")
    contract = bundle.get("processor_contract")
    rng = np.random.default_rng(args.seed)
    n_views = bundle["valid_views"]
    chunk_size = bundle["chunk_size"]
    lang_len = bundle["lang_len"]

    images = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(n_views)]
    state_dim = int((contract or {}).get("state", {}).get(
        "dim", bundle["max_state_dim"]))
    state = rng.standard_normal(state_dim).astype(np.float32)

    LOG.info("loading PyTorch reference (CPU, ~3.5 GB) ...")
    # The checkpoint's config says device=cuda and from_pretrained honours it; the sample
    # tensors here are CPU, and putting 3.5 GB of weights into the Orin's unified memory
    # would compete with the engines this is meant to check.
    from lerobot.configs.policies import PreTrainedConfig

    cfg_obj = PreTrainedConfig.from_pretrained(str(args.checkpoint))
    cfg_obj.device = "cpu"
    ref = XVLAPolicy.from_pretrained(str(args.checkpoint), config=cfg_obj)
    ref.to("cpu").eval()
    model = ref.model

    x1 = rng.standard_normal((1, chunk_size, model.dim_action)).astype(np.float32)

    pixels = np.stack([preprocess_image(img) for img in images]).astype(np.float32)
    image_input = torch.from_numpy(pixels).unsqueeze(0)
    n_pad = ref.config.num_image_views - image_input.shape[1]
    if n_pad > 0:
        image_input = torch.cat(
            [image_input, image_input.new_zeros((1, n_pad, *image_input.shape[2:]))], dim=1
        )
    image_mask = torch.zeros(1, ref.config.num_image_views, dtype=torch.bool)
    image_mask[0, :n_views] = True

    from transformers import AutoTokenizer

    tokenizer_path = (Path(args.tokenizer) if args.tokenizer
                      else args.split_dir / bundle["tokenizer"]["path"])
    if bundle.get("schema_version", 1) >= 2:
        expected_tokenizer_sha = bundle["tokenizer"]["tree_sha256"]
        if tree_sha256(tokenizer_path) != expected_tokenizer_sha:
            raise ValueError("reference tokenizer does not match bundle identity")
    tok = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=bundle.get("schema_version", 1) >= 2
    )(
        args.instruction, max_length=lang_len, padding="max_length",
        truncation=True, padding_side="right", return_tensors="pt",
    )
    domain_id = torch.tensor([bundle["domain_id"]], dtype=torch.long)
    proprio = torch.zeros(1, model.dim_proprio)
    model_state = normalize_vector(state, contract["state"]) if contract else state
    proprio[0, : len(model_state)] = torch.from_numpy(model_state)

    with torch.no_grad():
        enc = model.forward_vlm(tok["input_ids"], image_input, image_mask)
        n_act = ref.config.chunk_size
        cond = torch.cat(
            [
                model.transformer.vlm_proj(enc["vlm_features"]),
                model.transformer.aux_visual_proj(enc["aux_visual_inputs"]),
            ],
            dim=1,
        )
        cond = cond + model.transformer.pos_emb[:, n_act : n_act + cond.shape[1]]

    steps = (args.steps if args.steps is not None else bundle["num_denoising_steps"])
    x1_tensor = torch.from_numpy(x1)
    action_model = torch.zeros_like(x1_tensor)
    with torch.no_grad():
        for i in range(steps, 0, -1):
            t = torch.full((1,), i / steps, dtype=x1_tensor.dtype)
            x_t = x1_tensor * t.view(-1, 1, 1) + action_model * (
                1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = model.action_space.preprocess(proprio, x_t)
            action_model = model.transformer(
                domain_id=domain_id, action_with_noise=x_t_m, proprio=proprio_m,
                t=t, **enc,
            )
        normalized = model.action_space.postprocess(action_model)

    normalized_action = normalized[0].numpy()
    physical_action = (unnormalize_vector(normalized_action, contract["action"])
                       if contract else normalized_action)

    args.reference.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.reference,
        signature=np.array(json.dumps(_reference_signature(bundle, args))),
        images=np.stack(images), state=state, x1=x1, steps=np.array(steps),
        instruction=np.array(args.instruction),
        vlm_features=enc["vlm_features"].numpy(),
        aux_visual=enc["aux_visual_inputs"].numpy(),
        cond_tokens=cond.numpy(), model_action=action_model[0].numpy(),
        normalized_action=normalized_action, action=physical_action,
    )
    LOG.info("wrote %s", args.reference)


# ======================================================================================
# phase 2 -- split engines, compared against the saved reference
# ======================================================================================


def compare(args) -> int:
    from xvla_runtime.split_ort import XVLASplitPolicy

    ref = np.load(args.reference, allow_pickle=False)

    bundle = verify_bundle(args.split_dir, verify_manifest=True)
    want = _reference_signature(bundle, args)
    got = json.loads(str(ref["signature"])) if "signature" in ref.files else None
    if got != want:
        if got is None:
            detail = "it predates signature stamping"
        else:
            diff = [f"{k}: reference {got.get(k)!r} vs bundle {v!r}"
                    for k, v in want.items() if got.get(k) != v]
            detail = "; ".join(diff) or "unknown difference"
        sys.exit(
            f"reference {args.reference} does not match {args.split_dir}\n"
            f"  {detail}\n"
            f"  Re-emit it for this bundle:  --refresh-reference\n"
            f"  (or keep one per bundle with --reference <path>)")

    images = [img for img in ref["images"]]
    instruction = str(ref["instruction"])
    steps = int(ref["steps"])

    policy = XVLASplitPolicy(
        args.split_dir, cache_dir=args.cache_dir, precision=args.precision,
        tokenizer_dir=args.tokenizer, num_denoising_steps=steps,
    )

    print(f"\nparity vs PyTorch reference  (precision={args.precision}, "
          f"threshold cos >= {args.threshold})")
    print(f"  views={policy.valid_views}  lang_len={policy.lang_len}  "
          f"tokens_per_view={policy.tokens_per_view}  steps={steps}\n")

    ok = True
    cond_got = policy.encode_observation(images, instruction)
    ok &= report("cond_tokens", ref["cond_tokens"], cond_got, args.threshold)

    action_got = policy.sample_actions(images, instruction, ref["state"], x1=ref["x1"])
    if "model_action" in ref.files:
        ok &= report("model action 20D", ref["model_action"],
                     policy.last_model_action[0], args.threshold)
    if "normalized_action" in ref.files:
        ok &= report(
            "normalized", ref["normalized_action"], policy.last_normalized_action[0],
            args.threshold)
    ok &= report("physical action", ref["action"], action_got, args.threshold)

    t = policy.last_timings
    print(f"\n  timings: vision {t.get('vision_ms', 0):.0f} ms, text {t.get('text_ms', 0):.0f} ms, "
          f"cond {t.get('cond_ms', 0):.0f} ms, denoise {t.get('denoise_ms', 0):.0f} ms "
          f"({steps} steps)")
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", type=Path, default=REPO / "exports" / "split")
    ap.add_argument("--checkpoint", type=Path, default=REPO / "models" / "xvla-base")
    ap.add_argument("--reference", type=Path, default=REPO / "exports" / "parity_reference.npz")
    ap.add_argument("--tokenizer", default=None,
                    help="override tokenizer directory; schema-v2 requires an exact hash match")
    ap.add_argument("--cache-dir", default=None,
                    help="TRT engine cache; defaults to <split-dir>/trt_cache")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--instruction", default="pick up the rock and place it in the bucket")
    ap.add_argument("--threshold", type=float, default=0.997,
                    help="cosine floor; smolvla-runtime uses 0.997 for the fp16 path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--refresh-reference", action="store_true")
    ap.add_argument("--_emit-reference", dest="emit", action="store_true",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.steps is not None and args.steps <= 0:
        ap.error("--steps must be positive")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.emit:
        emit_reference(args)
        return

    if args.refresh_reference or not args.reference.exists():
        LOG.info("running reference phase in a separate process ...")
        cmd = [sys.executable, __file__, "--_emit-reference",
               "--split-dir", str(args.split_dir), "--checkpoint", str(args.checkpoint),
               "--reference", str(args.reference), "--instruction", args.instruction,
               "--seed", str(args.seed)]
        if args.tokenizer:
            cmd += ["--tokenizer", args.tokenizer]
        if args.steps:
            cmd += ["--steps", str(args.steps)]
        if subprocess.run(cmd, text=True).returncode != 0:
            sys.exit("reference phase failed")

    sys.exit(compare(args))


if __name__ == "__main__":
    main()
