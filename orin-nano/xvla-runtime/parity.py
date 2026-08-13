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


def emit_reference(args) -> None:
    import torch

    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    from xvla_runtime.split_ort import preprocess_image

    bundle = json.loads((args.split_dir / "bundle.json").read_text())
    rng = np.random.default_rng(args.seed)
    n_views = bundle["valid_views"]
    chunk_size = bundle["chunk_size"]
    lang_len = bundle["lang_len"]

    images = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(n_views)]
    state = rng.standard_normal(bundle["max_state_dim"]).astype(np.float32)

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

    tok = AutoTokenizer.from_pretrained(args.tokenizer)(
        args.instruction, max_length=lang_len, padding="max_length",
        truncation=True, padding_side="right", return_tensors="pt",
    )
    domain_id = torch.tensor([bundle["domain_id"]], dtype=torch.long)
    proprio = torch.zeros(1, model.dim_proprio)
    proprio[0, : len(state)] = torch.from_numpy(state[: model.dim_proprio])

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

    steps = args.steps or bundle["num_denoising_steps"]
    real_randn = torch.randn
    target = (1, chunk_size, model.dim_action)

    def fixed_randn(*a, **kw):
        shape = tuple(a[0]) if len(a) == 1 and not isinstance(a[0], int) else tuple(a)
        if shape == target:
            return torch.from_numpy(x1)
        return real_randn(*a, **kw)

    torch.randn = fixed_randn
    try:
        with torch.no_grad():
            action = model.generate_actions(
                input_ids=tok["input_ids"], image_input=image_input,
                image_mask=image_mask, domain_id=domain_id, proprio=proprio, steps=steps,
            )
    finally:
        torch.randn = real_randn

    args.reference.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.reference,
        images=np.stack(images), state=state, x1=x1, steps=np.array(steps),
        instruction=np.array(args.instruction),
        vlm_features=enc["vlm_features"].numpy(),
        aux_visual=enc["aux_visual_inputs"].numpy(),
        cond_tokens=cond.numpy(),
        action=action[0].numpy(),
    )
    LOG.info("wrote %s", args.reference)


# ======================================================================================
# phase 2 -- split engines, compared against the saved reference
# ======================================================================================


def compare(args) -> int:
    from xvla_runtime.split_ort import XVLASplitPolicy

    ref = np.load(args.reference, allow_pickle=False)
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
    ok &= report("action", ref["action"], action_got, args.threshold)

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
    ap.add_argument("--tokenizer", default=str(REPO / "models" / "tokenizer"))
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

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.emit:
        emit_reference(args)
        return

    if args.refresh_reference or not args.reference.exists():
        LOG.info("running reference phase in a separate process ...")
        cmd = [sys.executable, __file__, "--_emit-reference",
               "--split-dir", str(args.split_dir), "--checkpoint", str(args.checkpoint),
               "--reference", str(args.reference), "--tokenizer", args.tokenizer,
               "--instruction", args.instruction, "--seed", str(args.seed)]
        if args.steps:
            cmd += ["--steps", str(args.steps)]
        if subprocess.run(cmd, text=True).returncode != 0:
            sys.exit("reference phase failed")

    sys.exit(compare(args))


if __name__ == "__main__":
    main()
