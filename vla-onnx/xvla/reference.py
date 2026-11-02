#!/usr/bin/env python3
"""Store a stock-X-VLA reference chunk in a bundle: the fixture the robot gates every load on.

Ported from jetson-orin-nano-vla `export/xvla/reference.py`, the exporter the Orin runtime
(kaivuriprokkis-python `lerobot_vla/runtime/vendor/xvla_trt.py`) was written against. Since
2026-10-29 that runtime refuses a bundle without `bundle.json` `fixture` + `fixture.npz`: it
runs the engines on the saved inputs and noise, and checks the chunk, its own image
preprocessing and its own tokenizer against what stock LeRobot produced here.

Seeded synthetic inputs (two random 640x480 frames, random state) go through the
checkpoint's own preprocessor pipeline and the policy's own image padding, so `input_ids`,
`pixel_values` and `proprio` are exactly what the model sees. The noise draw inside
`generate_actions` is replaced by a seeded one that is saved too.

Two differences from the Orin repo's copy, both found shipping the dual-camera run:
  * float32 config dtype. Fine-tunes train with --policy.dtype=bfloat16, and the stock
    sampler follows the config dtype ("expected m1 and m2 to have the same dtype"). The
    graphs are fp32 upcasts of the same weights (parity.py does the same).
  * the untrimmed model-width chunk. lerobot 0.6.1's `auto` action space trims [.., 20] to
    the robot's 4 columns in postprocess; the board compares the engines' raw 20-wide
    output, so `action_pred` must be 20 wide.

Runs on the CPU in float32, the precision the graphs were traced in.

    python reference.py --checkpoint <pretrained_model> --bundle <bundle dir> [--task "..."]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1] / "common"
sys.path.insert(0, str(REPO))
from vla_common.bundle import write_manifest  # noqa: E402

FIXTURE = "fixture.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--task", default=None, help="default: the bundle's first task")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    bundle = json.loads((a.bundle / "bundle.json").read_text())
    task = a.task or (bundle.get("tasks") or [None])[0]
    if not task:
        raise SystemExit(f"{a.bundle}: bundle.json names no task; pass --task")
    views = int(bundle["valid_views"])

    cfg = PreTrainedConfig.from_pretrained(str(a.checkpoint))
    cfg.device = "cpu"
    cfg.dtype = "float32"
    policy = XVLAPolicy.from_pretrained(str(a.checkpoint), config=cfg)
    policy.to("cpu").float().eval()
    pre, _ = make_pre_post_processors(
        cfg, pretrained_path=str(a.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}})

    rng = np.random.default_rng(a.seed)
    keys = list(cfg.image_features)[:views]
    raw = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in keys]
    state_dim = cfg.input_features["observation.state"].shape[0]
    state = rng.standard_normal(state_dim).astype(np.float32)
    obs = {k: torch.from_numpy(im.transpose(2, 0, 1).copy()).float() for k, im in zip(keys, raw)}
    obs["observation.state"] = torch.from_numpy(state)
    obs["task"] = task
    batch = pre(obs)

    model = policy.model
    x1 = torch.from_numpy(
        rng.standard_normal((1, model.chunk_size, model.dim_action)).astype(np.float32))
    inputs = policy._build_model_inputs(batch)
    model.action_space.postprocess = lambda x: x  # keep the model-width chunk (see above)
    real_randn = torch.randn
    torch.randn = lambda *s, **kw: x1.clone().to(kw.get("dtype") or x1.dtype)
    try:
        with torch.no_grad():
            action = model.generate_actions(**inputs, steps=cfg.num_denoising_steps)
    finally:
        torch.randn = real_randn
    if action.shape[-1] != model.dim_action:
        raise SystemExit(f"reference chunk is {tuple(action.shape)}, expected model width {model.dim_action}")

    np.savez(
        a.bundle / FIXTURE,
        raw=np.stack(raw),
        pixel_values=inputs["image_input"][0, :views].numpy(),
        input_ids=inputs["input_ids"].numpy(),
        proprio=inputs["proprio"].numpy(),
        domain_id=inputs["domain_id"].numpy(),
        state=state, x1=x1.numpy(), action_pred=action.numpy(),
        meta=np.array(json.dumps({"task": task, "views": views, "seed": a.seed,
                                  "steps": cfg.num_denoising_steps})),
    )
    bundle["fixture"] = {"file": FIXTURE, "task": task, "seed": a.seed,
                         "source": "stock LeRobot XVLAPolicy, float32 on the CPU"}
    (a.bundle / "bundle.json").write_text(json.dumps(bundle, indent=2))
    write_manifest(a.bundle)
    print(f"action_pred {tuple(action.shape)} -> {a.bundle / FIXTURE}")


if __name__ == "__main__":
    main()
