#!/usr/bin/env python
"""Derive an excavator-ready X-VLA checkpoint dir from the stock `lerobot/xvla-base`.

The stock config.json declares the input contract of the robots X-VLA was pretrained on:

    observation.images.image   VISUAL (3, 256, 256)
    observation.images.image2  VISUAL (3, 256, 256)
    observation.images.image3  VISUAL (3, 224, 224)
    observation.state          STATE  (8,)

None of which is our excavator (`observation.images.cam1`, state `(3,)`). That matters
because `make_policy` only fills `input_features` from the dataset **when the config leaves
it empty** (`if not cfg.input_features:`), so with `--policy.path` the pretrained contract
survives and two things go wrong:

1. `validate_visual_features_consistency` fails outright — neither {cam1} nor
   {image,image2,image3} is a subset of the other.
2. Worse if you dodge (1) with `--rename_map`: that skips validation entirely, and the
   normalizer is then built with `features=policy.config.input_features` (state shape (8,))
   against `stats=dataset.meta.stats` (state shape (3,)). Mismatched shapes, silently.

`--policy.input_features={}` does not help — draccus merges the CLI dict into the one from
config.json rather than replacing it, so the four stock entries come back.

So we empty it here instead, in a derived directory. Nothing else changes, and nothing about
the weights depends on the emptied keys:

  * `dim_proprio = max_state_dim` (20), and `_prepare_state` pads our 3-dim state to 20 —
    the declared `(8,)` only ever fed the normalizer.
  * `num_image_views` stays 3, so `_prepare_images` pads our 1 (or 2) real views out to 3
    with zeroed, mask=False slots. `forward_vlm` runs the vision encoder over valid views
    only, so the padding is free. This is exactly the Orin runtime's `--valid-views 1`.

`model.safetensors` is symlinked, not copied — 3.5 GB is not worth duplicating.

Usage:
    python prepare_checkpoint.py                    # models/xvla-base -> models/xvla-base-excavator
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from vla_common.paths import dataset, playbook

ROOT = playbook("xvla")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=ROOT / "models/xvla-base")
    p.add_argument("--dst", type=Path, default=ROOT / "models/xvla-base-excavator")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    src, dst = args.src.resolve(), args.dst.resolve()
    if not (src / "model.safetensors").exists():
        raise SystemExit(f"no checkpoint at {src} — run fetch_checkpoint.sh first")

    if dst.exists():
        if not args.force:
            raise SystemExit(f"{dst} exists (use --force)")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    cfg = json.loads((src / "config.json").read_text())
    dropped = dict(cfg.get("input_features", {}))
    cfg["input_features"] = {}
    # output_features is repopulated from the dataset by make_policy unconditionally, but
    # emptying it too keeps the file honest about what this checkpoint no longer claims.
    cfg["output_features"] = {}
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)

    if (src / "REVISION").exists():
        shutil.copy2(src / "REVISION", dst / "REVISION")

    (dst / "model.safetensors").symlink_to(src / "model.safetensors")

    print(f"{dst}")
    print(f"  weights symlinked from {src}")
    print(f"  emptied input_features so the dataset defines them; dropped:")
    for k, v in dropped.items():
        print(f"    {k:32s} {tuple(v['shape'])}")
    print(f"  kept num_image_views={cfg.get('num_image_views')} "
          f"max_state_dim={cfg.get('max_state_dim')} max_action_dim={cfg.get('max_action_dim')}")


if __name__ == "__main__":
    main()
