#!/usr/bin/env python3
"""Mixed-FP16 EVO1 bootstrap bundle for Orin resident-memory testing.

`token_embedding` stays FP32: it runs on the CPU EP, so casting it buys nothing and
costs precision. The shared recipe lives in `vla_common.fp16_weights`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_common.fp16_weights import convert_bundle

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-dir", type=Path,
                    default=ROOT / "exports" / "split-bootstrap")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "exports" / "split-bootstrap-fp16")
    args = ap.parse_args()

    # This converter only ever handles the nondeployable random-head bootstrap.
    # Refusing anything else keeps a trained bundle from silently acquiring the
    # bootstrap's provenance.
    bundle = json.loads((args.split_dir / "bundle.json").read_text())
    if bundle.get("deployable") or not bundle.get("random_action_head"):
        raise SystemExit("bootstrap converter refuses an unrecognized bundle")

    convert_bundle(args.split_dir, args.out_dir, keep_fp32=("token_embedding",))


if __name__ == "__main__":
    main()
