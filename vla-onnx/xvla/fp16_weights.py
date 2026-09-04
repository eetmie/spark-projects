#!/usr/bin/env python3
"""Mixed-FP16 X-VLA bundle — the shared recipe, with this model's defaults.

Twelve sessions sit at ~6.7 GB on a 7.4 GB board, so residency is the constraint here.
The precision recipe and the reasoning behind it live in `vla_common.fp16_weights`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_common.fp16_weights import convert_bundle

REPO = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-dir", type=Path, default=REPO / "exports" / "split")
    ap.add_argument("--out-dir", type=Path, default=REPO / "exports" / "split_fp16")
    ap.add_argument("--only", nargs="*", default=None,
                    help="convert only these graphs; the rest are copied as-is")
    args = ap.parse_args()
    convert_bundle(args.split_dir, args.out_dir, only=args.only)


if __name__ == "__main__":
    main()
