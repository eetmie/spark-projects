#!/usr/bin/env python3
"""N1.7 bundle (ORT on the Spark, or TensorRT on the Orin) vs the stock-PyTorch reference.

    python parity17.py --bundle work/groot-n17-split.fp32 --ref work/ref17.npz
    python parity17.py --bundle ~/bundles/groot-n17-base-split --ref ref17.npz --backend trt --cache DIR
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from parity import compare, ort_runner
from pipeline17 import Bundle, encode_frames, infer


def unpatchify(pv: np.ndarray, n: int, gh: int, gw: int) -> np.ndarray:
    x = pv.reshape(n, gh // 2, gw // 2, 2, 2, 3, 2, 16, 16)[..., 0, :, :]
    return np.ascontiguousarray(x.transpose(0, 5, 1, 3, 6, 2, 4, 7).reshape(n, 3, gh * 16, gw * 16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--backend", default="ort", choices=["ort", "trt"])
    ap.add_argument("--cache", help="trt: engine directory")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()

    bundle = Bundle(a.bundle)
    r = np.load(a.ref)
    if a.backend == "ort":
        run = ort_runner(bundle, a.threads)
    else:
        from trt_runtime import Engines
        run = Engines(bundle, a.cache).run

    b = bundle.b
    thw = r["image_grid_thw"]
    pix = unpatchify(r["pixel_values"].astype(np.float32), thw.shape[0], int(thw[0, 1]), int(thw[0, 2]))
    v = b["views"]
    t0 = time.perf_counter()
    frames = [encode_frames(bundle, run, pix[i * v:(i + 1) * v]) for i in range(b["frames"])]
    t1 = time.perf_counter()
    timings = {}
    act = infer(bundle, run, frames, r["state"], r["noise"], timings=timings)
    timings["vision_ms_per_slot"] = (t1 - t0) * 1e3 / b["frames"]
    print({k: round(v, 1) for k, v in timings.items()})
    res = compare("action_pred", act, r["action_pred"])
    first = compare("first action", act[:, 0], r["action_pred"][:, 0])
    print(json.dumps({"chunk": res, "first": first}))


if __name__ == "__main__":
    main()
