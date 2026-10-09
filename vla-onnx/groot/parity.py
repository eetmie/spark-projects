#!/usr/bin/env python3
"""Bundle (ORT on the Spark, or TensorRT on the Orin) vs the stock-PyTorch reference.

    python parity.py --bundle work/groot-n16-split.fp32 --ref work/ref_robocasa_v3.npz
    python parity.py --bundle ~/bundles/groot-n16-split --ref ref.npz --backend trt --cache DIR
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from pipeline import Bundle, infer


def ort_runner(bundle: Bundle, threads: int):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    # ORT's optimizer fuses com.microsoft.Gelu, which has no FP16 CPU kernel.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sessions = {}

    def run(name, feeds):
        if name not in sessions:
            sessions[name] = ort.InferenceSession(str(bundle.root / f"{name}.onnx"), so,
                                                  providers=["CPUExecutionProvider"])
        s = sessions[name]
        # A chunk whose blocks all attend to one modality has the other bias pruned.
        out = s.run(None, {i.name: feeds[i.name] for i in s.get_inputs()})
        return dict(zip([o.name for o in s.get_outputs()], out))
    return run


def compare(name, a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    d = np.abs(a - b)
    rng = float(b.max() - b.min())
    res = {"cosine": cos, "max_abs": float(d.max()), "max_pct_range": float(d.max() / rng * 100),
           "p99_pct_range": float(np.percentile(d, 99) / rng * 100)}
    print(f"{name:16s} cos {cos:.6f}  max {res['max_pct_range']:.3f} %  "
          f"p99 {res['p99_pct_range']:.3f} % of range")
    return res


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

    # The processor's own pixels: its 256 -> 252 bicubic resize is a camera-side concern.
    pv = r["pixel_values"].astype(np.float32)
    timings = {}
    act = infer(bundle, run, pv, r["state"], r["noise"], timings=timings)
    print({k: round(v, 1) for k, v in timings.items()})
    res = compare("action_pred", act, r["action_pred"])
    first = compare("first action", act[:, 0], r["action_pred"][:, 0])
    print(json.dumps({"chunk": res, "first": first}))


if __name__ == "__main__":
    main()
