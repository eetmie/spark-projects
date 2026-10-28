"""Bundle-vs-reference helpers shared by parity17.py: the ORT CPU runner and the metrics."""

from __future__ import annotations


import numpy as np

from pipeline17 import Bundle


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


