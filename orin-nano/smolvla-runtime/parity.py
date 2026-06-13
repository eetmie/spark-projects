#!/usr/bin/env python3
"""Parity-check the ORT TensorRT-EP runtime against the FP32 ONNX, on this Orin Nano.

Before trusting any action the runtime emits, confirm the reduced-precision
(FP16 by default) ORT/TensorRT-EP path still matches the FP32 reference. This is
the guard for FP16 overflow (NaN/divergence in softmax/layernorm): if cosine
similarity drops or the output goes non-finite, the precision-sensitive ops aren't
being kept in FP32 and the export / provider options need a look.

How it works
------------
* Reference  = the FP32 ONNX run through ONNX Runtime on the **CPU** EP. CPU gives
  *true* FP32; the CUDA EP would use TF32 and wouldn't be a clean gold.
* Candidate  = the SAME ONNX run through ONNX Runtime with the **TensorRT EP**
  (fp16 by default; bf16 with --precision bf16), i.e. the actual deployment path.
* Identical, seeded inputs go to both (same image, tokens, state, and the same
  noise draw per sample -- flow-matching is noise-sensitive, so this must match).
* Reference and candidate run **sequentially**, and the reference session is
  released before the candidate builds its engine, so peak memory stays within the
  8 GB shared RAM.

Inputs/outputs are mapped by the same io_spec.resolve_io the runtime uses, so this
survives the `image` vs `image0` export-naming difference automatically.

Precision note (Orin Nano, compute 8.7)
---------------------------------------
FP16 is the deployment mode and should PASS at cos >= 0.997. BF16 is experimental:
keep it ONLY if (a) the TRT-EP logs show real BF16 tactics on this image, (b) it
PASSES here, and (c) it beats FP16 on latency / action drift. See tools/probe_precision.py.

Examples
--------
    # Standard FP16 check (3 seeded samples, CPU FP32 reference):
    python parity.py --onnx exports/smolvla.onnx

    # BF16 experiment, more samples, a real image:
    python parity.py --onnx exports/smolvla.onnx --precision bf16 \
        --num-samples 5 --image some_frame.png
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

from smolvla_runtime.backends.ort import build_providers
from smolvla_runtime.io_spec import TensorSpec, resolve_io
from smolvla_runtime.preprocess import InputBuilder

LOG = logging.getLogger("parity")


# --- comparison metrics ------------------------------------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a, b) / (na * nb))


def _compare(ref: np.ndarray, cand: np.ndarray) -> dict:
    ref = np.asarray(ref, dtype=np.float32)
    cand = np.asarray(cand, dtype=np.float32)
    finite = bool(np.all(np.isfinite(cand)))
    diff = np.abs(ref.astype(np.float64) - cand.astype(np.float64))
    denom = float(np.max(np.abs(ref))) + 1e-9
    return {
        "cosine": _cosine(ref, cand),
        "max_abs": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs": float(np.mean(diff)) if diff.size else 0.0,
        "max_rel": (float(np.max(diff)) / denom) if diff.size else 0.0,
        "finite": finite,
        "n_nan": int(np.sum(np.isnan(cand))),
        "n_inf": int(np.sum(np.isinf(cand))),
    }


_ORT_TYPE_TO_NP = {
    "tensor(float)": np.float32, "tensor(float16)": np.float16,
    "tensor(double)": np.float64, "tensor(int64)": np.int64,
    "tensor(int32)": np.int32, "tensor(bool)": np.bool_, "tensor(uint8)": np.uint8,
}


def _ort_shape(shape):
    return tuple(d if isinstance(d, int) else -1 for d in shape)


def _specs(sess):
    inputs = [TensorSpec(i.name, np.dtype(_ORT_TYPE_TO_NP.get(i.type, np.float32)), _ort_shape(i.shape))
              for i in sess.get_inputs()]
    outputs = [TensorSpec(o.name, np.dtype(_ORT_TYPE_TO_NP.get(o.type, np.float32)), _ort_shape(o.shape))
               for o in sess.get_outputs()]
    return inputs, outputs


# --- reference: FP32 ONNX via ORT CPU (true FP32) ----------------------------
def run_reference(onnx_path, feeds_per_sample):
    """Run the FP32 ONNX on the CPU EP; return io + per-sample output dicts. The
    session is created and torn down here so its memory frees before the candidate
    builds its TensorRT engine."""
    import onnxruntime as ort

    LOG.info("Reference: ONNX Runtime FP32 on CPU (%s)", onnx_path)
    LOG.info("  (CPU FP32 is the trustworthy gold reference but slow -- be patient.)")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs, outputs = _specs(sess)
    io = resolve_io(inputs, outputs)
    out_names = [o.name for o in sess.get_outputs()]

    results = []
    for logical in feeds_per_sample:
        feeds = {io.role_to_name[role]: arr for role, arr in logical.items()}
        results.append(dict(zip(out_names, sess.run(out_names, feeds))))
    del sess  # release the FP32 session before the candidate's engine is built
    return io, results


# --- candidate: the deployment runtime (ORT + TensorRT EP) -------------------
def run_candidate(onnx_path, feeds_per_sample, precision, cache_dir):
    import onnxruntime as ort

    LOG.info("Candidate: ONNX Runtime + TensorRT EP, precision=%s (%s)", precision, onnx_path)
    LOG.info("  (first run builds + caches the TRT engine into %s — minutes.)", cache_dir)
    sess = ort.InferenceSession(onnx_path, providers=build_providers(cache_dir, precision=precision))
    LOG.info("  active providers: %s", sess.get_providers())
    inputs, outputs = _specs(sess)
    io = resolve_io(inputs, outputs)
    out_names = [o.name for o in sess.get_outputs()]

    results, latencies = [], []
    for logical in feeds_per_sample:
        feeds = {io.role_to_name[role]: arr for role, arr in logical.items()}
        t0 = time.perf_counter()
        outs = sess.run(out_names, feeds)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        results.append(dict(zip(out_names, outs)))
    return io, results, latencies


def make_image(rng, size, image_path):
    if image_path:
        from PIL import Image
        return np.asarray(Image.open(image_path).convert("RGB"))
    # A seeded synthetic frame: realism doesn't matter for parity, only that ref
    # and candidate see the *same* pixels.
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="FP32 ONNX (with .onnx.data sidecar if split).")
    ap.add_argument("--precision", choices=("fp16", "bf16"), default="fp16",
                    help="Candidate TRT-EP precision. fp16 = deploy default; bf16 = experiment.")
    ap.add_argument("--engine-cache-dir", default="/tmp/smolvla_trt_cache")
    ap.add_argument("--model-id", default="lerobot/smolvla_base", help="HF id / local dir for tokenizer.")
    ap.add_argument("--num-samples", type=int, default=3, help="Distinct seeded input draws to compare.")
    ap.add_argument("--instruction", default="pick up the object")
    ap.add_argument("--state", default=None, help="Comma-separated state vector; default zeros.")
    ap.add_argument("--image", default=None, help="Optional image file; default seeded synthetic.")
    ap.add_argument("--cos-threshold", type=float, default=0.997,
                    help="Min cosine similarity on the action chunk to PASS.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    state = None
    if args.state and args.state.strip():
        state = np.asarray([float(p) for p in args.state.split(",")], dtype=np.float32)

    # We need the graph's dims to build inputs. Peek them from the ONNX I/O via a
    # throwaway ORT session, *before* building the (memory-heavy) reference run.
    import onnxruntime as ort
    peek = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    peek_in, peek_out = _specs(peek)
    io0 = resolve_io(peek_in, peek_out)
    del peek

    # Build all sample feeds ONCE so reference and candidate see identical inputs
    # (same image, tokens, state, and the same noise draw per sample).
    builder = InputBuilder(
        model_id=args.model_id, image_size=io0.image_size, lang_max_len=io0.lang_max_len,
        state_dim=io0.state_dim, chunk_size=io0.chunk_size, action_dim=io0.action_dim,
        fixed_noise=False, seed=args.seed,
    )
    img_rng = np.random.default_rng(args.seed + 1000)
    feeds_per_sample = [
        builder.build(make_image(img_rng, io0.image_size, args.image), args.instruction, state)
        for _ in range(args.num_samples)
    ]
    LOG.info("Prepared %d sample(s); action chunk = (%d, %d).",
             args.num_samples, io0.chunk_size, io0.action_dim)

    # Reference first (then freed), candidate second.
    ref_io, ref_outs = run_reference(args.onnx, feeds_per_sample)
    cand_io, cand_outs, latencies = run_candidate(
        args.onnx, feeds_per_sample, args.precision, args.engine_cache_dir)

    primary = cand_io.primary_output
    common = [n for n in cand_outs[0] if n in ref_outs[0]]
    LOG.info("Comparing outputs %s (primary action output: %s)", common, primary)

    # --- report ---
    title = f"PARITY: ORT/TRT-EP {args.precision}  vs  FP32 ONNX (CPU)"
    print(f"\n================  {title}  ================")
    worst_primary_cos = 1.0
    any_nonfinite = False
    for s in range(args.num_samples):
        print(f"\n-- sample {s}  (infer {latencies[s]:.1f} ms) " + ("-" * 32))
        for name in common:
            m = _compare(ref_outs[s][name], cand_outs[s][name])
            tag = " *ACTION*" if name == primary else ""
            flag = "" if m["finite"] else f"  <-- NON-FINITE (nan={m['n_nan']} inf={m['n_inf']})"
            print(f"  {name:<24}{tag:<9} cos={m['cosine']:.6f}  "
                  f"max_abs={m['max_abs']:.3e}  mean_abs={m['mean_abs']:.3e}  "
                  f"max_rel={m['max_rel']:.3e}{flag}")
            if name == primary:
                worst_primary_cos = min(worst_primary_cos, m["cosine"])
            any_nonfinite = any_nonfinite or not m["finite"]

    # head-to-head preview of the first action vector, last sample
    ref_a = np.asarray(ref_outs[-1][primary], np.float32).reshape(-1, cand_io.action_dim)[0]
    cand_a = np.asarray(cand_outs[-1][primary], np.float32).reshape(-1, cand_io.action_dim)[0]
    k = min(8, ref_a.size)
    print(f"\n  action[0][:{k}] FP32 ref       : [{', '.join(f'{v:+.4f}' for v in ref_a[:k])}]")
    print(f"  action[0][:{k}] {args.precision:<4} candidate : [{', '.join(f'{v:+.4f}' for v in cand_a[:k])}]")
    if latencies:
        print(f"\n  candidate infer: avg {np.mean(latencies):.1f} ms  "
              f"p95 {np.percentile(latencies, 95):.1f} ms")

    passed = (worst_primary_cos >= args.cos_threshold) and not any_nonfinite
    print("\n" + "=" * 69)
    print(f"  worst action cosine = {worst_primary_cos:.6f}  "
          f"(threshold {args.cos_threshold})   non-finite: {any_nonfinite}")
    print(f"  RESULT: {'PASS ✅' if passed else 'FAIL ❌'}")
    if not passed:
        print(f"  -> {args.precision} reduced precision diverged (likely softmax/layernorm overflow).")
        print("     trt_layer_norm_fp32_fallback is on by default; if it still fails, keep more")
        print("     sensitive ops in FP32 in the export, or fall those subgraphs back to the CUDA EP.")
    print("=" * 69 + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
