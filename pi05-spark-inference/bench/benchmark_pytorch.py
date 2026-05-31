#!/usr/bin/env python3
"""Benchmark π0.5 inference latency on GB10.

Loads an openpi policy (auto-detects PyTorch vs JAX by the presence of
model.safetensors in the checkpoint dir) and times policy.infer() over a
LIBERO-shaped observation. Reports latency percentiles and writes JSON.

Example:
    python bench/benchmark_pytorch.py \
        --config pi05_libero \
        --checkpoint checkpoints/pi05_libero_pytorch \
        --warmup 5 --iters 50
"""
import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_libero", help="openpi train config name")
    ap.add_argument("--checkpoint", required=True, help="checkpoint dir (PyTorch or JAX)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--duration", type=float, default=None,
                    help="if set, run timed iters for at least this many seconds (overrides --iters)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-steps", type=int, default=None,
                    help="flow-matching denoising steps (sample_kwargs); default = model default")
    ap.add_argument("--out", default=None, help="results JSON path")
    args = ap.parse_args()

    # Imports are slow (jax/torch); do them after arg parse for fast --help.
    import os
    import numpy as np  # noqa: F401
    from openpi.training import config as _config
    from openpi.policies import libero_policy
    # NOTE: we deliberately build the Policy directly instead of using
    # openpi.policies.policy_config.create_trained_policy, because that module
    # imports openpi.training.checkpoints -> data_loader -> lerobot (a heavy,
    # torch-pinning, training-only dep). The inference path needs none of it.
    from openpi.policies import policy as _policy
    from openpi.models_pytorch import pi0_pytorch
    from openpi import transforms
    from openpi.shared import normalize as _normalize
    import safetensors.torch as st

    try:
        import torch
        has_torch = True
    except Exception:
        has_torch = False

    train_config = _config.get_config(args.config)
    sample_kwargs = {"num_steps": args.num_steps} if args.num_steps is not None else None

    print(f"[load] config={args.config} checkpoint={args.checkpoint} device={args.device}")
    t0 = time.perf_counter()

    weight_path = os.path.join(args.checkpoint, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)
    if not is_pytorch:
        raise SystemExit(f"No model.safetensors in {args.checkpoint}; this harness is for the PyTorch path.")

    # Build the model and load weights with strict=False: save_model dedups the
    # tied embedding (language_model.embed_tokens.weight shares storage with
    # paligemma.lm_head.weight, which IS in the file), so strict=False loads with
    # missing=∅, unexpected=∅ and the tie auto-fills the embedding. openpi's own
    # load_pytorch uses strict=True and trips over this in our transformers build.
    model = pi0_pytorch.PI0Pytorch(config=train_config.model)
    missing, unexpected = st.load_model(model, weight_path, strict=False)
    if unexpected or any("embed_tokens" not in m for m in missing):
        raise SystemExit(f"Unexpected state_dict mismatch. missing={missing} unexpected={unexpected}")
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise SystemExit("Asset id required to load norm stats.")
    norm_stats = _normalize.load(os.path.join(args.checkpoint, "assets", data_config.asset_id))

    policy = _policy.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        pytorch_device=args.device,
        is_pytorch=True,
    )
    load_s = time.perf_counter() - t0
    print(f"[load] policy ready in {load_s:.1f}s (is_pytorch={is_pytorch})")

    example = libero_policy.make_libero_example()

    def sync():
        if has_torch and torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.synchronize()

    print(f"[warmup] {args.warmup} iters")
    for _ in range(args.warmup):
        out = policy.infer(example)
        sync()

    action_shape = list(np.asarray(out["actions"]).shape) if "actions" in out else None

    lat_ms = []
    if args.duration is not None:
        print(f"[bench] timed run for >= {args.duration:.0f}s")
        bench_start = time.perf_counter()
        while time.perf_counter() - bench_start < args.duration:
            sync()
            t = time.perf_counter()
            policy.infer(example)
            sync()
            lat_ms.append((time.perf_counter() - t) * 1e3)
    else:
        print(f"[bench] {args.iters} iters")
        for _ in range(args.iters):
            sync()
            t = time.perf_counter()
            policy.infer(example)
            sync()
            lat_ms.append((time.perf_counter() - t) * 1e3)

    lat_ms.sort()
    def pct(p): return lat_ms[min(len(lat_ms) - 1, int(p / 100 * len(lat_ms)))]
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "arch": platform.machine(),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "is_pytorch": is_pytorch,
        "num_steps": args.num_steps,
        "warmup": args.warmup,
        "iters": len(lat_ms),
        "duration_s": args.duration,
        "action_shape": action_shape,
        "load_s": round(load_s, 2),
        "latency_ms": {
            "mean": round(statistics.mean(lat_ms), 2),
            "median": round(statistics.median(lat_ms), 2),
            "p90": round(pct(90), 2),
            "p99": round(pct(99), 2),
            "min": round(lat_ms[0], 2),
            "max": round(lat_ms[-1], 2),
            "stdev": round(statistics.pstdev(lat_ms), 2),
        },
        "hz": round(1000.0 / statistics.mean(lat_ms), 2),
    }
    if has_torch and torch.cuda.is_available():
        summary["gpu"] = torch.cuda.get_device_name(0)
        summary["torch"] = torch.__version__

    print(json.dumps(summary, indent=2))

    out_path = Path(args.out) if args.out else Path("results") / (
        f"{args.config}_{'pt' if summary['is_pytorch'] else 'jax'}_"
        f"steps{args.num_steps or 'def'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
