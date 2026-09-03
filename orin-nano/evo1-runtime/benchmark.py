#!/usr/bin/env python3
"""Benchmark EVO1 runtime variants against the fixed native 32-step fixture."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from evo1_runtime.split_ort import (
    Evo1SplitPolicy,
    _MemorySampler,
    process_memory,
    verify_bundle,
)


def similarity(reference: np.ndarray, actual: np.ndarray) -> dict:
    lhs = reference.astype(np.float64).reshape(-1)
    rhs = actual.astype(np.float64).reshape(-1)
    denominator = np.linalg.norm(lhs) * np.linalg.norm(rhs)
    cosine = float(np.dot(lhs, rhs) / denominator) if denominator else float("nan")
    return {
        "cosine": cosine,
        "max_abs": float(np.max(np.abs(lhs - rhs))),
        "mean_abs": float(np.mean(np.abs(lhs - rhs))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("bundle"))
    parser.add_argument("--cache", type=Path, default=Path("trt_cache"))
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--embedding-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--action-hot", type=Path)
    parser.add_argument("--device-resident-action", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.runs <= 0
        or args.warmup < 0
        or (args.steps is not None and args.steps <= 0)
    ):
        parser.error(
            "runs and steps must be positive and warmup must be non-negative"
        )

    bundle = verify_bundle(args.bundle)
    steps = (
        int(bundle["num_inference_timesteps"])
        if args.steps is None
        else args.steps
    )
    print("WARNING:", bundle["warning"], flush=True)
    before = process_memory()
    policy = Evo1SplitPolicy(
        args.bundle,
        args.cache,
        args.precision,
        embedding_device=args.embedding_device,
        action_hot=args.action_hot,
        device_resident_action=args.device_resident_action,
        allow_bootstrap=True,
    )
    after_load = process_memory()
    fixture = np.load(args.bundle / bundle["fixture"]["file"], allow_pickle=False)
    try:
        for _ in range(args.warmup):
            policy.run_fixture(fixture, steps=steps)

        sampler = _MemorySampler()
        sampler.start()
        wall_ms: list[float] = []
        stage_ms: dict[str, list[float]] = {}
        output = None
        for _ in range(args.runs):
            started = time.perf_counter()
            output = policy.run_fixture(fixture, steps=steps)
            wall_ms.append((time.perf_counter() - started) * 1000)
            for name, seconds in output["timings_s"].items():
                stage_ms.setdefault(name, []).append(seconds * 1000)
        low_water = sampler.finish()
        assert output is not None
        mean_ms = statistics.fmean(wall_ms)
        action_parity = similarity(fixture["expected_action"], output["action"])
        document = {
            "status": (
                "PASS" if action_parity["cosine"] >= args.threshold else "FAIL"
            ),
            "threshold": args.threshold,
            "variant": {
                "precision": args.precision,
                "embedding_device": args.embedding_device,
                "cuda_fallback": not bool(os.environ.get("TRT_DROP_CUDA_EP")),
                "steps": steps,
                "action_hot": str(args.action_hot) if args.action_hot else None,
                "device_resident_action": args.device_resident_action,
                "warmup": args.warmup,
                "runs": args.runs,
            },
            "latency_ms": {
                "mean": mean_ms,
                "p50": float(np.percentile(wall_ms, 50)),
                "p95": float(np.percentile(wall_ms, 95)),
                "min": min(wall_ms),
                "max": max(wall_ms),
                "achieved_hz": 1000.0 / mean_ms,
            },
            "stage_mean_ms": {
                name: statistics.fmean(values) for name, values in stage_ms.items()
            },
            "action_vs_native_32_step": action_parity,
            "memory": {
                "before": before,
                "after_load": after_load,
                "after_benchmark": process_memory(),
                "benchmark_low_water": low_water,
            },
            "providers": {
                name: session.get_providers()
                for name, session in policy.sessions.items()
            },
        }
    finally:
        fixture.close()
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    if document["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
