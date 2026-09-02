#!/usr/bin/env python3
"""Run EVO1's deterministic Spark fixture on Orin and report split parity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from evo1_runtime.split_ort import Evo1SplitPolicy, process_memory, verify_bundle


def compare(name: str, expected: np.ndarray, actual: np.ndarray) -> dict:
    lhs = expected.astype(np.float64).reshape(-1)
    rhs = actual.astype(np.float64).reshape(-1)
    denominator = np.linalg.norm(lhs) * np.linalg.norm(rhs)
    cosine = float(np.dot(lhs, rhs) / denominator) if denominator else float("nan")
    return {
        "name": name,
        "cosine": cosine,
        "max_abs": float(np.max(np.abs(lhs - rhs))),
        "mean_abs": float(np.mean(np.abs(lhs - rhs))),
        "actual_min": float(rhs.min()),
        "actual_max": float(rhs.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("bundle"))
    parser.add_argument("--cache", type=Path, default=Path("trt_cache"))
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--threshold", type=float, default=0.999)
    args = parser.parse_args()

    bundle = verify_bundle(args.bundle)
    print("WARNING:", bundle["warning"], flush=True)
    before = process_memory()
    policy = Evo1SplitPolicy(
        args.bundle, args.cache, args.precision, allow_bootstrap=True
    )
    after_load = process_memory()
    with np.load(args.bundle / bundle["fixture"]["file"], allow_pickle=False) as fixture:
        output = policy.run_fixture(fixture)
        fused_valid = np.broadcast_to(
            fixture["context_mask"][..., None], fixture["expected_fused"].shape
        )
        reports = [
            compare("vision", fixture["expected_vision"], output["vision"]),
            compare(
                "fused_valid",
                fixture["expected_fused"][fused_valid],
                output["fused"][fused_valid],
            ),
            compare("action", fixture["expected_action"], output["action"]),
        ]
        fused_all = compare(
            "fused_all_diagnostic", fixture["expected_fused"], output["fused"]
        )
    document = {
        "status": "PASS"
        if all(item["cosine"] >= args.threshold for item in reports)
        else "FAIL",
        "threshold": args.threshold,
        "providers": {
            name: session.get_providers() for name, session in policy.sessions.items()
        },
        "memory": {"before": before, "after_load": after_load, "after_run": process_memory()},
        "load_timings_s": policy.load_timings_s,
        "inference_timings_s": output["timings_s"],
        "parity": reports,
        "diagnostics": {"masked_padding_excluded_from_gate": fused_all},
    }
    print(json.dumps(document, indent=2, sort_keys=True))
    if document["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
