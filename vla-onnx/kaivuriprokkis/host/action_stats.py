"""Action-chunk statistics for RTC, from the training dataset (runs in the OpenPI venv).

    python host/action_stats.py --step CHECKPOINT_STEP_DIR --out WORK/action_stats.npz [--config pi05_excavator]

Reads the LeRobot v2.1 parquet files directly (no video decode), normalizes `action` with
the checkpoint's own quantile statistics exactly as OpenPI does, and takes every full
in-episode window of `action_horizon` rows. Windows that would run past the end of an
episode are skipped (OpenPI pads them with the last action, which is not the motion's
statistics). Output: mean [H*D] and covariance [H*D, H*D] in float64, flattened row-major
(row, dim), plus provenance. The exporter turns this into the bundle's rtc/ files.

Tied to H, the action rate and the action dims: changing any of them invalidates it.
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "pi05" / "excavator"))
import rtc  # noqa: E402
import spark_env  # noqa: E402, F401  (HF_LEROBOT_HOME)
import config  # noqa: E402

config.register()
from openpi.training import config as _config  # noqa: E402


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--step", type=pathlib.Path, required=True, help="OpenPI checkpoint step dir (assets/)")
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--config", default="pi05_excavator")
    a = p.parse_args()

    cfg = _config.get_config(a.config)
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    horizon = cfg.model.action_horizon
    dims = cfg.data.action_dim
    root = pathlib.Path(spark_env.os.environ["HF_LEROBOT_HOME"]) / data.repo_id
    stats_path = a.step / "assets" / data.asset_id / "norm_stats.json"
    stats = json.loads(stats_path.read_text())["norm_stats"]["actions"]
    if not data.use_quantile_norm:
        raise SystemExit("pi0.5 excavator configs use quantile normalization")
    q01 = np.asarray(stats["q01"], np.float64)[:dims]
    q99 = np.asarray(stats["q99"], np.float64)[:dims]
    info = json.loads((root / "meta" / "info.json").read_text())

    chunks, episodes, short = [], 0, 0
    for path in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
        table = pq.read_table(path, columns=["action", "episode_index", "frame_index"])
        if len(set(table.column("episode_index").to_pylist())) != 1:
            raise SystemExit(f"{path}: more than one episode")
        frames = np.asarray(table.column("frame_index").to_pylist())
        if not np.array_equal(frames, np.arange(len(frames))):
            raise SystemExit(f"{path}: frame_index is not 0..n-1")
        raw = np.stack(table.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
        if raw.shape[1] < dims:
            raise SystemExit(f"{path}: action has {raw.shape[1]} dims, config wants {dims}")
        # OpenPI Normalize (use_quantiles): float32 data against float64 stats, numpy promotion.
        norm = (raw[:, :dims] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        episodes += 1
        n = len(norm) - horizon + 1
        if n <= 0:
            short += 1
            continue
        windows = np.lib.stride_tricks.sliding_window_view(norm, (horizon, dims))[:, 0]
        chunks.append(windows)
    if not chunks:
        raise SystemExit(f"no episode in {root} is at least {horizon} frames long")
    chunks = np.concatenate(chunks)
    mean, cov = rtc.chunk_moments(chunks)
    eig = np.linalg.eigvalsh(cov)
    meta = {"dataset": data.repo_id, "dataset_info_sha256": sha(root / "meta" / "info.json"),
            "norm_stats_sha256": sha(stats_path), "fps": info["fps"], "rows": horizon, "dims": dims,
            "action_names": info["features"]["action"].get("names", [])[:dims],
            "space": "OpenPI quantile-normalized actions, flattened row-major (row, dim)",
            "chunks": int(len(chunks)), "episodes": episodes, "episodes_shorter_than_horizon": short,
            "covariance_ddof": 1, "covariance_min_eigenvalue": float(eig.min()),
            "covariance_max_eigenvalue": float(eig.max())}
    np.savez(a.out, mean=mean, covariance=cov, meta=np.asarray(json.dumps(meta)))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
