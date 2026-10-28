"""Open-loop replay of dataset takes through a fine-tuned GR00T N1.7 checkpoint.

NVIDIA's Gr00tPolicy runs the checkpoint's own processor (crop/resize, state normalization,
action unnormalization), so this checks the whole PyTorch policy in physical units: every
`--stride` frames it predicts a chunk and compares it with the recorded valve commands.
Zero and dataset-mean chunks are the baselines. The observations, chunks and noise-free
summary are saved to OUT so engine runtimes can be checked on real frames.

    python open_loop.py --checkpoint ../runs/NAME --dataset ../datasets/VIEW --episodes 0 50 --out OUT
"""
import argparse
import json
import pathlib
import sys
import time
from copy import deepcopy

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import finetune  # noqa: E402  (VLM-from-config and cached-snapshot loading)
import modality  # noqa: E402, F401  (registers NEW_EMBODIMENT)


def parse_observation(obs: dict, mc: dict) -> dict:
    """Batch-of-one policy input (gr00t/eval/open_loop_eval.py, which imports matplotlib)."""
    out = {}
    for m in ("video", "state", "language"):
        out[m] = {}
        for k in mc[m].modality_keys:
            v = obs[k if m == "language" else f"{m}.{k}"]
            out[m][k] = [[v]] if isinstance(v, str) else v[None, :]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--episodes", type=int, nargs="+", default=[0])
    p.add_argument("--stride", type=int, default=40)
    p.add_argument("--max-frames", type=int, default=0, help="per episode, 0 = all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=pathlib.Path, required=True)
    a = p.parse_args()

    vlm_files = finetune.local_snapshot(finetune.COSMOS)
    finetune.vlm_files_from(vlm_files)
    finetune.vlm_from_config(vlm_files)

    import torch
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    torch.manual_seed(a.seed)
    tag = EmbodimentTag.NEW_EMBODIMENT
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=a.checkpoint, device="cuda")
    mc = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(dataset_path=a.dataset, modality_configs=mc,
                                  video_backend="torchcodec", video_backend_kwargs=None)
    obs_mc = deepcopy(mc)
    obs_mc.pop("action")
    akey = mc["action"].modality_keys[0]
    horizon = len(mc["action"].delta_indices)

    stats = json.loads((pathlib.Path(a.checkpoint) / "processor/statistics.json").read_text())
    mean = np.asarray(stats[tag.value]["action"][akey]["mean"], np.float32)

    a.out.mkdir(parents=True, exist_ok=True)
    rows, saved, times = [], {}, []
    for ep in a.episodes:
        traj = loader[ep]
        gt = np.vstack(traj[f"action.{akey}"].to_numpy()).astype(np.float32)
        n = len(traj) if a.max_frames <= 0 else min(len(traj), a.max_frames)
        preds, starts, obs_saved = [], [], []
        for t in range(0, n, a.stride):
            dp = extract_step_data(traj, t, obs_mc, tag)
            obs = {f"state.{k}": v for k, v in dp.states.items()}
            obs |= {f"video.{k}": np.array(v) for k, v in dp.images.items()}
            for lk in mc["language"].modality_keys:
                obs[lk] = dp.text
            parsed = parse_observation(obs, mc)
            t0 = time.perf_counter()
            act, _ = policy.get_action(parsed)
            times.append(time.perf_counter() - t0)
            preds.append(np.asarray(act[akey])[0].astype(np.float32))      # [H, A]
            starts.append(t)
            obs_saved.append(parsed)
        preds = np.stack(preds)
        err, err0, zero, base = [], [], [], []
        for t, c in zip(starts, preds):
            g = gt[t:t + horizon]
            err.append(np.abs(c[:len(g)] - g))
            err0.append(np.abs(c[0] - g[0]))
            zero.append(np.abs(g))
            base.append(np.abs(mean - g))
        cat = lambda xs: np.concatenate([x.reshape(-1, gt.shape[1]) for x in xs])  # noqa: E731
        row = {
            "episode": ep, "task": str(dp.text),
            "frames": n, "chunks": len(starts),
            "mae_chunk": cat(err).mean(0).round(4).tolist(), "mae_row0": cat(err0).mean(0).round(4).tolist(),
            "mae_zero": cat(zero).mean(0).round(4).tolist(), "mae_mean": cat(base).mean(0).round(4).tolist(),
        }
        rows.append(row)
        print(json.dumps(row))
        saved[f"ep{ep}_pred"] = preds
        saved[f"ep{ep}_starts"] = np.asarray(starts)
        saved[f"ep{ep}_gt"] = gt[:n + horizon]
        first = obs_saved[0]
        for k, v in first.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    saved[f"ep{ep}_obs0_{k}.{kk}"] = np.asarray(vv)
    np.savez_compressed(a.out / "open_loop.npz", **saved)
    keys = ("mae_chunk", "mae_row0", "mae_zero", "mae_mean")
    summary = {
        "checkpoint": str(pathlib.Path(a.checkpoint).resolve()), "dataset": a.dataset,
        "stride": a.stride, "horizon": horizon, "action_key": akey,
        "mean": {k: np.mean([r[k] for r in rows], 0).round(4).tolist() for k in keys},
        "latency_ms_p50": round(1000 * float(np.median(times[1:] or times)), 1),
        "episodes": rows,
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("mean", "latency_ms_p50")}))


if __name__ == "__main__":
    main()
