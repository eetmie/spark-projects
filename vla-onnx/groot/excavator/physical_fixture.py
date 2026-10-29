"""Real-frame, physical-unit fixture for a fine-tuned GR00T N1.7 export.

Takes dataset observations (decoded camera frames, joint angles in degrees, task text),
injects seeded noise and runs NVIDIA's Gr00tPolicy, so the reference covers the whole
boundary: image transform, state normalization, model and action unnormalization. Saves
the raw inputs, noise and valve chunks, plus the checkpoint's normalization (contract).

    python physical_fixture.py --checkpoint ../runs/NAME --dataset ../datasets/VIEW \
        --samples 50:100 75:300 99:0 --task "scoop the sand ..." --out fixture_physical.npz

`--rtc-delays 0 6 12` (cycled over the samples) runs rtc_patch.py's training-time RTC sampler:
the committed rows are the recorded valves of the same frames, normalized and clipped as the
processor does, padded to the model width. Saved as `rtc_delay` [N] and `rtc_prefix` [N,H,D].

A checkpoint trained with GROOT_VIDEO_DELTAS="-k,0" also gets `raw_past` [N,V,H,W,3], each
camera k frames before `raw`. Samples need FRAME >= k: the loader does not pad by default, and
a negative index would read the end of the episode.
"""
import argparse
import json
import pathlib
import sys
from copy import deepcopy

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import finetune  # noqa: E402
import modality  # noqa: E402, F401
import rtc_patch  # noqa: E402
from open_loop import parse_observation  # noqa: E402


def contract(checkpoint: pathlib.Path, tag: str, mc) -> dict:
    """What a runtime needs to reproduce Gr00tPolicy's state/action boundary."""
    cfg = json.loads((checkpoint / "processor/processor_config.json").read_text())["processor_kwargs"]
    stats = json.loads((checkpoint / "processor/statistics.json").read_text())[tag]
    if cfg.get("apply_sincos_state_encoding") or mc["state"].mean_std_embedding_keys \
            or mc["action"].mean_std_embedding_keys or cfg.get("use_mean_std"):
        raise ValueError("only the min-max boundary is implemented")
    for ac in mc["action"].action_configs or []:
        if ac.rep.name != "ABSOLUTE":
            raise ValueError("relative actions are not implemented")
    lo, hi = ("q01", "q99") if cfg["use_percentiles"] else ("min", "max")
    side = lambda m, k: {"min": stats[m][k][lo], "max": stats[m][k][hi]}  # noqa: E731
    return {
        "embodiment": tag, "normalization": f"minmax[{lo},{hi}] -> [-1,1]",
        "clip_state": bool(cfg.get("clip_outliers", True)), "clip_action": True,
        "state": {k: side("state", k) for k in mc["state"].modality_keys},
        "action": {k: side("action", k) for k in mc["action"].modality_keys},
        "formalize_language": bool(cfg.get("formalize_language", True)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=pathlib.Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--samples", nargs="+", required=True, help="EPISODE:FRAME pairs")
    p.add_argument("--task", help="override the episode's task (the bundle bakes one)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32",
                   help="Gr00tPolicy casts model and inputs to bfloat16; float32 is the exact reference")
    p.add_argument("--rtc-delays", type=int, nargs="+",
                   help="training-time RTC: committed rows per sample, cycled (recorded valves as prefix)")
    p.add_argument("--out", type=pathlib.Path, required=True)
    a = p.parse_args()

    vlm_files = finetune.local_snapshot(finetune.COSMOS)
    finetune.vlm_files_from(vlm_files)
    finetune.vlm_from_config(vlm_files)

    import torch
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    import gr00t.policy.gr00t_policy as gp
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    tag = EmbodimentTag.NEW_EMBODIMENT
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=str(a.checkpoint), device="cuda")
    if a.dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        policy.model.to(torch.float32)
        cast = gp._rec_to_dtype
        gp._rec_to_dtype = lambda x, dtype: cast(x, torch.float32)
    mc = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(dataset_path=a.dataset, modality_configs=mc,
                                  video_backend="torchcodec", video_backend_kwargs=None)
    obs_mc = deepcopy(mc)
    obs_mc.pop("action")
    cfg = policy.model.config
    gen = torch.Generator().manual_seed(a.seed)
    ct = contract(a.checkpoint, tag.value, mc)
    rtc = {"delay": 0, "prefix": None}
    if a.rtc_delays:
        if max(a.rtc_delays) >= cfg.action_horizon or min(a.rtc_delays) < 0:
            raise SystemExit("--rtc-delays must lie in [0, action_horizon)")
        rtc_patch.apply(None)                       # per-row-time sampling; weights unchanged
        head = policy.model.action_head
        sampler = head.get_action_with_features

        def with_rtc(**kw):
            kw["action_input"]["rtc_prefix"] = rtc["prefix"]
            return sampler(**{**kw, "options": {"rtc_delay": rtc["delay"]}})
        head.get_action_with_features = with_rtc
        (side,) = ct["action"].values()
        a_lo, a_hi = np.asarray(side["min"]), np.asarray(side["max"])
    vkeys, skey = mc["video"].modality_keys, mc["state"].modality_keys[0]
    akey, lkey = mc["action"].modality_keys[0], mc["language"].modality_keys[0]
    deltas = list(mc["video"].delta_indices)
    if len(deltas) > 2:
        raise SystemExit(f"video deltas {deltas}: the board runtime handles [0] and [-k, 0]")

    out = {k: [] for k in ("raw", "state_deg", "noise", "actions", "gt", "episode", "frame")}
    if len(deltas) == 2:
        out["raw_past"] = []
    if a.rtc_delays:
        out |= {"rtc_delay": [], "rtc_prefix": []}
    tasks = []
    real_randn = torch.randn
    for k_sample, s in enumerate(a.samples):
        ep, fr = (int(x) for x in s.split(":"))
        if fr + min(deltas) < 0:
            raise SystemExit(f"sample {s}: frame {fr} has no frame {-min(deltas)} earlier in its episode")
        traj = loader[ep]
        gt = np.vstack(traj[f"action.{akey}"].to_numpy())[fr:fr + cfg.action_horizon]
        gt = np.pad(gt, ((0, cfg.action_horizon - len(gt)), (0, 0)), mode="edge")
        if a.rtc_delays:
            # gr00t.data.utils.normalize_values_minmax + clip_outliers, then zero padding.
            z = np.clip(2 * (gt - a_lo) / (a_hi - a_lo) - 1, -1.0, 1.0)
            pre = np.zeros((cfg.action_horizon, cfg.max_action_dim), np.float32)
            pre[:, :z.shape[1]] = z
            rtc["delay"] = a.rtc_delays[k_sample % len(a.rtc_delays)]
            rtc["prefix"] = torch.from_numpy(pre)[None].to("cuda")
        dp = extract_step_data(traj, fr, obs_mc, tag)
        obs = {f"state.{k}": v for k, v in dp.states.items()}
        obs |= {f"video.{k}": np.array(v) for k, v in dp.images.items()}
        task = a.task or str(dp.text)
        obs[lkey] = task
        noise = torch.randn(1, cfg.action_horizon, cfg.max_action_dim, generator=gen)
        torch.randn = lambda *sz, **kw: noise.clone().to(device=kw.get("device"), dtype=kw.get("dtype"))
        try:
            act, _ = policy.get_action(parse_observation(obs, mc))
        finally:
            torch.randn = real_randn
        video = np.stack([obs[f"video.{k}"] for k in vkeys], 1)      # [T,V,H,W,3], oldest first
        out["raw"].append(video[-1])                                  # current
        if len(deltas) == 2:
            out["raw_past"].append(video[0])                          # deltas[0] frames earlier
        out["state_deg"].append(obs[f"state.{skey}"][-1].astype(np.float32))
        out["noise"].append(noise.numpy()[0])
        out["actions"].append(np.asarray(act[akey])[0].astype(np.float32))
        out["gt"].append(gt.astype(np.float32))
        if a.rtc_delays:
            out["rtc_delay"].append(rtc["delay"])
            out["rtc_prefix"].append(pre)
        out["episode"].append(ep)
        out["frame"].append(fr)
        tasks.append(task)
        print(f"ep {ep} frame {fr}: state {out['state_deg'][-1].round(2)} row0 {out['actions'][-1][0].round(3)}"
              + (f" rtc delay {rtc['delay']}" if a.rtc_delays else ""))
    meta = {"checkpoint": str(a.checkpoint.resolve()), "dataset": a.dataset, "tasks": tasks,
            "cameras": vkeys, "video_delta_indices": deltas, "dtype": a.dtype, "contract": ct, "seed": a.seed,
            "rtc": {"sampler": "rtc_patch training-time", "delays": a.rtc_delays} if a.rtc_delays else None}
    np.savez_compressed(a.out, **{k: np.stack(v) for k, v in out.items()},
                        meta=np.array(json.dumps(meta)))
    (a.out.with_suffix(".contract.json")).write_text(json.dumps(meta["contract"], indent=1) + "\n")
    print("saved", a.out)


if __name__ == "__main__":
    main()
