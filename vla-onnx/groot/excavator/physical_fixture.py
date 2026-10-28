"""Real-frame, physical-unit fixture for a fine-tuned GR00T N1.7 export.

Takes dataset observations (decoded camera frames, joint angles in degrees, task text),
injects seeded noise and runs NVIDIA's Gr00tPolicy, so the reference covers the whole
boundary: image transform, state normalization, model and action unnormalization. Saves
the raw inputs, noise and valve chunks, plus the checkpoint's normalization (contract).

    python physical_fixture.py --checkpoint ../runs/NAME --dataset ../datasets/VIEW \
        --samples 50:100 75:300 99:0 --task "scoop the sand ..." --out fixture_physical.npz
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
    vkeys, skey = mc["video"].modality_keys, mc["state"].modality_keys[0]
    akey, lkey = mc["action"].modality_keys[0], mc["language"].modality_keys[0]

    out = {k: [] for k in ("raw", "state_deg", "noise", "actions", "gt", "episode", "frame")}
    tasks = []
    real_randn = torch.randn
    for s in a.samples:
        ep, fr = (int(x) for x in s.split(":"))
        traj = loader[ep]
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
        gt = np.vstack(traj[f"action.{akey}"].to_numpy())[fr:fr + cfg.action_horizon]
        out["raw"].append(np.stack([obs[f"video.{k}"][-1] for k in vkeys]))   # [V,H,W,3] current
        out["state_deg"].append(obs[f"state.{skey}"][-1].astype(np.float32))
        out["noise"].append(noise.numpy()[0])
        out["actions"].append(np.asarray(act[akey])[0].astype(np.float32))
        out["gt"].append(np.pad(gt, ((0, cfg.action_horizon - len(gt)), (0, 0)), mode="edge"))
        out["episode"].append(ep)
        out["frame"].append(fr)
        tasks.append(task)
        print(f"ep {ep} frame {fr}: state {out['state_deg'][-1].round(2)} row0 {out['actions'][-1][0].round(3)}")
    meta = {"checkpoint": str(a.checkpoint.resolve()), "dataset": a.dataset, "tasks": tasks,
            "cameras": vkeys, "dtype": a.dtype, "contract": contract(a.checkpoint, tag.value, mc), "seed": a.seed}
    np.savez_compressed(a.out, **{k: np.stack(v) for k, v in out.items()},
                        meta=np.array(json.dumps(meta)))
    (a.out.with_suffix(".contract.json")).write_text(json.dumps(meta["contract"], indent=1) + "\n")
    print("saved", a.out)


if __name__ == "__main__":
    main()
