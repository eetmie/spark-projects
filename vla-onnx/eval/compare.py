#!/usr/bin/env python
"""Compare VLA runs on a common time base.

The runs cannot be compared by training loss: variants may be trained on a
different temporal decimation of the same data, or on a different set of camera
features, so their action distributions, their normalization statistics and their
loss scales all differ. A model at 6 fps predicting 50 actions is describing 8.3 s
of machine motion; a model at 30 fps predicting 50 actions is describing 1.67 s.
Those numbers are not commensurable.

So this script evaluates every run the same way:

  * the observation is taken from ONE source dataset at its native rate, so every
    model sees byte-identical images and states,
  * each model is fed exactly the cameras it was trained on (read from its own
    config), so a 1-camera and a 2-camera run can be scored side by side,
  * each model predicts its action chunk at its own rate,
  * the chunk is expanded to a common grid by zero-order hold, which is how it
    would actually execute on the machine,
  * it is scored against the source ground-truth actions over a fixed wall-clock
    horizon.

Because the actions are rate commands, the headline metric is displacement
error: the integral of (predicted - actual) command over the horizon. That is
what decides whether the bucket ends up in the right place. Per-sample MAE is
reported too, but a model can have decent MAE and still drift.

Two trivial baselines are included. If a run does not beat them it has not
learned the task, and the comparison between runs is moot.

EVERYTHING IS READ FROM THE CHECKPOINT. There used to be a PRESETS table here holding,
per experiment, the source recording, the pinned instruction and a hand-written list of
held-out episodes. Every field of it drifted:

  * `digging_dry2` was written when that recording had 78 episodes. It grew to 242, and
    the preset went on scoring 8 of the 24 held-out episodes and reporting the result as
    the model's error -- silently, with a plausible table.
  * `digging_clean` listed 10 "held-out" episodes that the trainer's own rule puts in the
    TRAINING set. It was defused only because one queue script happened to export a
    matching override; running the trainer directly leaked the eval set with no error.
  * `digging_clean` also pinned "move the sand to the container" while its dataset says
    "move sand to container". The policy conditions on that embedding, so it was scored
    on an instruction it never saw. Well-formed prefix, no error, wrong numbers.

A checkpoint already records all of it, and `run_fps()` below was already reading the
run's own train_config.json for exactly this reason. So the table is gone: point this at
a checkpoint and the source, the split and the instructions come from the run itself.
Drift is not fixed here, it is made unrepresentable.

Usage:
    python compare.py --ckpt outputs/digging_demo/dry2_ir/checkpoints/001000
    python compare.py --sweep outputs/digging_demo --horizons 0.17 0.3
    python compare.py --ckpt A=<path> --ckpt B=<path> --only-task "move rock to container"
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from vla_common.dataset.split import episode_count

# Runs whose dataset fed observation.state as all zeros, with the normalizer patched to
# mean 0 / std 1 so normalization is the identity. Feeding them the source recording's
# real state would push raw joint values (|state| up to ~120) into a model that trained
# on zeros. This is the one thing a checkpoint does not record -- the dataset it names is
# gone-by-design -- so it stays a literal, keyed on the repo_id train_config.json carries.
STATE_BLIND_REPOS = {"local/masi_kaivuri_nostate"}

DEFAULT_STRIDE = 15


@dataclass(frozen=True)
class RunSpec:
    """What one checkpoint says about how it was trained.

    Built by `resolve_run`. Replaces the old hand-written Preset: every field here is
    read out of the run rather than typed next to it.
    """

    ckpt: Path              # .../checkpoints/<step>/pretrained_model
    src: Path               # source recording, native fps, ALL cameras
    train_episodes: list[int]
    val_episodes: list[int]
    repo_id: str
    policy_type: str
    fps: int                # rate the checkpoint's own dataset was clocked at


def resolve_run(ckpt: Path) -> RunSpec:
    """Read a checkpoint's training setup out of its own train_config.json.

    THE SOURCE RECORDING IS RESOLVED THROUGH THE SYMLINK. `dataset.root` names what the
    run trained on, which for a camera-subset run is a *variant* -- a metadata-only view
    whose `data/` is a symlink into the recording (vla_common.dataset.camera_variant).
    Scoring must use the recording itself, because it is the only copy that still has
    every camera, and a two-camera run and a one-camera run have to be fed from the same
    frames to be comparable. Resolving `<root>/data` and taking its parent gives the
    recording for a variant and the root itself for a plain dataset, with no table.

    HELD OUT IS THE COMPLEMENT, NEVER A LIST. `dataset.episodes` is what the trainer was
    told to train on; everything else in the recording was held out by construction. That
    is what makes this immune to the drift that killed the presets.
    """
    ckpt = Path(ckpt).expanduser().resolve()
    if ckpt.name != "pretrained_model" and (ckpt / "pretrained_model").is_dir():
        ckpt = ckpt / "pretrained_model"
    cfg_path = ckpt / "train_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"{ckpt}: no train_config.json -- not a LeRobot checkpoint?")
    cfg = json.loads(cfg_path.read_text())

    root = Path(cfg["dataset"]["root"]).expanduser()
    if not root.exists():
        raise SystemExit(
            f"{ckpt}: trained on {root}, which no longer exists. The recording is the "
            f"only copy of the source data -- it cannot be scored without it.")
    src = (root / "data").resolve().parent

    train_eps = cfg["dataset"].get("episodes")
    n = episode_count(src)
    if train_eps is None:
        raise SystemExit(
            f"{ckpt}: train_config.json records no dataset.episodes, so it trained on the "
            f"whole recording and nothing was held out. There is no honest split to score.")
    train_eps = sorted(int(e) for e in train_eps)
    val_eps = [e for e in range(n) if e not in set(train_eps)]
    if not val_eps:
        raise SystemExit(f"{ckpt}: trained on all {n} episodes -- nothing held out to score.")

    info = json.loads((root / "meta" / "info.json").read_text())
    return RunSpec(ckpt=ckpt, src=src, train_episodes=train_eps, val_episodes=val_eps,
                   repo_id=cfg["dataset"]["repo_id"], policy_type=cfg["policy"]["type"],
                   fps=int(info["fps"]))


def agree(specs: dict) -> RunSpec:
    """One RunSpec for a set of runs, or a hard stop.

    Blending runs that held out different episodes, or that trained on different
    recordings, produces a table whose rows are not comparable -- which is the exact
    failure this rewrite exists to remove, so it is an error and not a warning.
    """
    first_label, first = next(iter(specs.items()))
    for label, spec in specs.items():
        if spec.src != first.src:
            raise SystemExit(
                f"{label} trained on {spec.src.name} but {first_label} trained on "
                f"{first.src.name}. Scoring them in one table would compare different data.")
        if spec.val_episodes != first.val_episodes:
            raise SystemExit(
                f"{label} held out {len(spec.val_episodes)} episodes and {first_label} held "
                f"out {len(first.val_episodes)}. One of them would be scored on episodes it "
                f"trained on. Score them separately.")
    return first


def _ckpt_arg(spec: str):
    """`LABEL=PATH` or a bare PATH whose run directory names it."""
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label, Path(path).expanduser()
    path = Path(spec).expanduser()
    parts = [q for q in path.parts if q not in ("pretrained_model", "checkpoints")]
    # .../<run>/checkpoints/<step>/pretrained_model -> "<run>@<step>"
    return (f"{parts[-2]}@{parts[-1]}" if len(parts) >= 2 else path.name), path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", action="append", default=[], metavar="[LABEL=]PATH",
                   help="a checkpoint to score; repeat for a multi-run table")
    p.add_argument("--sweep", type=Path, default=None,
                   help="score every run under this sweep dir at --checkpoint")
    p.add_argument("--checkpoint", default="last", help="which checkpoint --sweep picks (default: last)")
    p.add_argument("--only-task", default=None, metavar="INSTRUCTION",
                   help="score only the held-out episodes carrying this instruction")
    p.add_argument("--horizons", type=float, nargs="+", default=[1.5, 4.0], help="horizons in seconds")
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                   help=f"spacing between eval start points, in source frames (default {DEFAULT_STRIDE})")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-draws", type=int, default=None,
                   help="average this many flow-matching noise draws per eval point. Default is 1 "
                        "for a single architecture and 4 when the table spans more than one: "
                        "X-VLA gains ~18.5%% from averaging draws and SmolVLA ~0%%, so scoring a "
                        "mixed table at 1 charges X-VLA for sampler variance SmolVLA does not have")
    p.add_argument("--device", default="cuda")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()
    if not args.ckpt and args.sweep is None:
        raise SystemExit("nothing to score: pass --ckpt <path> or --sweep <dir>")
    return args


def collect_ckpts(args) -> dict:
    """{label: checkpoint dir} from --ckpt and --sweep."""
    out = {}
    if args.sweep is not None:
        sweep = args.sweep.expanduser()
        if not sweep.is_dir():
            raise SystemExit(f"no such sweep dir: {sweep}")
        for run in sorted(d for d in sweep.iterdir() if (d / "checkpoints").is_dir()):
            ck = run / "checkpoints" / args.checkpoint / "pretrained_model"
            if ck.exists():
                out[f"{run.name}@{args.checkpoint}"] = ck
        if not out:
            raise SystemExit(f"no run under {sweep} has a '{args.checkpoint}' checkpoint")
    for spec in args.ckpt:
        label, path = _ckpt_arg(spec)
        out[label] = path
    return out


def source_meta(spec):
    """Native fps and action joint names of the source recording."""
    info = json.loads((spec.src / "meta" / "info.json").read_text())
    return int(info["fps"]), list(info["features"]["action"]["names"])


def read_shards(dirpath):
    """Concatenate every parquet shard under `dirpath`, in file order.

    LeRobot v3 rolls over to a new shard once one passes `data_files_size_in_mb`
    (100 MB) -- `data/chunk-000/file-001.parquet`, and a matching second file under
    `meta/episodes/`. Every recording up to masi_digging_dry fitted in a single shard,
    so this used to read `chunk-000/file-000.parquet` by name. masi_digging_dry_2 has
    two, and reading only the first drops episodes 63-77 -- the entire "move rock to
    container" task -- out of the ground truth while the table still prints plausible
    numbers for the episodes that survived. Shard order is the row order the global
    `dataset_from_index`/`dataset_to_index` are counted in, so sorted() is load-bearing,
    not cosmetic (the names are zero-padded, so it is a numeric sort).
    """
    files = sorted(Path(dirpath).rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet shards under {dirpath}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_ground_truth(spec):
    df = read_shards(spec.src / "data")
    eps = read_shards(spec.src / "meta" / "episodes")
    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
    bounds = {
        int(r["episode_index"]): (int(r["dataset_from_index"]), int(r["dataset_to_index"]))
        for _, r in eps.iterrows()
    }
    # The shards must line up end to end with the global indices the episode table
    # counts in; if they ever do not, every eval window is silently off by a shard.
    span = max(hi for _, hi in bounds.values())
    if span != len(actions):
        raise SystemExit(
            f"{spec.src}: episode table ends at row {span} but data has {len(actions)} rows "
            f"-- shard concatenation is out of step with dataset_from_index/dataset_to_index")
    return actions, bounds


def episode_tasks(spec):
    """{episode_index: instruction} straight from the recording's own metadata."""
    out = {}
    for _, r in read_shards(spec.src / "meta" / "episodes").iterrows():
        names = [str(t) for t in r["tasks"]]
        if len(names) != 1:
            raise SystemExit(f"episode {int(r['episode_index'])} has {len(names)} tasks {names}; "
                             "this script assumes one instruction per episode")
        out[int(r["episode_index"])] = names[0]
    return out


def select_episodes(spec, only_task):
    """Held-out episodes, optionally narrowed to one instruction.

    This is what the old `digging_dry2_sand` / `_rock` presets were: a slice of one run's
    held-out set, so the halves of a two-task table decompose. As a preset it went stale
    the moment the recording grew -- the rock slice was pinned to 2 episodes when the
    session had come to hold 10. Derived from the recording, it cannot.
    """
    if only_task is None:
        return spec.val_episodes
    per_ep = episode_tasks(spec)
    recorded = sorted(set(per_ep.values()))
    if only_task not in recorded:
        raise SystemExit(f"--only-task {only_task!r} is not in {spec.src.name}: {recorded}")
    picked = [e for e in spec.val_episodes if per_ep[e] == only_task]
    if not picked:
        raise SystemExit(f"no held-out episode carries {only_task!r} "
                         f"(held out: {len(spec.val_episodes)} episodes)")
    return picked


def tasks_for_points(spec, bounds, val_episodes, points):
    """The instruction to feed at each eval start point.

    Every point takes ITS OWN episode's instruction. The old code could also pin one
    string across a whole preset, which on a two-task recording scored every rock episode
    under the sand instruction -- and because the policy conditions on the language
    embedding, that produces a plausible table rather than an error. There is no reason to
    ever want it, so the option is gone.
    """
    per_ep = episode_tasks(spec)
    owner = {}
    for ep in val_episodes:
        start, stop = bounds[ep]
        owner.update(dict.fromkeys(range(start, stop), per_ep[ep]))
    return [owner[p] for p in points]


def eval_points(bounds, val_episodes, max_horizon_s, src_fps, stride=DEFAULT_STRIDE):
    """Source-frame indices to evaluate from, with a full horizon left in the episode."""
    need = int(round(max_horizon_s * src_fps))
    points = []
    for ep in val_episodes:
        start, stop = bounds[ep]
        points += [i for i in range(start, stop - need, stride)]
    return points


def load_policy(ckpt):
    """Load whatever architecture this checkpoint is, by its own recorded type.

    Comparing SmolVLA against X-VLA means the loader cannot be hardcoded to one class.
    Both expose `predict_action_chunk`, `config.image_features` and the pre/post
    processors, so everything downstream is architecture-agnostic.
    """
    policy_type = json.loads((ckpt / "train_config.json").read_text())["policy"]["type"]
    return policy_type, get_policy_class(policy_type).from_pretrained(ckpt)


def run_fps(ckpt, src_fps):
    """Playback rate of the dataset this checkpoint was trained on.

    Read from the run's own train_config.json -> dataset root -> meta/info.json,
    rather than a hardcoded repo_id table: a new run used to be a KeyError here,
    which is exactly the moment you least want the eval to fall over.
    """
    cfg = json.loads((ckpt / "train_config.json").read_text())
    root = cfg["dataset"].get("root")
    if root and (Path(root) / "meta" / "info.json").exists():
        return int(json.loads((Path(root) / "meta" / "info.json").read_text())["fps"])
    return src_fps


def predict_chunks(ckpt, src_ds, points, batch_size, device, spec, src_fps, point_tasks,
                   n_draws=1):
    """Run one model over all eval points. Returns (N, chunk, action_dim) unnormalized
    actions and the fps its chunk is clocked at.

    Both architectures here are flow-matching samplers seeded by a random draw, so a
    single pass scores one sample of a distribution rather than the policy itself. How
    much that matters is NOT the same for the two: measured on digging_dry2, averaging
    4 draws moves SmolVLA by -0.4% (its sampler is effectively deterministic) and X-VLA
    by -18.5%, and pulls X-VLA's move_ratio from 1.131 to 1.043. Scoring both at
    n_draws=1 therefore charges X-VLA for sampler variance that SmolVLA does not have.

    Default stays 1 so every number recorded before 2026-09-01 still reproduces. Pass
    --n-draws 4 when the question is "which architecture fits better" rather than "what
    does one forward pass do". Note that n_draws > 1 is NOT free at deploy time: it costs
    a full denoise pass each, and on the Orin denoise is 86% of wall clock.
    """
    # Seed BEFORE load_policy, exactly where the single-draw version seeded. Model
    # construction itself consumes the RNG (xavier/normal init runs before the checkpoint
    # is loaded over it), so seeding after this line would shift the noise draw and
    # silently move every number recorded before n_draws existed.
    torch.manual_seed(0)
    policy_type, policy = load_policy(ckpt)
    policy.eval().to(device)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt)
    repo_id = json.loads((ckpt / "train_config.json").read_text())["dataset"]["repo_id"]
    fps = run_fps(ckpt, src_fps)
    state_blind = repo_id in STATE_BLIND_REPOS

    # Feed exactly the cameras this checkpoint was trained on. An IR-only run and an
    # IR+RGB run are then scored on the same frames without either seeing an input it
    # never saw in training.
    cam_keys = list(policy.config.image_features)
    missing = [k for k in cam_keys if k not in src_ds.meta.features]
    if missing:
        raise SystemExit(f"{ckpt} expects {missing}, absent from {spec.src}")

    draws = []
    for draw in range(n_draws):
        # Draw 0 deliberately inherits the RNG state left by load_policy above, which is
        # what the single-draw version sampled from -- so --n-draws 1 reproduces every
        # earlier number bit for bit. Extra draws get their own seeds.
        if draw:
            torch.manual_seed(1000 + draw)
        chunks = []
        for i in range(0, len(points), batch_size):
            group = points[i : i + batch_size]
            items = [src_ds[j] for j in group]
            state = torch.stack([it["observation.state"] for it in items])
            if state_blind:
                state = torch.zeros_like(state)
            batch = {"observation.state": state, "task": point_tasks[i : i + batch_size]}
            for key in cam_keys:
                batch[key] = torch.stack([it[key] for it in items])
            with torch.no_grad():
                chunks.append(post(policy.predict_action_chunk(pre(batch))).float().cpu())
        draws.append(torch.cat(chunks))
    del policy
    torch.cuda.empty_cache()
    # Average in unnormalized action space, which is where the commands are executed.
    return torch.stack(draws).mean(0).numpy(), fps, cam_keys, policy_type


def to_30hz(chunk, fps, n_samples, src_fps):
    """Zero-order hold a chunk predicted at `fps` onto the source grid."""
    reps = src_fps // fps
    expanded = np.repeat(chunk, reps, axis=1)
    if expanded.shape[1] < n_samples:
        raise ValueError(f"chunk covers {expanded.shape[1] / src_fps:.2f}s, need {n_samples / src_fps:.2f}s")
    return expanded[:, :n_samples]


def score(pred, gt, joints, src_fps):
    """pred, gt: (N, T, action_dim) rate commands on the source grid."""
    dt = 1.0 / src_fps
    mae = np.abs(pred - gt).mean()
    # Integrated command over the horizon: where the joint actually ends up.
    disp_err = np.abs((pred - gt).sum(axis=1) * dt)
    return {
        "mae": float(mae),
        "disp_err": float(disp_err.mean()),
        "disp_err_per_joint": {j: float(v) for j, v in zip(joints, disp_err.mean(axis=0))},
        "move_ratio": float(np.abs(pred).mean() / np.abs(gt).mean()),
    }


def main():
    args = parse_args()
    ckpts = collect_ckpts(args)

    # Resolve every run first, then insist they describe the same experiment. Doing this
    # before any inference means a mismatched table fails in a second rather than after
    # the GPU has chewed through the first model.
    specs = {label: resolve_run(path) for label, path in ckpts.items()}
    spec = agree(specs)

    n_draws = args.n_draws
    if n_draws is None:
        n_draws = 4 if len({s.policy_type for s in specs.values()}) > 1 else 1

    src_fps, joints = source_meta(spec)
    actions, bounds = load_ground_truth(spec)
    val_episodes = select_episodes(spec, args.only_task)
    max_h = max(args.horizons)
    points = eval_points(bounds, val_episodes, max_h, src_fps, args.stride)
    if not points:
        raise SystemExit(f"no eval point survives a {max_h}s horizon in episodes {val_episodes}")
    point_tasks = tasks_for_points(spec, bounds, val_episodes, points)

    json_out = args.json_out
    if json_out is None:
        json_out = (args.sweep.expanduser() if args.sweep else spec.ckpt.parent) / "comparison.json"

    print(f"source {spec.src.name} @ {src_fps}fps, joints {joints}")
    print(f"held out {len(spec.val_episodes)} of {len(spec.train_episodes) + len(spec.val_episodes)} "
          f"episodes, derived from the run's own dataset.episodes")
    if args.only_task:
        print(f"scoring the {len(val_episodes)} of them that carry {args.only_task!r}")
    if n_draws > 1:
        why = "mixed architectures" if args.n_draws is None else "requested"
        print(f"averaging {n_draws} noise draws per eval point ({why})")
    print(f"evaluating on {len(points)} start points from episodes {val_episodes}")
    counts = {t: point_tasks.count(t) for t in sorted(set(point_tasks))}
    print("instructions: " + ", ".join(f"{t!r} x{n}" for t, n in counts.items()) + "\n")

    src_ds = LeRobotDataset(repo_id="local/src", root=spec.src, video_backend="torchcodec")

    # Ground truth on the source grid, per horizon.
    n_max = int(round(max_h * src_fps))
    gt_full = np.stack([actions[p : p + n_max] for p in points])
    act_dim = gt_full.shape[-1]

    results = {}
    for name, ckpt in ckpts.items():
        print(f"[{name}] running inference ...", flush=True)
        chunk, fps, cams, ptype = predict_chunks(
            specs[name].ckpt, src_ds, points, args.batch_size, args.device, spec, src_fps,
            point_tasks, n_draws=n_draws)
        covered = chunk.shape[1] / fps
        cam_short = [c.rsplit(".", 1)[-1] for c in cams]
        print(f"[{name}] {ptype}, cameras {cam_short}, chunk {chunk.shape[1]} @ {fps}fps = {covered:.2f}s")
        results[name] = {"fps": fps, "chunk": int(chunk.shape[1]), "covers_s": covered,
                         "n_draws": n_draws, "checkpoint": str(specs[name].ckpt),
                         "cameras": cam_short, "policy": ptype, "horizons": {}}
        for h in args.horizons:
            n = int(round(h * src_fps))
            if covered + 1e-6 < h:
                # A run whose chunk is shorter than the horizon used to vanish from the
                # table with no explanation. Say so instead.
                print(f"[{name}] chunk covers {covered:.2f}s, short of the {h}s horizon -- not scored")
                continue
            results[name]["horizons"][f"{h}s"] = score(
                to_30hz(chunk, fps, n, src_fps), gt_full[:, :n], joints, src_fps)

    # Trivial baselines on the same points and horizons. The mean is taken over the
    # training episodes only, so the baseline gets no more information than the models.
    train_mask = np.zeros(len(actions), bool)
    for ep in spec.train_episodes:
        train_mask[slice(*bounds[ep])] = True
    train_mean = actions[train_mask].mean(axis=0)
    for label, const in [("zero-action", np.zeros(act_dim, np.float32)), ("mean-action", train_mean)]:
        results[label] = {"fps": src_fps, "chunk": n_max, "covers_s": max_h, "cameras": [],
                          "policy": "-", "horizons": {}}
        for h in args.horizons:
            n = int(round(h * src_fps))
            pred = np.broadcast_to(const, (len(points), n, act_dim))
            results[label]["horizons"][f"{h}s"] = score(pred, gt_full[:, :n], joints, src_fps)

    width = max(14, max(len(n) for n in results) + 2)
    for h in args.horizons:
        key = f"{h}s"
        rows = [(n, r) for n, r in results.items() if key in r["horizons"]]
        if not rows:
            continue
        print(f"\n{'=' * (83 + width)}\nhorizon {h}s   (lower is better; move_ratio near 1.0 = right amount of motion)")
        print(f"{'run':<{width}}{'policy':>9}{'cams':>12}{'fps':>5}{'chunk':>7}{'covers':>8}{'MAE':>9}{'disp_err':>10}{'move_ratio':>12}")
        print("-" * (83 + width))
        for name, r in sorted(rows, key=lambda x: x[1]["horizons"][key]["disp_err"]):
            s = r["horizons"][key]
            print(
                f"{name:<{width}}{r.get('policy','?'):>9}{'+'.join(r['cameras']) or '-':>12}"
                f"{r['fps']:>5}{r['chunk']:>7}{r['covers_s']:>7.1f}s"
                f"{s['mae']:>9.4f}{s['disp_err']:>10.4f}{s['move_ratio']:>12.2f}"
            )
        print("\nper-joint displacement error:")
        print(f"{'run':<{width}}" + "".join(f"{j:>10}" for j in joints))
        for name, r in sorted(rows, key=lambda x: x[1]["horizons"][key]["disp_err"]):
            pj = r["horizons"][key]["disp_err_per_joint"]
            print(f"{name:<{width}}" + "".join(f"{pj[j]:>10.4f}" for j in joints))

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps({
        "source": str(spec.src),
        "held_out_episodes": spec.val_episodes,
        "scored_episodes": val_episodes,
        "only_task": args.only_task,
        "runs": results,
    }, indent=2))
    print(f"\nwrote {json_out}")


if __name__ == "__main__":
    main()
