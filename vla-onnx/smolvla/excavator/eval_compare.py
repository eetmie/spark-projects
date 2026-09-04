#!/usr/bin/env python
"""Compare SmolVLA runs on a common time base.

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

Usage:
    python eval_compare.py --preset digging
    python eval_compare.py --preset kaivuri --runs A B --horizons 1.5
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from vla_common.paths import dataset, playbook

import numpy as np
import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

ROOT = playbook("smolvla")


@dataclass
class Preset:
    """One evaluation setup: which recording is ground truth, and how to slice it."""

    src: Path                      # source dataset, always at native fps with ALL cameras
    out_dir: Path                  # sweep dir holding the runs
    task: str | None               # instruction fed to every model; None = per-episode
    val_episodes: list[int]        # held out from training, scored here
    stride_eval: int               # spacing between eval start points, in source frames
    state_blind_repos: set[str] = field(default_factory=set)


PRESETS = {
    # DEPRECATED as a source of results (2026-08-19): this recording predates the
    # frame-arrival pacing fix (kaivuriprokkis b8b2ccd), so its loop timing slipped while
    # `timestamp` reported a flawless frame_index/fps. Do not draw conclusions from runs
    # trained on it. Kept as a REGRESSION FIXTURE for this script — A@020000 must still
    # score disp_err 0.1320 against a 0.3901 zero-action baseline.
    # The 2026-08-12/13 sweep: one camera, 4-dim state, blocks task. Start points land
    # on multiples of 15 so the same source frame exists in the 30/10/6 fps variants.
    "kaivuri": Preset(
        src=dataset("vanhat/masi_kaivuri_juusto"),  # moved out of Desktop/ 2026-08-19
        out_dir=ROOT / "outputs/excavator",
        task="scoop blocks and dump it to the left",
        val_episodes=[3, 11, 19, 27],
        stride_eval=15,
        # Runs trained on a state-blind dataset saw observation.state == 0 at every frame,
        # and their normalizer was patched to mean 0 / std 1, which makes normalization the
        # identity. Feeding them the source dataset's real state would push raw joint values
        # (|state| up to ~120) straight into the model instead of the zeros it trained on.
        state_blind_repos={"local/masi_kaivuri_nostate"},
    ),
    # The 2026-08-19 camera-count sweep: IR-only vs IR+RGB, 3-dim state (slew left out,
    # its yaw origin drifts), sand-to-container task. All runs are 30 fps, so the stride
    # only controls how many start points we score.
    "digging": Preset(
        src=dataset("masi_digging"),
        out_dir=ROOT / "outputs/digging",
        task="move the sand to the container",
        val_episodes=[5, 15, 25, 35, 45, 55, 65, 75],
        stride_eval=15,
    ),
    # The 2026-08-20 re-run on the GROWN dataset: 189 episodes / 113781 frames. The first
    # 82 episodes are byte-identical to the sweep above (verified: they still sum to 41765
    # frames) and 107 were appended, so `digging` above remains reproducible and this
    # held-out list is a clean superset of it. Same task, same 3-dim state, same cameras.
    # val_episodes must stay in step with run_digging.sh, which derives the same
    # range(5, N_EPS, 10) from the dataset's own info.json.
    "digging189": Preset(
        src=dataset("masi_digging"),
        out_dir=ROOT / "outputs/digging189",
        task="move the sand to the container",
        val_episodes=list(range(5, 189, 10)),
        stride_eval=15,
    ),
    # Boundary dead air trimmed off every episode and the dirty 83-90 block skipped
    # (built by make_trim_variant.py): 181 episodes / 103356 frames.
    # val_episodes are the RENUMBERED ids of the same recordings digging189 held out,
    # less source ep 85 which fell inside the dropped block -- so scores here are
    # directly comparable to that sweep. Source ep -> clean ep is x if x < 83 else x-8.
    "digging_clean": Preset(
        src=ROOT / "datasets/masi_digging_clean",
        out_dir=ROOT / "outputs/digging_clean",
        task="move the sand to the container",
        val_episodes=[5, 15, 25, 35, 45, 55, 65, 75,
                      87, 97, 107, 117, 127, 137, 147, 157, 167, 177],
        stride_eval=15,
    ),
    # The 2026-08-28 dry-sand session: 62 episodes / 100490 frames, same rig and
    # schema as digging (cam1 IR + cam2 RGB, 3-dim state, 4-dim action). Episodes are
    # ~2.9x longer than digging's (1621 vs 571 frames), so 62 episodes carry nearly the
    # same frame budget -- and make_trim_variant reports only 0.2% cuttable, hence no
    # _clean variant to point at: src is the Desktop recording itself.
    #
    # THE TASK STRING IS NOT THE SAME as every other digging preset above. This
    # recording says "move sand to container"; the others say "move the sand to the
    # container". SmolVLA conditions on the language embedding, so scoring a dry run
    # with the older phrasing feeds it an instruction it never trained on -- and
    # nothing errors: the prefix is well-formed and the numbers come out plausible but
    # wrong. Copied from meta/tasks.parquet, not from the preset above.
    "digging_dry": Preset(
        src=dataset("masi_digging_dry"),
        out_dir=ROOT / "outputs/digging_dry",
        task="move sand to container",
        val_episodes=list(range(5, 62, 10)),
        stride_eval=15,
    ),
    # The 2026-08-31 session: 78 episodes / 65655 frames, same rig and schema again --
    # and the FIRST TWO-TASK recording. Episodes 0-62 are "move sand to container",
    # 63-77 are "move rock to container", one instruction per episode.
    #
    # `task=None` is the point of this preset: it makes tasks_for_points feed each
    # episode its OWN instruction. Pinning a string the way every preset above does
    # would score all 15 rock episodes under the sand instruction and quietly report
    # the result as the model's error.
    #
    # Held out is the usual every-10th-from-5, which lands on 5/15/25/35/45/55 in the
    # sand block and 65/75 in the rock block -- both tasks are represented, roughly in
    # proportion (2 of 15 rock, 6 of 63 sand). Keep it in step with run_digging.sh,
    # which derives range(5, 78, 10) from the dataset's own info.json.
    #
    # It is also the first recording to span TWO parquet shards; see read_shards.
    "digging_dry2": Preset(
        src=dataset("masi_digging_dry_2"),
        out_dir=ROOT / "outputs/digging_dry2",
        task=None,
        val_episodes=list(range(5, 78, 10)),
        stride_eval=15,
    ),
    # The same run and the same held-out episodes as digging_dry2, split by instruction.
    # The headline table blends both tasks into one mean, which cannot say whether the
    # model learned one instruction and is dragging the other along: 87 of the 409 points
    # are rock, so a rock-only failure would move the blended number by very little.
    # These two score the halves separately. They still set task=None -- each episode gets
    # its own instruction, exactly as in the combined run, so the numbers decompose.
    # Rock rests on only 2 held-out episodes: read it as a smoke test, not a tight estimate.
    "digging_dry2_sand": Preset(
        src=dataset("masi_digging_dry_2"),
        out_dir=ROOT / "outputs/digging_dry2",
        task=None,
        val_episodes=[5, 15, 25, 35, 45, 55],
        stride_eval=15,
    ),
    "digging_dry2_rock": Preset(
        src=dataset("masi_digging_dry_2"),
        out_dir=ROOT / "outputs/digging_dry2",
        task=None,
        val_episodes=[65, 75],
        stride_eval=15,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), default="digging")
    p.add_argument("--runs", nargs="*", default=None, help="run names (default: every run with a checkpoint)")
    p.add_argument("--checkpoint", default="last", help="checkpoint to load (default: last)")
    p.add_argument("--horizons", type=float, nargs="+", default=[1.5, 4.0], help="horizons in seconds")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-draws", type=int, default=1,
                   help="average this many flow-matching noise draws per eval point "
                        "(default 1). Use 4 to compare architectures without charging "
                        "X-VLA for sampler variance SmolVLA does not have")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", type=Path, default=None, help="override the preset's sweep dir")
    p.add_argument("--extra-runs", nargs="*", default=[], metavar="LABEL=PATH",
                   help="runs from another sweep dir (e.g. xvla_ir=/path/to/xvla/outputs/digging/ir), "
                        "so architectures kept in separate projects still score in one table")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()
    args.preset = PRESETS[args.preset]
    if args.out_dir is None:
        args.out_dir = args.preset.out_dir
    if args.json_out is None:
        args.json_out = args.out_dir / "comparison.json"
    return args


def source_meta(preset):
    """Native fps and action joint names of the source recording."""
    info = json.loads((preset.src / "meta" / "info.json").read_text())
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


def load_ground_truth(preset):
    df = read_shards(preset.src / "data")
    eps = read_shards(preset.src / "meta" / "episodes")
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
            f"{preset.src}: episode table ends at row {span} but data has {len(actions)} rows "
            f"-- shard concatenation is out of step with dataset_from_index/dataset_to_index")
    return actions, bounds


def episode_tasks(preset):
    """{episode_index: instruction} straight from the recording's own metadata."""
    out = {}
    for _, r in read_shards(preset.src / "meta" / "episodes").iterrows():
        names = [str(t) for t in r["tasks"]]
        if len(names) != 1:
            raise SystemExit(f"episode {int(r['episode_index'])} has {len(names)} tasks {names}; "
                             "this script assumes one instruction per episode")
        out[int(r["episode_index"])] = names[0]
    return out


def tasks_for_points(preset, bounds, points):
    """The instruction to feed at each eval start point.

    SmolVLA conditions on the language embedding, so the instruction is an input like
    any other -- and a wrong-but-well-formed one produces a plausible table rather than
    an error (that is how the "move the sand to the container" / "move sand to
    container" mismatch could have gone unnoticed). Hence both branches here check
    themselves against the recording:

    * single-task preset -- `task` is pinned, and must be the string the dataset
      actually carries, and the dataset must really only have one.
    * multi-task preset -- `task=None`, and each point takes ITS OWN episode's
      instruction. Pinning one string across a two-task recording would score every
      rock episode under the sand instruction, which is not a number about anything.
    """
    per_ep = episode_tasks(preset)
    recorded = sorted(set(per_ep.values()))

    if preset.task is not None:
        if len(recorded) > 1:
            raise SystemExit(
                f"{preset.src} carries {len(recorded)} tasks {recorded}, but the preset pins "
                f"one ({preset.task!r}). Set task=None to score each episode with its own.")
        if preset.task not in recorded:
            raise SystemExit(f"preset task {preset.task!r} is not in {preset.src}: {recorded}")
        return [preset.task] * len(points)

    owner = {}
    for ep in preset.val_episodes:
        start, stop = bounds[ep]
        owner.update(dict.fromkeys(range(start, stop), per_ep[ep]))
    return [owner[p] for p in points]


def eval_points(preset, bounds, max_horizon_s, src_fps):
    """Source-frame indices to evaluate from, with a full horizon left in the episode."""
    need = int(round(max_horizon_s * src_fps))
    points = []
    for ep in preset.val_episodes:
        start, stop = bounds[ep]
        points += [i for i in range(start, stop - need, preset.stride_eval)]
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


def predict_chunks(ckpt, src_ds, points, batch_size, device, preset, src_fps, point_tasks,
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
    state_blind = repo_id in preset.state_blind_repos

    # Feed exactly the cameras this checkpoint was trained on. An IR-only run and an
    # IR+RGB run are then scored on the same frames without either seeing an input it
    # never saw in training.
    cam_keys = list(policy.config.image_features)
    missing = [k for k in cam_keys if k not in src_ds.meta.features]
    if missing:
        raise SystemExit(f"{ckpt} expects {missing}, absent from {preset.src}")

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
    preset = args.preset
    src_fps, joints = source_meta(preset)
    actions, bounds = load_ground_truth(preset)
    max_h = max(args.horizons)
    points = eval_points(preset, bounds, max_h, src_fps)
    point_tasks = tasks_for_points(preset, bounds, points)
    print(f"source {preset.src.name} @ {src_fps}fps, joints {joints}")
    if args.n_draws > 1:
        print(f"averaging {args.n_draws} noise draws per eval point")
    print(f"evaluating on {len(points)} start points from held-out episodes {preset.val_episodes}")
    counts = {t: point_tasks.count(t) for t in sorted(set(point_tasks))}
    print("instructions: " + ", ".join(f"{t!r} x{n}" for t, n in counts.items()) + "\n")

    runs = args.runs
    if runs is None:
        runs = sorted(d.name for d in args.out_dir.iterdir() if (d / "checkpoints" / args.checkpoint).exists())
    run_dirs = {name: args.out_dir / name for name in runs}
    for spec in args.extra_runs:
        if "=" not in spec:
            raise SystemExit(f"--extra-runs wants LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        run_dirs[label] = Path(path).expanduser().resolve()
    if not run_dirs:
        raise SystemExit(f"no runs with a '{args.checkpoint}' checkpoint under {args.out_dir}")

    src_ds = LeRobotDataset(repo_id="local/src", root=preset.src, video_backend="torchcodec")

    # Ground truth on the source grid, per horizon.
    n_max = int(round(max_h * src_fps))
    gt_full = np.stack([actions[p : p + n_max] for p in points])
    act_dim = gt_full.shape[-1]

    results = {}
    for name, run_dir in run_dirs.items():
        ckpt = run_dir / "checkpoints" / args.checkpoint / "pretrained_model"
        if not ckpt.exists():
            print(f"[{name}] no checkpoint at {ckpt}, skipping")
            continue
        print(f"[{name}] running inference ...", flush=True)
        chunk, fps, cams, ptype = predict_chunks(
            ckpt, src_ds, points, args.batch_size, args.device, preset, src_fps, point_tasks,
            n_draws=args.n_draws)
        covered = chunk.shape[1] / fps
        cam_short = [c.rsplit(".", 1)[-1] for c in cams]
        print(f"[{name}] {ptype}, cameras {cam_short}, chunk {chunk.shape[1]} @ {fps}fps = {covered:.2f}s")
        results[name] = {"fps": fps, "chunk": int(chunk.shape[1]), "covers_s": covered,
                         "n_draws": args.n_draws,
                         "cameras": cam_short, "policy": ptype, "horizons": {}}
        for h in args.horizons:
            n = int(round(h * src_fps))
            if covered + 1e-6 < h:
                continue
            results[name]["horizons"][f"{h}s"] = score(
                to_30hz(chunk, fps, n, src_fps), gt_full[:, :n], joints, src_fps)

    # Trivial baselines on the same points and horizons. The mean is taken over the
    # training episodes only, so the baseline gets no more information than the models.
    train_mask = np.ones(len(actions), bool)
    for ep in preset.val_episodes:
        train_mask[slice(*bounds[ep])] = False
    train_mean = actions[train_mask].mean(axis=0)
    for label, const in [("zero-action", np.zeros(act_dim, np.float32)), ("mean-action", train_mean)]:
        results[label] = {"fps": src_fps, "chunk": n_max, "covers_s": max_h, "cameras": [],
                          "policy": "-", "horizons": {}}
        for h in args.horizons:
            n = int(round(h * src_fps))
            pred = np.broadcast_to(const, (len(points), n, act_dim))
            results[label]["horizons"][f"{h}s"] = score(pred, gt_full[:, :n], joints, src_fps)

    for h in args.horizons:
        key = f"{h}s"
        rows = [(n, r) for n, r in results.items() if key in r["horizons"]]
        if not rows:
            continue
        print(f"\n{'=' * 97}\nhorizon {h}s   (lower is better; move_ratio near 1.0 = right amount of motion)")
        print(f"{'run':<14}{'policy':>9}{'cams':>12}{'fps':>5}{'chunk':>7}{'covers':>8}{'MAE':>9}{'disp_err':>10}{'move_ratio':>12}")
        print("-" * 97)
        for name, r in sorted(rows, key=lambda x: x[1]["horizons"][key]["disp_err"]):
            s = r["horizons"][key]
            print(
                f"{name:<14}{r.get('policy','?'):>9}{'+'.join(r['cameras']) or '-':>12}"
                f"{r['fps']:>5}{r['chunk']:>7}{r['covers_s']:>7.1f}s"
                f"{s['mae']:>9.4f}{s['disp_err']:>10.4f}{s['move_ratio']:>12.2f}"
            )
        print("\nper-joint displacement error:")
        print(f"{'run':<14}" + "".join(f"{j:>10}" for j in joints))
        for name, r in sorted(rows, key=lambda x: x[1]["horizons"][key]["disp_err"]):
            pj = r["horizons"][key]["disp_err_per_joint"]
            print(f"{name:<14}" + "".join(f"{pj[j]:>10.4f}" for j in joints))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
