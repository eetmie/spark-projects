"""Score pi0.5 checkpoints on held-out episodes, the way eval/compare.py scores X-VLA.

    python excavator/score_heldout.py --sweep checkpoints/pi05_excavator_rtc/digging_complete_rtc20_10k
    python excavator/score_heldout.py --ckpt A=<step dir> --ckpt B=<step dir> --horizons 0.33 1.0

compare.py loads LeRobot policies only; this is its OpenPI twin. The metric, the baselines and
the table are the same, so the numbers mean the same thing:
  * one observation source at its native rate: the full recording the training copy was cut
    from, so every checkpoint sees byte-identical frames;
  * the predicted chunk (unnormalized, 30 fps) is compared with the recorded actions over a
    fixed wall-clock horizon;
  * disp_err is the integral of (predicted - recorded) rate command over the horizon, i.e.
    where the joint ends up; MAE per sample is reported too; move_ratio near 1 = right amount
    of motion;
  * zero-action and mean-action (mean over training episodes) baselines on the same points.

Everything comes from the checkpoint. Its `assets/<asset_id>` names the training dataset, and
that dataset's `meta/subset.json` (vla_common subset_v21) names the source recording and the
held-out episodes. Nothing is typed in here, so the split cannot drift.

Sampling: the stock sampler (RTC delay 0), 10 flow steps. The noise is drawn once per eval
point from a fixed seed and shared by every checkpoint, so differences between rows are the
weights, not the noise. --n-draws averages several draws, as in compare.py.

Continuations: a run restarted from params as `<run>-from-<k>` numbers its steps from 0.
--sweep picks up `<run>` and every `<run>-from-*`, and labels each checkpoint by global step.
"""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PI05 = HERE.parent
DEFAULT_STRIDE = 15


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", action="append", default=[], metavar="[LABEL=]STEP_DIR")
    p.add_argument("--sweep", type=pathlib.Path, default=None,
                   help="a run dir under checkpoints/<config>/; scores every kept step of it and of its -from-* continuations")
    p.add_argument("--config", default="pi05_excavator_rtc")
    p.add_argument("--datasets", type=pathlib.Path, default=None,
                   help="HF_LEROBOT_HOME (default: $HF_LEROBOT_HOME, else <pi05>/datasets)")
    p.add_argument("--horizons", type=float, nargs="+", default=[0.33, 1.0, 1.5],
                   help="seconds; 0.33 = one 10-row execute window, 1.0 = an X-VLA c30 chunk; pi0.5's 50 rows cover 1.67")
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="frames between eval start points")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-draws", type=int, default=1)
    p.add_argument("--only-task", default=None, metavar="INSTRUCTION")
    p.add_argument("--limit", type=int, default=None, help="score only the first N eval points (smoke test)")
    p.add_argument("--json-out", type=pathlib.Path, default=None)
    a = p.parse_args()
    if not a.ckpt and a.sweep is None:
        raise SystemExit("nothing to score: pass --ckpt <step dir> or --sweep <run dir>")
    return a


def step_dirs(run: pathlib.Path):
    """{label: step dir} for a run and its -from-<k> continuations, labelled by global step."""
    run = run.expanduser().resolve()
    out = {}
    for d in sorted([run, *run.parent.glob(run.name + "-from-*")]):
        if not d.is_dir():
            continue
        off = int(d.name.rsplit("-from-", 1)[1]) if "-from-" in d.name else 0
        for s in d.iterdir():
            if s.name.isdigit() and (s / "params").is_dir() and (s / "_CHECKPOINT_METADATA").exists():
                out[off + int(s.name)] = s
    if not out:
        raise SystemExit(f"no complete checkpoint under {run} or its -from-* continuations")
    return {f"@{g}": out[g] for g in sorted(out)}


def ckpt_label(spec: str):
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label, pathlib.Path(path).expanduser().resolve()
    path = pathlib.Path(spec).expanduser().resolve()
    return f"{path.parent.name}@{path.name}", path


def resolve_split(ckpts: dict, datasets: pathlib.Path):
    """Source recording and held-out episodes, read from the checkpoints. All must agree."""
    found = {}
    for label, ck in ckpts.items():
        ids = [d.name for d in (ck / "assets").iterdir() if d.is_dir()] if (ck / "assets").is_dir() else []
        if len(ids) != 1:
            raise SystemExit(f"{label}: expected one assets/<dataset> dir in {ck}, found {ids}")
        found[label] = ids[0]
    if len(set(found.values())) != 1:
        raise SystemExit(f"checkpoints trained on different datasets, score them separately: {found}")
    train_id = next(iter(found.values()))
    sub = datasets / train_id / "meta" / "subset.json"
    if not sub.exists():
        raise SystemExit(f"{train_id} has no meta/subset.json: it was not cut from a recording with a held-out "
                         f"split, so there is nothing honest to score")
    s = json.loads(sub.read_text())
    src = pathlib.Path(s["source"])
    if not src.is_dir():
        src = datasets / src.name
    held = sorted(int(e) for e in s["held_out"])
    train = sorted(int(e) for e in s["kept_source_episodes"])
    if set(held) & set(train):
        raise SystemExit(f"{sub}: held-out and kept episodes overlap")
    # Held-out episodes the source's audit FAILed (a frozen sensor, a stalled camera) are not
    # ground truth: scoring against them measures the fault, not the model.
    failed = []
    audit = src / "meta" / "audit.json"
    if audit.exists():
        failed = sorted(set(held) & set(json.loads(audit.read_text()).get("bad_episodes", [])))
        held = [e for e in held if e not in failed]
    else:
        print(f"!! {src.name} has no audit (python -m vla_common.dataset.audit {src}); scoring every held-out episode")
    return train_id, src, train, held, failed


def score(pred, gt, joints, fps):
    """Identical to eval/compare.py score(): pred, gt (N, T, D) rate commands on the source grid."""
    dt = 1.0 / fps
    mae = np.abs(pred - gt).mean()
    disp_err = np.abs((pred - gt).sum(axis=1) * dt)
    return {
        "mae": float(mae),
        "disp_err": float(disp_err.mean()),
        "disp_err_per_joint": {j: float(v) for j, v in zip(joints, disp_err.mean(axis=0))},
        "move_ratio": float(np.abs(pred).mean() / np.abs(gt).mean()),
    }


def main():
    a = parse_args()
    datasets = (a.datasets or pathlib.Path(os.environ.get("HF_LEROBOT_HOME", PI05 / "datasets"))).expanduser().resolve()
    os.environ["HF_LEROBOT_HOME"] = str(datasets)

    ckpts = dict(step_dirs(a.sweep)) if a.sweep else {}
    for spec in a.ckpt:
        label, path = ckpt_label(spec)
        ckpts[label] = path
    train_id, src, train_eps, held, failed = resolve_split(ckpts, datasets)

    # config.py reads the dataset name at import: it must name the training copy, whose
    # norm stats the checkpoint carries.
    os.environ["KAIVURI_DATASET"] = train_id
    sys.path.insert(0, str(HERE))
    import spark_env  # noqa: F401
    import config

    config.register()
    import jax
    import jax.numpy as jnp
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.models import model as _model
    from openpi.policies import policy_config
    from openpi.training import config as _config

    cfg = _config.get_config(a.config)
    H, fps_model = cfg.model.action_horizon, 30

    ds = LeRobotDataset(src.name, root=src)
    fps = int(ds.fps)
    joints = list(ds.meta.features["action"]["names"])
    actions = np.stack(ds.hf_dataset.with_format("numpy")["action"]).astype(np.float32)
    lo = np.asarray(ds.episode_data_index["from"]).astype(int)
    hi = np.asarray(ds.episode_data_index["to"]).astype(int)
    bounds = {e: (int(lo[e]), int(hi[e])) for e in range(len(lo))}
    if hi.max() != len(actions):
        raise SystemExit(f"{src}: episode index ends at {hi.max()} but data has {len(actions)} rows")
    tasks = {e: ds.meta.episodes[e]["tasks"] for e in bounds}
    if any(len(t) != 1 for t in tasks.values()):
        raise SystemExit("an episode carries more than one instruction; this script assumes one")
    tasks = {e: t[0] for e, t in tasks.items()}

    eps = held if a.only_task is None else [e for e in held if tasks[e] == a.only_task]
    if not eps:
        raise SystemExit(f"no held-out episode carries {a.only_task!r}; recorded: {sorted(set(tasks.values()))}")
    max_h = max(a.horizons)
    n_max = int(round(max_h * fps))
    points = [i for e in eps for i in range(bounds[e][0], bounds[e][1] - n_max, a.stride)]
    if a.limit:
        points = points[: a.limit]
    owner = {i: e for e in eps for i in range(*bounds[e])}
    gt = np.stack([actions[p : p + n_max] for p in points])

    print(f"source {src.name} @ {fps} fps, joints {joints}; trained on {train_id} ({len(train_eps)} episodes)")
    print(f"held out {len(held)} episodes (from {train_id}/meta/subset.json): {held}")
    if failed:
        print(f"skipped {len(failed)} held-out episodes the source's audit FAILed: {failed}")
    if a.only_task:
        print(f"scoring the {len(eps)} that carry {a.only_task!r}")
    counts = {}
    for p in points:
        counts[tasks[owner[p]]] = counts.get(tasks[owner[p]], 0) + 1
    print(f"{len(points)} eval points, stride {a.stride}; " + ", ".join(f"{t!r} x{n}" for t, n in counts.items()))
    print(f"checkpoints: {', '.join(f'{k} ({v})' for k, v in ckpts.items())}\n", flush=True)

    # Observations once, shared by every checkpoint (video decode is the slow part on CPU).
    def hwc(img):
        return (np.asarray(img).transpose(1, 2, 0) * 255).astype(np.uint8)

    t0 = time.time()
    obs = []
    for p in points:
        it = ds[p]
        obs.append({
            "observation/cam1": hwc(it["observation.images.cam1"]),
            "observation/cam2": hwc(it["observation.images.cam2"]),
            "observation/state": np.asarray(it["observation.state"]),
            "prompt": tasks[owner[p]],
        })
    print(f"decoded {len(obs)} observations in {time.time() - t0:.0f} s", flush=True)

    # One noise tensor per draw and point, shared by all checkpoints.
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((a.n_draws, len(points), H, cfg.model.action_dim)).astype(np.float32)

    results = {}
    for label, ck in ckpts.items():
        t0 = time.time()
        policy = policy_config.create_trained_policy(cfg, ck)
        # Batched Policy.infer: the same transforms per sample, one sampler call per batch.
        draws = []
        for d in range(a.n_draws):
            chunks = []
            for i in range(0, len(points), a.batch_size):
                group = [policy._input_transform(jax.tree.map(lambda x: x, o)) for o in obs[i : i + a.batch_size]]
                n = len(group)
                group += [group[-1]] * (a.batch_size - n)  # pad: one compiled shape
                batch = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *group)
                nz = noise[d, i : i + a.batch_size]
                nz = np.concatenate([nz, np.repeat(nz[-1:], a.batch_size - n, 0)])
                out = policy._sample_actions(jax.random.key(0), _model.Observation.from_dict(batch),
                                             noise=jnp.asarray(nz))
                out = np.asarray(out)
                st = np.asarray(batch["state"])
                for j in range(n):
                    chunks.append(policy._output_transform({"state": st[j], "actions": out[j]})["actions"])
            draws.append(np.stack(chunks))
        chunk = np.mean(draws, axis=0)
        del policy
        covered = chunk.shape[1] / fps_model
        print(f"[{label}] {len(points)} chunks {chunk.shape[1]} rows = {covered:.2f}s in {time.time() - t0:.0f} s",
              flush=True)
        results[label] = {"checkpoint": str(ck), "chunk": int(chunk.shape[1]), "covers_s": covered,
                          "n_draws": a.n_draws, "horizons": {}}
        for h in a.horizons:
            n = int(round(h * fps))
            if covered + 1e-6 < h:
                print(f"[{label}] chunk covers {covered:.2f}s, short of {h}s: not scored")
                continue
            results[label]["horizons"][f"{h}s"] = score(chunk[:, :n], gt[:, :n], joints, fps)

    mask = np.zeros(len(actions), bool)
    for e in train_eps:
        mask[slice(*bounds[e])] = True
    for label, const in [("zero-action", np.zeros(len(joints), np.float32)), ("mean-action", actions[mask].mean(0))]:
        results[label] = {"chunk": n_max, "covers_s": max_h, "horizons": {}}
        for h in a.horizons:
            n = int(round(h * fps))
            pred = np.broadcast_to(const, (len(points), n, len(joints)))
            results[label]["horizons"][f"{h}s"] = score(pred, gt[:, :n], joints, fps)

    width = max(14, max(len(k) for k in results) + 2)
    for h in a.horizons:
        key = f"{h}s"
        rows = [(k, r) for k, r in results.items() if key in r["horizons"]]
        print(f"\n{'=' * (31 + width)}\nhorizon {h}s   (lower is better; move_ratio near 1.0 = right amount of motion)")
        print(f"{'run':<{width}}{'MAE':>9}{'disp_err':>10}{'move_ratio':>12}")
        print("-" * (31 + width))
        rows.sort(key=lambda x: x[1]["horizons"][key]["disp_err"])
        for k, r in rows:
            s = r["horizons"][key]
            print(f"{k:<{width}}{s['mae']:>9.4f}{s['disp_err']:>10.4f}{s['move_ratio']:>12.2f}")
        print("per-joint displacement error:")
        print(f"{'run':<{width}}" + "".join(f"{j:>10}" for j in joints))
        for k, r in rows:
            pj = r["horizons"][key]["disp_err_per_joint"]
            print(f"{k:<{width}}" + "".join(f"{pj[j]:>10.4f}" for j in joints))

    out = a.json_out or ((a.sweep.expanduser().resolve() if a.sweep else next(iter(ckpts.values())).parent)
                         / "heldout_scores.json")
    out.write_text(json.dumps({"source": str(src), "train_dataset": train_id, "held_out_episodes": held,
                               "scored_episodes": eps, "skipped_audit_fail": failed, "only_task": a.only_task, "stride": a.stride,
                               "points": len(points), "runs": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
