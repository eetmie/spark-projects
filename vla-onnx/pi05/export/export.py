"""Export a fine-tuned OpenPI pi0.5 checkpoint (converted to PyTorch) in the pinned container.

Adapted from jetson-orin-nano-vla export/pi05/export.py, which is pinned to the LIBERO weights.
Here the checkpoint is ours: its hashes are recorded instead of checked, fixtures are real dataset
frames (PI05_FRAMES from excavator/fixture_frames.py) and, with discrete_state_input, the bundle
carries the state prompt's token pieces and the `kaivuri` runtime contract (BUNDLE.md).
"""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def link_or_copy(source, dest):
    """Share large weights when possible, including across Docker bind mounts."""
    try:
        os.link(source, dest)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(source, dest)
    return dest


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def finalize(bundle, initial, frames_path, tokenizer, lock):
    import numpy as np
    from prompt import StatePrompt
    info_path = bundle / "bundle.json"
    info = json.loads(info_path.read_text())
    references = json.loads((initial / "manifest.json").read_text())
    frames = np.load(frames_path)
    tasks = [str(t) for t in frames["tasks"]]
    if info["policy"] == "pi05_openpi_state":
        prompt = StatePrompt(tokenizer)
        verified = sum(prompt.verify(t, info["state_dim"]) for t in tasks)
        info["tokenizer"] = {"sep": prompt.sep, "neg": prompt.neg, "pieces": prompt.pieces,
                             "tail": prompt.tail, "verified_compositions": verified}
        info["tasks"] = {t: {"head": prompt.head(t)} for t in tasks}
        for fixture in references["fixtures"]:
            with np.load(initial / f"policy_fixture_{fixture['id']}.npz") as arrays:
                ids = arrays["tokens"].reshape(-1)[arrays["token_mask"].reshape(-1)].tolist()
            # The reference tokenized its normalized state; the composed prompt must match it.
            assert ids[:len(info["tasks"][fixture["prompt"]]["head"])] == info["tasks"][fixture["prompt"]]["head"]
            assert ids[-len(prompt.tail):] == prompt.tail
    else:
        info["tasks"] = {}
        for fixture in references["fixtures"]:
            with np.load(initial / f"policy_fixture_{fixture['id']}.npz") as arrays:
                info["tasks"][fixture["prompt"]] = {"tokens": arrays["tokens"].reshape(-1).tolist(),
                                                    "mask": arrays["token_mask"].reshape(-1).tolist()}
    features = json.loads(str(frames["features"]))
    provenance = json.loads(str(frames["provenance"]))
    info["kaivuri"] = {
        "fps": int(frames["fps"]),
        "tasks": tasks,
        "state": {"names": features["observation.state"], "units": "deg",
                  "discrete_state_input": references["discrete_state_input"]},
        "action": {"names": features["action"][:info["robot_action_dim"]], "range": [-1.0, 1.0]},
        "action_horizon": info["action_horizon"], "num_steps": info["num_steps"],
        "cameras": [{"slot": slot, "dataset_key": key, "camera": camera, "rotation_deg": rotation}
                    for slot, key, camera, rotation in zip(info["image_keys"], ("observation.images.cam1", "observation.images.cam2"),
                                                           provenance["camera_order"], provenance["camera_rotations_deg"])],
        "profile_files": provenance["profile_files"],
        "recorder_revision": provenance["revision"],
        "preprocess": "OpenPI resize_with_pad to 224 (antialiased), gray repeated to 3 channels, x/255*2-1",
    }
    info["schema"] = "pi05-split/1"
    info["source_lock"] = lock
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    paths = sorted(p for p in bundle.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    (bundle / "MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(bundle)}\n" for p in paths))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--from-compact", action="store_true",
                    help="reuse a finished initial/baseline in --work-dir; rebuild only the compact bundle")
    a = ap.parse_args(argv)
    a.checkpoint, a.frames, a.out, a.work_dir = (p.resolve() for p in (a.checkpoint, a.frames, a.out, a.work_dir))
    if a.out.exists() or (a.work_dir.exists() and not a.from_compact):
        ap.error("--out and --work-dir must be new directories")
    if a.work_dir.is_relative_to(a.out) or a.out.is_relative_to(a.work_dir):
        ap.error("--out and --work-dir must be separate")
    weights = a.checkpoint / "model.safetensors"
    from weight_digest import tensor_digest
    lock = {"openpi_commit": "c23745b5ad24e98f66967ea795a07b2588ed6c79",
            "policy_config": os.environ.get("PI05_CONFIG", "pi05_excavator"),
            "converted_checkpoint_sha256": sha(weights),
            "converted_active_tensors_sha256": tensor_digest(weights, ["paligemma_with_expert.gemma_expert.lm_head.weight"]),
            "frames_sha256": sha(a.frames)}
    initial, baseline = a.work_dir / "initial", a.work_dir / "baseline"
    if not a.from_compact:
        initial.mkdir(parents=True)
        baseline.mkdir()
        from download import download as download_locked
    tokenizer_lock = json.loads((HERE / "tokenizer-lock.json").read_text())
    if not a.from_compact:
        download_locked(tokenizer_lock, a.work_dir / "tokenizer")
    lock["tokenizer"] = tokenizer_lock
    tokenizer = a.work_dir / "tokenizer" / "paligemma_tokenizer.model"
    env = {**os.environ, "PI05_CHECKPOINT": str(a.checkpoint), "PI05_INITIAL_ROOT": str(initial),
           "PI05_BASELINE_ROOT": str(baseline), "PYTHONNOUSERSITE": "1", "PI05_FRAMES": str(a.frames),
           "PI05_TOKENIZER": str(tokenizer)}
    for script, stage in [("prepare_blocks.py", initial), ("prepare_conditioning.py", initial),
                          ("prepare_full.py", baseline), ("prepare_compact.py", a.out)]:
        if a.from_compact and stage != a.out:
            continue
        if stage == a.out:
            a.out.mkdir(parents=True)
            shutil.copytree(baseline / "weights", a.out / "weights", copy_function=link_or_copy)
            shutil.copytree(baseline / "templates", a.out / "templates")
            link_or_copy(baseline / "embedding_fp16.npy", a.out / "embedding_fp16.npy")
        print(f"Export stage {script}: {stage}", flush=True)
        with (a.work_dir / f"{Path(script).stem}.log").open("w" if a.from_compact else "x") as log:
            subprocess.run([sys.executable, str(HERE / script)], env={**env, "PI05_STAGE_ROOT": str(stage)},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    finalize(a.out, initial, a.frames, tokenizer, lock)
    print(f"Complete compact bundle: {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
