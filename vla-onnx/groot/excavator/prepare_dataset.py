"""GR00T view of an excavator LeRobot v2.1 dataset.

    python prepare_dataset.py SRC_V21_DATASET OUT_DIR [--state-dims N] [--action-dims N]

GR00T reads LeRobot v2 plus meta/modality.json, and it rewrites meta/stats.json in place
when q01/q99 are missing (LeRobot's are). So it gets its own directory: data/ and videos/
are symlinks to the source, meta/ is a copy without stats.json, plus modality.json. The
source dataset (shared with OpenPI training) is never written.

The v3.0 -> v2.1 step is `vla_common.dataset.to_v21`. Videos must be H.264 (torchcodec).

`--action-bounds fixed` writes meta/stats.json itself: GR00T's own statistics, except the action's
min/max/q01/q99 are the valves' physical bounds -1/+1. GR00T keeps a complete stats.json, and its
min-max normalization is then the identity: no percentile clipping, neutral zero stays zero.
The empirical statistics are kept beside it as meta/stats_empirical.json. State is unchanged.
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess

COPY = ("info.json", "episodes.jsonl", "tasks.jsonl")


def video_codec(path: pathlib.Path) -> str:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src", type=pathlib.Path)
    p.add_argument("out", type=pathlib.Path)
    p.add_argument("--state-dims", type=int, help="leading observation.state dims to use (default: all)")
    p.add_argument("--action-dims", type=int, default=4, help="leading action dims: 4 valves, 6 with tracks")
    p.add_argument("--cameras", nargs="+", default=["cam1", "cam2"])
    p.add_argument("--action-bounds", choices=("dataset", "fixed"), default="dataset",
                   help="dataset: GR00T's q01/q99 (computed at training); fixed: valves' physical -1/+1")
    a = p.parse_args()

    src = a.src.resolve()
    info = json.loads((src / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise SystemExit(f"{src}: codebase_version {info.get('codebase_version')}, expected v2.1 (run to_v21)")
    feats = info["features"]
    state_n = feats["observation.state"]["shape"][0]
    action_n = feats["action"]["shape"][0]
    state_dims = a.state_dims or state_n
    if not (0 < state_dims <= state_n and 0 < a.action_dims <= action_n):
        raise SystemExit(f"dims out of range: state {state_dims}/{state_n}, action {a.action_dims}/{action_n}")
    for cam in a.cameras:
        key = f"observation.images.{cam}"
        if feats.get(key, {}).get("dtype") != "video":
            raise SystemExit(f"{key} is not a video feature")
        first = next((src / "videos").glob(f"chunk-*/{key}/episode_*.mp4"))
        if (codec := video_codec(first)) != "h264":
            raise SystemExit(f"{first}: {codec}; GR00T's torchcodec path is verified for H.264 only")

    if a.out.exists():
        raise SystemExit(f"{a.out} exists; use a new directory")
    (a.out / "meta").mkdir(parents=True)
    for name in ("data", "videos"):
        # relative, so the view survives the tree being moved
        (a.out / name).symlink_to(os.path.relpath(src.resolve() / name, a.out.resolve()), target_is_directory=True)
    for name in COPY:
        shutil.copy2(src / "meta" / name, a.out / "meta" / name)

    modality = {
        "state": {"joints": {"start": 0, "end": state_dims}},
        "action": {"valves": {"start": 0, "end": a.action_dims}},
        "video": {cam: {"original_key": f"observation.images.{cam}"} for cam in a.cameras},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    (a.out / "meta" / "modality.json").write_text(json.dumps(modality, indent=2) + "\n")
    names = lambda k, n: (feats[k].get("names") or [])[:n]  # noqa: E731
    provenance = {"source": str(src), "fps": info["fps"], "episodes": info["total_episodes"],
                  "frames": info["total_frames"], "state_names": names("observation.state", state_dims),
                  "action_names": names("action", a.action_dims), "cameras": a.cameras}
    if a.action_bounds == "fixed":
        from gr00t.data.stats import calculate_dataset_statistics

        floats = [k for k, f in feats.items() if "float" in f["dtype"]]
        stats = calculate_dataset_statistics(list(a.out.glob("data/*/*.parquet")), floats)
        (a.out / "meta" / "stats_empirical.json").write_text(json.dumps(stats, indent=4) + "\n")
        act = stats["action"]
        n = len(act["min"])
        act.update({"min": [-1.0] * n, "q01": [-1.0] * n, "max": [1.0] * n, "q99": [1.0] * n})
        (a.out / "meta" / "stats.json").write_text(json.dumps(stats, indent=4) + "\n")
        provenance["action_bounds"] = {"kind": "fixed physical", "min": -1.0, "max": 1.0,
                                       "empirical": "meta/stats_empirical.json"}
    (a.out / "meta" / "kaivuri_source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
