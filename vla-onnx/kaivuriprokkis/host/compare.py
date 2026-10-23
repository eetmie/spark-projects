"""Compare two pi05 split bundles of the same checkpoint (e.g. this exporter vs vla-onnx/pi05).

    python host/compare.py NEW REFERENCE

Reports, per artifact class, whether the files are byte-identical: refit weights, templates
(structure hash and ONNX bytes), conditioning/RoPE/embedding tables, norm stats, and the
fixtures' inputs and FP16 split references. Exit 1 if weights, tables or FP16 references differ.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(new, ref):
    new, ref = Path(new), Path(ref)
    bad = 0
    a, b = json.loads((new / "bundle.json").read_text()), json.loads((ref / "bundle.json").read_text())
    for key in ("policy", "prefix_tokens", "prompt_capacity", "action_horizon", "action_dim", "robot_action_dim",
                "state_dim", "num_steps", "tokenizer", "tasks", "checkpoint_sha256"):
        x, y = a.get(key), b.get(key)
        if key == "tokenizer":
            x, y = ({k: v for k, v in d.items() if k != "verified_compositions"} for d in (x, y))
        same = x == y
        bad += not same
        print(f"{'same' if same else 'DIFF'}  bundle.{key}")
    kinds = {c["label"]: c for c in b["components"]}
    weights_diff, structure_diff = [], []
    for c in a["components"]:
        r = kinds.get(c["label"])
        if r is None or r["kind"] != c["kind"]:
            weights_diff.append(c["label"])
            continue
        if c["structure_sha256"] != r["structure_sha256"]:
            structure_diff.append(c["label"])
        x = json.loads((new / "weights" / c["label"] / "index.json").read_text())["weights"]
        y = json.loads((ref / "weights" / c["label"] / "index.json").read_text())["weights"]
        if [(w["name"], w["sha256"]) for w in x] != [(w["name"], w["sha256"]) for w in y]:
            weights_diff.append(c["label"])
    bad += bool(weights_diff) or len(a["components"]) != len(b["components"])
    print(f"{'same' if not weights_diff else 'DIFF'}  weights of {len(a['components'])} components {weights_diff or ''}")
    print(f"{'same' if not structure_diff else 'note'}  component structure hashes {structure_diff or ''}")
    for kind in sorted({c["kind"] for c in a["components"]}):
        x, y = new / "templates" / f"{kind}.onnx", ref / "templates" / f"{kind}.onnx"
        print(f"{'same' if y.exists() and sha(x) == sha(y) else 'note'}  templates/{kind}.onnx bytes")
    for name in (a.get("conditioning", "conditioning_10step_fp16.npy"), "rope_cos_fp16.npy", "rope_sin_fp16.npy",
                 "embedding_fp16.npy", "norm_stats.json"):
        same = sha(new / name) == sha(ref / name)
        bad += not same
        print(f"{'same' if same else 'DIFF'}  {name}")
    for path in sorted((new / "fixtures").glob("fixture_*.npz")):
        other = ref / "fixtures" / path.name
        if not other.exists():
            print(f"note  {path.name}: no reference fixture")
            continue
        with np.load(path) as x, np.load(other) as y:
            inputs = [k for k in ("raw_cam1", "raw_cam2", "raw_state", "noise", "tokens", "token_mask", "state",
                                  "image_base_0_rgb", "image_left_wrist_0_rgb") if not np.array_equal(x[k], y[k])]
            fp16 = [k for k in ("fp16_actions_normalized", "fp16_actions_robot") if not np.array_equal(x[k], y[k])]
            d = np.abs(x["actions"].astype(np.float64) - y["actions"]).max()
            bad += bool(inputs or fp16)
            print(f"{'same' if not (inputs or fp16) else 'DIFF'}  {path.name}: inputs {inputs or 'equal'}, "
                  f"FP16 references {fp16 or 'equal'}; stock vs reference stock max |diff| {d:.4g}")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
