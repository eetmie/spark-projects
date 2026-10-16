"""Compare a bundle's FP16 split references and OpenPI's mixed BF16/FP32 reference to an FP32 policy.

Runs in the pinned container with the export.sh environment:
    python check_fp32.py --checkpoint <converted> --bundle <bundle>
Same fixture inputs and noise; prints cosine / max |diff| of normalized and robot-unit chunks.
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch

from paths import train_config


def metric(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return {"cosine": round(float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)), 6),
            "max_abs": round(float(np.abs(a - b).max()), 5)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    a = ap.parse_args()
    from openpi.models_pytorch import pi0_pytorch
    from openpi.policies import policy as policy_module
    from openpi import transforms
    from openpi.shared import normalize

    torch.backends.cuda.matmul.allow_tf32 = False
    train = train_config()
    config = dataclasses.replace(train.model, pytorch_compile_mode=None)
    model = pi0_pytorch.PI0Pytorch(config)
    missing, unexpected = st.load_model(model, str(a.checkpoint / "model.safetensors"), strict=False)
    assert not unexpected and all("embed_tokens" in k for k in missing)
    model = model.float().eval()
    data = train.data.create(train.assets_dirs, config)
    norm = normalize.load(a.checkpoint / "assets" / data.asset_id)
    policy = policy_module.Policy(
        model, transforms=[transforms.InjectDefaultPrompt(None), *data.data_transforms.inputs,
                           transforms.Normalize(norm, use_quantiles=data.use_quantile_norm), *data.model_transforms.inputs],
        output_transforms=[*data.model_transforms.outputs, transforms.Unnormalize(norm, use_quantiles=data.use_quantile_norm),
                           *data.data_transforms.outputs],
        sample_kwargs={"num_steps": 10}, pytorch_device="cuda", is_pytorch=True)
    info = json.loads((a.bundle / "bundle.json").read_text())
    report = []
    for i in range(len(info["reference_comparisons"])):
        f = np.load(a.bundle / "fixtures" / f"fixture_{i}.npz")
        obs = {"observation/cam1": f["raw_cam1"], "observation/cam2": f["raw_cam2"],
               "observation/state": f["raw_state"], "prompt": json.loads((a.bundle / "bundle.json").read_text())
               ["reference_comparisons"] and None}
        obs["prompt"] = [t for t in info["kaivuri"]["tasks"] if info["tasks"][t]["head"] ==
                         f["tokens"].reshape(-1)[:len(info["tasks"][t]["head"])].tolist()][0]
        with torch.inference_mode():
            robot = policy.infer(obs, noise=f["noise"])["actions"]
        row = {"fixture": i, "fp16_split_vs_fp32_robot": metric(f["fp16_actions_robot"], robot),
               "mixed_bf16_vs_fp32_robot": metric(f["actions_robot"], robot)}
        report.append(row)
        print(json.dumps(row), flush=True)
    return report


if __name__ == "__main__":
    main()
