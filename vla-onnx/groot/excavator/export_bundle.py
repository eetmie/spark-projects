"""Export a GR00T N1.7 excavator fine-tune as a board bundle with runtime prompts.

Graphs come from the pinned jetson-orin-nano-vla exporter (`--exporter`, its K/V + AdaLN
layout) with `--full-kv`: no graph bakes the prompt, the text/image masks are runtime inputs.
The rotary table is baked, but it is identical for any text length (the images precede the
text and the pads continue its positions), so one bundle serves every prompt that fits.

Added to bundle.json:
  prompt   prefix ids (chat start + images), suffix ids, tokenizer.json, formalization,
           text capacity, the dataset's trained tasks, and per-prompt physical fixtures
           (FP32 Gr00tPolicy on real frames, plus its bfloat16 run as the error envelope)
  kaivuri  fps, state/action names and the normalization contract (physical_fixture.py)
  rtc      (checkpoints with rtc_training.json) the DiT graphs take `rtc_mask` + `prefix`
           (rtc_export.py); fixtures from rtc_patch's sampler with recorded-valve prefixes

    python export_bundle.py --checkpoint ../runs/NAME --dataset ../datasets/VIEW --out OUT

Runtime: kaivuriprokkis native_stack/tools/groot_live.py.
"""
import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
GROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(GROOT.parent / "common"))

DEFAULT_EXPORTER = GROOT / "third_party/jetson-orin-nano-vla-6e2d857/export"
HALF_NAMES = r"kv_[0-9]+|mod_[0-9]+|mod_out|state_features|ds_[0-9]+"   # as export.sh, N1.7


def formalize(text: str) -> str:
    """Gr00tN1d7Processor's formalize_language: lower case, drop punctuation."""
    return re.sub(r"[^\w\s]", "", text.lower())


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def task_samples(dataset: pathlib.Path, per_task: int) -> dict[str, list[str]]:
    """EPISODE:FRAME samples per trained task, spread over its episodes, a third into each take."""
    eps = [json.loads(l) for l in (dataset / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    out = {}
    for t in [json.loads(l)["task"] for l in (dataset / "meta/tasks.jsonl").read_text().splitlines() if l.strip()]:
        mine = [e for e in eps if t in e["tasks"]]
        pick = [mine[int(i)] for i in np.linspace(0, len(mine) - 1, per_task)]
        out[t] = [f"{e['episode_index']}:{e['length'] // 3}" for e in pick]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=pathlib.Path, required=True)
    p.add_argument("--dataset", type=pathlib.Path, required=True, help="the GR00T dataset view it trained on")
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--exporter", type=pathlib.Path, default=DEFAULT_EXPORTER)
    p.add_argument("--seq-len", type=int, default=256, help="prompt + images + pads; text capacity follows")
    p.add_argument("--samples-per-task", type=int, default=3)
    p.add_argument("--extra-prompt", action="append", default=None,
                   help="off-list prompts given a fixture too (default: 'dig a hole')")
    p.add_argument("--vlm-files", help="Cosmos-Reason2-2B config/tokenizer dir (default: cached snapshot)")
    a = p.parse_args()

    import finetune
    from physical_fixture import contract as physical_contract
    from vla_common.bundle.manifest import write_manifest

    py = sys.executable
    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty")
    vlm = a.vlm_files or finetune.local_snapshot(finetune.COSMOS)
    finetune.vlm_files_from(vlm)
    samples = task_samples(a.dataset, a.samples_per_task)
    tasks = list(samples)
    first = formalize(tasks[0])

    # 1. Stock FP32 reference (this repo's reference17.py reads processor/), then the graphs.
    ref = out.parent / (out.name + ".fixture.npz")
    run([py, GROOT / "reference17.py", "--checkpoint", a.checkpoint, "--vlm-files", vlm,
         "--embodiment", "new_embodiment", "--task", first, "--out", ref])
    env = {**os.environ, "PYTHONPATH": f"{a.exporter}:{a.exporter / 'groot'}"}
    rtc_cfg = a.checkpoint / "rtc_training.json"
    rtc = json.loads(rtc_cfg.read_text()) if rtc_cfg.exists() else None
    script = HERE / "rtc_export.py" if rtc else a.exporter / "groot/export_split_onnx17.py"
    run([py, script, "--ref", ref, "--out", out, "--full-kv", "--seq-len", a.seq_len], env=env)
    run([py, "-m", "vla_common.half_boundary", "--bundle", out, "--names", HALF_NAMES], env=env)
    ref.unlink()

    # 2. Prompt contract: prefix + tokenize(formalize(text)) + suffix == the processor's ids.
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(pathlib.Path(vlm) / "tokenizer.json"))
    b = json.loads((out / "bundle.json").read_text())
    ids = b["input_ids"][:b["prompt_tokens"]]
    last_img = max(i for i, t in enumerate(ids) if t == b["image_token"])
    prefix = ids[:last_img + 2]                                   # ..., image tokens, <|vision_end|>
    text_ids = tok.encode(first, add_special_tokens=False).ids
    suffix = ids[len(prefix) + len(text_ids):]
    if prefix + text_ids + suffix != ids:
        sys.exit("prefix + text + suffix does not reproduce the exported prompt")
    shutil.copy(pathlib.Path(vlm) / "tokenizer.json", out / "tokenizer.json")

    # 3. Physical fixtures per prompt: FP32 reference and the bfloat16 envelope.
    extra = a.extra_prompt if a.extra_prompt is not None else ["dig a hole"]
    jobs = [(t, samples[t]) for t in tasks] + [(x, samples[tasks[-1]]) for x in extra]
    (out / "prompt_fixtures").mkdir()
    fixtures, contract = [], None
    for k, (text, smp) in enumerate(jobs):
        entry = {"task": text, "trained": text in tasks, "samples": smp}
        for dtype in ("float32", "bfloat16"):
            f = out / "prompt_fixtures" / f"{k}_{dtype}.npz"
            run([py, HERE / "physical_fixture.py", "--checkpoint", a.checkpoint, "--dataset", a.dataset,
                 "--samples", *smp, "--task", text, "--dtype", dtype, "--out", f])
            f.with_suffix(".contract.json").unlink()
            entry[dtype] = str(f.relative_to(out))
        fixtures.append(entry)
    rtc_fixtures = []
    if rtc:
        d_max = int(rtc["max_delay"])
        delays = sorted({max(1, d_max // 3), max(1, 2 * d_max // 3), max(1, d_max - 2), d_max})
        smp = task_samples(a.dataset, len(delays))[tasks[0]]
        entry = {"task": tasks[0], "samples": smp, "delays": delays}
        for dtype in ("float32", "bfloat16"):
            f = out / "prompt_fixtures" / f"rtc_{dtype}.npz"
            run([py, HERE / "physical_fixture.py", "--checkpoint", a.checkpoint, "--dataset", a.dataset,
                 "--samples", *smp, "--task", tasks[0], "--dtype", dtype, "--rtc-delays", *delays, "--out", f])
            f.with_suffix(".contract.json").unlink()
            entry[dtype] = str(f.relative_to(out))
        rtc_fixtures.append(entry)

    from gr00t.data.embodiment_tags import EmbodimentTag
    import modality  # noqa: F401  (registers the excavator config)
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    proc = Gr00tN1d7Processor.from_pretrained(str(a.checkpoint / "processor"))
    mc = proc.modality_configs[EmbodimentTag.NEW_EMBODIMENT.value]
    contract = physical_contract(a.checkpoint, EmbodimentTag.NEW_EMBODIMENT.value, mc)
    info = json.loads((a.dataset / "meta/info.json").read_text())
    b["prompt"] = {
        "prefix_ids": prefix, "suffix_ids": suffix, "tokenizer": "tokenizer.json",
        "formalize": "lowercase, then drop every character matching [^\\w\\s]",
        "capacity_tokens": b["seq_len"] - len(prefix) - len(suffix),
        "tasks": tasks, "fixtures": fixtures,
    }
    b["kaivuri"] = {
        "fps": info["fps"],
        "state_names": info["features"]["observation.state"]["names"],
        "action_names": info["features"]["action"]["names"],
        "cameras": mc["video"].modality_keys, "contract": contract,
        "dataset": str(a.dataset.resolve()), "checkpoint": str(a.checkpoint.resolve()),
    }
    if rtc:
        b["rtc"] = {
            **rtc, "clean_bucket": 1000,
            "inputs": {"rtc_mask": "[1,H,1] float, 1 on committed rows (row < delay), every dit graph",
                       "prefix": "[1,H,D] committed actions, normalized model space clipped to [-1,1], "
                                 "zero-padded past the real width; first and last dit graph"},
            "delay": "rows from the observation time that execute before the new chunk can take over",
            "fixtures": rtc_fixtures,
        }
    (out / "bundle.json").write_text(json.dumps(b, indent=1) + "\n")
    n = write_manifest(out)
    print(f"bundle {out}: {len(tasks)} trained tasks, capacity {b['prompt']['capacity_tokens']} text tokens, "
          f"{len(fixtures)} prompt fixtures, {n} files in the manifest")


if __name__ == "__main__":
    main()
