"""kaivuriprokkis pi0.5 exporter: fine-tuned OpenPI checkpoint -> kaivuri-inference bundle.

Runs in the pinned OpenPI container (Dockerfile here); export.sh is the host entry point.

One pass over one loaded model, replacing vla-onnx/pi05/export's four prototype stages:

1. Stock references: the OpenPI policy in FP32 on CUDA (transforms, normalization,
   tokenizer, Euler loop) on real dataset frames with seeded noise. Same choice as
   jetson-orin-nano-vla's custom-checkpoint exporter: BF16 arithmetic itself drifts from
   FP32 more than the split export does.
2. The FP16 split model in the compact layout kaivuri-inference runs (BUNDLE.md in
   vla-onnx/pi05): two valid cameras, the prompt's real tokens, zero tail; FP32 language
   residual stream; K/V-only last language layer. Its outputs are the parity references.
3. Templates and per-layer weights, conditioning tables, RoPE, embedding, norm stats.
4. RTC (kaivuriprokkis native_stack/notes/RTC_PLAN.md, P5): action-chunk covariance of
   the training data, the t = 0 style table, and reference fixtures of the NumPy oracle
   (rtc.py) blended into the FP16 split denoise loop. `--row-time` also traces the action
   templates with per-row modulation [1, H, 3072] for a Mode C (clean-prefix) model.

The bundle is `pi05_openpi_state` / `pi05-split/1`, which the current runtime loads; the
RTC files are additions it ignores. With `--row-time` the action templates change shape
and need runtime support (P4) first.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), os.environ.get("PI05_EXPORT_DIR", "/pi05")]

import numpy as np  # noqa: E402

import rtc  # noqa: E402

IMAGE_KEYS = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
DATASET_CAMERAS = ("observation.images.cam1", "observation.images.cam2")
IMAGE_TOKENS = 256
HIDDEN = 2048
MAX_POSITIONS = 2048


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def fetch_tokenizer(lock: dict, target: Path) -> Path:
    """The generation-pinned PaliGemma tokenizer, checked against its lock."""
    (item,) = lock["objects"]
    path = target / item["file"]
    if not path.exists():
        target.mkdir(parents=True, exist_ok=True)
        url = (f"https://storage.googleapis.com/storage/v1/b/{lock['bucket']}/o/"
               f"{urllib.request.quote(lock['prefix'] + item['file'], safe='')}?alt=media&generation={item['generation']}")
        partial = path.with_suffix(".partial")
        with urllib.request.urlopen(url) as response, partial.open("wb") as out:
            out.write(response.read())
        partial.rename(path)
    if path.stat().st_size != item["size"] or sha(path) != item["sha256"]:
        raise SystemExit(f"tokenizer checksum mismatch: {path}")
    return path


def pin_downloads(tokenizer: Path):
    """OpenPI otherwise fetches a moving GCS tokenizer object; refuse any other download."""
    from openpi.shared import download
    original = download.maybe_download

    def pinned(url, *args, **kwargs):
        if str(url) == "gs://big_vision/paligemma_tokenizer.model":
            return tokenizer
        if str(url).startswith(("gs://", "http://", "https://")):
            raise ValueError(f"unpinned OpenPI download: {url}")
        return original(url, *args, **kwargs)

    download.maybe_download = pinned


def train_config(name: str):
    sys.path.insert(0, os.environ.get("PI05_CONFIG_DIR", "/cfg"))
    import config  # the excavator TrainConfigs (vla-onnx/pi05/excavator/config.py)
    config.register()
    from openpi.training import config as cfg
    return cfg.get_config(name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True, help="converted checkpoint (model.safetensors + assets/)")
    ap.add_argument("--frames", type=Path, required=True, help="excavator/fixture_frames.py output")
    ap.add_argument("--action-stats", type=Path, required=True, help="host/action_stats.py output")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True, help="scratch (tokenizer, temporary ONNX)")
    ap.add_argument("--config", default=os.environ.get("PI05_CONFIG", "pi05_excavator"))
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--row-time", action="store_true",
                    help="trace action templates with per-row modulation (Mode C); needs runtime P4")
    ap.add_argument("--rtc-rows", type=int, default=15, help="anchor rows d of the RTC reference fixtures")
    ap.add_argument("--rtc-elapsed", type=int, default=15,
                    help="old-chunk rows played between observations (2 Hz at 30 fps = 15)")
    ap.add_argument("--rtc-latency-s", type=float, default=0.434,
                    help="observation-to-plan latency; the runtime switches plans at ceil(latency x fps)")
    ap.add_argument("--rtc-time-threshold", type=float, default=0.3)
    ap.add_argument("--rtc-beta", type=float, default=0.5, help="correlation shrinkage of the reference")
    ap.add_argument("--rtc-stability-limit", type=float, default=1.0)
    ap.add_argument("--rtc-max-rows", type=int, default=25, help="largest anchor the bundle declares")
    a = ap.parse_args(argv)
    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        ap.error(f"{out} must be new or empty")

    import torch
    import torch.nn.functional as F
    import onnx
    import safetensors.torch as st
    from openpi import transforms
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, create_sinusoidal_pos_embedding
    from openpi.policies.policy import Policy
    from openpi.shared import normalize
    from layers import (MASK_NEG, ActionBlock, ActionInput, ActionOutput, GemmaBlock, KVBlock, VisionBlock,
                        VisionStem, VisionTail, accumulation, array, metric, pack)
    from prompt import StatePrompt, bins as state_bins
    from weight_digest import tensor_digest

    lock_path = Path(os.environ.get("PI05_EXPORT_DIR", "/pi05")) / "tokenizer-lock.json"
    tokenizer_lock = json.loads(lock_path.read_text())
    tokenizer = fetch_tokenizer(tokenizer_lock, a.work / "tokenizer")
    pin_downloads(tokenizer)
    train = train_config(a.config)
    cfg = dataclasses.replace(train.model, pytorch_compile_mode=None)
    if not getattr(cfg, "pi05", False) or not cfg.discrete_state_input:
        raise SystemExit("kaivuri-inference runs pi0.5 with discrete_state_input (joint angles in the prompt)")
    if cfg.paligemma_variant != "gemma_2b" or cfg.action_expert_variant != "gemma_300m":
        raise SystemExit("templates support gemma_2b with gemma_300m")
    horizon, width, steps = cfg.action_horizon, cfg.action_dim, a.num_steps
    data = train.data.create(train.assets_dirs, cfg)
    if not data.use_quantile_norm:
        raise SystemExit("the runtime implements OpenPI quantile normalization only")
    output_names = [type(t).__name__ for t in data.data_transforms.outputs]
    if output_names != ["ExcavatorOutputs"]:
        raise SystemExit(f"output transforms {output_names}: only ExcavatorOutputs (slice) is implemented")
    stats = normalize.load(a.checkpoint / "assets" / data.asset_id)
    frames = np.load(a.frames)
    tasks = [str(t) for t in frames["tasks"]]
    count = int(frames["count"])
    action_stats = np.load(a.action_stats)
    stats_meta = json.loads(str(action_stats["meta"]))
    if stats_meta["rows"] != horizon or stats_meta["fps"] != int(frames["fps"]):
        raise SystemExit("action stats were computed for another horizon or rate")
    if stats_meta["norm_stats_sha256"] != sha(a.checkpoint / "assets" / data.asset_id / "norm_stats.json"):
        raise SystemExit("action stats used other normalization statistics than this checkpoint")
    prompt = StatePrompt(tokenizer)
    verified = sum(prompt.verify(t, int(frames["frame_0/observation/state"].shape[-1])) for t in tasks)

    torch.manual_seed(3010)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    weights = a.checkpoint / "model.safetensors"
    model = PI0Pytorch(cfg)
    missing, unexpected = st.load_model(model, str(weights), strict=False)
    if unexpected or any("embed_tokens" not in k for k in missing):
        raise SystemExit(f"checkpoint does not match the model: missing={missing}, unexpected={unexpected}")
    model.float().eval().cuda()

    # ── 1. stock FP32 references on real frames ───────────────────────────────
    norm_in = transforms.Normalize(stats, use_quantiles=True)
    policy = Policy(model, transforms=[transforms.InjectDefaultPrompt(None), *data.data_transforms.inputs, norm_in,
                                       *data.model_transforms.inputs],
                    output_transforms=[*data.model_transforms.outputs, transforms.Unnormalize(stats, use_quantiles=True),
                                       *data.data_transforms.outputs],
                    sample_kwargs={"num_steps": steps}, pytorch_device="cuda", is_pytorch=True)
    captured = {}
    sample_actions = policy._sample_actions

    def sample(device, observation, **kwargs):
        result = sample_actions(device, observation, **kwargs)
        captured.update(actions=array(result), tokens=array(observation.tokenized_prompt),
                        token_mask=array(observation.tokenized_prompt_mask), state=array(observation.state))
        for key, image in observation.images.items():
            captured["image_" + key] = array(image)
            captured["image_mask_" + key] = array(observation.image_masks[key])
        return result

    policy._sample_actions = sample
    fixtures, state_dim, robot_dim = [], None, None
    for i in range(count):
        obs = {k[len(f"frame_{i}/"):]: frames[k] for k in frames.files if k.startswith(f"frame_{i}/observation/")}
        task = str(frames[f"frame_{i}/prompt"])
        obs["prompt"] = task
        noise = np.random.default_rng(1000 + i).standard_normal((horizon, width)).astype(np.float32)
        captured.clear()
        with torch.inference_mode():
            result = policy.infer(obs, noise=noise)
        fx = {k: v.copy() for k, v in captured.items()}
        with torch.inference_mode():
            again = policy.infer(obs, noise=noise)
        if not np.array_equal(result["actions"], again["actions"]):
            raise SystemExit("stock reference does not reproduce")
        keys = [k for k in IMAGE_KEYS if bool(fx["image_mask_" + k].item())]
        if keys != IMAGE_KEYS[:2]:
            raise SystemExit(f"valid cameras {keys}; the runtime feeds base_0_rgb and left_wrist_0_rgb")
        state_dim = int(obs["observation/state"].shape[-1])
        robot_dim = int(result["actions"].shape[-1])
        ids = fx["tokens"].reshape(-1)[fx["token_mask"].reshape(-1)].tolist()
        composed = prompt.compose(prompt.head(task), state_bins(fx["state"].reshape(-1)[:state_dim]))
        if composed != ids:
            raise SystemExit(f"fixture {i}: board-composed prompt differs from OpenPI's tokenization")
        fx.update(noise=noise, actions_robot=np.asarray(result["actions"]), dataset_index=frames[f"frame_{i}/index"],
                  raw_cam1=obs["observation/cam1"], raw_cam2=obs["observation/cam2"], raw_state=obs["observation/state"],
                  task=task)
        fixtures.append(fx)
        print(f"stock fixture {i}: {task!r}, {len(ids)} prompt tokens, state {obs['observation/state']}", flush=True)
    policy._sample_actions = sample_actions
    del policy, sample_actions, captured
    torch.cuda.empty_cache()

    prompt_capacity = prompt.capacity(tasks, state_dim)
    capacity = 2 * IMAGE_TOKENS + prompt_capacity
    if prompt_capacity > cfg.max_token_len or capacity + horizon >= MAX_POSITIONS:
        raise SystemExit(f"prompt capacity {prompt_capacity} / prefix {capacity} too large")

    # ── 2. FP16 split model ───────────────────────────────────────────────────
    core = model.paligemma_with_expert.paligemma.model
    language, vision = core.language_model, core.vision_tower.vision_model
    expert = model.paligemma_with_expert.gemma_expert.model
    rope_frequencies = [m.rotary_emb.inv_freq.clone().float() for m in (language, expert)]
    model.half()
    for module, frequencies in zip((language, expert), rope_frequencies):
        module.rotary_emb.inv_freq = frequencies
        module.rotary_emb.original_inv_freq = frequencies
        module.config._attn_implementation = "eager"
    vision.config._attn_implementation = "eager"
    stem = VisionStem(vision.embeddings)
    vision_blocks = [VisionBlock(layer) for layer in vision.encoder.layers]
    tail = VisionTail(vision.post_layernorm, core.multi_modal_projector, fp32_out=True)
    language_blocks = [GemmaBlock(layer, residual32=True) for layer in language.layers[:-1]]
    last_kv = KVBlock(language.layers[-1], residual32=True)
    last_full = GemmaBlock(language.layers[-1], residual32=True)
    action_input = ActionInput(model.action_in_proj)
    action_blocks = [ActionBlock(layer) for layer in expert.layers]
    action_output = ActionOutput(expert.norm, model.action_out_proj)
    n_lang, n_action = len(language.layers), len(action_blocks)
    expert_width = model.action_in_proj.out_features

    times = torch.from_numpy(rtc.openpi_times(steps)[:steps]).cuda()
    dt = torch.tensor(-1.0 / steps, dtype=torch.float32, device="cuda")
    sites = [(expert.layers[i], norm) for i in range(n_action) for norm in ("input_layernorm", "post_attention_layernorm")]

    def condition(t):
        emb = create_sinusoidal_pos_embedding(t, expert_width, 4e-3, 4.0, device=t.device).float().half()
        return F.silu(model.time_mlp_out(F.silu(model.time_mlp_in(emb))))

    def style_table(cond):
        table = torch.empty((len(sites) + 1, 3 * expert_width), dtype=torch.float16, device="cuda")
        for j, (layer, norm) in enumerate(sites):
            table[j] = getattr(layer, norm).dense(cond)[0]
        table[-1] = expert.norm.dense(cond)[0]
        return table

    with torch.inference_mode():
        conds = [condition(times[s:s + 1]) for s in range(steps)]  # one row each, as live inference
        styles = torch.stack([style_table(c) for c in conds])
        style_clean = style_table(condition(torch.zeros(1, dtype=torch.float32, device="cuda")))
        positions = torch.arange(-1, MAX_POSITIONS, device="cuda")[None]
        dummy = torch.empty((1, positions.shape[1], HIDDEN), dtype=torch.float16, device="cuda")
        rope_cos, rope_sin = language.rotary_emb(dummy, positions)
    del dummy
    capture = {}

    def prefix(fx, record=False):
        with torch.inference_mode():
            parts = []
            for v, key in enumerate(IMAGE_KEYS[:2]):
                image = torch.from_numpy(fx["image_" + key]).cuda().half()
                if image.shape[1] != 3:
                    image = image.permute(0, 3, 1, 2)
                if record and v == 0:
                    capture["stem"] = (image,)
                hidden = stem(image)
                for i, block in enumerate(vision_blocks):
                    if record and v == 0:
                        capture[f"vision_{i:02d}"] = (hidden.clone(),)
                    hidden = block(hidden)
                if record and v == 0:
                    capture["vision_tail"] = (hidden.clone(),)
                projected = tail(hidden)
                if record:
                    direct = core.get_image_features(image)
                    assert torch.equal(projected.to(direct.dtype), direct), "split vision differs from the FP16 tower"
                parts.append(projected)
            ids = torch.from_numpy(fx["tokens"].reshape(-1)[fx["token_mask"].reshape(-1)].astype(np.int64)).cuda()[None]
            # FP16 embedding x sqrt(2048) rounded to FP16, then widened for the FP32 stream.
            parts.append((language.embed_tokens(ids) * (HIDDEN ** 0.5)).float())
            n_valid = 2 * IMAGE_TOKENS + ids.shape[1]
            assert n_valid <= capacity, ("prefix capacity exceeded", n_valid, capacity)
            parts.append(torch.zeros((1, capacity - n_valid, HIDDEN), dtype=torch.float32, device="cuda"))
            hidden = torch.cat(parts, dim=1)
            pad = torch.arange(capacity, device="cuda")[None] < n_valid
            mask = torch.where(pad[:, None, :, None] & pad[:, None, None, :], 0.0, MASK_NEG)
            pos = torch.cumsum(pad, dim=-1) - 1
            cos, sin = rope_cos[:, pos[0] + 1], rope_sin[:, pos[0] + 1]
            kv = []
            for i, block in enumerate(language_blocks):
                if record:
                    capture[f"language_{i:02d}"] = (hidden.clone(), mask, cos, sin)
                hidden, k, v = block(hidden, mask, cos, sin)
                assert torch.isfinite(hidden).all(), ("nonfinite language block", i)
                kv.append((k, v))
            if record:
                capture[f"language_{n_lang - 1:02d}"] = (hidden.clone(), cos, sin)
                _, k_full, v_full = last_full(hidden, mask, cos, sin)
            k, v = last_kv(hidden, cos, sin)
            if record:
                assert torch.equal(k, k_full) and torch.equal(v, v_full), "K/V-only last layer differs"
            kv.append((k, v))
            return kv, pad

    def denoise(kv, pad, z, record=False, live=False, sched=None, x0=None, c=None, row_time=False):
        """FP16 split Euler loop; `sched` blends with the NumPy oracle after every step.
        `row_time` feeds per-row modulation: rows < d get the t = 0 style in Mode C."""
        rows = 0 if x0 is None else len(x0)
        clean_rows = rows if sched is not None and sched.mode == "clean" else 0
        log, velocities, trace = [], [], []
        with torch.inference_mode():
            spos = pad.sum(-1)[:, None] + torch.arange(horizon, device="cuda")[None]
            s_cos, s_sin = rope_cos[:, spos[0] + 1], rope_sin[:, spos[0] + 1]
            s_valid = torch.cat((pad[:, None, :].expand(1, horizon, -1),
                                 torch.ones((1, horizon, horizon), dtype=torch.bool, device="cuda")), dim=-1)
            s_mask = torch.where(s_valid[:, None], 0.0, MASK_NEG)
            start = rtc.initial(z, x0, sched) if sched is not None else z
            actions = torch.from_numpy(np.ascontiguousarray(start, np.float32)).cuda()[None]
            trace.append(array(actions))
            row = (torch.arange(horizon, device="cuda") < clean_rows)[None, :, None]

            def mod(s, j):
                m = styles[s, j][None, None]
                if row_time:
                    m = torch.where(row, style_clean[j][None, None], m.expand(1, horizon, -1))
                return m

            for s in range(steps):
                if record and s == 0:
                    capture["action_input"] = (actions.clone(),)
                hidden = action_input(actions)
                for i, block in enumerate(action_blocks):
                    mod_in, mod_post = mod(s, 2 * i), mod(s, 2 * i + 1)
                    if live:
                        mod_in = expert.layers[i].input_layernorm.dense(conds[s])[:, None]
                        mod_post = expert.layers[i].post_attention_layernorm.dense(conds[s])[:, None]
                    args = (hidden, s_mask, s_cos, s_sin, *kv[i], mod_in, mod_post)
                    if record and s == 0:
                        capture[f"action_{i:02d}"] = tuple(x.clone() for x in args)
                    hidden = block(*args)
                    assert torch.isfinite(hidden).all(), ("nonfinite action block", s, i)
                mod_final = mod(s, len(sites))
                if live:
                    mod_final = expert.norm.dense(conds[s])[:, None]
                if record and s == 0:
                    capture["action_output"] = (hidden.clone(), mod_final)
                velocity = action_output(hidden, mod_final)
                assert torch.isfinite(velocity).all(), ("nonfinite velocity", s)
                actions = actions + dt * velocity
                if sched is not None:
                    x, info = rtc.blend(array(actions)[0], z, x0, sched, s, c, a.rtc_stability_limit)
                    actions = torch.from_numpy(x).cuda()[None]
                    log.append(info)
                velocities.append(array(velocity))
                trace.append(array(actions))
        return array(actions)[0], {"log": log, "velocities": np.stack(velocities), "steps": np.stack(trace)}

    output_transform = transforms.compose([*data.model_transforms.outputs, transforms.Unnormalize(stats, use_quantiles=True),
                                           *data.data_transforms.outputs])
    (out / "fixtures").mkdir(parents=True, exist_ok=True)
    comparisons, prefixes = [], []
    for i, fx in enumerate(fixtures):
        kv, pad = prefix(fx, record=(i == 0))
        prefixes.append((kv, pad))
        fp16, extra = denoise(kv, pad, fx["noise"], record=(i == 0))
        live, _ = denoise(kv, pad, fx["noise"], live=True)
        assert np.array_equal(fp16, live), "cached and live AdaRMS modulation differ"
        if a.row_time:
            per_row, _ = denoise(kv, pad, fx["noise"], record=(i == 0), row_time=True)
            assert np.array_equal(fp16, per_row), "per-row modulation differs from the broadcast one"
        robot = output_transform({"actions": fp16, "state": fx["state"][0].copy()})["actions"]
        c = {"fixture": i, "task": fx["task"], "prompt_tokens": int(fx["token_mask"].sum()),
             "fp16_vs_stock_fp32": metric(fp16[None], fx["actions"]),
             "fp16_vs_stock_fp32_robot": metric(robot, fx["actions_robot"])}
        assert c["fp16_vs_stock_fp32"]["finite"] and c["fp16_vs_stock_fp32"]["cosine"] >= 0.999, c
        comparisons.append(c)
        fx.update(fp16_actions_normalized=fp16[None], fp16_actions_robot=robot,
                  fp16_velocities=extra["velocities"], fp16_action_steps=extra["steps"])
        np.savez_compressed(out / "fixtures" / f"fixture_{i}.npz",
                            **{k: (np.asarray(v) if k == "task" else v) for k, v in fx.items()})
        print("fixture", i, "FP16 split vs stock FP32", json.dumps(c["fp16_vs_stock_fp32"]), flush=True)

    # ── 3. templates, weights, tables ─────────────────────────────────────────
    action_inputs = ["hidden", "mask", "cos", "sin", "prefix_k", "prefix_v", "mod_in", "mod_post"]
    groups = [("stem", stem, "stem", ["image"], ["hidden"]),
              *[(f"vision_{i:02d}", b, "vision", ["hidden"], ["output"]) for i, b in enumerate(vision_blocks)],
              ("vision_tail", tail, "vision_tail", ["hidden"], ["output"]),
              *[(f"language_{i:02d}", b, "language", ["hidden", "mask", "cos", "sin"], ["output", "key", "value"])
                for i, b in enumerate(language_blocks)],
              (f"language_{n_lang - 1:02d}", last_kv, "language_kv", ["hidden", "cos", "sin"], ["key", "value"]),
              ("action_input", action_input, "action_input", ["actions"], ["hidden"]),
              *[(f"action_{i:02d}", b, "action", action_inputs, ["output"]) for i, b in enumerate(action_blocks)],
              ("action_output", action_output, "action_output", ["hidden", "mod_final"], ["velocity"])]
    components, seen = [], set()
    a.work.mkdir(parents=True, exist_ok=True)
    temporary = a.work / "temporary_export.onnx"
    for label, module, kind, names, outputs in groups:
        with torch.no_grad():
            torch.onnx.export(module, capture[label], str(temporary), dynamo=False, opset_version=17,
                              input_names=names, output_names=outputs, do_constant_folding=True)
        graph = onnx.load(str(temporary))
        changed = accumulation(graph)
        record = pack(out, graph, kind, label, template=kind not in seen)
        seen.add(kind)
        components.append({"label": label, "kind": kind, "inputs": names, "outputs": outputs,
                           "raw_weights_bytes": record["initializer_bytes"], "accumulation_ops": changed,
                           "structure_sha256": record["structure_sha256"]})
        print(f"exported {label} ({record['initializer_bytes'] / 2**20:.1f} MiB)", flush=True)
        del graph
    temporary.unlink()
    conditioning = f"conditioning_{steps}step_fp16.npy"
    np.save(out / conditioning, array(styles))
    np.save(out / "conditioning_clean_fp16.npy", array(style_clean))
    np.save(out / "embedding_fp16.npy", array(language.embed_tokens.weight))
    np.save(out / "rope_cos_fp16.npy", array(rope_cos[0]))
    np.save(out / "rope_sin_fp16.npy", array(rope_sin[0]))
    (out / "norm_stats.json").write_bytes((a.checkpoint / "assets" / data.asset_id / "norm_stats.json").read_bytes())
    (out / "conditioning_manifest.json").write_text(json.dumps({
        "num_steps": steps, "dtype": "float16", "shape": list(styles.shape),
        "timesteps_fp32": times.cpu().tolist(), "timesteps_fp32_bits": times.cpu().numpy().view(np.uint32).tolist(),
        "sites": [f"paligemma_with_expert.gemma_expert.model.layers.{i}.{n}.dense" for i in range(n_action)
                  for n in ("input_layernorm", "post_attention_layernorm")] + ["paligemma_with_expert.gemma_expert.model.norm.dense"],
        "clean": {"file": "conditioning_clean_fp16.npy", "time": 0.0, "shape": list(style_clean.shape)},
    }, indent=2) + "\n")

    # ── 4. RTC ────────────────────────────────────────────────────────────────
    fps = int(frames["fps"])
    dims = int(stats_meta["dims"])
    if dims != robot_dim:
        raise SystemExit(f"action stats have {dims} dims, the policy outputs {robot_dim}")
    cov, mean = action_stats["covariance"], action_stats["mean"]
    (out / "rtc").mkdir()
    np.save(out / "rtc" / "action_covariance_f64.npy", cov)
    np.save(out / "rtc" / "action_mean_f64.npy", mean)
    sigma = rtc.reference_sigma(cov, a.rtc_beta)
    if not 0 < a.rtc_rows <= a.rtc_max_rows < horizon or a.rtc_elapsed + a.rtc_rows > horizon:
        raise SystemExit("RTC anchor does not fit the horizon")
    corrections = rtc.Corrections(sigma, dims, horizon, a.rtc_max_rows)
    switch_row = math.ceil(a.rtc_latency_s * fps)
    sched_a = rtc.schedule("inpaint", steps, a.rtc_time_threshold)
    sched_c = rtc.schedule("clean", steps)
    rtc_refs = []
    for i, fx in enumerate(fixtures):
        kv, pad = prefixes[i]
        previous = fx["fp16_actions_normalized"][0]
        z = np.random.default_rng(2000 + i).standard_normal((horizon, width)).astype(np.float32)
        x0 = rtc.anchor(previous, a.rtc_elapsed, a.rtc_rows, dims)
        c_d = corrections(a.rtc_rows)
        off, _ = denoise(kv, pad, z)
        inpaint, info_a = denoise(kv, pad, z, sched=sched_a, x0=x0, c=c_d)
        arrays = {"source_fixture": np.asarray(i), "elapsed": np.asarray(a.rtc_elapsed), "rows": np.asarray(a.rtc_rows),
                  "dims": np.asarray(dims), "time_threshold": np.asarray(a.rtc_time_threshold, np.float32),
                  "stability_limit": np.asarray(a.rtc_stability_limit, np.float32),
                  "previous": previous, "noise": z, "anchor": x0, "correction": c_d,
                  "alpha": sched_a.alpha, "beta": sched_a.beta, "gate": sched_a.gate, "times": sched_a.times,
                  "off": off, "inpaint": inpaint, "inpaint_steps": info_a["steps"],
                  "inpaint_corrected": np.asarray([e["corrected"] for e in info_a["log"]]),
                  "inpaint_max_correction": np.asarray([e["max_correction"] for e in info_a["log"]], np.float32)}
        entry = {"fixture": i, "file": f"rtc/rtc_fixture_{i}.npz",
                 "off": rtc.join_metrics(off, previous, a.rtc_elapsed, 0, dims, switch_row),
                 "inpaint": rtc.join_metrics(inpaint, previous, a.rtc_elapsed, a.rtc_rows, dims, switch_row),
                 "inpaint_steps_corrected": int(arrays["inpaint_corrected"].sum())}
        if a.row_time:
            clean, info_c = denoise(kv, pad, z, sched=sched_c, x0=x0, row_time=True)
            assert np.array_equal(clean[:a.rtc_rows, :dims], x0), "Mode C prefix is not the anchor"
            arrays.update(clean=clean, clean_steps=info_c["steps"])
            entry["clean"] = rtc.join_metrics(clean, previous, a.rtc_elapsed, a.rtc_rows, dims, switch_row)
        np.savez_compressed(out / entry["file"], **arrays)
        rtc_refs.append(entry)
        print(f"rtc fixture {i}: switch jump off {entry['off']['switch_jump']:.4f}, "
              f"inpaint {entry['inpaint']['switch_jump']:.4f} (median step {entry['off']['median_step']:.4f})", flush=True)

    # ── bundle.json ───────────────────────────────────────────────────────────
    features = json.loads(str(frames["features"]))
    provenance = json.loads(str(frames["provenance"]))
    lock = {"openpi_commit": "c23745b5ad24e98f66967ea795a07b2588ed6c79", "policy_config": train.name,
            "converted_checkpoint_sha256": sha(weights),
            "converted_active_tensors_sha256": tensor_digest(weights, ["paligemma_with_expert.gemma_expert.lm_head.weight"]),
            "frames_sha256": sha(a.frames), "action_stats_sha256": sha(a.action_stats), "tokenizer": tokenizer_lock}
    if os.environ.get("PI05_SOURCE_STEP"):
        lock["source_step"] = os.environ["PI05_SOURCE_STEP"]
    bundle = {
        "schema": "pi05-split/1", "policy": "pi05_openpi_state", "policy_config": train.name, "complete": True,
        "exporter": {"name": "spark-projects/vla-onnx/kaivuriprokkis",
                     "revision": os.environ.get("KP_EXPORTER_REVISION", "unknown"), "stock_precision": "float32"},
        "checkpoint_sha256": lock["converted_checkpoint_sha256"],
        "image_keys": IMAGE_KEYS, "valid_cameras": 2, "image_slots": 3, "valid_camera_compute_only": True,
        "layout": "compact", "prefix_tokens": capacity, "prompt_capacity": prompt_capacity,
        "packing": f"Keep every valid prefix token in order; masked padding at the tail to static capacity {capacity}.",
        "action_horizon": horizon, "action_dim": width, "robot_action_dim": robot_dim, "state_dim": state_dim,
        "num_steps": steps, "conditioning": conditioning,
        "vision_layers": len(vision_blocks), "language_layers": n_lang, "action_layers": n_action,
        "dtype": "FP16 weights and hidden states, FP32 language residual stream, FP32 normalization/accumulation/Euler",
        "language_residual": "float32", "use_quantiles": True,
        "tokenizer": {"sep": prompt.sep, "neg": prompt.neg, "pieces": prompt.pieces, "tail": prompt.tail,
                      "verified_compositions": verified},
        "tasks": {t: {"head": prompt.head(t)} for t in tasks},
        "reference_comparisons": comparisons,
        "embedding_sha256": sha(out / "embedding_fp16.npy"),
        "components": components,
        "raw_compute_weights_bytes": sum(c["raw_weights_bytes"] for c in components),
        "kaivuri": {
            "fps": fps, "tasks": tasks,
            "state": {"names": features["observation.state"], "units": "deg", "discrete_state_input": True},
            "action": {"names": features["action"][:robot_dim], "range": [-1.0, 1.0]},
            "action_horizon": horizon, "num_steps": steps,
            "cameras": [{"slot": slot, "dataset_key": key, "camera": camera, "rotation_deg": rotation}
                        for slot, key, camera, rotation in zip(IMAGE_KEYS, DATASET_CAMERAS, provenance["camera_order"],
                                                               provenance["camera_rotations_deg"])],
            "profile_files": provenance["profile_files"], "recorder_revision": provenance["revision"],
            "preprocess": "OpenPI resize_with_pad to 224 (antialiased), gray repeated to 3 channels, x/255*2-1",
            "rtc": {
                "version": 1,
                "row_time": bool(a.row_time),
                "noise": "iid",
                "style_clean": "conditioning_clean_fp16.npy",
                "max_prefix": a.rtc_max_rows,
                "action_statistics": {"covariance": "rtc/action_covariance_f64.npy", "mean": "rtc/action_mean_f64.npy",
                                      **{k: stats_meta[k] for k in ("rows", "dims", "fps", "space", "chunks", "episodes",
                                                                     "dataset", "dataset_info_sha256", "norm_stats_sha256",
                                                                     "covariance_ddof", "action_names")}},
                "inpaint_defaults": {"sigma": "correlation of the covariance, shrinkage beta*Sigma+(1-beta)*I",
                                     "shrinkage_beta": a.rtc_beta, "jitter": "1e-6*max(mean(diag(Sigma_OO)),1)",
                                     "time_threshold": a.rtc_time_threshold, "stability_limit": a.rtc_stability_limit},
                "reference_fixtures": {"elapsed": a.rtc_elapsed, "rows": a.rtc_rows, "switch_row": switch_row,
                                       "latency_s": a.rtc_latency_s, "noise_seed": "2000 + fixture",
                                       "previous": "the fixture's own FP16 chunk (noise seed 1000 + fixture)",
                                       "oracle": "vla-onnx/kaivuriprokkis/rtc.py, blended on the host between FP16 split steps",
                                       "fixtures": rtc_refs},
            },
        },
        "source_lock": lock,
    }
    (out / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    (out / "MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files))
    print(f"complete: {out} ({bundle['raw_compute_weights_bytes'] / 2**30:.2f} GiB compute weights)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
