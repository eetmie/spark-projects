"""Parity: split-graph ONNX pipeline vs the monolithic FP32 gold (the ship gate).

Runs the SmolVLA flow-matching loop over the 9 split graphs (vision/text/state +
expert prefill[KV]/decode + projectors) entirely in numpy+ORT, exactly as the Orin
runtime will, then compares the action chunk against the monolithic
`sample_actions` ONNX on identical seeded inputs. If cosine ~1.0 / max_abs tiny,
the split decomposition + Python loop are correct and the bundle is Orin-ready.

Numpy loop ported from github.com/aifoundry-org/ETARS modeling_smolvla_ort.py,
adapted to our exporter's graph names + lerobot 0.5.1 config.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * _sigmoid(x)


def make_att_2d_masks(pad_masks, att_masks):
    cumsum = np.cumsum(att_masks, axis=1)
    att = cumsum[:, None, :] <= cumsum[:, :, None]
    pad = pad_masks[:, None, :] & pad_masks[:, :, None]
    return att & pad


def sinusoidal_time_emb(time, dim, min_period, max_period):
    frac = np.linspace(0.0, 1.0, dim // 2, dtype=np.float64)
    period = min_period * (max_period / min_period) ** frac
    scale = 1.0 / period * 2 * math.pi
    x = scale[None, :] * time[:, None]
    return np.concatenate([np.sin(x), np.cos(x)], axis=1)


class Sess:
    def __init__(self, path):
        self.s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inames = [i.name for i in self.s.get_inputs()]

    def __call__(self, *args):
        return self.s.run(None, {n: a for n, a in zip(self.inames, args)})


class SplitPipeline:
    def __init__(self, d: Path, cfg):
        self.cfg = cfg
        self.vision = Sess(d / "smolvlm_vision.onnx")
        self.text = Sess(d / "smolvlm_text.onnx")
        self.state = Sess(d / "state_projector.onnx")
        self.action_in = Sess(d / "action_in_projector.onnx")
        self.action_out = Sess(d / "action_out_projector.onnx")
        self.time_in = Sess(d / "time_in_projector.onnx")
        self.time_out = Sess(d / "time_out_projector.onnx")
        self.prefill = Sess(d / "smolvlm_expert_prefill.onnx")
        self.decode = Sess(d / "smolvlm_expert_decode.onnx")
        self.exp_dim = int(960 * cfg.expert_width_multiplier)

    def embed_prefix(self, image, lang_tokens, lang_masks, state):
        embs, pad_masks, att = [], [], []
        img_emb = self.vision(image)[0]
        img_emb = img_emb * img_emb.shape[-1] ** 0.5
        n_img = img_emb.shape[1]
        embs.append(img_emb)
        pad_masks.append(np.ones((1, n_img), dtype=bool))
        att += [0] * n_img

        lang_emb = self.text(lang_tokens)[0]
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att += [0] * lang_emb.shape[1]

        state_emb = self.state(state)[0]
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        pad_masks.append(np.ones((1, state_emb.shape[1]), dtype=bool))
        att += [1] * state_emb.shape[1]

        embs = np.concatenate(embs, axis=1).astype(np.float32)
        pad_masks = np.concatenate(pad_masks, axis=1)
        att = np.array(att, dtype=bool)[None, :]
        return embs, pad_masks, att

    def embed_suffix(self, x_t, time):
        action_emb = self.action_in(x_t.astype(np.float32))[0]
        time_emb = sinusoidal_time_emb(time, self.exp_dim, self.cfg.min_period, self.cfg.max_period)
        time_emb = np.broadcast_to(time_emb[:, None, :], action_emb.shape).copy()
        at = np.concatenate([action_emb, time_emb], axis=2).astype(np.float32)
        at = self.time_in(at)[0]
        at = silu(at)
        at = self.time_out(at.astype(np.float32))[0]
        pad = np.ones((1, at.shape[1]), dtype=bool)
        att = np.ones(self.cfg.chunk_size, dtype=bool)[None, :]
        return at.astype(np.float32), pad, att

    def sample_actions(self, image, lang_tokens, lang_masks, state, noise):
        pe, pp, pa = self.embed_prefix(image, lang_tokens, lang_masks, state)
        pmask2d = make_att_2d_masks(pp, pa)
        ppos = (np.cumsum(pp, axis=1) - 1).astype(np.int64)
        kv = self.prefill(pmask2d, ppos, pe)  # 32 KV tensors

        dt = np.array(-1.0 / self.cfg.num_steps, dtype=np.float32)
        x_t = noise.astype(np.float32).copy()
        t = np.array(1.0, dtype=np.float32)
        chunk = self.cfg.chunk_size
        while t >= -dt / 2:
            se, sp, sa = self.embed_suffix(x_t, np.broadcast_to(t, 1))
            slen, plen = sp.shape[1], pp.shape[1]
            pref2d = np.broadcast_to(pp[:, None, :], (1, slen, plen)).copy()
            suf2d = make_att_2d_masks(sp, sa)
            full = np.concatenate([pref2d, suf2d], axis=2)
            pos = (np.sum(pp, axis=-1)[:, None] + np.cumsum(sp, axis=1) - 1).astype(np.int64)
            out = self.decode(full, pos, se, *kv)[0]
            out = out[:, -chunk:].astype(np.float32)
            v_t = self.action_out(out)[0]
            x_t = x_t + dt * v_t
            t = t + dt
        return x_t


def cosine(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-dir", default="exports-split")
    ap.add_argument("--monolith", default="exports/smolvla_base_fp32_static.onnx")
    ap.add_argument("--model-id", default="lerobot/smolvla_base")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    import lerobot.policies.smolvla.modeling_smolvla  # noqa: F401 (registers the 'smolvla' choice)
    from lerobot.configs.policies import PreTrainedConfig
    cfg = PreTrainedConfig.from_pretrained(args.model_id)

    rng = np.random.default_rng(args.seed)
    image = rng.standard_normal((1, 3, *cfg.resize_imgs_with_padding)).astype(np.float32)
    lang_tokens = np.ones((1, cfg.tokenizer_max_length), dtype=np.int64)
    lang_masks = np.ones((1, cfg.tokenizer_max_length), dtype=bool)
    state = rng.standard_normal((1, cfg.max_state_dim)).astype(np.float32)
    noise = rng.standard_normal((1, cfg.chunk_size, cfg.max_action_dim)).astype(np.float32)

    print("Running SPLIT pipeline (9 ORT graphs, Python denoise loop) ...")
    split_out = SplitPipeline(Path(args.split_dir), cfg).sample_actions(
        image, lang_tokens, lang_masks, state, noise)

    print(f"Running MONOLITH gold ({args.monolith}) ...")
    mono = ort.InferenceSession(args.monolith, providers=["CPUExecutionProvider"])
    feed = {"image0": image, "img_mask0": np.ones((1,), dtype=bool),
            "lang_tokens": lang_tokens, "lang_masks": lang_masks,
            "state": state, "noise": noise}
    mono_out = mono.run(None, feed)[0]

    cos = cosine(mono_out, split_out)
    absd = np.abs(mono_out.astype(np.float64) - split_out.astype(np.float64))
    print(f"\nsplit shape {split_out.shape}  mono shape {mono_out.shape}")
    print(f"cosine          = {cos:.7f}")
    print(f"max_abs_diff    = {absd.max():.6e}")
    print(f"mean_abs_diff   = {absd.mean():.6e}")
    passed = cos >= 0.999 and absd.max() <= 1e-2
    print(f"RESULT: {'PASS ✅ — split matches the monolith, Orin-ready' if passed else 'FAIL ❌ — investigate'}")


if __name__ == "__main__":
    main()
