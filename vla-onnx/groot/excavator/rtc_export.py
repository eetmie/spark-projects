"""Training-time RTC in the exported GR00T N1.7 DiT graphs (rtc_patch.py's sampler, as graphs).

Every dit_k graph gains `rtc_mask` [1,H,1] (1 = committed row, 0 = free row); dit_0 and the
last chunk also take `prefix` [1,H,D] (normalized, padded model-space actions). As
rtc_patch.get_action_with_features: committed rows are written into the iterate before every
step and after the last, and carry the clean flow time (bucket 1000) in the action encoder and
in every AdaLN; the state token and the free rows keep the step's time. The clean-time
encodings are constants baked into the graphs, so the time/mod graphs and the runtime's
per-step constants are unchanged. An all-zero `rtc_mask` is the stock sampler.

    PYTHONPATH=EXPORTER:EXPORTER/groot python rtc_export.py --ref REF --out OUT [exporter args]

runs EXPORTER/groot/export_split_onnx17.py with these DiT chunks; export_bundle.py does this
for checkpoints that carry rtc_training.json.
"""
from __future__ import annotations

import pathlib
import runpy
import sys

import torch
import torch.nn.functional as F
from torch import nn

import groot_split
from groot_split import DitChunk, ModChunk, TimeEmb, _is_cross, attends_text

CLEAN_BUCKET = 1000.0          # rtc_patch: clean time 1.0 * num_timestep_buckets


class _TokenAdaLN(nn.Module):
    """A block's norm1 with per-token modulation [1,S,2D]; the stock AdaLayerNorm math
    (linear/silu already applied by the mod graphs) with the [:, None] broadcast dropped."""

    def __init__(self, ada):
        super().__init__()
        self.norm = ada.norm

    def forward(self, x, temb):
        scale, shift = temb.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class RtcEuler(nn.Module):
    """where(hold, prefix, x + dt * v), x itself held first. Named `euler` in the chunk, so
    the mixed-FP16 pass keeps it FP32 (export_split_onnx.FP32_NODE) like the stock update."""

    def __init__(self, dt: float):
        super().__init__()
        self.dt = dt

    def forward(self, x, v, prefix, rtc_mask):
        hold = rtc_mask > 0.5
        x = torch.where(hold, prefix, x)
        return torch.where(hold, prefix, x + self.dt * v)


class RtcDitChunk(DitChunk):
    def __init__(self, policy, a: int, b: int, c, rows=None):
        super().__init__(policy, a, b, c, rows)
        if c.buckets != CLEAN_BUCKET:
            raise ValueError(f"clean time is bucket {CLEAN_BUCKET:g}; the contract has {c.buckets}")
        for blk in self.blocks:
            blk.norm1 = _TokenAdaLN(blk.norm1)
        if self.last:
            self.euler = RtcEuler(self.dt)
        with torch.no_grad():
            t_proj, tau = TimeEmb(policy).eval()(torch.tensor([CLEAN_BUCKET]))
            mods = ModChunk(policy, a, b, self.last).eval()(t_proj)
        for name, m in zip([f"mod_{i}" for i in range(a, b)] + (["mod_out"] if self.last else []), mods):
            self.register_buffer(f"clean_{name}", m.detach().clone()[:, None], persistent=False)   # [1,1,M]
        if self.first:
            self.register_buffer("clean_tau", tau.detach().clone(), persistent=False)             # [1,1,hd]
        self.in_names = self.in_names + ["rtc_mask"] + (["prefix"] if self.first or self.last else [])

    def example(self, c, hd, mod_dim, kv_dim, S):
        base = [n for n in self.in_names if n not in ("rtc_mask", "prefix")]
        saved, self.in_names = self.in_names, base
        try:
            args = dict(zip(base, super().example(c, hd, mod_dim, kv_dim, S)))
        finally:
            self.in_names = saved
        args["rtc_mask"] = torch.zeros(1, c.action_horizon, 1)
        args["prefix"] = torch.zeros(1, c.action_horizon, c.action_dim)
        return tuple(args[n] for n in self.in_names)

    def forward(self, *args):
        x = dict(zip(self.in_names, args))
        m = x["rtc_mask"]                                         # [1,H,1]
        hold = m > 0.5
        token = torch.cat([torch.zeros_like(m[:, :1]), m], dim=1) > 0.5   # state token keeps step time
        if self.first:
            actions = torch.where(hold, x["prefix"], x["actions"])
            a_emb = self.w1(actions)
            tau = torch.where(hold, self.clean_tau.to(a_emb.dtype), x["tau"].to(a_emb.dtype))
            h = torch.cat([a_emb, tau], dim=-1)
            h = self.w2(h)
            h = self.w3(h * torch.sigmoid(h)) + self.pos_emb
            h = torch.cat([x["state_features"], h], dim=1)
        else:
            h = x["h"]

        def per_token(name):
            mod = x[name][:, None]
            return torch.where(token, getattr(self, f"clean_{name}").to(mod.dtype), mod)

        for idx, block in zip(range(self.a, self.b), self.blocks):
            temb = per_token(f"mod_{idx}")
            if not _is_cross(idx):
                h = block(h, temb=temb)
            else:
                bias = None if self.rows is not None else \
                    x["text_bias" if attends_text(idx, self.every) else "image_bias"]
                h = block(h, encoder_hidden_states=x[f"kv_{idx}"], encoder_attention_mask=bias, temb=temb)
        if not self.last:
            return h
        shift, scale = per_token("mod_out").chunk(2, dim=-1)
        h = self.norm_out(h) * (1 + scale) + shift
        h = self.proj_out_2(h)
        v = self.d2(F.relu(self.d1(h)))[:, -self.horizon:]
        return self.euler(x["actions"], v, x["prefix"], m)


def install() -> None:
    """add_action_graphs (and run_split) build DiT chunks through groot_split.DitChunk."""
    groot_split.DitChunk = RtcDitChunk


def main():
    exporter = pathlib.Path(groot_split.__file__).resolve().parent / "export_split_onnx17.py"
    install()
    sys.argv = [str(exporter), *sys.argv[1:]]
    runpy.run_path(str(exporter), run_name="__main__")


if __name__ == "__main__":
    main()
