"""GR00T action-head pieces shared by the N1.7 split (groot17_split.py, export_split_onnx17.py).

The N1.6 split that also lived here was removed 2026-10-28 (N1.7 replaced it); the
published N1.6 exporter is in jetson-orin-nano-vla export/groot.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MASK_NEG = -1.0e4


# --------------------------------------------------------------------------------------
# loading


# --------------------------------------------------------------------------------------
# contract


# --------------------------------------------------------------------------------------
# vision


# --------------------------------------------------------------------------------------
# language model


# --------------------------------------------------------------------------------------
# action head


def _slice_cat_linear(lin, eid: int) -> nn.Linear:
    w = lin.W[eid].detach()               # [in, out]
    out = nn.Linear(w.shape[0], w.shape[1])
    out.weight.data = w.t().contiguous().clone()
    out.bias.data = lin.b[eid].detach().clone()
    return out


class TimeEmb(nn.Module):
    """The two sinusoidal encodings of the denoising time t, kept as their own FP32 graph.

    sin(t * freq) with t up to 750 loses ~0.4 rad of argument when the product is FP16,
    and the mixed-FP16 pass would put it there; stock PyTorch computes both in FP32.
    Measured: inline FP16 sinusoids cost cos 0.992 / max 11.7 % of range on the chunk.
    """

    def __init__(self, policy):
        super().__init__()
        self.time_proj = policy.action_head.model.timestep_encoder.time_proj
        self.hidden = policy.action_head.action_encoder.hidden_size

    def forward(self, t):
        half = self.hidden // 2
        exponent = -torch.arange(half, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half)
        freqs = t.float().reshape(1, 1, 1) * exponent.exp()
        tau = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)    # [1,1,hidden]
        return self.time_proj(t), tau


class DitChunk(nn.Module):
    """DiT blocks [a, b) of one denoising step.

    First chunk: action encoder (sliced to the embodiment), position embedding, concat
    with the state token, timestep embedding. Last chunk: output AdaLN, proj_out,
    action decoder and the Euler update, so it returns the next action iterate.
    `temb` crosses every chunk boundary because every block's AdaLN needs it.
    """

    def __init__(self, policy, a: int, b: int, c):  # c: groot17_split.Contract
        super().__init__()
        head = policy.action_head
        dit = head.model
        self.a, self.b = a, b
        self.first = a == 0
        self.last = b == len(dit.transformer_blocks)
        self.blocks = nn.ModuleList(dit.transformer_blocks[a:b])
        self.every = dit.attend_text_every_n_blocks
        self.dt = 1.0 / c.steps
        self.horizon = c.action_horizon
        eid = c.embodiment_id
        if self.first:
            enc = head.action_encoder
            self.w1 = _slice_cat_linear(enc.W1, eid)
            self.w2 = _slice_cat_linear(enc.W2, eid)
            self.w3 = _slice_cat_linear(enc.W3, eid)
            self.timestep_embedder = dit.timestep_encoder.timestep_embedder
            self.register_buffer(
                "pos_emb", head.position_embedding.weight[:c.action_horizon][None].detach().clone(),
                persistent=False)
        if self.last:
            self.norm_out = dit.norm_out
            self.proj_out_1 = dit.proj_out_1
            self.proj_out_2 = dit.proj_out_2
            dec = head.action_decoder
            self.d1 = _slice_cat_linear(dec.layer1, eid)
            self.d2 = _slice_cat_linear(dec.layer2, eid)

    def forward(self, *args):
        if self.first:
            actions, t_proj, tau, state_features, vl, text_bias, image_bias = args
            a_emb = self.w1(actions)
            x = torch.cat([a_emb, tau.to(a_emb.dtype).repeat(1, actions.shape[1], 1)], dim=-1)
            x = self.w2(x)
            x = self.w3(x * torch.sigmoid(x)) + self.pos_emb
            h = torch.cat([state_features, x], dim=1)
            temb = self.timestep_embedder(t_proj.to(a_emb.dtype))
        elif self.last:
            h, temb, vl, text_bias, image_bias, actions = args
        else:
            h, temb, vl, text_bias, image_bias = args
        for i, block in enumerate(self.blocks):
            idx = self.a + i
            if idx % 2 == 1:
                h = block(h, temb=temb)
            else:
                bias = text_bias if idx % (2 * self.every) == 0 else image_bias
                h = block(h, encoder_hidden_states=vl, encoder_attention_mask=bias, temb=temb)
        if not self.last:
            return (h, temb) if self.first else h
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        h = self.norm_out(h) * (1 + scale[:, None]) + shift[:, None]
        h = self.proj_out_2(h)
        v = self.d2(F.relu(self.d1(h)))[:, -self.horizon:]
        return actions + self.dt * v


# --------------------------------------------------------------------------------------
# planning


def plan(units: list[int], budget: int) -> list[tuple[int, int]]:
    """Greedy contiguous packing of per-layer param counts under `budget`."""
    out, start, acc = [], 0, 0
    for i, n in enumerate(units):
        if acc and acc + n > budget:
            out.append((start, i))
            start, acc = i, 0
        acc += n
    out.append((start, len(units)))
    return out


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------------------
# host-side glue (numpy-free here; the runtime mirrors it)


