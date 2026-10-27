"""Training-time RTC for NVIDIA GR00T N1.7 (Isaac-GR00T n1.7-release 23ace64).

Black, Ren, Equi, Levine, arXiv 2512.05964, Algorithm 1: per sample a delay d is drawn;
action rows < d are the clean ground truth with flow time "clean", and only rows >= d
carry loss. At inference the committed rows are written into rows < d before every step.

GR00T's time runs noise 0 -> actions 1 (the paper's convention), so clean is t = 1, which
discretizes to bucket `num_timestep_buckets` (1000). Training draws t <= 0.999, so bucket
1000 only ever means "clean prefix row".

Per-row time touches three places that take one timestep per sample:
- MultiEmbodimentActionEncoder: sinusoidal time per action token; accepts [B, T] now.
- DiT / AlternateVLDiT: AdaLayerNorm in every block and the output modulation use one temb
  [B, D]; they accept a per-token timestep [B, S] -> temb [B, S, D]. The state token keeps
  the sample's flow time; action tokens get their row time.
- The action head's loss is masked to rows >= d (on top of GR00T's padding mask).
A [B] timestep takes the original code path everywhere. No new parameters.

The patched functions are copies of the pinned source with these edits. `apply()` checks
each original's source hash and refuses to patch a different GR00T revision.

NVIDIA's own experimental inference RTC (`action_input["action"]` + rtc_* options, velocity
strength ramp) is untouched and still reachable; training-time RTC sampling is selected by
`options["rtc_delay"]`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from typing import Literal

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.gr00t_n1d7 import gr00t_n1d7 as _head
from gr00t.model.modules import dit as _dit
from gr00t.model.modules import embodiment_conditioned_mlp as _emb

PINNED = "23ace64f17aa5015259b8609d371eb61a357c776"
SOURCE_SHA256 = {
    "MultiEmbodimentActionEncoder.forward": "1b95873f91a3e205448562c023f4885d919254b736b1bf1ade8168ad3de7b1cb",
    "TimestepEncoder.forward": "0e978031c181745912c152827a86a0965d4329dbb387cb28bc69714cee584cd0",
    "AdaLayerNorm.forward": "589c5c4ae5d625d81fbc8f33da62827282cbfc9fffa9944686fbab8e2e737c0f",
    "DiT.forward": "7b62d5f1ff1cc63f91dbdaf63bf7d13ea3ef106723d8e41ee328cd93aefcc0c9",
    "AlternateVLDiT.forward": "ae892ce058edead2d6d88d9619cb8bae117438fa8efa4208e332e93aca5eece0",
    "Gr00tN1d7ActionHead.forward": "688f3f3816fe94f4cc522af6241bd47dba4e83838bff940f4681140f68f27a0f",
    "Gr00tN1d7ActionHead.get_action_with_features": "e0c82af8a6daaaa234b6781f1e56c85ed0c34bcb5a52c3479fff46630c75bcfb",
}
TARGETS = {
    "MultiEmbodimentActionEncoder.forward": _emb.MultiEmbodimentActionEncoder,
    "TimestepEncoder.forward": _dit.TimestepEncoder,
    "AdaLayerNorm.forward": _dit.AdaLayerNorm,
    "DiT.forward": _dit.DiT,
    "AlternateVLDiT.forward": _dit.AlternateVLDiT,
    "Gr00tN1d7ActionHead.forward": _head.Gr00tN1d7ActionHead,
    "Gr00tN1d7ActionHead.get_action_with_features": _head.Gr00tN1d7ActionHead,
}
ORIGINAL = {name: cls.__dict__[name.split(".")[1]] for name, cls in TARGETS.items()}


@dataclasses.dataclass(frozen=True)
class RtcTraining:
    max_delay: int = 12  # rows; >= the runtime's largest committed prefix (latency x fps)
    dist: Literal["uniform", "exp"] = "uniform"

    def __post_init__(self):
        if self.max_delay < 0:
            raise ValueError("max_delay must be >= 0")
        if self.dist not in ("uniform", "exp"):
            raise ValueError(f"unknown delay distribution {self.dist!r}")

    def to_json(self) -> str:
        return json.dumps({"method": "training_time_rtc", "paper": "arXiv:2512.05964", "gr00t": PINNED,
                           "clean_time": 1.0, **dataclasses.asdict(self)}, indent=2)


def sample_delay(batch: int, rtc: RtcTraining | None, device) -> torch.Tensor:
    """[batch] long in {0..max_delay}."""
    if rtc is None or rtc.max_delay == 0:
        return torch.zeros(batch, dtype=torch.long, device=device)
    if rtc.dist == "uniform":
        return torch.randint(0, rtc.max_delay + 1, (batch,), device=device)
    weights = torch.exp(-torch.arange(rtc.max_delay + 1, dtype=torch.float32))
    return torch.multinomial(weights, batch, replacement=True).to(device)


def prefix_mask(delay: torch.Tensor, horizon: int) -> torch.Tensor:
    """[B, H] bool, True on rows < delay."""
    return torch.arange(horizon, device=delay.device)[None, :] < delay[:, None]


# ── per-token time in the encoder and DiT ─────────────────────────────────────

def action_encoder_forward(self, actions, timesteps, cat_ids):
    B, T, _ = actions.shape
    if timesteps.dim() == 1 and timesteps.shape[0] == B:
        timesteps = timesteps.unsqueeze(1).expand(-1, T)
    elif tuple(timesteps.shape) != (B, T):
        raise ValueError(f"timesteps must be [B] or [B, T]; got {tuple(timesteps.shape)} for actions {(B, T)}")
    a_emb = self.W1(actions, cat_ids)
    tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
    x = torch.cat([a_emb, tau_emb], dim=-1)
    x = _emb.swish(self.W2(x, cat_ids))
    x = self.W3(x, cat_ids)
    return x


def timestep_encoder_forward(self, timesteps):
    if timesteps.dim() == 1:
        return ORIGINAL["TimestepEncoder.forward"](self, timesteps)
    b, s = timesteps.shape
    return ORIGINAL["TimestepEncoder.forward"](self, timesteps.reshape(-1)).reshape(b, s, -1)


def ada_layer_norm_forward(self, x, temb=None):
    temb = self.linear(self.silu(temb))
    if temb.dim() == 2:
        scale, shift = temb.chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]
    scale, shift = temb.chunk(2, dim=-1)
    return self.norm(x) * (1 + scale) + shift


def _dit_output(self, hidden_states, temb, return_all_hidden_states, all_hidden_states):
    if temb.dim() == 2:
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
    else:
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale) + shift
    if return_all_hidden_states:
        return self.proj_out_2(hidden_states), all_hidden_states
    return self.proj_out_2(hidden_states)


def dit_forward(self, hidden_states, encoder_hidden_states, timestep=None, encoder_attention_mask=None,
                return_all_hidden_states=False):
    temb = self.timestep_encoder(timestep)
    hidden_states = hidden_states.contiguous()
    encoder_hidden_states = encoder_hidden_states.contiguous()
    all_hidden_states = [hidden_states]
    for idx, block in enumerate(self.transformer_blocks):
        if idx % 2 == 1 and self.config.interleave_self_attention:
            hidden_states = block(hidden_states, attention_mask=None, encoder_hidden_states=None,
                                  encoder_attention_mask=None, temb=temb)
        else:
            hidden_states = block(hidden_states, attention_mask=None, encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=None, temb=temb)
        all_hidden_states.append(hidden_states)
    return _dit_output(self, hidden_states, temb, return_all_hidden_states, all_hidden_states)


def alternate_dit_forward(self, hidden_states, encoder_hidden_states, timestep=None, encoder_attention_mask=None,
                          return_all_hidden_states=False, image_mask=None, backbone_attention_mask=None):
    assert image_mask is not None, "Image mask is required"
    temb = self.timestep_encoder(timestep)
    hidden_states = hidden_states.contiguous()
    encoder_hidden_states = encoder_hidden_states.contiguous()
    image_attention_mask = image_mask & backbone_attention_mask
    non_image_attention_mask = (~image_mask) & backbone_attention_mask
    all_hidden_states = [hidden_states]
    assert self.config.interleave_self_attention, "Interleave self attention must be enabled"
    for idx, block in enumerate(self.transformer_blocks):
        if idx % 2 == 1:
            hidden_states = block(hidden_states, attention_mask=None, encoder_hidden_states=None,
                                  encoder_attention_mask=None, temb=temb)
        else:
            if idx % (2 * self.attend_text_every_n_blocks) == 0:
                curr_encoder_attention_mask = non_image_attention_mask
            else:
                curr_encoder_attention_mask = image_attention_mask
            hidden_states = block(hidden_states, attention_mask=None, encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=curr_encoder_attention_mask, temb=temb)
        all_hidden_states.append(hidden_states)
    return _dit_output(self, hidden_states, temb, return_all_hidden_states, all_hidden_states)


# ── action head: training loss and sampler ────────────────────────────────────

def _run_dit(self, sa_embs, vl_embeds, timestep, backbone_output, return_all_hidden_states):
    if self.config.use_alternate_vl_dit:
        return self.model(hidden_states=sa_embs, encoder_hidden_states=vl_embeds,
                          encoder_attention_mask=backbone_output.backbone_attention_mask, timestep=timestep,
                          return_all_hidden_states=return_all_hidden_states, image_mask=backbone_output.image_mask,
                          backbone_attention_mask=backbone_output.backbone_attention_mask)
    return self.model(hidden_states=sa_embs, encoder_hidden_states=vl_embeds,
                      encoder_attention_mask=backbone_output.backbone_attention_mask, timestep=timestep,
                      return_all_hidden_states=return_all_hidden_states)


def _embed_actions(self, actions, row_buckets, embodiment_id, device):
    action_features = self.action_encoder(actions, row_buckets, embodiment_id)
    if self.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
        action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
    return action_features


def head_forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
    self.set_frozen_modules_to_eval_mode()
    backbone_output = self.process_backbone_output(backbone_output)
    vl_embeds = backbone_output.backbone_features
    device = vl_embeds.device
    embodiment_id = action_input.embodiment_id

    assert action_input.state.shape[1] == self.config.state_history_length
    action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)
    state_features = self.state_encoder(action_input.state, embodiment_id)
    if self.training and self.state_dropout_prob > 0:
        do_dropout = torch.rand(state_features.shape[0], device=state_features.device) < self.state_dropout_prob
        do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
        state_features = state_features * (1 - do_dropout)

    actions = action_input.action
    B, H = actions.shape[0], actions.shape[1]
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
    t = self.sample_time(B, device=actions.device, dtype=actions.dtype)  # [B]
    rtc = getattr(self, "rtc", None) if self.training else None
    prefix = prefix_mask(sample_delay(B, rtc, device), H)  # [B, H]
    row_t = torch.where(prefix, torch.ones_like(t)[:, None], t[:, None])  # clean = 1
    row_t3 = row_t[:, :, None]
    noisy_trajectory = (1 - row_t3) * noise + row_t3 * actions
    velocity = actions - noise

    t_buckets = (t * self.num_timestep_buckets).long()  # [B], as the stock head
    row_buckets = (row_t * self.num_timestep_buckets).long()  # [B, H]; prefix -> num_timestep_buckets
    action_features = _embed_actions(self, noisy_trajectory, row_buckets, embodiment_id, device)
    sa_embs = torch.cat((state_features, action_features), dim=1)
    token_time = torch.cat([t_buckets[:, None].expand(-1, state_features.shape[1]), row_buckets], dim=1)

    model_output, _ = _run_dit(self, sa_embs, vl_embeds, token_time, backbone_output, True)
    pred = self.action_decoder(model_output, embodiment_id)
    pred_actions = pred[:, -actions.shape[1]:]

    action_mask = action_input.action_mask * (~prefix)[:, :, None].to(action_input.action_mask.dtype)
    action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
    loss = action_loss.sum() / (action_mask.sum() + 1e-6)
    return {"loss": loss, "action_loss": action_loss, "action_mask": action_mask,
            "backbone_features": vl_embeds, "state_features": state_features}


@torch.no_grad()
def get_action_with_features(self, backbone_features, state_features, embodiment_id, backbone_output,
                             action_input, options=None):
    """`options["rtc_delay"]` (int or [B]) with `action_input["rtc_prefix"]` [B, H, D] (normalized,
    padded model space) selects training-time RTC sampling. Anything else: stock behavior."""
    if not options or "rtc_delay" not in options:
        return ORIGINAL["Gr00tN1d7ActionHead.get_action_with_features"](
            self, backbone_features, state_features, embodiment_id, backbone_output, action_input, options)
    vl_embeds = backbone_features
    B, device = vl_embeds.shape[0], vl_embeds.device
    actions = torch.randn(size=(B, self.config.action_horizon, self.action_dim), dtype=vl_embeds.dtype, device=device)
    prefix_actions = action_input["rtc_prefix"].to(device=device, dtype=actions.dtype)
    if prefix_actions.shape != actions.shape:
        raise ValueError(f"rtc_prefix is {tuple(prefix_actions.shape)}, expected {tuple(actions.shape)}")
    delay = torch.as_tensor(options["rtc_delay"], device=device, dtype=torch.long).expand(B)
    if int(delay.max()) >= self.config.action_horizon:
        raise ValueError("rtc_delay must be below the action horizon")
    prefix = prefix_mask(delay, self.config.action_horizon)
    hold = prefix[:, :, None]

    dt = 1.0 / self.num_inference_timesteps
    for step in range(self.num_inference_timesteps):
        actions = torch.where(hold, prefix_actions, actions)
        t_bucket = int(step / float(self.num_inference_timesteps) * self.num_timestep_buckets)
        row_buckets = torch.where(prefix, self.num_timestep_buckets, t_bucket)
        action_features = _embed_actions(self, actions, row_buckets, embodiment_id, device)
        sa_embs = torch.cat((state_features, action_features), dim=1)
        token_time = torch.cat([torch.full((B, state_features.shape[1]), t_bucket, device=device), row_buckets], 1)
        model_output = _run_dit(self, sa_embs, vl_embeds, token_time, backbone_output, False)
        pred = self.action_decoder(model_output, embodiment_id)
        actions = actions + dt * pred[:, -self.action_horizon:]
    actions = torch.where(hold, prefix_actions, actions)
    return BatchFeature(data={"action_pred": actions, "backbone_features": vl_embeds,
                              "state_features": state_features})


PATCHES = {
    "MultiEmbodimentActionEncoder.forward": action_encoder_forward,
    "TimestepEncoder.forward": timestep_encoder_forward,
    "AdaLayerNorm.forward": ada_layer_norm_forward,
    "DiT.forward": dit_forward,
    "AlternateVLDiT.forward": alternate_dit_forward,
    "Gr00tN1d7ActionHead.forward": head_forward,
    "Gr00tN1d7ActionHead.get_action_with_features": get_action_with_features,
}


def source_mismatches() -> list[str]:
    return [name for name, fn in ORIGINAL.items()
            if hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() != SOURCE_SHA256[name]]


def apply(rtc: RtcTraining | None) -> None:
    """Install the per-row-time code paths. `rtc` sets the training delay distribution for
    every action head (None: stock training loss, per-row sampling still available)."""
    bad = source_mismatches()
    if bad:
        raise RuntimeError(f"GR00T source differs from {PINNED} in {bad}; re-verify rtc_patch.py first")
    for name, fn in PATCHES.items():
        setattr(TARGETS[name], name.split(".")[1], fn)
    _head.Gr00tN1d7ActionHead.rtc = rtc


def remove() -> None:
    for name, fn in ORIGINAL.items():
        setattr(TARGETS[name], name.split(".")[1], fn)
    _head.Gr00tN1d7ActionHead.rtc = None
