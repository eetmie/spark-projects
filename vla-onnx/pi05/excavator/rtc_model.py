"""Training-time RTC for OpenPI pi0.5 (Mode C of kaivuriprokkis RTC_PLAN.md).

Black, Ren, Equi, Levine, "Training-Time Action Conditioning for Efficient Real-Time
Chunking", arXiv 2512.05964, Algorithm 1. Per sample a delay d is drawn; action rows < d
are the clean ground truth with their own flow time set to "clean", and the loss covers
only rows >= d. At inference the committed rows of the previous chunk are written into
rows < d before every step. No extra forward pass and no new parameters.

OpenPI's time runs noise 1 -> actions 0, so "clean" is t = 0 here (the paper's 1.0).

What changes in pi0.5: the action expert's AdaRMS takes one time vector per sample
([B, emb], `modulation[:, None, :]` in gemma.RMSNorm). Per-row time needs [B, H, emb].
`RowRMSNorm` replaces gemma.RMSNorm: identical math and parameter names, and a 2-D cond
takes the original path, so plain configs and checkpoints are unaffected. The exporter's
`--row-time` templates and `conditioning_clean_fp16.npy` are the inference side of this.

Delay d is drawn from {0..rtc_max_delay} (inclusive), uniform or with exponentially
decreasing weights (the paper's simulation setting).
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import einops
import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at

CLEAN_TIME = 0.0  # OpenPI convention: t = 1 noise, t = 0 actions


class RowRMSNorm(nn.Module):
    """gemma.RMSNorm that also accepts a per-token cond [B, T, D]. Same parameters."""

    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        if cond is None:
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (1 + scale)
            return normed_inputs.astype(dtype), None
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        if modulation.ndim == 2:
            modulation = modulation[:, None, :]
        elif modulation.shape[:2] != x.shape[:2]:
            raise ValueError(f"per-token cond {cond.shape} does not match tokens {x.shape}")
        scale, shift, gate = jnp.split(modulation, 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift
        return normed_inputs.astype(dtype), gate


STOCK_RMSNORM = _gemma.RMSNorm if _gemma.RMSNorm.__name__ == "RMSNorm" else None


def install_row_rmsnorm() -> None:
    """Swap gemma.RMSNorm for RowRMSNorm. Block/Module look it up by global name at call or
    setup time, so this must run before a model is created; Pi0RTCConfig.create does it."""
    _gemma.RMSNorm = RowRMSNorm


def sample_delay(rng, batch: int, max_delay: int, dist: str) -> jax.Array:
    """[batch] int32 in {0..max_delay}."""
    if max_delay == 0:
        return jnp.zeros((batch,), jnp.int32)
    if dist == "uniform":
        return jax.random.randint(rng, (batch,), 0, max_delay + 1)
    if dist == "exp":
        # weights e^-d, normalized: higher delays get less supervision (paper, simulation).
        logits = -jnp.arange(max_delay + 1, dtype=jnp.float32)
        return jax.random.categorical(rng, logits, shape=(batch,)).astype(jnp.int32)
    raise ValueError(f"unknown delay distribution {dist!r}")


def prefix_mask(delay: jax.Array, horizon: int) -> jax.Array:
    """[B, H] bool, True on rows < delay."""
    return jnp.arange(horizon)[None, :] < delay[:, None]


@dataclasses.dataclass(frozen=True)
class Pi0RTCConfig(pi0_config.Pi0Config):
    rtc_max_delay: int = 20  # rows; >= the runtime's largest committed prefix (latency x fps)
    rtc_delay_dist: Literal["uniform", "exp"] = "uniform"

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("training-time RTC is implemented for pi0.5 (AdaRMS time) only")
        if not 0 <= self.rtc_max_delay < self.action_horizon:
            raise ValueError("rtc_max_delay must be in [0, action_horizon)")

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0RTC":
        install_row_rmsnorm()
        return Pi0RTC(self, rngs=nnx.Rngs(rng))


class Pi0RTC(_pi0.Pi0):
    def __init__(self, config: Pi0RTCConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.rtc_max_delay = config.rtc_max_delay
        self.rtc_delay_dist = config.rtc_delay_dist

    def embed_suffix(self, obs, noisy_actions, timestep):
        """`timestep` [B] (stock) or [B, H] (per row). Per row returns adarms_cond [B, H, emb]."""
        if timestep.ndim == 1:
            return super().embed_suffix(obs, noisy_actions, timestep)
        b, h = timestep.shape
        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = _pi0.posemb_sincos(
            timestep.reshape(-1), self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        )
        time_emb = nnx.swish(self.time_mlp_in(time_emb))
        time_emb = nnx.swish(self.time_mlp_out(time_emb))
        adarms_cond = time_emb.reshape(b, h, -1)
        input_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
        ar_mask = jnp.array([True] + [False] * (self.action_horizon - 1))
        return action_tokens, input_mask, ar_mask, adarms_cond

    def _llm(self, *args, **kwargs):
        # The Module signature annotates adarms_cond as [b, d]; a per-row cond is [b, h, d].
        with at.disable_typechecking():
            return self.PaliGemma.llm(*args, **kwargs)

    @override
    def compute_loss(self, rng, observation, actions, *, train: bool = False):
        preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        b, h = actions.shape[0], actions.shape[1]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (b,)) * 0.999 + 0.001
        delay = sample_delay(delay_rng, b, self.rtc_max_delay, self.rtc_delay_dist)
        prefix = prefix_mask(delay, h)
        row_time = jnp.where(prefix, CLEAN_TIME, time[:, None])  # [B, H]
        t = row_time[..., None]
        x_t = t * noise + (1 - t) * actions  # prefix rows are exactly the clean actions
        u_t = noise - actions

        prefix_tokens, prefix_mask_, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, row_time)
        input_mask = jnp.concatenate([prefix_mask_, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self._llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # train.py takes jnp.mean of this [B, H] array. Scaling the masked rows by
        # count/kept turns that mean into the mean over kept (postfix) rows only.
        per_row = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        keep = jnp.logical_not(prefix).astype(per_row.dtype)
        return per_row * keep * (keep.size / jnp.maximum(jnp.sum(keep), 1.0))

    @override
    def sample_actions(
        self,
        rng,
        observation,
        *,
        num_steps: int = 10,
        noise=None,
        action_prefix=None,
        delay=None,
    ):
        """Stock sampler when `delay` is None. Otherwise rows < delay[b] are set to
        action_prefix [B, H, D] (normalized model space) before every step, fed clean time,
        and returned unchanged."""
        if delay is None:
            return super().sample_actions(rng, observation, num_steps=num_steps, noise=noise)
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        b = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (b, self.action_horizon, self.action_dim))
        delay = jnp.broadcast_to(jnp.asarray(delay, jnp.int32), (b,))
        prefix = prefix_mask(delay, self.action_horizon)
        hold = prefix[..., None]

        prefix_tokens, pmask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(pmask, prefix_ar_mask)
        positions = jnp.cumsum(pmask, axis=1) - 1
        _, kv_cache = self._llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            x_t = jnp.where(hold, action_prefix, x_t)
            row_time = jnp.where(prefix, CLEAN_TIME, time)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, row_time)
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(pmask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            pos = jnp.sum(pmask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self._llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=pos, kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            return carry[1] >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return jnp.where(hold, action_prefix, x_0)
