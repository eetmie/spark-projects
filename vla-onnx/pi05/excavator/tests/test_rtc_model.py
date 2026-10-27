"""CPU tests for training-time RTC on OpenPI pi0.5 (dummy-size Gemma, random weights).

    cd vla-onnx/pi05 && JAX_PLATFORMS=cpu third_party/openpi/.venv/bin/python -m pytest excavator/tests -q
"""
import pathlib
import sys

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import rtc_model  # noqa: E402

from openpi.models import pi0_config  # noqa: E402

H = 10


def small(cls=rtc_model.Pi0RTCConfig, **kw):
    base = dict(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=H,
                dtype="float32", max_token_len=16)
    if cls is rtc_model.Pi0RTCConfig:
        base["rtc_max_delay"] = kw.pop("rtc_max_delay", 4)
    return cls(**base, **kw)


@pytest.fixture(scope="module")
def model():
    """Random weights everywhere. At init OpenPI zeroes the AdaRMS modulation, so the gates
    are 0 and action tokens never mix; that would hide what these tests check."""
    graph, state = nnx.split(small().create(jax.random.key(0)))
    leaves, tree = jax.tree.flatten(state)
    keys = jax.random.split(jax.random.key(1), len(leaves))
    leaves = [x + 0.05 * jax.random.normal(k, x.shape, x.dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x
              for x, k in zip(leaves, keys)]
    return nnx.merge(graph, jax.tree.unflatten(tree, leaves))


def test_rmsnorm_2d_cond_is_stock_bit_exact():
    assert rtc_model.STOCK_RMSNORM is not None
    x = jax.random.normal(jax.random.key(1), (2, 5, 64))
    cond = jax.random.normal(jax.random.key(2), (2, 64))
    stock, row = rtc_model.STOCK_RMSNORM(), rtc_model.RowRMSNorm()
    params = stock.init(jax.random.key(3), x, cond)
    params = jax.tree.map(lambda p: p + 0.1 * jax.random.normal(jax.random.key(4), p.shape), params)
    a, ga = stock.apply(params, x, cond)
    b, gb = row.apply(params, x, cond)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ga, gb)
    # Same cond on every token through the per-token path.
    c, gc = row.apply(params, x, jnp.broadcast_to(cond[:, None], (2, 5, 64)))
    np.testing.assert_allclose(a, c, rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.broadcast_to(ga, gc.shape), gc, rtol=0, atol=1e-6)


def test_same_parameters_as_stock_pi05():
    """A plain pi0.5 checkpoint loads into the RTC model unchanged."""
    shapes = lambda cfg: jax.tree.map(  # noqa: E731
        lambda a: (a.shape, a.dtype), nnx.state(nnx.eval_shape(cfg.create, jax.random.key(0))).to_pure_dict())
    assert shapes(small()) == shapes(small(pi0_config.Pi0Config))


def test_zero_delay_sampler_matches_stock(model):
    obs = small().fake_obs(2)
    noise = jax.random.normal(jax.random.key(5), (2, H, 32))
    stock = model.sample_actions(None, obs, num_steps=5, noise=noise)
    row = model.sample_actions(None, obs, num_steps=5, noise=noise,
                               action_prefix=jnp.zeros_like(noise), delay=jnp.zeros(2, jnp.int32))
    np.testing.assert_allclose(stock, row, rtol=0, atol=2e-5)


def test_prefix_rows_are_returned_exactly(model):
    obs = small().fake_obs(2)
    noise = jax.random.normal(jax.random.key(6), (2, H, 32))
    prefix = jax.random.normal(jax.random.key(7), (2, H, 32))
    delay = jnp.array([3, 0], jnp.int32)
    out = np.asarray(model.sample_actions(None, obs, num_steps=5, noise=noise, action_prefix=prefix, delay=delay))
    np.testing.assert_array_equal(out[0, :3], np.asarray(prefix)[0, :3])
    assert not np.allclose(out[0, 3:], np.asarray(prefix)[0, 3:])
    free = np.asarray(model.sample_actions(None, obs, num_steps=5, noise=noise))
    np.testing.assert_allclose(out[1], free[1], rtol=0, atol=2e-5)


def test_prefix_changes_the_free_rows(model):
    """The prefix is visible to the free rows through attention (what training teaches)."""
    obs = small().fake_obs(1)
    noise = jax.random.normal(jax.random.key(8), (1, H, 32))
    a = model.sample_actions(None, obs, num_steps=5, noise=noise, action_prefix=jnp.zeros((1, H, 32)), delay=3)
    b = model.sample_actions(None, obs, num_steps=5, noise=noise, action_prefix=jnp.ones((1, H, 32)), delay=3)
    assert float(jnp.max(jnp.abs(a[:, 3:] - b[:, 3:]))) > 1e-6


def test_loss_covers_postfix_rows_only(model):
    cfg = small()
    obs, act = cfg.fake_obs(8), jax.random.normal(jax.random.key(9), (8, H, 32))
    rng = jax.random.key(10)
    loss = np.asarray(model.compute_loss(rng, obs, act, train=False))
    delay = np.asarray(rtc_model.sample_delay(jax.random.split(rng, 4)[3], 8, cfg.rtc_max_delay, "uniform"))
    for b in range(8):
        assert np.all(loss[b, :delay[b]] == 0)
        assert np.all(loss[b, delay[b]:] > 0)
    kept = (np.arange(H)[None] >= delay[:, None]).sum()
    # train.py's jnp.mean equals the mean over kept rows: un-scale and compare.
    raw = loss * kept / loss.size
    np.testing.assert_allclose(loss.mean(), raw.sum() / kept, rtol=1e-6)


def test_delay_distributions():
    rng = jax.random.key(11)
    u = np.asarray(rtc_model.sample_delay(rng, 20000, 4, "uniform"))
    e = np.asarray(rtc_model.sample_delay(rng, 20000, 4, "exp"))
    assert u.min() == 0 and u.max() == 4 and abs(np.mean(u == 4) - 0.2) < 0.02
    assert e.min() == 0 and e.max() <= 4 and np.mean(e == 0) > 0.6
    assert np.all(np.asarray(rtc_model.sample_delay(rng, 5, 0, "uniform")) == 0)


def test_config_guards():
    with pytest.raises(ValueError):
        small(rtc_max_delay=H)
    with pytest.raises(ValueError):
        rtc_model.Pi0RTCConfig(pi05=False)
