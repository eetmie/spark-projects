"""CPU tests for the RTC DiT graphs (rtc_export.py) against rtc_patch.py's sampler, tiny random head.

    vla-onnx/groot/.venv-groot17/bin/python -m pytest vla-onnx/groot/excavator/tests -q
"""
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

HERE = pathlib.Path(__file__).resolve().parents[1]
EXPORTER = HERE.parent / "third_party/jetson-orin-nano-vla-6e2d857/export/groot"
sys.path[:0] = [str(HERE), str(EXPORTER)]
import rtc_patch  # noqa: E402
import rtc_export  # noqa: E402
from groot_split import DitChunk, KvChunk, ModChunk, TimeEmb, _is_cross  # noqa: E402
from groot17_split import Contract, MASK_NEG  # noqa: E402

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config  # noqa: E402
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead  # noqa: E402

H, A, S, V, EID = 10, 8, 6, 12, 1


@pytest.fixture
def head():
    rtc_patch.remove()
    torch.manual_seed(0)
    cfg = Gr00tN1d7Config(
        backbone_embedding_dim=32, hidden_size=32, input_embedding_dim=48, max_action_dim=A, max_state_dim=S,
        action_horizon=H, max_num_embodiments=4, state_dropout_prob=0.0, use_alternate_vl_dit=True,
        diffusion_model_cfg=dict(positional_embeddings=None, num_layers=4, num_attention_heads=2,
                                 attention_head_dim=24, norm_type="ada_norm", dropout=0.0, final_dropout=False,
                                 output_dim=32, interleave_self_attention=True))
    h = Gr00tN1d7ActionHead(cfg).float().eval()
    yield h
    rtc_patch.remove()


def observation(seed=0):
    g = torch.Generator().manual_seed(seed)
    image = torch.zeros(1, V, dtype=torch.bool)
    image[:, :6] = True
    valid = torch.ones(1, V, dtype=torch.bool)
    valid[:, -2:] = False                                      # two pads, as the right-padded sequence
    vl = torch.randn(1, V, 32, generator=g)
    state = torch.randn(1, 1, S, generator=g)
    noise = torch.randn(1, H, A, generator=g)
    prefix = torch.randn(1, H, A, generator=g)
    return vl, image, valid, state, noise, prefix


def reference(h, delay, seed=0):
    """rtc_patch's sampler (delay None: the stock sampler) on processed features."""
    vl, image, valid, state, noise, prefix = observation(seed)
    bo = BatchFeature(data=dict(backbone_features=vl, backbone_attention_mask=valid, image_mask=image))
    ai = BatchFeature(data=dict(state=state, embodiment_id=torch.tensor([EID])))
    sf = h.state_encoder(state, ai.embodiment_id)
    real = torch.randn
    torch.randn = lambda *a, **k: noise.clone()
    try:
        if delay is None:
            return h.get_action_with_features(vl, sf, ai.embodiment_id, bo, ai, None).action_pred
        ai["rtc_prefix"] = prefix
        return h.get_action_with_features(vl, sf, ai.embodiment_id, bo, ai, {"rtc_delay": delay}).action_pred
    finally:
        torch.randn = real


def split(h, chunk_cls, delay, seed=0):
    """The exported graphs' pipeline in PyTorch: time, mod, kv, then the DiT chunks per step."""
    vl, image, valid, state, noise, prefix = observation(seed)
    pol = SimpleNamespace(action_head=h)
    c = Contract("new_embodiment", EID, 1, 1, (32, 32), V, action_horizon=H, action_dim=A, state_dim=S)
    n = len(h.model.transformer_blocks)
    neg = torch.tensor(MASK_NEG)
    pool = {"state_features": h.state_encoder(state, torch.tensor([EID])),
            "text_bias": torch.where(~image & valid, 0.0, neg)[:, None],
            "image_bias": torch.where(image & valid, 0.0, neg)[:, None],
            "rtc_mask": (torch.arange(H) < (delay or 0)).float()[None, :, None],
            "prefix": prefix}
    kv = KvChunk(pol, [i for i in range(n) if _is_cross(i)])
    pool.update(zip(kv.out_names, kv(vl)))
    mods = [ModChunk(pol, 0, 2, False), ModChunk(pol, 2, n, True)]
    chunks = [chunk_cls(pol, 0, 1, c), chunk_cls(pol, 1, 3, c), chunk_cls(pol, 3, n, c)]
    time_emb = TimeEmb(pol)
    actions = noise
    for t in c.timesteps():
        t_proj, pool["tau"] = time_emb(torch.tensor([t]))
        for m in mods:
            pool.update(zip(m.out_names, m(t_proj)))
        pool["actions"] = actions
        for ch in chunks:
            pool["h"] = ch(*[pool[k] for k in ch.in_names])
        actions = pool["h"]
    return actions, prefix


@torch.no_grad()
def test_stock_split_matches_stock_sampler(head):
    """The harness itself: the stock export chunks reproduce the stock sampler."""
    out, _ = split(head, DitChunk, None)
    torch.testing.assert_close(out, reference(head, None), rtol=0, atol=2e-5)


@torch.no_grad()
def test_zero_mask_is_the_stock_export(head):
    stock, _ = split(head, DitChunk, None)
    rtc, _ = split(head, rtc_export.RtcDitChunk, 0)
    torch.testing.assert_close(rtc, stock, rtol=0, atol=1e-6)


@torch.no_grad()
@pytest.mark.parametrize("delay", [0, 1, 4, 9])
def test_rtc_split_matches_rtc_sampler(head, delay):
    rtc_patch.apply(None)
    out, prefix = split(head, rtc_export.RtcDitChunk, delay)
    ref = reference(head, delay)
    torch.testing.assert_close(out, ref, rtol=0, atol=2e-5)
    torch.testing.assert_close(out[:, :delay], prefix[:, :delay], rtol=0, atol=0)


@torch.no_grad()
def test_free_rows_see_the_prefix(head):
    a, _ = split(head, rtc_export.RtcDitChunk, 4, seed=0)
    stock, _ = split(head, DitChunk, None, seed=0)
    assert float((a[:, 4:] - stock[:, 4:]).abs().max()) > 1e-4


def test_graph_inputs(head):
    pol = SimpleNamespace(action_head=head)
    c = Contract("new_embodiment", EID, 1, 1, (32, 32), V, action_horizon=H, action_dim=A, state_dim=S)
    n = len(head.model.transformer_blocks)
    first, mid, last = (rtc_export.RtcDitChunk(pol, a, b, c) for a, b in ((0, 1), (1, 3), (3, n)))
    assert first.in_names[-2:] == ["rtc_mask", "prefix"] and last.in_names[-2:] == ["rtc_mask", "prefix"]
    assert mid.in_names[-1] == "rtc_mask" and "prefix" not in mid.in_names
    for ch in (first, mid, last):
        ex = ch.example(c, 48, 96, 96, V)
        assert len(ex) == len(ch.in_names)
        ch(*ex)
