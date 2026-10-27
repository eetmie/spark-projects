"""CPU tests for training-time RTC on GR00T N1.7 (tiny random action head, no backbone).

    vla-onnx/groot/.venv-groot17/bin/python -m pytest vla-onnx/groot/excavator/tests -q
"""
import pathlib
import sys

import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import rtc_patch  # noqa: E402

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead  # noqa: E402
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config  # noqa: E402

B, H, A, S, V = 3, 10, 8, 6, 12


def config(alternate=True):
    return Gr00tN1d7Config(
        backbone_embedding_dim=32, hidden_size=32, input_embedding_dim=48, max_action_dim=A, max_state_dim=S,
        action_horizon=H, max_num_embodiments=4, state_dropout_prob=0.0, use_alternate_vl_dit=alternate,
        diffusion_model_cfg=dict(positional_embeddings=None, num_layers=4, num_attention_heads=2,
                                 attention_head_dim=24, norm_type="ada_norm", dropout=0.0, final_dropout=False,
                                 output_dim=32, interleave_self_attention=True))


def inputs(seed=0, valid_rows=H):
    g = torch.Generator().manual_seed(seed)
    image = torch.zeros(B, V, dtype=torch.bool)
    image[:, :6] = True
    backbone = BatchFeature(data=dict(backbone_features=torch.randn(B, V, 32, generator=g),
                                      backbone_attention_mask=torch.ones(B, V, dtype=torch.bool), image_mask=image))
    mask = torch.zeros(B, H, A)
    mask[:, :valid_rows, :4] = 1  # 4 real action dims, padded rows/dims masked as GR00T does
    action = BatchFeature(data=dict(state=torch.randn(B, 1, S, generator=g), action=torch.randn(B, H, A, generator=g),
                                    embodiment_id=torch.tensor([1, 1, 1]), action_mask=mask))
    return backbone, action


def clone(bf):
    return BatchFeature(data={k: v.clone() for k, v in bf.items()})


@pytest.fixture(params=[True, False], ids=["alternate_dit", "dit"])
def head(request):
    rtc_patch.remove()
    torch.manual_seed(0)
    h = Gr00tN1d7ActionHead(config(request.param)).float()
    yield h
    rtc_patch.remove()


def run_loss(h, seed=1):
    backbone, action = inputs()
    torch.manual_seed(seed)
    return h(clone(backbone), clone(action))


def sample(h, seed=2, **kw):
    backbone, action = inputs()
    del action["action"]  # the stock sampler reads an "action" key as NVIDIA's inference RTC
    features = h._encode_features(backbone, action)
    torch.manual_seed(seed)
    if kw:
        action["rtc_prefix"] = kw.pop("prefix")
    return h.get_action_with_features(features.backbone_features, features.state_features, action.embodiment_id,
                                      h.process_backbone_output(backbone), action, kw or None).action_pred


def test_pinned_source():
    assert rtc_patch.source_mismatches() == []


def test_refuses_another_revision(monkeypatch):
    monkeypatch.setitem(rtc_patch.SOURCE_SHA256, "DiT.forward", "0" * 64)
    with pytest.raises(RuntimeError, match="re-verify"):
        rtc_patch.apply(None)


def test_patched_loss_without_rtc_matches_stock(head):
    head.train()
    stock = run_loss(head)["loss"]
    rtc_patch.apply(None)
    patched = run_loss(head)["loss"]
    torch.testing.assert_close(patched, stock, rtol=0, atol=1e-6)


def test_patched_sampler_matches_stock(head):
    head.eval()
    stock = sample(head)
    rtc_patch.apply(None)
    torch.testing.assert_close(sample(head), stock, rtol=0, atol=0)  # stock code path
    zero = sample(head, prefix=torch.zeros(B, H, A), rtc_delay=0)  # per-row path, no prefix
    torch.testing.assert_close(zero, stock, rtol=0, atol=1e-5)


def test_rtc_loss_masks_prefix_rows(head):
    rtc_patch.apply(rtc_patch.RtcTraining(max_delay=5))
    head.train()
    out = run_loss(head, seed=3)
    torch.manual_seed(3)
    torch.randn(B, H, A)  # noise
    head.sample_time(B, device="cpu", dtype=torch.float32)
    delay = rtc_patch.sample_delay(B, head.rtc, "cpu")
    mask = out["action_mask"]
    for b in range(B):
        assert torch.all(mask[b, :delay[b]] == 0)
        assert torch.all(mask[b, delay[b]:, :4] == 1)
    torch.testing.assert_close(out["loss"], out["action_loss"].sum() / (mask.sum() + 1e-6))
    out["loss"].backward()  # differentiable through the per-token paths


def test_eval_mode_has_no_prefix(head):
    rtc_patch.apply(rtc_patch.RtcTraining(max_delay=5))
    head.eval()
    out = run_loss(head)
    torch.testing.assert_close(out["action_mask"], inputs()[1].action_mask)


def test_prefix_rows_returned_and_used(head):
    head.eval()
    rtc_patch.apply(None)
    prefix = torch.randn(B, H, A, generator=torch.Generator().manual_seed(9))
    out = sample(head, prefix=prefix, rtc_delay=4)
    torch.testing.assert_close(out[:, :4], prefix[:, :4], rtol=0, atol=0)
    other = sample(head, prefix=prefix + 1.0, rtc_delay=4)
    assert float((out[:, 4:] - other[:, 4:]).abs().max()) > 1e-5  # free rows see the prefix


def test_per_row_delay(head):
    head.eval()
    rtc_patch.apply(None)
    prefix = torch.randn(B, H, A)
    out = sample(head, prefix=prefix, rtc_delay=torch.tensor([0, 2, 7]))
    torch.testing.assert_close(out[1, :2], prefix[1, :2], rtol=0, atol=0)
    torch.testing.assert_close(out[2, :7], prefix[2, :7], rtol=0, atol=0)
    with pytest.raises(ValueError):
        sample(head, prefix=prefix, rtc_delay=H)


def test_delay_distributions():
    torch.manual_seed(0)
    u = rtc_patch.sample_delay(20000, rtc_patch.RtcTraining(4, "uniform"), "cpu")
    e = rtc_patch.sample_delay(20000, rtc_patch.RtcTraining(4, "exp"), "cpu")
    assert u.min() == 0 and u.max() == 4 and abs((u == 4).float().mean() - 0.2) < 0.02
    assert e.max() <= 4 and (e == 0).float().mean() > 0.6
    assert torch.all(rtc_patch.sample_delay(5, None, "cpu") == 0)
