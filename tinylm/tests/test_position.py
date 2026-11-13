"""第 4 章配套测试（tests/test_position.py）。"""
import math

import pytest
import torch

from ch04 import (
    ModelConfig, RoPE, RoPEScalingConfig, TinyLM,
    alibi_slopes, rope_inv_freq, sinusoidal_positions,
)


def test_sinusoidal_shift():
    d, L, delta = 16, 64, 7
    pe = sinusoidal_positions(L, d)
    omega = 10000.0 ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
    err = 0.0
    for t in range(L - delta):
        for i in range(d // 2):
            a = float(delta * omega[i])
            R = torch.tensor([[math.cos(a), math.sin(a)],
                              [-math.sin(a), math.cos(a)]])
            err = max(err, float(
                (pe[t + delta, 2 * i:2 * i + 2] - R @ pe[t, 2 * i:2 * i + 2]
                 ).abs().max()))
    assert err < 1e-5


def test_rope_shift_invariance():
    d_head, L = 16, 64
    rope = RoPE(d_head, 10000.0, None, L)
    q = torch.randn(1, 1, 1, d_head)
    k = torch.randn(1, 1, 1, d_head)
    qr, kr = rope(q.expand(1, 1, L, d_head), k.expand(1, 1, L, d_head),
                  torch.arange(L))
    S = (qr @ kr.transpose(-2, -1))[0, 0]
    err = 0.0
    for c in range(1, L):
        err = max(err, float((S[:L - c, :L - c] - S[c:, c:]).abs().max()))
    assert err < 1e-4


def test_rope_norm():
    d_head, L = 16, 32
    rope = RoPE(d_head, 10000.0, None, L)
    q = torch.randn(2, 3, L, d_head)
    k = torch.randn(2, 3, L, d_head)
    qr, kr = rope(q, k, torch.arange(L))
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    assert torch.allclose(kr.norm(dim=-1), k.norm(dim=-1), atol=1e-5)


def test_rope_scaling_shapes():
    d_head = 64
    for method in ("none", "pi", "ntk", "yarn"):
        inv = rope_inv_freq(d_head, 10000.0,
                            RoPEScalingConfig(method, 4.0), 8192)
        assert inv.shape == (d_head // 2,)
        assert (inv > 0).all()
    none = rope_inv_freq(d_head, 10000.0, None, 8192)
    yarn = rope_inv_freq(d_head, 10000.0, RoPEScalingConfig("yarn", 4.0), 8192)
    assert yarn[0] == pytest.approx(none[0], rel=1e-5)
    assert yarn[-1] < none[-1] / 2


def test_alibi_slopes():
    slopes = alibi_slopes(4)
    expected = torch.tensor([0.25, 0.0625, 0.015625, 0.00390625])
    assert torch.allclose(slopes, expected)


def test_alibi_bias_shape():
    cfg = ModelConfig(vocab_size=16, d_model=32, n_layers=2, n_heads=4,
                      d_ff=64, max_seq_len=32, pos_encoding="alibi")
    model = TinyLM(cfg)
    bias = model.blocks[0].attn.alibi(8, torch.device("cpu"))
    assert bias.shape == (4, 8, 8)
    assert torch.allclose(bias[0, 7, 0], torch.tensor(-0.25 * 7))


def test_rope_extrapolation_shape():
    cfg = ModelConfig(vocab_size=16, d_model=32, n_layers=2, n_heads=4,
                      d_ff=64, max_seq_len=64, pos_encoding="rope")
    model = TinyLM(cfg).eval()
    ids = torch.randint(0, 16, (1, 128))
    with torch.no_grad():
        out = model(ids)
    assert out.logits.shape == (1, 128, 16)


def test_learned_cannot_exceed_table():
    cfg = ModelConfig(vocab_size=16, d_model=32, n_layers=2, n_heads=4,
                      d_ff=64, max_seq_len=64, pos_encoding="learned")
    model = TinyLM(cfg).eval()
    with pytest.raises(AssertionError):
        model(torch.randint(0, 16, (1, 65)))


def test_causality_rope():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=16, d_model=64, n_layers=2, n_heads=4,
                      d_ff=256, max_seq_len=128, pos_encoding="rope")
    model = TinyLM(cfg).eval()
    ids = torch.randint(0, 16, (2, 32))
    changed = ids.clone()
    changed[:, 5] = (changed[:, 5] + 1) % 16
    with torch.no_grad():
        a = model(ids).logits
        b = model(changed).logits
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-6)
    assert not torch.allclose(a[:, 5:], b[:, 5:], atol=1e-6)
