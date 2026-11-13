"""第 6 章自测：MHA/MQA/GQA/MLA 的形状、因果性、参数量、缓存与吸收等价性。"""
import pytest
import torch

import ch06
from ch04 import PositionAwareAttention


def base_config(**kw):
    common = dict(vocab_size=32, d_model=64, n_layers=2, n_heads=8,
                  d_ff=128, max_seq_len=64, bias=False,
                  norm="rmsnorm", ffn="swiglu")
    common.update(kw)
    return ch06.ModelConfig(**common)


def test_mha_matches_chapter4_attention():
    """n_kv_heads = n_heads 时，变体实现必须与第 4 章 MHA 完全一致。"""
    cfg = base_config(attention="mha", pos_encoding="rope")
    ours = ch06.VariantAttention(cfg).eval()
    ref = PositionAwareAttention(cfg).eval()
    ref.load_state_dict(ours.state_dict())
    x = torch.randn(2, 12, cfg.d_model)
    pos = torch.arange(12)
    with torch.no_grad():
        assert torch.allclose(ours(x, position_ids=pos),
                              ref(x, position_ids=pos), atol=1e-6)


def test_gqa_shares_kv_across_heads():
    cfg = base_config(attention="gqa", n_kv_heads=2, pos_encoding="rope")
    attn = ch06.VariantAttention(cfg)
    assert attn.n_kv_heads == 2
    assert attn.repeat == cfg.n_heads // 2
    assert attn.k_proj.out_features == 2 * attn.d_head
    assert attn.v_proj.out_features == 2 * attn.d_head


def test_mqa_has_single_kv_head():
    cfg = base_config(attention="mqa", pos_encoding="rope")
    assert cfg.n_kv_heads == 1
    attn = ch06.VariantAttention(cfg)
    assert attn.k_proj.out_features == attn.d_head
    assert attn.v_proj.out_features == attn.d_head


@pytest.mark.parametrize("name,n_kv", [("mha", 8), ("gqa", 4), ("gqa", 2), ("mqa", 1)])
def test_attention_param_formula(name, n_kv):
    """公式 (6.5)：2 d² + 2 d d_kv（无偏置）。"""
    cfg = base_config(attention=name, n_kv_heads=n_kv, pos_encoding="rope")
    d = cfg.d_model
    d_kv = cfg.n_kv_heads * (d // cfg.n_heads)
    expected = 2 * d * d + 2 * d * d_kv
    actual = sum(p.numel() for p in ch06.make_attention(cfg).parameters())
    assert actual == expected == ch06.attention_params(cfg)


def test_mla_param_formula():
    """公式 (6.13)：2 d² + 3 d d_c + d d_r (h + 1)。"""
    cfg = base_config(attention="mla", d_c=16, d_head_rope=8, pos_encoding="rope")
    d, h, dc, dr = cfg.d_model, cfg.n_heads, cfg.d_c, cfg.d_head_rope
    expected = 2 * d * d + 3 * d * dc + d * dr * (h + 1)
    actual = sum(p.numel() for p in ch06.make_attention(cfg).parameters())
    assert actual == expected == ch06.attention_params(cfg)


def test_kv_cache_formula():
    """公式 (6.6)：2 T L n_kv d_head b。"""
    cfg = base_config(attention="gqa", n_kv_heads=2, pos_encoding="rope")
    T, b = 16, 3
    expected = b * T * cfg.n_layers * 2 * cfg.n_kv_heads * (cfg.d_model // cfg.n_heads) * 2
    assert ch06.kv_cache_bytes(cfg, T, batch=b) == expected


def test_mla_cache_formula():
    """公式 (6.11)：T L (d_c + d_r) b。"""
    cfg = base_config(attention="mla", d_c=16, d_head_rope=8, pos_encoding="rope")
    expected = 4 * 32 * cfg.n_layers * (cfg.d_c + cfg.d_head_rope) * 2
    assert ch06.kv_cache_bytes(cfg, 32, batch=4) == expected


@pytest.mark.parametrize("name,kw", [
    ("mha", {}), ("mqa", {}), ("gqa", {"n_kv_heads": 2}),
    ("mla", {"d_c": 16, "d_head_rope": 8}),
])
def test_causality(name, kw):
    """改变位置 t 的输入，不得影响位置 < t 的输出。"""
    cfg = base_config(attention=name, pos_encoding="rope", **kw)
    attn = ch06.make_attention(cfg).eval()
    x = torch.randn(1, 10, cfg.d_model)
    pos = torch.arange(10)
    with torch.no_grad():
        y = attn(x, position_ids=pos)
        x2 = x.clone()
        x2[:, 6] += 5.0
        y2 = attn(x2, position_ids=pos)
    assert torch.allclose(y[:, :6], y2[:, :6], atol=1e-6)
    assert not torch.allclose(y[:, 6:], y2[:, 6:], atol=1e-4)


def test_divisibility_is_checked():
    with pytest.raises(AssertionError):
        base_config(attention="gqa", n_kv_heads=3, pos_encoding="rope")


def test_mla_absorption_identity():
    """公式 (6.12)：权重吸收前后的内容通路输出一致。"""
    cfg = base_config(attention="mla", d_c=16, d_head_rope=8, pos_encoding="rope")
    assert ch06.mla_absorption_error(cfg) < 1e-5


def test_mla_cache_beats_gqa_and_mha():
    cfgs = {
        "mha": base_config(attention="mha", pos_encoding="rope"),
        "gqa": base_config(attention="gqa", n_kv_heads=2, pos_encoding="rope"),
        "mla": base_config(attention="mla", d_c=16, d_head_rope=8, pos_encoding="rope"),
    }
    sizes = {k: ch06.kv_cache_bytes(c, 32) for k, c in cfgs.items()}
    assert sizes["mla"] < sizes["gqa"] < sizes["mha"]


@pytest.mark.parametrize("name,kw", [
    ("mha", {}), ("mqa", {}), ("gqa", {"n_kv_heads": 2}),
    ("mla", {"d_c": 16, "d_head_rope": 8}),
])
def test_tinylm_forward_and_loss(name, kw):
    cfg = base_config(attention=name, pos_encoding="rope", **kw)
    model = ch06.TinyLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(ids, labels=ids)
    assert out.logits.shape == (2, 12, cfg.vocab_size)
    assert torch.isfinite(out.loss)
