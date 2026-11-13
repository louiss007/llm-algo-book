"""第 8 章自测：线性注意力、选择性 SSM、混合块、状态账本与因果性。"""
import math

import pytest
import torch

import ch07
import ch08


def base_config(**kw):
    common = dict(vocab_size=32, d_model=64, n_layers=2, n_heads=4,
                  d_ff=128, max_seq_len=64, bias=False,
                  norm="rmsnorm", ffn="swiglu", pos_encoding="rope")
    common.update(kw)
    return ch08.ModelConfig(**common)


def ssm_config(**kw):
    c = dict(block_type="ssm", ssm_d_state=16, ssm_expand=2)
    c.update(kw)
    return base_config(**c)


def test_attention_default_matches_chapter7():
    """block_type="attention" 时，第 8 章必须与第 7 章逐位一致。"""
    cfg7 = ch07.ModelConfig(vocab_size=32, d_model=64, n_layers=2, n_heads=4,
                            d_ff=128, max_seq_len=64, bias=False,
                            norm="rmsnorm", ffn="swiglu", pos_encoding="rope")
    cfg8 = base_config(block_type="attention")
    ref, ours = ch07.TinyLM(cfg7).eval(), ch08.TinyLM(cfg8).eval()
    ours.load_state_dict(ref.state_dict())
    ids = torch.randint(0, 32, (2, 12))
    with torch.no_grad():
        a, b = ref(ids), ours(ids)
    assert torch.allclose(a.logits, b.logits, atol=1e-6)
    assert b.aux_loss is None


def test_linear_attention_matches_quadratic_form():
    """公式 (8.1)–(8.6)：线性注意力的扫描形式等于显式二次形式。"""
    cfg = base_config(block_type="linear_attention")
    torch.manual_seed(0)
    layer = ch08.LinearAttention(cfg).eval()
    x = torch.randn(2, 9, cfg.d_model)
    with torch.no_grad():
        y = layer(x)
        q = torch.nn.functional.elu(layer.q_proj(x)) + 1.0
        k = torch.nn.functional.elu(layer.k_proj(x)) + 1.0
        v = layer.v_proj(x)
        scores = q @ k.transpose(-1, -2)
        mask = torch.tril(torch.ones(9, 9, dtype=torch.bool))
        scores = scores.masked_fill(~mask, 0.0)
        ref = layer.out_proj((scores @ v) / scores.sum(-1, keepdim=True).clamp_min(1e-6))
    assert torch.allclose(y, ref, atol=1e-5)


@pytest.mark.parametrize("T", [1, 2, 17, 32])
def test_parallel_scan_matches_sequential(T):
    """公式 (8.17)：并行扫描与逐步递推一致。"""
    torch.manual_seed(0)
    B, C = 2, 5
    a = torch.rand(B, T, C) * 0.9
    b = torch.randn(B, T, C)
    h = torch.zeros(B, C)
    ref = []
    for t in range(T):
        h = a[:, t] * h + b[:, t]
        ref.append(h)
    ref = torch.stack(ref, dim=1)
    assert torch.allclose(ch08.scan_states(a, b), ref, atol=1e-6)


def test_ssm_shapes_and_state():
    cfg = ssm_config()
    model = ch08.TinyLM(cfg).eval()
    ids = torch.randint(0, 32, (2, 11))
    with torch.no_grad():
        out = model(ids, labels=ids)
    state = model.blocks[0].mixer.last_state
    assert out.logits.shape == (2, 11, 32)
    assert torch.isfinite(out.loss)
    assert state.shape == (2, cfg.ssm_expand * cfg.d_model, cfg.ssm_d_state)


def test_state_size_independent_of_sequence_length():
    cfg = ssm_config()
    torch.manual_seed(0)
    model = ch08.TinyLM(cfg).eval()
    with torch.no_grad():
        for T in (4, 64):
            model(torch.randint(0, 32, (1, T)))
            state = model.blocks[0].mixer.last_state
            assert state.shape == (1, cfg.ssm_expand * cfg.d_model, cfg.ssm_d_state)


def test_ssm_layer_param_formula():
    """公式 (8.16)：单层 SSM 参数公式与实测一致。"""
    cfg = ssm_config()
    mixer = ch08.SelectiveSSM(cfg)
    actual = sum(p.numel() for p in mixer.parameters())
    assert actual == ch08.ssm_layer_params(cfg)


@pytest.mark.parametrize("kind", ["attention", "ssm", "linear_attention", "hybrid"])
def test_model_params_formula_matches_model(kind):
    """按层累加的参数量账本与实测一致。"""
    cfg = ssm_config(block_type=kind, hybrid_attn_every=2)
    actual = sum(p.numel() for p in ch08.TinyLM(cfg).parameters())
    assert ch08.model_params(cfg) == actual


@pytest.mark.parametrize("kind", ["ssm", "linear_attention", "hybrid"])
def test_causality(kind):
    """改动位置 t 的输入，位置 < t 的 logits 不变。"""
    cfg = ssm_config(block_type=kind, hybrid_attn_every=2)
    torch.manual_seed(0)
    model = ch08.TinyLM(cfg).eval()
    ids = torch.randint(0, 32, (1, 10))
    with torch.no_grad():
        y = model(ids).logits
        ids2 = ids.clone()
        ids2[:, 6] = (ids2[:, 6] + 5) % 32
        y2 = model(ids2).logits
    assert torch.allclose(y[:, :6], y2[:, :6], atol=1e-6)
    assert not torch.allclose(y[:, 6:], y2[:, 6:], atol=1e-6)


def test_long_sequence_is_stable():
    cfg = ssm_config()
    torch.manual_seed(0)
    model = ch08.TinyLM(cfg).eval()
    with torch.no_grad():
        out = model(torch.randint(0, 32, (1, 512)))
        state = model.blocks[0].mixer.last_state
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(state).all()
    assert state.abs().max() < 1e3


def test_hybrid_layer_mix():
    cfg = ssm_config(block_type="hybrid", hybrid_attn_every=4, n_layers=4)
    assert ch08.layer_kinds(cfg) == ["ssm", "ssm", "ssm", "attn"]
    model = ch08.TinyLM(cfg)
    assert isinstance(model.blocks[3], ch07.TransformerBlock)


def test_state_bytes_grows_only_for_attention():
    common = dict(vocab_size=32, d_model=64, n_layers=2, n_heads=4,
                  d_ff=128, max_seq_len=64, bias=False, norm="rmsnorm",
                  ffn="swiglu", pos_encoding="rope")
    attn = ch08.ModelConfig(block_type="attention", **common)
    ssm = ch08.ModelConfig(block_type="ssm", **common)
    linear = ch08.ModelConfig(block_type="linear_attention", **common)
    assert ch08.state_bytes(ssm, 1, 8192) == ch08.state_bytes(ssm, 1, 131072)
    assert ch08.state_bytes(linear, 1, 8192) == ch08.state_bytes(linear, 1, 131072)
    assert ch08.state_bytes(attn, 1, 131072) > 10 * ch08.state_bytes(attn, 1, 8192)
    assert ch08.state_bytes(ssm, 1, 8192) < ch08.state_bytes(linear, 1, 8192)


def test_hybrid_forward_and_loss():
    cfg = ssm_config(block_type="hybrid", hybrid_attn_every=2)
    model = ch08.TinyLM(cfg)
    ids = torch.randint(0, 32, (2, 12))
    out = model(ids, labels=ids)
    assert out.logits.shape == (2, 12, 32)
    assert torch.isfinite(out.loss)


def test_invalid_block_type_is_rejected():
    with pytest.raises(AssertionError):
        base_config(block_type="transformer")


def test_zero_features_do_not_produce_nan():
    """φ 为非负特征，归一化分母不应为零。"""
    cfg = base_config(block_type="linear_attention")
    layer = ch08.LinearAttention(cfg).eval()
    x = torch.zeros(1, 3, cfg.d_model)
    with torch.no_grad():
        y = layer(x)
    assert torch.isfinite(y).all()
