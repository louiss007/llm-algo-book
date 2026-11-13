"""第 5 章配套测试（tests/test_modern.py）。"""
import pytest
import torch

from ch05 import (
    LayerNorm,
    ModelConfig,
    RMSNorm,
    SwiGLU,
    TinyLM,
    TransformerBlock,
    exact_params,
    flops_per_token,
)


def make_cfg(**kw):
    base = dict(vocab_size=32, d_model=32, n_layers=2, n_heads=4,
                d_ff=64, max_seq_len=32)
    base.update(kw)
    return ModelConfig(**base)


def test_rmsnorm_formula():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16)
    norm = RMSNorm(16, eps=1e-5)
    with torch.no_grad():
        norm.gamma.uniform_(0.5, 1.5)
    expected = norm.gamma * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    assert torch.allclose(norm(x), expected, atol=1e-6)


def test_rmsnorm_scale_invariance():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16)
    norm = RMSNorm(16, eps=0.0)
    assert torch.allclose(norm(10 * x), norm(x), atol=1e-5)


def test_rmsnorm_equals_ln_on_zero_mean_input():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16)
    x = x - x.mean(-1, keepdim=True)
    rms = RMSNorm(16, eps=1e-5)
    ln = LayerNorm(16, 1e-5, bias=True)
    with torch.no_grad():
        ln.gamma.copy_(rms.gamma)
        ln.beta.zero_()
    assert torch.allclose(rms(x), ln(x), atol=1e-5)


def test_shift_sensitivity():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16)
    rms = RMSNorm(16, eps=1e-5)
    ln = LayerNorm(16, 1e-5, bias=True)
    with torch.no_grad():
        ln.gamma.copy_(rms.gamma)
        ln.beta.zero_()
    with torch.no_grad():
        assert (rms(x + 5) - rms(x)).abs().max() > 0.1
        assert (ln(x + 5) - ln(x)).abs().max() < 1e-5


def test_swiglu_params():
    cfg = make_cfg(ffn="swiglu", d_ff=64, bias=False)
    ffn = SwiGLU(cfg)
    assert sum(p.numel() for p in ffn.parameters()) == 3 * 32 * 64
    assert ffn(torch.randn(2, 5, 32)).shape == (2, 5, 32)


def test_weight_tying():
    cfg = make_cfg(tie_embeddings=True)
    model = TinyLM(cfg)
    assert model.lm_head.weight is model.tok_emb.weight
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


@pytest.mark.parametrize("kw", [
    {},
    {"norm": "rmsnorm"},
    {"ffn": "swiglu"},
    {"norm": "rmsnorm", "ffn": "swiglu", "bias": False},
    {"norm": "rmsnorm", "ffn": "swiglu", "bias": False, "tie_embeddings": True},
    {"norm": "rmsnorm", "ffn": "swiglu", "bias": False, "parallel_block": True},
])
def test_exact_parameter_count(kw):
    cfg = make_cfg(**kw)
    model = TinyLM(cfg)
    assert sum(p.numel() for p in model.parameters()) == exact_params(cfg)


def test_equal_flops_at_eight_thirds():
    base = make_cfg(d_ff=256)
    modern = make_cfg(ffn="swiglu", d_ff=170)
    rel = abs(flops_per_token(base, 64) - flops_per_token(modern, 64))
    assert rel / flops_per_token(base, 64) < 0.01


def test_parallel_block_definition():
    torch.manual_seed(0)
    cfg = make_cfg(parallel_block=True)
    block = TransformerBlock(cfg).eval()
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        h = block.norm1(x)
        expected = x + block.attn(h) + block.mlp(h)
        assert torch.allclose(block(x), expected, atol=1e-6)


def test_causality_modern():
    torch.manual_seed(0)
    cfg = make_cfg(norm="rmsnorm", ffn="swiglu", bias=False,
                   pos_encoding="rope", max_seq_len=64)
    model = TinyLM(cfg).eval()
    ids = torch.randint(0, 32, (2, 16))
    changed = ids.clone()
    changed[:, 5] = (changed[:, 5] + 1) % 32
    with torch.no_grad():
        a = model(ids).logits
        b = model(changed).logits
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-6)
    assert not torch.allclose(a[:, 5:], b[:, 5:], atol=1e-6)
