"""第 3 章配套测试（tests/test_model.py）。"""
import pytest
import torch

from ch03 import ModelConfig, TinyLM


def make_model(**kw):
    base = dict(vocab_size=32, d_model=32, n_layers=2, n_heads=4,
                d_ff=64, max_seq_len=64)
    base.update(kw)
    return TinyLM(ModelConfig(**base))


@pytest.mark.parametrize("batch,seq", [(1, 1), (2, 17), (1, 64)])
def test_shapes(batch, seq):
    model = make_model()
    ids = torch.randint(0, 32, (batch, seq))
    logits = model(ids).logits
    assert logits.shape == (batch, seq, 32)


def test_causality():
    torch.manual_seed(0)
    model = make_model().eval()
    ids = torch.randint(0, 32, (2, 16))
    changed = ids.clone()
    changed[:, 5] = (changed[:, 5] + 1) % 32
    with torch.no_grad():
        a = model(ids).logits
        b = model(changed).logits
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-6)
    assert not torch.allclose(a[:, 5:], b[:, 5:], atol=1e-6)


def test_parameter_count():
    cfg = ModelConfig(vocab_size=16, d_model=64, n_layers=2, n_heads=4,
                      d_ff=256, max_seq_len=64)
    model = TinyLM(cfg)
    d, f, V, L = cfg.d_model, cfg.d_ff, cfg.vocab_size, cfg.max_seq_len
    expected = cfg.n_layers * (4 * d**2 + 2 * d * f + 9 * d + f) \
        + 2 * d + 2 * V * d + L * d
    assert sum(p.numel() for p in model.parameters()) == expected


def test_loss_backward():
    torch.manual_seed(0)
    model = make_model(vocab_size=16)
    ids = torch.randint(0, 16, (4, 12))
    out = model(ids[:, :-1], labels=ids[:, 1:])
    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) == sum(1 for p in model.parameters() if p.requires_grad)
    assert all(torch.isfinite(g).all() for g in grads)
