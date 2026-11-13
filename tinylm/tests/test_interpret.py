"""第 28 章：探针、消融、修补与 SAE 的只读分析。"""
import torch

from tinylm.config import ModelConfig
from tinylm.interpret.core import (ablate_heads, activation_at,
                                   attention_maps, linear_probe_accuracy,
                                   logit_margin, patched_output, recovery,
                                   sae_metrics, train_sae)
from tinylm.model import TinyLM


def make_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4,
                      d_ff=64, max_seq_len=16, pos_encoding="nope")
    return TinyLM(cfg).eval(), cfg


def test_logit_margin_is_finite():
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    value = logit_margin(logits, targets)
    assert value.dim() == 0 and torch.isfinite(value)


def test_linear_probe_learns_separable_data():
    torch.manual_seed(0)
    y = torch.randint(0, 2, (120,))
    centers = torch.tensor([[2.0, 0.0], [-2.0, 0.0]])
    X = centers[y] + 0.3 * torch.randn(120, 2)
    acc = linear_probe_accuracy(X[:80], y[:80], X[80:], y[80:], 2,
                                steps=300, lr=1e-2, seed=0)
    assert acc > 0.9


def test_attention_maps_are_row_stochastic():
    model, _ = make_model()
    ids = torch.randint(0, 32, (2, 6))
    maps = attention_maps(model, ids, layer=0)
    assert maps.shape == (2, 4, 6, 6)
    assert torch.allclose(maps.sum(-1), torch.ones(2, 4, 6), atol=1e-5)


def test_ablate_heads_restores_hook():
    model, _ = make_model()
    ids = torch.randint(0, 32, (1, 6))
    with torch.no_grad():
        base = model(ids).logits
        with ablate_heads(model, 0, [0, 2]):
            ablated = model(ids).logits
        after = model(ids).logits
    assert not torch.allclose(base, ablated)
    assert torch.allclose(base, after)
    assert len(model.blocks[0].attn.out_proj._forward_pre_hooks) == 0


def test_patched_output_with_same_value_is_identity():
    model, _ = make_model()
    ids = torch.randint(0, 32, (1, 5))
    module = model.blocks[0]
    value = activation_at(model, module, 3, ids)
    with torch.no_grad():
        base = model(ids).logits
        with patched_output(module, 3, value):
            patched = model(ids).logits
    assert torch.allclose(base, patched, atol=1e-6)


def test_recovery_definition():
    assert recovery(0.5, 1.0, 0.0) == 0.5
    try:
        recovery(0.0, 1.0, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("相同的两端应报错")


def test_sae_metrics_are_finite():
    torch.manual_seed(0)
    X = torch.randn(128, 8)
    sae = train_sae(X, dict_size=16, l1=0.02, steps=120, batch_size=32, seed=0)
    metrics = sae_metrics(sae, X)
    assert metrics["mse"] >= 0.0
    assert metrics["l0"] >= 0.0
    assert isinstance(metrics["dead"], int)
