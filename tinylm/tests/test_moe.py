"""第 7 章自测：MoE 路由、分派、容量、参数账本、辅助损失与因果性。"""
import math

import pytest
import torch

import ch06
import ch07


def base_config(**kw):
    common = dict(vocab_size=32, d_model=64, n_layers=2, n_heads=4,
                  d_ff=128, max_seq_len=64, bias=False,
                  norm="rmsnorm", ffn="swiglu", pos_encoding="rope")
    common.update(kw)
    return ch07.ModelConfig(**common)


def moe_config(**kw):
    m = dict(n_experts=8, top_k=2, d_ff_expert=64)
    m.update(kw)
    return base_config(moe=ch07.MoEConfig(**m))


def test_dense_moe_none_matches_chapter6():
    """moe=None 时，第 7 章的模型必须与第 6 章逐位一致。"""
    cfg6 = ch06.ModelConfig(vocab_size=32, d_model=64, n_layers=2, n_heads=4,
                            d_ff=128, max_seq_len=64, bias=False,
                            norm="rmsnorm", ffn="swiglu", pos_encoding="rope")
    cfg7 = base_config()
    ref, ours = ch06.TinyLM(cfg6).eval(), ch07.TinyLM(cfg7).eval()
    ours.load_state_dict(ref.state_dict())
    ids = torch.randint(0, 32, (2, 12))
    with torch.no_grad():
        a = ref(ids)
        b = ours(ids)
    assert torch.allclose(a.logits, b.logits, atol=1e-6)
    assert b.aux_loss is None


def test_param_formula_matches_model():
    """公式 (7.10)–(7.12)：总参数逐位一致，含共享专家。"""
    cfg = moe_config(n_shared=1)
    total = ch07.model_params(cfg)
    active = ch07.model_params(cfg, active=True)
    actual = sum(p.numel() for p in ch07.TinyLM(cfg).parameters())
    d, f, E = cfg.d_model, cfg.moe.d_ff_expert, 8
    expert = 3 * d * f
    assert total == actual
    assert total - active == cfg.n_layers * (E - cfg.moe.top_k) * expert


def test_router_experts_and_shared_count():
    cfg = moe_config(n_shared=2)
    model = ch07.TinyLM(cfg)
    moe = model.blocks[0].mlp
    assert len(moe.experts) == 8
    assert len(moe.shared) == 2
    assert moe.router.weight.shape == (8, cfg.d_model)


def test_routing_weights_sum_to_one():
    """公式 (7.5)：Top-K 的归一化门控之和为 1。"""
    cfg = moe_config()
    moe = ch07.MoE(cfg).eval()
    x = torch.randn(5, 7, cfg.d_model)
    with torch.no_grad():
        probs = moe.router(x.reshape(-1, cfg.d_model)).softmax(-1)
        top_p, _ = probs.topk(moe.top_k, dim=-1)
        gates = top_p / top_p.sum(-1, keepdim=True)
    assert torch.allclose(gates.sum(-1), torch.ones(35), atol=1e-6)


def test_identical_experts_reduce_to_dense():
    """所有专家权重相同时，无论路由如何，输出等于单个专家。"""
    cfg = moe_config(capacity_factor=None)
    torch.manual_seed(0)
    moe = ch07.MoE(cfg).eval()
    for expert in moe.experts[1:]:
        expert.load_state_dict(moe.experts[0].state_dict())
    x = torch.randn(3, 9, cfg.d_model)
    with torch.no_grad():
        y, _ = moe(x)
        y_ref = moe.experts[0](x)
    assert torch.allclose(y, y_ref, atol=1e-6)


def test_capacity_limits_dispatch():
    """公式 (7.7)–(7.8)：每个专家接收的 Token 不超过容量。"""
    cfg = moe_config(capacity_factor=0.4, top_k=2)
    torch.manual_seed(1)
    moe = ch07.MoE(cfg).eval()
    x = torch.randn(16, 16, cfg.d_model)
    n = 16 * 16
    capacity = math.ceil(0.4 * n * 2 / 8)
    with torch.no_grad():
        moe(x)
    assert int(moe.last_load.max()) <= capacity
    assert moe.last_dropped > 0
    assert int(moe.last_load.sum()) + moe.last_dropped == n * 2


def test_no_capacity_means_no_drop():
    cfg = moe_config(capacity_factor=None)
    moe = ch07.MoE(cfg).eval()
    x = torch.randn(4, 10, cfg.d_model)
    with torch.no_grad():
        moe(x)
    assert moe.last_dropped == 0
    assert int(moe.last_load.sum()) == 4 * 10 * 2


def test_load_balance_loss_formula():
    """公式 (7.18)：均衡时为 α，失衡时大于 α。"""
    balanced = ch07.load_balance_loss(
        torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.5]), 0.01, 2)
    skewed = ch07.load_balance_loss(
        torch.tensor([0.9, 0.1]), torch.tensor([0.9, 0.1]), 0.01, 2)
    assert abs(balanced.item() - 0.01) < 1e-9
    assert skewed > balanced


def test_aux_loss_only_when_enabled_and_training():
    cfg = moe_config(aux_loss_coef=0.01)
    model = ch07.TinyLM(cfg)
    ids = torch.randint(0, 32, (2, 10))
    model.train()
    out = model(ids, labels=ids)
    assert out.aux_loss is not None and out.aux_loss.item() >= 0.0
    model.eval()
    with torch.no_grad():
        out_eval = model(ids, labels=ids)
    assert out_eval.aux_loss is None

    none_cfg = moe_config(aux_loss_coef=0.0)
    model0 = ch07.TinyLM(none_cfg).train()
    assert model0(ids, labels=ids).aux_loss is None


@pytest.mark.parametrize("kw", [
    {}, {"n_shared": 1}, {"capacity_factor": None},
])
def test_causality(kw):
    """改动位置 t 的输入，位置 < t 的 logits 不变。"""
    cfg = moe_config(**kw)
    torch.manual_seed(0)
    model = ch07.TinyLM(cfg).eval()
    ids = torch.randint(0, 32, (1, 10))
    with torch.no_grad():
        y = model(ids).logits
        ids2 = ids.clone()
        ids2[:, 6] = (ids2[:, 6] + 5) % 32
        y2 = model(ids2).logits
    assert torch.allclose(y[:, :6], y2[:, :6], atol=1e-6)
    assert not torch.allclose(y[:, 6:], y2[:, 6:], atol=1e-6)


@pytest.mark.parametrize("kw", [
    {}, {"n_shared": 1}, {"capacity_factor": None},
])
def test_tinylm_forward_and_loss(kw):
    cfg = moe_config(**kw)
    model = ch07.TinyLM(cfg)
    ids = torch.randint(0, 32, (2, 12))
    out = model(ids, labels=ids)
    assert out.logits.shape == (2, 12, 32)
    assert torch.isfinite(out.loss)


def test_invalid_top_k_is_rejected():
    with pytest.raises(AssertionError):
        moe_config(top_k=9)
