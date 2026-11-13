"""第 11 章自测：AdamW、学习率调度、梯度裁剪与 Muon。"""
import math

import pytest
import torch

from tinylm.train import optim as O


def test_adamw_first_step_matches_closed_form():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    p.grad = torch.tensor([2.0])
    opt = O.AdamW([p], lr=0.1, betas=(0.9, 0.95), eps=0.0, weight_decay=0.1)
    opt.step()
    expected = 0.99 - 0.1 * 2.0 / math.sqrt(4.0)
    assert p.item() == pytest.approx(expected, rel=1e-6)


def test_bias_correction_makes_first_step_lr_sized():
    p = torch.nn.Parameter(torch.tensor([0.0]))
    p.grad = torch.tensor([1e-3])
    opt = O.AdamW([p], lr=0.05, betas=(0.9, 0.95), eps=0.0, weight_decay=0.0)
    opt.step()
    assert p.item() == pytest.approx(-0.05, rel=1e-4)


def test_decoupled_decay_is_not_amplified_by_second_moment():
    p1 = torch.nn.Parameter(torch.tensor([1.0]))
    p1.grad = torch.tensor([1e-3])
    O.AdamW([p1], lr=0.1, betas=(0.9, 0.95), eps=0.0,
            weight_decay=0.1).step()

    p2 = torch.nn.Parameter(torch.tensor([1.0]))
    p2.grad = torch.tensor([1e-3])
    torch.optim.Adam([p2], lr=0.1, betas=(0.9, 0.95), eps=0.0,
                     weight_decay=0.1).step()

    # 耦合 L2 的衰减项会除以 sqrt(v)，在本例中被放大约 10 倍
    assert p2.item() - p1.item() == pytest.approx(0.01, abs=1e-4)


def test_param_groups_exclude_biases_and_norms():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3))
    groups = O.make_param_groups(model, weight_decay=0.2)
    assert groups[0]["weight_decay"] == 0.2 and groups[1]["weight_decay"] == 0.0
    assert len(groups[0]["params"]) == 1          # 只有 Linear.weight
    assert len(groups[1]["params"]) == 3          # Linear.bias + LN 的权重与偏置

def test_warmup_endpoints_and_midpoint():
    assert O.lr_factor(0, 100, warmup=10) == 0.0
    assert O.lr_factor(5, 100, warmup=10) == pytest.approx(0.5)
    assert O.lr_factor(10, 100, warmup=10) == 1.0
    assert O.lr_factor(100, 100, warmup=10) == pytest.approx(0.1)


def test_cosine_is_monotone_after_warmup():
    factors = [O.lr_factor(s, 100, warmup=10, kind="cosine") for s in range(10, 101)]
    assert all(a >= b for a, b in zip(factors, factors[1:]))
    assert factors[-1] == pytest.approx(0.1)


def test_wsd_is_flat_then_decays():
    assert O.lr_factor(50, 100, warmup=0, kind="wsd", decay_start=50) == 1.0
    assert O.lr_factor(75, 100, warmup=0, kind="wsd", decay_start=50) == pytest.approx(0.55)
    assert O.lr_factor(100, 100, warmup=0, kind="wsd", decay_start=50) == pytest.approx(0.1)
    assert abs(O.lr_factor(50, 100, warmup=0, kind="wsd", decay_start=50)
               - O.lr_factor(51, 100, warmup=0, kind="wsd", decay_start=50)) < 0.02


def test_unknown_schedule_raises():
    with pytest.raises(ValueError):
        O.lr_factor(1, 10, kind="trapezoid")


def test_clip_grad_norm_scales_globally():
    a = torch.nn.Parameter(torch.zeros(2))
    b = torch.nn.Parameter(torch.zeros(1))
    a.grad = torch.tensor([3.0, 0.0])
    b.grad = torch.tensor([4.0])
    before = O.clip_grad_norm_([a, b], max_norm=1.0)
    after = torch.sqrt((a.grad ** 2).sum() + (b.grad ** 2).sum())
    assert before == pytest.approx(5.0)
    assert after.item() == pytest.approx(1.0, rel=1e-6)


def test_clip_keeps_small_gradients():
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([0.1, 0.2])
    before = O.clip_grad_norm_([p], max_norm=1.0)
    assert before == pytest.approx(math.sqrt(0.05))
    assert torch.allclose(p.grad, torch.tensor([0.1, 0.2]))


def test_spike_detector_flags_outlier():
    det = O.LossSpikeDetector(window=20, z_threshold=3.0, min_history=10)
    for i in range(15):
        z, spike = det.update(1.0 + 0.01 * (i % 2))
        assert not spike
    z, spike = det.update(3.0)
    assert spike and z > 3.0
    z, spike = det.update(1.0)
    assert not spike


def test_newton_schulz_drives_singular_values_to_one():
    torch.manual_seed(0)
    G = torch.randn(8, 5)
    before = torch.linalg.svdvals(G / G.norm())
    Q = O.zeropower_newtonschulz5(G)
    after = torch.linalg.svdvals(Q)
    assert Q.shape == G.shape
    # 五次迭代只是近似正交：条件数显著下降，谱范数接近 1
    assert after[0] / after[-1] < before[0] / before[-1]
    assert 0.5 < after[-1] and after[0] < 1.3


def test_muon_update_reduces_quadratic():
    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.randn(4, 3))
    m = torch.zeros_like(p)
    before = 0.5 * (p ** 2).sum().item()
    m = O.muon_update(p, m, p.grad if p.grad is not None else p.detach().clone(),
                      lr=0.05)
    after = 0.5 * (p ** 2).sum().item()
    assert after < before
    assert m.shape == p.shape
