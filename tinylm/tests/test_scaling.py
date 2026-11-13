"""第 9 章自测：幂律拟合、计算最优分配、数据重复与算力账本。"""
import math

import pytest
import torch

import ch09


TRUE = dict(E=1.7, A=400.0, B=400.0, alpha=0.34, beta=0.28)


def test_power_law_and_flops():
    """(9.1) 与 (9.2)：手算两点核对。"""
    L = ch09.power_law_loss(1e9, 1e10, 1.7, 400.0, 400.0, 0.34, 0.28)
    expected = 1.7 + 400.0 / (1e9 ** 0.34) + 400.0 / (1e10 ** 0.28)
    assert abs(L - expected) < 1e-12
    assert ch09.training_flops(1e9, 1e10) == 6e19


def test_optimal_exponents_sum_to_one():
    """(9.5)：a+b=1，且 a<b（α>β）。"""
    a, b = ch09.optimal_exponents(0.34, 0.28)
    assert abs(a + b - 1.0) < 1e-12
    assert abs(a - 0.28 / 0.62) < 1e-12
    assert a < b


def test_allocation_satisfies_compute_budget():
    """(9.6)：N_opt 与 D_opt 必须严格满足 6ND=C。"""
    for e in (20, 22, 24):
        C = 10.0 ** e
        N, D = ch09.scaling_allocation(C)
        assert abs(6.0 * N * D / C - 1.0) < 1e-12
    N1, D1 = ch09.scaling_allocation(1e22)
    N2, D2 = ch09.scaling_allocation(1e24)
    assert N2 > N1 and D2 > D1
    assert D2 / N2 > D1 / N1          # b>a，Token/参数比缓慢上升


def test_compute_optimal_satisfies_balance_condition():
    """(9.3)–(9.4)：直接数值最小化应满足 αA N^-α=βB D^-β。"""
    C = 1e22
    N, D, _ = ch09.compute_optimal(C, **TRUE)
    assert abs(6.0 * N * D / C - 1.0) < 1e-9
    left = TRUE["alpha"] * TRUE["A"] * N ** (-TRUE["alpha"])
    right = TRUE["beta"] * TRUE["B"] * D ** (-TRUE["beta"])
    assert abs(left / right - 1.0) < 1e-3


def test_fit_recovers_exponents_from_synthetic_data():
    """固定随机种子，35 个带 1% 噪声的点应恢复 α、β 与 E。"""
    pts = ch09.synthetic_points(TRUE, seed=0)
    got = ch09.fit_scaling_law(pts, seed=0)
    assert abs(got["E"] - TRUE["E"]) / TRUE["E"] < 0.05
    assert abs(got["alpha"] - TRUE["alpha"]) < 0.05
    assert abs(got["beta"] - TRUE["beta"]) < 0.05
    assert got["rmse_log"] < 0.02


def test_effective_data_saturates():
    """(9.8)：首轮全值，重复收益递减，上限为 U(1+R*)。"""
    U = 1e12
    assert ch09.effective_data(U, U) == U
    assert ch09.effective_data(0.5 * U, U) == 0.5 * U
    # 第二轮几乎等于全新数据：R=1 时 D_eff/U 已到 1.95 以上
    assert 1.95 * U < ch09.effective_data(2 * U, U) < 2 * U
    # 无限重复也只能拿到 U(1+R*) 的有效数据
    assert ch09.effective_data(1e6 * U, U) < 1.001 * (1 + ch09.R_STAR) * U
    assert ch09.effective_data(2 * U, U) < ch09.effective_data(8 * U, U)


@pytest.mark.parametrize("C", [1e19, 1e21, 1e23])
def test_compute_optimal_loss_decreases_with_budget(C):
    N1, D1, L1 = ch09.compute_optimal(C, **TRUE)
    N2, D2, L2 = ch09.compute_optimal(10 * C, **TRUE)
    assert N2 > N1 and D2 > D1
    assert L2 < L1
