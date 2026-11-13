"""第 27 章：指标与统计。"""
import math

import pytest

from tinylm.eval.metrics import (accuracy, binomial_se, bradley_terry_prob,
                                 cohen_kappa, macro_f1, mcnemar_test,
                                 paired_bootstrap_diff, pass_at_k,
                                 wilson_interval)


def test_accuracy_and_macro_f1():
    preds = [1, 1, 0, 1]
    labels = [1, 0, 0, 1]
    assert accuracy(preds, labels) == 0.75
    assert 0.0 <= macro_f1(preds, labels) <= 1.0
    with pytest.raises(ValueError):
        accuracy([], [])


def test_pass_at_k_edges():
    assert pass_at_k(10, 10, 1) == pytest.approx(1.0)
    assert pass_at_k(10, 0, 1) == pytest.approx(0.0)
    assert pass_at_k(4, 1, 4) == pytest.approx(1.0)
    assert pass_at_k(4, 2, 2) == pytest.approx(5 / 6)


def test_binomial_and_wilson():
    se = binomial_se(0.5, 100)
    assert se == pytest.approx(0.05)
    lo, hi = wilson_interval(50, 100)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    assert lo < 0.5 and hi > 0.5


def test_statistical_tools():
    observed, lo, hi = paired_bootstrap_diff([1.0, 2.0, 3.0], [0.5, 1.5, 2.5],
                                             n_boot=200, seed=1)
    assert observed == pytest.approx(0.5)
    assert lo <= observed <= hi
    assert 0.0 <= mcnemar_test(5, 5) <= 1.0
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert bradley_terry_prob(2.0, 0.0) > 0.5
    assert math.isclose(bradley_terry_prob(0.0, 0.0), 0.5)
