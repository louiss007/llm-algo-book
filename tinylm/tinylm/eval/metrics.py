"""第 27 章：评测指标与统计工具（tinylm/eval/metrics.py）。

只依赖 Python 标准库。指标函数接收逐样本结果，统计函数用固定种子，
便于在 CPU 上复现实验。
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Hashable, Sequence

def _check_scores(preds: Sequence, labels: Sequence) -> None:
    if len(preds) != len(labels) or len(preds) == 0:
        raise ValueError("preds and labels must be non-empty and aligned")

def accuracy(preds: Sequence, labels: Sequence) -> float:
    """(27.1)：逐样本正确率的算术平均。"""
    _check_scores(preds, labels)
    return sum(1 for p, y in zip(preds, labels) if p == y) / len(labels)

def macro_f1(preds: Sequence, labels: Sequence) -> float:
    """(27.2)：先算每类 F1，再对类别等权平均。"""
    _check_scores(preds, labels)
    classes = sorted(set(labels) | set(preds), key=repr)
    scores: list[float] = []
    for c in classes:
        tp = sum(1 for p, y in zip(preds, labels) if p == c and y == c)
        fp = sum(1 for p, y in zip(preds, labels) if p == c and y != c)
        fn = sum(1 for p, y in zip(preds, labels) if p != c and y == c)
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return sum(scores) / len(scores)

def pass_at_k(n: int, c: int, k: int) -> float:
    """(27.3)：从 n 个样本中抽 k 个，至少一个正确的无偏估计。"""
    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= n")
    if not 0 <= c <= n:
        raise ValueError("need 0 <= c <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def binomial_se(p: float, n: int) -> float:
    """(27.4)：二项比例的标准误。"""
    if n < 1 or not 0.0 <= p <= 1.0:
        raise ValueError("invalid p or n")
    return math.sqrt(p * (1.0 - p) / n)

def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """(27.5)：二项比例的 Wilson 置信区间，小样本比正态近似稳。"""
    if n < 1 or not 0 <= successes <= n or z <= 0:
        raise ValueError("invalid counts or z")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)

def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float],
                          n_boot: int = 4000, level: float = 0.95,
                          seed: int = 0) -> tuple[float, float, float]:
    """(27.6)：逐样本配对差的 bootstrap 均值与百分位区间。"""
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("a and b must be non-empty and aligned")
    if n_boot < 1 or not 0.0 < level < 1.0:
        raise ValueError("invalid n_boot or level")
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    observed = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(len(diffs)):
            total += diffs[rng.randrange(len(diffs))]
        boots.append(total / len(diffs))
    boots.sort()
    lo_index = int((1.0 - level) / 2.0 * n_boot)
    hi_index = min(n_boot - 1, int((1.0 + level) / 2.0 * n_boot) - 1)
    return observed, boots[lo_index], boots[hi_index]

def mcnemar_test(b: int, c: int) -> float:
    """(27.7)：配对二分类的精确 McNemar 双侧 p 值。

    b 是“A 对、B 错”的样本数，c 是“A 错、B 对”的样本数。
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    total = b + c
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, i) for i in range(min(b, c) + 1)) / 2.0 ** total
    return min(1.0, 2.0 * tail)

def expected_calibration_error(confidences: Sequence[float],
                               correct: Sequence[bool | int],
                               n_bins: int = 10) -> float:
    """(27.11)：按置信度分箱，累加 |准确率 - 平均置信度|。"""
    if len(confidences) != len(correct) or len(confidences) == 0:
        raise ValueError("confidences and correct must be non-empty and aligned")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for conf, ok in zip(confidences, correct):
        if not 0.0 <= conf <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        index = min(int(conf * n_bins), n_bins - 1)
        buckets[index].append((conf, float(bool(ok))))
    total = len(confidences)
    return sum(len(bucket) / total * abs(
        sum(ok for _, ok in bucket) / len(bucket) - sum(c for c, _ in bucket) / len(bucket)
    ) for bucket in buckets if bucket)

def ngrams(tokens: Sequence[Hashable], n: int) -> set[tuple]:
    if n < 1:
        raise ValueError("n must be positive")
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

def contamination_rate(eval_docs: Sequence[Sequence[Hashable]],
                       train_docs: Sequence[Sequence[Hashable]],
                       n: int = 13) -> float:
    """(27.10)：评测文档中至少命中一个训练 n-gram 的比例。"""
    if not eval_docs:
        raise ValueError("eval_docs must be non-empty")
    train_ngrams: set[tuple] = set()
    for doc in train_docs:
        train_ngrams |= ngrams(doc, n)
    hit = sum(1 for doc in eval_docs if ngrams(doc, n) & train_ngrams)
    return hit / len(eval_docs)

def bradley_terry_prob(score_i: float, score_j: float) -> float:
    """(27.12)：Elo 尺度下 Bradley–Terry 模型中 i 胜过 j 的概率。"""
    return 1.0 / (1.0 + 10.0 ** (-(score_i - score_j) / 400.0))

def elo_update(rating_w: float, rating_l: float,
               k: float = 32.0) -> tuple[float, float]:
    """(27.13)：按胜负更新双方 Elo 分。"""
    if k <= 0:
        raise ValueError("k must be positive")
    expected_w = 1.0 / (1.0 + 10.0 ** ((rating_l - rating_w) / 400.0))
    delta = k * (1.0 - expected_w)
    return rating_w + delta, rating_l - delta

def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """(27.14)：两名标注者的 Cohen kappa。"""
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("ratings must be non-empty and aligned")
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[key] * cb[key] for key in ca) / (n * n)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)

def cost_aware_utility(score: float, cost: float, lam: float) -> float:
    """(27.15)：把分数与成本折算到同一尺度。"""
    if cost < 0 or lam < 0:
        raise ValueError("cost and lam must be non-negative")
    return score - lam * cost
