"""第 24 章：测试时计算扩展（tinylm/inference/search.py）。"""
import math
from collections import Counter
from dataclasses import dataclass, field

@dataclass(frozen=True)
class SearchConfig:
    """测试时搜索参数，不进入 ``ModelConfig``。"""

    max_samples: int = 16
    max_tokens: int = 512
    temperature: float = 1.0
    vote: str = "majority"          # majority | weighted
    min_samples: int = 2
    stop_margin: float = 0.0        # 0 表示不提前停止
    c_puct: float = 1.0

    def __post_init__(self):
        assert self.max_samples >= 1
        assert self.max_tokens >= 1
        assert self.temperature > 0.0
        assert self.vote in ("majority", "weighted")
        assert self.min_samples >= 2
        assert self.stop_margin >= 0.0
        assert self.c_puct > 0.0

@dataclass
class SearchResult:
    answer: object
    counts: dict
    scores: dict
    n_samples: int
    stopped_early: bool = False

    @property
    def margin(self):
        """最高分与次高分之差；只有一个候选时返回其票数。"""
        if len(self.scores) == 1:
            return float(next(iter(self.scores.values())))
        ordered = sorted(self.scores.values(), reverse=True)
        return float(ordered[0] - ordered[1])

def coverage_probability(p, n):
    """(24.1)：至少一次采样正确的概率。"""
    assert 0.0 <= p <= 1.0 and n >= 1
    return 1.0 - (1.0 - p) ** n

def pass_at_k(n, c, k):
    """(24.3)：从 n 条样本中取 k 条、至少一条正确的无偏估计。"""
    assert n >= 1 and 0 <= c <= n and k >= 1
    if k > n:
        raise ValueError("k 不能大于 n")
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod

def majority_accuracy(p, n):
    """(24.4)：独立同分布、n 为奇数时的多数投票准确率。"""
    assert 0.0 <= p <= 1.0 and n >= 1 and n % 2 == 1
    return sum(
        math.comb(n, i) * p ** i * (1.0 - p) ** (n - i)
        for i in range(n // 2 + 1, n + 1)
    )

def majority_vote(answers, weights=None):
    """(24.5)：按出现次数或权重求和投票；平分时取最早出现的答案。"""
    assert len(answers) > 0
    if weights is not None:
        assert len(weights) == len(answers)
    order = {}
    counts = Counter(answers)
    scores = {}
    for i, ans in enumerate(answers):
        order.setdefault(ans, i)
        scores[ans] = scores.get(ans, 0.0) + (
            1.0 if weights is None else float(weights[i])
        )
    best = max(scores, key=lambda a: (scores[a], -order[a]))
    return SearchResult(answer=best, counts=dict(counts), scores=scores,
                        n_samples=len(answers))

def best_of_n(candidates, scores):
    """选择验证器分数最高的候选；平分时取第一个。"""
    assert len(candidates) == len(scores) and len(candidates) > 0
    idx = max(range(len(candidates)), key=lambda i: (scores[i], -i))
    return candidates[idx], idx

def selection_accuracy(p, n, q):
    """(24.6)：验证器两两判别正确率为 q 时的选择成功率（成对判别近似）。"""
    assert 0.0 <= p <= 1.0 and n >= 1 and 0.0 <= q <= 1.0
    total = 0.0
    for k in range(1, n + 1):
        if k * q + (n - k) * (1.0 - q) == 0.0:
            continue
        mass = math.comb(n, k) * p ** k * (1.0 - p) ** (n - k)
        total += mass * (k * q) / (k * q + (n - k) * (1.0 - q))
    return total

def should_stop(result, cfg):
    """(24.11)：领先优势足够大且样本数达到下限时提前停止。"""
    if cfg.stop_margin <= 0.0 or result.n_samples < cfg.min_samples:
        return False
    return result.margin >= cfg.stop_margin

def self_consistency(sample_fn, cfg):
    """重复采样并投票，直到样本预算用尽或满足提前停止条件。"""
    answers, weights = [], []
    for i in range(cfg.max_samples):
        ans, weight = sample_fn(i)
        answers.append(ans)
        weights.append(weight)
        result = majority_vote(
            answers, weights if cfg.vote == "weighted" else None
        )
        if should_stop(result, cfg):
            result.stopped_early = True
            return result
    return result

def token_cost(cfg, prompt_tokens=0):
    """(24.9)：按样本数记账的总 Token 预算。"""
    assert prompt_tokens >= 0
    return cfg.max_samples * (prompt_tokens + cfg.max_tokens)

def puct_score(q, prior, total_visits, visits, c_puct):
    """(24.7)：PUCT 的选择分数。"""
    assert 0.0 <= prior <= 1.0 and total_visits >= 0 and visits >= 0
    explore = c_puct * prior * math.sqrt(total_visits + 1) / (1 + visits)
    return q + explore

@dataclass
class TreeNode:
    state: object
    prior: float = 1.0
    parent: "TreeNode | None" = None
    children: list = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False

    @property
    def q(self):
        """(24.8)：节点价值的访问平均。"""
        return self.value_sum / self.visits if self.visits else 0.0

def tree_search(root_state, expand_fn, value_fn, budget, c_puct=1.0):
    """PUCT 搜索：用价值函数代替随机回放，返回最佳状态与统计。

    ``expand_fn(state)`` 返回 ``[(child_state, prior), ...]``，叶子返回空表；
    ``value_fn(state)`` 返回 ``[0, 1]`` 内的价值。预算按价值函数调用次数计。
    """
    assert budget >= 0 and c_puct > 0.0
    root = TreeNode(root_state)
    evaluations = 0
    best_state, best_value = root_state, None
    while evaluations < budget:
        node = root
        path = [node]
        while True:
            if not node.expanded:
                node.expanded = True
                for state, prior in expand_fn(node.state):
                    node.children.append(TreeNode(state, prior, node))
                if not node.children:
                    break
                node = max(node.children, key=lambda c: c.prior)
                path.append(node)
                continue
            if not node.children:
                break
            total = sum(c.visits for c in node.children)
            node = max(
                node.children,
                key=lambda c: puct_score(c.q, c.prior, total, c.visits, c_puct),
            )
            path.append(node)
        value = value_fn(node.state)
        evaluations += 1
        if best_value is None or value > best_value:
            best_state, best_value = node.state, value
        for item in path:
            item.visits += 1
            item.value_sum += value
    stats = {
        "evaluations": evaluations,
        "root_visits": root.visits,
        "best_value": best_value if best_value is not None
        else value_fn(root_state),
    }
    return best_state, stats
