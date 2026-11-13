"""第 10 章：数据配比与 Token 预算（tinylm/data/mixture.py）。"""
import math
import random


def temperature_probs(weights, temperature=1.0):
    """(10.9)：p_i ∝ w_i^{1/T}，T=1 为按来源比例，T>1 更均匀。"""
    if temperature <= 0:
        raise ValueError("temperature 必须为正")
    logits = [math.log(w) / temperature for w in weights]
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def allocate_counts(total, probs):
    """(10.10)：最大余数法分配预算，保证各域计数之和等于 total。"""
    raw = [total * p for p in probs]
    counts = [int(math.floor(x)) for x in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:remainder]:
        counts[i] += 1
    return counts


def interleave(domain_docs, counts, seed=0):
    """按配额轮流取文档；域内先打乱，配额超过文档数时循环复用。"""
    rng = random.Random(seed)
    pools = {}
    for name, docs in domain_docs.items():
        pool = list(docs)
        rng.shuffle(pool)
        pools[name] = pool
    remaining = dict(zip(domain_docs, counts))
    cursor = {name: 0 for name in pools}
    out = []
    while sum(remaining.values()) > 0:
        for name in domain_docs:
            if remaining[name] > 0 and pools[name]:
                out.append(pools[name][cursor[name] % len(pools[name])])
                cursor[name] += 1
                remaining[name] -= 1
    return out
