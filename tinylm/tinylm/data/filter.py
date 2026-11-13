"""第 10 章：去重、质量过滤与污染检测（tinylm/data/filter.py）。"""
import hashlib
import random
import re
from collections import Counter

_MERSENNE = (1 << 61) - 1


def tokenize(text):
    """教学用混合分词：中文按空白分隔的连续汉字串，英文与代码按词。"""
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[一-鿿]+|[^\s]", text)


def exact_dedup(docs):
    """(10.2)：按规范化文本的哈希去重，保留首次出现。"""
    seen, kept, dropped = set(), [], []
    for doc in docs:
        key = hashlib.blake2b(" ".join(tokenize(doc)).encode("utf-8"),
                              digest_size=16).hexdigest()
        if key in seen:
            dropped.append(doc)
        else:
            seen.add(key)
            kept.append(doc)
    return kept, dropped


def shingles(tokens, n=5):
    """把 Token 序列切成连续 n 元组集合。"""
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a, b):
    """(10.3)：J(A,B)=|A∩B|/|A∪B|。"""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _base_hash(gram):
    data = "\x1f".join(map(str, gram)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(gram_set, num_perm=64, seed=0):
    """(10.4)：用随机仿射哈希构造 MinHash 签名。"""
    rng = random.Random(seed)
    coeffs = [(rng.randrange(1, _MERSENNE), rng.randrange(_MERSENNE))
              for _ in range(num_perm)]
    if not gram_set:
        return tuple([_MERSENNE] * num_perm)
    bases = [_base_hash(g) for g in gram_set]
    return tuple(min((a * x + b) % _MERSENNE for x in bases) for a, b in coeffs)


def lsh_candidates(signatures, bands=8, rows=8):
    """(10.5)：把签名切成 bands 段，同段同桶的文档成为候选对。"""
    assert bands * rows <= len(signatures[0])
    buckets = {}
    for idx, sig in enumerate(signatures):
        for b in range(bands):
            key = (b, sig[b * rows:(b + 1) * rows])
            buckets.setdefault(key, []).append(idx)
    pairs = set()
    for ids in buckets.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.add((ids[i], ids[j]))
    return pairs


def near_dedup(docs, threshold=0.7, num_perm=64, bands=8, rows=8, n=5):
    """先 LSH 找候选，再用精确 Jaccard 复核；保留每簇的第一个文档。"""
    gram_sets = [shingles(tokenize(d), n) for d in docs]
    sigs = [minhash_signature(g, num_perm) for g in gram_sets]
    removed = set()
    for i, j in sorted(lsh_candidates(sigs, bands, rows)):
        if i in removed or j in removed:
            continue
        if jaccard(gram_sets[i], gram_sets[j]) >= threshold:
            removed.add(j)
    kept = [d for i, d in enumerate(docs) if i not in removed]
    dropped = [docs[i] for i in sorted(removed)]
    return kept, dropped


def quality_score(text):
    """启发式质量分：长度、去重二元组、字母/汉字比例、词频集中度。"""
    toks = tokenize(text)
    n = len(toks)
    if n == 0:
        return 0.0
    length = min(1.0, n / 40.0)
    uniq_bigram = len(set(zip(toks, toks[1:]))) / max(1, n - 1)
    alpha = sum(bool(re.match(r"[A-Za-z0-9一-鿿]", t)) for t in toks) / n
    top = Counter(toks).most_common(1)[0][1] / n
    score = 0.30 * length + 0.40 * uniq_bigram + 0.15 * alpha + 0.15 * (1.0 - top)
    if n < 12:                      # 太短的文档信息量不足
        score = min(score, 0.30)
    if alpha < 0.5:                 # 符号噪声按比例惩罚
        score *= alpha / 0.5
    return score


def quality_filter(docs, tau=0.55):
    """(10.6)：保留质量分不低于阈值的文档。"""
    scores = [quality_score(d) for d in docs]
    kept = [d for d, s in zip(docs, scores) if s >= tau]
    dropped = [d for d, s in zip(docs, scores) if s < tau]
    return kept, dropped, scores


def ngram_set(tokens, n):
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def contamination_hits(train_docs, eval_docs, n=8):
    """(10.13)：每个评测文档被训练集 n-gram 覆盖的比例。"""
    train_grams = set()
    for d in train_docs:
        train_grams |= ngram_set(tokenize(d), n)
    hits = []
    for d in eval_docs:
        grams = ngram_set(tokenize(d), n)
        hits.append(len(grams & train_grams) / max(1, len(grams)))
    return hits


def decontaminate(train_docs, eval_docs, n=8, threshold=0.5):
    """(10.14)：移除与评测集 n-gram 重叠超阈值的训练文档。"""
    eval_grams = set()
    for d in eval_docs:
        eval_grams |= ngram_set(tokenize(d), n)
    hits = []
    for d in train_docs:
        grams = ngram_set(tokenize(d), n)
        hits.append(len(grams & eval_grams) / max(1, len(grams)))
    kept = [d for d, h in zip(train_docs, hits) if h < threshold]
    dropped = [d for d, h in zip(train_docs, hits) if h >= threshold]
    return kept, dropped
