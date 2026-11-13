"""第 10 章自测：配比采样、去重、质量过滤与污染检测。"""
import pytest

from tinylm.data import filter as F
from tinylm.data import mixture as M


GOOD_A = "山脉 河流 高原 气候 植被 森林 草原 湖泊 海岸 岛屿 峡谷 冰川 湿地 土壤 矿物 物种 迁徙 聚落 灌溉 交通 贸易 城市 人口 语言 建筑 遗址 平原 盆地 丘陵 瀑布 温泉 沙漠 绿洲 冻土 溶洞 珊瑚 红树 湿地 苔原 沼泽 溪流 三角洲 峡湾"
GOOD_B = "整数 实数 素数 因数 倍数 方程 函数 极限 导数 积分 矩阵 向量 概率 期望 方差 集合 映射 群 环 域 证明 定理 引理 归纳 反证 充分 必要 存在 任意 唯一 连续 可导 有界 收敛 发散 级数 多项式 特征值 行列式 秩 迹 对偶 同构"
NEAR_A = GOOD_A.replace("高原", "盆地")


def test_tokenize_splits_words_and_punctuation():
    toks = F.tokenize("def f(x): return x，山脉")
    assert "def" in toks and "f" in toks and "(" in toks
    assert "山脉" in toks and "，" in toks


def test_exact_dedup_keeps_first_occurrence():
    kept, dropped = F.exact_dedup([GOOD_A, GOOD_B, GOOD_A, GOOD_B])
    assert kept == [GOOD_A, GOOD_B]
    assert dropped == [GOOD_A, GOOD_B]


def test_jaccard_edges():
    assert F.jaccard(set(), set()) == 1.0
    assert F.jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert F.jaccard({1, 2}, {1, 2}) == 1.0


def test_minhash_estimate_tracks_jaccard():
    a = F.shingles(F.tokenize(GOOD_A), 3)
    b = F.shingles(F.tokenize(NEAR_A), 3)
    sa = F.minhash_signature(a, num_perm=256)
    sb = F.minhash_signature(b, num_perm=256)
    estimate = sum(x == y for x, y in zip(sa, sb)) / 256
    assert abs(estimate - F.jaccard(a, b)) < 0.12


def test_near_dedup_catches_small_edit():
    docs = [GOOD_A, NEAR_A, GOOD_B]
    kept, dropped = F.near_dedup(docs, threshold=0.7, num_perm=128)
    assert len(kept) == 2 and len(dropped) == 1
    assert GOOD_B in kept


def test_lsh_puts_identical_signatures_in_one_bucket():
    sig = F.minhash_signature(F.shingles(F.tokenize(GOOD_A), 3), 64)
    pairs = F.lsh_candidates([sig, sig], bands=8, rows=8)
    assert (0, 1) in pairs


def test_quality_filter_rejects_noise():
    repetitive = " ".join(["山脉"] * 80)
    short = " ".join(["山脉", "河流", "高原", "气候"])
    symbols = "".join(["#"] * 80)
    kept, dropped, scores = F.quality_filter([GOOD_A, repetitive, short, symbols])
    assert GOOD_A in kept
    assert repetitive in dropped and short in dropped and symbols in dropped
    assert F.quality_score(GOOD_A) > 0.7


def test_contamination_detects_leak_not_clean_doc():
    train = [GOOD_A, GOOD_B]
    eval_docs = [GOOD_A, "用户 请 检查 这个 步骤 首先 打开 文件"]
    hits = F.contamination_hits(train, eval_docs, n=4)
    assert hits[0] == 1.0
    assert hits[1] < 0.5


def test_decontaminate_removes_leak():
    train = [GOOD_A, GOOD_B]
    kept, dropped = F.decontaminate(train, [GOOD_A], n=4, threshold=0.5)
    assert kept == [GOOD_B]
    assert dropped == [GOOD_A]


def test_temperature_probs_flat_and_sharp():
    weights = [80, 40, 25, 15]
    base = M.temperature_probs(weights, 1.0)
    flat = M.temperature_probs(weights, 2.0)
    sharp = M.temperature_probs(weights, 0.5)
    assert sum(base) == pytest.approx(1.0)
    assert max(flat) - min(flat) < max(base) - min(base)
    assert max(sharp) - min(sharp) > max(base) - min(base)


def test_allocate_counts_sums_to_budget():
    probs = M.temperature_probs([80, 40, 25, 15], 2.0)
    counts = M.allocate_counts(1000, probs)
    assert sum(counts) == 1000
    assert all(c >= 0 for c in counts)


def test_interleave_respects_quotas_and_recycles():
    domains = {"a": ["a1", "a2"], "b": ["b1"]}
    out = M.interleave(domains, [3, 2], seed=0)
    assert len(out) == 5
    assert out.count("b1") == 2 and sum(x.startswith("a") for x in out) == 3


def test_invalid_temperature_raises():
    with pytest.raises(ValueError):
        M.temperature_probs([1, 2], 0.0)
