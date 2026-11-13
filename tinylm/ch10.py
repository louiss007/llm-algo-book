"""第 10 章：预训练数据工程实验（examples/ch10_data.py）。"""
import random
import re
from collections import Counter

import torch

from ch08 import ModelConfig, TinyLM
from tinylm.data.filter import (contamination_hits, decontaminate, exact_dedup,
                                near_dedup, quality_filter, tokenize)
from tinylm.data.mixture import allocate_counts, interleave, temperature_probs

DOMAINS = ("百科", "代码", "数学", "对话")
BANK = {
    "百科": "山脉 河流 高原 平原 盆地 气候 降水 植被 森林 草原 湖泊 海岸 岛屿 峡谷 冰川 湿地 土壤 矿物 物种 迁徙 聚落 灌溉 交通 贸易 城市 人口 语言 建筑 遗址".split(),
    "代码": "def return for while if else import from class self None True False print len range list dict set tuple append sort key value total result data model token batch".split(),
    "数学": "整数 实数 有理数 素数 因数 倍数 方程 函数 极限 导数 积分 矩阵 向量 概率 期望 方差 集合 映射 群 环 域 证明 定理 引理 归纳 反证 充分 必要 存在 任意".split(),
    "对话": "用户 助手 问题 回答 请 帮 我 你 这个 可以 需要 建议 步骤 首先 然后 最后 注意 例如 如果 因为 所以 好的 谢谢 明白 继续 试试 检查".split(),
}


def make_doc(domain, rng):
    """按域随机拼句：固定短语很短，避免不同文档共享长 n-gram。"""
    words = BANK[domain]
    sentences = []
    for _ in range(rng.randint(3, 5)):
        length = rng.randint(6, 12)
        parts = [rng.choice(words) for _ in range(length)]
        if domain in ("百科", "数学"):
            sentences.append(" ".join(parts) + "。")
        elif domain == "代码":
            sentences.append(" ".join(parts) + " ;")
        else:
            sentences.append(rng.choice(["用户：", "助手："]) + " ".join(parts) + "。")
    return "".join(sentences)


def make_good(domain_counts, rng):
    return [(d, make_doc(d, rng)) for d, n in domain_counts.items() for _ in range(n)]


def near_copy(text, rng):
    chars = list(text)
    idxs = [i for i, c in enumerate(chars) if re.match(r"[A-Za-z0-9一-鿿]", c)]
    i = rng.choice(idxs)
    pool = "甲乙丙丁戊己庚辛壬癸山脉河海云风力" if re.match(r"[一-鿿]", chars[i]) \
        else "abcdefghijklmnopqrstuvwxyz0123456789"
    chars[i] = rng.choice(pool)
    return "".join(chars)


def build_corpus(seed=0):
    """248 篇：160 正常 + 60 重复 + 20 低质 + 8 条评测泄漏。"""
    rng = random.Random(seed)
    good = make_good({"百科": 80, "代码": 40, "数学": 25, "对话": 15}, rng)
    eval_rng = random.Random(2024)
    eval_docs = [(d, make_doc(d, eval_rng)) for d in DOMAINS for _ in range(5)]
    leak = []
    for i, (d, text) in enumerate(eval_docs[:8]):
        leak.append((d, text if i < 6 else near_copy(text, rng)))
    exact = [(d, text) for d, text in rng.sample(good, 30)]
    near = [(d, near_copy(text, rng)) for d, text in rng.sample(good, 30)]
    noise = []
    for i in range(20):
        d = DOMAINS[i % 4]
        if i % 3 == 0:
            noise.append((d, " ".join([rng.choice(BANK[d])] * 80)))
        elif i % 3 == 1:
            noise.append((d, " ".join(BANK[d][:5])))
        else:
            noise.append((d, "".join(rng.choice("###$%^&*(){}[]<>~`|") for _ in range(80))))
    return good + exact + near + noise + leak, eval_docs


def build_clean_val(seed=777):
    rng = random.Random(seed)
    return {d: [make_doc(d, rng) for _ in range(10)] for d in DOMAINS}


def build_vocab(texts):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {"<eos>": 0, "<unk>": 1}
    for tok, _ in counter.most_common():
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def encode_stream(docs, vocab):
    ids = []
    for doc in docs:
        ids.extend(vocab.get(t, 1) for t in tokenize(doc))
        ids.append(0)
    return torch.tensor(ids, dtype=torch.long)


def train_model(docs, vocab, steps=150, seed=0, batch=16, seq=32, lr=2e-3):
    cfg = ModelConfig(vocab_size=len(vocab), d_model=32, n_layers=2, n_heads=4,
                      d_ff=64, max_seq_len=64, norm="rmsnorm", ffn="swiglu",
                      bias=False, pos_encoding="rope", attention="gqa",
                      n_kv_heads=2)
    torch.manual_seed(seed)
    model = TinyLM(cfg)
    stream = encode_stream(docs, vocab)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for _ in range(steps):
        starts = torch.randint(0, len(stream) - seq - 1, (batch,), generator=gen)
        x = torch.stack([stream[s:s + seq] for s in starts])
        y = torch.stack([stream[s + 1:s + seq + 1] for s in starts])
        out = model(x, labels=y)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    return model


@torch.no_grad()
def evaluate(model, docs, vocab, seq=32):
    model.eval()
    stream = encode_stream(docs, vocab)
    n = (len(stream) - 1) // seq
    losses = []
    for k in range(n):
        x = stream[k * seq:(k + 1) * seq].unsqueeze(0)
        y = stream[k * seq + 1:(k + 1) * seq + 1].unsqueeze(0)
        losses.append(model(x, labels=y).loss.item())
    return sum(losses) / len(losses)


def pipeline(raw, eval_texts):
    q_kept, q_drop, _ = quality_filter(raw)
    e_kept, e_drop = exact_dedup(q_kept)
    n_kept, n_drop = near_dedup(e_kept)
    d_kept, d_drop = decontaminate(n_kept, eval_texts)
    return {"raw": raw, "quality": q_kept, "exact+near": n_kept,
            "decontam": d_kept}, [len(q_drop), len(e_drop), len(n_drop), len(d_drop)]


def funnel_table(raw, eval_texts):
    stages, _ = pipeline(raw, eval_texts)
    print("\n数据管道漏斗（按文档数与 Token 数）")
    print(f"{'阶段':<12s}{'文档数':>8s}{'Token 数':>10s}{'本阶段删除':>10s}")
    prev = len(raw)
    for name, docs in stages.items():
        tokens = sum(len(tokenize(d)) for d in docs)
        removed = prev - len(docs)
        print(f"{name:<12s}{len(docs):>8d}{tokens:>10d}{removed:>10d}")
        prev = len(docs)
    return stages


def contamination_demo(raw, eval_texts):
    hits = contamination_hits(raw, eval_texts)
    leaked = sum(h >= 0.5 for h in hits)
    clean_val = [t for docs in build_clean_val().values() for t in docs]
    clean_hits = contamination_hits(raw, clean_val)
    print(f"\n污染检测（8-gram，覆盖率阈值 0.5）")
    print(f"评测集 {len(eval_texts)} 篇中命中 {leaked} 篇；"
          f"未泄漏的干净文档 {len(clean_val)} 篇中命中 "
          f"{sum(h >= 0.5 for h in clean_hits)} 篇")
    print(f"命中评测文档的覆盖率：{sorted(round(h, 2) for h in hits)[-8:]}")


def mixture_table(clean_docs):
    by_domain = {d: [t for dd, t in clean_docs if dd == d] for d in DOMAINS}
    sizes = [max(1, len(by_domain[d])) for d in DOMAINS]
    print("\n数据配比：按来源比例与温度采样（总预算 400 篇）")
    print(f"{'策略':<14s}" + "".join(f"{d:>8s}" for d in DOMAINS) + f"{'最大/最小':>10s}")
    for name, probs in (("按来源比例", temperature_probs(sizes, 1.0)),
                        ("温度 T=2", temperature_probs(sizes, 2.0)),
                        ("等权", [0.25] * 4)):
        counts = allocate_counts(400, probs)
        print(f"{name:<14s}" + "".join(f"{c:>8d}" for c in counts)
              + f"{max(counts) / min(counts):>10.1f}")
    return by_domain, sizes


def mixture_ablation(by_domain, sizes, vocab, clean_val):
    source = {d: by_domain[d] for d in DOMAINS}
    equal = {d: 0.25 for d in DOMAINS}
    source_probs = temperature_probs(sizes, 1.0)
    equal_probs = [equal[d] for d in DOMAINS]
    print("\n配比消融：按来源比例 vs 等权（2 个种子、120 步）")
    print(f"{'策略':<12s}{'百科':>10s}{'代码':>10s}{'数学':>10s}{'对话':>10s}")
    for name, probs in (("按来源比例", source_probs), ("等权", equal_probs)):
        counts = allocate_counts(400, probs)
        docs = interleave(source, counts, seed=0)
        rows = []
        for seed in (0, 1):
            model = train_model(docs, vocab, steps=120, seed=seed)
            rows.append([evaluate(model, clean_val[d], vocab) for d in DOMAINS])
        means = [sum(r[i] for r in rows) / len(rows) for i in range(4)]
        print(f"{name:<12s}" + "".join(f"{m:>10.4f}" for m in means))


def main():
    raw_docs, eval_docs = build_corpus()
    raw = [t for _, t in raw_docs]
    eval_texts = [t for _, t in eval_docs]
    clean_val = build_clean_val()

    print("合成语料：4 个域，含正常文档、重复、低质噪声与评测泄漏")
    print(f"原始文档 {len(raw)} 篇，评测集 {len(eval_texts)} 篇，干净验证集 "
          f"{sum(len(v) for v in clean_val.values())} 篇")
    stages = funnel_table(raw, eval_texts)
    contamination_demo(raw, eval_texts)

    clean = stages["decontam"]
    text_to_domain = {t: d for d, t in raw_docs}
    by_domain, sizes = mixture_table([(text_to_domain[t], t) for t in clean])

    vocab = build_vocab(raw + eval_texts + [t for v in clean_val.values() for t in v])
    print(f"\n词表大小 {len(vocab)}；训练 150 步、批 16、序列 32，3 个种子取均值")
    corpora = {
        "原始": raw,
        "仅质量过滤": stages["quality"],
        "仅去重": stages["exact+near"],
        "质量+去重+防污染": clean,
    }
    print(f"{'语料':<18s}{'去重后 Token':>11s}{'干净验证':>14s}{'泄漏评测':>14s}")
    for name, docs in corpora.items():
        unique_docs = near_dedup(exact_dedup(docs)[0])[0]
        unique = sum(len(tokenize(d)) for d in unique_docs)
        clean_loss, leak_loss = [], []
        for seed in (0, 1, 2):
            model = train_model(docs, vocab, steps=150, seed=seed)
            clean_loss.append(evaluate(model, [t for v in clean_val.values() for t in v], vocab))
            leak_loss.append(evaluate(model, eval_texts, vocab))
        c = torch.tensor(clean_loss)
        l = torch.tensor(leak_loss)
        print(f"{name:<18s}{unique:>11d}"
              f"{c.mean():>9.4f}±{c.std():.3f}{l.mean():>9.4f}±{l.std():.3f}")

    mixture_ablation(by_domain, sizes, vocab, clean_val)

if __name__ == "__main__":
    main()
