"""第 24 章：测试时计算扩展（examples/ch24_search.py）。"""

import math
import time

import torch
import torch.nn.functional as F
from ch08 import ModelConfig, TinyLM

from tinylm.inference.search import (SearchConfig, best_of_n,
                                     coverage_probability, majority_accuracy,
                                     majority_vote, selection_accuracy,
                                     self_consistency, tree_search)

def coverage_demo():
    print("覆盖概率：N 次独立采样至少一次正确")
    print(f"{'p':>6s}" + "".join(f"{'N=' + str(n):>10s}" for n in (1, 4, 16, 64)))
    for p in (0.05, 0.10, 0.20, 0.30):
        print(f"{p:>6.2f}" + "".join(
            f"{coverage_probability(p, n):>10.3f}" for n in (1, 4, 16, 64)))

def voting_demo():
    print("\n多数投票准确率：独立同分布、错误可交换")
    print(f"{'p':>6s}" + "".join(f"{'N=' + str(n):>10s}" for n in (1, 3, 5, 9, 15)))
    for p in (0.35, 0.50, 0.65):
        print(f"{p:>6.2f}" + "".join(
            f"{majority_accuracy(p, n):>10.3f}" for n in (1, 3, 5, 9, 15)))

    gen = torch.Generator().manual_seed(0)
    trials, n, p = 5000, 15, 0.65
    single = diverse = 0
    for _ in range(trials):
        correct = torch.rand(n, generator=gen) < p
        one_wrong = torch.where(correct, torch.zeros(n, dtype=torch.long),
                                torch.ones(n, dtype=torch.long)).tolist()
        many_wrong = torch.where(
            correct, torch.zeros(n, dtype=torch.long),
            torch.randint(1, 10, (n,), generator=gen),
        ).tolist()
        single += int(majority_vote(one_wrong).answer == 0)
        diverse += int(majority_vote(many_wrong).answer == 0)
    print(f"  模拟 p=0.65, N=15：单一错误答案 {single / trials:.3f}"
          f"（理论 {majority_accuracy(p, n):.3f}），"
          f"多个错误答案 {diverse / trials:.3f}")

def verifier_demo():
    p, trials, n = 0.10, 20000, 64
    print(f"\n验证器选择（p={p:.2f}，N={n}，按 (24.6) 计算）")
    print(f"{'两两判别正确率 q':>18s}{'选择成功率':>12s}")
    for q in (1.0, 0.9, 0.7, 0.5):
        print(f"{q:>18.1f}{selection_accuracy(p, n, q):>12.3f}")

    gen = torch.Generator().manual_seed(0)
    correct = torch.rand(trials, n, generator=gen) < p
    scores = correct.float() * 3.0 + torch.randn(trials, n, generator=gen)
    clean = correct[torch.arange(trials), scores.argmax(1)].float().mean().item()
    biased = correct[
        torch.arange(trials),
        (scores + (~correct).float() * 1.5).argmax(1),
    ].float().mean().item()
    print(f"  分数模拟（正确 N(3,1)、错误 N(0,1)，等效 q≈0.98）："
          f"干净 {clean:.3f}")
    print(f"  给错误候选加 1.5 分后：{biased:.3f}（奖励作弊）")

def _trap_tree():
    children = {
        "root": [("A", 0.5), ("B", 0.5)],
        "A": [("A1", 1 / 3), ("A2", 1 / 3), ("A3", 1 / 3)],
        "B": [("B1", 1 / 3), ("B2", 1 / 3), ("B3", 1 / 3)],
        "B3": [("C1", 0.5), ("C2", 0.5)],
    }
    values = {"A1": 0.8, "A2": 0.8, "A3": 0.8, "B1": 0.2, "B2": 0.2,
              "C1": 1.0, "C2": 0.2}
    return (lambda s: children.get(s, [])), (lambda s: values.get(s, 0.0))

def tree_demo():
    expand, value = _trap_tree()
    leaves = ["A1", "A2", "A3", "B1", "B2", "C1", "C2"]
    gen = torch.Generator().manual_seed(0)
    trials, budget = 2000, 8
    best = 0.0
    success = 0
    for _ in range(trials):
        idx = torch.randint(0, len(leaves), (budget,), generator=gen)
        picked = [leaves[i] for i in idx.tolist()]
        vals = [value(s) for s in picked]
        best += max(vals)
        success += int(1.0 in vals)
    print("\n陷阱树：贪心停在局部最优，搜索用预算换取覆盖")
    print(f"{'策略':>12s}{'平均最佳值':>14s}{'找到最优':>12s}")
    print(f"{'贪心':>12s}{0.800:>14.3f}{0.000:>12.3f}")
    print(f"{'随机 8 叶':>12s}{best / trials:>14.3f}{success / trials:>12.3f}")
    for b in (4, 8, 12, 16):
        state, stats = tree_search("root", expand, value, budget=b, c_puct=2.0)
        print(f"{'PUCT 预算 ' + str(b):>12s}{stats['best_value']:>14.3f}"
              f"{float(stats['best_value'] == 1.0):>12.3f}")

def make_chain_data(vocab):
    pairs = torch.cartesian_prod(torch.arange(vocab), torch.arange(vocab))
    a, b = pairs[:, 0], pairs[:, 1]
    hint = (3 * b + 1) % vocab
    answer = (a + hint) % vocab
    sep = torch.full_like(a, vocab)
    eq = torch.full_like(a, vocab + 1)
    prompt = torch.stack([a, b, eq], dim=1)
    train = torch.stack([a, b, eq, hint], dim=1)
    labels = torch.stack([b, eq, hint, answer], dim=1)
    return prompt, train, labels, answer

def train_chain(steps, vocab, seed=0):
    prompt, train, labels, answer = make_chain_data(vocab)
    cfg = ModelConfig(vocab_size=vocab + 2, d_model=64, n_layers=2,
                      n_heads=4, n_kv_heads=2, d_ff=128, max_seq_len=8,
                      norm="rmsnorm", ffn="swiglu", bias=False,
                      pos_encoding="rope", attention="gqa")
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = TinyLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for _ in range(steps):
        j = torch.randint(0, len(prompt), (128,), generator=gen)
        loss = model(train[j], labels=labels[j]).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, prompt, answer

@torch.no_grad()
def chain_accuracy(model, prompt, answer, n_samples, temperature=1.0, seed=1234):
    B = len(prompt)
    gen = torch.Generator().manual_seed(seed)
    hint_logits = model(prompt).logits[:, -1] / temperature
    hints = torch.multinomial(
        F.softmax(hint_logits, dim=-1), n_samples,
        replacement=True, generator=gen,
    ).t()                                   # (n_samples, B)
    expanded = prompt.unsqueeze(0).expand(n_samples, -1, -1).reshape(
        n_samples * B, prompt.size(1))
    full = torch.cat([expanded, hints.reshape(-1, 1)], dim=1)
    ans_logits = model(full).logits[:, -1] / temperature
    answers = torch.multinomial(
        F.softmax(ans_logits, dim=-1), 1, generator=gen,
    ).reshape(n_samples, B)
    results = {}
    for n in (1, 4, 8, 16):
        ok = sum(
            int(majority_vote(answers[:n, q].tolist()).answer == answer[q].item())
            for q in range(B)
        )
        results[n] = ok / B
    return results

def tiny_lm_demo():
    vocab = 29
    print("\nTinyLM 链式算术：训练更久，还是采样更多？")
    print(f"{'训练步数':>10s}{'贪心':>10s}" + "".join(
        f"{'投票@' + str(n):>12s}" for n in (1, 4, 8, 16)))
    times = {}
    for steps in (600, 1000):
        start = time.time()
        model, prompt, answer = train_chain(steps, vocab)
        results = chain_accuracy(model, prompt, answer, 16)
        with torch.no_grad():
            hint = model(prompt).logits[:, -1].argmax(-1)
            full = torch.cat([prompt, hint[:, None]], dim=1)
            greedy = (model(full).logits[:, -1].argmax(-1) == answer).float().mean()
        print(f"{steps:>10d}{greedy.item():>10.3f}" + "".join(
            f"{results[n]:>12.3f}" for n in (1, 4, 8, 16)))
        times[steps] = time.time() - start
    print(f"  训练耗时：600 步 {times[600]:.1f} s，1000 步 {times[1000]:.1f} s；"
          f"每样本 5 个 Token")

def stopping_demo():
    p, trials, max_samples = 0.65, 1000, 32
    print(f"\n提前停止（p={p:.2f}，最多 {max_samples} 样本，{trials} 题）")
    print(f"{'停止阈值':>10s}{'平均样本':>12s}{'准确率':>10s}")
    for margin in (0.0, 2.0, 3.0, 5.0):
        gen = torch.Generator().manual_seed(0)
        cfg = SearchConfig(max_samples=max_samples, min_samples=4,
                           stop_margin=margin)
        used = 0
        hits = 0
        for _ in range(trials):
            def sample(i, gen=gen):
                hit = torch.rand((), generator=gen).item() < p
                return (0 if hit else 1), 1.0
            result = self_consistency(sample, cfg)
            used += result.n_samples
            hits += int(result.answer == 0)
        label = "不停止" if margin == 0.0 else f"δ={margin:.0f}"
        print(f"{label:>10s}{used / trials:>12.1f}{hits / trials:>10.3f}")

if __name__ == "__main__":
    coverage_demo()
    voting_demo()
    verifier_demo()
    tree_demo()
    tiny_lm_demo()
    stopping_demo()
