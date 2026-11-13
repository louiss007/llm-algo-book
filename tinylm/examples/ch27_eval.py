"""第 27 章：评测方法论（examples/ch27_eval.py）。"""
import random
import sys

sys.path.insert(0, ".")
from tinylm.eval.metrics import (accuracy, binomial_se, bradley_terry_prob, cohen_kappa,
                                 contamination_rate, cost_aware_utility,
                                 elo_update, expected_calibration_error,
                                 macro_f1, mcnemar_test, paired_bootstrap_diff,
                                 pass_at_k, wilson_interval)

def comparison_demo():
    n = 1000
    both_ok, a_only, b_only = 680, 20, 40
    both_bad = n - both_ok - a_only - b_only
    a = [1] * both_ok + [1] * a_only + [0] * b_only + [0] * both_bad
    b = [1] * both_ok + [0] * a_only + [1] * b_only + [0] * both_bad
    acc_a, acc_b = accuracy(a, [1] * n), accuracy(b, [1] * n)
    diff, lo, hi = paired_bootstrap_diff(a, b, n_boot=4000, seed=0)
    p_value = mcnemar_test(a_only, b_only)
    lo_a, hi_a = wilson_interval(sum(a), n)
    lo_b, hi_b = wilson_interval(sum(b), n)
    print("两个模型的配对比较（n=1000）")
    print(f"  A 准确率 {acc_a:.3f}，标准误 {binomial_se(acc_a, n):.4f}，"
          f"Wilson 95% 区间 [{lo_a:.3f}, {hi_a:.3f}]")
    print(f"  B 准确率 {acc_b:.3f}，Wilson 95% 区间 [{lo_b:.3f}, {hi_b:.3f}]")
    print(f"  配对差 B-A {diff:+.3f}，bootstrap 95% 区间 [{lo:+.3f}, {hi:+.3f}]")
    print(f"  不一致对 b={a_only}, c={b_only}，McNemar 精确 p={p_value:.4f}")

def passk_demo():
    n, c = 10, 2
    rng = random.Random(0)
    correct = set(range(c))
    print(f"pass@k 的公式与抽样核对（n={n}, c={c}, 20000 次）")
    print(f"{'k':>2} {'公式':>8} {'模拟':>8}")
    for k in range(1, 6):
        hits = sum(1 for _ in range(20000)
                   if correct & set(rng.sample(range(n), k)))
        print(f"{k:>2} {pass_at_k(n, c, k):>8.4f} {hits / 20000:>8.4f}")

def contamination_demo():
    train = "the quick brown fox jumps over the lazy dog and then it sleeps all day".split()
    copied = list(train)
    paraphrase = train[:6] + "while the crowd watches quietly".split()
    novel = "a completely different sentence about evaluation protocol".split()
    docs = [copied, paraphrase, novel]
    print("训练—评测 n-gram 重叠（3 条评测文档）")
    for n in (5, 13):
        print(f"  n={n:>2}: 命中比例 {contamination_rate(docs, [train], n):.3f}")

def judge_demo():
    rng = random.Random(1)
    trials = 2000
    a_first = sum(1 for _ in range(trials) if rng.random() < 0.6)
    swapped = sum(1 for _ in range(trials) if rng.random() < 0.6)
    print("裁判的位置偏差（两模型真实质量相同）")
    print(f"  只按 A 在前：A 胜率 {a_first / trials:.3f}")
    print(f"  交换顺序后：A 胜率 {(a_first + (trials - swapped)) / (2 * trials):.3f}")
    print("Bradley–Terry 胜率与 Elo 更新")
    for gap in (0, 100, 200):
        print(f"  分差 {gap:>3}: P(i 胜 j)={bradley_terry_prob(gap, 0):.3f}")
    w, l = elo_update(1500, 1700)
    print(f"  1500 战胜 1700 后：{w:.1f} / {l:.1f}")

def calibration_demo():
    confidences, correct = [], []
    for bucket in range(10):
        confidences += [(bucket + 0.5) / 10] * 10
        correct += [1] * bucket + [0] * (10 - bucket)
    ece = expected_calibration_error(confidences, correct, n_bins=10)
    print(f"校准：ECE = {ece:.3f}（每箱 10 个样本，准确率比置信度低 0.05）")

def agreement_demo():
    a = [1] * 50 + [0] * 50
    b = [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40
    print(f"标注一致性：一致率 0.800，Cohen kappa = {cohen_kappa(a, b):.3f}")
    preds, labels = [0, 0, 1, 2], [0, 1, 1, 2]
    print(f"macro-F1（3 类，两类各错一个）= {macro_f1(preds, labels):.3f}"
          f"（准确率 {accuracy(preds, labels):.3f}）")

def utility_demo():
    u_a = cost_aware_utility(0.72, 1.0, 0.02)
    u_b = cost_aware_utility(0.74, 3.0, 0.02)
    print("成本感知效用（lambda=0.02，成本按相对单位）")
    print(f"  A: 0.72 - 0.02*1 = {u_a:.2f}")
    print(f"  B: 0.74 - 0.02*3 = {u_b:.2f}")

def main():
    comparison_demo()
    print()
    passk_demo()
    print()
    contamination_demo()
    print()
    judge_demo()
    print()
    calibration_demo()
    print()
    agreement_demo()
    print()
    utility_demo()

if __name__ == "__main__":
    main()
