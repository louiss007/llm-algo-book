"""第 22 章：投机解码与多 Token 预测（examples/ch22_speculative.py）。"""

import torch

from ch08 import ModelConfig, TinyLM
from tinylm.inference.kv_cache import KVCache, cached_forward, cached_prefill
from tinylm.inference.speculative import (MTPHead, SpecConfig,
                                          acceptance_rate,
                                          expected_output_tokens, mtp_loss,
                                          rejection_sample, speculative_sample,
                                          speculative_speedup,
                                          tree_attention_mask, truncate_cache)

def lossless_check():
    p = torch.tensor([0.05, 0.10, 0.15, 0.20, 0.50])
    q = torch.tensor([0.40, 0.25, 0.15, 0.10, 0.10])
    n = 200_000
    gen = torch.Generator().manual_seed(0)
    tokens, accepted = rejection_sample(
        q.expand(n, -1), p.expand(n, -1), gen)
    empirical = torch.bincount(tokens, minlength=5).float() / n
    print("单步拒绝采样的分布（20 万次）")
    print(f"{'Token':>6s}{'目标 p':>10s}{'实测':>10s}{'误差':>10s}")
    for i in range(5):
        print(f"{i:>6d}{p[i]:>10.3f}{empirical[i]:>10.3f}"
              f"{(empirical[i] - p[i]).abs():>10.3f}")
    print(f"  实测接受率       {accepted.float().mean():.4f}")
    print(f"  理论 α=Σmin(p,q) {acceptance_rate(p[None], q[None]).item():.4f}")

def acceptance_ledger():
    p = torch.tensor([0.05, 0.10, 0.15, 0.20, 0.50])
    r = torch.tensor([0.40, 0.30, 0.15, 0.10, 0.05])
    print("\n草稿质量与加速比（γ=4，草稿开销 c=0.1）")
    print(f"{'混合系数':>10s}{'α':>8s}{'期望 Token':>12s}{'加速比':>10s}")
    for beta in (0.2, 0.4, 0.6, 0.8):
        q = (1.0 - beta) * p + beta * r
        alpha = acceptance_rate(p[None], q[None]).item()
        tokens = expected_output_tokens(alpha, 4)
        speed = speculative_speedup(alpha, 4, 0.1)
        print(f"{beta:>10.1f}{alpha:>8.3f}{tokens:>12.3f}{speed:>10.3f}")
    print("\nγ 的边际收益（α=0.8，c=0.1）")
    print(f"{'γ':>4s}{'期望 Token':>12s}{'加速比':>10s}")
    for gamma in range(1, 9):
        print(f"{gamma:>4d}{expected_output_tokens(0.8, gamma):>12.3f}"
              f"{speculative_speedup(0.8, gamma, 0.1):>10.3f}")

def acceptance_simulation():
    p = torch.tensor([0.05, 0.10, 0.15, 0.20, 0.50])
    q = torch.tensor([0.40, 0.25, 0.15, 0.10, 0.10])
    n, gamma = 100_000, 4
    q_seq = q.expand(n, gamma, -1)
    p_seq = p.expand(n, gamma + 1, -1)
    _, n_accept, _ = speculative_sample(
        q_seq, p_seq, torch.Generator().manual_seed(1))
    alpha = acceptance_rate(p[None], q[None]).item()
    empirical = n_accept.float().mean().item()
    theory = alpha * (1 - alpha ** gamma) / (1 - alpha)
    print(f"\n接受数模拟（10 万轮，α={alpha:.2f}，γ={gamma}）")
    print(f"  实测平均接受草稿数 {empirical:.4f}")
    print(f"  理论 α(1-α^γ)/(1-α) {theory:.4f}")
    print(f"  每轮平均输出 Token   {1 + empirical:.4f}")

def rollback_check():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=32, d_model=64, n_layers=2, n_heads=8,
                      n_kv_heads=2, d_ff=128, max_seq_len=64,
                      norm="rmsnorm", ffn="swiglu", bias=False,
                      pos_encoding="rope", attention="gqa")
    model = TinyLM(cfg).eval()
    prompt = torch.randint(0, 32, (1, 8))
    draft = torch.randint(0, 32, (1, 4))
    accepted = 2

    cache = KVCache(cfg)
    cached_prefill(model, prompt, cache)
    cached_forward(model, draft, cache)
    truncate_cache(cache, prompt.size(1) + accepted)
    with torch.no_grad():
        nxt = torch.randint(0, 32, (1, 1))
        got = cached_forward(model, nxt, cache, logits_to_keep=1)

    ref = KVCache(cfg)
    cached_prefill(model, torch.cat([prompt, draft[:, :accepted]], dim=1), ref)
    with torch.no_grad():
        want = cached_forward(model, nxt, ref, logits_to_keep=1)
    print("\nKV Cache 回滚（接受 2/4 个草稿 Token）")
    print(f"  回滚后长度        {cache.length}")
    print(f"  与重算最大误差    {(got - want).abs().max().item():.2e}")

def mtp_check():
    torch.manual_seed(0)
    head = MTPHead(d_model=16, vocab_size=11, num_steps=3)
    hidden = torch.randn(1, 6, 16)
    targets = torch.randint(0, 11, (1, 6))
    logits = head(hidden)
    loss = mtp_loss(logits, targets)
    print("\n多 Token 预测头")
    print(f"  每步 logits 形状  {tuple(logits[0].shape)} × {len(logits)}")
    print(f"  三步平均损失      {loss.item():.4f}")
    print(f"  仅第一步损失      {mtp_loss(logits, targets, [1.0, 0.0, 0.0]).item():.4f}")

def tree_check():
    mask = tree_attention_mask([-1, 0, 0, 1, 2])
    print("\n树状草稿的祖先掩码（1 表示可见）")
    for row in mask:
        print("  " + " ".join("1" if x else "0" for x in row))

def markov_demo():
    target = torch.tensor([[0.7, 0.1, 0.1, 0.1],
                           [0.1, 0.7, 0.1, 0.1],
                           [0.1, 0.1, 0.7, 0.1],
                           [0.1, 0.1, 0.1, 0.7]])
    draft = torch.tensor([[0.4, 0.2, 0.2, 0.2],
                          [0.2, 0.4, 0.2, 0.2],
                          [0.2, 0.2, 0.4, 0.2],
                          [0.2, 0.2, 0.2, 0.4]])
    alpha = acceptance_rate(target, draft)
    print("\n玩具马尔可夫模型")
    print(f"  平均每步接受率    {alpha.mean().item():.3f}")
    print(f"  γ=4 的期望 Token  {expected_output_tokens(alpha.mean().item(), 4):.3f}")

if __name__ == "__main__":
    lossless_check()
    acceptance_ledger()
    acceptance_simulation()
    rollback_check()
    mtp_check()
    tree_check()
    markov_demo()
