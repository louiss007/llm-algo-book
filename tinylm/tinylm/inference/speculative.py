"""第 22 章：投机解码与多 Token 预测（tinylm/inference/speculative.py）。"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True)
class SpecConfig:
    """投机解码参数，不进入 ``ModelConfig``。"""

    num_draft: int = 4
    seed: int | None = None

    def __post_init__(self):
        assert self.num_draft >= 1

def acceptance_rate(p, q):
    """(22.5)：α=Σ min(p,q)=1-TV(p,q)，批内逐行计算。"""
    return torch.minimum(p, q).sum(dim=-1)

def residual_probs(p, q):
    """(22.2)：r(x) ∝ max(0, p(x)-q(x))；p=q 时退回 p 以保持定义。"""
    r = (p - q).clamp_min(0.0)
    mass = r.sum(dim=-1, keepdim=True)
    return torch.where(mass > 0, r / mass.clamp_min(1e-12), p)

def rejection_sample(q, p, generator=None):
    """(22.1)–(22.3)：从草稿分布 q 提议，按 p 修正；返回 (token, accepted)。"""
    q = q / q.sum(dim=-1, keepdim=True)
    p = p / p.sum(dim=-1, keepdim=True)
    x = torch.multinomial(q, 1, generator=generator)
    u = torch.rand(q.size(0), 1, generator=generator)
    ratio = p.gather(1, x) / q.gather(1, x).clamp_min(1e-12)
    accepted = u < ratio.clamp(max=1.0)
    residual = residual_probs(p, q)
    fallback = torch.multinomial(residual, 1, generator=generator)
    token = torch.where(accepted, x, fallback)
    return token.squeeze(1), accepted.squeeze(1)

def speculative_sample(q_seq, p_seq, generator=None):
    """(22.1)–(22.3)：一次验证 γ 个草稿 Token，返回接受的草稿、前缀与修正 Token。

    ``q_seq`` 形状 ``(B, γ, V)``；``p_seq`` 形状 ``(B, γ+1, V)``，
    前 γ 个是目标模型对草稿 Token 的分布，最后一个是全部接受后的奖励 Token。
    """
    B, gamma, V = q_seq.shape
    assert p_seq.shape == (B, gamma + 1, V)
    q = q_seq / q_seq.sum(dim=-1, keepdim=True)
    p = p_seq / p_seq.sum(dim=-1, keepdim=True)

    drafts = torch.multinomial(q.reshape(B * gamma, V), 1,
                               generator=generator).reshape(B, gamma, 1)
    u = torch.rand(B, gamma, 1, generator=generator)
    ratio = p[:, :gamma].gather(-1, drafts) / q.gather(-1, drafts)
    accepted = u < ratio.clamp(max=1.0)
    prefix_ok = accepted.squeeze(-1).cumprod(dim=-1).bool()   # 首次拒绝后全为假
    n_accept = prefix_ok.sum(dim=-1)

    residual = residual_probs(p[:, :gamma], q)                # (B, γ, V)
    idx = n_accept.clamp(max=gamma - 1)
    fallback = residual.gather(
        1, idx[:, None, None].expand(B, 1, V)).squeeze(1)
    out_probs = torch.where((n_accept == gamma)[:, None],
                            p[:, gamma], fallback)
    correction = torch.multinomial(out_probs, 1, generator=generator)
    return drafts.squeeze(-1), n_accept, correction.squeeze(1)

def expected_output_tokens(alpha, gamma):
    """(22.7)：一次目标模型验证平均产出的 Token 数。"""
    if alpha >= 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)

def speculative_speedup(alpha, gamma, cost_ratio):
    """(22.8)：收益 = 期望产出 Token / (1 + γ·草稿相对开销)。"""
    return expected_output_tokens(alpha, gamma) / (1.0 + cost_ratio * gamma)

def truncate_cache(cache, length):
    """(22.13)：拒绝后把 KV Cache 回滚到已接受前缀。"""
    assert 0 <= length <= cache.length
    cache.length = int(length)
    return cache

class MTPHead(nn.Module):
    """(22.11)：共享主干隐状态上的多步预测头（简化版 MTP）。"""

    def __init__(self, d_model, vocab_size, num_steps=3):
        super().__init__()
        self.num_steps = num_steps
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, vocab_size, bias=False)
             for _ in range(num_steps)])

    def forward(self, hidden):
        return [head(hidden) for head in self.heads]

def mtp_loss(logits_list, targets, weights=None, ignore_index=-100):
    """(22.11)–(22.12)：第 i 个头预测 t+i，按步加权。"""
    T = targets.size(1)
    losses = []
    for i, logits in enumerate(logits_list):
        if i >= T:
            break
        pred = logits[:, :T - i]
        target = targets[:, i:]
        losses.append(F.cross_entropy(
            pred.reshape(-1, pred.size(-1)), target.reshape(-1),
            ignore_index=ignore_index))
    losses = torch.stack(losses)
    if weights is None:
        return losses.mean()
    weights = torch.as_tensor(weights, dtype=losses.dtype,
                              device=losses.device)[:losses.numel()]
    return (losses * weights).sum() / weights.sum()

def tree_attention_mask(parents):
    """(22.10)：树状草稿的祖先可见掩码，True 表示可以注意。"""
    n = len(parents)
    mask = torch.zeros(n, n, dtype=torch.bool)
    for node in range(n):
        cur = node
        while cur != -1:
            mask[node, cur] = True
            cur = parents[cur]
    return mask

def _generator(seed, device):
    if seed is None:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen

@torch.no_grad()
def speculative_decode(target_fn, draft_fn, prefix, max_new_tokens,
                       cfg=SpecConfig(), eos_token_id=None):
    """简化版投机解码：草稿自回归提议，目标模型一次验证。

    ``target_fn``/``draft_fn`` 输入 ``(1, T)``，返回 ``(1, T, V)`` 的
    逐位置下一 Token 分布。本实现每轮重算目标前向，只用于验证算法。
    """
    if prefix.dim() != 2 or prefix.size(0) != 1:
        raise ValueError("speculative_decode 只接受 (1, T) 的提示")
    device = prefix.device
    gen = _generator(cfg.seed, device)
    seq = prefix
    start = prefix.size(1)
    while seq.size(1) - start < max_new_tokens:
        remaining = max_new_tokens - (seq.size(1) - start)
        gamma = min(cfg.num_draft, remaining)
        draft_seq, qs, drafts = seq, [], []
        for _ in range(gamma):
            q = draft_fn(draft_seq)[:, -1]
            qs.append(q)
            x = torch.multinomial(q, 1, generator=gen)
            drafts.append(x)
            draft_seq = torch.cat([draft_seq, x], dim=1)
        q_seq = torch.stack(qs, dim=1)                     # (1, γ, V)
        L = seq.size(1)
        p_all = target_fn(torch.cat([seq] + drafts, dim=1))
        p_seq = torch.stack([p_all[:, L - 1 + i] for i in range(gamma + 1)],
                            dim=1)                         # (1, γ+1, V)
        _, n_accept, correction = speculative_sample(q_seq, p_seq, gen)
        a = int(n_accept.item())
        new_tokens = drafts[:a] + [correction[:, None]]
        seq = torch.cat([seq] + new_tokens, dim=1)
        if eos_token_id is not None and int(seq[0, -1]) == eos_token_id:
            break
    return seq[:, :start + max_new_tokens]
