"""第 18 章：可验证奖励与组相对策略优化（tinylm/posttrain/grpo.py）。"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tinylm.posttrain import ppo as P

@dataclass(frozen=True)
class GRPOConfig:
    """GRPO 训练脚本参数，不进入 ``ModelConfig``。"""

    group_size: int = 8             # 每个提示采样的回答数 G
    lr: float = 1e-3
    beta: float = 0.0               # KL 系数；0 表示不使用参考模型
    clip_low: float = 0.2           # 下侧裁剪半径
    clip_high: float = 0.2          # 上侧裁剪半径，DAPO 取 0.28
    epochs: int = 2                 # 同一组 rollout 重复更新次数
    minibatch: int = 64
    max_new: int = 4                # 响应最多生成多少个 Token（含 EOS）
    temperature: float = 1.0
    normalize_std: bool = True      # 是否除以组内标准差
    leave_one_out: bool = False     # 基线是否排除当前样本
    token_level: bool = True        # 按批内 Token 平均，而非逐序列平均
    max_grad_norm: float = 1.0
    pad_id: int = 0
    eos_id: int = 4

    def __post_init__(self):
        assert self.group_size >= 1
        assert self.lr > 0.0
        assert self.beta >= 0.0
        assert 0.0 <= self.clip_low < 1.0
        assert 0.0 <= self.clip_high < 1.0
        assert self.epochs >= 1
        assert self.minibatch >= 1
        assert self.max_new >= 1
        assert self.temperature > 0.0
        assert self.max_grad_norm > 0.0

def group_advantages(rewards, group_size, normalize_std=True,
                     leave_one_out=False, eps=1e-4):
    """(18.4)–(18.6)：组内均值基线，可选除以组内标准差。

    ``rewards`` 形状 ``(B * G,)``，组内连续排列。G=1 且使用组均值基线时，
    每个样本的基线就是它自己，优势恒为零，这是 GRPO 至少需要两个样本的
    直接原因。``leave_one_out=True`` 用其余 G-1 个样本的均值作基线，可以
    消除有限 G 带来的 (1 - 1/G) 收缩。
    """
    groups = rewards.view(-1, group_size)
    if group_size == 1:
        if leave_one_out:
            raise ValueError("leave-one-out 基线要求 group_size >= 2")
        return torch.zeros_like(groups).view(-1)
    if leave_one_out:
        total = groups.sum(dim=1, keepdim=True)
        baseline = (total - groups) / (group_size - 1)
    else:
        baseline = groups.mean(dim=1, keepdim=True)
    centered = groups - baseline
    if normalize_std:
        scale = groups.std(dim=1, unbiased=False, keepdim=True)
        centered = centered / (scale + eps)
    return centered.view(-1)

def zero_variance_groups(rewards, group_size, eps=1e-6):
    """(18.12)：返回 ``(B,)`` 布尔张量，标记全对或全错的组。"""
    groups = rewards.view(-1, group_size)
    return groups.std(dim=1, unbiased=False) < eps

def mixed_group_probability(p, group_size):
    """(18.12)：组内既有正确又有错误回答的概率。

    单条回答正确率为 ``p`` 时，全对概率为 ``p^G``，全错为 ``(1-p)^G``，
    两者都给出零优势。只有混合组提供策略梯度。
    """
    return 1.0 - p ** group_size - (1.0 - p) ** group_size

@torch.no_grad()
def sample_groups(policy, prompts, cfg, generator):
    """(18.2)：把每个提示重复 G 次后采样，返回 ``(B*G, T)`` 的序列与掩码。

    采样与日志概率使用同一个温度，保证重要性比的分母就是行为策略。
    """
    policy.eval()
    seqs = prompts.repeat_interleave(cfg.group_size, dim=0)
    done = torch.zeros(seqs.size(0), dtype=torch.bool)
    for _ in range(cfg.max_new):
        logits = policy(seqs).logits[:, -1] / cfg.temperature
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, cfg.pad_id), nxt)
        seqs = torch.cat([seqs, nxt[:, None]], dim=1)
        done = done | (nxt == cfg.eos_id)
        if bool(done.all()):
            break
    return seqs, seqs != cfg.pad_id

def tempered_logprobs(model, seqs, attn, temperature=1.0):
    """(18.7)：与采样温度一致的逐 Token 对数概率，形状 ``(B, T)``。"""
    logits = model(seqs, attention_mask=attn).logits[:, :-1] / temperature
    logp = F.log_softmax(logits, dim=-1)
    target = seqs[:, 1:]
    out = logp.gather(-1, target[..., None]).squeeze(-1)
    out = torch.where(attn[:, 1:], out, torch.zeros_like(out))
    return F.pad(out, (1, 0))

def grpo_loss(logp, old_logp, adv, mask, ref_logp=None, beta=0.0,
              clip_low=0.2, clip_high=0.2, token_level=True):
    """(18.7)–(18.11)：GRPO 的裁剪代理目标与 KL 惩罚。

    ``logp``、``old_logp``、``mask`` 形状 ``(N, T)``，``adv`` 形状 ``(N,)``。
    返回损失与诊断量；损失已经取负，可以直接反向。
    """
    # (18.7) 重要性比
    ratio = torch.exp(logp - old_logp)
    # (18.8) 裁剪代理目标
    surrogate = torch.minimum(
        ratio * adv[:, None],
        torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * adv[:, None],
    )
    if token_level:
        # (18.11) 左式：批内所有响应 Token 一起平均
        pg = (surrogate * mask).sum() / mask.sum().clamp_min(1)
    else:
        # (18.11) 右式：先对每条回答平均，再对回答平均
        lengths = mask.sum(1).clamp_min(1)
        pg = ((surrogate * mask).sum(1) / lengths).mean()
    loss = -pg
    diagnostics = {
        "pg": float(pg.detach()),
        "ratio": float(((ratio * mask).sum() / mask.sum().clamp_min(1)).detach()),
        "clip_frac": float((((ratio < 1.0 - clip_low) |
                             (ratio > 1.0 + clip_high)) * mask).sum()
                           / mask.sum().clamp_min(1)),
    }
    if ref_logp is not None and beta > 0.0:
        # (18.10) k3 估计量：r - log r - 1，其中 r = π_ref / π_θ
        log_ratio = torch.clamp(ref_logp - logp, -20.0, 20.0)
        k3 = torch.exp(log_ratio) - log_ratio - 1.0
        kl = (k3 * mask).sum() / mask.sum().clamp_min(1)
        loss = loss + beta * kl
        diagnostics["kl"] = float(kl.detach())
    return loss, diagnostics

def answer_positions(seqs, prompt_len, pad_id=0, eos_id=4):
    """返回每条响应中最后一个非 padding、非 EOS 的 Token 位置。"""
    positions = torch.arange(seqs.size(1), device=seqs.device)[None, :]
    valid = (seqs != pad_id) & (seqs != eos_id) & (positions >= prompt_len)
    has = valid.any(dim=1)
    reverse = valid.long().flip(-1).argmax(dim=-1)
    idx = seqs.size(1) - 1 - reverse
    return torch.where(has, idx, torch.full_like(idx, -1))

def verifiable_reward(seqs, prompt_len, targets, mode="final",
                      pad_id=0, eos_id=4):
    """(18.1)：规则可验证奖励。``final`` 检查最后一个答案 Token，
    ``anywhere`` 只要求答案出现在响应中的任意位置。"""
    mask = (seqs != pad_id) & (seqs != eos_id)
    if mode == "anywhere":
        return ((seqs == targets[:, None]) & mask).any(dim=1).float()
    if mode != "final":
        raise ValueError(f"未知奖励模式 {mode!r}")
    idx = answer_positions(seqs, prompt_len, pad_id, eos_id)
    safe = idx.clamp_min(0)
    token = seqs.gather(1, safe[:, None]).squeeze(1)
    return ((token == targets) & (idx >= 0)).float()
