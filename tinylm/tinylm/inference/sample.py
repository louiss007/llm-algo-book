"""第 20 章：解码策略（tinylm/inference/sample.py）。"""
import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class SamplingConfig:
    """生成脚本参数，不进入 ``ModelConfig``。"""

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None

    def __post_init__(self):
        assert self.temperature > 0.0
        assert self.top_k is None or self.top_k >= 1
        assert self.top_p is None or 0.0 < self.top_p <= 1.0
        assert 0.0 <= self.min_p <= 1.0
        assert self.repetition_penalty > 0.0

def apply_repetition_penalty(logits, token_ids, penalty, pad_id=None):
    """(20.11)：对已出现 Token 的 logit 做分段缩放。"""
    if penalty == 1.0:
        return logits
    out = logits.clone()
    for b in range(out.size(0)):
        seen = torch.unique(token_ids[b])
        if pad_id is not None:
            seen = seen[seen != pad_id]
        values = out[b, seen]
        out[b, seen] = torch.where(values > 0, values / penalty,
                                   values * penalty)
    return out

def top_k_mask(logits, k):
    """(20.7)：保留 logit 最大的 k 个 Token。"""
    k = min(int(k), logits.size(-1))
    indices = logits.topk(k, dim=-1).indices
    mask = torch.zeros_like(logits, dtype=torch.bool)
    return mask.scatter_(-1, indices, True)

def top_p_mask(logits, p):
    """(20.8)：保留累积概率首次达到 p 的最小前缀。"""
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = cumulative > p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    keep_sorted = ~remove
    mask = torch.zeros_like(logits, dtype=torch.bool)
    return mask.scatter_(-1, sorted_idx, keep_sorted)

def min_p_mask(logits, min_p):
    """(20.9)：保留概率不低于最大概率 min_p 倍的 Token。"""
    probs = torch.softmax(logits, dim=-1)
    threshold = min_p * probs.max(dim=-1, keepdim=True).values
    return probs >= threshold

def decode_mask(logits, cfg):
    """依次叠加 top-k、top-p、min-p 的保留集合。"""
    mask = torch.ones_like(logits, dtype=torch.bool)
    if cfg.top_k is not None:
        mask &= top_k_mask(logits, cfg.top_k)           # (20.7)
    if cfg.top_p is not None:
        mask &= top_p_mask(logits, cfg.top_p)           # (20.8)
    if cfg.min_p > 0.0:
        mask &= min_p_mask(logits, cfg.min_p)           # (20.9)
    return mask

def filtered_probs(logits, cfg, token_ids=None, pad_id=None):
    """(20.6)–(20.10)：返回截断并重新归一化后的分布。"""
    adjusted = logits
    if token_ids is not None and cfg.repetition_penalty != 1.0:
        adjusted = apply_repetition_penalty(
            adjusted, token_ids, cfg.repetition_penalty, pad_id)  # (20.11)
    adjusted = adjusted / cfg.temperature               # (20.6)
    mask = decode_mask(adjusted, cfg)
    adjusted = adjusted.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(adjusted, dim=-1)             # (20.10)
    return probs, mask

def _make_generator(seed, device):
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator

@torch.no_grad()
def sample_decode(model, input_ids, max_new_tokens, cfg=SamplingConfig(),
                  eos_token_id=None, pad_id=0):
    """逐 Token 采样，返回 ``(seqs, mask)``。"""
    model.eval()
    seqs = input_ids
    done = torch.zeros(seqs.size(0), dtype=torch.bool, device=seqs.device)
    generator = _make_generator(cfg.seed, seqs.device)
    for _ in range(max_new_tokens):
        logits = model(seqs).logits[:, -1]
        probs, _ = filtered_probs(logits, cfg, seqs, pad_id)
        nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        if eos_token_id is not None:
            nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
        seqs = torch.cat([seqs, nxt[:, None]], dim=1)
        if eos_token_id is not None:
            done = done | (nxt == eos_token_id)
            if bool(done.all()):
                break
    return seqs, seqs != pad_id

@torch.no_grad()
def greedy_decode(model, input_ids, max_new_tokens, eos_token_id=None,
                  pad_id=0):
    """(20.2)：逐 Token 取 argmax，返回 ``(seqs, mask)``。"""
    model.eval()
    seqs = input_ids
    done = torch.zeros(seqs.size(0), dtype=torch.bool, device=seqs.device)
    for _ in range(max_new_tokens):
        nxt = model(seqs).logits[:, -1].argmax(dim=-1)
        if eos_token_id is not None:
            nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
        seqs = torch.cat([seqs, nxt[:, None]], dim=1)
        if eos_token_id is not None:
            done = done | (nxt == eos_token_id)
            if bool(done.all()):
                break
    return seqs, seqs != pad_id

@torch.no_grad()
def beam_search(model, prompt, beam_width=4, max_new_tokens=32,
                eos_token_id=None, length_penalty=0.0, pad_id=0):
    """(20.3)–(20.5)：单条提示的束搜索，返回得分最高的序列。"""
    model.eval()
    if prompt.dim() != 2 or prompt.size(0) != 1:
        raise ValueError("beam_search 只接受单条提示，形状为 (1, T)")
    beams = [(0.0, prompt)]
    completed = []
    for _ in range(max_new_tokens):
        pairs = []
        batch = torch.cat([seq for _, seq in beams], dim=0)
        log_probs = F.log_softmax(model(batch).logits[:, -1], dim=-1)
        for i, (score, seq) in enumerate(beams):
            values, indices = log_probs[i].topk(beam_width, dim=-1)
            for value, token in zip(values.tolist(), indices.tolist()):
                pairs.append((score + value, torch.cat(
                    [seq, torch.tensor([[token]], device=seq.device)], dim=1)))
        if eos_token_id is not None:
            unfinished = []
            for score, seq in pairs:
                if int(seq[0, -1]) == eos_token_id:
                    completed.append((score, seq))
                else:
                    unfinished.append((score, seq))
        else:
            unfinished = pairs
        if not unfinished:
            break
        beams = sorted(unfinished, key=lambda item: item[0], reverse=True)
        beams = beams[:beam_width]
    completed.extend(beams)
    if not completed:
        raise RuntimeError("束搜索没有产生任何候选")

    def rank(item):
        score, seq = item
        length = seq.size(1) ** length_penalty      # (20.5)
        return score / length if length else score

    return max(completed, key=rank)[1]

@torch.no_grad()
def sequence_logprob(model, seqs, prompt_len, pad_id=None):
    """(20.3)：给定序列在模型下的对数似然，跳过提示与 padding。"""
    logits = model(seqs[:, :-1]).logits
    log_probs = F.log_softmax(logits, dim=-1)
    targets = seqs[:, 1:]
    token_logp = log_probs.gather(-1, targets[..., None]).squeeze(-1)
    valid = torch.ones_like(targets, dtype=torch.bool)
    valid[:, :max(prompt_len - 1, 0)] = False
    if pad_id is not None:
        valid &= targets != pad_id
    return (token_logp * valid).sum(dim=-1)

def pass_at_k(n, c, k):
    """(20.13)：从 n 条样本中取 k 条，至少一条正确的无偏估计。"""
    if not 1 <= k <= n:
        raise ValueError("需要 1 <= k <= n")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

@torch.no_grad()
def generate(model, input_ids, *, max_new_tokens=64, sampling=None,
             eos_token_id=None, use_cache=True):
    """F.2.4 冻结入口：``sampling=None`` 表示贪心。"""
    if use_cache:
        warnings.warn("第 20 章尚未实现 KV Cache，use_cache 被忽略")
    if sampling is None:
        seqs, _ = greedy_decode(model, input_ids, max_new_tokens,
                                eos_token_id=eos_token_id)
    else:
        seqs, _ = sample_decode(model, input_ids, max_new_tokens, sampling,
                                eos_token_id=eos_token_id)
    return seqs
