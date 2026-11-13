"""第 21 章：分块注意力与在线 Softmax（tinylm/inference/flash.py）。"""
import math

import torch

def dense_attention(q, k, v, causal=True, q_start=None):
    """参考实现：一次算完整个注意力矩阵，形状 (B, H, Tq, Tk)。"""
    Tq, Tk = q.size(-2), k.size(-2)
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = q @ k.transpose(-2, -1) * scale
    if causal:
        start = Tk - Tq if q_start is None else q_start
        qi = torch.arange(start, start + Tq, device=q.device)
        ki = torch.arange(Tk, device=q.device)
        scores = scores.masked_fill(ki[None, :] > qi[:, None], float("-inf"))
    return scores.softmax(dim=-1) @ v

def flash_attention(q, k, v, block_q=64, block_k=64, causal=True):
    """公式 (21.8)–(21.11)：分块读取 K/V，在线 Softmax 累积输出。"""
    B, H, Tq, D = q.shape
    Tk = k.size(-2)
    scale = 1.0 / math.sqrt(D)
    start = Tk - Tq
    out = torch.zeros_like(q)
    for i0 in range(0, Tq, block_q):
        i1 = min(i0 + block_q, Tq)
        q_blk = q[:, :, i0:i1]
        row_max = torch.full((B, H, i1 - i0, 1), -1e30,
                             device=q.device, dtype=q.dtype)
        row_sum = torch.zeros_like(row_max)
        acc = torch.zeros_like(q_blk)
        for j0 in range(0, Tk, block_k):
            j1 = min(j0 + block_k, Tk)
            if causal and j0 > start + i1 - 1:
                break
            s = q_blk @ k[:, :, j0:j1].transpose(-2, -1) * scale
            if causal:
                qi = torch.arange(start + i0, start + i1, device=q.device)
                ki = torch.arange(j0, j1, device=q.device)
                s = s.masked_fill(ki[None, :] > qi[:, None], float("-inf"))
            m_new = torch.maximum(row_max, s.max(dim=-1, keepdim=True).values)
            alpha = torch.exp(row_max - m_new)
            p = torch.exp(s - m_new)
            row_sum = row_sum * alpha + p.sum(dim=-1, keepdim=True)
            acc = acc * alpha + p @ v[:, :, j0:j1]
            row_max = m_new
        out[:, :, i0:i1] = acc / row_sum.clamp_min(1e-9)
    return out
