"""位置编码（tinylm/position.py）。

第 4 章：正弦编码、RoPE（含 PI/NTK/YaRN 缩放）与 ALiBi。
"""
import math

import torch
import torch.nn as nn


def sinusoidal_positions(max_len, d_model, base=10000.0):
    """公式 (4.1)：正弦位置编码，返回 (max_len, d_model)。"""
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, d_model, 2, dtype=torch.float32)
    omega = base ** (-i / d_model)                    # 公式 (4.2)
    angles = pos * omega
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


def rope_inv_freq(d_head, theta=10000.0, scaling=None, train_len=1024):
    """公式 (4.6)、(4.15)–(4.17)：RoPE 各维频率及缩放。"""
    i = torch.arange(0, d_head, 2, dtype=torch.float32)
    theta_i = theta ** (-i / d_head)                  # 公式 (4.6)
    if scaling is None or scaling.method in ("none", "pi"):
        return theta_i
    s = scaling.factor
    if scaling.method == "ntk":                      # 公式 (4.15)
        base = theta * s ** (d_head / (d_head - 2))
        return base ** (-i / d_head)
    if scaling.method == "yarn":                     # 公式 (4.16)–(4.17)
        lam = 2 * math.pi / theta_i
        r = train_len / lam                          # 低频 r 小，高频 r 大
        alpha, beta = 1.0, 32.0
        w = ((beta - r) / (beta - alpha)).clamp(0, 1)
        return theta_i * ((1 - w) + w / s)
    raise ValueError(scaling.method)


def apply_rope(x, cos, sin):
    """公式 (4.7)：对最后一维做分半旋转。"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RoPE(nn.Module):
    def __init__(self, d_head, theta=10000.0, scaling=None, train_len=1024):
        super().__init__()
        self.scaling = scaling
        self.register_buffer(
            "inv_freq", rope_inv_freq(d_head, theta, scaling, train_len),
            persistent=False,
        )
        self.temp = 1.0
        if scaling is not None and scaling.method == "yarn":
            self.temp = 0.1 * math.log(scaling.factor) + 1.0   # 公式 (4.18)

    def forward(self, q, k, position_ids):
        pos = position_ids if position_ids.dim() == 2 else position_ids.unsqueeze(0)
        if self.scaling is not None and self.scaling.method == "pi":
            pos = pos / self.scaling.factor                 # 公式 (4.14)
        freqs = pos.float().unsqueeze(-1) * self.inv_freq
        cos = freqs.cos().unsqueeze(1)
        sin = freqs.sin().unsqueeze(1)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


def alibi_slopes(n_heads):
    """公式 (4.12)：几何递减的 ALiBi 斜率。"""
    return 2.0 ** (-8.0 * (torch.arange(n_heads, dtype=torch.float32) + 1) / n_heads)


class ALiBi(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.register_buffer("slopes", alibi_slopes(n_heads), persistent=False)

    def forward(self, T, device):
        idx = torch.arange(T, device=device)
        dist = (idx[:, None] - idx[None, :]).clamp(min=0)   # j <= i 时的距离
        return -self.slopes[:, None, None] * dist[None, :, :]
