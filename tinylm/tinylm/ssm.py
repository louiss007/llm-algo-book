"""替代架构（tinylm/ssm.py，第 8 章）。

线性注意力、选择性 SSM、两种替换块与按 ``block_type`` 的分派。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock
from .moe import make_ffn_or_moe
from .norm import make_norm


def scan_states(a, b):
    """公式 (8.17)：h_t = a_t h_{t-1} + b_t 的并行关联扫描。

    Hillis-Steele 扫描，把 (a_t, b_t) 复合为 (a2 a1, a2 b1 + b2)，
    共 log2(T) 轮，每轮对序列维做一次向量化更新。
    """
    A, Bv = a, b
    T = a.shape[1]
    step = 1
    while step < T:
        A_prev = A
        A = torch.cat([A[:, :step], A[:, step:] * A_prev[:, :-step]], dim=1)
        Bv = torch.cat(
            [Bv[:, :step], A_prev[:, step:] * Bv[:, :-step] + Bv[:, step:]], dim=1)
        step *= 2
    return Bv


class SelectiveSSM(nn.Module):
    """公式 (8.8)–(8.16)：输入相关的选择性状态空间层（Mamba 风格）。"""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.inner = cfg.ssm_expand * d
        self.n_state = cfg.ssm_d_state
        self.in_proj = nn.Linear(d, 2 * self.inner, bias=cfg.bias)
        self.conv_w = nn.Parameter(torch.empty(self.inner, 4))  # 深度因果卷积
        nn.init.kaiming_uniform_(self.conv_w, a=math.sqrt(5))
        self.conv_b = nn.Parameter(torch.zeros(self.inner)) if cfg.bias else None
        self.x_proj = nn.Linear(self.inner, 2 * self.n_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.inner, bias=True)
        a = torch.empty(self.inner, self.n_state).uniform_(1.0, 16.0)
        self.A_log = nn.Parameter(torch.log(a))          # A = -exp(A_log) < 0
        self.D = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, d, bias=cfg.bias)
        self.last_state = None

    def forward(self, x, attention_mask=None, position_ids=None):
        B, T, _ = x.shape
        u, z = self.in_proj(x).chunk(2, dim=-1)
        xp = F.pad(u.transpose(1, 2), (3, 0))                      # 左侧补 3 个 0
        conv = sum(xp[:, :, k:k + T] * self.conv_w[:, k].view(1, -1, 1)
                   for k in range(4))
        if self.conv_b is not None:
            conv = conv + self.conv_b.view(1, -1, 1)
        u = F.silu(conv.transpose(1, 2))                            # 因果卷积

        bc_dt = self.x_proj(u)
        B_t, C_t, dt_raw = torch.split(
            bc_dt, [self.n_state, self.n_state, 1], dim=-1)           # (8.13)
        dt = F.softplus(self.dt_proj(dt_raw))                         # (8.13)

        A = -torch.exp(self.A_log)                                    # (8.10)
        A_bar = torch.exp(dt.unsqueeze(-1) * A)                       # (8.14)
        ratio = torch.where(
            A.abs() < 1e-6,
            dt.unsqueeze(-1).expand_as(A_bar),
            (A_bar - 1.0) / A,
        )
        B_bar = ratio * B_t.unsqueeze(-2)                             # (8.14)

        h = scan_states(A_bar, B_bar)                                 # (8.17)
        y = (C_t.unsqueeze(-2) * h).sum(-1) + self.D * u              # (8.10)
        self.last_state = h[:, -1].detach()
        return self.out_proj(y * F.silu(z))


class LinearAttention(nn.Module):
    """公式 (8.1)–(8.6)：以 φ(q)ᵀφ(k) 替代 softmax 的线性注意力。"""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.q_proj = nn.Linear(d, d, bias=cfg.bias)
        self.k_proj = nn.Linear(d, d, bias=cfg.bias)
        self.v_proj = nn.Linear(d, d, bias=cfg.bias)
        self.out_proj = nn.Linear(d, d, bias=cfg.bias)
        self.eps = 1e-6
        self.last_state = None

    def forward(self, x, attention_mask=None, position_ids=None):
        q = F.elu(self.q_proj(x)) + 1.0                               # φ(q)
        k = F.elu(self.k_proj(x)) + 1.0
        v = self.v_proj(x)
        S = (k.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(dim=1)         # (8.4)
        z = k.cumsum(dim=1)
        num = torch.einsum("btd,btdv->btv", q, S)                     # (8.3)
        den = torch.einsum("btd,btd->bt", q, z).clamp_min(self.eps)
        self.last_state = S[:, -1].detach()
        return self.out_proj(num / den.unsqueeze(-1))                 # (8.5)


class SSMBlock(nn.Module):
    """与 TransformerBlock 同构：只把注意力换成 SSM，前馈子层保持不变。"""

    def __init__(self, cfg):
        super().__init__()
        self.parallel = cfg.parallel_block
        self.norm1 = make_norm(cfg)
        self.mixer = SelectiveSSM(cfg)
        self.norm2 = None if self.parallel else make_norm(cfg)
        self.mlp = make_ffn_or_moe(cfg)

    def forward(self, x, attention_mask=None, position_ids=None):
        if self.parallel:
            h = self.norm1(x)
            delta, aux = self._ffn(h)
            return x + self.mixer(h) + delta, aux
        x = x + self.mixer(self.norm1(x))
        delta, aux = self._ffn(self.norm2(x))
        return x + delta, aux

    def _ffn(self, h):
        out = self.mlp(h)
        return out if isinstance(out, tuple) else (out, None)


class LinearAttentionBlock(SSMBlock):
    """同样的块结构，mixer 换成线性注意力。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.mixer = LinearAttention(cfg)


def make_block(cfg, layer_idx):
    """按 block_type 分派；默认 attention 与第 7 章逐位一致。"""
    if cfg.block_type == "attention":
        return TransformerBlock(cfg)
    if cfg.block_type == "linear_attention":
        return LinearAttentionBlock(cfg)
    if cfg.block_type == "ssm":
        return SSMBlock(cfg)
    m = cfg.hybrid_attn_every
    return TransformerBlock(cfg) if (layer_idx + 1) % m == 0 else SSMBlock(cfg)


def layer_kinds(cfg):
    if cfg.block_type == "attention":
        return ["attn"] * cfg.n_layers
    if cfg.block_type == "linear_attention":
        return ["linear"] * cfg.n_layers
    if cfg.block_type == "ssm":
        return ["ssm"] * cfg.n_layers
    m = cfg.hybrid_attn_every
    return ["attn" if (i + 1) % m == 0 else "ssm" for i in range(cfg.n_layers)]
