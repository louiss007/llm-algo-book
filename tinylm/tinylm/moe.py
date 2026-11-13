"""混合专家层（tinylm/moe.py，第 7 章）。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ffn import make_ffn


class Expert(nn.Module):
    """公式 (7.10)：一个专家就是一个独立的前馈网络。"""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        f = cfg.moe.d_ff_expert or cfg.d_ff
        self.is_glu = cfg.ffn == "swiglu"
        if self.is_glu:
            self.w_gate = nn.Linear(d, f, bias=cfg.bias)
            self.w_up = nn.Linear(d, f, bias=cfg.bias)
            self.w_down = nn.Linear(f, d, bias=cfg.bias)
        else:
            self.fc_in = nn.Linear(d, f, bias=cfg.bias)
            self.fc_out = nn.Linear(f, d, bias=cfg.bias)

    def forward(self, x):
        if self.is_glu:
            return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        return self.fc_out(F.gelu(self.fc_in(x)))


def load_balance_loss(f, p, coef, n_experts):
    """公式 (7.18)：α E Σ_e f_e P_e。f、p 都是长度为 E 的向量。"""
    return coef * n_experts * (f * p).sum()


class MoE(nn.Module):
    """稀疏激活的前馈层：路由、Top-K 分派、容量、共享专家与辅助损失。"""

    def __init__(self, cfg):
        super().__init__()
        m = cfg.moe
        self.n_experts = m.n_experts
        self.top_k = m.top_k
        self.capacity_factor = m.capacity_factor
        self.aux_coef = m.aux_loss_coef
        self.router = nn.Linear(cfg.d_model, m.n_experts, bias=False)  # (7.2)
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(m.n_experts)])
        self.shared = nn.ModuleList([Expert(cfg) for _ in range(m.n_shared)])
        self.last_load = None
        self.last_dropped = 0

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        probs = self.router(flat).softmax(dim=-1)                  # (7.3)
        top_p, top_idx = probs.topk(self.top_k, dim=-1)            # (7.4)
        gates = top_p / top_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # (7.5)

        n_tokens = flat.shape[0]
        if self.capacity_factor is None:
            capacity = None
        else:                                                       # (7.7)
            capacity = max(1, math.ceil(
                self.capacity_factor * n_tokens * self.top_k / self.n_experts))

        out = torch.zeros_like(flat)
        load = torch.zeros(self.n_experts, dtype=torch.long, device=flat.device)
        dropped = 0
        for e, expert in enumerate(self.experts):
            slot = top_idx == e                                     # (7.8)
            chosen = slot.any(dim=-1)
            if capacity is not None:
                keep = torch.cumsum(chosen.long(), dim=0) <= capacity
                dropped += int((chosen & ~keep).sum())
                chosen = chosen & keep
            n_e = int(chosen.sum())
            load[e] = n_e
            if n_e == 0:
                continue
            idx = chosen.nonzero(as_tuple=True)[0]
            g = (gates * slot).sum(dim=-1)[idx]                     # (7.6)
            out[idx] += g.unsqueeze(-1) * expert(flat[idx])

        for expert in self.shared:                                  # (7.21)
            out = out + expert(flat)

        self.last_load = load.detach()
        self.last_dropped = dropped

        aux = None
        if self.training and self.aux_coef > 0:
            f = load.float() / max(1, n_tokens * self.top_k)        # (7.16)
            p = probs.mean(dim=0)                                   # (7.17)
            aux = load_balance_loss(f, p, self.aux_coef, self.n_experts)  # (7.18)
        return out.reshape(B, T, D), aux


def make_ffn_or_moe(cfg):
    if cfg.moe is None:
        return make_ffn(cfg)
    return MoE(cfg)
