"""前馈网络（tinylm/ffn.py）。

第 3 章两层 MLP，第 5 章增加 SwiGLU 与 ``make_ffn`` 分派。
"""
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """公式 (3.15)：两层前馈网络，中间维度 d_ff。"""

    def __init__(self, cfg):
        super().__init__()
        self.fc_in = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.fc_out = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)

    def forward(self, x):
        return self.fc_out(F.gelu(self.fc_in(x)))


class SwiGLU(nn.Module):
    """公式 (5.7)：SiLU 门控的三投影前馈网络。"""

    def __init__(self, cfg):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))         # (5.7)


def make_ffn(cfg):
    if cfg.ffn == "swiglu":
        return SwiGLU(cfg)
    return MLP(cfg)
