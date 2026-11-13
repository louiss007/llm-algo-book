"""归一化层（tinylm/norm.py）。

第 3 章 LayerNorm，第 5 章增加 RMSNorm 与 ``make_norm`` 分派。
"""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """公式 (3.2)：逐位置标准化，再仿射变换。"""

    def __init__(self, d_model, eps=1e-5, bias=True):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model)) if bias else None
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)                      # (3.1) 均值
        var = x.var(dim=-1, unbiased=False, keepdim=True)      # (3.1) 方差
        x_hat = (x - mu) / torch.sqrt(var + self.eps)
        out = self.gamma * x_hat
        return out + self.beta if self.beta is not None else out


class RMSNorm(nn.Module):
    """公式 (5.2)：只除以均方根，不减均值。"""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (5.1)
        return self.gamma * x / rms                                       # (5.2)


def make_norm(cfg):
    """按 ``cfg.norm`` 返回归一化层，默认保持第 3 章 LayerNorm。"""
    if cfg.norm == "rmsnorm":
        return RMSNorm(cfg.d_model, cfg.norm_eps)
    return LayerNorm(cfg.d_model, cfg.norm_eps, cfg.bias)
