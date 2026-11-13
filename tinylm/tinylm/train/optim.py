"""第 11 章：优化器、学习率调度与训练稳定性（tinylm/train/optim.py）。"""
import math
from collections import deque

import torch


class AdamW(torch.optim.Optimizer):
    """公式 (11.3)–(11.7)：偏差修正的 Adam 与解耦权重衰减。"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1):
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError("lr、eps、weight_decay 必须非负")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("betas 必须在 [0, 1) 内")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)              # (11.3)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)       # (11.3)
                m_hat = m / (1 - b1 ** t)                     # (11.4)
                v_hat = v / (1 - b2 ** t)                     # (11.4)
                p.mul_(1 - lr * wd)                           # (11.7) 解耦衰减
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
        return loss


def make_param_groups(model, weight_decay=0.1):
    """权重衰减只作用于二维以上参数；偏置与归一化增益不衰减。"""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_factor(step, total, warmup=0, kind="cosine", min_ratio=0.1,
              decay_start=None, power=1.0):
    """公式 (11.8)–(11.10)：返回第 step 步的学习率乘子。

    step 是已完成的更新次数，取值 0..total；warmup 结束处因子恰好为 1。
    """
    if total <= 0 or step < 0:
        raise ValueError("total 必须为正，step 必须非负")
    step = min(step, total)
    if warmup > 0 and step < warmup:
        return step / warmup                              # (11.8)
    if kind == "constant":
        return 1.0
    if kind == "cosine":
        span = max(1, total - warmup)
        t = min(1.0, (step - warmup) / span)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))  # (11.9)
    if kind == "wsd":
        start = int(0.8 * total) if decay_start is None else decay_start
        start = max(start, warmup)
        if step <= start:
            return 1.0                                    # (11.10) 稳定段
        t = min(1.0, (step - start) / max(1, total - start))
        return min_ratio + (1 - min_ratio) * (1 - t) ** power  # (11.10) 衰减段
    raise ValueError(f"未知调度: {kind}")


def clip_grad_norm_(parameters, max_norm):
    """公式 (11.14)：按全局 L2 范数缩放所有梯度，返回裁剪前的范数。"""
    if max_norm <= 0:
        raise ValueError("max_norm 必须为正")
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    total = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in params))
    coef = min(1.0, max_norm / (total.item() + 1e-6))
    if coef < 1.0:
        for p in params:
            p.grad.mul_(coef)
    return total.item()


class LossSpikeDetector:
    """公式 (11.13)：用滑动窗口的均值与标准差判定损失尖峰。"""

    def __init__(self, window=50, z_threshold=4.0, min_history=10):
        self.history = deque(maxlen=window)
        self.z_threshold = z_threshold
        self.min_history = min_history

    def update(self, loss):
        """先判定再更新，避免当前点抬高自己的基线。"""
        x = float(loss)
        z = 0.0
        spike = False
        if len(self.history) >= self.min_history:
            mean = sum(self.history) / len(self.history)
            var = sum((h - mean) ** 2 for h in self.history) / len(self.history)
            if var > 0:
                z = (x - mean) / math.sqrt(var)
                spike = z > self.z_threshold
        self.history.append(x)
        return z, spike


def zeropower_newtonschulz5(G, steps=5, eps=1e-7):
    """公式 (11.20)：五次迭代把矩阵的奇异值推向 1。"""
    if G.ndim != 2:
        raise ValueError("Muon 只处理二维参数")
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


def muon_update(param, momentum, grad, lr, mu=0.95, ns_steps=5):
    """公式 (11.19)：动量正交化后更新，返回新动量。"""
    momentum = mu * momentum + grad
    ortho = zeropower_newtonschulz5(momentum, steps=ns_steps)
    scale = max(1.0, param.size(0) / param.size(1)) ** 0.5
    with torch.no_grad():
        param.add_(ortho, alpha=-lr * scale)
    return momentum
