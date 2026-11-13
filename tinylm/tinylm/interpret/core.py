"""第 28 章：可解释性工具（tinylm/interpret/core.py）。

探针、注意力分析、激活修补与稀疏自编码器的最小实现。
所有函数只读或临时挂 hook，不修改模型定义。
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

def logit_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """(28.5)：目标 Token 的 logit 减去 log-sum-exp。"""
    if logits.dim() != 2 or targets.dim() != 1 or logits.size(0) != targets.size(0):
        raise ValueError("logits must be (B, V) and targets (B,)")
    rows = torch.arange(logits.size(0), device=logits.device)
    return (logits[rows, targets] - logits.logsumexp(dim=-1)).mean()

def linear_probe_accuracy(X_train: torch.Tensor, y_train: torch.Tensor,
                          X_test: torch.Tensor, y_test: torch.Tensor,
                          n_classes: int, steps: int = 300,
                          lr: float = 1e-2, seed: int = 0) -> float:
    """(28.1)：线性探针在留出集上的准确率。"""
    if X_train.size(0) != y_train.size(0) or X_test.size(0) != y_test.size(0):
        raise ValueError("features and labels must be aligned")
    if n_classes < 2 or steps < 1:
        raise ValueError("need n_classes >= 2 and steps >= 1")
    torch.manual_seed(seed)
    probe = nn.Linear(X_train.size(1), n_classes)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(steps):
        loss = F.cross_entropy(probe(X_train), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((probe(X_test).argmax(dim=-1) == y_test).float().mean())

def probe_selectivity(acc_probe: float, acc_control: float) -> float:
    """(28.2)：探针准确率减去对照任务准确率。"""
    return acc_probe - acc_control

@torch.no_grad()
def attention_maps(model: nn.Module, input_ids: torch.Tensor,
                   layer: int) -> torch.Tensor:
    """(28.3)：返回第 layer 层每个头的注意力矩阵 (B, H, T, T)。

    只支持 MHA/MQA/GQA 与 learned/sine/rope/alibi 位置编码；
    不处理 padding mask 与 MLA。
    """
    attn = model.blocks[layer].attn
    if not all(hasattr(attn, name) for name in
               ("q_proj", "k_proj", "n_heads", "n_kv_heads", "d_head")):
        raise NotImplementedError("attention_maps 只支持 MHA/MQA/GQA")
    store: dict[str, torch.Tensor] = {}

    def pre_hook(_module, inputs):
        store["h"] = inputs[0]

    handle = attn.register_forward_pre_hook(pre_hook)
    try:
        model(input_ids)
    finally:
        handle.remove()
    h = store["h"]
    B, T, _ = h.shape
    q = attn.q_proj(h).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
    k = attn.k_proj(h).view(B, T, attn.n_kv_heads, attn.d_head).transpose(1, 2)
    if attn.pos_mode == "rope":
        pos = torch.arange(T, device=h.device)
        q, k = attn.rope(q, k, pos)
    if getattr(attn, "repeat", 1) > 1:
        k = k.repeat_interleave(attn.repeat, dim=1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(attn.d_head)
    if attn.pos_mode == "alibi":
        scores = scores + attn.alibi(T, h.device)
    mask = torch.triu(torch.full((T, T), float("-inf"), device=h.device), diagonal=1)
    return (scores + mask).softmax(dim=-1)

@contextmanager
def ablate_heads(model: nn.Module, layer: int, heads: list[int] | tuple[int, ...]):
    """(28.4)：在前向时把指定注意力头的输出置零。"""
    attn = model.blocks[layer].attn
    d_head = attn.d_head

    def pre_hook(_module, inputs):
        x = inputs[0].clone()
        for head in heads:
            x[..., head * d_head:(head + 1) * d_head] = 0
        return (x,)

    handle = attn.out_proj.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()

@torch.no_grad()
def activation_at(model: nn.Module, module: nn.Module, position: int,
                  input_ids: torch.Tensor) -> torch.Tensor:
    """抓取 module 输出在 position 处的激活 (B, d)。"""
    store: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        z = output[0] if isinstance(output, tuple) else output
        store["z"] = z[:, position].detach().clone()

    handle = module.register_forward_hook(hook)
    try:
        model(input_ids)
    finally:
        handle.remove()
    return store["z"]

@contextmanager
def patched_output(module: nn.Module, position: int, value: torch.Tensor):
    """把 module 输出在 position 处替换为 value，作激活修补。"""
    def hook(_module, _inputs, output):
        z = output[0] if isinstance(output, tuple) else output
        z = z.clone()
        z[:, position] = value.to(dtype=z.dtype, device=z.device)
        return (z, *output[1:]) if isinstance(output, tuple) else z

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()

def recovery(patched: float, clean: float, corrupt: float) -> float:
    """(28.6)：修补后指标相对干净/损坏两端的恢复比例。"""
    denom = clean - corrupt
    if abs(denom) < 1e-12:
        raise ValueError("clean and corrupt scores are indistinguishable")
    return (patched - corrupt) / denom

class SparseAutoencoder(nn.Module):
    """(28.7)：ReLU 编码器 + 线性解码器的字典学习模型。"""

    def __init__(self, d_model: int, dict_size: int):
        super().__init__()
        if d_model < 1 or dict_size < 1:
            raise ValueError("d_model and dict_size must be positive")
        self.encoder = nn.Linear(d_model, dict_size)
        self.decoder = nn.Linear(dict_size, d_model)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = F.relu(self.encoder(x))
        return features, self.decoder(features)

def train_sae(X: torch.Tensor, dict_size: int = 128, l1: float = 0.02,
              steps: int = 1200, batch_size: int = 256,
              lr: float = 2e-3, seed: int = 0) -> SparseAutoencoder:
    """(28.7)：训练 SAE，并在每步后把解码器列归一化。"""
    if X.dim() != 2 or X.size(0) < 1:
        raise ValueError("X must be (N, d)")
    if l1 < 0 or steps < 1 or batch_size < 1:
        raise ValueError("invalid l1, steps or batch_size")
    torch.manual_seed(seed)
    sae = SparseAutoencoder(X.size(1), dict_size)
    opt = torch.optim.AdamW(sae.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        index = torch.randint(0, X.size(0), (min(batch_size, X.size(0)),), generator=gen)
        features, recon = sae(X[index])
        loss = F.mse_loss(recon, X[index]) + l1 * features.abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            norm = sae.decoder.weight.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
            sae.decoder.weight.data /= norm
    return sae

@torch.no_grad()
def sae_metrics(sae: SparseAutoencoder, X: torch.Tensor,
                eps: float = 1e-6) -> dict[str, float]:
    """(28.8)：重构误差、平均 L0 与死特征数。"""
    features, recon = sae(X)
    return {
        "mse": float(F.mse_loss(recon, X)),
        "l0": float((features > eps).float().sum(dim=-1).mean()),
        "dead": int(((features > eps).sum(dim=0) == 0).sum()),
    }

@torch.no_grad()
def feature_purity(sae: SparseAutoencoder, X: torch.Tensor,
                   labels: torch.Tensor, n_classes: int) -> tuple[float, list[float]]:
    """(28.9)：每个特征最大激活样本的标签纯度，再对特征等权平均。"""
    if X.size(0) != labels.size(0):
        raise ValueError("X and labels must be aligned")
    features, _ = sae(X)
    assigned = features.argmax(dim=-1)
    purities: list[float] = []
    for k in range(features.size(1)):
        mask = assigned == k
        support = int(mask.sum())
        if support == 0:
            continue
        counts = torch.bincount(labels[mask], minlength=n_classes)
        purities.append(float(counts.max()) / support)
    return (sum(purities) / len(purities) if purities else 0.0), purities

def project_out(x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """把 direction 方向从 x 中投影掉。"""
    unit = direction / direction.norm().clamp_min(1e-12)
    return x - (x @ unit).unsqueeze(-1) * unit

@contextmanager
def ablated_direction(module: nn.Module, position: int, direction: torch.Tensor):
    """在 module 输入的 position 处投影掉 direction，用于特征消融。"""
    def pre_hook(_module, inputs):
        x = inputs[0].clone()
        x[:, position] = project_out(x[:, position], direction)
        return (x,)

    handle = module.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()
