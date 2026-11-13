"""第 15 章：LoRA、QLoRA 与参数账本（tinylm/posttrain/lora.py）。"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# QLoRA 使用的 16 级 NF4 码本（Dettmers et al., 2023，附录 E）
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495,
    0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
])

# ---------------------------------------------------------------- config
@dataclass(frozen=True)
class LoraConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "out_proj",
        "w_gate", "w_up", "w_down", "fc_in", "fc_out",
    )
    bias: str = "none"              # none | all | lora_only

    def __post_init__(self):
        assert self.r >= 1
        assert self.alpha > 0
        assert 0.0 <= self.dropout < 1.0
        assert self.bias in ("none", "all", "lora_only")

# ---------------------------------------------------------------- lora.py
class LoRALinear(nn.Module):
    """公式 (15.1)：y = W x + (α/r) B A x。"""

    def __init__(self, base: nn.Linear, cfg: LoraConfig):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        if cfg.bias in ("all", "lora_only") and base.bias is not None:
            base.bias.requires_grad_(True)
        self.r = cfg.r
        self.scaling = cfg.alpha / cfg.r
        self.dropout = nn.Dropout(cfg.dropout)
        self.lora_A = nn.Parameter(torch.empty(cfg.r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, cfg.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.merged = False
        self._delta = None

    def forward(self, x):
        out = self.base(x)
        if self.merged:
            return out
        delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return out + delta * self.scaling                       # (15.1)

    @torch.no_grad()
    def merge(self):
        """把 B A 写回基座权重，推理时不再计算低秩分支。"""
        if self.merged:
            return
        self._delta = (self.lora_B @ self.lora_A) * self.scaling  # (15.2)
        self.base.weight.add_(self._delta)
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if not self.merged:
            return
        self.base.weight.sub_(self._delta)
        self.merged = False
        self._delta = None

def _resolve_target(name, cfg):
    return name in cfg.target_modules or name.split(".")[-1] in cfg.target_modules

def apply_lora(model, cfg: LoraConfig):
    """冻结全部参数，再把目标 nn.Linear 换成 LoRALinear。"""
    for p in model.parameters():
        p.requires_grad_(False)

    targets = [name for name, m in model.named_modules()
               if isinstance(m, nn.Linear) and _resolve_target(name, cfg)]
    for name in targets:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        leaf = name.rsplit(".", 1)[-1]
        setattr(parent, leaf, LoRALinear(getattr(parent, leaf), cfg))

    if cfg.bias == "all":
        for name, p in model.named_parameters():
            if name.endswith("bias"):
                p.requires_grad_(True)
    return model

def lora_parameters(model):
    for name, p in model.named_parameters():
        if name.rsplit(".", 1)[-1] in ("lora_A", "lora_B"):
            yield name, p

def lora_state_dict(model):
    """只保存适配器，支持同一基座挂多个任务适配器。"""
    return {name: p.detach().clone() for name, p in lora_parameters(model)}

def load_lora_state_dict(model, state):
    current = set(lora_state_dict(model))
    missing = current - set(state)
    extra = set(state) - current
    if missing or extra:
        raise KeyError(f"适配器键不匹配：缺 {sorted(missing)[:3]}，多 {sorted(extra)[:3]}")
    with torch.no_grad():
        for name, p in lora_parameters(model):
            p.copy_(state[name])

def trainable_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable,
            "ratio": trainable / max(total, 1)}

def lora_module_params(model, cfg: LoraConfig):
    """公式 (15.3)：每个目标模块 r(d_in + d_out) 个参数。"""
    rows = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and _resolve_target(name, cfg):
            rows.append((name, cfg.r * (m.in_features + m.out_features)))
    return rows

def state_bytes(model, *, param=4, grad=4, optim=8):
    """训练静态状态：可训练参数 (param+grad+optim)，冻结参数只有 param。

    若模型做过 NF4 量化，被量化的基座按 4 比特记账，其余冻结参数仍是 fp32。
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    quant = getattr(model, "_nf4", None)
    q_numel = quant["numel"] if quant else 0
    q_bytes = quant["bytes"] if quant else 0
    train_b = trainable * (param + grad + optim)
    frozen_b = (frozen - q_numel) * param + q_bytes
    return {"trainable": trainable, "frozen": frozen,
            "train_bytes": train_b, "frozen_bytes": frozen_b,
            "total_bytes": train_b + frozen_b}

# ---------------------------------------------------------------- qlora.py
def _blocks(w, block_size):
    flat = w.reshape(-1)
    n = flat.numel()
    pad = (-n) % block_size
    flat = F.pad(flat, (0, pad))
    return flat.view(-1, block_size), n

def nf4_quantize(w, block_size=64):
    """按块 absmax 归一化后，投影到最近的 NF4 码字。"""
    levels = NF4_LEVELS.to(w.device)
    blocks, n = _blocks(w.detach().float(), block_size)
    absmax = blocks.abs().amax(dim=1).clamp_min(1e-8)
    norm = blocks / absmax[:, None]
    codes = (norm.unsqueeze(-1) - levels).abs().argmin(-1).to(torch.uint8)
    deq = levels[codes.long()] * absmax[:, None]
    return deq.reshape(-1)[:n].view(w.shape), codes, absmax

def nf4_dequantize(codes, absmax, shape):
    levels = NF4_LEVELS.to(codes.device)
    deq = levels[codes.long()] * absmax[:, None]
    return deq.reshape(-1)[:math.prod(shape)].view(shape)

def nf4_storage_bytes(numel, block_size=64, double_quant=True):
    """公式 (15.6)：4 比特权重 + 每块尺度（双层量化时再压到 8 比特）。"""
    weight_bytes = math.ceil(numel / 2)
    n_blocks = math.ceil(numel / block_size)
    if double_quant:
        const_bytes = n_blocks + math.ceil(n_blocks / 256) * 4
    else:
        const_bytes = n_blocks * 4
    total = weight_bytes + const_bytes
    return {"weight": weight_bytes, "constant": const_bytes,
            "total": total, "bits_per_param": 8 * total / numel}

@torch.no_grad()
def quantize_lora_bases(model, block_size=64, double_quant=True):
    """把 LoRALinear 的基座权重做 NF4 量化-反量化（模拟 QLoRA 的数值）。"""
    numel = total = 0
    for m in model.modules():
        if not isinstance(m, LoRALinear):
            continue
        w = m.base.weight
        deq, codes, absmax = nf4_quantize(w.data, block_size)
        w.data.copy_(deq)
        w.requires_grad_(False)
        m.nf4_codes = codes
        m.nf4_scales = absmax
        numel += w.numel()
        total += nf4_storage_bytes(w.numel(), block_size, double_quant)["total"]
    info = {"numel": numel, "bytes": total,
            "bits_per_param": 8 * total / max(numel, 1)}
    model._nf4 = info
    return info
