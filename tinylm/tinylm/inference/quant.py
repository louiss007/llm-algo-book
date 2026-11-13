"""第 23 章：量化（tinylm/inference/quant.py）。"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class QuantConfig:
    """量化参数，不进入 ``ModelConfig``。"""

    bits: int = 8
    granularity: str = "channel"       # tensor | channel | group
    group_size: int = 64
    symmetric: bool = True
    eps: float = 1e-8

    def __post_init__(self):
        assert self.bits in (2, 3, 4, 5, 6, 7, 8)
        assert self.granularity in ("tensor", "channel", "group")
        assert self.group_size >= 1
        if self.granularity == "group":
            assert self.group_size >= 2

@dataclass
class QuantizedTensor:
    q: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    cfg: QuantConfig
    shape: tuple

    def dequantize(self):
        return dequantize(self.q, self.scale, self.zero, self.cfg, self.shape)

def qrange(cfg):
    """(23.1)：对称量化的整数范围，或非对称量化的无符号范围。"""
    if cfg.symmetric:
        qmax = 2 ** (cfg.bits - 1) - 1
        return -qmax, qmax
    return 0, 2 ** cfg.bits - 1

def _group_view(x, cfg):
    """把最后两维整理成 (行, 组, 组内元素)，并记录还原信息。"""
    orig = x.shape
    x2 = x.reshape(-1, orig[-1])
    n = x2.size(1)
    if cfg.granularity == "tensor":
        return x2.reshape(1, 1, -1), (orig, n, 0)
    if cfg.granularity == "channel":
        return x2.unsqueeze(1), (orig, n, 0)
    pad = (-n) % cfg.group_size
    if pad:
        x2 = F.pad(x2, (0, pad))
    return x2.reshape(x2.size(0), -1, cfg.group_size), (orig, n, pad)

def _ungroup(g, meta):
    """把分组结果还原成原始形状。"""
    orig, n, pad = meta
    x2 = g.reshape(-1, n + pad)
    if pad:
        x2 = x2[..., :n]
    return x2.reshape(orig)

def quant_params(x, cfg):
    """(23.4)–(23.5)：按粒度计算缩放因子与零点。"""
    g, _ = _group_view(x.detach().float(), cfg)
    qmin, qmax = qrange(cfg)
    if cfg.symmetric:
        amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(cfg.eps)
        scale = amax / qmax
        zero = torch.zeros_like(scale)
    else:
        lo = g.amin(dim=-1, keepdim=True)
        hi = g.amax(dim=-1, keepdim=True)
        scale = ((hi - lo) / (qmax - qmin)).clamp_min(cfg.eps)
        zero = torch.round(qmin - lo / scale).clamp(qmin, qmax)
    return scale.squeeze(-1), zero.squeeze(-1)

def quantize(x, cfg):
    """(23.2)：量化到整数网格，返回 (q, scale, zero)。"""
    scale, zero = quant_params(x, cfg)
    g, meta = _group_view(x.detach().float(), cfg)
    qmin, qmax = qrange(cfg)
    q = torch.round(g / scale.unsqueeze(-1) + zero.unsqueeze(-1))
    q = q.clamp(qmin, qmax)
    dtype = torch.int8 if cfg.symmetric else torch.uint8
    q = _ungroup(q, meta).to(dtype)
    return q, scale, zero

def dequantize(q, scale, zero, cfg, shape):
    """(23.3)：把整数码点还原成实数。"""
    orig = tuple(shape)
    x2 = q.float().reshape(-1, orig[-1])
    n = x2.size(1)
    if cfg.granularity == "tensor":
        out = (x2 - zero.reshape(1)) * scale.reshape(1)
    elif cfg.granularity == "channel":
        out = (x2 - zero.reshape(-1, 1)) * scale.reshape(-1, 1)
    else:
        pad = (-n) % cfg.group_size
        if pad:
            x2 = F.pad(x2, (0, pad))
        g = x2.reshape(x2.size(0), -1, cfg.group_size)
        out = (g - zero.unsqueeze(-1)) * scale.unsqueeze(-1)
        out = out.reshape(x2.size(0), -1)
        if pad:
            out = out[:, :n]
    return out.reshape(orig)

def fake_quant(x, cfg):
    """模拟量化：在浮点里做“量化—反量化”，用于误差分析与 QAT。"""
    q, scale, zero = quantize(x, cfg)
    return dequantize(q, scale, zero, cfg, x.shape)

def sqnr_db(x, x_hat):
    """(23.8)：信号功率与量化噪声功率之比，单位 dB。"""
    signal = x.float().pow(2).mean()
    noise = (x.float() - x_hat.float()).pow(2).mean()
    if noise <= 0:
        return float("inf")
    return float(10.0 * torch.log10(signal / noise))

def theoretical_sqnr(bits):
    """(23.9)：满量程正弦波下的 6.02b+1.76 dB 上界。"""
    return 6.0206 * bits + 1.7609

def quantization_error(x, x_hat):
    x = x.float()
    x_hat = x_hat.float()
    diff = x - x_hat
    cos = F.cosine_similarity(x.flatten(), x_hat.flatten(), dim=0)
    return {
        "mse": float(diff.pow(2).mean()),
        "max_abs": float(diff.abs().max()),
        "cosine": float(cos),
        "sqnr_db": sqnr_db(x, x_hat),
    }

def storage_bits(cfg, n_cols):
    """(23.10)：每个权重的平均比特数，含缩放因子与零点开销。"""
    assert n_cols >= 1
    per_scale = 16 + (0 if cfg.symmetric else 8)
    if cfg.granularity in ("tensor", "channel"):
        return cfg.bits + per_scale / n_cols
    return cfg.bits + per_scale / min(cfg.group_size, n_cols)

def compression_ratio(cfg, n_cols):
    """(23.11)：相对 FP16 的存储压缩比。"""
    return 16.0 / storage_bits(cfg, n_cols)

def pack_int4(q):
    """把 [-8,7] 的 int4 码点两两打包进一个 uint8。"""
    flat = q.to(torch.int32).reshape(-1)
    if flat.numel() % 2:
        flat = F.pad(flat, (0, 1))
    lo = flat[0::2] & 0x0F
    hi = (flat[1::2] & 0x0F) << 4
    return (lo | hi).to(torch.uint8)

def unpack_int4(packed, numel):
    """还原 int4 码点；超过 ``numel`` 的填充被截掉。"""
    p = packed.to(torch.int32).reshape(-1)
    lo = (p & 0x0F)
    hi = (p >> 4) & 0x0F
    out = torch.empty(p.numel() * 2, dtype=torch.int32)
    out[0::2] = lo
    out[1::2] = hi
    out = torch.where(out >= 8, out - 16, out)
    return out[:numel].to(torch.int8)

def quantized_linear(x, weight, cfg, bias=None):
    """W-only 量化：权重反量化后与浮点激活做矩阵乘。"""
    return F.linear(x, fake_quant(weight, cfg), bias)

def gptq_quantize(weight, hessian, cfg, damp=0.01):
    """(23.13)：逐列量化并用 Hessian 逆补偿尚未处理的列。"""
    assert cfg.granularity == "channel" and cfg.symmetric
    W = weight.detach().float().clone()
    out_f, in_f = W.shape
    H = hessian.detach().float().clone()
    assert H.shape == (in_f, in_f)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += damp * torch.diag(H).mean() * torch.eye(in_f, dtype=H.dtype)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    qmin, qmax = qrange(cfg)
    scale = W.abs().amax(dim=1, keepdim=True).clamp_min(cfg.eps) / qmax
    Q = torch.zeros_like(W)
    for i in range(in_f):
        w = W[:, i:i + 1]
        q = torch.round(w / scale).clamp(qmin, qmax) * scale
        Q[:, i:i + 1] = q
        if i + 1 < in_f:
            err = (w - q) / Hinv[i, i]          # (23.13)
            W[:, i + 1:] -= err @ Hinv[i, i + 1:].unsqueeze(0)
    return Q, scale.squeeze(1)

def smooth_scale(x, weight, alpha=0.5):
    """(23.15)：把激活离群值迁移到权重上的逐通道缩放。"""
    assert 0.0 <= alpha <= 1.0
    ax = x.detach().float().abs().amax(dim=0).clamp_min(1e-8)
    aw = weight.detach().float().abs().amax(dim=0).clamp_min(1e-8)
    return (ax.pow(alpha) / aw.pow(1.0 - alpha)).clamp(1e-2, 1e2)

def smooth_quant(x, weight, alpha=0.5):
    """(23.15)：返回等价变换后的 (x/s, W·diag(s))。"""
    s = smooth_scale(x, weight, alpha)
    return x / s, weight * s

def kv_quantize(tensor, cfg, per_channel=False):
    """KV 量化：K 按通道（沿 T），V 按 Token（沿 D）。"""
    if per_channel:
        t = tensor.transpose(-1, -2)
        q, scale, zero = quantize(t, cfg)
        return dequantize(q, scale, zero, cfg, t.shape).transpose(-1, -2)
    q, scale, zero = quantize(tensor, cfg)
    return dequantize(q, scale, zero, cfg, tensor.shape)

def attention_sqnr(q, k, v, k_cfg, v_cfg, per_channel_k=True):
    """端到端注意力输出的信噪比，用于比较 KV 量化方案。"""
    def attend(kk, vv):
        s = q @ kk.transpose(-1, -2) / math.sqrt(q.size(-1))
        p = torch.softmax(s, dim=-1)
        return p @ vv

    ref = attend(k, v)
    k_hat = kv_quantize(k, k_cfg, per_channel=per_channel_k)
    v_hat = kv_quantize(v, v_cfg)
    return sqnr_db(ref, attend(k_hat, v_hat))
