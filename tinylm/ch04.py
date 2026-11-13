"""第 4 章：位置编码与长度外推（tinylm/position.py 与 model.py 的增量）。"""
import math
import sys
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from ch03 import CausalLMOutput, LayerNorm, MLP, ModelConfig as BaseConfig


# ---------------------------------------------------------------- config.py
@dataclass(frozen=True)
class RoPEScalingConfig:
    """RoPE 的长度外推配置。factor = 目标长度 / 训练长度。"""
    method: str = "none"          # none | pi | ntk | yarn
    factor: float = 1.0


@dataclass(frozen=True)
class ModelConfig(BaseConfig):
    # 第 3 章字段全部继承；本章只增加一个可选开关
    rope_scaling: RoPEScalingConfig = field(default_factory=RoPEScalingConfig)


# ---------------------------------------------------------------- position.py
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


# ---------------------------------------------------------------- attention.py
class PositionAwareAttention(nn.Module):
    """在第 3 章 MultiHeadAttention 上加入位置分派。"""

    def __init__(self, cfg):
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        self.n_heads = h
        self.d_head = d // h
        self.pos_mode = cfg.pos_encoding
        self.q_proj = nn.Linear(d, d, bias=cfg.bias)
        self.k_proj = nn.Linear(d, d, bias=cfg.bias)
        self.v_proj = nn.Linear(d, d, bias=cfg.bias)
        self.out_proj = nn.Linear(d, d, bias=cfg.bias)
        if self.pos_mode == "rope":
            self.rope = RoPE(self.d_head, cfg.rope_theta, cfg.rope_scaling,
                             cfg.max_seq_len)
        if self.pos_mode == "alibi":
            self.alibi = ALiBi(h)
        mask = torch.triu(
            torch.full((cfg.max_seq_len, cfg.max_seq_len), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x, attention_mask=None, position_ids=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if self.pos_mode == "rope":
            q, k = self.rope(q, k, position_ids)              # 公式 (4.7)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        if self.pos_mode == "rope" and self.rope.temp != 1.0:
            scores = scores / math.sqrt(self.rope.temp)       # 公式 (4.18)
        if self.pos_mode == "alibi":
            scores = scores + self.alibi(T, x.device)         # 公式 (4.11)

        if T > self.causal_mask.size(0):                      # 允许 RoPE 超出表长
            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device,
                           dtype=scores.dtype),
                diagonal=1,
            )
        else:
            mask = self.causal_mask[:T, :T]
        scores = scores + mask
        if attention_mask is not None:
            pad = attention_mask[:, None, None, :] == 0
            scores = scores.masked_fill(pad, float("-inf"))

        attn = scores.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)
        return self.out_proj(out)


# ---------------------------------------------------------------- block.py
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = LayerNorm(cfg.d_model, cfg.norm_eps, cfg.bias)
        self.attn = PositionAwareAttention(cfg)
        self.ln2 = LayerNorm(cfg.d_model, cfg.norm_eps, cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, attention_mask=None, position_ids=None):
        x = x + self.attn(self.ln1(x), attention_mask, position_ids)
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------- model.py
class TinyLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pos_mode = cfg.pos_encoding
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if self.pos_mode == "learned":
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        elif self.pos_mode == "sine":
            self.register_buffer(
                "sin_pe", sinusoidal_positions(cfg.max_seq_len, cfg.d_model),
                persistent=False,
            )
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_f = LayerNorm(cfg.d_model, cfg.norm_eps, cfg.bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc_out.weight"):
                nn.init.normal_(p, mean=0.0, std=std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, *, attention_mask=None, position_ids=None,
                past_kv=None, use_cache=False, labels=None, logits_to_keep=None):
        B, T = input_ids.shape
        if self.pos_mode in ("learned", "sine"):
            assert T <= self.cfg.max_seq_len, "序列长度超过 max_seq_len"
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device)

        x = self.tok_emb(input_ids)
        if self.pos_mode == "learned":
            x = x + self.pos_emb(position_ids)                # 第 3 章式 (3.18)
        elif self.pos_mode == "sine":
            x = x + self.sin_pe[position_ids]                 # 公式 (4.1)
        for block in self.blocks:
            x = block(x, attention_mask, position_ids)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(logits=logits, loss=loss)


# ---------------------------------------------------------------- 实验
def copy_task(B, T, lag, vocab, gen):
    """前 lag 个 Token 随机，之后每个 Token 复制 lag 步之前的值。"""
    seq = torch.randint(0, vocab, (B, lag), generator=gen)
    for t in range(lag, T):
        seq = torch.cat([seq, seq[:, t - lag:t - lag + 1]], dim=1)
    return seq


def make_labels(seq, lag):
    inp, tgt = seq[:, :-1], seq[:, 1:].clone()
    tgt[:, :lag - 1] = -100          # 前 lag-1 个目标落在随机前缀内
    return inp, tgt


def train_model(pos_encoding, steps=400, seed=0, train_len=32, lag=8):
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=16, d_model=64, n_layers=2, n_heads=4, d_ff=256,
        max_seq_len=128, pos_encoding=pos_encoding,
    )
    model = TinyLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    gen = torch.Generator().manual_seed(123)
    model.train()
    for _ in range(steps):
        seq = copy_task(64, train_len, lag, 16, gen)
        inp, tgt = make_labels(seq, lag)
        out = model(inp, labels=tgt)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    return model


@torch.no_grad()
def evaluate(model, T, lag=8, B=256, seed=999):
    gen = torch.Generator().manual_seed(seed + T)
    seq = copy_task(B, T, lag, 16, gen)
    inp, tgt = make_labels(seq, lag)
    model.eval()
    out = model(inp, labels=tgt)
    mask = tgt != -100
    acc = (out.logits.argmax(-1)[mask] == tgt[mask]).float().mean().item()
    return out.loss.item(), acc


def scaling_table():
    d_head, theta, s, train_len = 64, 10000.0, 4.0, 8192
    print("RoPE 缩放：d_head=64, theta=10000, s=4, L_train=8192")
    print(f"{'i':>2s}  " + "  ".join(f"{m:>10s}" for m in
                                     ("none", "PI", "NTK", "YaRN")) + "   （有效波长）")
    for i in (0, 8, 16, 24, 31):
        vals = []
        for method in ("none", "pi", "ntk", "yarn"):
            inv = rope_inv_freq(d_head, theta,
                                RoPEScalingConfig(method=method, factor=s),
                                train_len)
            lam = (2 * math.pi / inv[i]).item()
            if method == "pi":
                lam *= s
            vals.append(lam)
        print(f"{i:2d}  " + "  ".join(f"{v:10.1f}" for v in vals))


def main():
    scaling_table()
    print("\n外推实验：训练长度 32，lag=8；损失/准确率只在可预测位置统计")
    lengths = [32, 48, 64, 96, 128]
    print(f"{'编码':<8s} " + " ".join(f"{'T=' + str(T):>13s}" for T in lengths))
    for pe in ("learned", "sine", "rope", "nope", "alibi"):
        model = train_model(pe, steps=400)
        cells = [evaluate(model, T) for T in lengths]
        print(f"{pe:<8s} " + " ".join(f"{loss:7.4f}/{acc:.3f}"
                                      for loss, acc in cells))


if __name__ == "__main__":
    main()
