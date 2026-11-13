"""第 6 章：注意力机制的变体（tinylm/attention.py、config.py 的增量）。"""
import math
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from ch03 import CausalLMOutput
from ch04 import ALiBi, RoPE, sinusoidal_positions
from ch05 import ModelConfig as Config5, make_ffn, make_norm


# ---------------------------------------------------------------- config.py
@dataclass(frozen=True)
class ModelConfig(Config5):
    attention: str = "mha"          # mha | mqa | gqa | mla
    n_kv_heads: int | None = None   # GQA 的 KV 头数
    d_c: int = 128                  # MLA 的 KV 压缩维度
    d_head_rope: int = 32           # MLA 解耦 RoPE 的键维度

    def __post_init__(self):
        super().__post_init__()
        assert self.attention in ("mha", "mqa", "gqa", "mla")
        if self.attention == "mla":
            assert self.d_c > 0 and self.d_head_rope > 0
            return
        if self.attention == "mha":
            n_kv = self.n_heads
        elif self.attention == "mqa":
            n_kv = 1
        else:
            assert self.n_kv_heads is not None, "GQA 必须指定 n_kv_heads"
            n_kv = self.n_kv_heads
        object.__setattr__(self, "n_kv_heads", n_kv)
        assert self.n_heads % n_kv == 0, "n_heads 必须是 n_kv_heads 的整数倍"


# ---------------------------------------------------------------- attention.py
class VariantAttention(nn.Module):
    """公式 (6.2)、(6.3)：Q 有 n_heads 个头，K/V 只有 n_kv_heads 个头。"""

    def __init__(self, cfg):
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        self.n_heads = h
        self.n_kv_heads = cfg.n_kv_heads
        self.d_head = d // h
        self.repeat = h // cfg.n_kv_heads
        self.pos_mode = cfg.pos_encoding
        self.q_proj = nn.Linear(d, d, bias=cfg.bias)
        self.k_proj = nn.Linear(d, cfg.n_kv_heads * self.d_head, bias=cfg.bias)
        self.v_proj = nn.Linear(d, cfg.n_kv_heads * self.d_head, bias=cfg.bias)
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
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.pos_mode == "rope":
            q, k = self.rope(q, k, position_ids)
        if self.repeat > 1:
            k = k.repeat_interleave(self.repeat, dim=1)          # 公式 (6.2)
            v = v.repeat_interleave(self.repeat, dim=1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        if self.pos_mode == "rope" and self.rope.temp != 1.0:
            scores = scores / math.sqrt(self.rope.temp)
        if self.pos_mode == "alibi":
            scores = scores + self.alibi(T, x.device)

        if T > self.causal_mask.size(0):
            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device,
                           dtype=scores.dtype),
                diagonal=1,
            )
        else:
            mask = self.causal_mask[:T, :T]
        scores = scores + mask
        if attention_mask is not None:
            if attention_mask.dim() == 4:                     # 块对角文档掩码
                scores = scores.masked_fill(attention_mask == 0, float("-inf"))
            else:
                pad = attention_mask[:, None, None, :] == 0
                scores = scores.masked_fill(pad, float("-inf"))

        attn = scores.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)
        return self.out_proj(out)


class LatentAttention(nn.Module):
    """公式 (6.8)–(6.10)：KV 联合低秩压缩 + 解耦 RoPE。"""

    def __init__(self, cfg):
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        self.n_heads = h
        self.d_head = d // h
        self.d_c = cfg.d_c
        self.d_rope = cfg.d_head_rope
        self.pos_mode = cfg.pos_encoding
        self.q_proj = nn.Linear(d, d, bias=cfg.bias)
        self.kv_down = nn.Linear(d, cfg.d_c, bias=cfg.bias)      # (6.8)
        self.k_up = nn.Linear(cfg.d_c, d, bias=cfg.bias)         # (6.9)
        self.v_up = nn.Linear(cfg.d_c, d, bias=cfg.bias)         # (6.9)
        self.q_rope = nn.Linear(d, h * cfg.d_head_rope, bias=cfg.bias)
        self.k_rope = nn.Linear(d, cfg.d_head_rope, bias=cfg.bias)
        self.out_proj = nn.Linear(d, d, bias=cfg.bias)
        if self.pos_mode == "rope":
            self.rope = RoPE(cfg.d_head_rope, cfg.rope_theta,
                             cfg.rope_scaling, cfg.max_seq_len)
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
        c = self.kv_down(x)                                       # (6.8)
        k = self.k_up(c).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_up(c).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q_r = self.q_rope(x).view(B, T, self.n_heads, self.d_rope).transpose(1, 2)
        k_r = self.k_rope(x).view(B, T, 1, self.d_rope).transpose(1, 2)

        if self.pos_mode == "rope":
            q_r, k_r = self.rope(q_r, k_r, position_ids)          # (6.10)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        scores = scores + q_r @ k_r.transpose(-2, -1) / math.sqrt(self.d_rope)
        if self.pos_mode == "alibi":
            scores = scores + self.alibi(T, x.device)

        if T > self.causal_mask.size(0):
            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device,
                           dtype=scores.dtype),
                diagonal=1,
            )
        else:
            mask = self.causal_mask[:T, :T]
        scores = scores + mask
        if attention_mask is not None:
            if attention_mask.dim() == 4:                     # 块对角文档掩码
                scores = scores.masked_fill(attention_mask == 0, float("-inf"))
            else:
                pad = attention_mask[:, None, None, :] == 0
                scores = scores.masked_fill(pad, float("-inf"))

        attn = scores.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)
        return self.out_proj(out)


def make_attention(cfg):
    if cfg.attention == "mla":
        return LatentAttention(cfg)
    return VariantAttention(cfg)


# ---------------------------------------------------------------- block.py
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.parallel = cfg.parallel_block
        self.norm1 = make_norm(cfg)
        self.attn = make_attention(cfg)
        self.norm2 = None if self.parallel else make_norm(cfg)
        self.mlp = make_ffn(cfg)

    def forward(self, x, attention_mask=None, position_ids=None):
        if self.parallel:
            h = self.norm1(x)
            return x + self.attn(h, attention_mask, position_ids) + self.mlp(h)
        x = x + self.attn(self.norm1(x), attention_mask, position_ids)
        return x + self.mlp(self.norm2(x))


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
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "fc_out.weight", "w_down.weight")):
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
            x = x + self.pos_emb(position_ids)
        elif self.pos_mode == "sine":
            x = x + self.sin_pe[position_ids]
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


# ---------------------------------------------------------------- 账本
def attention_params(cfg):
    """公式 (6.5)、(6.13)：注意力子层的参数量。"""
    d, h = cfg.d_model, cfg.n_heads
    if cfg.attention == "mla":
        p = 2 * d * d + 3 * d * cfg.d_c + d * cfg.d_head_rope * (h + 1)
        if cfg.bias:
            p += 4 * d + cfg.d_c + (h + 1) * cfg.d_head_rope
        return p
    d_kv = cfg.n_kv_heads * (d // h)
    p = 2 * d * d + 2 * d * d_kv
    if cfg.bias:
        p += 2 * d + 2 * d_kv
    return p


def exact_params(cfg):
    """把 (6.5) 代入第 5 章 (5.11) 的整模型账本。"""
    d, f, V, L = cfg.d_model, cfg.d_ff, cfg.vocab_size, cfg.max_seq_len
    attn = attention_params(cfg)
    if cfg.ffn == "swiglu":
        ffn = 3 * d * f + ((2 * f + d) if cfg.bias else 0)
    else:
        ffn = 2 * d * f + ((f + d) if cfg.bias else 0)
    norm_p = d + (d if cfg.norm == "layernorm" and cfg.bias else 0)
    layer_norms = norm_p if cfg.parallel_block else 2 * norm_p
    per_layer = attn + ffn + layer_norms
    emb = V * d * (1 if cfg.tie_embeddings else 2)
    pos = L * d if cfg.pos_encoding == "learned" else 0
    return cfg.n_layers * per_layer + norm_p + emb + pos


def kv_cache_bytes(cfg, T, batch=1, dtype_bytes=2):
    """公式 (6.6)、(6.11)：推理时 KV Cache 的字节数。"""
    if cfg.attention == "mla":
        per_token_layer = (cfg.d_c + cfg.d_head_rope) * dtype_bytes
    else:
        per_token_layer = 2 * cfg.n_kv_heads * (cfg.d_model // cfg.n_heads) * dtype_bytes
    return batch * T * cfg.n_layers * per_token_layer


def mla_absorption_error(cfg, seed=0):
    """公式 (6.12)：内容通路吸收前后的输出误差。"""
    torch.manual_seed(seed)
    attn = LatentAttention(cfg).eval()
    x = torch.randn(2, 7, cfg.d_model)
    H, D, dc = attn.n_heads, attn.d_head, attn.d_c
    with torch.no_grad():
        q = attn.q_proj(x).view(2, 7, H, D).transpose(1, 2)
        c = attn.kv_down(x)
        k = attn.k_up(c).view(2, 7, H, D).transpose(1, 2)
        v = attn.v_up(c).view(2, 7, H, D).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D)
        mask = attn.causal_mask[:7, :7]
        w = (scores + mask).softmax(dim=-1)
        out = (w @ v).transpose(1, 2).reshape(2, 7, -1)
        out = attn.out_proj(out)

        Wq = attn.q_proj.weight.view(H, D, cfg.d_model)
        Wuk = attn.k_up.weight.view(H, D, dc)
        q_abs = torch.einsum("hdc,hde->hec", Wuk, Wq)
        q_c = torch.einsum("hec,bte->bhtc", q_abs, x)
        s_abs = torch.einsum("bhtc,bjc->bhtj", q_c, c) / math.sqrt(D)
        w_abs = (s_abs + mask).softmax(dim=-1)
        c_w = torch.einsum("bhtj,bjc->bhtc", w_abs, c)
        Wo = attn.out_proj.weight.view(cfg.d_model, H, D)
        Wuv = attn.v_up.weight.view(H, D, dc)
        O_h = torch.stack([Wo[:, h, :] @ Wuv[h] for h in range(H)])
        out_abs = torch.einsum("hoc,bhtc->bto", O_h, c_w)
    return (out - out_abs).abs().max().item()


def human(nbytes):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:.2f} {unit}" if unit != "B" else f"{nbytes:.0f} B"
        nbytes /= 1024


def ledger():
    common = dict(vocab_size=32000, d_model=4096, n_layers=32, n_heads=32,
                  d_ff=11008, max_seq_len=8192, norm="rmsnorm", ffn="swiglu",
                  bias=False, pos_encoding="rope")
    cfgs = {
        "MHA": ModelConfig(attention="mha", **common),
        "GQA-8": ModelConfig(attention="gqa", n_kv_heads=8, **common),
        "MQA": ModelConfig(attention="mqa", **common),
        "MLA": ModelConfig(attention="mla", d_c=512, d_head_rope=64, **common),
    }
    print("注意力变体账本（d=4096, L=32, H=32, T=8192, FP16）")
    print(f"{'变体':<8s}{'注意力参数':>14s}{'总参数':>16s}"
          f"{'每Token缓存':>14s}{'8K缓存':>12s}")
    for name, cfg in cfgs.items():
        per_token = kv_cache_bytes(cfg, 1)
        total8k = kv_cache_bytes(cfg, 8192)
        print(f"{name:<8s}{attention_params(cfg):>14,d}{exact_params(cfg):>16,d}"
              f"{human(per_token):>14s}{human(total8k):>12s}")


def industrial_cases():
    print("\n工业配置的 KV Cache（FP16）")
    cases = [
        ("Llama 2 7B MHA", 32, 32, 128, 4096, None),
        ("Llama 3 8B GQA-8", 32, 8, 128, 8192, None),
        ("PaLM 540B MQA", 118, 1, 384, 2048, None),
        ("DeepSeek-V2 MLA", 60, None, None, 8192, (512, 64)),
    ]
    for name, layers, n_kv, d_head, T, mla in cases:
        if mla is not None:
            per_token = layers * (mla[0] + mla[1]) * 2
        else:
            per_token = layers * 2 * n_kv * d_head * 2
        print(f"{name:<20s} 每Token {human(per_token):>10s}"
              f"   T={T:<5d} 总计 {human(per_token * T):>10s}")


def absorption_demo():
    cfg = ModelConfig(vocab_size=32, d_model=64, n_layers=2, n_heads=8,
                      d_ff=128, max_seq_len=64, attention="mla",
                      d_c=16, d_head_rope=8, bias=False)
    err = mla_absorption_error(cfg)
    print(f"\nMLA 吸收等价性（内容通路）：最大输出误差 {err:.2e}")


# ---------------------------------------------------------------- 复制任务消融
def copy_task(B, T, lag, vocab, gen):
    seq = torch.randint(0, vocab, (B, lag), generator=gen)
    for t in range(lag, T):
        seq = torch.cat([seq, seq[:, t - lag:t - lag + 1]], dim=1)
    return seq


def make_labels(seq):
    return seq[:, :-1], seq[:, 1:].clone()


def train_one(cfg, steps=300, seed=0, train_len=32, lag=8):
    torch.manual_seed(seed)
    model = TinyLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    gen = torch.Generator().manual_seed(1000 + seed)
    model.train()
    for _ in range(steps):
        seq = copy_task(64, train_len, lag, cfg.vocab_size, gen)
        inp, tgt = make_labels(seq)
        out = model(inp, labels=tgt)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    return model


@torch.no_grad()
def evaluate(model, T=32, lag=8, B=256, seed=7):
    gen = torch.Generator().manual_seed(seed)
    seq = copy_task(B, T, lag, 16, gen)
    inp, tgt = make_labels(seq)
    model.eval()
    out = model(inp, labels=tgt)
    acc = (out.logits.argmax(-1) == tgt).float().mean().item()
    return out.loss.item(), acc


def ablation():
    common = dict(vocab_size=16, d_model=64, n_layers=2, n_heads=4,
                  d_ff=256, max_seq_len=64, pos_encoding="rope")
    cfgs = {
        "MHA": ModelConfig(attention="mha", **common),
        "GQA-2": ModelConfig(attention="gqa", n_kv_heads=2, **common),
        "MQA": ModelConfig(attention="mqa", **common),
        "MLA": ModelConfig(attention="mla", d_c=16, d_head_rope=8, **common),
    }
    floor = (8 - 1) / (32 - 1) * math.log(16)
    ceiling = ((32 - 8) + (8 - 1) / 16) / (32 - 1)
    print("\n复制任务消融（3 个种子，600 步，损失/准确率取均值±标准差）")
    print(f"参照损失下界 {floor:.4f}，随机猜测准确率 {ceiling:.4f}")
    print(f"{'变体':<8s}{'参数量':>10s}{'每Token缓存':>13s}"
          f"{'损失':>16s}{'准确率':>18s}")
    for name, cfg in cfgs.items():
        n = exact_params(cfg)
        cache = kv_cache_bytes(cfg, 1)
        results = []
        for seed in range(3):
            model = train_one(cfg, steps=600, seed=seed)
            results.append(evaluate(model))
        losses = torch.tensor([r[0] for r in results])
        accs = torch.tensor([r[1] for r in results])
        print(f"{name:<8s}{n:>10,d}{cache:>13d}"
              f"{losses.mean():>10.4f} ± {losses.std():.4f}"
              f"{accs.mean():>11.4f} ± {accs.std():.4f}")


def main():
    ledger()
    industrial_cases()
    absorption_demo()
    ablation()


if __name__ == "__main__":
    main()
