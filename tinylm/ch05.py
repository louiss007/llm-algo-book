"""第 5 章：现代 LLM 的架构演进（tinylm/norm.py、ffn.py、model.py 的增量）。"""
import math
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from ch03 import CausalLMOutput, LayerNorm, MLP
from ch04 import (
    ModelConfig as Config4,
    PositionAwareAttention,
    sinusoidal_positions,
)


# ---------------------------------------------------------------- config.py
@dataclass(frozen=True)
class ModelConfig(Config4):
    parallel_block: bool = False

    def __post_init__(self):
        if self.d_ff is None:
            ratio = 8 / 3 if self.ffn == "swiglu" else 4   # 公式 (5.8)
            object.__setattr__(self, "d_ff", int(round(ratio * self.d_model)))
        super().__post_init__()
        assert self.norm in ("layernorm", "rmsnorm")
        assert self.ffn in ("mlp", "swiglu")


# ---------------------------------------------------------------- norm.py
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
    if cfg.norm == "rmsnorm":
        return RMSNorm(cfg.d_model, cfg.norm_eps)
    return LayerNorm(cfg.d_model, cfg.norm_eps, cfg.bias)


# ---------------------------------------------------------------- ffn.py
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


# ---------------------------------------------------------------- block.py
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.parallel = cfg.parallel_block
        self.norm1 = make_norm(cfg)
        self.attn = PositionAwareAttention(cfg)
        self.norm2 = None if self.parallel else make_norm(cfg)
        self.mlp = make_ffn(cfg)

    def forward(self, x, attention_mask=None, position_ids=None):
        if self.parallel:                                      # 公式 (5.10)
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


# ---------------------------------------------------------------- 参数与算力账本
def exact_params(cfg):
    """公式 (5.11)：RMSNorm + SwiGLU + 可选偏置/绑定/并行层。"""
    d, f, V, L = cfg.d_model, cfg.d_ff, cfg.vocab_size, cfg.max_seq_len
    attn = 4 * d * d + (4 * d if cfg.bias else 0)
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


def flops_per_token(cfg, T):
    """公式 (5.12)：每个 Token 的前向 FLOPs，忽略偏置与归一化。"""
    d, f = cfg.d_model, cfg.d_ff
    attn_proj = 8 * d * d
    ffn = (6 * d * f) if cfg.ffn == "swiglu" else (4 * d * f)
    scores = 4 * T * d
    return cfg.n_layers * (attn_proj + ffn + scores)


def ledger_small():
    base = ModelConfig(vocab_size=1024, d_model=256, n_layers=4, n_heads=8,
                       d_ff=1024, max_seq_len=256)
    modern = ModelConfig(vocab_size=1024, d_model=256, n_layers=4, n_heads=8,
                         norm="rmsnorm", ffn="swiglu", bias=False, max_seq_len=256)
    modern_tied = ModelConfig(vocab_size=1024, d_model=256, n_layers=4, n_heads=8,
                              norm="rmsnorm", ffn="swiglu", bias=False,
                              tie_embeddings=True, max_seq_len=256)
    modern_par = ModelConfig(vocab_size=1024, d_model=256, n_layers=4, n_heads=8,
                             norm="rmsnorm", ffn="swiglu", bias=False,
                             parallel_block=True, max_seq_len=256)
    print("小模型账本（d=256, n_l=4, V=1024, L=256, T=256）")
    print(f"{'配置':<18s}{'实测':>10s}{'公式':>10s}{'每Token FLOPs':>16s}")
    for name, cfg in (("基线 LN+MLP", base), ("现代 RMS+SwiGLU", modern),
                      ("现代+权重绑定", modern_tied), ("现代+并行层", modern_par)):
        model = TinyLM(cfg)
        actual = sum(p.numel() for p in model.parameters())
        print(f"{name:<18s}{actual:>10,d}{exact_params(cfg):>10,d}"
              f"{flops_per_token(cfg, 256):>16,d}")


def llama2_ledger():
    """Llama 2 7B 的账本；只算公式，不实例化 67 亿参数。"""
    cfg = ModelConfig(vocab_size=32000, d_model=4096, n_layers=32, n_heads=32,
                      d_ff=11008, max_seq_len=4096, norm="rmsnorm",
                      ffn="swiglu", bias=False, pos_encoding="rope")
    n = exact_params(cfg)
    print("\nLlama 2 7B 账本")
    print(f"公式 (5.11): {n:,}    公开参数量: 6,738,415,616    差 {n - 6_738_415_616:+,}")
    d, f, nl, V = 4096, 11008, 32, 32000
    attn = nl * 4 * d * d
    ffn = nl * 3 * d * f
    norms = (nl * 2 + 1) * d
    emb = 2 * V * d
    print(f"注意力 {attn:,}  前馈 {ffn:,}  归一化 {norms:,}  嵌入/输出 {emb:,}")
    print(f"前馈占非嵌入参数: {ffn / (attn + ffn + norms):.1%}")


def llama3_delta():
    """Llama 2 7B 到 Llama 3 8B 的参数增量拆解。"""
    d, nl = 4096, 32
    vocab = (128256 - 32000) * d * 2
    ffn = 3 * d * (14336 - 11008) * nl
    d_kv = d * 8 // 32
    gqa = ((2 * d * d + 2 * d * d_kv) - 4 * d * d) * nl
    print("\nLlama 2 7B → Llama 3 8B 的参数增量")
    print(f"词表 32K→128K: {vocab:+,}")
    print(f"前馈 11008→14336: {ffn:+,}")
    print(f"注意力 MHA→GQA(8 KV 头): {gqa:+,}")
    total = 6_738_415_616 + vocab + ffn + gqa
    print(f"合计: {vocab + ffn + gqa:+,}   →   {total:,}")


def norm_demo():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 64)
    rms, ln = RMSNorm(64), LayerNorm(64, 1e-5, True)
    with torch.no_grad():
        ln.gamma.copy_(rms.gamma)
        ln.beta.zero_()
    with torch.no_grad():
        scale_diff = (rms(10 * x) - rms(x)).abs().max().item()
        shift_rms = (rms(x + 5) - rms(x)).abs().max().item()
        shift_ln = (ln(x + 5) - ln(x)).abs().max().item()
        x0 = x - x.mean(dim=-1, keepdim=True)
        zero_mean_diff = (rms(x0) - ln(x0)).abs().max().item()
    print("\nRMSNorm 与 LayerNorm 的行为差异")
    print(f"输入放大 10 倍，RMSNorm 输出最大差异: {scale_diff:.2e}")
    print(f"输入平移 +5，RMSNorm 输出最大差异: {shift_rms:.2e}")
    print(f"输入平移 +5，LayerNorm 输出最大差异: {shift_ln:.2e}")
    print(f"零均值输入下两者输出最大差异: {zero_mean_diff:.2e}")


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
                  max_seq_len=64, pos_encoding="learned")
    cfgs = {
        "基线 LN+MLP": ModelConfig(d_ff=256, **common),
        "仅 RMSNorm": ModelConfig(d_ff=256, norm="rmsnorm", **common),
        "仅 SwiGLU": ModelConfig(ffn="swiglu", **common),
        "现代组合": ModelConfig(norm="rmsnorm", ffn="swiglu", bias=False, **common),
    }
    floor = (8 - 1) / (32 - 1) * math.log(16)
    ceiling = ((32 - 8) + (8 - 1) / 16) / (32 - 1)
    print("\n复制任务消融（3 个种子，300 步，损失/准确率取均值±标准差）")
    print(f"理论损失下界 {floor:.4f}，理论上界（准确率）{ceiling:.4f}")
    print(f"{'配置':<14s}{'参数量':>9s}{'损失':>16s}{'准确率':>18s}")
    for name, cfg in cfgs.items():
        n = exact_params(cfg)
        results = []
        for seed in range(3):
            model = train_one(cfg, seed=seed)
            results.append(evaluate(model))
        losses = torch.tensor([r[0] for r in results])
        accs = torch.tensor([r[1] for r in results])
        print(f"{name:<14s}{n:>9,d}"
              f"{losses.mean():>10.4f} ± {losses.std():.4f}"
              f"{accs.mean():>11.4f} ± {accs.std():.4f}")


def main():
    ledger_small()
    llama2_ledger()
    llama3_delta()
    norm_demo()
    ablation()


if __name__ == "__main__":
    main()
