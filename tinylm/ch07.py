"""第 7 章：混合专家模型（tinylm/moe.py、block.py 的增量）。"""
import math
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from ch03 import CausalLMOutput
from ch04 import sinusoidal_positions
from ch05 import make_ffn, make_norm
from ch06 import ModelConfig as Config6, make_attention


# ---------------------------------------------------------------- config.py
@dataclass(frozen=True)
class MoEConfig:
    n_experts: int = 8
    top_k: int = 2
    n_shared: int = 0
    d_ff_expert: int | None = None       # None 表示沿用 cfg.d_ff
    capacity_factor: float | None = 1.25  # None 表示不丢弃
    aux_loss_coef: float = 0.01

    def __post_init__(self):
        assert self.n_experts >= 1
        assert 1 <= self.top_k <= self.n_experts
        assert self.n_shared >= 0
        assert self.d_ff_expert is None or self.d_ff_expert > 0
        assert self.capacity_factor is None or self.capacity_factor > 0
        assert self.aux_loss_coef >= 0


@dataclass(frozen=True)
class ModelConfig(Config6):
    moe: MoEConfig | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.moe is not None:
            assert self.moe.top_k <= self.moe.n_experts


# ---------------------------------------------------------------- moe.py
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


# ---------------------------------------------------------------- block.py
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.parallel = cfg.parallel_block
        self.norm1 = make_norm(cfg)
        self.attn = make_attention(cfg)
        self.norm2 = None if self.parallel else make_norm(cfg)
        self.mlp = make_ffn_or_moe(cfg)

    def forward(self, x, attention_mask=None, position_ids=None):
        if self.parallel:
            h = self.norm1(x)
            delta, aux = self._ffn(h)
            return x + self.attn(h, attention_mask, position_ids) + delta, aux
        x = x + self.attn(self.norm1(x), attention_mask, position_ids)
        delta, aux = self._ffn(self.norm2(x))
        return x + delta, aux

    def _ffn(self, h):
        out = self.mlp(h)
        return out if isinstance(out, tuple) else (out, None)


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

        aux_total = None
        for block in self.blocks:
            x, aux = block(x, attention_mask, position_ids)
            if aux is not None:
                aux_total = aux if aux_total is None else aux_total + aux

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(logits=logits, loss=loss, aux_loss=aux_total)


# ---------------------------------------------------------------- 账本
def expert_params(cfg):
    """公式 (7.10)：单个专家的参数量（SwiGLU 为 3 d d_f，MLP 为 2 d d_f）。"""
    d, f = cfg.d_model, (cfg.moe.d_ff_expert or cfg.d_ff)
    if cfg.ffn == "swiglu":
        return 3 * d * f + ((2 * f + d) if cfg.bias else 0)
    return 2 * d * f + ((f + d) if cfg.bias else 0)


def dense_ffn_params(cfg):
    d, f = cfg.d_model, cfg.d_ff
    if cfg.ffn == "swiglu":
        return 3 * d * f + ((2 * f + d) if cfg.bias else 0)
    return 2 * d * f + ((f + d) if cfg.bias else 0)


def moe_layer_params(cfg, active=False):
    """公式 (7.11)、(7.13)：路由 + 参与计算的专家个数。"""
    m = cfg.moe
    n = m.n_experts
    used = (m.top_k + m.n_shared) if active else (n + m.n_shared)
    return cfg.d_model * n + used * expert_params(cfg)


def model_params(cfg, active=False):
    """公式 (7.12)、(7.13)：整模型的总参数与每 Token 激活参数。"""
    d, V, L = cfg.d_model, cfg.vocab_size, cfg.max_seq_len
    from ch06 import attention_params
    attn = attention_params(cfg)
    if cfg.moe is None:
        ffn = dense_ffn_params(cfg)
    else:
        ffn = moe_layer_params(cfg, active=active)
    norm_p = d + (d if cfg.norm == "layernorm" and cfg.bias else 0)
    norms = norm_p if cfg.parallel_block else 2 * norm_p
    per_layer = attn + ffn + norms
    emb = V * d * (1 if cfg.tie_embeddings else 2)
    pos = L * d if cfg.pos_encoding == "learned" else 0
    return cfg.n_layers * per_layer + norm_p + emb + pos


def human(nbytes):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if nbytes < 1024 or unit == "TiB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes:.0f} B"
        nbytes /= 1024


def ledger():
    common = dict(vocab_size=32000, d_model=4096, n_layers=32, n_heads=32,
                  n_kv_heads=8, d_ff=14336, max_seq_len=32768, norm="rmsnorm",
                  ffn="swiglu", bias=False, pos_encoding="rope", attention="gqa")
    mixtral = ModelConfig(moe=MoEConfig(n_experts=8, top_k=2,
                                        d_ff_expert=14336), **common)
    v3 = ModelConfig(
        vocab_size=129280, d_model=7168, n_layers=61, n_heads=128,
        d_ff=18432, max_seq_len=16384, norm="rmsnorm", ffn="swiglu",
        bias=False, pos_encoding="rope", attention="mla", d_c=512,
        d_head_rope=64,
        moe=MoEConfig(n_experts=256, top_k=8, n_shared=1, d_ff_expert=2048),
    )
    toy = ModelConfig(vocab_size=32000, d_model=512, n_layers=8, n_heads=8,
                      d_ff=2048, max_seq_len=1024, norm="rmsnorm",
                      ffn="swiglu", bias=False, pos_encoding="rope",
                      moe=MoEConfig(n_experts=8, top_k=2, d_ff_expert=1024))

    print("MoE 账本（FP16 权重）")
    print(f"{'配置':<14s}{'总参数':>16s}{'激活参数':>16s}{'激活比':>8s}"
          f"{'权重显存':>10s}{'每Token FLOPs':>14s}")
    for name, cfg in (("Mixtral 8x7B", mixtral), ("DeepSeek-V3", v3),
                      ("TinyLM MoE", toy)):
        total = model_params(cfg)
        active = model_params(cfg, active=True)
        if name == "TinyLM MoE":
            actual = sum(p.numel() for p in TinyLM(cfg).parameters())
            assert total == actual, (total, actual)
        print(f"{name:<14s}{total:>16,d}{active:>16,d}"
              f"{active / total:>8.1%}{human(2 * total):>10s}"
              f"{2 * active / 1e9:>11.1f} GFLOP")

    v3_adjusted = model_params(v3) - 3 * (moe_layer_params(v3) - dense_ffn_params(v3))
    print(f"DeepSeek-V3 前 3 层为稠密前馈：扣除后总参数 {v3_adjusted:,d}"
          f"（公开值约 671B）")
    dense = ModelConfig(**common)
    sparse = mixtral
    print(f"\n稠密对照：Mixtral 尺寸的稠密前馈总参数 "
          f"{model_params(dense):,d}，MoE 总参数 {model_params(sparse):,d}，"
          f"激活参数 {model_params(sparse, active=True):,d}")
    print(f"MoE 用 {model_params(sparse, active=True) / model_params(dense):.2f} "
          f"倍的激活参数承载 {model_params(sparse) / model_params(dense):.2f} 倍的容量")


# ---------------------------------------------------------------- 受控实验
def copy_task(B, T, lag, vocab, gen):
    seq = torch.randint(0, vocab, (B, lag), generator=gen)
    for t in range(lag, T):
        seq = torch.cat([seq, seq[:, t - lag:t - lag + 1]], dim=1)
    return seq


def make_labels(seq):
    return seq[:, :-1], seq[:, 1:].clone()


def train_one(cfg, steps=600, seed=0, train_len=32, lag=8):
    torch.manual_seed(seed)
    model = TinyLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    gen = torch.Generator().manual_seed(1000 + seed)
    model.train()
    for _ in range(steps):
        seq = copy_task(64, train_len, lag, cfg.vocab_size, gen)
        inp, tgt = make_labels(seq)
        out = model(inp, labels=tgt)
        total = out.loss + (out.aux_loss if out.aux_loss is not None else 0.0)
        opt.zero_grad()
        total.backward()
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
    try:
        moe = model.blocks[0].mlp
        load = moe.last_load.float()
        load = load / load.sum()
        entropy = -(load * load.clamp_min(1e-9).log()).sum().item()
        max_share = load.max().item()
    except AttributeError:
        entropy, max_share = float("nan"), float("nan")
    return out.loss.item(), acc, entropy, max_share


def ablation():
    common = dict(vocab_size=16, d_model=64, n_layers=2, n_heads=4,
                  d_ff=256, max_seq_len=64, pos_encoding="rope",
                  norm="rmsnorm", ffn="swiglu")
    cfgs = {
        "Dense": ModelConfig(**common),
        "MoE-aux0": ModelConfig(moe=MoEConfig(
            n_experts=8, top_k=2, d_ff_expert=128, aux_loss_coef=0.0), **common),
        "MoE-aux": ModelConfig(moe=MoEConfig(
            n_experts=8, top_k=2, d_ff_expert=128, aux_loss_coef=0.01), **common),
    }
    floor = (8 - 1) / (32 - 1) * math.log(16)
    print("\n复制任务消融（3 个种子，600 步，损失/准确率取均值±标准差）")
    print(f"参照损失下界 {floor:.4f}")
    print(f"{'配置':<10s}{'总参数':>10s}{'激活参数':>10s}{'损失':>16s}"
          f"{'准确率':>18s}{'负载熵':>8s}{'最大份额':>9s}")
    for name, cfg in cfgs.items():
        total, active = model_params(cfg), model_params(cfg, active=True)
        results = []
        for seed in range(3):
            model = train_one(cfg, seed=seed)
            results.append(evaluate(model))
        losses = torch.tensor([r[0] for r in results])
        accs = torch.tensor([r[1] for r in results])
        ent = torch.tensor([r[2] for r in results])
        mx = torch.tensor([r[3] for r in results])
        ent_s = "—" if math.isnan(ent.mean()) else f"{ent.mean():.2f}"
        mx_s = "—" if math.isnan(mx.mean()) else f"{mx.mean():.2f}"
        print(f"{name:<10s}{total:>10,d}{active:>10,d}"
              f"{losses.mean():>10.4f} ± {losses.std():.4f}"
              f"{accs.mean():>11.4f} ± {accs.std():.4f}"
              f"{ent_s:>8s}{mx_s:>9s}")


def capacity_demo():
    cfg = ModelConfig(vocab_size=16, d_model=64, n_layers=2, n_heads=4,
                      d_ff=128, max_seq_len=64, pos_encoding="rope",
                      norm="rmsnorm", ffn="swiglu",
                      moe=MoEConfig(n_experts=8, top_k=2, d_ff_expert=64,
                                    capacity_factor=None, aux_loss_coef=0.0))
    torch.manual_seed(0)
    model = TinyLM(cfg)
    ids = torch.randint(0, 16, (8, 32))
    with torch.no_grad():
        model.eval()
        base = model(ids).logits
        for m in model.modules():
            if isinstance(m, MoE):
                m.capacity_factor = 0.5
        tight = model(ids).logits
        loads = [int(m.last_dropped) for m in model.modules() if isinstance(m, MoE)]
    diff = (base - tight).abs().max().item()
    capacity = max(1, math.ceil(0.5 * 8 * 32 * 2 / 8))
    print(f"\n容量收紧（CF=1.25 → 0.5，容量 {capacity}/专家）："
          f"两层共丢弃 {sum(loads)}/{2 * 8 * 32 * 2} 个分派，"
          f"logits 最大变化 {diff:.3f}")


def main():
    ledger()
    capacity_demo()
    ablation()


if __name__ == "__main__":
    main()
