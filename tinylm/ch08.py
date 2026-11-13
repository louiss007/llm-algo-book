"""第 8 章：超越 Transformer（tinylm/ssm.py、block.py 的增量）。"""
import math
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from ch03 import CausalLMOutput
from ch04 import sinusoidal_positions
from ch05 import make_norm
from ch06 import attention_params
from ch07 import (ModelConfig as Config7, TransformerBlock, dense_ffn_params,
                 make_ffn_or_moe, moe_layer_params)


# ---------------------------------------------------------------- config.py
@dataclass(frozen=True)
class ModelConfig(Config7):
    block_type: str = "attention"      # attention | linear_attention | ssm | hybrid
    ssm_d_state: int = 16              # 每个通道的状态维度 N
    ssm_expand: int = 2                # 内部宽度 I = expand * d_model
    hybrid_attn_every: int = 4         # 混合架构中每隔多少层放一个注意力层

    def __post_init__(self):
        super().__post_init__()
        assert self.block_type in ("attention", "linear_attention", "ssm", "hybrid")
        assert self.ssm_d_state >= 1
        assert self.ssm_expand >= 1
        assert self.hybrid_attn_every >= 1


# ---------------------------------------------------------------- ssm.py
def scan_states(a, b):
    """公式 (8.17)：h_t = a_t h_{t-1} + b_t 的并行关联扫描。

    Hillis-Steele 扫描，把 (a_t, b_t) 复合为 (a2 a1, a2 b1 + b2)，
    共 log2(T) 轮，每轮对序列维做一次向量化更新。
    """
    A, Bv = a, b
    T = a.shape[1]
    step = 1
    while step < T:
        A_prev = A
        A = torch.cat([A[:, :step], A[:, step:] * A_prev[:, :-step]], dim=1)
        Bv = torch.cat(
            [Bv[:, :step], A_prev[:, step:] * Bv[:, :-step] + Bv[:, step:]], dim=1)
        step *= 2
    return Bv


class SelectiveSSM(nn.Module):
    """公式 (8.8)–(8.16)：输入相关的选择性状态空间层（Mamba 风格）。"""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.inner = cfg.ssm_expand * d
        self.n_state = cfg.ssm_d_state
        self.in_proj = nn.Linear(d, 2 * self.inner, bias=cfg.bias)
        self.conv_w = nn.Parameter(torch.empty(self.inner, 4))  # 深度因果卷积
        nn.init.kaiming_uniform_(self.conv_w, a=math.sqrt(5))
        self.conv_b = nn.Parameter(torch.zeros(self.inner)) if cfg.bias else None
        self.x_proj = nn.Linear(self.inner, 2 * self.n_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.inner, bias=True)
        a = torch.empty(self.inner, self.n_state).uniform_(1.0, 16.0)
        self.A_log = nn.Parameter(torch.log(a))          # A = -exp(A_log) < 0
        self.D = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, d, bias=cfg.bias)
        self.last_state = None

    def forward(self, x, attention_mask=None, position_ids=None):
        B, T, _ = x.shape
        u, z = self.in_proj(x).chunk(2, dim=-1)
        xp = F.pad(u.transpose(1, 2), (3, 0))                      # 左侧补 3 个 0
        conv = sum(xp[:, :, k:k + T] * self.conv_w[:, k].view(1, -1, 1)
                   for k in range(4))
        if self.conv_b is not None:
            conv = conv + self.conv_b.view(1, -1, 1)
        u = F.silu(conv.transpose(1, 2))                            # 因果卷积

        bc_dt = self.x_proj(u)
        B_t, C_t, dt_raw = torch.split(
            bc_dt, [self.n_state, self.n_state, 1], dim=-1)           # (8.13)
        dt = F.softplus(self.dt_proj(dt_raw))                         # (8.13)

        A = -torch.exp(self.A_log)                                    # (8.10)
        A_bar = torch.exp(dt.unsqueeze(-1) * A)                       # (8.14)
        ratio = torch.where(
            A.abs() < 1e-6,
            dt.unsqueeze(-1).expand_as(A_bar),
            (A_bar - 1.0) / A,
        )
        B_bar = ratio * B_t.unsqueeze(-2)                             # (8.14)

        h = scan_states(A_bar, B_bar)                                 # (8.17)
        y = (C_t.unsqueeze(-2) * h).sum(-1) + self.D * u              # (8.10)
        self.last_state = h[:, -1].detach()
        return self.out_proj(y * F.silu(z))


class LinearAttention(nn.Module):
    """公式 (8.1)–(8.6)：以 φ(q)ᵀφ(k) 替代 softmax 的线性注意力。"""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.q_proj = nn.Linear(d, d, bias=cfg.bias)
        self.k_proj = nn.Linear(d, d, bias=cfg.bias)
        self.v_proj = nn.Linear(d, d, bias=cfg.bias)
        self.out_proj = nn.Linear(d, d, bias=cfg.bias)
        self.eps = 1e-6
        self.last_state = None

    def forward(self, x, attention_mask=None, position_ids=None):
        q = F.elu(self.q_proj(x)) + 1.0                               # φ(q)
        k = F.elu(self.k_proj(x)) + 1.0
        v = self.v_proj(x)
        S = (k.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(dim=1)         # (8.4)
        z = k.cumsum(dim=1)
        num = torch.einsum("btd,btdv->btv", q, S)                     # (8.3)
        den = torch.einsum("btd,btd->bt", q, z).clamp_min(self.eps)
        self.last_state = S[:, -1].detach()
        return self.out_proj(num / den.unsqueeze(-1))                 # (8.5)


class SSMBlock(nn.Module):
    """与 TransformerBlock 同构：只把注意力换成 SSM，前馈子层保持不变。"""

    def __init__(self, cfg):
        super().__init__()
        self.parallel = cfg.parallel_block
        self.norm1 = make_norm(cfg)
        self.mixer = SelectiveSSM(cfg)
        self.norm2 = None if self.parallel else make_norm(cfg)
        self.mlp = make_ffn_or_moe(cfg)

    def forward(self, x, attention_mask=None, position_ids=None):
        if self.parallel:
            h = self.norm1(x)
            delta, aux = self._ffn(h)
            return x + self.mixer(h) + delta, aux
        x = x + self.mixer(self.norm1(x))
        delta, aux = self._ffn(self.norm2(x))
        return x + delta, aux

    def _ffn(self, h):
        out = self.mlp(h)
        return out if isinstance(out, tuple) else (out, None)


class LinearAttentionBlock(SSMBlock):
    """同样的块结构，mixer 换成线性注意力。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.mixer = LinearAttention(cfg)


def make_block(cfg, layer_idx):
    """按 block_type 分派；默认 attention 与第 7 章逐位一致。"""
    if cfg.block_type == "attention":
        return TransformerBlock(cfg)
    if cfg.block_type == "linear_attention":
        return LinearAttentionBlock(cfg)
    if cfg.block_type == "ssm":
        return SSMBlock(cfg)
    m = cfg.hybrid_attn_every
    return TransformerBlock(cfg) if (layer_idx + 1) % m == 0 else SSMBlock(cfg)


def layer_kinds(cfg):
    if cfg.block_type == "attention":
        return ["attn"] * cfg.n_layers
    if cfg.block_type == "linear_attention":
        return ["linear"] * cfg.n_layers
    if cfg.block_type == "ssm":
        return ["ssm"] * cfg.n_layers
    m = cfg.hybrid_attn_every
    return ["attn" if (i + 1) % m == 0 else "ssm" for i in range(cfg.n_layers)]


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
        self.blocks = nn.ModuleList([make_block(cfg, i) for i in range(cfg.n_layers)])
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
def ssm_layer_params(cfg):
    """公式 (8.16)：单层 SSM 的参数量。"""
    I = cfg.ssm_expand * cfg.d_model
    N = cfg.ssm_d_state
    p = 3 * cfg.d_model * I + I * (3 * N + 7)
    p += (4 * I + cfg.d_model) if cfg.bias else I  # dt_proj 始终带偏置
    return p


def norm_params(cfg):
    d = cfg.d_model
    return d + (d if cfg.norm == "layernorm" and cfg.bias else 0)


def block_params(cfg, kind):
    """单层参数量：三种块都只替换 mixer，前馈子层与归一化保持一致。"""
    d, n = cfg.d_model, norm_params(cfg)
    ffn = moe_layer_params(cfg) if cfg.moe is not None else dense_ffn_params(cfg)
    norms = n if cfg.parallel_block else 2 * n
    if kind == "attn":
        mixer = attention_params(cfg)
    elif kind == "linear":
        mixer = 4 * d * d + (4 * d if cfg.bias else 0)
    else:
        mixer = ssm_layer_params(cfg)
    return mixer + ffn + norms


def model_params(cfg):
    """按块类型逐层累加，避免为 4096 维配置实例化整个模型。"""
    emb = cfg.vocab_size * cfg.d_model * (1 if cfg.tie_embeddings else 2)
    pos = cfg.max_seq_len * cfg.d_model if cfg.pos_encoding == "learned" else 0
    return (sum(block_params(cfg, k) for k in layer_kinds(cfg))
            + norm_params(cfg) + emb + pos)


def state_bytes(cfg, batch=1, seq_len=8192, b_elem=2):
    """每层解码状态：注意力缓存随 T 增长，SSM/线性注意力状态与 T 无关。"""
    d = cfg.d_model
    total = 0
    for kind in layer_kinds(cfg):
        if kind == "attn":
            if cfg.attention == "mla":
                total += batch * seq_len * (cfg.d_c + cfg.d_head_rope) * b_elem
            else:
                d_kv = cfg.n_kv_heads * (d // cfg.n_heads)
                total += 2 * batch * seq_len * d_kv * b_elem
        elif kind == "linear":
            total += batch * (d * d + d) * b_elem
        else:
            total += batch * (cfg.ssm_expand * d) * cfg.ssm_d_state * b_elem
    return total


def decode_flops(cfg, seq_len):
    """每 Token 解码的粗略 FLOPs：权重读取 + 序列相关项。"""
    total = 2 * model_params(cfg)
    d = cfg.d_model
    for kind in layer_kinds(cfg):
        if kind == "attn":
            total += 4 * seq_len * d          # QK^T + AV，按查询头计 4 T d
        elif kind == "linear":
            total += 4 * d * d
        else:
            total += 4 * (cfg.ssm_expand * d) * cfg.ssm_d_state
    return total


def human(nbytes):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if nbytes < 1024 or unit == "TiB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes:.0f} B"
        nbytes /= 1024


def ledger():
    common = dict(vocab_size=32000, d_model=4096, n_layers=32, n_heads=32,
                  n_kv_heads=8, d_ff=14336, max_seq_len=32768, norm="rmsnorm",
                  ffn="swiglu", bias=False, pos_encoding="rope", attention="gqa")
    cfgs = {
        "Attention": ModelConfig(**common),
        "LinearAttn": ModelConfig(block_type="linear_attention", **common),
        "SSM": ModelConfig(block_type="ssm", ssm_d_state=16, ssm_expand=2, **common),
        "Hybrid(1:3)": ModelConfig(block_type="hybrid", hybrid_attn_every=4,
                                   ssm_d_state=16, ssm_expand=2, **common),
    }
    print("状态与算力账本（FP16 状态，B=1，单序列解码）")
    print(f"{'配置':<12s}{'参数量':>14s}{'状态@8K':>12s}{'状态@128K':>12s}"
          f"{'FLOPs@8K':>12s}{'FLOPs@128K':>13s}")
    for name, cfg in cfgs.items():
        p = model_params(cfg)
        print(f"{name:<12s}{p:>14,d}{human(state_bytes(cfg, 1, 8192)):>12s}"
              f"{human(state_bytes(cfg, 1, 131072)):>12s}"
              f"{decode_flops(cfg, 8192) / 1e9:>9.1f} G"
              f"{decode_flops(cfg, 131072) / 1e9:>10.1f} G")

    print("\n每层状态大小（B=1，FP16）")
    print(f"{'层类型':<10s}{'8K':>12s}{'128K':>12s}{'随 T 增长':>10s}")
    for name, cfg in cfgs.items():
        kinds = layer_kinds(cfg)
        a8 = state_bytes_all(cfg, 8192) / max(1, len(kinds))
        a128 = state_bytes_all(cfg, 131072) / max(1, len(kinds))
        grows = "是" if a128 > 1.01 * a8 else "否"
        print(f"{name:<10s}{human(a8):>12s}{human(a128):>12s}{grows:>10s}")

    one = ModelConfig(vocab_size=16, d_model=64, n_layers=2, n_heads=4,
                      d_ff=256, max_seq_len=64, norm="rmsnorm", ffn="swiglu",
                      bias=False, pos_encoding="rope", block_type="ssm",
                      ssm_d_state=16, ssm_expand=2)
    measured = sum(p.numel() for p in TinyLM(one).parameters())
    print(f"\n单层 SSM 参数：公式 {ssm_layer_params(one):,d}，"
          f"实测 {sum(p.numel() for p in SelectiveSSM(one).parameters()):,d}")
    print(f"两层 SSM 模型：公式 {model_params(one):,d}，实测 {measured:,d}")


def state_bytes_all(cfg, seq_len):
    """无批次的每层平均状态字节数，便于逐层对照。"""
    return state_bytes(cfg, 1, seq_len)


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
    mixer = model.blocks[0].mixer if hasattr(model.blocks[0], "mixer") else None
    state = getattr(mixer, "last_state", None)
    norm = state.norm().item() if state is not None else float("nan")
    return out.loss.item(), acc, norm


def ablation():
    common = dict(vocab_size=16, d_model=64, n_layers=2, n_heads=4,
                  d_ff=256, max_seq_len=64, pos_encoding="rope",
                  norm="rmsnorm", ffn="swiglu")
    cfgs = {
        "Attention": ModelConfig(block_type="attention", **common),
        "LinearAttn": ModelConfig(block_type="linear_attention", **common),
        "SSM": ModelConfig(block_type="ssm", ssm_d_state=16, ssm_expand=2, **common),
        "Hybrid(1:1)": ModelConfig(block_type="hybrid", hybrid_attn_every=2,
                                   ssm_d_state=16, ssm_expand=2, **common),
    }
    floor = (8 - 1) / (32 - 1) * math.log(16)
    print("\n复制任务消融（3 个种子，600 步，损失/准确率取均值±标准差）")
    print(f"参照损失下界 {floor:.4f}")
    print(f"{'配置':<12s}{'参数量':>10s}{'损失':>16s}{'准确率':>18s}{'状态范数':>10s}")
    for name, cfg in cfgs.items():
        params = model_params(cfg)
        results = []
        for seed in range(3):
            model = train_one(cfg, seed=seed)
            results.append(evaluate(model))
        losses = torch.tensor([r[0] for r in results])
        accs = torch.tensor([r[1] for r in results])
        norms = torch.tensor([r[2] for r in results])
        norm_s = "—" if math.isnan(norms.mean()) else f"{norms.mean():.2f}"
        print(f"{name:<12s}{params:>10,d}"
              f"{losses.mean():>10.4f} ± {losses.std():.4f}"
              f"{accs.mean():>11.4f} ± {accs.std():.4f}{norm_s:>10s}")


def long_context_demo():
    cfg = ModelConfig(vocab_size=16, d_model=64, n_layers=2, n_heads=4,
                      d_ff=256, max_seq_len=64, norm="rmsnorm", ffn="swiglu",
                      bias=False, pos_encoding="rope", block_type="ssm",
                      ssm_d_state=16, ssm_expand=2)
    torch.manual_seed(0)
    model = TinyLM(cfg).eval()
    with torch.no_grad():
        for T in (8, 64, 512):
            ids = torch.randint(0, 16, (1, T))
            out = model(ids)
            state = model.blocks[0].mixer.last_state
            print(f"T={T:>3d}  输出有限 {bool(torch.isfinite(out.logits).all())}"
                  f"  状态形状 {tuple(state.shape)}  状态范数 {state.norm():.2f}")


def main():
    ledger()
    long_context_demo()
    ablation()


if __name__ == "__main__":
    main()
