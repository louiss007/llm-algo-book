"""第 3 章：Transformer 基线（tinylm/model 的核心实现）。"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- config.py
@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int | None = None
    d_ff: int | None = None
    max_seq_len: int = 1024
    attention: str = "mha"
    norm: str = "layernorm"
    norm_eps: float = 1e-5
    ffn: str = "mlp"
    pos_encoding: str = "learned"
    rope_theta: float = 10_000.0
    bias: bool = True
    tie_embeddings: bool = False
    dropout: float = 0.0

    def __post_init__(self):
        if self.d_ff is None:
            object.__setattr__(self, "d_ff", 4 * self.d_model)
        if self.n_kv_heads is None:
            object.__setattr__(self, "n_kv_heads", self.n_heads)
        assert self.d_model % self.n_heads == 0, "d_model 必须能被 n_heads 整除"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads 必须是 n_kv_heads 的整数倍"


# ---------------------------------------------------------------- norm.py
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


# ---------------------------------------------------------------- attention.py
class MultiHeadAttention(nn.Module):
    """因果多头自注意力：公式 (3.5)–(3.14)。"""

    def __init__(self, cfg):
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        self.n_heads = h
        self.d_head = d // h
        self.q_proj = nn.Linear(d, d, bias=cfg.bias)
        self.k_proj = nn.Linear(d, d, bias=cfg.bias)
        self.v_proj = nn.Linear(d, d, bias=cfg.bias)
        self.out_proj = nn.Linear(d, d, bias=cfg.bias)
        # 公式 (3.10)：上三角为 -inf，位置 i 只能看到 j <= i
        mask = torch.triu(torch.full((cfg.max_seq_len, cfg.max_seq_len), float("-inf")),
                          diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x, attention_mask=None, return_weights=False):
        B, T, _ = x.shape
        # 公式 (3.5)：三个投影，再拆成 n_heads 个头
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # 公式 (3.6)(3.11)：缩放点积打分 + 因果掩码 + 行 softmax
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        scores = scores + self.causal_mask[:T, :T]
        if attention_mask is not None:
            pad = attention_mask[:, None, None, :] == 0
            scores = scores.masked_fill(pad, float("-inf"))
        attn = scores.softmax(dim=-1)

        # 公式 (3.12)(3.14)：加权求和，拼接后做输出投影
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)
        out = self.out_proj(out)
        return (out, attn) if return_weights else out


# ---------------------------------------------------------------- ffn.py
class MLP(nn.Module):
    """公式 (3.15)：两层前馈网络，中间维度 d_ff。"""

    def __init__(self, cfg):
        super().__init__()
        self.fc_in = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.fc_out = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)

    def forward(self, x):
        return self.fc_out(F.gelu(self.fc_in(x)))


# ---------------------------------------------------------------- block.py
class TransformerBlock(nn.Module):
    """公式 (3.4)：Pre-LN 残差块。"""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = LayerNorm(cfg.d_model, cfg.norm_eps, cfg.bias)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = LayerNorm(cfg.d_model, cfg.norm_eps, cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)   # 子层 1：Token 间通信
        x = x + self.mlp(self.ln2(x))                    # 子层 2：逐位置计算
        return x


# ---------------------------------------------------------------- model.py
@dataclass
class CausalLMOutput:
    logits: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    past_kv: object | None = None


class TinyLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)   # 公式 (3.18)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = LayerNorm(cfg.d_model, cfg.norm_eps, cfg.bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        # GPT-2 做法：残差分支的输出投影按 1/sqrt(2 n_l) 缩放
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

    def forward(
        self,
        input_ids,
        *,
        attention_mask=None,
        position_ids=None,
        past_kv=None,
        use_cache=False,
        labels=None,
        logits_to_keep=None,
    ):
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, "序列长度超过 max_seq_len"
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device)
        # 公式 (3.18)：Token 嵌入 + 位置嵌入
        x = self.tok_emb(input_ids) + self.pos_emb(position_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)                        # 公式 (3.19)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(                         # 公式 (3.20)
                logits.reshape(-1, self.cfg.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        # past_kv / use_cache / logits_to_keep 在第 4、18、21 章才生效
        return CausalLMOutput(logits=logits, loss=loss)


# ---------------------------------------------------------------- 实验
def copy_task(batch_size, seq_len, lag, vocab_size, generator=None):
    """前 lag 个 Token 随机，之后每个 Token 复制 lag 步之前的值。"""
    seq = torch.randint(0, vocab_size, (batch_size, lag), generator=generator)
    for t in range(lag, seq_len):
        seq = torch.cat([seq, seq[:, t - lag : t - lag + 1]], dim=1)
    return seq


def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    cfg = ModelConfig(vocab_size=16, d_model=64, n_layers=2, n_heads=4,
                      d_ff=256, max_seq_len=64)
    model = TinyLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    d, f, V, L = cfg.d_model, cfg.d_ff, cfg.vocab_size, cfg.max_seq_len
    exact = cfg.n_layers * (4 * d**2 + 2 * d * f + 9 * d + f) + 2 * d + 2 * V * d + L * d
    simple = cfg.n_layers * (4 * d**2 + 2 * d * f) + 2 * V * d
    print(f"参数量: {n_params:,}")
    print(f"精确公式 (3.22): {exact:,}    简化估算 (3.23): {simple:,}")

    lag, T, B = 8, 32, 64
    x = copy_task(B, T, lag, cfg.vocab_size, gen)

    # 形状与因果性检查
    model.eval()
    with torch.no_grad():
        out = model(x)
        print("logits 形状:", tuple(out.logits.shape))

        x2 = x.clone()
        x2[:, 10] = (x2[:, 10] + 1) % cfg.vocab_size
        out2 = model(x2)
        diff = (out.logits - out2.logits).abs().amax(dim=(0, 2))
        print(f"因果性: 位置 0..9 最大差异 = {diff[:10].max().item():.2e}, "
              f"位置 10..31 最大差异 = {diff[10:].max().item():.2e}")

        _, attn = model.blocks[0].attn(model.blocks[0].ln1(
            model.tok_emb(x) + model.pos_emb(torch.arange(T))), return_weights=True)
        row_sums = attn.sum(-1)
        upper = torch.triu(attn, diagonal=1)
        print(f"注意力: 行和范围 [{row_sums.min().item():.4f}, {row_sums.max().item():.4f}], "
              f"上三角绝对值最大 = {upper.abs().max().item():.2e}")

    # 训练：预测下一个 Token
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    for step in range(1, 401):
        batch_x = copy_task(B, T, lag, cfg.vocab_size, gen)
        inp, tgt = batch_x[:, :-1], batch_x[:, 1:]
        out = model(inp, labels=tgt)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        if step in (1, 100, 200, 300, 400):
            print(f"step {step:3d}  loss={out.loss.item():.4f}  "
                  f"ppl={math.exp(out.loss.item()):.2f}")

    model.eval()
    with torch.no_grad():
        test = copy_task(B, T, lag, cfg.vocab_size, gen)
        logits = model(test[:, :-1]).logits
        acc = (logits.argmax(-1) == test[:, 1:]).float().mean().item()
    floor = (lag - 1) / (T - 1) * math.log(cfg.vocab_size)
    ceiling = ((T - lag) + (lag - 1) / cfg.vocab_size) / (T - 1)
    print(f"测试集下一个 Token 准确率: {acc:.4f}（理论上界 {ceiling:.4f}）")
    print(f"理论损失下界: {floor:.4f}（前 {lag - 1} 个目标不可预测）")


if __name__ == "__main__":
    main()
