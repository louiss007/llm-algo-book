"""注意力层（tinylm/attention.py）。

第 3 章多头注意力；第 4 章加入位置分派；第 6 章扩展 MQA/GQA/MLA。
"""
import math

import torch
import torch.nn as nn

from .position import ALiBi, RoPE


def _causal_mask(max_seq_len):
    return torch.triu(
        torch.full((max_seq_len, max_seq_len), float("-inf")),
        diagonal=1,
    )


def _apply_attention_mask(scores, attention_mask):
    if attention_mask is None:
        return scores
    if attention_mask.dim() == 4:                     # 块对角文档掩码
        return scores.masked_fill(attention_mask == 0, float("-inf"))
    pad = attention_mask[:, None, None, :] == 0
    return scores.masked_fill(pad, float("-inf"))


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
        self.register_buffer("causal_mask", _causal_mask(cfg.max_seq_len),
                             persistent=False)

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
        scores = _apply_attention_mask(scores, attention_mask)

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
        self.register_buffer("causal_mask", _causal_mask(cfg.max_seq_len),
                             persistent=False)

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
        scores = _apply_attention_mask(scores, attention_mask)

        attn = scores.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)
        return self.out_proj(out)


def make_attention(cfg):
    if cfg.attention == "mla":
        return LatentAttention(cfg)
    return VariantAttention(cfg)


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


def kv_cache_bytes(cfg, T, batch=1, dtype_bytes=2):
    """公式 (6.6)、(6.11)：推理时 KV Cache 的字节数。"""
    if cfg.attention == "mla":
        per_token_layer = (cfg.d_c + cfg.d_head_rope) * dtype_bytes
    else:
        per_token_layer = 2 * cfg.n_kv_heads * (cfg.d_model // cfg.n_heads) * dtype_bytes
    return batch * T * cfg.n_layers * per_token_layer
