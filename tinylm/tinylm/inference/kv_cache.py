"""第 21 章：KV Cache 与增量解码（tinylm/inference/kv_cache.py）。"""
import math

import torch


class KVCache:
    """按层预分配的 K/V 缓冲区，形状 (B, n_kv_heads, max_len, d_head)。

    所有层共享一个 ``length`` 计数器：每层只写入新 Token 的 K/V，
    因此解码一步只做一次线性投影，不再重算历史 Token。
    """

    def __init__(self, cfg, batch_size=1, max_len=None, dtype=torch.float32,
                 device=None):
        self.n_layers = cfg.n_layers
        self.n_kv_heads = cfg.n_kv_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.batch_size = batch_size
        self.max_len = max_len or cfg.max_seq_len
        self.dtype = dtype
        self.device = device
        self.k = [None] * self.n_layers
        self.v = [None] * self.n_layers
        self.length = 0

    def _grow(self, layer, k):
        shape = (self.batch_size, self.n_kv_heads, self.max_len, self.d_head)
        self.k[layer] = torch.zeros(shape, dtype=self.dtype, device=k.device)
        self.v[layer] = torch.zeros(shape, dtype=self.dtype, device=k.device)

    def append(self, layer, k, v):
        """把新 Token 的 K/V 写到 ``length`` 处；不推进计数器。"""
        if self.k[layer] is None:
            self._grow(layer, k)
        start = self.length
        end = start + k.size(-2)
        assert end <= self.max_len, "KV Cache 溢出：增大 max_len"
        self.k[layer][:, :, start:end] = k
        self.v[layer][:, :, start:end] = v

    def advance(self, n):
        self.length += n

    def keys(self, layer):
        return self.k[layer][:, :, :self.length]

    def values(self, layer):
        return self.v[layer][:, :, :self.length]

    def reset(self):
        self.length = 0

def cache_bytes(cfg, batch_size, seq_len, b_elem=2):
    """(21.1)：KV Cache 显存；MLA 只存压缩潜向量与解耦 RoPE 键。"""
    if cfg.attention == "mla":
        per_token = cfg.d_c + cfg.d_head_rope
        return cfg.n_layers * batch_size * seq_len * per_token * b_elem
    per_token = 2 * cfg.n_kv_heads * (cfg.d_model // cfg.n_heads)
    n_attn = cfg.n_layers
    if cfg.block_type == "hybrid":
        n_attn = sum((i + 1) % cfg.hybrid_attn_every == 0
                     for i in range(cfg.n_layers))
    elif cfg.block_type != "attention":
        n_attn = 0
    return n_attn * batch_size * seq_len * per_token * b_elem

def cache_tokens_per_gib(cfg, b_elem=2):
    """反向计算：每 GiB 显存能缓存多少 Token。"""
    one = cache_bytes(cfg, 1, 1, b_elem)
    return (1 << 30) / one if one else float("inf")

def weight_bytes(cfg, b_elem=2):
    from ..scaling import model_params

    return model_params(cfg) * b_elem

def arithmetic_intensity(cfg, batch_size, seq_len, b_elem=2):
    """公式 (21.5)：解码一步的 FLOPs / 搬运字节数；批越大越接近算力上限。"""
    from ..scaling import decode_flops

    flops = batch_size * decode_flops(cfg, seq_len)
    moved = weight_bytes(cfg, b_elem) + cache_bytes(cfg, batch_size, seq_len,
                                                    b_elem)
    return flops / moved

def roofline_time(cfg, batch_size, seq_len, peak_flops, bandwidth,
                  b_elem=2):
    """公式 (21.6)：max(算力时间, 带宽时间)，返回 (秒, FLOPs, 字节)。"""
    from ..scaling import decode_flops

    flops = batch_size * decode_flops(cfg, seq_len)
    moved = weight_bytes(cfg, b_elem) + cache_bytes(cfg, batch_size, seq_len,
                                                    b_elem)
    return max(flops / peak_flops, moved / bandwidth), flops, moved

def ttft(prefill_tokens, prefill_tps):
    """公式 (21.17)：首 Token 延迟。"""
    return prefill_tokens / prefill_tps

def tpot(decode_seconds, decode_tokens):
    """公式 (21.18)：平均每输出 Token 时间。"""
    return decode_seconds / decode_tokens

def throughput(batch_size, tpot_seconds):
    """公式 (21.19)：稳态吞吐，条/秒。"""
    return batch_size / tpot_seconds

def cached_attention(attn, x, cache, layer, attention_mask=None,
                     position_ids=None):
    """投影新 Token，拼接历史 K/V，再对全部键做注意力。"""
    needed = ("q_proj", "k_proj", "v_proj", "out_proj",
              "n_heads", "n_kv_heads", "d_head", "pos_mode")
    assert all(hasattr(attn, name) for name in needed), "缓存实现只覆盖 MHA/GQA/MQA"
    B, T, _ = x.shape
    q = attn.q_proj(x).view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
    k = attn.k_proj(x).view(B, T, attn.n_kv_heads, attn.d_head).transpose(1, 2)
    v = attn.v_proj(x).view(B, T, attn.n_kv_heads, attn.d_head).transpose(1, 2)

    if attn.pos_mode == "rope":
        q, k = attn.rope(q, k, position_ids)

    cache.append(layer, k, v)
    L = cache.length + T
    k_all = cache.k[layer][:, :, :L]
    v_all = cache.v[layer][:, :, :L]
    if attn.repeat > 1:
        k_all = k_all.repeat_interleave(attn.repeat, dim=1)
        v_all = v_all.repeat_interleave(attn.repeat, dim=1)

    scores = q @ k_all.transpose(-2, -1) / math.sqrt(attn.d_head)
    if attn.pos_mode == "rope" and attn.rope.temp != 1.0:
        scores = scores / math.sqrt(attn.rope.temp)
    if attn.pos_mode == "alibi":
        pos = position_ids.reshape(-1)
        key_pos = torch.arange(L, device=x.device)
        dist = (pos[:, None] - key_pos[None, :]).clamp(min=0)
        scores = scores - attn.alibi.slopes[:, None, None] * dist[None, None]

    pos = position_ids.reshape(-1)
    key_pos = torch.arange(L, device=x.device)
    visible = key_pos[None, :] <= pos[:, None]
    scores = scores.masked_fill(~visible[None, None], float("-inf"))

    if attention_mask is not None:
        assert attention_mask.dim() == 2
        pad = attention_mask[:, None, None, :L] == 0
        scores = scores.masked_fill(pad, float("-inf"))

    attn_w = scores.softmax(dim=-1)
    out = attn_w @ v_all
    out = out.transpose(1, 2).reshape(B, T, attn.n_heads * attn.d_head)
    return attn.out_proj(out)

def cached_forward(model, input_ids, cache, *, attention_mask=None,
                   position_ids=None, logits_to_keep=None):
    """与 ``TinyLM.forward`` 等价，但读写 Cache；返回 logits。"""
    cfg = model.cfg
    assert cfg.block_type == "attention", "缓存前向只支持纯注意力块"
    B, T = input_ids.shape
    if position_ids is None:
        position_ids = torch.arange(cache.length, cache.length + T,
                                    device=input_ids.device)
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    if model.pos_mode in ("learned", "sine"):
        assert cache.length + T <= cfg.max_seq_len

    x = model.tok_emb(input_ids)
    if model.pos_mode == "learned":
        x = x + model.pos_emb(position_ids)
    elif model.pos_mode == "sine":
        x = x + model.sin_pe[position_ids]

    for layer, block in enumerate(model.blocks):
        assert all(hasattr(block, name)
                   for name in ("norm1", "attn", "mlp", "parallel"))
        if block.parallel:
            h = block.norm1(x)
            mix = cached_attention(block.attn, h, cache, layer,
                                   attention_mask, position_ids)
            delta = block.mlp(h)
            if isinstance(delta, tuple):
                delta = delta[0]
            x = x + mix + delta
        else:
            x = x + cached_attention(block.attn, block.norm1(x), cache, layer,
                                     attention_mask, position_ids)
            out = block.mlp(block.norm2(x))
            x = x + (out[0] if isinstance(out, tuple) else out)

    cache.advance(T)
    x = model.ln_f(x)
    if logits_to_keep:
        x = x[:, -logits_to_keep:]
    return model.lm_head(x)

def cached_prefill(model, input_ids, cache, **kwargs):
    """整段提示词一次写入 Cache，返回最后位置的 logits。"""
    return cached_forward(model, input_ids, cache, logits_to_keep=1, **kwargs)

@torch.no_grad()
def cached_greedy_decode(model, input_ids, max_new_tokens, cache=None,
                         eos_token_id=None):
    """预填充一次，然后每步只喂 1 个 Token。"""
    if cache is None:
        cache = KVCache(model.cfg, batch_size=input_ids.size(0))
    tokens = input_ids
    logits = cached_prefill(model, tokens, cache)
    steps = 0
    for _ in range(max_new_tokens):
        nxt = logits[:, -1:].argmax(dim=-1)
        tokens = torch.cat([tokens, nxt], dim=1)
        steps += 1
        if eos_token_id is not None and bool((nxt == eos_token_id).all()):
            break
        if cache.length >= cache.max_len:
            break
        logits = cached_forward(model, nxt, cache, logits_to_keep=1)
    return tokens, steps
