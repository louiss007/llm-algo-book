"""第 21 章：推理加速（examples/ch21_inference.py）。"""

import time

import torch

from ch08 import ModelConfig, TinyLM
from tinylm.inference.flash import dense_attention, flash_attention
from tinylm.inference.kv_cache import (KVCache, arithmetic_intensity,
                                       cache_bytes, cached_forward,
                                       cached_prefill, roofline_time)
from tinylm.inference.paged import fragmentation_table

def human(nbytes):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if nbytes < 1024 or unit == "TiB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes:.0f} B"
        nbytes /= 1024

def check_equivalence():
    cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=3, n_heads=8,
                      n_kv_heads=2, d_ff=128, max_seq_len=128,
                      norm="rmsnorm", ffn="swiglu", bias=False,
                      pos_encoding="rope", attention="gqa")
    torch.manual_seed(0)
    model = TinyLM(cfg).eval()
    prompt = torch.randint(0, 64, (1, 16))

    with torch.no_grad():
        full = model(prompt).logits
        prefill = cached_forward(model, prompt, KVCache(cfg))
        d_prefill = (full - prefill).abs().max().item()

        cache = KVCache(cfg)
        logits = cached_prefill(model, prompt, cache)
        ids = prompt
        d_decode = 0.0
        for _ in range(8):
            nxt = logits[:, -1:].argmax(-1)
            ids = torch.cat([ids, nxt], dim=1)
            logits = cached_forward(model, nxt, cache, logits_to_keep=1)
            ref = model(ids).logits[:, -1:]
            d_decode = max(d_decode, (logits - ref).abs().max().item())

        cache2 = KVCache(cfg)
        cached_forward(model, prompt[:, :7], cache2)
        chunked = cached_forward(model, prompt[:, 7:], cache2)
        d_chunk = (full[:, 7:] - chunked).abs().max().item()

    print("缓存等价性（最大绝对误差）")
    print(f"  整段预填充      {d_prefill:.2e}")
    print(f"  分块预填充      {d_chunk:.2e}")
    print(f"  逐 Token 解码    {d_decode:.2e}")

def kv_ledger():
    common = dict(vocab_size=32000, max_seq_len=32768, norm="rmsnorm",
                  ffn="swiglu", bias=False, pos_encoding="rope")
    cfgs = {
        "7B GQA (8 KV)": ModelConfig(d_model=4096, n_layers=32, n_heads=32,
                                     n_kv_heads=8, d_ff=14336,
                                     attention="gqa", **common),
        "13B MHA": ModelConfig(d_model=5120, n_layers=40, n_heads=40,
                               n_kv_heads=40, d_ff=13824,
                               attention="mha", **common),
        "MLA (V2 规模)": ModelConfig(d_model=5120, n_layers=60, n_heads=128,
                                    d_ff=12288, attention="mla", d_c=512,
                                    d_head_rope=64, **common),
    }
    print("\nKV Cache 账本（FP16，单序列）")
    print(f"{'配置':<16s}{'4K':>10s}{'32K':>10s}"
          f"{'每 GiB Token':>14s}{'24G 卡最大批':>12s}")
    for name, cfg in cfgs.items():
        b4 = cache_bytes(cfg, 1, 4096)
        b32 = cache_bytes(cfg, 1, 32768)
        per_gib = (1 << 30) / cache_bytes(cfg, 1, 1)
        max_b = int(20 * (1 << 30) // b4)
        print(f"{name:<16s}{human(b4):>10s}{human(b32):>10s}"
              f"{per_gib:>14,.0f}{max_b:>12d}")

@torch.no_grad()
def speed_compare():
    cfg = ModelConfig(vocab_size=256, d_model=128, n_layers=4, n_heads=8,
                      n_kv_heads=2, d_ff=256, max_seq_len=512,
                      norm="rmsnorm", ffn="swiglu", bias=False,
                      pos_encoding="rope", attention="gqa")
    torch.manual_seed(0)
    model = TinyLM(cfg).eval()
    prompt = torch.randint(0, 256, (1, 32))
    new = 96

    t0 = time.perf_counter()
    ids = prompt
    for _ in range(new):
        logits = model(ids).logits
        ids = torch.cat([ids, logits[:, -1:].argmax(-1)], dim=1)
    full_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    cache = KVCache(cfg)
    logits = cached_prefill(model, prompt, cache)
    for _ in range(new):
        nxt = logits[:, -1:].argmax(-1)
        logits = cached_forward(model, nxt, cache, logits_to_keep=1)
    cache_time = time.perf_counter() - t0

    print(f"\n纯 PyTorch CPU 实测（提示 {prompt.size(1)}，生成 {new}）")
    print(f"  每步重算全序列  {full_time:8.3f} s")
    print(f"  KV Cache 增量   {cache_time:8.3f} s")
    print(f"  加速比          {full_time / cache_time:8.2f}×")

def flash_check():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 8, 256, 64) for _ in range(3))
    dense = dense_attention(q, k, v)
    tiled = flash_attention(q, k, v, block_q=64, block_k=64)
    print("\nFlashAttention 与稠密注意力")
    print(f"  T=256 最大绝对误差 { (dense - tiled).abs().max().item():.2e}")
    print(f"  输出形状          {tuple(tiled.shape)}")

def paged_ledger():
    lengths = [2048, 17, 4000, 256, 1, 512]
    row = fragmentation_table(lengths, 16)
    print("\n分页分配 vs 朴素预分配（block=16）")
    print(f"  有效 Token          {row['tokens']}")
    print(f"  分页预留槽位        {row['paged_slots']}，利用率 "
          f"{row['paged_util']:.3%}")
    print(f"  朴素预留槽位        {row['naive_slots']}，利用率 "
          f"{row['naive_util']:.3%}")

def roofline_ledger():
    cfg = ModelConfig(vocab_size=32000, d_model=4096, n_layers=32,
                      n_heads=32, n_kv_heads=8, d_ff=14336,
                      max_seq_len=32768, norm="rmsnorm", ffn="swiglu",
                      bias=False, pos_encoding="rope", attention="gqa")
    peak, bw = 989e12, 3.35e12          # H100 SXM FP16 稠密算力与显存带宽
    print("\n解码 roofline（7B，T=4096，H100 峰值 989 TFLOPS / 3.35 TB/s）")
    print(f"{'批':>4s}{'算术强度':>12s}{'搬运':>10s}{'下界':>10s}{'吞吐':>12s}")
    for batch in (1, 8, 32, 64):
        ai = arithmetic_intensity(cfg, batch, 4096)
        sec, flops, moved = roofline_time(cfg, batch, 4096, peak, bw)
        print(f"{batch:>4d}{ai:>10.1f} F/B{moved / 2**30:>8.1f}G"
              f"{sec * 1e3:>8.1f}ms{batch / sec:>10.0f} tok/s")

if __name__ == "__main__":
    check_equivalence()
    kv_ledger()
    speed_compare()
    flash_check()
    paged_ledger()
    roofline_ledger()
