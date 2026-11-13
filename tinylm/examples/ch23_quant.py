"""第 23 章：量化（examples/ch23_quant.py）。"""

import math
import time

import torch

from ch08 import ModelConfig, TinyLM
from tinylm.inference.kv_cache import cache_bytes
from tinylm.inference.quant import (QuantConfig, attention_sqnr,
                                    compression_ratio, fake_quant,
                                    gptq_quantize, pack_int4,
                                    quantization_error, quantized_linear,
                                    smooth_quant, sqnr_db, storage_bits)

def granularity_demo():
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    w *= torch.logspace(-1, 1.5, 64)[:, None]       # 行尺度差异
    idx = torch.randint(0, 128, (64, 2))
    w[torch.arange(64)[:, None], idx] *= 20.0       # 行内离群值
    print("粒度与权重量化误差（4 比特，64×128）")
    print(f"{'粒度':>10s}{'信噪比 dB':>12s}{'最大误差':>12s}{'比特/权重':>12s}")
    for name, cfg in (("逐张量", QuantConfig(bits=4, granularity="tensor")),
                      ("逐通道", QuantConfig(bits=4, granularity="channel")),
                      ("逐组 128", QuantConfig(bits=4, granularity="group", group_size=128)),
                      ("逐组 64", QuantConfig(bits=4, granularity="group", group_size=64)),
                      ("逐组 32", QuantConfig(bits=4, granularity="group", group_size=32))):
        err = quantization_error(w, fake_quant(w, cfg))
        print(f"{name:>10s}{err['sqnr_db']:>12.2f}{err['max_abs']:>12.4f}"
              f"{storage_bits(cfg, 128):>12.3f}")

def bit_demo():
    torch.manual_seed(0)
    w = torch.randn(256, 1024)
    print("\n比特宽度与信噪比（逐组 64，256×1024）")
    print(f"{'比特':>6s}{'信噪比 dB':>12s}{'压缩比':>10s}")
    for bits in (2, 3, 4, 6, 8):
        cfg = QuantConfig(bits=bits, granularity="group", group_size=64)
        err = quantization_error(w, fake_quant(w, cfg))
        print(f"{bits:>6d}{err['sqnr_db']:>12.2f}"
              f"{compression_ratio(cfg, 1024):>10.2f}")

def packing_demo():
    q = (torch.randint(0, 256, (4096 * 4096,), dtype=torch.int32)
         % 16 - 8).to(torch.int8)
    packed = pack_int4(q)
    fp16 = q.numel() * 2
    int4 = packed.numel()
    cfg = QuantConfig(bits=4, granularity="group", group_size=64)
    print("\nINT4 打包（4096×4096 的单层权重）")
    print(f"  FP16 字节        {fp16 / 2**20:>10.1f} MiB")
    print(f"  INT4 码点字节    {int4 / 2**20:>10.1f} MiB")
    print(f"  含缩放因子       {storage_bits(cfg, 4096):>10.3f} 比特/权重")
    print(f"  理论压缩比       {compression_ratio(cfg, 4096):>10.2f}×")

def gptq_demo():
    torch.manual_seed(0)
    n, in_f, out_f = 1024, 128, 64
    z = torch.randn(n, 32)
    a = torch.randn(32, in_f) * 0.3
    x = z @ a + 0.1 * torch.randn(n, in_f)
    w = torch.randn(out_f, in_f) * 0.05
    h = x.t() @ x / n
    cfg = QuantConfig(bits=4, granularity="channel")
    ref = x @ w.t()
    w_rtn = fake_quant(w, cfg)
    w_gptq, _ = gptq_quantize(w, h, cfg)
    print("\nGPTQ 与 RTN（4 比特，线性层 64×128，1024 条校准输入）")
    print(f"{'方案':>8s}{'输出 MSE':>14s}{'输出信噪比 dB':>16s}")
    for name, wq in (("FP16", w), ("RTN", w_rtn), ("GPTQ", w_gptq)):
        y = x @ wq.t()
        print(f"{name:>8s}{(ref - y).pow(2).mean():>14.6f}"
              f"{sqnr_db(ref, y):>16.2f}")

def smooth_demo():
    torch.manual_seed(0)
    n, in_f, out_f = 2048, 128, 64
    x = torch.randn(n, in_f)
    x[:, :4] *= 30.0
    w = torch.randn(out_f, in_f) * 0.05
    ref = x @ w.t()
    cfg = QuantConfig(bits=8, granularity="channel")
    print("\nSmoothQuant（W8A8，激活有 4 个离群通道）")
    print(f"  激活最大绝对值 {x.abs().max():.2f}")
    base = (ref - quantized_linear(fake_quant(x, cfg), w, cfg)).pow(2).mean()
    print(f"{'α':>6s}{'激活最大绝对值':>16s}{'输出 MSE':>14s}{'信噪比 dB':>12s}")
    for alpha in (0.0, 0.3, 0.5, 0.7):
        xs, ws = smooth_quant(x, w, alpha)
        y = quantized_linear(fake_quant(xs, cfg), ws, cfg)
        print(f"{alpha:>6.1f}{xs.abs().max():>16.2f}"
              f"{(ref - y).pow(2).mean():>14.6f}{sqnr_db(ref, y):>12.2f}")
    print(f"  不平滑的输出 MSE {base:.6f}")

def kv_demo():
    torch.manual_seed(1)
    b, h, t, d = 2, 4, 64, 16
    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t, d)
    k[..., :2] *= 20.0
    v = torch.randn(b, h, t, d)
    print("\nKV Cache 量化（K 有两个通道离群）")
    print(f"{'精度':>10s}{'粒度':>10s}{'注意力输出信噪比 dB':>22s}")
    for bits, gran, gs in ((8, "channel", 64), (4, "group", 8)):
        cfg = QuantConfig(bits=bits, granularity=gran, group_size=gs)
        token = attention_sqnr(q, k, v, cfg, cfg, per_channel_k=False)
        channel = attention_sqnr(q, k, v, cfg, cfg, per_channel_k=True)
        tag = f"INT{bits}"
        print(f"{tag:>10s}{'逐 Token':>10s}{token:>22.2f}")
        print(f"{'':>10s}{'逐通道':>10s}{channel:>22.2f}")

def make_data(n, length, gen, prob=0.55, vocab=128):
    seq = torch.zeros(n, length, dtype=torch.long)
    seq[:, 0] = torch.randint(0, vocab, (n,), generator=gen)
    seq[:, 1] = torch.randint(0, vocab, (n,), generator=gen)
    for t in range(2, length):
        nxt = (seq[:, t - 1] * 7 + seq[:, t - 2] * 3 + 1) % vocab
        mask = torch.rand(n, generator=gen) < prob
        noise = torch.randint(0, vocab, (n,), generator=gen)
        seq[:, t] = torch.where(mask, nxt, noise)
    return seq

def model_demo():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    vocab = 128
    train = make_data(512, 48, gen)
    val = make_data(128, 48, gen)
    cfg = ModelConfig(vocab_size=vocab, d_model=64, n_layers=2, n_heads=4,
                      n_kv_heads=2, d_ff=128, max_seq_len=64, norm="rmsnorm",
                      ffn="swiglu", bias=False, pos_encoding="rope",
                      attention="gqa")
    model = TinyLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start = time.time()
    for _ in range(400):
        idx = train[torch.randint(0, 512, (32,), generator=gen)]
        opt.zero_grad()
        loss = model(idx[:, :-1], labels=idx[:, 1:]).loss
        loss.backward()
        opt.step()

    def evaluate():
        with torch.no_grad():
            return model(val[:, :-1], labels=val[:, 1:]).loss.item()

    base = evaluate()
    saved = [(p, p.detach().clone()) for p in model.parameters() if p.dim() >= 2]
    print(f"\nTinyLM 权重仅量化（{sum(p.numel() for p in model.parameters()):,} 参数，"
          f"训练 {time.time() - start:.1f} s）")
    print(f"{'配置':>16s}{'验证损失':>12s}{'Δ损失':>10s}{'困惑度倍数':>12s}")
    print(f"{'FP16':>16s}{base:>12.4f}{0.0:>10.4f}{1.000:>12.3f}")
    for name, quant_cfg in (
            ("W8 逐通道", QuantConfig(bits=8, granularity="channel")),
            ("W4 逐通道", QuantConfig(bits=4, granularity="channel")),
            ("W4 逐组 32", QuantConfig(bits=4, granularity="group", group_size=32)),
            ("W2 逐通道", QuantConfig(bits=2, granularity="channel"))):
        for p, orig in saved:
            p.data.copy_(orig)
        for p, _ in saved:
            p.data.copy_(fake_quant(p, quant_cfg))
        loss = evaluate()
        print(f"{name:>16s}{loss:>12.4f}{loss - base:>10.4f}"
              f"{math.exp(loss - base):>12.3f}")
    for p, orig in saved:
        p.data.copy_(orig)

def ledger():
    cfg = ModelConfig(vocab_size=32000, d_model=4096, n_layers=32, n_heads=32,
                      n_kv_heads=8, d_ff=11008, max_seq_len=4096,
                      norm="rmsnorm", ffn="swiglu", bias=False,
                      pos_encoding="rope", attention="gqa")
    params = 6_738_415_616
    bw = 3.35e12
    print("\n7B 模型解码带宽账本（T=4096，H100 3.35 TB/s，理想下界）")
    print(f"{'方案':>10s}{'权重 GB':>10s}{'KV GB':>10s}{'总 GB':>10s}"
          f"{'ms/token':>10s}{'Token/s':>10s}")
    kv16 = cache_bytes(cfg, 1, 4096, b_elem=2)
    for name, wb, kbb in (("FP16", 2.0, 2.0), ("INT8", 1.0, 1.0),
                          ("INT4", 0.5, 0.5)):
        total = params * wb + kv16 * (kbb / 2.0)
        ms = total / bw * 1e3
        print(f"{name:>10s}{params * wb / 1e9:>10.2f}{kv16 * (kbb / 2.0) / 1e9:>10.2f}"
              f"{total / 1e9:>10.2f}{ms:>10.3f}{1e3 / ms:>10.1f}")
    print("批 32、长上下文时只量化权重（KV 仍 FP16）")
    kv32 = cache_bytes(cfg, 32, 4096, b_elem=2)
    for name, wb, kbb in (("FP16", 2.0, 2.0), ("W4", 0.5, 2.0), ("W4+KV4", 0.5, 0.5)):
        total = params * wb + kv32 * (kbb / 2.0)
        print(f"{name:>10s}{params * wb / 1e9:>10.2f}{kv32 * (kbb / 2.0) / 1e9:>10.2f}"
              f"{total / 1e9:>10.2f}{total / bw * 1e3:>10.3f}{bw / total:>10.1f}")

if __name__ == "__main__":
    granularity_demo()
    bit_demo()
    packing_demo()
    gptq_demo()
    smooth_demo()
    kv_demo()
    model_demo()
    ledger()
