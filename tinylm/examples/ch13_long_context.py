"""第 13 章：长上下文的算力、打包与长度外推实验（examples/ch13_long_context.py）。"""
import time

import torch
import torch.nn.functional as F

from ch04 import RoPEScalingConfig
from ch08 import ModelConfig, TinyLM
from tinylm.data.collate import pack_documents, token_utilization
from tinylm.eval.needle import (bucket_accuracy, needle_positions,
                                passkey_batch, retrieval_correct)
from tinylm.train import long_context as LC

VOCAB = 17
QUERY = 16
CHANCE = 1.0 / 16
T_TRAIN = 32
TARGET = 128
STEPS = 400
SEEDS = (0, 1, 2)
LENGTHS = (32, 64, 128)


def retrieval_batch(batch, seq_len, gen):
    x = torch.randint(0, QUERY, (batch, seq_len), generator=gen)
    x[:, -1] = QUERY
    y = torch.full((batch, seq_len), -100, dtype=torch.long)
    y[:, -1] = x[:, 0]
    return x, y


@torch.no_grad()
def evaluate(model, seq_len, batch=256, seed=7):
    gen = torch.Generator().manual_seed(seed)
    x, y = retrieval_batch(batch, seq_len, gen)
    model.eval()
    out = model(x, labels=y)
    acc = (out.logits[:, -1].argmax(-1) == y[:, -1]).float().mean().item()
    return out.loss.item(), acc


def train_one(regime, method, factor, seed):
    torch.manual_seed(seed)
    max_len = T_TRAIN if regime == "fixed" else TARGET
    cfg = ModelConfig(
        vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2, d_ff=128,
        max_seq_len=max_len, pos_encoding="rope", norm="rmsnorm",
        ffn="swiglu", bias=False,
        rope_scaling=RoPEScalingConfig(method=method, factor=factor),
    )
    model = TinyLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    gen = torch.Generator().manual_seed(1000 + seed)
    model.train()

    def train_at(seq_len, steps):
        for _ in range(steps):
            x, y = retrieval_batch(32, seq_len, gen)
            out = model(x, labels=y)
            opt.zero_grad()
            out.loss.backward()
            opt.step()

    if regime == "fixed":
        train_at(T_TRAIN, STEPS)
    elif regime == "mixed":
        for i in range(STEPS):
            train_at(LENGTHS[i % len(LENGTHS)], 1)
    elif regime == "curriculum":
        for seq_len, share in LC.length_schedule(T_TRAIN, TARGET, 3):
            train_at(seq_len, round(STEPS * share))
    return [evaluate(model, seq_len)[1] for seq_len in LENGTHS]


def extrapolation_table():
    configs = (
        ("固定 32", "none", 1.0, "fixed"),
        ("固定 32", "pi", 4.0, "fixed"),
        ("固定 32", "ntk", 4.0, "fixed"),
        ("固定 32", "yarn", 4.0, "fixed"),
        ("混合长度", "none", 1.0, "mixed"),
        ("课程", "none", 1.0, "curriculum"),
        ("课程", "pi", 4.0, "curriculum"),
    )
    print(f"\n首 Token 检索（训练 400 步，3 种子；随机水平 {CHANCE:.3f}）")
    print(f"{'训练方式':<10s}{'RoPE':<6s}{'T=32':>14s}{'T=64':>14s}"
          f"{'T=128':>14s}{'达标种子':>10s}{'有效长度':>8s}")
    for regime_name, method, factor, regime in configs:
        rows = torch.tensor([train_one(regime, method, factor, seed)
                             for seed in SEEDS])
        mean, std = rows.mean(0), rows.std(0)
        eff = LC.effective_context(LENGTHS, mean.tolist(), threshold=0.5)
        hit = int((rows[:, -1] >= 0.5).sum())
        cells = "".join(f"{m:>9.2f}±{s:.2f}" for m, s in zip(mean, std))
        print(f"{regime_name:<10s}{method:<6s}{cells}"
              f"{hit:>7d}/{len(SEEDS)}{eff:>8d}")


def ledger_table():
    d, layers, heads, kv_heads, d_ff = 4096, 32, 32, 8, 14336
    print("\n7B 模型账本（B=1，BF16；分数矩阵指不使用 FlashAttention 时的显式 T×T）")
    print(f"{'T':>8s}{'注意力 TFLOPs':>14s}{'线性 TFLOPs':>13s}"
          f"{'注意力占比':>10s}{'KV GiB':>9s}{'激活 GiB':>10s}"
          f"{'分数矩阵 GiB':>14s}")
    for T in (4096, 32768, 131072):
        a = LC.attention_flops(1, T, d, layers) / 1e12
        l = LC.linear_flops(1, T, d, d_ff, layers) / 1e12
        share = LC.attention_share(1, T, d, d_ff, layers)
        kv = LC.kv_cache_bytes(1, T, layers, kv_heads, d // heads) / 2**30
        act = LC.activation_bytes(1, T, d, layers, heads) / 2**30
        score = (LC.activation_bytes(1, T, d, layers, heads,
                                     materialize_scores=True)
                 - LC.activation_bytes(1, T, d, layers, heads)) / 2**30
        print(f"{T:>8d}{a:>14.1f}{l:>13.1f}{share:>10.1%}"
              f"{kv:>9.1f}{act:>10.1f}{score:>14.1f}")


def kv_model_table():
    models = (("7B", 32), ("70B", 80), ("405B", 126))
    print("\n单序列 BF16 KV Cache（GiB，8 个 KV 头、d_head=128）")
    print(f"{'模型':<6s}{'层数':>6s}" + "".join(f"{T//1024:>8d}k" for T in
                                          (4096, 32768, 131072)))
    for name, layers in models:
        cells = "".join(f"{LC.kv_cache_bytes(1, T, layers, 8, 128)/2**30:>8.1f}"
                        for T in (4096, 32768, 131072))
        print(f"{name:<6s}{layers:>6d}{cells}")


def packing_demo():
    gen = torch.Generator().manual_seed(0)
    lengths = (9, 12, 7, 15, 11, 8)
    docs = [torch.randint(0, VOCAB, (n,), generator=gen) for n in lengths]
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2,
                      d_ff=128, max_seq_len=128, pos_encoding="rope",
                      norm="rmsnorm", ffn="swiglu", bias=False)
    model = TinyLM(cfg).eval()
    batch = pack_documents(docs)

    def token_losses(input_ids, labels, mask=None, positions=None):
        out = model(input_ids[None], attention_mask=mask,
                    position_ids=positions)
        per_token = F.cross_entropy(out.logits[0], labels, reduction="none",
                                    ignore_index=-100)
        return out.logits[0], per_token, per_token[labels != -100].sum()

    logits_u, losses_u, sum_u = [], [], 0.0
    for doc in docs:
        lg, ls, s = token_losses(doc[:-1], doc[1:],
                                 positions=torch.arange(len(doc) - 1))
        logits_u.append(lg)
        losses_u.append(ls)
        sum_u = sum_u + s
    logits_u = torch.cat(logits_u)
    losses_u = torch.cat(losses_u)

    lg_p, ls_p, sum_p = token_losses(batch.input_ids, batch.labels,
                                     batch.attention_mask, batch.position_ids)
    lg_c, ls_c, _ = token_losses(batch.input_ids, batch.labels,
                                 positions=batch.position_ids)
    keep = batch.labels != -100
    mean_u = (losses_u.sum() / keep.sum()).item()
    mean_p = ls_p[keep].mean().item()
    mean_c = ls_c[keep].mean().item()
    n_pred = sum(lengths) - len(lengths)
    print(f"\n打包等价性（6 篇文档，长度 {'/'.join(map(str, lengths))}，"
          f"共 {n_pred} 个预测位置）")
    print(f"{'方式':<20s}{'平均损失':>10s}{'logits 最大偏差':>16s}"
          f"{'梯度最大偏差':>14s}")
    print(f"{'逐条前向':<20s}{mean_u:>10.4f}{0.0:>16.1e}{0.0:>14.1e}")
    print(f"{'打包 + 块对角掩码':<20s}{mean_p:>10.4f}"
          f"{(logits_u - lg_p[keep]).abs().max().item():>16.1e}", end="")
    model.zero_grad()
    sum_u.backward()
    grads_u = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    sum_p.backward()
    gdiff = max((gu - p.grad).abs().max().item()
                for gu, p in zip(grads_u, model.parameters()))
    print(f"{gdiff:>14.1e}")
    print(f"{'打包 + 仅因果掩码':<20s}{mean_c:>10.4f}"
          f"{(logits_u - lg_c[keep]).abs().max().item():>16.1e}"
          f"{'—':>14s}")
    pad, packed = token_utilization(lengths, 32).values()
    print(f"Token 利用率：逐条 padding {pad:.1%} → 打包 {packed:.1%}")


def needle_sanity():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=18, d_model=32, n_layers=2, n_heads=2,
                      d_ff=128, max_seq_len=TARGET, pos_encoding="rope",
                      norm="rmsnorm", ffn="swiglu", bias=False)
    model = TinyLM(cfg).eval()
    gen = torch.Generator().manual_seed(3)
    depths = needle_positions(TARGET, 8)
    positions = [depths[i % len(depths)] for i in range(512)]
    ids, targets = passkey_batch(512, TARGET, positions, 16, 17, 16, gen)
    with torch.no_grad():
        correct = retrieval_correct(model(ids).logits[:, -1], targets)
    edges = [0, 32, 64, 96, TARGET]
    print("\n单针评测口径自检（随机初始化模型，预期接近 1/16）")
    for lo, hi, acc, n in bucket_accuracy(positions, correct, edges):
        print(f"深度 {lo:>3d}-{hi-1:<3d}：准确率 {acc:.3f}（{n} 条）")


def main():
    t0 = time.time()
    ledger_table()
    kv_model_table()
    packing_demo()
    needle_sanity()
    extrapolation_table()
    print(f"\n总耗时 {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
