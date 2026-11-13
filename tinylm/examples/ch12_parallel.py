"""第 12 章：分布式训练的显存、通信与等价性实验（examples/ch12_parallel.py）。"""
import copy
import math
import time

import torch
import torch.nn as nn

import ch10
import ch11
from ch08 import ModelConfig, TinyLM
from tinylm.train import optim as O
from tinylm.train import parallel as P


def make_model(vocab_size, seed):
    cfg = ModelConfig(vocab_size=vocab_size, d_model=32, n_layers=2, n_heads=4,
                      d_ff=64, max_seq_len=64, norm="rmsnorm", ffn="swiglu",
                      bias=False, pos_encoding="rope", attention="gqa",
                      n_kv_heads=2)
    torch.manual_seed(seed)
    return TinyLM(cfg)


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def flat_grad(model):
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


def assign_flat(model, flat):
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[offset:offset + n].reshape(p.shape))
            offset += n


def make_batches(stream, steps, batch, seq, seed):
    gen = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(steps):
        starts = torch.randint(0, len(stream) - seq - 1, (batch,), generator=gen)
        x = torch.stack([stream[s:s + seq] for s in starts])
        y = torch.stack([stream[s + 1:s + seq + 1] for s in starts])
        batches.append((x, y))
    return batches


def optimizer(params, lr):
    if isinstance(params, nn.Module):
        params = list(params.parameters())
    elif isinstance(params, torch.Tensor):
        params = [params]
    groups = [{"params": list(params), "weight_decay": 0.01}]
    return O.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def train_full(model, opt, batches):
    losses = []
    for x, y in batches:
        opt.zero_grad()
        out = model(x, labels=y)
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())
    return losses


def train_accum(model, opt, batches, accum):
    losses = []
    for x, y in batches:
        opt.zero_grad()
        xs, ys = x.chunk(accum), y.chunk(accum)
        step_loss = 0.0
        for xb, yb in zip(xs, ys):
            out = model(xb, labels=yb)
            (out.loss / accum).backward()          # 按 (12.7) 的全局批归一化
            step_loss += out.loss.item() / accum
        opt.step()
        losses.append(step_loss)
    return losses


def average_grads(replicas):
    grads = [flat_grad(m) for m in replicas]
    avg = torch.stack(grads).mean(dim=0)
    offset = 0
    for m in replicas:
        for p in m.parameters():
            n = p.numel()
            p.grad.copy_(avg[offset:offset + n].reshape(p.shape))
            offset += n
        offset = 0


def train_dp(replicas, opts, batches):
    losses = []
    for x, y in batches:
        for m in replicas:
            m.zero_grad(set_to_none=True)
        step_losses = []
        for m, xb, yb in zip(replicas, x.chunk(len(replicas)),
                            y.chunk(len(replicas))):
            out = m(xb, labels=yb)
            out.loss.backward()
            step_losses.append(out.loss.item())
        average_grads(replicas)                     # (12.5) all-reduce 后平均
        for opt in opts:
            opt.step()
        losses.append(sum(step_losses) / len(step_losses))
    return losses


def make_zero_shards(model, dp, lr):
    flat = flat_params(model).clone()
    n = flat.numel()
    pad = (-n) % dp
    padded = torch.cat([flat, flat.new_zeros(pad)])
    shards = [nn.Parameter(c.clone()) for c in padded.chunk(dp)]
    opts = [optimizer(s, lr) for s in shards]
    return shards, opts, n, pad


def train_zero1(replicas, opts, shards, shard_opts, n, pad, batches):
    losses = []
    dp = len(replicas)
    for x, y in batches:
        for m in replicas:
            m.zero_grad(set_to_none=True)
        step_losses = []
        for m, xb, yb in zip(replicas, x.chunk(dp), y.chunk(dp)):
            out = m(xb, labels=yb)
            out.loss.backward()
            step_losses.append(out.loss.item())
        average_grads(replicas)
        grad = flat_grad(replicas[0])
        grad_pad = torch.cat([grad, grad.new_zeros(pad)])
        for shard, opt, g in zip(shards, shard_opts, grad_pad.chunk(dp)):
            shard.grad = g.clone()
            opt.step()
        full = P.all_gather_shards(shards)[0][:n]   # 分片更新后聚合
        for m in replicas:
            assign_flat(m, full)
        losses.append(sum(step_losses) / len(step_losses))
    return losses


def run_seed(docs, val_texts, vocab, seed, steps=50, lr=2e-3):
    stream = ch10.encode_stream(docs, vocab)
    batches = make_batches(stream, steps, 16, 32, seed)
    base = make_model(len(vocab), seed)
    results = {}

    dense = copy.deepcopy(base)
    dense_losses = train_full(dense, optimizer(dense, lr), batches)
    results["整批 B=16"] = (dense, dense_losses)

    accum = copy.deepcopy(base)
    accum_losses = train_accum(accum, optimizer(accum, lr), batches, 4)
    results["梯度累积 A=4"] = (accum, accum_losses)

    replicas = [copy.deepcopy(base) for _ in range(4)]
    opts = [optimizer(m, lr) for m in replicas]
    dp_losses = train_dp(replicas, opts, batches)
    results["数据并行 DP=4"] = (replicas[0], dp_losses)

    replicas = [copy.deepcopy(base) for _ in range(4)]
    opts = [optimizer(m, lr) for m in replicas]
    shards, shard_opts, n, pad = make_zero_shards(replicas[0], 4, lr)
    zero_losses = train_zero1(replicas, opts, shards, shard_opts, n, pad,
                              batches)
    results["ZeRO-1 分片"] = (replicas[0], zero_losses)
    shard_ratio = ((n + pad) / 4) / n

    dense_flat = flat_params(dense)
    finals = []
    for name, (model, losses) in results.items():
        diff = (flat_params(model) - dense_flat).abs().max().item()
        val = ch10.evaluate(model, val_texts, vocab)
        finals.append((name, losses[-5:], val, diff))
    return finals, shard_ratio


def equivalence_table(docs, val_texts, vocab, seeds=(0, 1, 2)):
    rows = {}
    ratios = []
    for seed in seeds:
        finals, ratio = run_seed(docs, val_texts, vocab, seed)
        ratios.append(ratio)
        for name, last, val, diff in finals:
            rows.setdefault(name, []).append((val, diff, last))
    print("\n等价性（50 步，3 个种子，与整批训练的偏差）")
    print(f"{'模式':<16s}{'验证损失':>16s}{'最大参数偏差':>18s}")
    for name, vals in rows.items():
        v = torch.tensor([x[0] for x in vals])
        d = max(x[1] for x in vals)
        print(f"{name:<16s}{v.mean().item():.4f} ± {v.std().item():.4f}"
              f"{d:>16.2e}")
    print(f"每卡优化器状态占参数比例：{min(ratios):.3f}（4 卡分片，含 padding）")


def memory_table():
    print("\n7B 模型显存账本（BF16 权重与梯度、FP32 主权重与 Adam 状态，8 卡）")
    print(f"{'ZeRO':<8s}{'状态 GB':>12s}{'激活 GB':>12s}{'合计 GB':>12s}")
    for stage in range(4):
        cfg = P.ParallelConfig(dp=8, tp=1, pp=1, zero_stage=stage,
                               micro_batch=1, seq_len=4096,
                               d_model=4096, n_layers=32)
        m = P.memory_ledger(7e9, cfg, "bf16", activation_coef=12.0)
        print(f"{stage:<8d}{m['state']/1e9:>12.1f}"
              f"{m['activation']/1e9:>12.1f}{m['total']/1e9:>12.1f}")


def comm_table():
    print("\n通信账本（7B，BF16 激活，单步 4M Token，DP=64）")
    dp_cfg = P.ParallelConfig(dp=64, micro_batch=1, grad_accum=16, seq_len=4096,
                              d_model=4096, n_layers=32)
    v_dp = P.dp_grad_bytes(7e9, dp_cfg, "bf16")
    compute = 6 * 7e9 * 4e6
    t_comp = compute / (64 * 400e12)
    t_dp = v_dp / 50e9
    print(f"DP 梯度 all-reduce 每卡 {v_dp/1e9:.2f} GB，"
          f"按 50 GB/s 约 {t_dp:.2f} s")
    print(f"整步计算 {compute:.2e} FLOPs，64 卡 × 400 TFLOP/s 约 {t_comp:.2f} s")
    print(f"通信/计算 = {t_dp/t_comp:.1%}")

    tp_cfg = P.ParallelConfig(tp=8, micro_batch=1, seq_len=4096,
                              d_model=4096, n_layers=32)
    v_tp = P.tp_block_bytes(tp_cfg, "bf16") * 32
    print(f"TP=8 每层 {P.tp_block_bytes(tp_cfg)/1e6:.1f} MB，"
          f"32 层共 {v_tp/1e9:.2f} GB/步")
    for pp, m in ((16, 16), (16, 64), (16, 256)):
        print(f"PP={pp}、在途微批 {m:>3d}：气泡 {P.pp_bubble_fraction(pp, m):.1%}")
    ep = P.ep_all_to_all_bytes(4096, 2, 4096, 64, 2) * 32
    print(f"EP=64、每卡 4096 Token、Top-2、32 层：all-to-all {ep/1e9:.2f} GB/步")


def precision_table():
    print("\n精度与每参数状态字节")
    print(f"{'dtype':<10s}{'状态字节':>10s}{'最大':>12s}{'最小正规数':>14s}"
          f"{'尾数位':>8s}")
    mantissa = {"fp32": 23, "fp16": 10, "bf16": 7, "fp8": 3}
    for name, torch_dtype in (("fp32", torch.float32), ("fp16", torch.float16),
                              ("bf16", torch.bfloat16),
                              ("fp8", torch.float8_e4m3fn)):
        info = torch.finfo(torch_dtype)
        print(f"{name:<10s}{P.param_state_bytes(name):>10d}"
              f"{info.max:>12.3g}{info.tiny:>14.3g}{mantissa[name]:>8d}")


def loss_scaling_demo():
    g = 1e-8
    h = torch.tensor(g, dtype=torch.float16).item()
    scaled = torch.tensor(g * 1024.0, dtype=torch.float16).item()
    print("\n损失缩放（梯度 1e-8）")
    print(f"直接转 FP16：{h:.3e}；乘 1024 后转 FP16：{scaled:.3e}，"
          f"再除回：{scaled/1024:.3e}")
    scaler = P.LossScaler(init_scale=8.0, growth_interval=2)
    print(f"初始 scale={scaler.scale:.0f}")
    for i in range(2):
        scaler.update(True)
    print(f"连续 2 步正常后 scale={scaler.scale:.0f}")
    scaler.update(False)
    print(f"检测到溢出后 scale={scaler.scale:.0f}，"
          f"跳过步数={scaler.skipped_steps}")


def main():
    t0 = time.time()
    docs, val_texts, vocab = ch11.build_data()
    equivalence_table(docs, val_texts, vocab)
    memory_table()
    comm_table()
    precision_table()
    loss_scaling_demo()
    print(f"\n总耗时 {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
