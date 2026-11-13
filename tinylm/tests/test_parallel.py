"""第 12 章自测：显存账本、通信量、分片集合通信、张量并行与损失缩放。"""
import math

import pytest
import torch

from tinylm.train import optim as O
from tinylm.train import parallel as P


def cfg(**kw):
    base = dict(dp=8, tp=2, pp=1, sp=1, zero_stage=0, micro_batch=1,
                grad_accum=1, seq_len=1024, d_model=64, n_layers=8)
    base.update(kw)
    return P.ParallelConfig(**base)


def test_param_state_bytes():
    assert P.param_state_bytes("fp32") == 16
    assert P.param_state_bytes("fp16") == 16
    assert P.param_state_bytes("bf16") == 16
    # FP8 只压缩权重与激活；主权重与优化器状态仍是 FP32
    assert P.param_state_bytes("fp8") == 15


def test_zero_stage_memory_7b():
    model = 7e9
    gb = 1e9
    assert P.zero_state_bytes(model, 8, 0, "bf16") / gb == pytest.approx(112.0)
    assert P.zero_state_bytes(model, 8, 1, "bf16") / gb == pytest.approx(38.5)
    assert P.zero_state_bytes(model, 8, 2, "bf16") / gb == pytest.approx(26.25)
    assert P.zero_state_bytes(model, 8, 3, "bf16") / gb == pytest.approx(14.0)


def test_zero_memory_is_monotone():
    model = 1e9
    values = [P.zero_state_bytes(model, 16, s, "bf16") for s in range(4)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_memory_ledger_activation_and_recompute():
    c = cfg(tp=1, sp=1, pp=2, micro_batch=2, seq_len=128, d_model=64, n_layers=8)
    plain = P.memory_ledger(1e6, c, "bf16", activation_coef=12.0)
    recompute = P.memory_ledger(1e6, c, "bf16", activation_coef=12.0,
                                recompute=True)
    assert plain["total"] == pytest.approx(plain["state"] + plain["activation"])
    assert recompute["activation"] < plain["activation"]
    # 每卡层数 4、序列 128、微批 2、宽度 64、系数 12、每元素 2 字节
    expected = 12.0 * 4 * 2 * 128 * 64 * 2
    assert plain["activation"] == pytest.approx(expected)


def test_ring_all_reduce_volume():
    assert P.ring_all_reduce_bytes(100, 1) == 0
    assert P.ring_all_reduce_bytes(100, 4) == pytest.approx(150.0)
    assert P.ring_all_reduce_bytes(100, 64) == pytest.approx(196.875)


def test_dp_and_tp_bytes():
    c = cfg(dp=4, tp=2, micro_batch=1, seq_len=8, d_model=4)
    # 1000 个参数、BF16 梯度 2 字节、4 卡：2*(3/4)*2000
    assert P.dp_grad_bytes(1000, c, "bf16") == pytest.approx(3000.0)
    # 激活 1*8*4*2=64 字节；每次 all-reduce 2*(1/2)*64=64；每块 4 次
    assert P.tp_block_bytes(c, "bf16") == pytest.approx(256.0)


def test_pipeline_bubble():
    assert P.pp_bubble_fraction(1, 8) == 0
    assert P.pp_bubble_fraction(4, 8) == pytest.approx(3 / 11)
    assert P.pp_bubble_fraction(4, 32) < 0.5 * P.pp_bubble_fraction(4, 8)
    with pytest.raises(ValueError):
        P.pp_bubble_fraction(0, 8)


def test_global_batch_and_lr_scaling():
    assert P.global_batch_size(4, 2, 8) == 64
    assert P.scale_lr(3e-4, 64, 8, 1.0) == pytest.approx(2.4e-3)
    assert P.scale_lr(3e-4, 64, 8, 0.5) == pytest.approx(3e-4 * math.sqrt(8))
    with pytest.raises(ValueError):
        P.scale_lr(3e-4, 64, 8, 1.5)


def test_all_reduce_and_shard_roundtrip():
    torch.manual_seed(0)
    parts = [torch.randn(2, 3) for _ in range(4)]
    reduced = P.all_reduce_sum(parts)
    expected = torch.stack(parts).sum(dim=0)
    assert all(torch.allclose(r, expected) for r in reduced)
    gathered = P.shard_reduce_gather(parts)
    assert all(torch.allclose(g, expected) for g in gathered)
    # 元素数不能被卡数整除时，padding 不能污染结果
    odd = [torch.arange(5.0) + i for i in range(3)]
    out = P.shard_reduce_gather(odd)
    assert torch.allclose(out[0], torch.stack(odd).sum(dim=0))


def test_column_and_row_parallel_equal_dense():
    torch.manual_seed(1)
    x = torch.randn(2, 3)
    w = torch.randn(3, 6)
    world = 3
    col = P.column_parallel_linear(x, w, world)
    assert torch.allclose(torch.cat(col, dim=1), x @ w, atol=1e-6)
    row = P.row_parallel_linear(list(x.chunk(world, dim=1)), w)
    assert torch.allclose(row, x @ w, atol=1e-6)


def test_tensor_parallel_mlp_equal_dense():
    torch.manual_seed(2)
    x = torch.randn(2, 4)
    w1 = torch.randn(4, 8)
    w2 = torch.randn(8, 4)
    world = 2
    hidden = [torch.relu(h) for h in P.column_parallel_linear(x, w1, world)]
    out = P.row_parallel_linear(hidden, w2)
    assert torch.allclose(out, torch.relu(x @ w1) @ w2, atol=1e-6)


def test_loss_scaler_overflow_and_growth():
    s = P.LossScaler(init_scale=8.0, growth_interval=2)
    assert s.update(True) is True and s.scale == 8.0
    assert s.update(True) is True and s.scale == 16.0
    assert s.update(False) is False and s.scale == 8.0
    assert s.good_steps == 0 and s.skipped_steps == 1


def test_loss_scaler_unscale_detects_inf():
    s = P.LossScaler(init_scale=4.0)
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([8.0, -4.0])
    assert s.unscale_([p]) is True
    assert torch.allclose(p.grad, torch.tensor([2.0, -1.0]))
    p.grad = torch.tensor([float("inf"), 1.0])
    assert s.unscale_([p]) is False


def test_shard_flat_roundtrip():
    t = torch.arange(8.0)
    parts = [P.shard_flat(t, 4, r) for r in range(4)]
    assert torch.equal(torch.cat(parts), t)
    with pytest.raises(ValueError):
        P.shard_flat(torch.arange(7.0), 4, 0)


def test_sharded_adamw_equals_dense():
    torch.manual_seed(3)
    full = torch.randn(16)
    grad = torch.randn(16)
    dense = torch.nn.Parameter(full.clone())
    dense.grad = grad.clone()
    O.AdamW([dense], lr=0.05, betas=(0.9, 0.95), eps=1e-8,
            weight_decay=0.01).step()

    world = 4
    shards = [torch.nn.Parameter(x.clone()) for x in full.chunk(world)]
    for shard, g in zip(shards, grad.chunk(world)):
        shard.grad = g.clone()
        O.AdamW([shard], lr=0.05, betas=(0.9, 0.95), eps=1e-8,
                weight_decay=0.01).step()
    merged = P.merge_shards(shards)
    assert torch.allclose(merged, dense.detach(), atol=1e-6)


def test_invalid_configs():
    with pytest.raises(ValueError):
        cfg(zero_stage=4)
    with pytest.raises(ValueError):
        cfg(tp=4, sp=2)
    with pytest.raises(ValueError):
        cfg(d_model=30, tp=4)
    with pytest.raises(ValueError):
        cfg(n_layers=7, pp=2)
    with pytest.raises(ValueError):
        P.all_reduce_sum([torch.zeros(2), torch.zeros(3)])
