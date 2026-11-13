"""第 12 章：分布式训练的显存、通信与混合精度工具（tinylm/train/parallel.py）。"""
import math
from dataclasses import dataclass

import torch


# 每种精度下每参数占用的字节数：权重、梯度、FP32 主权重、单个 Adam 矩
# FP32 训练没有额外的主权重副本，master=0
PRECISION_BYTES = {
    "fp32": {"weight": 4, "grad": 4, "master": 0, "moment": 4, "act": 4},
    "tf32": {"weight": 4, "grad": 4, "master": 0, "moment": 4, "act": 4},
    "fp16": {"weight": 2, "grad": 2, "master": 4, "moment": 4, "act": 2},
    "bf16": {"weight": 2, "grad": 2, "master": 4, "moment": 4, "act": 2},
    "fp8": {"weight": 1, "grad": 2, "master": 4, "moment": 4, "act": 1},
}


@dataclass(frozen=True)
class ParallelConfig:
    """并行度与账本所需的形状参数；不改变 TinyLM 的结构。"""

    dp: int = 1              # 数据并行组大小，也是 ZeRO 分片组大小
    tp: int = 1              # 张量并行度
    pp: int = 1              # 流水并行段数
    sp: int = 1              # 序列并行度，1 表示关闭；只允许 1 或 tp
    zero_stage: int = 0      # 0/1/2/3
    micro_batch: int = 1     # 每卡的微批大小
    grad_accum: int = 1      # 梯度累积的微步数，也是流水线中的在途微批数
    seq_len: int = 1024
    d_model: int = 512
    n_layers: int = 8

    def __post_init__(self):
        for name in ("dp", "tp", "pp", "sp", "micro_batch", "grad_accum",
                     "seq_len", "d_model", "n_layers"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} 必须为正整数")
        if self.zero_stage not in (0, 1, 2, 3):
            raise ValueError("zero_stage 必须是 0、1、2 或 3")
        if self.sp not in (1, self.tp):
            raise ValueError("序列并行度必须为 1 或等于张量并行度")
        if self.d_model % self.tp != 0:
            raise ValueError("d_model 必须被张量并行度整除")
        if self.n_layers % self.pp != 0:
            raise ValueError("n_layers 必须被流水并行段数整除")

    @property
    def world_size(self):
        return self.dp * self.tp * self.pp

    @property
    def global_batch(self):
        """公式 (12.7)：全局批大小。"""
        return self.micro_batch * self.grad_accum * self.dp

    @property
    def tokens_per_step(self):
        return self.global_batch * self.seq_len


def param_state_bytes(precision="bf16"):
    """公式 (12.1)：每参数的权重、梯度、主权重和两份优化器状态之和。"""
    if precision not in PRECISION_BYTES:
        raise ValueError(f"未知精度 {precision}")
    b = PRECISION_BYTES[precision]
    return b["weight"] + b["grad"] + b["master"] + 2 * b["moment"]


def zero_state_bytes(P, dp, stage, precision="bf16"):
    """公式 (12.4)：ZeRO 各阶段每卡的模型状态字节数。"""
    if P <= 0 or dp < 1:
        raise ValueError("P 必须为正，dp 必须不小于 1")
    b = PRECISION_BYTES[precision]
    full = param_state_bytes(precision)
    if stage == 0:
        per_param = full
    elif stage == 1:                          # 只分片优化器状态与主权重
        per_param = b["weight"] + b["grad"] + (b["master"] + 2 * b["moment"]) / dp
    elif stage == 2:                          # 再分片梯度
        per_param = b["weight"] + (b["grad"] + b["master"] + 2 * b["moment"]) / dp
    elif stage == 3:                          # 再分片参数
        per_param = full / dp
    else:
        raise ValueError("stage 必须是 0、1、2 或 3")
    return P * per_param


def memory_ledger(P, cfg: ParallelConfig, precision="bf16",
                  activation_coef=12.0, recompute=False):
    """公式 (12.1)–(12.3)：模型状态、激活与合计的显存账本（字节）。"""
    b = PRECISION_BYTES[precision]
    state = zero_state_bytes(P, cfg.dp, cfg.zero_stage, precision)
    layers = cfg.n_layers / cfg.pp
    seq = cfg.seq_len / cfg.sp
    act = (activation_coef * layers * cfg.micro_batch * seq
           * cfg.d_model * b["act"])
    if recompute:
        act /= math.sqrt(layers)              # 激活重算：显存降，算力升
    return {
        "state": state,
        "activation": act,
        "total": state + act,
        "per_param_state": state / P,
    }


def ring_all_reduce_bytes(size_bytes, world):
    """环形 all-reduce 每卡收发量：2(N-1)/N 倍张量大小。"""
    if world < 1:
        raise ValueError("world 必须为正")
    return 2 * (world - 1) / world * size_bytes


def dp_grad_bytes(P, cfg: ParallelConfig, precision="bf16"):
    """公式 (12.5)：一步数据并行梯度 all-reduce 的每卡字节数。"""
    grad = P * PRECISION_BYTES[precision]["grad"]
    return ring_all_reduce_bytes(grad, cfg.dp)


def tp_block_bytes(cfg: ParallelConfig, precision="bf16", collectives=4):
    """公式 (12.9)：每个 Transformer 块在张量并行下的通信字节数。

    前向两个 row-parallel 输出各一次 all-reduce；反向两个 column-parallel
    的输入梯度各一次 all-reduce。每次 all-reduce 的量按环形算法计。
    """
    act = (cfg.micro_batch * cfg.seq_len * cfg.d_model
           * PRECISION_BYTES[precision]["act"])
    return collectives * ring_all_reduce_bytes(act, cfg.tp)


def pp_bubble_fraction(pp, micro_batches):
    """公式 (12.10)：GPipe/1F1B 同步流水线的气泡占比。"""
    if pp < 1 or micro_batches < 1:
        raise ValueError("pp 与 micro_batches 必须为正")
    return (pp - 1) / (micro_batches + pp - 1)


def ep_all_to_all_bytes(n_tokens, top_k, d_model, ep, element_bytes=2):
    """公式 (12.11)：每层专家并行的 all-to-all 字节数（发出与收回各一份）。"""
    if min(n_tokens, top_k, d_model, ep) < 1:
        raise ValueError("参数必须为正")
    return 2 * n_tokens * top_k * d_model * element_bytes


def global_batch_size(micro_batch, grad_accum, dp):
    """公式 (12.7) 的函数形式。"""
    if min(micro_batch, grad_accum, dp) < 1:
        raise ValueError("参数必须为正")
    return micro_batch * grad_accum * dp


def scale_lr(base_lr, global_batch, base_batch, kappa=1.0):
    """公式 (12.8)：按全局批大小缩放学习率，kappa=1 线性，0.5 平方根。"""
    if base_lr <= 0 or base_batch <= 0 or global_batch <= 0:
        raise ValueError("学习率与批大小必须为正")
    if not 0.0 <= kappa <= 1.0:
        raise ValueError("kappa 必须在 [0, 1] 内")
    return base_lr * (global_batch / base_batch) ** kappa


# ---------------------------------------------------------------- 集合通信模拟
def all_reduce_sum(parts):
    """模拟 all-reduce(sum)：输入每个 rank 一份张量，输出每个 rank 的相同和。"""
    if not parts:
        raise ValueError("至少需要一个 rank 的张量")
    if len({tuple(p.shape) for p in parts}) != 1:
        raise ValueError("各 rank 的张量形状必须一致")
    total = torch.stack(parts).sum(dim=0)
    return [total.clone() for _ in parts]


def reduce_scatter_sum(tensors):
    """模拟 reduce-scatter(sum)：rank r 只保留总和中的第 r 段。"""
    if not tensors:
        raise ValueError("至少需要一个 rank 的张量")
    shape = tensors[0].shape
    if any(t.shape != shape for t in tensors):
        raise ValueError("各 rank 的张量形状必须一致")
    world = len(tensors)
    n = tensors[0].numel()
    pad = (-n) % world
    flats = [torch.cat([t.reshape(-1), t.new_zeros(pad)]) for t in tensors]
    total = torch.stack(flats).sum(dim=0)
    chunks = total.chunk(world)
    return [chunks[r].clone() for r in range(world)]


def all_gather_shards(shards):
    """模拟 all-gather：每个 rank 收集全部 shard 并拼接。"""
    if not shards:
        raise ValueError("至少需要一个 shard")
    full = torch.cat([s.reshape(-1) for s in shards])
    return [full.clone() for _ in shards]


def shard_flat(tensor, world, rank):
    """把张量按扁平化后等分为 world 份，返回第 rank 份。"""
    if world < 1 or not 0 <= rank < world:
        raise ValueError("world 与 rank 不合法")
    flat = tensor.reshape(-1)
    if flat.numel() % world != 0:
        raise ValueError("元素数必须能被 world 整除")
    return flat.chunk(world)[rank].clone()


def merge_shards(parts):
    """把参数分片拼回一个扁平张量。"""
    return torch.cat([p.detach().reshape(-1) for p in parts])


def shard_reduce_gather(tensors):
    """reduce-scatter 后 all-gather：在所有 rank 上重建求和结果。"""
    shards = reduce_scatter_sum(tensors)
    full = all_gather_shards(shards)[0]
    n = tensors[0].numel()
    return [full[:n].reshape(tensors[0].shape).clone() for _ in tensors]


# ---------------------------------------------------------------- 张量并行原语
def column_parallel_linear(x, weight, world):
    """列并行：输出维度切分，前向不需要通信。"""
    if weight.ndim != 2 or weight.shape[1] % world != 0:
        raise ValueError("weight 的输出维度必须能被 world 整除")
    return [x @ w for w in weight.chunk(world, dim=1)]


def row_parallel_linear(x_parts, weight):
    """行并行：输入维度切分，前向用一次 all-reduce 求和。"""
    world = len(x_parts)
    if weight.shape[0] % world != 0:
        raise ValueError("weight 的输入维度必须能被 world 整除")
    partial = [x @ w for x, w in zip(x_parts, weight.chunk(world, dim=0))]
    return all_reduce_sum(partial)[0]


class LossScaler:
    """公式 (12.12)：FP16 动态损失缩放。"""

    def __init__(self, init_scale=2.0 ** 16, backoff=0.5, growth=2.0,
                 growth_interval=2000):
        if init_scale <= 0 or not 0 < backoff < 1 or growth <= 1:
            raise ValueError("缩放参数不合法")
        if growth_interval < 1:
            raise ValueError("growth_interval 必须为正")
        self.scale = float(init_scale)
        self.backoff = float(backoff)
        self.growth = float(growth)
        self.growth_interval = int(growth_interval)
        self.good_steps = 0
        self.skipped_steps = 0

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale_(self, params):
        """把梯度除回真实尺度；返回梯度是否全部有限。"""
        inv = 1.0 / self.scale
        finite = True
        for p in params:
            if p.grad is None:
                continue
            p.grad.mul_(inv)
            finite = finite and bool(torch.isfinite(p.grad).all())
        return finite

    def update(self, finite):
        """溢出时减半并跳过；连续若干次正常后加倍。返回是否执行更新。"""
        if not finite:
            self.scale *= self.backoff
            self.good_steps = 0
            self.skipped_steps += 1
            return False
        self.good_steps += 1
        if self.good_steps >= self.growth_interval:
            self.scale *= self.growth
            self.good_steps = 0
        return True
