"""最小训练循环（tinylm/train/loop.py，第 9 章）。"""
import itertools
from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from .optim import AdamW, clip_grad_norm_, make_param_groups
from .schedule import ScheduleConfig, lr_at


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 3e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup: int = 20
    schedule: str = "cosine"
    min_ratio: float = 0.1
    grad_clip: float = 1.0
    max_steps: int = 100
    log_every: int = 0
    device: str = "cpu"

    def __post_init__(self):
        if self.max_steps < 1:
            raise ValueError("max_steps 必须为正")
        if self.grad_clip < 0:
            raise ValueError("grad_clip 必须非负")


def make_optimizer(model, cfg: TrainConfig) -> AdamW:
    return AdamW(
        make_param_groups(model, cfg.weight_decay),
        lr=cfg.lr, betas=cfg.betas, eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )


def _next_batch(batches, step):
    if callable(batches):
        return batches(step)
    if isinstance(batches, Iterable):
        return next(batches)
    raise TypeError("batches 必须是可迭代对象或返回批次的函数")


def _unpack(batch):
    if isinstance(batch, dict):
        return batch["input_ids"], batch.get("labels")
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch[0], batch[1]
    return batch, None


@torch.no_grad()
def evaluate(model, batches, device="cpu") -> float:
    """在给定批次上计算平均损失。"""
    model.eval()
    total, count = 0.0, 0
    for batch in batches:
        input_ids, labels = _unpack(batch)
        out = model(input_ids.to(device), labels=None if labels is None else labels.to(device))
        if out.loss is not None:
            total += float(out.loss)
            count += 1
    model.train()
    if count == 0:
        raise ValueError("没有任何批次可用于评测")
    return total / count


def train(model, batches, cfg: TrainConfig | None = None,
          on_step: Callable[[int, dict], None] | None = None) -> dict:
    """运行 ``cfg.max_steps`` 步训练，返回损失、学习率与辅助损失历史。

    ``batches`` 可以是可迭代批次，也可以是 ``step -> batch`` 的函数；
    有限批次会被循环复用。批次为 ``(input_ids, labels)`` 或同名字典。
    """
    cfg = cfg or TrainConfig()
    model.to(cfg.device)
    model.train()
    opt = make_optimizer(model, cfg)
    schedule = ScheduleConfig(
        kind=cfg.schedule, total=cfg.max_steps, warmup=cfg.warmup,
        min_ratio=cfg.min_ratio,
    )
    stream = batches if callable(batches) else itertools.cycle(batches)

    history = {"losses": [], "lr": [], "aux": []}
    for step in range(cfg.max_steps):
        input_ids, labels = _unpack(_next_batch(stream, step))
        out = model(input_ids.to(cfg.device),
                    labels=None if labels is None else labels.to(cfg.device))
        if out.loss is None:
            raise ValueError("批次没有标签，无法训练")
        loss = out.loss
        if out.aux_loss is not None:
            loss = loss + out.aux_loss

        lr = cfg.lr * lr_at(step, schedule)
        for group in opt.param_groups:
            group["lr"] = lr

        opt.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        history["losses"].append(float(out.loss.detach()))
        history["lr"].append(lr)
        history["aux"].append(None if out.aux_loss is None
                              else float(out.aux_loss.detach()))
        if on_step is not None:
            on_step(step, history)
        if cfg.log_every and (step + 1) % cfg.log_every == 0:
            print(f"step {step + 1:>5d}  loss {float(out.loss.detach()):.4f}  lr {lr:.2e}")
    return history
