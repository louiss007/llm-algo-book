"""学习率调度（tinylm/train/schedule.py，第 11 章）。"""
from dataclasses import dataclass

from .optim import lr_factor


@dataclass(frozen=True)
class ScheduleConfig:
    """完整的调度参数；``kind`` 取 ``constant``、``cosine`` 或 ``wsd``。"""

    kind: str = "cosine"
    total: int = 1000
    warmup: int = 0
    min_ratio: float = 0.1
    decay_start: int | None = None
    power: float = 1.0

    def __post_init__(self):
        if self.total <= 0:
            raise ValueError("total 必须为正")
        if not 0 <= self.warmup <= self.total:
            raise ValueError("warmup 必须在 [0, total] 内")
        if self.kind not in ("constant", "cosine", "wsd"):
            raise ValueError(f"未知调度: {self.kind}")
        if not 0 <= self.min_ratio <= 1:
            raise ValueError("min_ratio 必须在 [0, 1] 内")


def lr_at(step: int, cfg: ScheduleConfig) -> float:
    """返回第 ``step`` 步（0 起）的学习率乘子。"""
    return lr_factor(step, cfg.total, cfg.warmup, cfg.kind,
                     cfg.min_ratio, cfg.decay_start, cfg.power)


def schedule_factors(cfg: ScheduleConfig) -> list[float]:
    """展开整条调度曲线，便于画图或对账。"""
    return [lr_at(step, cfg) for step in range(cfg.total + 1)]
