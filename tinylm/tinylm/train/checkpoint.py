"""检查点保存与加载（tinylm/train/checkpoint.py）。"""
from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer=None, step: int = 0, **extra) -> Path:
    """保存模型、优化器状态与训练步数；返回实际写入的路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "step": int(step),
    }
    payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(path, model=None, optimizer=None, *, map_location="cpu"):
    """加载检查点；给出 ``model``/``optimizer`` 时原地恢复权重。"""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(payload["model"])
    if optimizer is not None:
        if payload.get("optimizer") is None:
            raise ValueError("检查点不含优化器状态")
        optimizer.load_state_dict(payload["optimizer"])
    return payload
