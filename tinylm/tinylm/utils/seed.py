"""可复现性（tinylm/utils/seed.py）。"""
import os
import random

import torch


def set_seed(seed: int) -> int:
    """固定 Python、NumPy（可选）与 PyTorch 的随机源。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def worker_init_fn(worker_id: int) -> None:
    """DataLoader 的 worker 初始化，保证每个 worker 的随机源可复现。"""
    seed = torch.initial_seed() % 2**32
    set_seed(seed + worker_id)
