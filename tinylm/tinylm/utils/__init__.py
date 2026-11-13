"""通用工具（tinylm/utils）。"""
from .seed import set_seed, worker_init_fn
from .shapes import assert_shape, same_shape

__all__ = ["set_seed", "worker_init_fn", "assert_shape", "same_shape"]
