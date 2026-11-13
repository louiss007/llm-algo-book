"""形状断言（tinylm/utils/shapes.py）。"""
import torch


def assert_shape(tensor: torch.Tensor, *dims) -> torch.Tensor:
    """断言张量维度；``None`` 表示该维不检查。

    例如 ``assert_shape(x, 1, None, 64)`` 要求三维且第 1、3 维匹配。
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"期望 torch.Tensor，得到 {type(tensor).__name__}")
    if tensor.dim() != len(dims):
        raise AssertionError(
            f"期望 {len(dims)} 维，得到 {tensor.dim()} 维（shape={tuple(tensor.shape)}）"
        )
    for i, (actual, expected) in enumerate(zip(tensor.shape, dims)):
        if expected is not None and actual != expected:
            raise AssertionError(
                f"第 {i} 维期望 {expected}，得到 {actual}（shape={tuple(tensor.shape)}）"
            )
    return tensor


def same_shape(*tensors: torch.Tensor) -> bool:
    if not tensors:
        return True
    return all(t.shape == tensors[0].shape for t in tensors[1:])
