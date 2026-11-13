"""自定义算子的 float64 梯度检查。"""
import torch

from tinylm.norm import LayerNorm, RMSNorm
from tinylm.ssm import scan_states


def test_layernorm_gradcheck():
    torch.manual_seed(0)
    layer = LayerNorm(8, eps=1e-5).double()
    x = torch.randn(2, 3, 8, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-4)


def test_rmsnorm_gradcheck():
    torch.manual_seed(0)
    layer = RMSNorm(8, eps=1e-5).double()
    x = torch.randn(2, 3, 8, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-4)


def test_scan_states_gradcheck():
    torch.manual_seed(0)
    a = torch.sigmoid(torch.randn(1, 4, 2, 3, dtype=torch.double))
    a.requires_grad_(True)
    b = torch.randn(1, 4, 2, 3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, y: scan_states(x, y), (a, b),
                                    eps=1e-6, atol=1e-4)
