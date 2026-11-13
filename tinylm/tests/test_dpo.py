"""第 17 章测试：DPO 损失、序列对数概率与隐式奖励。"""
import math

import pytest
import torch

from ch08 import ModelConfig, TinyLM
from tinylm.posttrain import dpo as D

def tiny_cfg():
    return ModelConfig(vocab_size=16, d_model=32, n_layers=2, n_heads=2,
                       d_ff=128, max_seq_len=32, pos_encoding="rope",
                       norm="rmsnorm", ffn="swiglu", bias=False)

def model(seed=0):
    torch.manual_seed(seed)
    return TinyLM(tiny_cfg())

def test_dpo_loss_zero_margin_is_log_two():
    z = torch.zeros(4)
    assert torch.allclose(D.dpo_loss(z, z, z, z, beta=0.1),
                          torch.tensor(math.log(2.0)), atol=1e-6)

def test_dpo_loss_decreases_with_margin():
    margins = torch.tensor([0.0, 0.5, 1.0, 2.0])
    zeros = torch.zeros(4)
    losses = [float(D.dpo_loss(margins[:i + 1], zeros[:i + 1], zeros[:i + 1],
                               zeros[:i + 1], beta=1.0))
              for i in range(4)]
    assert losses[0] > losses[1] > losses[2] > losses[3]

def test_dpo_loss_ignores_common_shift():
    pc, pr = torch.tensor([0.3, -1.0]), torch.tensor([0.1, 0.5])
    rc, rr = torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])
    a = D.dpo_loss(pc, pr, rc, rr)
    b = D.dpo_loss(pc + 3.0, pr + 3.0, rc, rr)
    assert torch.allclose(a, b)

def test_label_smoothing_increases_loss_for_confident_pair():
    pc, pr, rc, rr = (torch.tensor([2.0]), torch.tensor([0.0]),
                      torch.tensor([0.0]), torch.tensor([0.0]))
    hard = D.dpo_loss(pc, pr, rc, rr, beta=1.0)
    smooth = D.dpo_loss(pc, pr, rc, rr, beta=1.0, label_smoothing=0.1)
    assert smooth > hard

def test_ipo_has_zero_at_target_margin():
    beta = 0.5
    h = 1.0 / (2.0 * beta)
    pc = torch.tensor([h])
    zeros = torch.zeros(1)
    assert torch.allclose(D.dpo_loss(pc, zeros, zeros, zeros, beta=beta,
                                     loss_type="ipo"), torch.tensor(0.0))

def test_implicit_reward_scale():
    logp = torch.tensor([0.2, -0.4])
    ref = torch.tensor([0.1, 0.0])
    reward = D.implicit_reward(logp, ref, beta=0.5)
    assert torch.allclose(reward, torch.tensor([0.05, -0.2]))

def test_dpo_gradient_pushes_chosen_up_and_rejected_down():
    pc = torch.tensor([0.5], requires_grad=True)
    pr = torch.tensor([0.0], requires_grad=True)
    zero = torch.zeros(1)
    loss = D.dpo_loss(pc, pr, zero, zero, beta=0.1)
    loss.backward()
    assert pc.grad < 0 and pr.grad > 0
    assert torch.allclose(pc.grad, -pr.grad)

def test_completion_logprobs_matches_manual():
    m = model()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    attn = torch.ones(1, 5, dtype=torch.bool)
    completion = torch.tensor([[False, False, True, True, True]])
    got = D.completion_logprobs(m, ids, attn, completion)
    with torch.no_grad():
        logits = m(ids, attention_mask=attn).logits[:, :-1]
        logp = torch.log_softmax(logits, dim=-1)
        manual = sum(logp[0, t, ids[0, t + 1]] for t in (1, 2, 3))
    assert torch.allclose(got[0], manual)

def test_completion_logprobs_excludes_prompt_and_padding():
    m = model()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    attn = torch.ones(1, 5, dtype=torch.bool)
    completion = torch.tensor([[False, False, True, True, True]])
    base = D.completion_logprobs(m, ids, attn, completion)
    padded_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0]])
    padded_attn = torch.tensor([[True] * 5 + [False] * 2])
    padded_completion = torch.tensor([[False, False, True, True, True, False, False]])
    padded = D.completion_logprobs(m, padded_ids, padded_attn, padded_completion)
    assert torch.allclose(base, padded)

def test_length_normalize_divides_by_completion_length():
    m = model()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    attn = torch.ones(1, 5, dtype=torch.bool)
    completion = torch.tensor([[False, False, True, True, True]])
    total = D.completion_logprobs(m, ids, attn, completion)
    mean = D.completion_logprobs(m, ids, attn, completion, length_normalize=True)
    assert torch.allclose(mean, total / 3.0)

def test_preference_accuracy_ties_fail():
    margin = torch.tensor([1.0, -1.0, 0.0, 2.0])
    assert float(D.preference_accuracy(margin)) == 0.5

def test_dpo_config_validation():
    with pytest.raises(AssertionError):
        D.DPOConfig(beta=0.0)
    with pytest.raises(AssertionError):
        D.DPOConfig(loss_type="orpo")
    with pytest.raises(AssertionError):
        D.DPOConfig(label_smoothing=0.5)
    with pytest.raises(AssertionError):
        D.DPOConfig(epochs=0)
