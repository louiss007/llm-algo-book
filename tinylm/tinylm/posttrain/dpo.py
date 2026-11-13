"""第 17 章：直接偏好优化（tinylm/posttrain/dpo.py）。"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class DPOConfig:
    """DPO 训练脚本参数，不进入 ``ModelConfig``。"""

    beta: float = 0.1               # 隐式奖励的温度，也控制偏离参考模型的力度
    lr: float = 5e-5
    epochs: int = 1
    batch_size: int = 8
    loss_type: str = "dpo"          # dpo | ipo
    label_smoothing: float = 0.0    # cDPO 的平滑系数
    length_normalize: bool = False  # 按响应 Token 数平均对数概率
    max_grad_norm: float = 1.0

    def __post_init__(self):
        assert self.beta > 0.0
        assert self.lr > 0.0
        assert self.epochs >= 1
        assert self.batch_size >= 1
        assert self.loss_type in ("dpo", "ipo")
        assert 0.0 <= self.label_smoothing < 0.5
        assert self.max_grad_norm > 0.0

def completion_logprobs(model, input_ids, attention_mask, completion_mask,
                        length_normalize=False):
    """返回每条样本完成部分的对数概率，形状 ``(B,)``。

    ``completion_mask[:, t]`` 标记位置 ``t`` 的 Token 属于回答（含 EOS）。
    第 ``t`` 位的 logits 预测 Token ``t+1``，因此参与求和的掩码要左移一位。
    """
    logits = model(input_ids, attention_mask=attention_mask).logits[:, :-1]
    logp = F.log_softmax(logits, dim=-1)
    target = input_ids[:, 1:]
    token_logp = logp.gather(-1, target[..., None]).squeeze(-1)
    valid = completion_mask[:, 1:] & attention_mask[:, 1:]
    token_logp = torch.where(valid, token_logp, torch.zeros_like(token_logp))
    total = token_logp.sum(-1)
    if length_normalize:
        # (17.6) 按响应 Token 数平均
        total = total / valid.sum(-1).clamp_min(1).to(total.dtype)
    return total

def log_ratio_margin(policy_chosen, policy_rejected, ref_chosen, ref_rejected):
    """(17.3) 括号内不含 β 的项：赢家与输家的对数比之差。"""
    win = policy_chosen - ref_chosen
    lose = policy_rejected - ref_rejected
    return win - lose

def implicit_reward(logp, ref_logp, beta):
    """(17.2)：隐式奖励，策略与参考模型的对数比乘以 β。"""
    return beta * (logp - ref_logp)

def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected,
             beta=0.1, label_smoothing=0.0, loss_type="dpo"):
    """(17.4)：直接偏好优化的成对损失，返回批平均标量。"""
    ratio_margin = log_ratio_margin(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    margin = beta * ratio_margin
    if loss_type == "ipo":
        # (17.7) IPO 把 log 比拉向 1/(2β)，而不是无限拉大
        return ((ratio_margin - 0.5 / beta) ** 2).mean()
    if label_smoothing > 0.0:
        eps = label_smoothing
        return -(F.logsigmoid(margin) * (1.0 - eps)
                 + F.logsigmoid(-margin) * eps).mean()
    return -F.logsigmoid(margin).mean()

def preference_accuracy(chosen_margin):
    """隐式奖励把赢家排在输家之前的比例；平局按失败计。"""
    return (chosen_margin > 0).float().mean()
