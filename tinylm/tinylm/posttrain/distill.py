"""第 19 章：知识蒸馏与模型压缩（tinylm/posttrain/distill.py）。"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class DistillConfig:
    """蒸馏训练脚本参数，不进入 ``ModelConfig``。"""

    temperature: float = 2.0       # 软标签温度 T
    alpha: float = 0.5             # 硬标签权重；0 表示纯蒸馏
    lr: float = 1e-3
    max_new: int = 4               # 教师生成的最大 Token 数
    max_grad_norm: float = 1.0
    pad_id: int = 0
    eos_id: int = 4

    def __post_init__(self):
        assert self.temperature > 0.0
        assert 0.0 <= self.alpha <= 1.0
        assert self.lr > 0.0
        assert self.max_new >= 1
        assert self.max_grad_norm > 0.0

def softmax_with_temperature(logits, temperature):
    """(19.1)：温度缩放后的教师/学生分布。"""
    if temperature <= 0.0:
        raise ValueError("temperature 必须为正")
    return torch.softmax(logits / temperature, dim=-1)

def distill_loss(student_logits, teacher_logits, labels=None,
                 cfg=DistillConfig()):
    """(19.3)–(19.4)：T^2 缩放的 KL 蒸馏，可选叠加硬标签交叉熵。

    ``student_logits`` 与 ``teacher_logits`` 形状为 ``(N, V)``，对应同一个
    预测位置；``labels`` 形状为 ``(N,)``。返回损失与诊断量。
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("学生与教师 logits 形状必须一致")
    t = cfg.temperature
    student_log_prob = F.log_softmax(student_logits / t, dim=-1)
    teacher_prob = softmax_with_temperature(teacher_logits, t)
    kd = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
    kd = kd * (t * t)                                  # (19.3)
    loss = (1.0 - cfg.alpha) * kd
    ce = torch.zeros((), device=student_logits.device)
    if labels is not None and cfg.alpha > 0.0:
        ce = F.cross_entropy(student_logits, labels, ignore_index=-100)
        loss = loss + cfg.alpha * ce                   # (19.4)
    diagnostics = {
        "kd": float(kd.detach()),
        "ce": float(ce.detach()),
        "teacher_entropy": float(
            -(teacher_prob * teacher_prob.clamp_min(1e-12).log()).sum(-1).mean()
        ),
    }
    return loss, diagnostics

def sft_labels(seqs, prompt_len, ignore_index=-100):
    """(19.7)：把完整序列转成下一 Token 标签，只监督提示之后的位置。

    标签在位置 ``t`` 上等于 ``seqs[t+1]``；提示最后一个位置的标签是回答的
    第一个 Token，因此监督从 ``prompt_len - 1`` 开始，而不是 ``prompt_len``。
    """
    labels = torch.full_like(seqs, ignore_index)
    labels[:, prompt_len - 1:-1] = seqs[:, prompt_len:]
    return labels

@torch.no_grad()
def generate_greedy(model, prompts, cfg=DistillConfig()):
    """(19.7)：教师贪心生成，返回 ``(B, T)`` 序列与布尔注意力掩码。"""
    model.eval()
    seqs = prompts
    done = torch.zeros(seqs.size(0), dtype=torch.bool,
                       device=seqs.device)
    for _ in range(cfg.max_new):
        logits = model(seqs).logits[:, -1]
        nxt = logits.argmax(dim=-1)
        nxt = torch.where(done, torch.full_like(nxt, cfg.pad_id), nxt)
        seqs = torch.cat([seqs, nxt[:, None]], dim=1)
        done = done | (nxt == cfg.eos_id)
        if bool(done.all()):
            break
    return seqs, seqs != cfg.pad_id

def teacher_targets(teacher, prompts):
    """教师提示末位的 logits，作为 (19.3) 的软目标。"""
    with torch.no_grad():
        teacher.eval()
        return teacher(prompts).logits[:, -1]

def magnitude_prune_(model, fraction, skip=("tok_emb", "lm_head")):
    """(19.9)：全局幅度剪枝，原地把最小的 ``fraction`` 比例权重置零。

    只处理二维以上参数，并跳过嵌入与输出头；返回实际阈值与可剪参数总数。
    非结构化剪枝不改变参数张量的形状，因此不会自动降低密集团计算量。
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction 必须在 [0, 1) 内")
    params = [(n, p) for n, p in model.named_parameters()
              if p.dim() >= 2 and not any(s in n for s in skip)]
    if not params:
        return 0.0, 0, []
    values = torch.cat([p.detach().abs().reshape(-1) for _, p in params])
    threshold = torch.quantile(values, fraction)
    masks = []
    with torch.no_grad():
        for _, p in params:
            mask = (p.abs() > threshold).to(p.dtype)
            p.mul_(mask)
            masks.append((p, mask))
    return float(threshold), int(values.numel()), masks

def apply_masks_(masks):
    """恢复阶段重新施加剪枝掩码，防止被置零的权重重新长出来。"""
    with torch.no_grad():
        for p, mask in masks:
            p.mul_(mask)

def global_sparsity(model, skip=("tok_emb", "lm_head")):
    """(19.9)：可剪参数中零元素的比例。"""
    params = [p for n, p in model.named_parameters()
              if p.dim() >= 2 and not any(s in n for s in skip)]
    if not params:
        return 0.0
    zeros = sum(int((p == 0).sum()) for p in params)
    total = sum(int(p.numel()) for p in params)
    return zeros / total

def parameter_count(model):
    """(19.11)：模型参数量。"""
    return sum(p.numel() for p in model.parameters())

def compression_ratio(teacher, student):
    """(19.11)：教师参数量除以学生参数量。"""
    return parameter_count(teacher) / parameter_count(student)
