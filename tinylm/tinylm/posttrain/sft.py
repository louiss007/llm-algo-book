"""第 14 章：监督微调的数据整理、损失与采样工具（tinylm/posttrain/sft.py）。"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tinylm.data.collate import block_causal_mask

IGNORE_INDEX = -100


@dataclass(frozen=True)
class SFTConfig:
    """SFT 的训练脚本参数，不进入 ``ModelConfig``。"""

    lr: float = 2e-5                 # 全参数微调的典型峰值学习率
    epochs: float = 3.0
    batch_size: int = 8
    max_len: int = 1024
    answer_only: bool = True
    replay_ratio: float = 0.0        # 混入预训练数据的比例
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0


def masked_cross_entropy(logits, labels):
    """(14.1)：只对 ``labels != -100`` 的位置求平均。

    ``logits`` 形状 ``(B, T, V)``，``labels`` 形状 ``(B, T)``。
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def token_mean_loss(per_token, labels):
    """(14.2) 左式：整个批次的监督 Token 一起平均。"""
    mask = labels != IGNORE_INDEX
    return per_token[mask].sum() / mask.sum().clamp_min(1)


def sequence_mean_loss(per_token, labels):
    """(14.2) 右式：先对每条样本平均，再对样本平均。"""
    mask = labels != IGNORE_INDEX
    lengths = mask.sum(1).clamp_min(1)
    per_example = (per_token * mask).sum(1) / lengths
    return per_example.mean()


def collate_examples(examples, pad_token=0, max_len=None):
    """把单条渲染结果补齐成一个批；padding 的标签为 ``-100``。"""
    examples = list(examples)
    if max_len is not None:
        examples = [e for e in examples if e["input_ids"].numel() <= max_len]
    length = max(int(e["input_ids"].numel()) for e in examples)
    batch = len(examples)
    input_ids = torch.full((batch, length), pad_token, dtype=torch.long)
    labels = torch.full((batch, length), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((batch, length), dtype=torch.bool)
    for i, e in enumerate(examples):
        n = int(e["input_ids"].numel())
        input_ids[i, :n] = e["input_ids"]
        labels[i, :n] = e["labels"]
        attention_mask[i, :n] = True
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": attention_mask}


@dataclass(frozen=True)
class PackedSFTBatch:
    """多条对话拼成的一条训练序列。"""

    input_ids: torch.Tensor        # (T,)
    labels: torch.Tensor           # (T,)
    doc_ids: torch.Tensor          # (T,)
    position_ids: torch.Tensor     # (T,)，每条对话内从 0 开始
    attention_mask: torch.Tensor   # (1, 1, T, T)，块对角因果掩码
    supervised: int                # 监督标签数
    tokens: int                    # 非 padding 输入数


def pack_sft_examples(examples, max_len=None):
    """把多条对话拼进一条序列，返回 :class:`PackedSFTBatch`。

    跨对话边界的标签置为 ``-100``；位置编号在每条对话内重置；
    注意力用块对角因果掩码，避免后一条对话看到前一条的内容。
    """
    examples = list(examples)
    lengths = [int(e["input_ids"].numel()) for e in examples]
    total = sum(lengths)
    if max_len is not None and total > max_len:
        raise ValueError("对话总长超过 max_len")

    input_ids = torch.cat([e["input_ids"] for e in examples])
    labels = torch.cat([e["labels"] for e in examples]).clone()
    doc_ids = torch.cat([torch.full((n,), i, dtype=torch.long)
                         for i, n in enumerate(lengths)])
    position_ids = torch.cat([torch.arange(n) for n in lengths])

    offsets = torch.tensor([0] + lengths[:-1]).cumsum(0)
    last = offsets + torch.tensor(lengths) - 1
    labels[last[:-1]] = IGNORE_INDEX          # (14.6) 跨对话边界不计损失

    return PackedSFTBatch(
        input_ids=input_ids,
        labels=labels,
        doc_ids=doc_ids,
        position_ids=position_ids,
        attention_mask=block_causal_mask(doc_ids),
        supervised=int((labels != IGNORE_INDEX).sum()),
        tokens=total,
    )


def supervision_stats(examples):
    """(14.3)：返回监督 Token 占比与平均回答长度。"""
    sup = sum(int((e["labels"] != IGNORE_INDEX).sum()) for e in examples)
    tok = sum(int(e["input_ids"].numel()) for e in examples)
    lengths = [int((e["labels"] != IGNORE_INDEX).sum()) for e in examples]
    return {
        "supervised": sup,
        "tokens": tok,
        "ratio": sup / tok if tok else 0.0,
        "mean_answer": sum(lengths) / len(lengths) if lengths else 0.0,
    }


def label_noise_mixture(p_true, epsilon, vocab_size):
    """(14.4)：均匀标签噪声下模型学到的分布。

    ``p_true`` 是干净条件分布（形状 ``(..., V)``），``epsilon`` 是标签
    被替换成均匀随机标签的比例。
    """
    uniform = torch.full_like(p_true, 1.0 / vocab_size)
    return (1.0 - epsilon) * p_true + epsilon * uniform


def acceptance_probability(p):
    """(14.7)：单条采样被接受的概率就是模型的精确匹配率 ``p``。"""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p 必须在 [0, 1] 内")
    return p


def expected_draws(p, accepted):
    """(14.7)：凑够 ``accepted`` 条需采样的期望条数。"""
    if p <= 0.0:
        return float("inf")
    return accepted / acceptance_probability(p)


def pass_at_n(p, n):
    """(14.7)：独立采样 ``n`` 条，至少一条正确的概率。"""
    if n < 1:
        raise ValueError("n 必须为正")
    return 1.0 - (1.0 - acceptance_probability(p)) ** n
