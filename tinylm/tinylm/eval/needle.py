"""第 13 章：大海捞针式检索评测（tinylm/eval/needle.py）。"""
import torch


def needle_positions(context_len, n_probes):
    """(13.11)：在可放置 KEY 的深度上等距取样。

    末两位留给 value 与 QUERY，因此 KEY 的位置范围是 ``[0, T-3]``。
    """
    if context_len < 3:
        raise ValueError("context_len 至少为 3")
    if n_probes < 1:
        raise ValueError("n_probes 必须为正")
    last = context_len - 3
    if n_probes == 1:
        return [last // 2]
    return [round(i * last / (n_probes - 1)) for i in range(n_probes)]


def passkey_batch(batch, context_len, positions, key_id, query_id,
                  value_vocab, generator=None):
    """构造单针检索批；返回 ``(input_ids, targets)``。

    ``KEY`` 后一位是随机 value，末位是 ``QUERY``；模型在末位预测 value。
    """
    if context_len < 3:
        raise ValueError("context_len 至少为 3")
    ids = torch.randint(0, value_vocab, (batch, context_len),
                        generator=generator)
    targets = torch.empty(batch, dtype=torch.long)
    for b, pos in enumerate(positions):
        if not 0 <= pos <= context_len - 3:
            raise ValueError("KEY 的位置越界")
        ids[b, pos] = key_id
        targets[b] = ids[b, pos + 1]
    ids[:, -1] = query_id
    return ids, targets


def retrieval_correct(logits, targets):
    """末位 logits 的 argmax 是否等于目标 value。"""
    return logits.argmax(dim=-1) == targets


def bucket_accuracy(positions, correct, edges):
    """按深度分桶统计准确率，返回 [(下界, 上界, 准确率, 样本数), ...]。"""
    positions = torch.as_tensor(positions)
    correct = torch.as_tensor(correct, dtype=torch.float32)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (positions >= lo) & (positions < hi)
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n else float("nan")
        out.append((lo, hi, acc, n))
    return out
