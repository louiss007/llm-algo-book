"""第 13 章：长文本打包与块对角注意力（tinylm/data/collate.py）。"""
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedBatch:
    """一条打包序列；所有张量长度都是 ``sum(len(doc)) - 1``。"""

    input_ids: torch.Tensor        # (T,)
    labels: torch.Tensor           # (T,)，跨文档位置为 -100
    doc_ids: torch.Tensor          # (T,)，每个输入位置所属文档
    position_ids: torch.Tensor     # (T,)，文档内位置，每篇从 0 开始
    attention_mask: torch.Tensor   # (1, 1, T, T)，块对角因果掩码


def block_causal_mask(doc_ids):
    """(13.8)：同文档且不越界的注意力掩码，形状 (1, 1, T, T)。"""
    T = doc_ids.numel()
    same_doc = doc_ids[:, None] == doc_ids[None, :]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    return (same_doc & causal)[None, None]


def pack_documents(docs, max_len=None):
    """把多篇文档拼成一条训练序列，返回 :class:`PackedBatch`。

    打包后第 t 个输入预测第 t+1 个 Token；若两者属于不同文档，
    ``labels[t] = -100``，配合 ``attention_mask`` 保证文档之间互不可见。
    """
    if max_len is not None and sum(len(d) for d in docs) > max_len:
        raise ValueError("文档总长超过 max_len")
    lengths = [len(d) for d in docs]
    flat = torch.cat([torch.as_tensor(d, dtype=torch.long) for d in docs])
    doc_ids = torch.cat([torch.full((n,), i, dtype=torch.long)
                         for i, n in enumerate(lengths)])
    position_ids = torch.cat([torch.arange(n) for n in lengths])

    input_ids = flat[:-1]
    labels = flat[1:].clone()
    labels[doc_ids[:-1] != doc_ids[1:]] = -100      # 跨文档边界不计损失
    return PackedBatch(
        input_ids=input_ids,
        labels=labels,
        doc_ids=doc_ids[:-1],
        position_ids=position_ids[:-1],
        attention_mask=block_causal_mask(doc_ids[:-1]),
    )


def token_utilization(lengths, max_len):
    """(13.7)：逐条 padding 与拼接打包的 Token 利用率。"""
    total = sum(lengths)
    padded_capacity = len(lengths) * max_len
    packed_capacity = math.ceil(total / max_len) * max_len
    return {"pad": total / padded_capacity, "packed": total / packed_capacity}
