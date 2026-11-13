"""打包后的 Token 数据集（tinylm/data/dataset.py）。"""
from collections.abc import Iterable, Sequence

import torch
from torch.utils.data import Dataset


def flatten_tokens(token_sequences: Iterable[Sequence[int]]) -> torch.Tensor:
    """把文档序列拼成一条长 Token 流。"""
    parts = [torch.as_tensor(seq, dtype=torch.long).reshape(-1)
             for seq in token_sequences]
    if not parts:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(parts)


class TokenDataset(Dataset):
    """把长 Token 流切成定长块，返回 ``(input_ids, labels)``。

    ``labels`` 是输入右移一位的目标；最后一个块不足 ``block_size + 1``
    时被丢弃，保证每条样本长度一致。
    """

    def __init__(self, tokens, block_size: int, stride: int | None = None):
        if block_size < 1:
            raise ValueError("block_size 必须为正")
        self.tokens = torch.as_tensor(tokens, dtype=torch.long).reshape(-1)
        self.block_size = int(block_size)
        self.stride = int(stride or block_size)
        if self.stride < 1:
            raise ValueError("stride 必须为正")
        usable = len(self.tokens) - self.block_size
        self.n_blocks = 0 if usable <= 0 else 1 + (usable - 1) // self.stride

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, index: int):
        if not 0 <= index < self.n_blocks:
            raise IndexError(index)
        start = index * self.stride
        chunk = self.tokens[start:start + self.block_size + 1]
        return chunk[:-1].clone(), chunk[1:].clone()


def packed_batches(tokens, block_size: int, batch_size: int, *, shuffle=True,
                   seed: int = 0):
    """按批产出 ``(input_ids, labels)``，形状均为 ``(B, block_size)``。"""
    ds = TokenDataset(tokens, block_size)
    order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed)) \
        if shuffle else torch.arange(len(ds))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        if len(idx) < batch_size:
            break
        xs, ys = zip(*(ds[int(i)] for i in idx))
        yield torch.stack(xs), torch.stack(ys)
