"""第 21 章：分页 KV Cache 与碎片账本（tinylm/inference/paged.py）。"""
import math
from dataclasses import dataclass, field

@dataclass
class BlockTable:
    """逻辑块 → 物理块的映射；物理块在池中按需分配、释放后立即可复用。"""

    block_size: int
    num_blocks: int
    free: list = field(default_factory=list)
    mapping: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.free:
            self.free = list(range(self.num_blocks))

    def blocks_needed(self, n_tokens):
        return math.ceil(n_tokens / self.block_size)

    def allocate(self, seq_id, n_tokens):
        """按 Token 数向上取整分配物理块；返回逻辑→物理块表。"""
        need = self.blocks_needed(n_tokens)
        if len(self.free) < need:
            raise MemoryError("物理块不足")
        table = [self.free.pop(0) for _ in range(need)]
        self.mapping[seq_id] = table
        return table

    def ensure_capacity(self, seq_id, total_tokens):
        """序列变长时补分配；已有块保留，只追加差量。"""
        table = self.mapping[seq_id]
        need = self.blocks_needed(total_tokens)
        extra = need - len(table)
        if extra <= 0:
            return 0
        if len(self.free) < extra:
            raise MemoryError("物理块不足")
        table.extend(self.free.pop(0) for _ in range(extra))
        return extra

    def free_sequence(self, seq_id):
        blocks = self.mapping.pop(seq_id, [])
        self.free.extend(blocks)
        return len(blocks)

    def used(self):
        return self.num_blocks - len(self.free)

def internal_fragmentation(lengths, block_size):
    """公式 (21.13)：块内未用槽位 / 已预留槽位。"""
    reserved = sum(math.ceil(n / block_size) * block_size for n in lengths)
    used = sum(lengths)
    return 0.0 if reserved == 0 else 1.0 - used / reserved

def padded_fragmentation(lengths, block_size):
    """朴素预分配：每条序列按 batch 内最长序列预留。"""
    longest = max(lengths)
    reserved = math.ceil(longest / block_size) * block_size * len(lengths)
    used = sum(lengths)
    return 1.0 - used / reserved

def fragmentation_table(lengths, block_size):
    total = sum(lengths)
    paged_slots = sum(math.ceil(n / block_size) * block_size for n in lengths)
    naive_slots = math.ceil(max(lengths) / block_size) * block_size * len(lengths)
    return {
        "tokens": total,
        "paged_slots": paged_slots,
        "naive_slots": naive_slots,
        "paged_util": total / paged_slots,
        "naive_util": total / naive_slots,
        "paged_waste": 1 - total / paged_slots,
        "naive_waste": 1 - total / naive_slots,
    }

def batch_capacity(free_bytes, cfg, seq_len, b_elem=2):
    """公式 (21.16)：给定剩余显存能同时驻留多少条等长序列。"""
    from tinylm.inference.kv_cache import cache_bytes

    per_seq = cache_bytes(cfg, 1, seq_len, b_elem)
    return int(free_bytes // per_seq) if per_seq else 0
