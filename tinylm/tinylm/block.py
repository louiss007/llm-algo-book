"""Transformer 块（tinylm/block.py，第 5、7 章增量）。"""
import torch.nn as nn

from .attention import make_attention
from .moe import make_ffn_or_moe
from .norm import make_norm


class TransformerBlock(nn.Module):
    """Pre-LN 残差块；第 5 章加入并行块，第 7 章前馈可换成 MoE。"""

    def __init__(self, cfg):
        super().__init__()
        self.parallel = cfg.parallel_block
        self.norm1 = make_norm(cfg)
        self.attn = make_attention(cfg)
        self.norm2 = None if self.parallel else make_norm(cfg)
        self.mlp = make_ffn_or_moe(cfg)

    def forward(self, x, attention_mask=None, position_ids=None):
        if self.parallel:
            h = self.norm1(x)
            delta, aux = self._ffn(h)
            return x + self.attn(h, attention_mask, position_ids) + delta, aux
        x = x + self.attn(self.norm1(x), attention_mask, position_ids)
        delta, aux = self._ffn(self.norm2(x))
        return x + delta, aux

    def _ffn(self, h):
        out = self.mlp(h)
        return out if isinstance(out, tuple) else (out, None)
