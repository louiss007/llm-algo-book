"""注意力等价性：MHA/GQA 退化、MLA 手算、块对角与 padding 掩码。"""
import math

import torch

from tinylm.attention import LatentAttention, VariantAttention
from tinylm.config import ModelConfig


def cfg(**kw):
    base = dict(vocab_size=32, d_model=32, n_layers=1, n_heads=4, d_ff=64,
                max_seq_len=16, bias=False, pos_encoding="nope")
    base.update(kw)
    return ModelConfig(**base)


def test_gqa_with_all_kv_heads_equals_mha():
    torch.manual_seed(0)
    mha = VariantAttention(cfg(attention="mha")).eval()
    gqa = VariantAttention(cfg(attention="gqa", n_kv_heads=4)).eval()
    gqa.load_state_dict(mha.state_dict())
    x = torch.randn(2, 6, 32)
    assert torch.allclose(mha(x), gqa(x), atol=1e-6)


def test_mqa_has_one_kv_head():
    attn = VariantAttention(cfg(attention="mqa"))
    assert attn.n_kv_heads == 1 and attn.repeat == 4
    x = torch.randn(1, 5, 32)
    assert attn(x).shape == (1, 5, 32)


def test_mla_matches_explicit_computation():
    torch.manual_seed(0)
    cfg_mla = cfg(attention="mla", d_c=8, d_head_rope=4)
    attn = LatentAttention(cfg_mla).eval()
    x = torch.randn(2, 7, 32)
    with torch.no_grad():
        q = attn.q_proj(x).view(2, 7, attn.n_heads, attn.d_head).transpose(1, 2)
        c = attn.kv_down(x)
        k = attn.k_up(c).view(2, 7, attn.n_heads, attn.d_head).transpose(1, 2)
        v = attn.v_up(c).view(2, 7, attn.n_heads, attn.d_head).transpose(1, 2)
        q_r = attn.q_rope(x).view(2, 7, attn.n_heads, attn.d_rope).transpose(1, 2)
        k_r = attn.k_rope(x).view(2, 7, 1, attn.d_rope).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(attn.d_head)
        scores = scores + q_r @ k_r.transpose(-2, -1) / math.sqrt(attn.d_rope)
        mask = torch.triu(torch.full((7, 7), float("-inf")), diagonal=1)
        w = (scores + mask).softmax(dim=-1)
        manual = (w @ v).transpose(1, 2).reshape(2, 7, -1)
        manual = attn.out_proj(manual)
        assert torch.allclose(attn(x), manual, atol=1e-6)


def test_block_diagonal_mask_matches_separate_documents():
    torch.manual_seed(0)
    attn = VariantAttention(cfg()).eval()
    x1 = torch.randn(1, 3, 32)
    x2 = torch.randn(1, 2, 32)
    x = torch.cat([x1, x2], dim=1)
    mask = torch.ones(1, 1, 5, 5)
    mask[0, 0, :3, 3:] = 0
    mask[0, 0, 3:, :3] = 0
    with torch.no_grad():
        packed = attn(x, attention_mask=mask)
        separate = torch.cat([attn(x1), attn(x2)], dim=1)
    assert torch.allclose(packed, separate, atol=1e-6)


def test_padding_mask_does_not_change_valid_positions():
    torch.manual_seed(0)
    attn = VariantAttention(cfg()).eval()
    x = torch.randn(1, 4, 32)
    padded = torch.cat([x, torch.randn(1, 3, 32)], dim=1)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]])
    with torch.no_grad():
        out_pad = attn(padded, attention_mask=mask)
        out = attn(x)
    assert torch.allclose(out_pad[:, :4], out, atol=1e-6)
