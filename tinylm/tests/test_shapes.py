"""形状边界：B=1、T=1、T=max_seq_len，以及各架构变体。"""
import pytest
import torch

from tinylm.config import ModelConfig, MoEConfig
from tinylm.data.dataset import TokenDataset, packed_batches
from tinylm.model import TinyLM

V, D, T_MAX = 32, 32, 16


def cfg(**kw):
    base = dict(vocab_size=V, d_model=D, n_layers=2, n_heads=4, d_ff=64,
                max_seq_len=T_MAX)
    base.update(kw)
    return ModelConfig(**base)


@pytest.mark.parametrize("batch,seq", [(1, 1), (2, 7), (1, T_MAX)])
def test_output_shapes(batch, seq):
    model = TinyLM(cfg())
    ids = torch.randint(0, V, (batch, seq))
    out = model(ids)
    assert out.logits.shape == (batch, seq, V)
    assert out.loss is None and out.aux_loss is None and out.past_kv is None


def test_loss_and_logits_to_keep():
    model = TinyLM(cfg())
    ids = torch.randint(0, V, (2, 8))
    labels = ids.clone()
    labels[:, :4] = -100
    out = model(ids, labels=labels, logits_to_keep=3)
    assert out.logits.shape == (2, 3, V)
    assert out.loss.dim() == 0 and torch.isfinite(out.loss)


@pytest.mark.parametrize("kw", [
    dict(norm="rmsnorm", ffn="swiglu", bias=False, pos_encoding="rope"),
    dict(attention="gqa", n_kv_heads=2, pos_encoding="rope"),
    dict(attention="mqa", pos_encoding="alibi"),
    dict(attention="mla", d_c=8, d_head_rope=4, pos_encoding="rope",
         bias=False),
    dict(block_type="linear_attention", bias=False),
    dict(block_type="ssm", ssm_d_state=8, ssm_expand=2, bias=False),
    dict(block_type="hybrid", hybrid_attn_every=2, ssm_d_state=8, bias=False),
])
def test_architecture_variants(kw):
    model = TinyLM(cfg(**kw))
    ids = torch.randint(0, V, (1, 5))
    out = model(ids)
    assert out.logits.shape == (1, 5, V)


def test_moe_aux_loss_in_training():
    model = TinyLM(cfg(moe=MoEConfig(n_experts=4, top_k=2)))
    model.train()
    ids = torch.randint(0, V, (2, 6))
    out = model(ids, labels=ids)
    assert out.aux_loss is not None and out.aux_loss.dim() == 0
    assert torch.isfinite(out.loss + out.aux_loss)


def test_dataset_boundaries():
    tokens = torch.arange(100)
    ds = TokenDataset(tokens, block_size=10)
    assert len(ds) == 9
    x, y = ds[0]
    assert x.shape == y.shape == (10,)
    assert torch.equal(x[1:], y[:-1])
    assert len(TokenDataset(torch.arange(10), block_size=10)) == 0
