"""第 13 章的账本、打包与检索评测测试。"""
import pytest
import torch
import torch.nn.functional as F

from ch08 import ModelConfig, TinyLM
from tinylm.data.collate import block_causal_mask, pack_documents, token_utilization
from tinylm.eval.needle import (bucket_accuracy, needle_positions, passkey_batch,
                                retrieval_correct)
from tinylm.train import long_context as LC


def tiny_model(vocab=17, max_len=32, seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=vocab, d_model=16, n_layers=1, n_heads=2,
                      d_ff=32, max_seq_len=max_len, pos_encoding="rope",
                      norm="rmsnorm", ffn="swiglu", bias=False)
    return TinyLM(cfg).eval()


def test_attention_and_linear_flops():
    d, f, L, T, B = 8, 16, 2, 4, 3
    assert LC.attention_flops(B, T, d, L) == 2 * B * L * d * T * (T + 1)
    assert LC.linear_flops(B, T, d, f, L) == 2 * B * T * L * (4 * d * d + 3 * d * f)
    # T 翻倍时，二次项按 2(2T+1)/(T+1) 增长，线性项正好翻倍
    assert LC.attention_flops(B, 2 * T, d, L) == pytest.approx(
        LC.attention_flops(B, T, d, L) * 2 * (2 * T + 1) / (T + 1))
    assert LC.linear_flops(B, 2 * T, d, f, L) == 2 * LC.linear_flops(B, T, d, f, L)


def test_attention_share_at_crossover():
    d, f, L = 32, 64, 2
    T = LC.crossover_length(d, f)
    assert T == 4 * d + 3 * f - 1
    share = LC.attention_share(1, T, d, f, L)
    assert share == pytest.approx(0.5, abs=1e-6)
    assert LC.attention_share(1, T // 2, d, f, L) < 0.5
    assert LC.attention_share(1, 2 * T, d, f, L) > 0.5


def test_kv_cache_and_activation():
    # 7B 风格：32 层、8 个 KV 头、d_head=128、128k 上下文
    kv = LC.kv_cache_bytes(1, 131072, 32, 8, 128)
    assert kv == 1.7179869184e10
    assert kv / 2 ** 30 == pytest.approx(16.0)
    assert LC.kv_cache_bytes(2, 131072, 32, 8, 128) == 2 * kv

    base = LC.activation_bytes(1, 4096, 64, 4, 4)
    assert base == 12 * 1 * 4096 * 4 * 64 * 2
    scores = LC.activation_bytes(1, 4096, 64, 4, 4, materialize_scores=True) - base
    assert scores == 1 * 4 * 4 * 4096 ** 2 * 2


def test_length_schedule_and_effective_context():
    sched = LC.length_schedule(32, 128, 3)
    assert [n for n, _ in sched] == [32, 64, 128]
    assert sum(share for _, share in sched) == pytest.approx(1.0)
    shares = LC.token_budget_shares([32, 64, 128])
    assert sum(shares) == pytest.approx(1.0)
    assert shares[0] > shares[1] > shares[2]
    assert LC.effective_context([32, 64, 128], [1.0, 0.6, 0.4], 0.5) == 64
    assert LC.effective_context([32, 64], [0.1, 0.2], 0.5) == 0


def test_block_causal_mask():
    doc_ids = torch.tensor([0, 0, 1, 1])
    mask = block_causal_mask(doc_ids)
    assert mask.shape == (1, 1, 4, 4)
    assert mask.sum().item() == 6          # 1 + 2 + 1 + 2
    assert not mask[0, 0, 0, 1]
    assert not mask[0, 0, 2, 1]
    assert mask[0, 0, 3, 2]


def test_pack_documents():
    docs = [torch.arange(5), torch.arange(10, 17), torch.arange(20, 24)]
    batch = pack_documents(docs)
    assert batch.input_ids.numel() == 5 + 7 + 4 - 1
    assert batch.labels.numel() == batch.input_ids.numel()
    assert batch.doc_ids.tolist() == [0] * 5 + [1] * 7 + [2] * 3
    # 每个文档的最后一个输入位置不能预测下一篇文档的首 Token
    assert batch.labels[4].item() == -100
    assert batch.labels[11].item() == -100
    assert batch.position_ids.tolist() == ([0, 1, 2, 3, 4] + list(range(7))
                                           + [0, 1, 2])
    with pytest.raises(ValueError):
        pack_documents(docs, max_len=10)


def test_packed_equivalence_and_leakage():
    docs = [torch.tensor([1, 4, 2, 7, 3]), torch.tensor([5, 0, 6, 8, 2, 9, 1]),
            torch.tensor([3, 3, 7, 0])]
    model = tiny_model()
    batch = pack_documents(docs)

    def logits_loss(ids, labels, mask=None, positions=None):
        out = model(ids[None], attention_mask=mask, position_ids=positions)
        per = F.cross_entropy(out.logits[0], labels, reduction="none",
                              ignore_index=-100)
        return out.logits[0], per

    lg_u, ls_u = [], []
    for doc in docs:
        lg, ls = logits_loss(doc[:-1], doc[1:],
                             positions=torch.arange(len(doc) - 1))
        lg_u.append(lg)
        ls_u.append(ls)
    lg_u, ls_u = torch.cat(lg_u), torch.cat(ls_u)
    lg_p, ls_p = logits_loss(batch.input_ids, batch.labels,
                             batch.attention_mask, batch.position_ids)
    lg_c, _ = logits_loss(batch.input_ids, batch.labels,
                          positions=batch.position_ids)
    keep = batch.labels != -100
    assert torch.allclose(lg_u, lg_p[keep], atol=1e-5)
    assert torch.allclose(ls_u, ls_p[keep], atol=1e-5)
    assert not torch.allclose(lg_u, lg_c[keep], atol=1e-4)


def test_2d_padding_mask_unchanged():
    model = tiny_model()
    x = torch.randint(0, 17, (2, 9))
    with torch.no_grad():
        plain = model(x).logits
        masked = model(x, attention_mask=torch.ones(2, 9, dtype=torch.long)).logits
    assert torch.allclose(plain, masked, atol=1e-6)


def test_token_utilization():
    util = token_utilization([9, 12, 7, 15, 11, 8], 32)
    assert util["pad"] == pytest.approx(62 / 192)
    assert util["packed"] == pytest.approx(62 / 64)


def test_needle_positions():
    pos = needle_positions(128, 5)
    assert pos[0] == 0 and pos[-1] == 125
    assert pos == sorted(set(pos))
    assert needle_positions(9, 1) == [3]
    with pytest.raises(ValueError):
        needle_positions(2, 4)
    with pytest.raises(ValueError):
        needle_positions(128, 0)


def test_passkey_batch_and_buckets():
    gen = torch.Generator().manual_seed(0)
    positions = [0, 3, 8, 12]
    ids, targets = passkey_batch(4, 16, positions, 16, 17, 16, gen)
    assert ids.shape == (4, 16)
    for b, pos in enumerate(positions):
        assert ids[b, pos] == 16
        assert targets[b] == ids[b, pos + 1]
        assert ids[b, -1] == 17
    with pytest.raises(ValueError):
        passkey_batch(1, 16, [14], 16, 17, 16, gen)

    correct = torch.tensor([True, True, True, False])
    rows = bucket_accuracy(positions, correct, [0, 4, 16])
    assert rows[0][2] == pytest.approx(1.0)     # 位置 0、3 都正确
    assert rows[0][3] == 2
    assert rows[1][2] == pytest.approx(0.5)     # 位置 8 正确、12 错误

    last = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert retrieval_correct(last, torch.tensor([1, 0])).tolist() == [True, True]
