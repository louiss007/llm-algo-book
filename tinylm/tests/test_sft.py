import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from ch08 import ModelConfig, TinyLM
from tinylm.posttrain import sft
from tinylm.tokenizer.chat import (ChatTemplate, count_supervised,
                                   encode_conversation)


def two_turn():
    return [("user", [5, 6]), ("assistant", [7, 8])]


def test_answer_only_masks_prompt():
    r = encode_conversation(two_turn())
    # ids: BOS USER 5 6 ASSIST 7 8 EOS
    assert r["input_ids"].tolist() == [1, 2, 5, 6, 3, 7, 8, 4]
    assert r["labels"].tolist() == [-100, -100, -100, -100, 7, 8, 4, -100]
    assert int((r["labels"] != -100).sum()) == 3


def test_full_loss_supervises_content():
    r = encode_conversation(two_turn(), answer_only=False)
    assert int((r["labels"] != -100).sum()) == 5


def test_multi_turn_each_answer_is_supervised():
    turns = [("user", [5]), ("assistant", [6]),
             ("user", [7]), ("assistant", [8])]
    r = encode_conversation(turns)
    ids = r["input_ids"].tolist()
    labels = r["labels"].tolist()
    # 两个回答 Token 与两个回合 EOS 都是监督目标
    assert int((r["labels"] != -100).sum()) == 4
    assert ids[3] == 3 and labels[3] == 6      # 第一个 ASSIST 之后
    assert ids[8] == 3 and labels[8] == 8      # 第二个 ASSIST 之后


def test_count_supervised_ratio():
    r = encode_conversation(two_turn())
    sup, tok, ratio = count_supervised(r)
    assert (sup, tok) == (3, 8)
    assert abs(ratio - 3 / 8) < 1e-9


def test_masked_cross_entropy_matches_manual():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 6)
    labels = torch.tensor([[1, -100, 3, -100], [2, 4, -100, -100]])
    got = sft.masked_cross_entropy(logits, labels)
    flat = F.cross_entropy(logits.reshape(-1, 6), labels.reshape(-1),
                           ignore_index=-100, reduction="none")
    want = flat[labels.reshape(-1) != -100].mean()
    assert torch.allclose(got, want)


def test_token_mean_differs_from_sequence_mean():
    per_token = torch.tensor([[1.0, 1.0, 9.0], [1.0, -100.0, -100.0]])
    labels = torch.tensor([[0, 1, 2], [0, -100, -100]])
    token_mean = sft.token_mean_loss(per_token, labels)
    seq_mean = sft.sequence_mean_loss(per_token, labels)
    assert abs(token_mean.item() - 3.0) < 1e-6
    assert abs(seq_mean.item() - 7 / 3) < 1e-6
    assert not torch.isclose(token_mean, seq_mean)


def test_collate_pads_labels_and_mask():
    a = encode_conversation([("user", [5]), ("assistant", [6])])
    b = encode_conversation([("user", [5, 6, 7]), ("assistant", [8, 9])])
    batch = sft.collate_examples([a, b], pad_token=0)
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"][0, -1].item() == -100
    assert not batch["attention_mask"][0, -1]
    assert batch["attention_mask"][1].all()


def test_pack_marks_boundaries_and_resets_positions():
    a = encode_conversation([("user", [5]), ("assistant", [6])])
    b = encode_conversation([("user", [7]), ("assistant", [8])])
    packed = sft.pack_sft_examples([a, b])
    n = a["input_ids"].numel()
    assert packed.labels[n - 1].item() == -100      # 跨对话边界
    assert packed.doc_ids.tolist() == [0] * n + [1] * n
    assert packed.position_ids.tolist() == list(range(n)) * 2
    assert packed.tokens == 2 * n
    assert packed.supervised == (int((a["labels"] != -100).sum())
                                 + int((b["labels"] != -100).sum()))


def test_pack_raises_when_too_long():
    a = encode_conversation([("user", [5]), ("assistant", [6])])
    with pytest.raises(ValueError):
        sft.pack_sft_examples([a, a], max_len=a["input_ids"].numel() + 1)


def test_packing_equals_per_example_forward():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=16, d_model=32, n_layers=2, n_heads=2,
                      d_ff=128, max_seq_len=32, pos_encoding="rope",
                      norm="rmsnorm", ffn="swiglu", bias=False)
    model = TinyLM(cfg).eval()
    examples = [
        encode_conversation([("user", [5, 6, 7]), ("assistant", [8, 9])]),
        encode_conversation([("user", [5, 9]), ("assistant", [6, 7, 8])]),
        encode_conversation([("user", [6]), ("assistant", [7])]),
    ]

    pieces, losses, total = [], 0.0, 0
    with torch.no_grad():
        for e in examples:
            out = model(e["input_ids"][None],
                        position_ids=torch.arange(e["input_ids"].numel())[None])
            per = F.cross_entropy(out.logits[0], e["labels"],
                                  ignore_index=-100, reduction="none")
            keep = e["labels"] != -100
            pieces.append(out.logits[0][keep])
            losses = losses + per[keep].sum()
            total += int(keep.sum())
        packed = sft.pack_sft_examples(examples)
        got = model(packed.input_ids[None],
                    attention_mask=packed.attention_mask,
                    position_ids=packed.position_ids[None])
        per_p = F.cross_entropy(got.logits[0], packed.labels,
                                ignore_index=-100, reduction="none")
        keep_p = packed.labels != -100
        logits_ref = torch.cat(pieces)
        assert torch.allclose(got.logits[0][keep_p], logits_ref, atol=1e-6)
        assert torch.allclose(per_p[keep_p].sum(), losses, atol=1e-6)


def test_supervision_stats():
    a = encode_conversation([("user", [5, 6]), ("assistant", [7, 8])])
    stats = sft.supervision_stats([a, a])
    assert stats["supervised"] == 2 * 3
    assert stats["tokens"] == 2 * a["input_ids"].numel()
    assert abs(stats["ratio"] - 3 / a["input_ids"].numel()) < 1e-9
    assert stats["mean_answer"] == 3.0


def test_uniform_label_noise_preserves_argmax_and_normalizes():
    p = torch.tensor([0.6, 0.3, 0.1])
    noisy = sft.label_noise_mixture(p, 0.4, 3)
    assert abs(noisy.sum().item() - 1.0) < 1e-9
    assert noisy.argmax().item() == p.argmax().item()
    assert abs(noisy[0].item() - 0.6 * 0.6 - 0.4 / 3) < 1e-6


def test_rejection_sampling_formulas():
    assert sft.acceptance_probability(0.25) == 0.25
    assert abs(sft.pass_at_n(0.25, 8) - (1 - 0.75 ** 8)) < 1e-9
    assert sft.expected_draws(0.25, 100) == 400.0
    assert sft.pass_at_n(0.0, 8) == 0.0
    with pytest.raises(ValueError):
        sft.acceptance_probability(1.2)


def test_rejection_sampling_empirical():
    torch.manual_seed(0)
    p, n, trials = 0.25, 8, 4000
    hits = 0
    for _ in range(trials):
        draws = torch.rand(n) < p
        hits += int(draws.any())
    assert abs(hits / trials - sft.pass_at_n(p, n)) < 0.03
