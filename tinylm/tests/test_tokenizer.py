"""第 2 章：BPE 往返、特殊 Token、保存加载与对话模板。"""
import torch

from tinylm.tokenizer.bpe import CORPUS, BPETokenizer
from tinylm.tokenizer.chat import ChatTemplate, count_supervised, encode_conversation

SPECIALS = ["<bos>", "<eos>", "<pad>"]


def make_tokenizer():
    return BPETokenizer().train(CORPUS, vocab_size=320, special_tokens=SPECIALS)


def test_roundtrip_chinese_and_english():
    tok = make_tokenizer()
    for text in ["语言模型学习下一个词元。", "Tokenization is the first step."]:
        assert tok.decode(tok.encode(text)) == text


def test_arbitrary_utf8_roundtrip():
    tok = make_tokenizer()
    text = "emoji 🙂 与生僻字 𠮷，以及换行\n和制表符\t。"
    assert tok.decode(tok.encode(text)) == text


def test_special_tokens_are_not_merged():
    tok = make_tokenizer()
    ids = tok.encode("语言<eos>模型")
    assert tok.vocab["<eos>"] in ids
    assert tok.decode(ids, skip_special=False).find("<eos>") >= 0
    assert "<eos>" not in tok.decode(ids)


def test_save_load_roundtrip(tmp_path):
    tok = make_tokenizer()
    path = tmp_path / "tok.json"
    tok.save(path)
    restored = BPETokenizer.load(path)
    text = "保存后编码结果必须一致。"
    assert restored.vocab_size == tok.vocab_size
    assert restored.encode(text) == tok.encode(text)


def test_chat_template_masks_prompt():
    template = ChatTemplate()
    rendered = template.render([
        ("user", [10, 11]),
        ("assistant", [12, 13]),
    ])
    n_sup, n_tok, ratio = count_supervised(rendered)
    assert n_tok == rendered["input_ids"].numel()
    assert 0 < n_sup < n_tok
    labels = rendered["labels"]
    supervised = labels[:-1] != -100
    assert torch.equal(labels[:-1][supervised],
                       rendered["input_ids"][1:][supervised])
    assert labels[-1] == -100


def test_encode_conversation_matches_template():
    turns = [("user", [7, 8]), ("assistant", [9])]
    a = encode_conversation(turns)
    b = ChatTemplate().render(turns)
    assert torch.equal(a["input_ids"], b["input_ids"])
    assert torch.equal(a["labels"], b["labels"])
