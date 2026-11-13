"""第 2 章：字节级 BPE 分词器（tinylm/tokenizer/bpe.py 的核心实现）。"""
import json
import re
from collections import Counter


def bytes_to_unicode():
    """把 256 个字节映射到可打印的 Unicode 字符（GPT-2 的做法）。"""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}


class BPETokenizer:
    def __init__(self, vocab=None, merges=None, special_tokens=None):
        self.vocab = vocab or {BYTE_ENCODER[b]: b for b in range(256)}
        self.merges = merges or []
        self.special_tokens = special_tokens or []
        self._build_ranks()

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _build_ranks(self):
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.id_to_token = {i: t for t, i in self.vocab.items()}

    @staticmethod
    def _merge(symbols, pair, merged):
        """从左到右、不重叠地替换一个相邻对。"""
        out, i = [], 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return out

    def train(self, text, vocab_size, special_tokens=None):
        assert vocab_size >= 256
        symbols = [BYTE_ENCODER[b] for b in text.encode("utf-8")]
        self.vocab = {BYTE_ENCODER[b]: b for b in range(256)}
        self.merges = []
        while len(self.vocab) < vocab_size:
            # 公式 (2.5)：统计相邻对频次
            counts = Counter(zip(symbols, symbols[1:]))
            if not counts:
                break
            # 公式 (2.7)：贪心选择当前最高频的对
            pair = max(counts, key=lambda p: (counts[p], p))
            if counts[pair] < 2:
                break
            merged = pair[0] + pair[1]
            self.vocab[merged] = len(self.vocab)          # 公式 (2.6)
            self.merges.append(pair)
            symbols = self._merge(symbols, pair, merged)
        self.special_tokens = special_tokens or []
        for tok in self.special_tokens:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        self._build_ranks()
        return self

    def _encode_ordinary(self, text):
        symbols = [BYTE_ENCODER[b] for b in text.encode("utf-8")]
        while len(symbols) >= 2:
            # 按训练时的合并顺序，优先执行 rank 最小的对
            pairs = set(zip(symbols, symbols[1:]))
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            symbols = self._merge(symbols, best, best[0] + best[1])
        return symbols

    def encode(self, text, *, add_special=True):
        ids = []
        if add_special and "<bos>" in self.vocab:
            ids.append(self.vocab["<bos>"])
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.vocab[part])
            elif part:
                ids.extend(self.vocab[s] for s in self._encode_ordinary(part))
        return ids

    def decode(self, ids, *, skip_special=True):
        pieces = []
        for i in ids:
            tok = self.id_to_token[i]
            if skip_special and tok in self.special_tokens:
                continue
            pieces.append(tok)
        data = bytes(BYTE_DECODER[ch] for ch in "".join(pieces))
        return data.decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab,
                       "merges": [list(p) for p in self.merges],
                       "special_tokens": self.special_tokens},
                      f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        tok = cls(vocab=d["vocab"],
                  merges=[tuple(p) for p in d["merges"]],
                  special_tokens=d["special_tokens"])
        tok._build_ranks()
        return tok


CORPUS = """
语言模型通过预测下一个词元来学习文本的统计规律。分词器把字符串切分成词元序列，是模型看到文本的第一步。
中文没有空格，分词器需要自己决定一个汉字或一个词应该占几个词元。英文单词之间用空格分隔，但标点和大小写仍然需要处理。
代码中的缩进、变量名和运算符对分词器提出了额外的要求。数字串经常被切成不规则的片段，这会影响算术能力。
Natural language processing studies how computers understand and generate human language. Tokenization is the first step of the pipeline.
Large language models are trained on trillions of tokens. The vocabulary size controls the trade-off between sequence length and embedding parameters.
Tokenizer design affects cost, context length, and even arithmetic. A Chinese character is three bytes in UTF-8, so byte-level BPE may need several merges to represent it.
def add(a, b):
    return a + b

for i in range(10):
    print(i, i ** 2)
2024年，模型的参数量从70亿增长到6710亿。上下文窗口从8千扩展到12.8万词元。
""" * 4


def main():
    tok = BPETokenizer().train(
        CORPUS, vocab_size=768, special_tokens=["<bos>", "<eos>"]
    )
    print(f"训练后: vocab_size={tok.vocab_size}  merges={len(tok.merges)}")
    print("往返一致:", tok.decode(tok.encode(CORPUS, add_special=False)) == CORPUS)

    test_zh = "大语言模型的分词器决定了文本被切成多少个词元。"
    test_en = "Tokenization determines how many tokens the model sees."
    for name, s in [("中文", test_zh), ("英文", test_en)]:
        ids = tok.encode(s, add_special=False)
        b = len(s.encode("utf-8"))
        print(f"{name}: chars={len(s)} bytes={b} tokens={len(ids)} "
              f"bytes/token={b / len(ids):.2f}")

    for s in ["strawberry", "1234567890", "def add(a, b):"]:
        ids = tok.encode(s, add_special=False)
        pieces = [tok.id_to_token[i] for i in ids]
        print(f"{s!r}: tokens={len(ids)} pieces={pieces}")

    print("\n词表大小对 Token 数的影响:")
    for V in [256, 384, 512, 768, 1024]:
        t = BPETokenizer().train(CORPUS, vocab_size=V)
        n_zh = len(t.encode(test_zh, add_special=False))
        n_en = len(t.encode(test_en, add_special=False))
        print(f"V_target={V:4d} V_actual={t.vocab_size:4d} "
              f"merges={len(t.merges):4d} zh={n_zh:3d} en={n_en:3d}")


if __name__ == "__main__":
    main()
