"""训练字节级 BPE 分词器。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinylm.config import load_config
from tinylm.tokenizer.bpe import CORPUS, BPETokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_cpu.yaml")
    parser.add_argument("--data", nargs="*", default=None,
                        help="待训练文本文件；省略时使用内置示例语料")
    parser.add_argument("--out", default="runs/tokenizer.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    texts = [Path(p).read_text(encoding="utf-8") for p in args.data] \
        if args.data else [CORPUS]
    text = "\n".join(texts)
    tokenizer = BPETokenizer().train(
        text, vocab_size=cfg.vocab_size,
        special_tokens=["<bos>", "<eos>", "<pad>"],
    )
    tokenizer.save(args.out)
    sample = "语言模型学习下一个词元。"
    ids = tokenizer.encode(sample)
    print(f"语料字符数 {len(text)}，词表 {tokenizer.vocab_size}，合并 {len(tokenizer.merges)}")
    print(f"示例编码 {ids}")
    print(f"往返一致 {tokenizer.decode(ids) == sample}，已保存到 {args.out}")


if __name__ == "__main__":
    main()
