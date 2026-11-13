"""生成入口：加载检查点并解码。"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinylm.config import load_config
from tinylm.inference.sample import SamplingConfig, generate
from tinylm.model import TinyLM
from tinylm.tokenizer.bpe import BPETokenizer
from tinylm.train.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_cpu.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--prompt", default="从前有一座山")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = TinyLM(cfg).eval()
    if args.ckpt:
        load_checkpoint(args.ckpt, model)
    tokenizer = BPETokenizer.load(args.tokenizer) if args.tokenizer else None

    if tokenizer is not None:
        ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
        ids = ids.clamp(max=cfg.vocab_size - 1)
    else:
        ids = torch.randint(5, cfg.vocab_size, (1, 4))

    sampling = None if args.temperature <= 0 else SamplingConfig(
        temperature=args.temperature, top_k=50, top_p=0.95, seed=0)
    out = generate(model, ids, max_new_tokens=args.max_new_tokens,
                   sampling=sampling, use_cache=False)
    ids = out[0].tolist()
    if tokenizer is not None and all(i in tokenizer.id_to_token for i in ids):
        print(tokenizer.decode(ids))
    else:
        print(ids)


if __name__ == "__main__":
    main()
