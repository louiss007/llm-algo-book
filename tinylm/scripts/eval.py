"""评测入口：留出集损失与困惑度。"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinylm.config import load_config
from tinylm.data.dataset import packed_batches
from tinylm.model import TinyLM
from tinylm.train.checkpoint import load_checkpoint
from tinylm.train.loop import evaluate
from tinylm.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_cpu.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--tokens", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    args = parser.parse_args()

    set_seed(0)
    cfg = load_config(args.config)
    model = TinyLM(cfg)
    if args.ckpt:
        load_checkpoint(args.ckpt, model)
    tokens = torch.randint(0, cfg.vocab_size, (args.tokens,))
    batches = packed_batches(tokens, min(args.block_size, cfg.max_seq_len),
                             args.batch_size, shuffle=False)
    loss = evaluate(model, batches)
    print(f"留出损失 {loss:.4f}，困惑度 {torch.exp(torch.tensor(loss)).item():.2f}")


if __name__ == "__main__":
    main()
