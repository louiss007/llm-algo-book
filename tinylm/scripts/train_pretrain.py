"""预训练入口：在合成 Token 流上跑通前向、反向与检查点。"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinylm.config import load_config
from tinylm.data.dataset import packed_batches
from tinylm.model import TinyLM
from tinylm.train.checkpoint import save_checkpoint
from tinylm.train.loop import TrainConfig, train
from tinylm.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_cpu.yaml")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--tokens", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--out", default="runs/cpu/last.pt")
    args = parser.parse_args()

    set_seed(0)
    cfg = load_config(args.config)
    model = TinyLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (args.tokens,))
    block = min(args.block_size, cfg.max_seq_len)
    batches = packed_batches(tokens, block, args.batch_size, seed=0)

    history = train(model, batches, TrainConfig(
        lr=3e-3, warmup=min(5, args.steps // 4), max_steps=args.steps,
        grad_clip=1.0, log_every=max(1, args.steps // 3),
    ))
    save_checkpoint(args.out, model, step=args.steps, config=vars(args))
    params = sum(p.numel() for p in model.parameters())
    print(f"参数量 {params:,d}；首步损失 {history['losses'][0]:.4f}，"
          f"末步损失 {history['losses'][-1]:.4f}；检查点 {args.out}")


if __name__ == "__main__":
    main()
