"""监督微调入口：对话模板 + 只对回答计损失。"""
import argparse
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinylm.config import load_config
from tinylm.model import TinyLM
from tinylm.posttrain.sft import SFTConfig, collate_examples, supervision_stats
from tinylm.tokenizer.chat import ChatTemplate
from tinylm.train.loop import TrainConfig, train
from tinylm.utils.seed import set_seed


def make_examples(cfg, template, n, generator):
    examples = []
    for _ in range(n):
        question = torch.randint(5, cfg.vocab_size, (6,), generator=generator).tolist()
        answer = torch.randint(5, cfg.vocab_size, (4,), generator=generator).tolist()
        examples.append(template.render([("user", question), ("assistant", answer)]))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_cpu.yaml")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--out", default="runs/sft/last.pt")
    args = parser.parse_args()

    set_seed(0)
    cfg = load_config(args.config)
    model = TinyLM(cfg)
    template = ChatTemplate()
    examples = make_examples(cfg, template, args.examples,
                             torch.Generator().manual_seed(0))
    stats = supervision_stats(examples)
    batch = collate_examples(examples, max_len=cfg.max_seq_len)
    batches = itertools.repeat(batch)

    history = train(model, batches, TrainConfig(
        lr=SFTConfig().lr, warmup=min(5, args.steps // 4),
        max_steps=args.steps, grad_clip=1.0, log_every=max(1, args.steps // 2),
    ))
    from tinylm.train.checkpoint import save_checkpoint
    save_checkpoint(args.out, model, step=args.steps)
    print(f"监督比例 {stats['ratio']:.3f}（{stats['supervised']}/{stats['tokens']}）；"
          f"首步损失 {history['losses'][0]:.4f}，末步 {history['losses'][-1]:.4f}")


if __name__ == "__main__":
    main()
