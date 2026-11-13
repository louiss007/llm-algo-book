"""GRPO 入口：组内相对优势与裁剪代理目标的最小训练。"""
import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinylm.config import load_config
from tinylm.model import TinyLM
from tinylm.posttrain.grpo import group_advantages, grpo_loss
from tinylm.posttrain.ppo import token_logprobs
from tinylm.train.optim import AdamW, make_param_groups
from tinylm.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_cpu.yaml")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=8)
    args = parser.parse_args()

    set_seed(0)
    cfg = load_config(args.config)
    policy = TinyLM(cfg)
    opt = AdamW(make_param_groups(policy, 0.0), lr=5e-3)
    seq_len = min(16, cfg.max_seq_len)
    seqs = torch.randint(5, cfg.vocab_size, (args.group_size, seq_len))
    attn = torch.ones_like(seqs, dtype=torch.bool)
    mask = torch.zeros_like(seqs, dtype=torch.bool)
    mask[:, seq_len // 2:] = True
    rewards = torch.rand(args.group_size, generator=torch.Generator().manual_seed(0))
    adv = group_advantages(rewards, args.group_size)
    old_policy = copy.deepcopy(policy).eval()
    with torch.no_grad():
        old_logp = token_logprobs(old_policy, seqs, attn)

    first = None
    for _ in range(args.steps):
        logp = token_logprobs(policy, seqs, attn)
        loss, diag = grpo_loss(logp, old_logp, adv, mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        first = float(loss.detach()) if first is None else first
    print(f"GRPO 组大小 {args.group_size}，优势均值 {float(adv.mean()):+.4f}；"
          f"首步损失 {first:.4f}，末步 {float(loss.detach()):.4f}，"
          f"裁剪比例 {diag['clip_frac']:.3f}")


if __name__ == "__main__":
    main()
