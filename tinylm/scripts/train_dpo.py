"""DPO 入口：用成对偏好数据做一次最小训练。"""
import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinylm.config import load_config
from tinylm.model import TinyLM
from tinylm.posttrain.dpo import DPOConfig, completion_logprobs, dpo_loss
from tinylm.train.optim import AdamW, make_param_groups
from tinylm.utils.seed import set_seed


def make_pairs(cfg, n, seq_len, prompt_len, generator):
    chosen = torch.randint(5, cfg.vocab_size, (n, seq_len), generator=generator)
    rejected = torch.randint(5, cfg.vocab_size, (n, seq_len), generator=generator)
    mask = torch.zeros(n, seq_len, dtype=torch.bool)
    mask[:, prompt_len:] = True
    attn = torch.ones(n, seq_len, dtype=torch.bool)
    return chosen, rejected, mask, attn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_cpu.yaml")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    set_seed(0)
    cfg = load_config(args.config)
    policy = TinyLM(cfg)
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = AdamW(make_param_groups(policy, 0.0), lr=5e-3)
    dpo_cfg = DPOConfig()
    seq_len, prompt_len = min(16, cfg.max_seq_len), 6
    chosen, rejected, mask, attn = make_pairs(
        cfg, args.batch_size, seq_len, prompt_len, torch.Generator().manual_seed(0))

    first = None
    for _ in range(args.steps):
        pc = completion_logprobs(policy, chosen, attn, mask)
        pr = completion_logprobs(policy, rejected, attn, mask)
        rc = completion_logprobs(ref, chosen, attn, mask)
        rr = completion_logprobs(ref, rejected, attn, mask)
        loss = dpo_loss(pc, pr, rc, rr, beta=dpo_cfg.beta,
                        label_smoothing=dpo_cfg.label_smoothing,
                        loss_type=dpo_cfg.loss_type)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        first = float(loss.detach()) if first is None else first
    print(f"DPO 步数 {args.steps}，首步损失 {first:.4f}，"
          f"末步损失 {float(loss.detach()):.4f}")


if __name__ == "__main__":
    main()
