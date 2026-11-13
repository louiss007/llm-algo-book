"""第 18 章：可验证奖励与组相对策略优化（examples/ch18_grpo.py）。"""
import time

import torch

from ch08 import ModelConfig, TinyLM
from tinylm.posttrain import grpo as G

PAD, BOS, Q, SEP, EOS = 0, 1, 2, 3, 4
DIGIT0 = 5
PROMPT_LEN = 4
N_PROMPTS = 10
SEEDS = (0, 1, 2)

def prompt(a):
    return [BOS, Q, DIGIT0 + a, SEP]

def answer(a):
    return DIGIT0 + ((a + 3) % 10)

def sft_example(a, gen, p_correct=0.3):
    """弱监督数据：只有 30% 的答案正确，其余从数字词表随机抽取。"""
    tok = answer(a) if float(torch.rand(1, generator=gen)) < p_correct else \
        int(torch.randint(DIGIT0, DIGIT0 + 10, (1,), generator=gen))
    ids = prompt(a) + [tok, EOS]
    labels = [-100] * len(ids)
    labels[-3] = tok          # 第 4 个位置（SEP）预测答案
    labels[-2] = EOS          # 第 5 个位置（答案）预测 EOS
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
            "attention_mask": torch.ones(len(ids), dtype=torch.bool)}

def collate(examples):
    n = max(e["input_ids"].numel() for e in examples)
    ids = torch.zeros(len(examples), n, dtype=torch.long)
    labels = torch.full((len(examples), n), -100, dtype=torch.long)
    mask = torch.zeros(len(examples), n, dtype=torch.bool)
    for i, e in enumerate(examples):
        k = e["input_ids"].numel()
        ids[i, :k] = e["input_ids"]
        labels[i, :k] = e["labels"]
        mask[i, :k] = True
    return ids, labels, mask

def make_model(seed):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=16, d_model=32, n_layers=2, n_heads=4,
                      d_ff=128, max_seq_len=32, pos_encoding="rope",
                      norm="rmsnorm", ffn="swiglu", bias=False)
    return TinyLM(cfg), cfg

def train_sft(model, seed, steps=50, p_correct=0.3):
    gen = torch.Generator().manual_seed(1000 + seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, N_PROMPTS, (32,), generator=gen)
        ids, labels, mask = collate([sft_example(int(i), gen, p_correct)
                                     for i in idx])
        out = model(ids, attention_mask=mask, labels=labels)
        opt.zero_grad()
        out.loss.backward()
        opt.step()

@torch.no_grad()
def greedy_accuracy(model, prompts=None):
    model.eval()
    a = list(range(N_PROMPTS)) if prompts is None else prompts
    seqs = torch.tensor([prompt(x) for x in a])
    done = torch.zeros(len(a), dtype=torch.bool)
    for _ in range(4):
        nxt = model(seqs).logits[:, -1].argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        seqs = torch.cat([seqs, nxt[:, None]], dim=1)
        done = done | (nxt == EOS)
        if bool(done.all()):
            break
    goals = torch.tensor([answer(x) for x in a])
    return float(G.verifiable_reward(seqs, PROMPT_LEN, goals).mean())

@torch.no_grad()
def sampled_accuracy(model, seed, group_size=8):
    """用固定评测种子采 ``group_size`` 条，返回 pass@1 与 pass@G。"""
    gen = torch.Generator().manual_seed(9000 + seed)
    prompts = torch.tensor([prompt(a) for a in range(N_PROMPTS)])
    seqs, attn = G.sample_groups(
        model, prompts, G.GRPOConfig(group_size=group_size, max_new=4), gen)
    goals = torch.tensor([answer(a) for a in range(N_PROMPTS)])
    rew = G.verifiable_reward(seqs, PROMPT_LEN, goals.repeat_interleave(group_size))
    rew = rew.view(N_PROMPTS, group_size)
    return float(rew.mean()), float(rew.any(1).float().mean())

def train_grpo(sft_state, cfg, gcfg, seed, rounds=300, batch=16):
    policy = TinyLM(cfg)
    policy.load_state_dict(sft_state)
    opt = torch.optim.AdamW(policy.parameters(), lr=gcfg.lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(7000 + seed)
    rew_hist, zero_hist = [], []
    for _ in range(rounds):
        idx = torch.randint(0, N_PROMPTS, (batch,), generator=gen)
        prompts = torch.tensor([prompt(int(i)) for i in idx])
        seqs, attn = G.sample_groups(policy, prompts, gcfg, gen)
        goals = torch.tensor([answer(int(i)) for i in idx])
        goals = goals.repeat_interleave(gcfg.group_size)
        rew = G.verifiable_reward(seqs, PROMPT_LEN, goals)
        adv = G.group_advantages(rew, gcfg.group_size, gcfg.normalize_std,
                                 gcfg.leave_one_out)
        zero = G.zero_variance_groups(rew, gcfg.group_size)
        rew_hist.append(float(rew.mean()))
        zero_hist.append(float(zero.float().mean()))

        mask = (seqs != PAD) & (
            torch.arange(seqs.size(1))[None, :] >= PROMPT_LEN)
        with torch.no_grad():
            old = G.tempered_logprobs(policy, seqs, attn, gcfg.temperature)
        policy.train()
        for _ in range(gcfg.epochs):
            perm = torch.randperm(seqs.size(0), generator=gen)
            for start in range(0, seqs.size(0), gcfg.minibatch):
                mb = perm[start:start + gcfg.minibatch]
                logp = G.tempered_logprobs(policy, seqs[mb], attn[mb],
                                           gcfg.temperature)
                loss, _ = G.grpo_loss(logp, old[mb], adv[mb], mask[mb], None,
                                      beta=gcfg.beta, clip_low=gcfg.clip_low,
                                      clip_high=gcfg.clip_high,
                                      token_level=gcfg.token_level)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                               gcfg.max_grad_norm)
                opt.step()
    return policy, sum(rew_hist) / len(rew_hist), sum(zero_hist) / len(zero_hist)

VARIANTS = (
    ("G=1", dict(group_size=1)),
    ("G=4", dict(group_size=4)),
    ("G=8", dict(group_size=8)),
)

def main():
    t0 = time.time()
    rows = {"SFT": ([], [], [])}
    for name, _ in VARIANTS:
        rows[name] = ([], [], [])
    for seed in SEEDS:
        model, cfg = make_model(seed)
        train_sft(model, seed)
        sft_state = {k: v.clone() for k, v in model.state_dict().items()}
        p1, p8 = sampled_accuracy(model, seed)
        g3 = greedy_accuracy(model)
        rows["SFT"][0].append(g3)
        rows["SFT"][1].append(p1)
        rows["SFT"][2].append(p8)
        print(f"seed {seed} SFT: greedy {g3:.2f} "
              f"pass@1 {p1:.2f} pass@8 {p8:.2f}")
        for name, kw in VARIANTS:
            gcfg = G.GRPOConfig(lr=1e-3, epochs=1, minibatch=64, max_new=4,
                                beta=0.0, **kw)
            ts = time.time()
            policy, rew, zero = train_grpo(sft_state, cfg, gcfg, seed)
            p1, p8 = sampled_accuracy(policy, seed)
            g3 = greedy_accuracy(policy)
            rows[name][0].append(g3)
            rows[name][1].append(p1)
            rows[name][2].append(p8)
            print(f"  {name:<11} greedy {g3:.2f} "
                  f"pass@1 {p1:.2f} pass@8 {p8:.2f} "
                  f"reward {rew:.2f} zero {zero:.2f} {time.time()-ts:.1f}s")
    print("\n3-seed means")
    print(f"{'':<11}{'greedy':>7}{'pass@1':>8}{'pass@8':>8}")
    for name, (g3, p1, p8) in rows.items():
        print(f"{name:<11}{sum(g3)/len(g3):>7.2f}"
              f"{sum(p1)/len(p1):>8.2f}{sum(p8)/len(p8):>8.2f}")
    print(f"total {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
