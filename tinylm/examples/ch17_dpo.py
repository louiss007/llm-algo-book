"""第 17 章：直接偏好优化（examples/ch17_dpo.py）。"""
import copy
import random
import time

import torch

from ch08 import ModelConfig, TinyLM
from tinylm.posttrain import dpo as D
from tinylm.posttrain import ppo as P
from tinylm.posttrain import sft
from tinylm.tokenizer.chat import ChatTemplate

PAD, BOS, EOS, USER, ASSISTANT = 0, 1, 4, 2, 3
TOPICS = (5, 6, 7, 8)
RESP_IDS = tuple(range(8, 16))
TARGET = {t: 8 + (t - 5) for t in TOPICS}
DECOY = {t: 12 + (t - 5) for t in TOPICS}
VOCAB = 16
PROMPT_LEN = 4
SEEDS = (0, 1, 2)
TEMPLATE = ChatTemplate()

def make_model(seed):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2,
                      d_ff=128, max_seq_len=32, pos_encoding="rope",
                      norm="rmsnorm", ffn="swiglu", bias=False)
    return TinyLM(cfg), cfg

def prompt_ids(topic):
    return [BOS, USER, int(topic), ASSISTANT]

def make_response(gen, topic, p_target=0.55, p_decoy=0.10, lo=3, hi=5):
    """行为策略：55% 目标词、10% 形近干扰词、其余从响应词表均匀采样。"""
    length = int(torch.randint(lo, hi + 1, (1,), generator=gen))
    out = []
    for _ in range(length):
        u = float(torch.rand(1, generator=gen))
        if u < p_target:
            out.append(TARGET[topic])
        elif u < p_target + p_decoy:
            out.append(DECOY[topic])
        else:
            out.append(RESP_IDS[int(torch.randint(0, len(RESP_IDS), (1,), generator=gen))])
    return out

def make_example(topic, ys):
    return TEMPLATE.render([("user", [int(topic)]), ("assistant", ys)])

def make_sft_data(seed, size):
    gen = torch.Generator().manual_seed(seed)
    data = []
    for _ in range(size):
        topic = TOPICS[int(torch.randint(0, len(TOPICS), (1,), generator=gen))]
        data.append(make_example(topic, make_response(gen, topic)))
    return data

def train_sft(model, data, seed, steps=400, lr=3e-3, batch_size=32):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(1000 + seed)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(data), (batch_size,), generator=gen)
        batch = sft.collate_examples([data[int(i)] for i in idx])
        out = model(batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    return model

def biased_reward(topic, ys, noise=0.0, rng=None):
    """标注者的真实偏好：目标词 + 形近词 + 长度，后两项是偏置。"""
    correct = sum(y == TARGET[topic] for y in ys) / len(ys)
    decoy = sum(y == DECOY[topic] for y in ys) / len(ys)
    length_bonus = 0.35 * (len(ys) - 3) / 3.0
    noise = rng.gauss(0.0, noise) if (noise and rng) else 0.0
    return correct + 0.30 * decoy + length_bonus + noise

def debiased_reward(topic, ys, noise=0.0, rng=None):
    """去掉长度与形近词偏置，只看目标词比例。"""
    correct = sum(y == TARGET[topic] for y in ys) / len(ys)
    noise = rng.gauss(0.0, noise) if (noise and rng) else 0.0
    return correct + noise

def make_pairs(seed, size, reward_fn, noise=0.03):
    gen = torch.Generator().manual_seed(seed + 77)
    rng = random.Random(seed + 77)
    pairs = []
    for _ in range(size):
        topic = TOPICS[int(torch.randint(0, len(TOPICS), (1,), generator=gen))]
        y1 = make_response(gen, topic)
        y2 = make_response(gen, topic)
        r1 = reward_fn(topic, y1, noise, rng)
        r2 = reward_fn(topic, y2, noise, rng)
        if abs(r1 - r2) < 1e-6:
            continue
        win, lose = (y1, y2) if r1 > r2 else (y2, y1)
        pairs.append({"topic": topic,
                      "win": make_example(topic, win)["input_ids"],
                      "lose": make_example(topic, lose)["input_ids"]})
    return pairs

def pad_batch(rows, pad=PAD, prompt_len=PROMPT_LEN):
    length = max(r.numel() for r in rows)
    ids = torch.full((len(rows), length), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), length), dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, :r.numel()] = r
        mask[i, :r.numel()] = True
    positions = torch.arange(length)[None, :]
    completion = mask & (positions >= prompt_len)
    return ids, mask, completion

def pair_tensors(batch):
    win_ids, win_mask, win_comp = pad_batch([b["win"] for b in batch])
    lose_ids, lose_mask, lose_comp = pad_batch([b["lose"] for b in batch])
    return win_ids, win_mask, win_comp, lose_ids, lose_mask, lose_comp

@torch.no_grad()
def pair_stats(policy, ref, pairs, beta):
    win_ids, win_mask, win_comp, lose_ids, lose_mask, lose_comp = pair_tensors(pairs)
    pc = D.completion_logprobs(policy, win_ids, win_mask, win_comp)
    pr = D.completion_logprobs(policy, lose_ids, lose_mask, lose_comp)
    rc = D.completion_logprobs(ref, win_ids, win_mask, win_comp)
    rr = D.completion_logprobs(ref, lose_ids, lose_mask, lose_comp)
    margin = beta * D.log_ratio_margin(pc, pr, rc, rr)
    return float(D.preference_accuracy(margin)), float(margin.mean())

def train_dpo(sft_state, cfg, pairs, dcfg, seed, steps=300, log_every=0):
    policy = TinyLM(cfg)
    policy.load_state_dict(sft_state)
    ref = TinyLM(cfg)
    ref.load_state_dict(sft_state)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    opt = torch.optim.AdamW(policy.parameters(), lr=dcfg.lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(4000 + seed)
    history = []
    done = 0
    for _ in range(dcfg.epochs):
        for _ in range(steps):
            idx = torch.randint(0, len(pairs), (dcfg.batch_size,), generator=gen)
            batch_data = [pairs[int(i)] for i in idx]
            win_ids, win_mask, win_comp, lose_ids, lose_mask, lose_comp = \
                pair_tensors(batch_data)
            pc = D.completion_logprobs(policy, win_ids, win_mask, win_comp,
                                       dcfg.length_normalize)
            pr = D.completion_logprobs(policy, lose_ids, lose_mask, lose_comp,
                                       dcfg.length_normalize)
            with torch.no_grad():
                rc = D.completion_logprobs(ref, win_ids, win_mask, win_comp,
                                           dcfg.length_normalize)
                rr = D.completion_logprobs(ref, lose_ids, lose_mask, lose_comp,
                                           dcfg.length_normalize)
            loss = D.dpo_loss(pc, pr, rc, rr, beta=dcfg.beta,
                              label_smoothing=dcfg.label_smoothing,
                              loss_type=dcfg.loss_type)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), dcfg.max_grad_norm)
            opt.step()
            done += 1
            if log_every and done % log_every == 0:
                history.append((done, float(loss.detach())))
    return policy, ref, history

@torch.no_grad()
def sample_for_topics(policy, topics, cfg, generator, batch=256):
    policy.eval()
    rows = []
    for start in range(0, len(topics), batch):
        chunk = topics[start:start + batch]
        prompts = torch.tensor([prompt_ids(t) for t in chunk], dtype=torch.long)
        rows.append(P.sample_responses(policy, prompts, cfg, generator))
    return torch.cat(rows)

@torch.no_grad()
def evaluate(policy, ref, seed, n=256):
    topics = [TOPICS[i % len(TOPICS)] for i in range(n)]
    gen = torch.Generator().manual_seed(5000 + seed)
    seqs = sample_for_topics(policy, topics, P.RLConfig(), gen)
    attn = seqs != PAD
    mask = P.response_mask(seqs, PROMPT_LEN, PAD)
    last = attn.long().sum(-1) - 1
    content = mask.clone()
    content[torch.arange(seqs.size(0)), last] = False
    target = torch.tensor([TARGET[int(t)] for t in topics])
    decoy = torch.tensor([DECOY[int(t)] for t in topics])
    denom = content.sum(1).clamp_min(1)
    true = ((seqs == target[:, None]) & content).sum(1) / denom
    fake = ((seqs == decoy[:, None]) & content).sum(1) / denom
    length = content.sum(1).float()
    logp = P.token_logprobs(policy, seqs, attn)
    ref_logp = P.token_logprobs(ref, seqs, attn)
    kl = P.kl_penalty(logp, ref_logp, mask)
    return {"true": true.mean().item(), "fake": fake.mean().item(),
            "len": length.mean().item(), "kl": float(kl)}

def fmt(values):
    t = torch.tensor(values, dtype=torch.float64)
    return f"{t.mean():.3f}±{t.std(unbiased=False):.3f}"

def main():
    t0 = time.time()
    keys = ("sft", "b01", "b05", "b20", "ipo", "deb05")
    rows = {k: [] for k in keys}
    pair_rows = {k: [] for k in keys}
    deb_rows = []
    traj = None
    for seed in SEEDS:
        model, cfg = make_model(seed)
        train_sft(model, make_sft_data(seed, 512), seed)
        sft_state = copy.deepcopy(model.state_dict())

        train_pairs = make_pairs(seed + 100, 2000, biased_reward)
        debiased_pairs = make_pairs(seed + 100, 2000, debiased_reward)
        eval_pairs = make_pairs(seed + 900, 400, biased_reward)
        eval_debiased = make_pairs(seed + 900, 400, debiased_reward)

        ref = TinyLM(cfg)
        ref.load_state_dict(sft_state)
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
        rows["sft"].append(evaluate(model, ref, seed))

        configs = (
            ("b01", train_pairs, D.DPOConfig(beta=0.10, lr=5e-4, batch_size=32)),
            ("b05", train_pairs, D.DPOConfig(beta=0.50, lr=5e-4, batch_size=32)),
            ("b20", train_pairs, D.DPOConfig(beta=2.00, lr=5e-4, batch_size=32)),
            ("ipo", train_pairs, D.DPOConfig(beta=0.50, lr=5e-4, batch_size=32,
                                             loss_type="ipo")),
            ("deb05", debiased_pairs, D.DPOConfig(beta=0.50, lr=5e-4, batch_size=32)),
        )
        line = f"seed {seed}: SFT 正确 {rows['sft'][-1]['true']:.3f}"
        for key, pairs, dcfg in configs:
            policy, ref2, history = train_dpo(
                sft_state, cfg, pairs, dcfg, seed,
                log_every=50 if (seed == 0 and key == "b01") else 0)
            result = evaluate(policy, ref2, seed)
            rows[key].append(result)
            pair_rows[key].append(pair_stats(policy, ref2, eval_pairs, dcfg.beta))
            if key == "deb05":
                deb_rows.append(pair_stats(policy, ref2, eval_debiased, dcfg.beta))
            line += (f" | {key} 正确 {result['true']:.3f} 长度 {result['len']:.2f}")
            if history:
                traj = history
        print(line)

    print("\nDPO 训练轨迹（seed 0，偏置偏好，β=0.10，每 50 步）")
    print("步数   批损失")
    for step, loss in traj:
        print(f"{step:>3d}   {loss:.4f}")

    print("\n方法                 正确率              干扰词              长度               逐 Token KL")
    labels = (("sft", "SFT 策略"), ("b01", "DPO β=0.10"),
              ("b05", "DPO β=0.50"), ("b20", "DPO β=2.00"),
              ("ipo", "IPO β=0.50"), ("deb05", "DPO 去偏 β=0.50"))
    for key, name in labels:
        r = rows[key]
        print(f"{name:<18}   {fmt([x['true'] for x in r])}   "
              f"{fmt([x['fake'] for x in r])}   {fmt([x['len'] for x in r])}   "
              f"{fmt([x['kl'] for x in r])}")

    print("\n留出偏好对：隐式奖励排序准确率 / 平均间隔（偏置标签）")
    for key, name in labels[1:]:
        acc = pair_rows[key]
        print(f"{name:<18}   准确率 {fmt([a for a, _ in acc])}   "
              f"间隔 {fmt([m for _, m in acc])}")
    print("\n去偏标签上的隐式奖励（DPO 去偏训练）")
    print(f"{'DPO 去偏':<18}   准确率 {fmt([a for a, _ in deb_rows])}   "
          f"间隔 {fmt([m for _, m in deb_rows])}")
    print(f"\n总耗时 {time.time() - t0:.1f} s")

if __name__ == "__main__":
    main()
