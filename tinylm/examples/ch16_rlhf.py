"""第 16 章：奖励模型与 PPO（examples/ch16_rlhf.py）。"""
import copy
import random
import time

import torch

from ch08 import ModelConfig, TinyLM
from tinylm.posttrain import ppo as P
from tinylm.posttrain import rm as RM
from tinylm.posttrain import sft
from tinylm.tokenizer.chat import ChatTemplate

PAD, BOS, EOS, USER, ASSISTANT = 0, 1, 4, 2, 3
TOPICS = (5, 6, 7, 8)
RESP_IDS = tuple(range(8, 16))
TARGET = {t: 8 + (t - 5) for t in TOPICS}
DECOY = {t: 12 + (t - 5) for t in TOPICS}
VOCAB = 16
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

def train_sft(model, data, seed, steps=500, lr=3e-3, batch_size=32):
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

def human_reward(topic, ys, noise=0.0, rng=None):
    """偏好标签背后的“人类”奖励。

    标注者把形近干扰词误当成正确答案、又偏好更长的回答；这两项偏置
    会被奖励模型学走，而真实任务只认目标词。
    """
    correct = sum(y == TARGET[topic] for y in ys) / len(ys)
    decoy = sum(y == DECOY[topic] for y in ys) / len(ys)
    length_bonus = 0.35 * (len(ys) - 3) / 3.0
    noise = rng.gauss(0.0, noise) if (noise and rng) else 0.0
    return correct + 0.30 * decoy + length_bonus + noise

def make_pairs(seed, size, noise=0.03):
    gen = torch.Generator().manual_seed(seed + 77)
    rng = random.Random(seed + 77)
    pairs = []
    for _ in range(size):
        topic = TOPICS[int(torch.randint(0, len(TOPICS), (1,), generator=gen))]
        y1 = make_response(gen, topic)
        y2 = make_response(gen, topic)
        r1 = human_reward(topic, y1, noise, rng)
        r2 = human_reward(topic, y2, noise, rng)
        if abs(r1 - r2) < 1e-6:
            continue
        win, lose = (y1, y2) if r1 > r2 else (y2, y1)
        pairs.append({
            "win": make_example(topic, win)["input_ids"],
            "lose": make_example(topic, lose)["input_ids"],
        })
    return pairs

def pad_batch(rows, pad=PAD):
    length = max(r.numel() for r in rows)
    ids = torch.full((len(rows), length), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), length), dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, :r.numel()] = r
        mask[i, :r.numel()] = True
    return ids, mask

def train_reward(model, pairs, seed, steps=400, lr=1e-3, batch_size=64):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(2000 + seed)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(pairs), (batch_size,), generator=gen)
        batch = [pairs[int(i)] for i in idx]
        win_ids, win_mask = pad_batch([b["win"] for b in batch])
        lose_ids, lose_mask = pad_batch([b["lose"] for b in batch])
        rw = model.score(win_ids, win_mask)
        rl = model.score(lose_ids, lose_mask)
        loss = RM.bradley_terry_loss(rw, rl)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model

@torch.no_grad()
def probe_reward_model(model):
    """探针：固定候选对，测量 RM 对长度与正确性的偏好。

    (1) 长度探针：两边全部命中目标，长度 3 vs 5；
    (2) 正确性探针：同为长度 4，4 个目标 vs 2 个目标；
    (3) 干扰词探针：同为长度 4，4 个目标 vs 2 个目标 + 2 个形近词。
    """
    long_win = correct_win = decoy_win = 0
    for topic in TOPICS:
        g, d = TARGET[topic], DECOY[topic]
        other = 8 + (TARGET[topic] - 8 + 4) % 8   # 同词表、非目标非干扰
        cases = (
            ("len", [g] * 3, [g] * 5),                 # 偏好更长的全对回答
            ("cor", [g] * 4, [g] * 3 + [other]),       # 偏好全对
            ("dec", [g] * 3 + [d], [g] * 3 + [other]), # 偏好形近词
        )
        for kind, a, b in cases:
            ids, mask = pad_batch([make_example(topic, a)["input_ids"],
                                   make_example(topic, b)["input_ids"]])
            score = model.score(ids, mask)
            win = int(score[1] > score[0]) if kind == "len" else int(score[0] > score[1])
            if kind == "len":
                long_win += win
            elif kind == "cor":
                correct_win += win
            else:
                decoy_win += win
    n = len(TOPICS)
    return {"len": long_win / n, "cor": correct_win / n, "dec": decoy_win / n}

@torch.no_grad()
def reward_accuracy(model, pairs):
    model.eval()
    win_ids, win_mask = pad_batch([p["win"] for p in pairs])
    lose_ids, lose_mask = pad_batch([p["lose"] for p in pairs])
    rw = model.score(win_ids, win_mask)
    rl = model.score(lose_ids, lose_mask)
    return float(RM.pairwise_accuracy(rw, rl))

@torch.no_grad()
def sample_for_topics(policy, topics, cfg, generator, batch=256):
    policy.eval()
    rows = []
    for start in range(0, len(topics), batch):
        chunk = topics[start:start + batch]
        prompts = torch.tensor([prompt_ids(t) for t in chunk], dtype=torch.long)
        rows.append(P.sample_responses(policy, prompts, cfg, generator))
    return torch.cat(rows)

def sequence_stats(policy, ref, reward_model, seqs, topics, prompt_len=4):
    """真实正确率、内容长度、RM 分数与逐 Token KL。"""
    attn = seqs != PAD
    mask = P.response_mask(seqs, prompt_len, PAD)
    last = attn.long().sum(-1) - 1
    content = mask.clone()
    content[torch.arange(seqs.size(0)), last] = False
    target = torch.tensor([TARGET[int(t)] for t in topics])
    decoy = torch.tensor([DECOY[int(t)] for t in topics])
    true = ((seqs == target[:, None]) & content).sum(1) / content.sum(1)
    fake = ((seqs == decoy[:, None]) & content).sum(1) / content.sum(1)
    length = content.sum(1).float()
    rm_score = reward_model.score(seqs, attn)
    logp = P.token_logprobs(policy, seqs, attn)
    ref_logp = P.token_logprobs(ref, seqs, attn)
    k1 = ((logp - ref_logp) * mask).sum(1) / mask.sum(1).clamp_min(1)
    return true, fake, length, rm_score, k1

def eval_topics(n=256):
    return [TOPICS[i % len(TOPICS)] for i in range(n)]

@torch.no_grad()
def evaluate(policy, ref, reward_model, seed, n=256):
    topics = eval_topics(n)
    gen = torch.Generator().manual_seed(5000 + seed)
    seqs = sample_for_topics(policy, topics, P.RLConfig(), gen)
    true, fake, length, rm_score, kl = sequence_stats(
        policy, ref, reward_model, seqs, topics)
    return {"true": true.mean().item(), "fake": fake.mean().item(),
            "len": length.mean().item(), "rm": rm_score.mean().item(),
            "kl": kl.mean().item()}

@torch.no_grad()
def evaluate_best_of_n(policy, ref, reward_model, n_best, seed, n=256):
    topics = eval_topics(n)
    gen = torch.Generator().manual_seed(6000 + seed)
    repeated = [t for t in topics for _ in range(n_best)]
    seqs = sample_for_topics(policy, repeated, P.RLConfig(), gen)
    true, fake, length, rm_score, kl = sequence_stats(
        policy, ref, reward_model, seqs, repeated)
    b = len(topics)
    pick = rm_score.view(b, n_best).argmax(1)
    idx = torch.arange(b) * n_best + pick
    return {"true": true[idx].mean().item(), "fake": fake[idx].mean().item(),
            "len": length[idx].mean().item(), "rm": rm_score[idx].mean().item(),
            "kl": kl[idx].mean().item()}

def run_ppo(sft_model, cfg_model, reward_model, beta, seed, iters=40,
            batch=48, log_every=0, eval_seed=None, **cfg_kwargs):
    cfg = P.RLConfig(beta=beta, **cfg_kwargs)
    policy = TinyLM(cfg_model)
    policy.load_state_dict(sft_model.state_dict())
    ref = TinyLM(cfg_model)
    ref.load_state_dict(sft_model.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    value = RM.ScalarModel(cfg_model)
    value.load_state_dict(reward_model.state_dict())

    opt_pi = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=0.0)
    opt_v = torch.optim.AdamW(value.parameters(), lr=cfg.lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(3000 + seed)
    trajectory = []
    for it in range(iters):
        topic_idx = torch.randint(0, len(TOPICS), (batch,), generator=gen)
        topics = [TOPICS[int(i)] for i in topic_idx]
        prompts = torch.tensor([prompt_ids(t) for t in topics], dtype=torch.long)
        batch_data = P.collect_rollout(
            policy, ref, value, reward_model, prompts, cfg, generator=gen)
        P.ppo_update(policy, value, opt_pi, opt_v, batch_data, cfg, gen)
        if log_every and (it + 1) % log_every == 0:
            s = evaluate(policy, ref, reward_model,
                         seed if eval_seed is None else eval_seed)
            trajectory.append((it + 1, s))
    final = evaluate(policy, ref, reward_model,
                     seed if eval_seed is None else eval_seed)
    return policy, final, trajectory

def fmt(values):
    t = torch.tensor(values, dtype=torch.float64)
    return f"{t.mean():.3f}±{t.std(unbiased=False):.3f}"

def main():
    t0 = time.time()
    rows = {k: [] for k in ("sft", "ppo01", "ppo05", "ppo20", "bon8")}
    traj = None
    for seed in SEEDS:
        model, cfg = make_model(seed)
        sft_data = make_sft_data(seed, 512)
        train_sft(model, sft_data, seed)
        sft_state = copy.deepcopy(model.state_dict())

        rm_model, _ = RM.init_scalar_from_lm(cfg, sft_state, seed=seed + 50)
        train_pairs = make_pairs(seed + 100, 2000)
        eval_pairs = make_pairs(seed + 900, 400)
        train_reward(rm_model, train_pairs, seed)
        acc = reward_accuracy(rm_model, eval_pairs)
        probes = probe_reward_model(rm_model)

        ref = TinyLM(cfg)
        ref.load_state_dict(sft_state)
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
        sft_eval = evaluate(model, ref, rm_model, seed)
        rows["sft"].append(sft_eval)
        rows["bon8"].append(evaluate_best_of_n(model, ref, rm_model, 8, seed))
        line = (f"seed {seed}: RM 偏好准确率 {acc:.3f}，长度探针 "
                f"{probes['len']:.3f}，正确性探针 {probes['cor']:.3f}，"
                f"干扰词探针 {probes['dec']:.3f}")
        for key, beta in (("ppo01", 0.10), ("ppo05", 0.50), ("ppo20", 2.00)):
            _, final, history = run_ppo(
                model, cfg, rm_model, beta, seed, iters=50,
                log_every=10 if (seed == 0 and abs(beta - 0.5) < 1e-9) else 0)
            rows[key].append(final)
            if history:
                traj = history
            line += f" | β={beta:.2f} 正确 {final['true']:.3f} 长度 {final['len']:.2f}"
        print(line)

    print("\nPPO 轨迹（seed 0，β=0.5，每 10 轮评测）")
    print("轮次   正确率   干扰词   长度   RM 分数   逐 Token log 比")
    for it, s in traj:
        print(f"{it:>3d}   {s['true']:.3f}   {s['fake']:.3f}   "
              f"{s['len']:.2f}   {s['rm']:.3f}    {s['kl']:.4f}")

    print("\n方法                 正确率              干扰词              长度                RM 分数")
    labels = (("sft", "SFT 策略"), ("ppo01", "PPO β=0.10"),
              ("ppo05", "PPO β=0.50"), ("ppo20", "PPO β=2.00"),
              ("bon8", "SFT + Best-of-8"))
    for key, name in labels:
        r = rows[key]
        print(f"{name:<18}   {fmt([x['true'] for x in r])}   "
              f"{fmt([x['fake'] for x in r])}   {fmt([x['len'] for x in r])}   "
              f"{fmt([x['rm'] for x in r])}")
    print("\nPPO 的逐 Token log 比（3 种子平均）")
    for key, name in labels[1:4]:
        print(f"{name:<12} {fmt([x['kl'] for x in rows[key]])}")
    print(f"\n总耗时 {time.time() - t0:.1f} s")

if __name__ == "__main__":
    main()
