"""第 19 章：知识蒸馏与模型压缩（examples/ch19_distill.py）。"""
import time

import torch
import torch.nn.functional as F

from ch08 import ModelConfig, TinyLM
from tinylm.posttrain import distill as D

PAD, BOS, Q, SEP, EOS = 0, 1, 2, 3, 4
DIGIT0 = 5
PROMPT_LEN = 5
ALL_PROMPTS = [(a, b) for a in range(10) for b in range(10)]
LABELED = [p for i, p in enumerate(ALL_PROMPTS) if i % 5 < 2]
HELD_OUT = [p for i, p in enumerate(ALL_PROMPTS) if i % 5 >= 2]
SEEDS = (0, 1, 2)
PRUNE_FRACTION = 0.8

def prompt(a, b):
    return [BOS, Q, DIGIT0 + a, DIGIT0 + b, SEP]

def answer(a, b):
    return DIGIT0 + ((a + b) % 10)

def make_model(seed, d_model, n_layers, n_heads, d_ff):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=16, d_model=d_model, n_layers=n_layers,
                      n_heads=n_heads, d_ff=d_ff, max_seq_len=32,
                      pos_encoding="rope", norm="rmsnorm", ffn="swiglu",
                      bias=False)
    return TinyLM(cfg), cfg

def labels_for_answer(prompts, answers):
    ids = torch.tensor([prompt(a, b) + [y, EOS]
                        for (a, b), y in zip(prompts, answers)])
    labels = torch.full_like(ids, -100)
    labels[:, PROMPT_LEN - 1] = torch.tensor(
        [int(y) for y in answers])          # 预测答案
    labels[:, PROMPT_LEN] = EOS             # 预测 EOS
    return ids, labels

@torch.no_grad()
def accuracy(model, prompts):
    model.eval()
    ids = torch.tensor([prompt(a, b) for a, b in prompts])
    pred = model(ids).logits[:, -1].argmax(-1)
    goals = torch.tensor([answer(a, b) for a, b in prompts])
    return float((pred == goals).float().mean())

def train_teacher(seed, steps=600, batch=64, lr=2e-3):
    model, _ = make_model(seed, 64, 2, 4, 256)
    gen = torch.Generator().manual_seed(1000 + seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(ALL_PROMPTS), (batch,), generator=gen)
        prompts = [ALL_PROMPTS[int(i)] for i in idx]
        ids = torch.tensor([prompt(a, b) for a, b in prompts])
        labels = torch.full((len(prompts), PROMPT_LEN), -100, dtype=torch.long)
        labels[:, -1] = torch.tensor([answer(a, b) for a, b in prompts])
        out = model(ids, labels=labels)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    return model

def train_student(kind, teacher, seed, steps=800, batch=64, lr=1e-3):
    model, _ = make_model(seed, 32, 2, 4, 128)
    gen = torch.Generator().manual_seed(2000 + seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    cfg = D.DistillConfig(temperature=2.0, alpha=0.0)
    model.train()
    for _ in range(steps):
        pool = LABELED if kind == "sft" else ALL_PROMPTS
        idx = torch.randint(0, len(pool), (batch,), generator=gen)
        prompts = [pool[int(i)] for i in idx]
        prompt_ids = torch.tensor([prompt(a, b) for a, b in prompts])
        if kind == "sft":
            goals = [answer(a, b) for a, b in prompts]
            ids, labels = labels_for_answer(prompts, goals)
            logits = model(ids).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1), ignore_index=-100)
        elif kind == "seq":
            seqs, _ = D.generate_greedy(teacher, prompt_ids, cfg)
            labels = D.sft_labels(seqs, PROMPT_LEN)
            logits = model(seqs).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1), ignore_index=-100)
        else:
            teacher_logits = D.teacher_targets(teacher, prompt_ids)
            student_logits = model(prompt_ids).logits[:, -1]
            loss, _ = D.distill_loss(student_logits, teacher_logits,
                                     labels=None, cfg=cfg)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       cfg.max_grad_norm)
        opt.step()
    return model

def recover(teacher, model, masks, seed, steps=200, batch=64, lr=5e-4):
    gen = torch.Generator().manual_seed(3000 + seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    cfg = D.DistillConfig(temperature=2.0, alpha=0.0)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(ALL_PROMPTS), (batch,), generator=gen)
        prompts = [ALL_PROMPTS[int(i)] for i in idx]
        ids = torch.tensor([prompt(a, b) for a, b in prompts])
        teacher_logits = D.teacher_targets(teacher, ids)
        student_logits = model(ids).logits[:, -1]
        loss, _ = D.distill_loss(student_logits, teacher_logits,
                                 labels=None, cfg=cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        D.apply_masks_(masks)
    return model

@torch.no_grad()
def mean_entropy(teacher, temperature=1.0):
    ids = torch.tensor([prompt(a, b) for a, b in ALL_PROMPTS])
    logits = D.teacher_targets(teacher, ids) / temperature
    probs = torch.softmax(logits, dim=-1)
    return float(-(probs * probs.clamp_min(1e-12).log()).sum(-1).mean())

def main():
    t0 = time.time()
    rows = {k: ([], []) for k in ("SFT-40", "Seq-KD", "Logit-KD")}
    for seed in SEEDS:
        teacher = train_teacher(seed)
        params_t = D.parameter_count(teacher)
        print(f"seed {seed} teacher all {accuracy(teacher, ALL_PROMPTS):.2f} "
              f"entropy {mean_entropy(teacher):.2f} params {params_t}")
        results = {}
        for name, kind in (("SFT-40", "sft"), ("Seq-KD", "seq"),
                           ("Logit-KD", "kd")):
            student = train_student(kind, teacher, seed)
            acc_all = accuracy(student, ALL_PROMPTS)
            acc_held = accuracy(student, HELD_OUT)
            results[name] = student
            rows[name][0].append(acc_all)
            rows[name][1].append(acc_held)
            print(f"  {name:<9} all {acc_all:.2f} held {acc_held:.2f} "
                  f"params {D.parameter_count(student)}")

        student = results["Logit-KD"]
        _, _, masks = D.magnitude_prune_(student, PRUNE_FRACTION)
        before = accuracy(student, ALL_PROMPTS)
        sparsity = D.global_sparsity(student)
        recover(teacher, student, masks, seed)
        after = accuracy(student, ALL_PROMPTS)
        print(f"  prune {PRUNE_FRACTION:.0%} sparsity {sparsity:.2f} "
              f"acc {before:.2f} -> {after:.2f}")
        print(f"  compression {D.compression_ratio(teacher, student):.2f}x")

    print("\n3-seed means")
    print(f"{'':<10}{'all':>7}{'held':>7}")
    for name, (all_acc, held_acc) in rows.items():
        print(f"{name:<10}{sum(all_acc)/len(all_acc):>7.2f}"
              f"{sum(held_acc)/len(held_acc):>7.2f}")
    print(f"total {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
