"""第 20 章：解码策略（examples/ch20_decode.py）。"""
import random
import time

import torch

from ch08 import ModelConfig, TinyLM
from tinylm.inference import sample as S

PAD, BOS, EOS, A, B, C, D = 7, 0, 1, 2, 3, 4, 5
VOCAB = 8
MAX_NEW = 16
TRANSITIONS = {
    BOS: [(A, 0.5), (C, 0.5)],
    A: [(B, 0.9), (EOS, 0.05), (C, 0.05)],
    B: [(A, 0.9), (EOS, 0.05), (D, 0.05)],
    C: [(D, 0.9), (EOS, 0.05), (A, 0.05)],
    D: [(C, 0.9), (EOS, 0.05), (B, 0.05)],
}
SEEDS = (0, 1, 2)
N_SAMPLES = 128

def make_sequence(rng):
    seq = [BOS]
    current = BOS
    for _ in range(20):
        choices = TRANSITIONS[current]
        current = rng.choices([t for t, _ in choices],
                              [p for _, p in choices])[0]
        seq.append(current)
        if current == EOS:
            break
    else:
        seq.append(EOS)
    return seq

def make_batch(seqs, max_len=None):
    max_len = max_len or max(len(s) for s in seqs)
    ids = torch.full((len(seqs), max_len), PAD, dtype=torch.long)
    labels = torch.full((len(seqs), max_len), -100, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, seq in enumerate(seqs):
        ids[i, :len(seq)] = torch.tensor(seq)
        labels[i, :len(seq) - 1] = torch.tensor(seq[1:])
        mask[i, :len(seq)] = True
    return ids, labels, mask

def train_model(seed, steps=700, batch=64, lr=3e-3):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=4,
                      d_ff=128, max_seq_len=24, pos_encoding="rope",
                      norm="rmsnorm", ffn="swiglu", bias=False)
    model = TinyLM(cfg)
    rng = random.Random(100 + seed)
    data = [make_sequence(rng) for _ in range(4000)]
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(data), (batch,), generator=gen)
        ids, labels, mask = make_batch([data[int(i)] for i in idx])
        out = model(ids, attention_mask=mask, labels=labels)
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model

def truncate(seq):
    ids = seq.tolist()
    if EOS in ids[1:]:
        return ids[:ids.index(EOS, 1) + 1]
    return ids

def metrics(seqs, prompt_len=1):
    cut = [truncate(seq) for seq in seqs]
    lengths = [len(s) - prompt_len for s in cut]
    eos = sum(s[-1] == EOS for s in cut) / len(cut)
    grams, total = set(), 0
    for s in cut:
        body = s[prompt_len:]
        for i in range(len(body) - 1):
            grams.add((body[i], body[i + 1]))
            total += 1
    distinct2 = len(grams) / total if total else 0.0
    unique = len({tuple(s) for s in cut}) / len(cut)
    return eos, sum(lengths) / len(lengths), distinct2, unique

def report(name, seqs, model, prompt_len=1):
    eos, length, distinct2, unique = metrics(seqs)
    logp = S.sequence_logprob(model, seqs, prompt_len, PAD).mean().item()
    print(f"  {name:<16} eos {eos:.2f} len {length:5.2f} "
          f"distinct2 {distinct2:.2f} unique {unique:.2f} logp {logp:6.2f}")
    return eos, length, distinct2, unique, logp

def main():
    t0 = time.time()
    prompt = torch.tensor([[BOS]])
    rows = {}
    for seed in SEEDS:
        model = train_model(seed)
        print(f"seed {seed}")
        b = torch.full((N_SAMPLES, 1), BOS)
        greedy = torch.cat([S.greedy_decode(model, prompt, MAX_NEW, EOS, PAD)[0]]
                           * N_SAMPLES, dim=0)
        beam = S.beam_search(model, prompt, 4, MAX_NEW, EOS, pad_id=PAD)
        beam = beam.repeat(N_SAMPLES, 1)
        beam_norm = S.beam_search(model, prompt, 4, MAX_NEW, EOS,
                                  length_penalty=1.0, pad_id=PAD)
        beam_norm = beam_norm.repeat(N_SAMPLES, 1)
        pure, _ = S.sample_decode(model, b, MAX_NEW, S.SamplingConfig(seed=seed),
                                  EOS, PAD)
        top_p, _ = S.sample_decode(
            model, b, MAX_NEW,
            S.SamplingConfig(temperature=0.8, top_p=0.9, seed=seed), EOS, PAD)
        top_p95, _ = S.sample_decode(
            model, b, MAX_NEW,
            S.SamplingConfig(temperature=0.8, top_p=0.95, seed=seed), EOS, PAD)
        min_p, _ = S.sample_decode(
            model, b, MAX_NEW,
            S.SamplingConfig(temperature=1.0, min_p=0.05, seed=seed), EOS, PAD)
        strategies = {
            "greedy": greedy, "beam w=4": beam, "beam w=4 norm": beam_norm,
            "pure T=1": pure, "T=0.8 top_p=.9": top_p,
            "T=0.8 top_p=.95": top_p95, "T=1.0 min_p=.05": min_p,
        }
        for name, seqs in strategies.items():
            vals = report(name, seqs, model)
            rows.setdefault(name, []).append(vals)
        for name, seqs in strategies.items():
            cut = [truncate(s) for s in seqs]
            c = sum(s[-1] == EOS for s in cut)
            p1 = S.pass_at_k(len(cut), c, 1)
            p8 = S.pass_at_k(len(cut), c, 8)
            rows.setdefault(name + " pass", []).append((p1, p8))
        print()

    print("3-seed means")
    print(f"{'':<17}{'eos':>6}{'len':>7}{'dist2':>7}{'uniq':>7}{'logp':>7}"
          f"{'pass@1':>8}{'pass@8':>8}")
    for name in ("greedy", "beam w=4", "beam w=4 norm", "pure T=1",
                 "T=0.8 top_p=.9", "T=0.8 top_p=.95", "T=1.0 min_p=.05"):
        vals = rows[name]
        ps = rows[name + " pass"]
        eos = sum(v[0] for v in vals) / len(vals)
        length = sum(v[1] for v in vals) / len(vals)
        dist2 = sum(v[2] for v in vals) / len(vals)
        uniq = sum(v[3] for v in vals) / len(vals)
        logp = sum(v[4] for v in vals) / len(vals)
        p1 = sum(v[0] for v in ps) / len(ps)
        p8 = sum(v[1] for v in ps) / len(ps)
        print(f"{name:<17}{eos:6.2f}{length:7.2f}{dist2:7.2f}{uniq:7.2f}"
              f"{logp:7.2f}{p1:8.2f}{p8:8.2f}")
    print(f"total {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
