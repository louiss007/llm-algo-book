"""第 11 章：优化器、学习率与训练稳定性实验（examples/ch11_optim.py）。"""
import math
import time

import torch

import ch10
from ch08 import ModelConfig, TinyLM
from tinylm.train import optim as O


def make_model(vocab_size, seed):
    cfg = ModelConfig(vocab_size=vocab_size, d_model=32, n_layers=2, n_heads=4,
                      d_ff=64, max_seq_len=64, norm="rmsnorm", ffn="swiglu",
                      bias=False, pos_encoding="rope", attention="gqa",
                      n_kv_heads=2)
    torch.manual_seed(seed)
    return TinyLM(cfg)


def build_data():
    raw_docs, eval_docs = ch10.build_corpus()
    raw = [t for _, t in raw_docs]
    eval_texts = [t for _, t in eval_docs]
    stages, _ = ch10.pipeline(raw, eval_texts)
    docs = stages["decontam"]
    clean_val = ch10.build_clean_val()
    val_texts = [t for v in clean_val.values() for t in v]
    vocab = ch10.build_vocab(raw + eval_texts + val_texts)
    return docs, val_texts, vocab


def make_optimizer(name, model, lr, weight_decay):
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if name == "adam":
        # PyTorch 的 weight_decay 是耦合 L2，用于对照 (11.6)
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                weight_decay=weight_decay)
    if name == "adamw":
        groups = O.make_param_groups(model, weight_decay)
        return O.AdamW(groups, lr=lr, betas=(0.9, 0.95))
    raise ValueError(name)


def train_lm(docs, val_texts, vocab, name="adamw", lr=2e-3, steps=300,
             warmup=30, kind="cosine", weight_decay=0.01, seed=0,
             batch=16, seq=32):
    model = make_model(len(vocab), seed)
    opt = make_optimizer(name, model, lr, weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]
    stream = ch10.encode_stream(docs, vocab)
    gen = torch.Generator().manual_seed(seed)
    history = []
    model.train()
    for step in range(steps):
        factor = O.lr_factor(step, steps, warmup=warmup, kind=kind)
        for group, base in zip(opt.param_groups, base_lrs):
            group["lr"] = base * factor
        starts = torch.randint(0, len(stream) - seq - 1, (batch,), generator=gen)
        x = torch.stack([stream[s:s + seq] for s in starts])
        y = torch.stack([stream[s + 1:s + seq + 1] for s in starts])
        out = model(x, labels=y)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        history.append(out.loss.item())
    return model, history, ch10.evaluate(model, val_texts, vocab)


def mean_std(values):
    t = torch.tensor(values)
    return t.mean().item(), t.std().item()


def optimizer_table(docs, val_texts, vocab):
    runs = (("SGD + 动量", "sgd", 0.1, 0.0),
            ("AdamW（解耦）", "adamw", 2e-3, 0.01),
            ("Adam（L2 耦合）", "adam", 2e-3, 0.01))
    print("\n优化器对照（300 步，3 个种子，验证损失）")
    print(f"{'优化器':<18s}{'验证损失':>16s}{'最后 10 步训练损失':>20s}")
    for label, name, lr, wd in runs:
        vals, tails = [], []
        for seed in range(3):
            _, hist, val = train_lm(docs, val_texts, vocab, name, lr,
                                    weight_decay=wd, seed=seed)
            vals.append(val)
            tails.append(sum(hist[-10:]) / 10)
        m, s = mean_std(vals)
        print(f"{label:<18s}{m:>10.4f} ± {s:.4f}{sum(tails) / 3:>20.4f}")


def warmup_table(docs, val_texts, vocab):
    print("\n预热对照（峰值学习率 0.5，150 步，3 个种子）")
    print(f"{'预热步数':>8s}{'验证损失':>18s}{'训练损失峰值':>16s}")
    for warmup in (0, 40):
        vals, peaks = [], []
        for seed in range(3):
            _, hist, val = train_lm(docs, val_texts, vocab, "adamw", lr=0.5,
                                    steps=150, warmup=warmup, kind="constant",
                                    seed=seed)
            vals.append(val)
            peaks.append(max(hist))
        m, s = mean_std(vals)
        print(f"{warmup:>8d}{m:>12.4f} ± {s:.4f}{sum(peaks) / 3:>16.2f}")


def schedule_table(docs, val_texts, vocab):
    print("\n学习率调度（峰值 0.01，300 步，3 个种子）")
    print(f"{'调度':<12s}{'验证损失':>18s}{'训练损失峰值':>16s}")
    for kind in ("constant", "cosine", "wsd"):
        vals, peaks = [], []
        for seed in range(3):
            _, hist, val = train_lm(docs, val_texts, vocab, "adamw", lr=0.01,
                                    steps=300, warmup=30, kind=kind, seed=seed)
            vals.append(val)
            peaks.append(max(hist))
        m, s = mean_std(vals)
        print(f"{kind:<12s}{m:>12.4f} ± {s:.4f}{sum(peaks) / 3:>16.2f}")


def quadratic_demo():
    """公式 (11.1)–(11.4)：条件数为 1000 的二次型，比较 SGD 与 Adam。"""
    A = torch.tensor([1.0, 1000.0])
    print("\n病态二次型：SGD 与 Adam 的损失轨迹")
    print(f"{'优化器':<10s}{'步 1':>12s}{'步 10':>12s}{'步 50':>12s}{'步 100':>12s}")
    for name, lr in (("SGD", 0.0015), ("Adam", 0.1)):
        x = torch.tensor([1.0, 1.0], requires_grad=True)
        opt = (torch.optim.SGD([x], lr=lr) if name == "SGD"
               else torch.optim.Adam([x], lr=lr))
        marks = {}
        for step in range(100):
            loss = 0.5 * (A * x * x).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step + 1 in (1, 10, 50, 100):
                marks[step + 1] = loss.item()
        print(f"{name:<10s}" + "".join(f"{marks[k]:>12.4f}"
                                       for k in (1, 10, 50, 100)))


def clip_demo():
    """公式 (11.14)：在第 10 步把梯度放大 50 倍，比较是否裁剪。"""
    print("\n梯度裁剪：SGD 动量，第 10 步梯度 ×50")
    print(f"{'裁剪阈值':<10s}{'最大 |x|':>10s}{'尖峰后损失':>12s}{'最终损失':>10s}")
    for clip in (None, 1.0):
        x = torch.tensor([5.0, 5.0], requires_grad=True)
        opt = torch.optim.SGD([x], lr=0.05, momentum=0.9)
        history, max_abs = [], 0.0
        for step in range(50):
            loss = 0.5 * (x * x).sum()
            opt.zero_grad()
            loss.backward()
            if step == 10:
                x.grad.mul_(50)
            if clip is not None:
                O.clip_grad_norm_([x], clip)
            opt.step()
            history.append(loss.item())
            max_abs = max(max_abs, x.detach().abs().max().item())
        label = "无" if clip is None else f"{clip:.1f}"
        print(f"{label:<10s}{max_abs:>10.2f}{history[14]:>12.2f}{history[-1]:>10.4f}")


def main():
    start = time.time()
    docs, val_texts, vocab = build_data()
    print(f"语料 {len(docs)} 篇，验证集 {len(val_texts)} 篇，词表 {len(vocab)}")
    print("模型：2 层、d_model=32、4 头 GQA-2、SwiGLU、RMSNorm、RoPE")
    quadratic_demo()
    optimizer_table(docs, val_texts, vocab)
    warmup_table(docs, val_texts, vocab)
    schedule_table(docs, val_texts, vocab)
    clip_demo()
    print(f"\n总耗时 {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
