"""第 28 章：可解释性（examples/ch28_interpret.py）。"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from ch08 import ModelConfig, TinyLM
from tinylm.interpret.core import (ablate_heads, ablated_direction,
                                   activation_at, attention_maps,
                                   feature_purity, linear_probe_accuracy,
                                   logit_margin, patched_output,
                                   probe_selectivity, recovery, sae_metrics,
                                   train_sae)

V, T = 16, 11
BOS, SEP, LO, HI = 0, 1, 4, 16
N_CLASSES = HI - LO

def make_batch(n, gen):
    ids = torch.randint(LO, HI, (n, T), generator=gen)
    ids[:, 0] = BOS
    ids[:, 9] = SEP
    answer = ids[:, 1].clone()
    labels = torch.full((n, T), -100, dtype=torch.long)
    labels[:, 10] = answer
    return ids, labels, answer

def train_model(steps=200):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    cfg = ModelConfig(vocab_size=V, d_model=64, n_layers=2, n_heads=4,
                      d_ff=128, max_seq_len=T)
    model = TinyLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    for _ in range(steps):
        ids, labels, _ = make_batch(128, gen)
        out = model(ids, labels=labels)
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return model

def predict(model, ids):
    with torch.no_grad():
        return model(ids).logits[:, -1].argmax(dim=-1)

def probe_demo(model, ids, answer):
    print("线性探针（层 1，训练 800 / 测试 200）")
    train, test = slice(0, 800), slice(800, 1000)
    for pos in (1, 2, 5):
        hidden = activation_at(model, model.blocks[1], pos, ids)
        acc = linear_probe_accuracy(
            hidden[train], ids[train, pos] - LO,
            hidden[test], ids[test, pos] - LO, N_CLASSES)
        print(f"  位置 {pos:>2} 的 Token：准确率 {acc:.3f}")
    hidden = activation_at(model, model.blocks[1], 10, ids)
    perm = torch.randperm(1000, generator=torch.Generator().manual_seed(3))
    control = linear_probe_accuracy(
        hidden[train], (answer[perm] - LO)[train],
        hidden[test], (answer[perm] - LO)[test], N_CLASSES)
    print(f"  打乱标签对照（位置 10）：{control:.3f}，"
          f"选择性 {probe_selectivity(1.0, control):+.3f}")

def causal_demo(model, ids, answer):
    gen = torch.Generator().manual_seed(7)
    other = torch.randint(LO, HI, (1000,), generator=gen)
    ids_filler = ids.clone()
    ids_filler[:, 5] = other
    keep_old = float((predict(model, ids_filler) == answer).float().mean())
    ids_answer = ids.clone()
    ids_answer[:, 1] = other
    follow_new = float((predict(model, ids_answer) == other).float().mean())
    print("只改输入的一个位置，观察输出是否跟随")
    print(f"  改位置 5（填充词）：仍复制原答案 {keep_old:.3f}")
    print(f"  改位置 1（被复制词）：改为复制新词 {follow_new:.3f}")

def attention_demo(model, ids, answer):
    print("最后一层位置对位置 1 的平均注意力")
    for layer in range(2):
        attn = attention_maps(model, ids, layer)
        weights = attn[:, :, -1, 1].mean(dim=0)
        shown = ", ".join(f"{float(w):.3f}" for w in weights)
        print(f"  层 {layer}：{shown}")
    with torch.no_grad():
        base = logit_margin(model(ids).logits[:, -1], answer)
    print("单头消融后的平均 logit margin 变化")
    for layer in range(2):
        drops = []
        for head in range(4):
            with ablate_heads(model, layer, [head]):
                with torch.no_grad():
                    lo = model(ids).logits[:, -1]
            drops.append(float(logit_margin(lo, answer) - base))
        shown = ", ".join(f"{d:+.3f}" for d in drops)
        print(f"  层 {layer}：{shown}")
    with ablate_heads(model, 0, [0, 1, 2, 3]):
        with torch.no_grad():
            lo = model(ids).logits[:, -1]
    acc = float((lo.argmax(-1) == answer).float().mean())
    print(f"  层 0 四头同时消融：准确率 {acc:.3f}，"
          f"margin {float(logit_margin(lo, answer)):+.3f}")

def patch_demo(model, ids, answer):
    gen = torch.Generator().manual_seed(11)
    corrupt = ids.clone()
    corrupt[:, 1] = torch.randint(LO, HI, (1000,), generator=gen)
    with torch.no_grad():
        clean_lo = model(ids).logits[:, -1]
        bad_lo = model(corrupt).logits[:, -1]
    clean = float(logit_margin(clean_lo, answer))
    bad = float(logit_margin(bad_lo, answer))
    plan = [
        ("tok_emb", model.tok_emb, 1),
        ("attn[0]", model.blocks[0].attn, 10),
        ("attn[0]", model.blocks[0].attn, 1),
        ("block[0]", model.blocks[0], 1),
        ("block[0]", model.blocks[0], 2),
        ("block[0]", model.blocks[0], 10),
        ("block[1]", model.blocks[1], 10),
    ]
    print("激活修补：把干净激活写回损坏输入，恢复率")
    for name, module, pos in plan:
        src = activation_at(model, module, pos, ids)
        with patched_output(module, pos, src):
            with torch.no_grad():
                lo = model(corrupt).logits[:, -1]
        rec = recovery(float(logit_margin(lo, answer)), clean, bad)
        print(f"  {name:<9} @ pos {pos:>2}：{rec:+.3f}")

def collect(model, position=10, n=4000, seed=123):
    gen = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for _ in range(n // 500):
        ids, _, answer = make_batch(500, gen)
        xs.append(activation_at(model, model.blocks[1], position, ids))
        ys.append(answer)
    return torch.cat(xs), torch.cat(ys)

def sae_demo(model):
    X, answer = collect(model)
    print("SAE 的稀疏—重构权衡（4000 条末位激活，字典 128）")
    print(f"{'λ':>6} {'MSE':>10} {'L0':>7} {'dead':>5}")
    best = None
    for lam in (0.0, 0.005, 0.02, 0.05):
        sae = train_sae(X, dict_size=128, l1=lam, steps=1200, seed=0)
        metrics = sae_metrics(sae, X)
        print(f"{lam:>6.3f} {metrics['mse']:>10.2e} {metrics['l0']:>7.2f} "
              f"{metrics['dead']:>5d}")
        if lam == 0.02:
            best = (sae, metrics)
    sae, _ = best
    purity, _ = feature_purity(sae, X, answer - LO, N_CLASSES)
    print(f"λ=0.020 的特征纯度（按最大激活样本的答案 Token）：{purity:.3f}")

    with torch.no_grad():
        mean_act = sae(X)[0][answer == 4].mean(dim=0)
    feature = int(mean_act.argmax())
    direction = sae.decoder.weight[:, feature].detach()
    target = torch.full((1000,), 4, dtype=torch.long)
    ids, _, _ = make_batch(1000, torch.Generator().manual_seed(99))
    with torch.no_grad():
        base = float(logit_margin(model(ids).logits[:, -1], target))
    with ablated_direction(model.ln_f, 10, direction):
        with torch.no_grad():
            lo = model(ids).logits[:, -1]
    ablated = float(logit_margin(lo, target))
    acc = float((lo.argmax(-1) == ids[:, 1]).float().mean())
    rand = torch.randn(64, generator=torch.Generator().manual_seed(5))
    with ablated_direction(model.ln_f, 10, rand):
        with torch.no_grad():
            lo_rand = model(ids).logits[:, -1]
    random_margin = float(logit_margin(lo_rand, target))
    acc_rand = float((lo_rand.argmax(-1) == ids[:, 1]).float().mean())
    print(f"Token 4 的最强特征 #{feature}：平均激活 {float(mean_act[feature]):.3f}")
    print(f"  投影掉后 margin {ablated:+.3f}（原 {base:+.3f}，随机方向 {random_margin:+.3f}）")
    print(f"  准确率 {acc:.3f}（随机方向 {acc_rand:.3f}）")

def main():
    model = train_model()
    ids, _, answer = make_batch(1000, torch.Generator().manual_seed(99))
    acc = float((predict(model, ids) == answer).float().mean())
    print(f"复制任务：1000 条留出样本，准确率 {acc:.3f}")
    print()
    probe_demo(model, ids, answer)
    print()
    causal_demo(model, ids, answer)
    print()
    attention_demo(model, ids, answer)
    print()
    patch_demo(model, ids, answer)
    print()
    sae_demo(model)

if __name__ == "__main__":
    main()
