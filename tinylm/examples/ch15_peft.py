"""第 15 章：LoRA、QLoRA 与参数账本（examples/ch15_peft.py）。"""
import copy
import random
import time

import torch

from ch08 import ModelConfig, TinyLM
from tinylm.posttrain import lora as L
from tinylm.posttrain import sft
from tinylm.tokenizer.chat import ChatTemplate, encode_conversation

D0 = 5
VOCAB = 16
TEMPLATE = ChatTemplate()
SEEDS = (0, 1, 2, 3, 4)

def plus2(x):
    return (x + 2) % 8

def plus1(x):
    return (x + 1) % 8

def make_turns(gen, length, rule):
    xs = [int(v) for v in torch.randint(0, 8, (length,), generator=gen)]
    ys = [rule(x) for x in xs]
    return [("user", [D0 + x for x in xs]),
            ("assistant", [D0 + y for y in ys])]

def make_data(seed, size, rule, answer_only, lo=3, hi=5):
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    data = []
    for _ in range(size):
        length = rng.randint(lo, hi)
        turns = make_turns(gen, length, rule)
        data.append(encode_conversation(turns, TEMPLATE,
                                        answer_only=answer_only))
    return data

def make_model(seed):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2,
                      d_ff=128, max_seq_len=32, pos_encoding="rope",
                      norm="rmsnorm", ffn="swiglu", bias=False)
    return TinyLM(cfg)

def train(model, data, seed, steps, lr, batch_size=32):
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

@torch.no_grad()
def evaluate(model, rule, n=200):
    model.eval()
    gen = torch.Generator().manual_seed(999)
    rng = random.Random(999)
    exact = tokens = correct = 0
    for _ in range(n):
        length = rng.randint(3, 5)
        r = encode_conversation(make_turns(gen, length, rule), TEMPLATE)
        out = model(r["input_ids"][None])
        keep = r["labels"] != -100
        pred = out.logits[0].argmax(-1)
        exact += int((pred[keep] == r["labels"][keep]).all())
        tokens += int(keep.sum())
        correct += int((pred[keep] == r["labels"][keep]).sum())
    return exact / n, correct / max(tokens, 1)

def build_bases(pretrain):
    bases = {}
    zero = []
    for seed in SEEDS:
        model = train(make_model(seed), pretrain, seed, 1200, 3e-3)
        bases[seed] = copy.deepcopy(model.state_dict())
        zero.append(evaluate(model, plus1))
    t = torch.tensor(zero)
    print(f"零样本基座（预训练学 +2，不微调）：ID 精确 {t[:, 0].mean():.3f}"
          f"±{t[:, 0].std():.3f}，Token {t[:, 1].mean():.3f}±{t[:, 1].std():.3f}")
    return bases

def run_one(bases, sft_data, seed, spec):
    kind, cfg, lr = spec
    model = make_model(seed)
    if bases is not None:
        model.load_state_dict(bases[seed])
    if cfg is not None:
        L.apply_lora(model, cfg)
        if kind == "qlora":
            L.quantize_lora_bases(model)
    train(model, sft_data, seed, 600, lr)
    return model

def method_table(bases, sft_data):
    specs = [
        ("全参数", ("full", None, 3e-3), None),
        ("LoRA q/v r=1", ("lora", L.LoraConfig(r=1, alpha=2,
         target_modules=("q_proj", "v_proj")), 1e-2), None),
        ("LoRA q/v r=2", ("lora", L.LoraConfig(r=2, alpha=4,
         target_modules=("q_proj", "v_proj")), 1e-2), None),
        ("LoRA q/v r=4", ("lora", L.LoraConfig(r=4, alpha=8,
         target_modules=("q_proj", "v_proj")), 1e-2), None),
        ("LoRA q/v r=8", ("lora", L.LoraConfig(r=8, alpha=16,
         target_modules=("q_proj", "v_proj")), 1e-2), None),
        ("LoRA 全线性 r=4", ("lora", L.LoraConfig(r=4, alpha=8), 1e-2), None),
        ("QLoRA 全线性 r=4", ("qlora", L.LoraConfig(r=4, alpha=8), 1e-2), None),
        ("随机基座+全参数", ("full", None, 3e-3), "random"),
        ("随机基座+LoRA r=4", ("lora", L.LoraConfig(r=4, alpha=8), 1e-2),
         "random"),
    ]
    print(f"\n{'方法':<18s}{'可训练':>8s}{'比例':>7s}{'状态KB':>8s}"
          f"{'ID 精确':>16s}{'ID Token':>16s}")
    for name, spec, base_kind in specs:
        rows = []
        for seed in SEEDS:
            used = None if base_kind == "random" else bases
            model = run_one(used, sft_data, seed, spec)
            rows.append(evaluate(model, plus1))
        t = torch.tensor(rows)
        stats = L.trainable_stats(model)
        mem = L.state_bytes(model)["total_bytes"]
        print(f"{name:<18s}{stats['trainable']:>8d}{stats['ratio']:>7.3f}"
              f"{mem/1024:>8.0f}{t[:, 0].mean():>10.3f}±{t[:, 0].std():.3f}"
              f"{t[:, 1].mean():>10.3f}±{t[:, 1].std():.3f}")

def saved_activation_bytes(model, batch):
    seen, total = set(), [0]

    def pack(t):
        if t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            total[0] += t.numel() * t.element_size()
        return t

    def unpack(t):
        return t

    model.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        out = model(batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        out.loss.backward()
    return total[0]

def memory_demo(bases, sft_data):
    gen = torch.Generator().manual_seed(77)
    idx = torch.randint(0, len(sft_data), (32,), generator=gen)
    batch = sft.collate_examples([sft_data[int(i)] for i in idx])
    specs = (
        ("全参数", None, False),
        ("LoRA q/v r=4", L.LoraConfig(r=4, alpha=8,
         target_modules=("q_proj", "v_proj")), False),
        ("LoRA 全线性 r=4", L.LoraConfig(r=4, alpha=8), False),
        ("QLoRA 全线性 r=4", L.LoraConfig(r=4, alpha=8), True),
    )
    print(f"\n静态状态与单步激活（32 条、长度 3–5）")
    print(f"{'方法':<18s}{'静态KB':>9s}{'激活KB':>9s}")
    for name, cfg, quant in specs:
        model = make_model(0)
        model.load_state_dict(bases[0])
        if cfg is not None:
            L.apply_lora(model, cfg)
            if quant:
                L.quantize_lora_bases(model)
        static = L.state_bytes(model)["total_bytes"]
        active = saved_activation_bytes(model, batch)
        print(f"{name:<18s}{static/1024:>9.0f}{active/1024:>9.0f}")

def merge_demo(bases, sft_data):
    model = make_model(0)
    model.load_state_dict(bases[0])
    L.apply_lora(model, L.LoraConfig(r=4, alpha=8))
    train(model, sft_data, 0, 200, 1e-2)
    model.eval()
    gen = torch.Generator().manual_seed(31)
    x = encode_conversation(make_turns(gen, 5, plus1), TEMPLATE)["input_ids"]
    x = x[None]
    before = model(x).logits
    modules = [m for m in model.modules() if isinstance(m, L.LoRALinear)]
    for m in modules:
        m.merge()
    after = model(x).logits
    for m in modules:
        m.unmerge()
    restored = model(x).logits
    print(f"\n合并前后 logits 最大偏差 {(before-after).abs().max():.1e}；"
          f"撤销合并后 {(before-restored).abs().max():.1e}")
    print(f"被替换的线性层 {len(modules)} 个，适配器参数 "
          f"{sum(p.numel() for _, p in L.lora_parameters(model))}")

def qlora_demo(bases):
    model = make_model(0)
    model.load_state_dict(bases[0])
    cfg = L.LoraConfig(r=4, alpha=8)
    L.apply_lora(model, cfg)
    model.eval()
    gen = torch.Generator().manual_seed(41)
    x = encode_conversation(make_turns(gen, 5, plus1), TEMPLATE)["input_ids"]
    x = x[None]
    before = model(x).logits
    numer, denom = 0.0, 0.0
    for m in model.modules():
        if isinstance(m, L.LoRALinear):
            w = m.base.weight.detach()
            deq, _, _ = L.nf4_quantize(w)
            numer += (w - deq).pow(2).sum().item()
            denom += w.pow(2).sum().item()
    info = L.quantize_lora_bases(model)
    quant = model(x).logits
    print(f"\nNF4 量化：码本 16 级，块大小 64；基座权重相对误差 "
          f"{(numer/denom) ** 0.5:.4f}")
    print(f"量化后 logits 相对偏差 "
          f"{(quant-before).norm().item()/before.norm().item():.4f}；"
          f"存储 {info['bits_per_param']:.3f} 比特/参数")

def main():
    t0 = time.time()
    pretrain = make_data(100, 1024, plus2, answer_only=False)
    sft_data = make_data(10, 256, plus1, answer_only=True)
    bases = build_bases(pretrain)
    method_table(bases, sft_data)
    memory_demo(bases, sft_data)
    merge_demo(bases, sft_data)
    qlora_demo(bases)
    print(f"\n总耗时 {time.time() - t0:.1f} s")

if __name__ == "__main__":
    main()
