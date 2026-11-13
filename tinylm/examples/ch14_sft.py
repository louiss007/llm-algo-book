"""第 14 章：SFT 的损失掩码、数据质量与拒绝采样（examples/ch14_sft.py）。"""
import random
import time

import torch
import torch.nn.functional as F

from ch08 import ModelConfig, TinyLM
from tinylm.posttrain import sft
from tinylm.tokenizer.chat import ChatTemplate, encode_conversation

PAD, BOS, USER, ASSISTANT, EOS, D0 = 0, 1, 2, 3, 4, 5
Q, A = 13, 14                     # OOD 模板使用的角色标记
VOCAB = 16
TEMPLATE = ChatTemplate()
SHIFTED = ChatTemplate(name="shifted", user_token=Q, assistant_token=A)
SEEDS = (0, 1, 2)
CHANCE = 1.0 / 8


def sample_turns(gen, length, noise=0.0, mode="random"):
    """生成一轮指令与回答：回答是输入的逐 Token 后继（模 8）。"""
    xs = [int(v) for v in torch.randint(0, 8, (length,), generator=gen)]
    ys = [(x + 1) % 8 for x in xs]
    flip = noise > 0 and torch.rand(1, generator=gen).item() < noise
    if flip and mode == "random":
        j = int(torch.randint(0, length, (1,), generator=gen).item())
        ys[j] = (ys[j] + 2) % 8
    elif flip and mode == "systematic":
        ys = [(x + 2) % 8 for x in xs]
    return [("user", [D0 + x for x in xs]),
            ("assistant", [D0 + y for y in ys])]


def make_dataset(seed, size, noise=0.0, mode="random", answer_only=True,
                 lo=3, hi=5):
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    data = []
    for _ in range(size):
        length = rng.randint(lo, hi)
        turns = sample_turns(gen, length, noise=noise, mode=mode)
        data.append(encode_conversation(turns, TEMPLATE,
                                        answer_only=answer_only))
    return data


def make_model(seed):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2,
                      d_ff=128, max_seq_len=32, pos_encoding="rope",
                      norm="rmsnorm", ffn="swiglu", bias=False)
    return TinyLM(cfg)


def train(data, seed, steps, lr=3e-3, batch_size=32):
    model = make_model(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
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
def evaluate(model, template, lo, hi, n=300):
    model.eval()
    gen = torch.Generator().manual_seed(999)
    rng = random.Random(999)
    exact = tokens = correct = 0
    for _ in range(n):
        length = rng.randint(lo, hi)
        r = encode_conversation(sample_turns(gen, length), template)
        out = model(r["input_ids"][None])
        keep = r["labels"] != -100
        pred = out.logits[0].argmax(-1)
        exact += int((pred[keep] == r["labels"][keep]).all())
        tokens += int(keep.sum())
        correct += int((pred[keep] == r["labels"][keep]).sum())
    return exact / n, correct / max(tokens, 1)


def masking_table():
    print("\n损失掩码对照（256 条干净数据，250 步，3 种子；随机水平 0.125）")
    print("OOD 两列为 Token 级命中率，便于观察部分泛化")
    print(f"{'掩码':<10s}{'ID 精确':>16s}{'ID Token':>16s}"
          f"{'长度 OOD':>16s}{'模板 OOD':>16s}")
    for answer_only in (True, False):
        rows = []
        for seed in SEEDS:
            data = make_dataset(10, 256, answer_only=answer_only)
            model = train(data, seed, steps=250)
            rows.append((*evaluate(model, TEMPLATE, 3, 5),
                         evaluate(model, TEMPLATE, 7, 8)[1],
                         evaluate(model, SHIFTED, 3, 5)[1]))
        t = torch.tensor(rows)
        mean, std = t.mean(0), t.std(0)
        name = "只算回答" if answer_only else "全序列"
        cells = "".join(f"{m:>10.3f}±{s:.3f}" for m, s in zip(mean[1:4], std[1:4]))
        print(f"{name:<10s}{mean[0]:>10.3f}±{std[0]:.3f}{cells}")


def quality_table():
    specs = (
        ("clean-64", 64, 0.0, "random"),
        ("clean-256", 256, 0.0, "random"),
        ("rand-50%-1024", 1024, 0.5, "random"),
        ("sys-50%-1024", 1024, 0.5, "systematic"),
        ("sys-25%-1024", 1024, 0.25, "systematic"),
    )
    print("\n数据质量与数量（600 步，3 种子；随机水平 0.125）")
    print("长度 OOD 为 Token 级命中率")
    print(f"{'数据':<16s}{'ID 精确':>16s}{'ID Token':>16s}{'长度 OOD':>16s}")
    keep_model = None
    for name, size, noise, mode in specs:
        rows = []
        for seed in SEEDS:
            data = make_dataset(20, size, noise=noise, mode=mode)
            model = train(data, seed, steps=600)
            rows.append((*evaluate(model, TEMPLATE, 3, 5),
                         evaluate(model, TEMPLATE, 7, 8)[1]))
            if name == "clean-64" and seed == 0:
                keep_model = model
        t = torch.tensor(rows)
        mean, std = t.mean(0), t.std(0)
        cells = "".join(f"{m:>10.3f}±{s:.3f}" for m, s in zip(mean[1:], std[1:]))
        print(f"{name:<16s}{mean[0]:>10.3f}±{std[0]:.3f}{cells}")
    return keep_model


def packing_demo():
    print("\n打包等价性（6 条对话，各自输出已做答案掩码）")
    model = make_model(0).eval()
    gen = torch.Generator().manual_seed(0)
    examples = []
    for length in (3, 4, 5, 3, 4, 5):
        examples.append(encode_conversation(sample_turns(gen, length)))

    logits_ref, loss_ref, sup_ref = [], 0.0, 0
    with torch.no_grad():
        for e in examples:
            out = model(e["input_ids"][None],
                        position_ids=torch.arange(e["input_ids"].numel())[None])
            per = F.cross_entropy(out.logits[0], e["labels"],
                                  ignore_index=-100, reduction="none")
            keep = e["labels"] != -100
            logits_ref.append(out.logits[0][keep])
            loss_ref = loss_ref + per[keep].sum()
            sup_ref += int(keep.sum())
        packed = sft.pack_sft_examples(examples)
        out = model(packed.input_ids[None],
                    attention_mask=packed.attention_mask,
                    position_ids=packed.position_ids[None])
        per = F.cross_entropy(out.logits[0], packed.labels,
                              ignore_index=-100, reduction="none")
        keep = packed.labels != -100
        diff = (out.logits[0][keep] - torch.cat(logits_ref)).abs().max()
    stats = sft.supervision_stats(examples)
    print(f"逐条前向  损失 {loss_ref.item():.4f}  监督 Token {sup_ref}")
    print(f"打包前向  损失 {per[keep].sum().item():.4f}  监督 Token "
          f"{int(keep.sum())}")
    print(f"logits 最大偏差 {diff.item():.1e}；"
          f"监督比例 {stats['ratio']:.3f}，平均回答长度 {stats['mean_answer']:.1f}")


@torch.no_grad()
def sample_answer(model, prompt, *, max_new=8, temperature=1.0, gen=None):
    ids = list(prompt)
    out = []
    for _ in range(max_new):
        logits = model(torch.tensor(ids)[None]).logits[0, -1]
        probs = torch.softmax(logits / temperature, dim=-1)
        tok = int(torch.multinomial(probs, 1, generator=gen).item())
        ids.append(tok)
        out.append(tok)
        if tok == EOS:
            break
    return out


def rejection_demo(model):
    print("\n拒绝采样（弱模型，64 个提示，每个采 8 条，温度 1.0）")
    gen = torch.Generator().manual_seed(7)
    rng = random.Random(7)
    prompts, targets = [], []
    for _ in range(64):
        length = rng.randint(3, 5)
        turns = sample_turns(gen, length)
        xs = turns[0][1]
        ys = turns[1][1]
        prompts.append([BOS, USER] + xs + [ASSISTANT])
        targets.append(ys)

    accepted = draws = pass8 = 0
    unique = set()
    for prompt, target in zip(prompts, targets):
        hits = 0
        for _ in range(8):
            toks = sample_answer(model, prompt, gen=gen)
            if toks and toks[-1] == EOS:
                toks = toks[:-1]
            draws += 1
            if toks == target:
                hits += 1
                unique.add(tuple(toks))
        accepted += hits
        pass8 += int(hits > 0)
    p_hat = accepted / draws
    print(f"接受率 {p_hat:.3f}；经验 pass@8 {pass8/len(prompts):.3f}；"
          f"理论 1-(1-p)^8 {sft.pass_at_n(p_hat, 8):.3f}")
    print(f"接受 {accepted} 条，去重后 {len(unique)} 条；"
          f"凑够 64 条期望采样 {sft.expected_draws(p_hat, 64):.0f} 条")


def main():
    t0 = time.time()
    masking_table()
    weak = quality_table()
    packing_demo()
    if weak is not None:
        rejection_demo(weak)
    print(f"\n总耗时 {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
