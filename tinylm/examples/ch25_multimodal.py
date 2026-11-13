"""第 25 章：多模态大模型（examples/ch25_multimodal.py）。"""
import math
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from ch08 import ModelConfig, TinyLM
from tinylm.multimodal import (MultimodalTinyLM, VisionConfig, attention_multiplier,
                               contrastive_loss, image_token_count,
                               vq_loss, vq_quantize)

PAD, BOS, IMG = 0, 1, 2
Q_SHAPE, Q_COLOR = 3, 4
ANS0 = 5
VOCAB = 11
PALETTE = torch.tensor([[1.0, 0.15, 0.15],
                        [0.15, 1.0, 0.15],
                        [0.15, 0.15, 1.0]])

def render(shape, color, center, gen):
    n = shape.shape[0]
    yy, xx = torch.meshgrid(torch.arange(12), torch.arange(12), indexing="ij")
    x = xx.float().view(1, 12, 12)
    y = yy.float().view(1, 12, 12)
    cx = center[:, 0].float().view(-1, 1, 1)
    cy = center[:, 1].float().view(-1, 1, 1)
    r = 2.6
    circle = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    square = (x - cx).abs().le(r * 0.8) & (y - cy).abs().le(r * 0.8)
    tri = (y >= cy - r) & (y <= cy + r) & \
          ((x - cx).abs() <= (y - (cy - r)) * 0.55)
    s = shape.view(-1, 1, 1)
    mask = torch.where(s == 0, circle, torch.where(s == 1, square, tri))
    img = 0.15 + 0.04 * torch.randn(n, 1, 12, 12, generator=gen)
    img = img + mask.float().unsqueeze(1) * PALETTE[color].view(-1, 3, 1, 1)
    return img.clamp(0, 1)

def make_data(n, seed, max_img_tokens):
    g = torch.Generator().manual_seed(seed)
    shape = torch.randint(0, 3, (n,), generator=g)
    color = torch.randint(0, 3, (n,), generator=g)
    q = torch.randint(0, 2, (n,), generator=g)
    center = torch.randint(4, 9, (n, 2), generator=g)
    images = render(shape, color, center, g)
    answers = torch.where(q == 0, shape, 3 + color) + ANS0
    T = 2 + max_img_tokens
    ids = torch.empty(n, T, dtype=torch.long)
    ids[:, 0] = BOS
    ids[:, 1] = torch.where(q == 0, Q_SHAPE, Q_COLOR)
    ids[:, 2:] = IMG
    labels = torch.full((n, T), -100, dtype=torch.long)
    labels[:, -1] = answers
    return ids, images, labels

def build(patch, projector, frozen, seed=0):
    torch.manual_seed(seed)
    vcfg = VisionConfig(image_size=12, patch_size=patch, projector=projector)
    lm = TinyLM(ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2,
                            n_heads=4, d_ff=64, max_seq_len=64))
    model = MultimodalTinyLM(lm, vcfg, image_token_id=IMG)
    if frozen:
        for p in model.vision.parameters():
            p.requires_grad_(False)
    return model

def evaluate(model, ids, images, labels):
    model.eval()
    with torch.no_grad():
        logits = model(ids, images).logits[:, -1]
        pred = logits.argmax(-1)
        return (pred == labels[:, -1]).float().mean().item()

def train_variant(name, patch, projector, frozen=False, steps=300, seeds=(0, 1, 2)):
    n_img = image_token_count(12, patch)
    n_train = 4096
    accs = []
    t0 = time.time()
    for seed in seeds:
        ids, images, labels = make_data(n_train, 100 + seed, n_img)
        tids, timages, tlabels = make_data(800, 999, n_img)
        model = build(patch, projector, frozen, seed)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=0.01)
        model.train()
        for step in range(steps):
            idx = torch.randint(0, n_train, (64,))
            out = model(ids[idx], images[idx], labels=labels[idx])
            opt.zero_grad()
            out.loss.backward()
            opt.step()
        accs.append(evaluate(model, tids, timages, tlabels))
    n_par = sum(p.numel() for p in model.parameters())
    acc = torch.tensor(accs)
    return dict(name=name, n_img=n_img, acc=acc.mean().item(), std=acc.std().item(),
                params=n_par, secs=time.time() - t0)

def token_budget_demo():
    rows = [(336, 14, 1), (336, 14, 2), (448, 14, 1), (448, 14, 2),
            (896, 14, 2), (12, 3, 1)]
    print("图像 Token 预算：N_p = (H/p)(W/p) / m^2，文本按 128 Token 计")
    print(f"{'分辨率':>10} {'切块':>5} {'合并':>4} {'图像Token':>9} "
          f"{'序列长':>7} {'注意力倍数':>9}")
    for size, patch, merge in rows:
        n = image_token_count(size, patch, merge)
        print(f"{size:>6}x{size:<3} {patch:>5} {merge:>4}x{merge:<2} "
              f"{n:>9} {n + 128:>7} {attention_multiplier(n, 128):>9.2f}")

def loss_demo():
    g = torch.Generator().manual_seed(0)
    text = torch.randn(64, 32, generator=g)
    image = 0.8 * text + 0.2 * torch.randn(64, 32, generator=g)
    perm = torch.randperm(64, generator=g)
    rand = torch.randn(64, 32, generator=g)
    print("对称 InfoNCE（batch=64）")
    print(f"  独立随机配对（τ=1.0）：{contrastive_loss(image, rand, 1.0):.3f}")
    print(f"  对齐配对（τ=0.07）：{contrastive_loss(image, text):.3f}")
    print(f"  错位配对（τ=0.07）：{contrastive_loss(image, text[perm]):.3f}")
    print(f"  随机水平 ln(64)：{math.log(64):.3f}")

def kmeans_codebook(z, k, iters=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    cb = z[torch.randperm(z.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        _, idx = vq_quantize(z, cb)
        for j in range(k):
            members = z[idx == j]
            if members.numel():
                cb[j] = members.mean(0)
    zq, _ = vq_quantize(z, cb)
    return F.mse_loss(zq, z).item()

def vq_demo():
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(16, 8, generator=g) * 2
    z = centers[torch.randint(0, 16, (2048,), generator=g)]
    z = z + 0.3 * torch.randn(2048, 8, generator=g)
    print("VQ 码本大小与重建误差（2048 个 8 维特征，16 个真实簇）")
    print(f"{'码本大小':>8} {'MSE':>8}")
    for k in (4, 8, 16, 32):
        print(f"{k:>8} {kmeans_codebook(z, k, seed=k):>8.4f}")
    _, idx = vq_quantize(z[:4], torch.randn(16, 8, generator=torch.Generator().manual_seed(1)))
    print(f"  码字范围检查：min={idx.min().item()}, max={idx.max().item()}")

def main():
    torch.set_num_threads(1)
    token_budget_demo()
    print()
    loss_demo()
    print()
    vq_demo()
    print()
    runs = [
        ("线性投影", 3, "linear", False),
        ("MLP 投影", 3, "mlp", False),
        ("MLP + 大块（4 Token）", 6, "mlp", False),
        ("MLP + 小块（36 Token）", 2, "mlp", False),
        ("MLP + 冻结视觉塔", 3, "mlp", True),
    ]
    print("合成视觉问答：3 种形状 × 3 种颜色，问形状或颜色")
    print(f"{'配置':<24} {'图像Token':>8} {'参数量':>10} {'测试准确率（3 种子）':>18} {'训练秒':>7}")
    for name, patch, proj, frozen in runs:
        r = train_variant(name, patch, proj, frozen)
        print(f"{name:<24} {r['n_img']:>8} {r['params']:>10,} "
              f"{r['acc']:>11.3f} ± {r['std']:.3f} {r['secs']:>7.1f}")

    ids, images, labels = make_data(800, 999, image_token_count(12, 3))
    model = build(3, "mlp", False)
    with torch.no_grad():
        no_img = model(ids).logits[:, -1]
        acc_no = (no_img.argmax(-1) == labels[:, -1]).float().mean().item()
    print(f"{'去掉图像（对照）':<24} {0:>8} {'—':>10} {acc_no:>11.3f}       {'—':>7}")

if __name__ == "__main__":
    main()
