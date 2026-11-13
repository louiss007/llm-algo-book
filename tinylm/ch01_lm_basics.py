"""第 1 章配套代码：n-gram、神经 n-gram 与 RNN 三种语言模型基线。
运行：python ch01_lm_basics.py
"""
import math
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# ---------- 1. 可控语料：二阶马尔可夫链 ----------
C = 6  # 词表大小

def transition(a: int, b: int):
    """给定前两个符号 (a, b)，返回下一个符号的分布。"""
    t1 = b
    t2 = (a + b + 1) % C
    if t1 == t2:
        t2 = (t2 + 1) % C
    return [(t1, 0.8), (t2, 0.2)]

T = {(a, b): transition(a, b) for a in range(C) for b in range(C)}

def exact_entropies():
    """在 36 个二阶状态上求平稳分布，得到精确的 H1/H2/H3。"""
    P = np.zeros((C * C, C * C))
    for a in range(C):
        for b in range(C):
            i = a * C + b
            for c, p in T[(a, b)]:
                P[i, b * C + c] += p
    pi = np.ones(C * C) / (C * C)
    for _ in range(10_000):
        pi = pi @ P
    pi /= pi.sum()

    H3 = 0.0
    for a in range(C):
        for b in range(C):
            H3 -= pi[a * C + b] * sum(p * math.log(p) for _, p in T[(a, b)])

    marg = np.zeros(C)
    for a in range(C):
        for b in range(C):
            marg[b] += pi[a * C + b]
    H1 = -sum(x * math.log(x) for x in marg)

    H2 = 0.0
    for b in range(C):
        pb = sum(pi[a * C + b] for a in range(C))
        cond = np.zeros(C)
        for a in range(C):
            for c, p in T[(a, b)]:
                cond[c] += pi[a * C + b] * p
        cond /= pb
        H2 -= pb * sum(x * math.log(x) for x in cond if x > 0)
    return H1, H2, H3

H1, H2, H3 = exact_entropies()
print(f"真实熵: H1={H1:.4f}  H2={H2:.4f}  H3={H3:.4f} (nats)")
print(f"真实 PPL: {math.exp(H1):.4f}  {math.exp(H2):.4f}  {math.exp(H3):.4f}")

rng = random.Random(42)
N = 120_000
seq = [0, 1]
for _ in range(N - 2):
    a, b = seq[-2], seq[-1]
    outcomes, probs = zip(*T[(a, b)])
    seq.append(rng.choices(outcomes, weights=probs)[0])

split = int(0.8 * N)
train, test = seq[:split], seq[split:]

# ---------- 2. n-gram（公式 1.8、1.9） ----------
class NGram:
    def __init__(self, n: int, k: float = 0.1):
        self.n, self.k = n, k  # k：加性平滑常数

    def fit(self, tokens):
        self.count = Counter()
        self.context_count = Counter()
        for i in range(self.n - 1, len(tokens)):
            ctx = tuple(tokens[i - self.n + 1:i]) if self.n > 1 else ()
            self.count[ctx + (tokens[i],)] += 1
            self.context_count[ctx] += 1
        return self

    def cross_entropy(self, tokens):
        total = 0.0
        for i in range(self.n - 1, len(tokens)):
            ctx = tuple(tokens[i - self.n + 1:i]) if self.n > 1 else ()
            num = self.count[ctx + (tokens[i],)] + self.k
            den = self.context_count[ctx] + self.k * C
            total -= math.log(num / den)          # 公式 (1.9)
        return total / (len(tokens) - self.n + 1)

for n in (1, 2, 3):
    ce = NGram(n).fit(train).cross_entropy(test)
    print(f"{n}-gram:  CE={ce:.4f}  PPL={math.exp(ce):.4f}")

# ---------- 3. 神经 n-gram（公式 1.11–1.14） ----------
def make_xy(tokens, order):
    x = torch.tensor([tokens[i - order:i] for i in range(order, len(tokens))])
    y = torch.tensor(tokens[order:])
    return x, y

xb1, yb1 = make_xy(train, 1); xt1, yt1 = make_xy(test, 1)
xb2, yb2 = make_xy(train, 2); xt2, yt2 = make_xy(test, 2)

class NeuralBigram(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.emb = nn.Embedding(C, d)      # 公式 (1.11)
        self.out = nn.Linear(d, C)         # 公式 (1.13)(1.14)

    def forward(self, x):
        return self.out(self.emb(x).squeeze(1))

m = NeuralBigram()
opt = torch.optim.Adam(m.parameters(), lr=0.02)
for _ in range(3_000):
    idx = torch.randint(0, len(xb1), (512,))
    loss = F.cross_entropy(m(xb1[idx]), yb1[idx])   # 公式 (1.3)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    ce = F.cross_entropy(m(xt1), yt1).item()
print(f"神经二元: CE={ce:.4f}  PPL={math.exp(ce):.4f}")

class NeuralTrigram(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.emb = nn.Embedding(C, d)
        self.mlp = nn.Sequential(nn.Linear(2 * d, 64), nn.ReLU(), nn.Linear(64, C))

    def forward(self, x):
        return self.mlp(self.emb(x).flatten(1))

m3 = NeuralTrigram()
opt = torch.optim.Adam(m3.parameters(), lr=0.02)
for _ in range(3_000):
    idx = torch.randint(0, len(xb2), (512,))
    loss = F.cross_entropy(m3(xb2[idx]), yb2[idx])
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    ce = F.cross_entropy(m3(xt2), yt2).item()
print(f"神经三元: CE={ce:.4f}  PPL={math.exp(ce):.4f}")

# ---------- 4. RNN（公式 1.12） ----------
class RNNLM(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.emb = nn.Embedding(C, d)
        self.rnn = nn.RNN(d, d, batch_first=True)   # 公式 (1.12)
        self.out = nn.Linear(d, C)                  # 公式 (1.13)(1.14)

    def forward(self, x, h=None):
        z, h = self.rnn(self.emb(x), h)
        return self.out(z), h

r = RNNLM()
opt = torch.optim.Adam(r.parameters(), lr=0.01)
tr = torch.tensor(train)
for _ in range(3_000):
    starts = torch.randint(1, len(tr) - 65, (32,))
    xb = torch.stack([tr[i - 1:i + 63] for i in starts])
    yb = torch.stack([tr[i:i + 64] for i in starts])
    logits, _ = r(xb)
    loss = F.cross_entropy(logits.reshape(-1, C), yb.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    te = torch.tensor(test).unsqueeze(0)
    h, total, count = None, 0.0, 0
    for i in range(0, len(test) - 1, 256):
        end = min(i + 256, len(test) - 1)
        logits, h = r(te[:, i:end], h)
        target = te[:, i + 1:end + 1]
        total += F.cross_entropy(logits.reshape(-1, C), target.reshape(-1),
                                 reduction="sum").item()
        count += target.numel()
    ce = total / count
print(f"RNN:     CE={ce:.4f}  PPL={math.exp(ce):.4f}")

# ---------- 5. 采样 ----------
def sample(model, length=24):
    ctx = tuple(random.choices(range(C), k=model.n - 1)) if model.n > 1 else ()
    out = []
    for _ in range(length):
        weights = [model.count[ctx + (c,)] + model.k for c in range(C)]
        c = random.choices(range(C), weights=weights)[0]
        out.append(c)
        ctx = (ctx + (c,))[-(model.n - 1):] if model.n > 1 else ()
    return "".join(chr(ord("a") + x) for x in out)

print("三元模型采样:", sample(NGram(3).fit(train)))
