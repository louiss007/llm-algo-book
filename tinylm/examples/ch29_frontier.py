"""第 29 章配套实验：持续学习、合成数据自举与稀疏奖励。

只依赖 PyTorch，不修改 TinyLM 的任何接口。所有随机性由固定种子控制。
"""

import math

import torch
import torch.nn as nn

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

# ---------------------------------------------------------------- 实验 1
def task_data(n: int, task: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g) * 2 - 1
    y = torch.sin(2 * math.pi * x) if task == 0 else torch.cos(2 * math.pi * x)
    flag = torch.full_like(x, float(task))
    return torch.cat([x, flag], dim=1), y

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

def mse(model, x, y):
    with torch.no_grad():
        return ((model(x) - y) ** 2).mean().item()

def train(model, x, y, steps, *, ewc=None, lam=0.0, replay=None, rho=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        if replay is not None and torch.rand(1, generator=g).item() < rho:  # (29.6)
            xb, yb = replay
        else:
            xb, yb = x, y
        loss = ((model(xb) - yb) ** 2).mean()
        if ewc is not None:  # (29.5)
            penalty = sum(
                (f * (p - star) ** 2).sum()
                for p, star, f in zip(model.parameters(), ewc["star"], ewc["fisher"])
            )
            loss = loss + 0.5 * lam * penalty
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model

def fisher_diag(model, x, y):
    fisher = [torch.zeros_like(p) for p in model.parameters()]
    for i in range(x.shape[0]):
        model.zero_grad()
        ((model(x[i:i + 1]) - y[i:i + 1]) ** 2).mean().backward()
        for f, p in zip(fisher, model.parameters()):
            f += p.grad.detach() ** 2
    return [f / x.shape[0] for f in fisher]

def continual_run(seed: int, steps: int = 500, lam: float = 10_000.0):
    xa, ya = task_data(200, 0, seed * 10 + 1)
    xb, yb = task_data(200, 1, seed * 10 + 2)
    result = {}
    for mode in ("naive", "ewc", "replay"):
        set_seed(seed + 100)
        model = MLP()
        train(model, xa, ya, steps, seed=seed)
        before = mse(model, xa, ya)
        if mode == "naive":
            train(model, xb, yb, steps, seed=seed + 1)
        elif mode == "ewc":
            star = [p.detach().clone() for p in model.parameters()]
            fisher = fisher_diag(model, xa, ya)
            train(model, xb, yb, steps,
                  ewc={"star": star, "fisher": fisher}, lam=lam, seed=seed + 1)
        else:
            train(model, xb, yb, steps, replay=(xa, ya), rho=0.5, seed=seed + 1)
        result[mode] = (before, mse(model, xa, ya), mse(model, xb, yb))
    return result

# ---------------------------------------------------------------- 实验 2
def collapse_trace(seed: int, n: int = 100, generations: int = 200, alpha: float = 0.0):
    g = torch.Generator().manual_seed(seed)
    mu, var = 0.0, 1.0
    trace = {}
    for t in range(generations + 1):
        if t in (0, 50, 100, 200):
            trace[t] = var
        if t == generations:
            break
        n_real = int(round(n * alpha))
        n_syn = n - n_real
        synthetic = mu + math.sqrt(var) * torch.randn(n_syn, generator=g)
        if n_real:
            data = torch.cat([synthetic, torch.randn(n_real, generator=g)])
        else:
            data = synthetic
        mu = data.mean().item()
        var = ((data - mu) ** 2).mean().item()  # (29.8)-(29.9)
    return trace

# ---------------------------------------------------------------- 实验 3
def sparse_rl(seed: int, steps: int = 8, batch: int = 4000,
              updates: int = 200, lr: float = 0.5, mode: str = "terminal"):
    g = torch.Generator().manual_seed(seed)
    theta = torch.zeros(steps)
    for _ in range(updates):
        p = torch.sigmoid(theta)
        actions = (torch.rand(batch, steps, generator=g) < p).float()
        if mode == "terminal":  # (29.1)-(29.2)
            reward = (actions.prod(dim=1) > 0).float()
            baseline = reward.mean()
            grad = ((reward - baseline).unsqueeze(1) * (actions - p)).mean(dim=0)
        elif mode == "step":
            dense = actions.clone()
            dense[:, -1] += 1.0
            returns = torch.flip(torch.cumsum(torch.flip(dense, [1]), dim=1), [1])
            baseline = returns.mean(dim=0, keepdim=True)
            grad = ((returns - baseline) * (actions - p)).mean(dim=0)
        else:  # (29.4): Phi(s_t) = 累计正确步数，gamma = 1
            phi = torch.cumsum(actions, dim=1)
            dense = torch.zeros_like(actions)
            dense[:, 0] = phi[:, 0]
            dense[:, 1:] = phi[:, 1:] - phi[:, :-1]
            dense[:, -1] += 1.0
            returns = torch.flip(torch.cumsum(torch.flip(dense, [1]), dim=1), [1])
            baseline = returns.mean(dim=0, keepdim=True)
            grad = ((returns - baseline) * (actions - p)).mean(dim=0)
        theta = theta + lr * grad
    p = torch.sigmoid(theta)
    sample = (torch.rand(20_000, steps, generator=g) < p).float()
    empirical = (sample.prod(dim=1) > 0).float().mean().item()
    return p.prod().item(), empirical

def mean_std(values):
    mu = sum(values) / len(values)
    sd = (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5
    return mu, sd

def main():
    print("持续学习：先学任务 A（sin），再学任务 B（cos），3 个种子")
    results = [continual_run(s) for s in range(3)]
    print("方法\tA 前\tA 后\tB\t遗忘")
    for mode, name in (("naive", "顺序训练"), ("ewc", "EWC λ=1e4"), ("replay", "回放 ρ=0.5")):
        before = mean_std([r[mode][0] for r in results])
        after = mean_std([r[mode][1] for r in results])
        b = mean_std([r[mode][2] for r in results])
        forget = (after[0] - before[0], (after[1] ** 2 + before[1] ** 2) ** 0.5)
        print(f"{name}\t{before[0]:.4f}±{before[1]:.4f}\t{after[0]:.4f}±{after[1]:.4f}"
              f"\t{b[0]:.4f}±{b[1]:.4f}\t{forget[0]:.4f}±{forget[1]:.4f}")

    print("\n合成数据自举：1 维高斯，N=100，200 代，200 次模拟")
    print("代数\t纯合成\t10% 真实数据\t理论期望")
    pure = [collapse_trace(s, alpha=0.0) for s in range(200)]
    anchor = [collapse_trace(s, alpha=0.1) for s in range(200)]
    for t in (0, 50, 100, 200):
        pure_mean = math.sqrt(sum(r[t] for r in pure) / len(pure))
        anchor_mean = math.sqrt(sum(r[t] for r in anchor) / len(anchor))
        theory = (0.99) ** (t / 2)
        print(f"{t}\t{pure_mean:.4f}\t{anchor_mean:.4f}\t{theory:.4f}")

    print("\n稀疏奖励信用分配：8 步链，4000 条轨迹/次，200 次更新，3 个种子")
    print("信号\t成功概率（解析）\t成功率（2 万次采样）")
    for mode, name in (("terminal", "终端奖励"), ("step", "逐步验证器"), ("potential", "势函数塑形")):
        vals = [sparse_rl(s, mode=mode) for s in range(3)]
        p = mean_std([v[0] for v in vals])
        e = mean_std([v[1] for v in vals])
        print(f"{name}\t{p[0]:.4f}±{p[1]:.4f}\t{e[0]:.4f}±{e[1]:.4f}")

if __name__ == "__main__":
    main()
