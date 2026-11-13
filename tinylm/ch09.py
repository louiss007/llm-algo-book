"""第 9 章：缩放定律（tinylm/scaling.py 与实验）。"""
import math

import torch
import torch.nn.functional as F

from ch08 import ModelConfig, TinyLM, decode_flops, model_params


# ---------------------------------------------------------------- scaling.py
def power_law_loss(N, D, E, A, B, alpha, beta):
    """公式 (9.1)：L(N,D)=E+A/N^α+B/D^β。"""
    return E + A / N**alpha + B / D**beta


def optimal_exponents(alpha, beta):
    """公式 (9.5)：计算最优分配的两个指数。"""
    s = alpha + beta
    return beta / s, alpha / s


def compute_optimal(C, E, A, B, alpha, beta, lo=6.0, hi=14.0):
    """公式 (9.3)–(9.6)：在 6ND=C 约束下最小化 (9.1)。

    固定 C 后 D=C/(6N)，问题化为一维搜索；在对数 N 上做黄金分割。
    """
    def objective(log_n):
        N = 10.0 ** log_n
        D = C / (6.0 * N)
        return float(power_law_loss(N, D, E, A, B, alpha, beta))

    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = objective(c), objective(d)
    for _ in range(200):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = objective(d)
    log_n = (a + b) / 2.0
    N = 10.0 ** log_n
    D = C / (6.0 * N)
    return N, D, float(power_law_loss(N, D, E, A, B, alpha, beta))


def fit_scaling_law(points, fix_E=None, steps=3000, restarts=3, seed=0):
    """用对数损失上的 Huber 回归拟合 (9.1)；返回参数字典与拟合误差。

    points: (M,3) 张量，三列分别是 N、D、观测损失 L。
    fix_E: 若给出，固定不可约损失 E，只拟合其余四个参数。
    """
    pts = torch.as_tensor(points, dtype=torch.float32)
    N, D, L = pts[:, 0], pts[:, 1], pts[:, 2]
    log_L = torch.log(L)
    best = None
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed + r)
        raw = torch.zeros(5)
        raw[0] = math.log(math.expm1(0.5)) if fix_E is None else 0.0
        raw[1] = math.log(400.0) + 0.5 * torch.randn((), generator=g).item()
        raw[2] = math.log(400.0) + 0.5 * torch.randn((), generator=g).item()
        raw[3] = 0.0
        raw[4] = 0.0
        raw = raw.requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=0.05)
        for _ in range(steps):
            if fix_E is None:
                E = F.softplus(raw[0]) + 1e-3
            else:
                E = torch.tensor(float(fix_E))
            A = torch.exp(raw[1])
            B = torch.exp(raw[2])
            alpha = torch.sigmoid(raw[3])
            beta = torch.sigmoid(raw[4])
            pred = torch.log(E + A / N**alpha + B / D**beta)
            loss = F.huber_loss(pred, log_L, delta=1e-3)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            if fix_E is None:
                E = float(F.softplus(raw[0]) + 1e-3)
            else:
                E = float(fix_E)
            params = dict(E=E, A=float(torch.exp(raw[1])), B=float(torch.exp(raw[2])),
                          alpha=float(torch.sigmoid(raw[3])),
                          beta=float(torch.sigmoid(raw[4])))
            pred = torch.log(torch.tensor(params["E"]) + params["A"] / N**params["alpha"]
                             + params["B"] / D**params["beta"])
            rmse = float(torch.sqrt(torch.mean((pred - log_L) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, params)
    rmse, params = best
    params["rmse_log"] = rmse
    return params


R_STAR = 15.4  # Muennighoff et al. 2023 拟合的重复半程常数


def effective_data(D, U, r_star=R_STAR):
    """公式 (9.8)：数据受限时的有效数据量（Muennighoff et al. 2023）。

    D 是计划训练的 Token 数，U 是唯一 Token 数；R=D/U-1 是重复轮数。
    """
    if D <= U:
        return float(D)
    R = D / U - 1.0
    return U + U * r_star * (1.0 - math.exp(-R / r_star))


def training_flops(N, D):
    """公式 (9.2)：训练算力 C≈6ND。"""
    return 6.0 * N * D


# ---------------------------------------------------------------- 合成数据验证
def synthetic_points(true, seed=0):
    """从已知幂律生成带噪声的观测点，用于验证拟合器。"""
    g = torch.Generator().manual_seed(seed)
    Ns = torch.logspace(7, 10, 5)
    Ds = torch.logspace(8, 11, 5)
    pts = []
    for N in Ns:
        for D in Ds:
            L = float(power_law_loss(float(N), float(D), **true))
            L *= math.exp(0.01 * float(torch.randn((), generator=g)))
            pts.append((float(N), float(D), L))
    return torch.tensor(pts)


# ---------------------------------------------------------------- 玩具扫描
V, T, LAG = 16, 32, 8
FLOOR = (LAG - 1) / (T - 1) * math.log(V)


def copy_batch(B, T, V, gen, lag=LAG):
    x = torch.randint(0, V, (B, lag), generator=gen)
    for t in range(lag, T):
        x = torch.cat([x, x[:, t - lag:t - lag + 1]], dim=1)
    return x


def toy_train(cfg, steps, seed=0, lr=3e-3, batch=32):
    torch.manual_seed(seed)
    model = TinyLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(1000 + seed)
    model.train()
    for _ in range(steps):
        seq = copy_batch(batch, T, V, gen)
        out = model(seq[:, :-1], labels=seq[:, 1:].clone())
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    return model


@torch.no_grad()
def toy_eval(model, seed=7, n=512):
    gen = torch.Generator().manual_seed(seed)
    seq = copy_batch(n, T, V, gen)
    inp, tgt = seq[:, :-1], seq[:, 1:].clone()
    model.eval()
    out = model(inp, labels=tgt)
    acc = (out.logits.argmax(-1) == tgt).float().mean().item()
    return out.loss.item(), acc


def toy_sweep(seeds=(0, 1, 2)):
    common = dict(vocab_size=V, n_heads=4, max_seq_len=T, norm="rmsnorm",
                  ffn="swiglu", bias=False, pos_encoding="rope",
                  attention="gqa", n_kv_heads=2)
    sizes = {
        "s": dict(d_model=32, n_layers=1, d_ff=64),
        "m": dict(d_model=48, n_layers=2, d_ff=96),
        "l": dict(d_model=64, n_layers=2, d_ff=128),
    }
    budgets = (100, 400, 1600)
    rows = []
    print("玩具扫描：复制任务（lag=8，T=32，V=16，3 个种子取均值）")
    print(f"理论损失下界 {FLOOR:.4f}，随机猜测损失 {math.log(V):.4f}")
    print(f"{'配置':<4s}{'N':>9s}{'D':>11s}{'损失':>16s}{'准确率':>16s}{'C=6ND':>11s}")
    for name, sz in sizes.items():
        cfg = ModelConfig(**common, **sz)
        for steps in budgets:
            losses, accs = [], []
            for seed in seeds:
                model = toy_train(cfg, steps, seed=seed)
                loss, acc = toy_eval(model)
                losses.append(loss)
                accs.append(acc)
            loss = torch.tensor(losses)
            acc = torch.tensor(accs)
            N = model_params(cfg)
            D = steps * 32 * T
            rows.append((N, D, float(loss.mean())))
            print(f"{name:<4s}{N:>9,d}{D:>11,d}"
                  f"{loss.mean():>10.4f} ± {loss.std():.4f}"
                  f"{acc.mean():>10.4f} ± {acc.std():.4f}"
                  f"{training_flops(N, D):>11.2e}")
    return rows


# ---------------------------------------------------------------- 账本
def chinchilla_params():
    return dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)


def scaling_allocation(C, a=0.46, b=0.54, G0=1.858):
    """公式 (9.6)：N_opt=g0(C/6)^a，D_opt=g0^{-1}(C/6)^b。

    a、b 取 Chinchilla Approach 3 的报告值；g0 由 70B/1.4T 的锚点校准。
    """
    N = G0 * (C / 6.0) ** a
    D = (C / 6.0) ** b / G0
    return N, D


def allocation_table():
    p = chinchilla_params()
    print("\n计算最优分配（a=0.46，b=0.54，g0=1.858；损失按 (9.1) 代入）")
    print(f"{'C (FLOPs)':>12s}{'N_opt':>14s}{'D_opt':>14s}{'D/N':>8s}{'L_pred':>9s}")
    for e in (20, 21, 22, 23, 24, 25):
        C = 10.0 ** e
        N, D = scaling_allocation(C)
        L = float(power_law_loss(N, D, **p))
        print(f"{C:>12.1e}{N:>14.3e}{D:>14.3e}{D/N:>8.1f}{L:>9.4f}")


def repetition_table():
    U = 1e12
    print("\n数据重复的有效数据量（U=1e12，R*=15.4）")
    print(f"{'重复轮数 R':>10s}{'D/U':>8s}{'D_eff/U':>10s}{'D_eff/D':>10s}")
    for r in (0, 1, 3, 7, 15, 31):
        D = (r + 1) * U
        de = effective_data(D, U)
        print(f"{r:>10d}{r + 1:>8d}{de/U:>10.3f}{de/D:>10.3f}")
    print(f"重复轮数趋于无穷时，D_eff 上限为 U(1+R*) = {1 + R_STAR:.1f}U")


def inference_ledger(infer_tokens=1e13):
    """训练 + 推理总成本；解码 FLOPs 复用第 8 章的 decode_flops。"""
    base = dict(vocab_size=128256, max_seq_len=8192, norm="rmsnorm",
                ffn="swiglu", bias=False, pos_encoding="rope", attention="gqa")
    m8 = ModelConfig(d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8,
                     d_ff=14336, **base)
    m70 = ModelConfig(d_model=8192, n_layers=80, n_heads=64, n_kv_heads=8,
                      d_ff=28672, **base)
    rows = [("8B / 15T", m8, 15e12), ("70B / 1.4T", m70, 1.4e12)]
    print(f"\n训练 + 推理账本（上下文 8192，推理 {infer_tokens:.0e} Token）")
    print(f"{'模型/数据':<12s}{'参数量':>14s}{'训练 FLOPs':>12s}"
          f"{'每 Token 解码':>13s}{'推理 FLOPs':>12s}{'合计':>12s}")
    for name, cfg, D in rows:
        N = model_params(cfg)
        train = training_flops(N, D)
        dec = decode_flops(cfg, 8192)
        infer = dec * infer_tokens
        print(f"{name:<12s}{N:>14,d}{train:>12.2e}{dec:>13.2e}"
              f"{infer:>12.2e}{train+infer:>12.2e}")
    return rows


def breakeven():
    """8B/15T 与 70B/1.4T 的训练 + 推理总成本相等时的推理 Token 数。"""
    base = dict(vocab_size=128256, max_seq_len=8192, norm="rmsnorm",
                ffn="swiglu", bias=False, pos_encoding="rope", attention="gqa")
    m8 = ModelConfig(d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8,
                     d_ff=14336, **base)
    m70 = ModelConfig(d_model=8192, n_layers=80, n_heads=64, n_kv_heads=8,
                      d_ff=28672, **base)
    t8, t70 = training_flops(model_params(m8), 15e12), training_flops(model_params(m70), 1.4e12)
    d8, d70 = decode_flops(m8, 8192), decode_flops(m70, 8192)
    q = (t70 - t8) / (d8 - d70)
    print(f"\n8B/15T 与 70B/1.4T 的总成本在推理 {q:.2e} Token 后相等；"
          f"此后 8B 更省。")


def main():
    true = dict(E=1.7, A=400.0, B=400.0, alpha=0.34, beta=0.28)
    pts = synthetic_points(true)
    got = fit_scaling_law(pts, seed=0)
    print("合成数据验证：已知幂律 + 1% 乘性噪声，35 个点")
    print(f"{'参数':<8s}{'真值':>10s}{'拟合':>10s}{'相对误差':>10s}")
    for k in ("E", "A", "B", "alpha", "beta"):
        print(f"{k:<8s}{true[k]:>10.4f}{got[k]:>10.4f}"
              f"{abs(got[k]-true[k])/abs(true[k]):>10.2%}")
    print(f"对数损失 RMSE：{got['rmse_log']:.5f}")

    allocation_table()
    repetition_table()

    toy_sweep()
    print("\n玩具扫描的算力跨度只有约 1.8 个数量级，且部分配置已经贴近损失下界，")
    print("不足以稳定识别 (9.1) 的五个参数；本章只用它验证方向，不用它估计指数。")

    inference_ledger()
    breakeven()


if __name__ == "__main__":
    main()
