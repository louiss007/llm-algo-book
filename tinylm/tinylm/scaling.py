"""缩放定律与算力预算（tinylm/scaling.py，第 8、9 章）。"""
import math

import torch
import torch.nn.functional as F

from .attention import attention_params
from .ssm import layer_kinds


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


def norm_params(cfg):
    d = cfg.d_model
    return d + (d if cfg.norm == "layernorm" and cfg.bias else 0)


def expert_params(cfg):
    """公式 (7.10)：单个专家的参数量。"""
    d, f = cfg.d_model, (cfg.moe.d_ff_expert or cfg.d_ff)
    if cfg.ffn == "swiglu":
        return 3 * d * f + ((2 * f + d) if cfg.bias else 0)
    return 2 * d * f + ((f + d) if cfg.bias else 0)


def dense_ffn_params(cfg):
    d, f = cfg.d_model, cfg.d_ff
    if cfg.ffn == "swiglu":
        return 3 * d * f + ((2 * f + d) if cfg.bias else 0)
    return 2 * d * f + ((f + d) if cfg.bias else 0)


def moe_layer_params(cfg, active=False):
    """公式 (7.11)、(7.13)：路由 + 参与计算的专家个数。"""
    m = cfg.moe
    used = (m.top_k + m.n_shared) if active else (m.n_experts + m.n_shared)
    return cfg.d_model * m.n_experts + used * expert_params(cfg)


def ssm_layer_params(cfg):
    """公式 (8.16)：单层 SSM 的参数量。"""
    I = cfg.ssm_expand * cfg.d_model
    N = cfg.ssm_d_state
    p = 3 * cfg.d_model * I + I * (3 * N + 7)
    p += (4 * I + cfg.d_model) if cfg.bias else I  # dt_proj 始终带偏置
    return p


def block_params(cfg, kind, active=False):
    """单层参数量：三种块只替换 mixer，前馈子层与归一化保持一致。"""
    d, n = cfg.d_model, norm_params(cfg)
    if cfg.moe is not None:
        ffn = moe_layer_params(cfg, active=active)
    else:
        ffn = dense_ffn_params(cfg)
    norms = n if cfg.parallel_block else 2 * n
    if kind == "attn":
        mixer = attention_params(cfg)
    elif kind == "linear":
        mixer = 4 * d * d + (4 * d if cfg.bias else 0)
    else:
        mixer = ssm_layer_params(cfg)
    return mixer + ffn + norms


def model_params(cfg, active=False):
    """按块类型逐层累加，避免为大配置实例化整个模型。"""
    emb = cfg.vocab_size * cfg.d_model * (1 if cfg.tie_embeddings else 2)
    pos = cfg.max_seq_len * cfg.d_model if cfg.pos_encoding == "learned" else 0
    return (sum(block_params(cfg, k, active=active) for k in layer_kinds(cfg))
            + norm_params(cfg) + emb + pos)


def state_bytes(cfg, batch=1, seq_len=8192, b_elem=2):
    """每层解码状态：注意力缓存随 T 增长，SSM/线性注意力状态与 T 无关。"""
    d = cfg.d_model
    total = 0
    for kind in layer_kinds(cfg):
        if kind == "attn":
            if cfg.attention == "mla":
                total += batch * seq_len * (cfg.d_c + cfg.d_head_rope) * b_elem
            else:
                d_kv = cfg.n_kv_heads * (d // cfg.n_heads)
                total += 2 * batch * seq_len * d_kv * b_elem
        elif kind == "linear":
            total += batch * (d * d + d) * b_elem
        else:
            total += batch * (cfg.ssm_expand * d) * cfg.ssm_d_state * b_elem
    return total


def decode_flops(cfg, seq_len):
    """每 Token 解码的粗略 FLOPs：权重读取 + 序列相关项。"""
    total = 2 * model_params(cfg)
    d = cfg.d_model
    for kind in layer_kinds(cfg):
        if kind == "attn":
            total += 4 * seq_len * d
        elif kind == "linear":
            total += 4 * d * d
        else:
            total += 4 * (cfg.ssm_expand * d) * cfg.ssm_d_state
    return total
