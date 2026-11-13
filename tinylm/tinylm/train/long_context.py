"""第 13 章：长上下文的算力、显存与长度调度（tinylm/train/long_context.py）。"""


def attention_flops(batch, seq_len, d_model, n_layers):
    """(13.1)：因果注意力的前向 FLOPs（QK^T 与 AV 各一次）。"""
    return 2 * batch * n_layers * d_model * seq_len * (seq_len + 1)


def linear_flops(batch, seq_len, d_model, d_ff, n_layers):
    """(13.2)：投影与 SwiGLU 前馈的前向 FLOPs。"""
    return 2 * batch * seq_len * n_layers * (4 * d_model ** 2 + 3 * d_model * d_ff)


def attention_share(batch, seq_len, d_model, d_ff, n_layers):
    """注意力 FLOPs 占总前向 FLOPs 的比例。"""
    attn = attention_flops(batch, seq_len, d_model, n_layers)
    linear = linear_flops(batch, seq_len, d_model, d_ff, n_layers)
    return attn / (attn + linear)


def crossover_length(d_model, d_ff):
    """(13.3)：注意力 FLOPs 追上线性项时的序列长度。"""
    return 4 * d_model + 3 * d_ff - 1


def kv_cache_bytes(batch, seq_len, n_layers, n_kv_heads, d_head,
                   bytes_per_elem=2):
    """(13.4)：推理 KV Cache 的字节数。"""
    return (2 * batch * seq_len * n_layers * n_kv_heads * d_head
            * bytes_per_elem)


def activation_bytes(batch, seq_len, d_model, n_layers, n_heads,
                     bytes_per_elem=2, coef=12.0, materialize_scores=False):
    """(13.5)：训练激活；显式分数矩阵是额外的 T^2 * n_heads 项。"""
    base = coef * batch * seq_len * n_layers * d_model * bytes_per_elem
    if materialize_scores:
        base += batch * n_layers * n_heads * seq_len ** 2 * bytes_per_elem
    return base


def length_schedule(base_len, target_len, stages):
    """(13.6)：几何增长的长度课程，返回 [(长度, 步数占比), ...]。"""
    assert stages >= 1 and target_len >= base_len >= 1
    if stages == 1:
        return [(target_len, 1.0)]
    ratio = (target_len / base_len) ** (1 / (stages - 1))
    lengths = [int(round(base_len * ratio ** i)) for i in range(stages)]
    lengths[-1] = target_len
    return [(n, 1.0 / stages) for n in lengths]


def token_budget_shares(lengths):
    """等 Token 预算时各长度应分配的步数占比。"""
    inv = [1.0 / n for n in lengths]
    total = sum(inv)
    return [x / total for x in inv]


def effective_context(lengths, scores, threshold=0.5):
    """(13.10)：指标不低于阈值的最长评测长度；全部不达标时返回 0。"""
    passed = [n for n, s in zip(lengths, scores) if s >= threshold]
    return max(passed) if passed else 0
