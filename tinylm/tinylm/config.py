"""模型配置（tinylm/config.py）。

第 3 章给出基线，第 4–8 章逐章增加可选字段；所有默认值保持第 3 章行为。
"""
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class RoPEScalingConfig:
    """RoPE 的长度外推配置。factor = 目标长度 / 训练长度。"""
    method: Literal["none", "pi", "ntk", "yarn"] = "none"
    factor: float = 1.0


@dataclass(frozen=True)
class MoEConfig:
    n_experts: int = 8
    top_k: int = 2
    n_shared: int = 0
    d_ff_expert: int | None = None        # None 表示沿用 ModelConfig.d_ff
    capacity_factor: float | None = 1.25  # None 表示不丢弃
    aux_loss_coef: float = 0.01

    def __post_init__(self):
        assert self.n_experts >= 1
        assert 1 <= self.top_k <= self.n_experts
        assert self.n_shared >= 0
        assert self.d_ff_expert is None or self.d_ff_expert > 0
        assert self.capacity_factor is None or self.capacity_factor > 0
        assert self.aux_loss_coef >= 0


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int | None = None
    d_ff: int | None = None
    max_seq_len: int = 1024
    attention: Literal["mha", "mqa", "gqa", "mla"] = "mha"
    d_c: int = 128
    d_head_rope: int = 32
    norm: Literal["layernorm", "rmsnorm"] = "layernorm"
    norm_eps: float = 1e-5
    ffn: Literal["mlp", "swiglu"] = "mlp"
    pos_encoding: Literal["learned", "sine", "rope", "nope", "alibi"] = "learned"
    rope_theta: float = 10_000.0
    rope_scaling: RoPEScalingConfig = field(default_factory=RoPEScalingConfig)
    bias: bool = True
    tie_embeddings: bool = False
    parallel_block: bool = False
    dropout: float = 0.0
    moe: MoEConfig | None = None
    block_type: Literal["attention", "linear_attention", "ssm", "hybrid"] = "attention"
    ssm_d_state: int = 16
    ssm_expand: int = 2
    hybrid_attn_every: int = 4

    def __post_init__(self):
        # 公式 (5.8)：SwiGLU 的中间维度取 8/3 d_model，使参数量与 MLP 可比
        if self.d_ff is None:
            ratio = 8 / 3 if self.ffn == "swiglu" else 4
            object.__setattr__(self, "d_ff", int(round(ratio * self.d_model)))

        assert self.d_model % self.n_heads == 0, "d_model 必须能被 n_heads 整除"
        assert self.norm in ("layernorm", "rmsnorm")
        assert self.ffn in ("mlp", "swiglu")
        assert self.pos_encoding in ("learned", "sine", "rope", "nope", "alibi")

        if self.attention == "mla":
            assert self.d_c > 0 and self.d_head_rope > 0
        else:
            if self.attention == "mha":
                n_kv = self.n_heads
            elif self.attention == "mqa":
                n_kv = 1
            elif self.attention == "gqa":
                assert self.n_kv_heads is not None, "GQA 必须指定 n_kv_heads"
                n_kv = self.n_kv_heads
            else:
                raise ValueError(f"未知注意力类型: {self.attention}")
            object.__setattr__(self, "n_kv_heads", n_kv)
            assert self.n_heads % n_kv == 0, "n_heads 必须是 n_kv_heads 的整数倍"

        if self.moe is not None:
            assert self.moe.top_k <= self.moe.n_experts

        assert self.block_type in ("attention", "linear_attention", "ssm", "hybrid")
        assert self.ssm_d_state >= 1
        assert self.ssm_expand >= 1
        assert self.hybrid_attn_every >= 1


def load_config(path: str | Path) -> ModelConfig:
    """读取扁平 YAML 配置。

    只支持 ``key: value`` 与 ``#`` 注释，避免给 TinyLM 引入 YAML 依赖；
    嵌套的 ``rope_scaling`` 与 ``moe`` 用 ``rope_scaling_method`` 等前缀字段表达。
    """
    raw = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        raw[key] = _parse_scalar(value)

    fields_by_name = {f.name: f for f in fields(ModelConfig)}
    unknown = set(raw) - set(fields_by_name) - {"rope_scaling_method",
                                                "rope_scaling_factor"}
    if unknown:
        raise ValueError(f"未知配置项: {sorted(unknown)}")

    raw["rope_scaling"] = RoPEScalingConfig(
        method=raw.pop("rope_scaling_method", "none"),
        factor=float(raw.pop("rope_scaling_factor", 1.0)),
    )
    return ModelConfig(**raw)


def _parse_scalar(text: str):
    low = text.lower()
    if low in ("null", "none", "~"):
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text
