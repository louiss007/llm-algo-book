# 附录 F TinyLM：设计、接口与复现指南

TinyLM 是贯穿全书的配套项目：一个只依赖 PyTorch、从零实现的小型语言模型。它的目的不是复现某个具体开源模型，而是让书中的每个设计选择都能被**单独打开或关闭**，从而做对照实验。

{% hint style="success" %}
TinyLM 的三条设计原则：
1. **接口先冻结，再逐章加功能**——后一章可以新增字段和方法，但不得修改前面章节已确立的公共签名。
2. **两档配置**——`tiny_cpu` 保证任何读者能跑通，`tiny_24g` 保证单张 24GB 显卡能做出有意义的实验。
3. **每个公式都能落到一行代码**——关键实现注释里标注对应的公式编号。
{% endhint %}

## F.1 目录结构

```text
tinylm/
├── pyproject.toml
├── configs/
│   ├── tiny_cpu.yaml          # CPU 可跑的最小配置
│   ├── tiny_24g.yaml          # 单卡 24GB 的主实验配置
│   └── scaling/
│       ├── s.yaml             # ≈7M
│       ├── m.yaml             # ≈17M
│       ├── l.yaml             # ≈36M
│       └── xl.yaml            # ≈64M
├── tinylm/
│   ├── __init__.py            # 公共 API 出口（唯一对外入口）
│   ├── tokenizer/
│   │   ├── base.py            # Tokenizer 协议
│   │   ├── bpe.py             # 第 2 章：字节级 BPE
│   │   └── chat.py            # 第 14 章：对话模板
│   ├── model/
│   │   ├── config.py          # ModelConfig / MoEConfig / RoPEScalingConfig
│   │   ├── norm.py            # LayerNorm、RMSNorm
│   │   ├── rope.py            # 第 4 章：RoPE 与长度外推
│   │   ├── attention.py       # MHA / MQA / GQA / MLA
│   │   ├── ffn.py             # MLP / SwiGLU
│   │   ├── moe.py             # 第 7 章：MoE 层
│   │   ├── block.py           # TransformerBlock
│   │   ├── ssm.py             # 第 8 章：可选的 Mamba 风格块
│   │   └── model.py           # TinyLM 主模型
│   ├── data/
│   │   ├── dataset.py         # 打包后的 Token 数据集
│   │   ├── mixture.py         # 第 10 章：数据配比
│   │   └── collate.py         # 第 12 章：序列打包
│   ├── train/
│   │   ├── loop.py            # Trainer
│   │   ├── optim.py           # AdamW / Muon
│   │   ├── schedule.py        # Warmup / Cosine / WSD
│   │   └── checkpoint.py
│   ├── posttrain/
│   │   ├── sft.py             # 第 14 章
│   │   ├── lora.py            # 第 15 章
│   │   ├── rm.py              # 第 16 章：奖励模型
│   │   ├── ppo.py             # 第 16 章
│   │   ├── dpo.py             # 第 17 章
│   │   ├── grpo.py            # 第 18 章
│   │   └── distill.py         # 第 19 章
│   ├── inference/
│   │   ├── sample.py          # 第 20 章：解码策略
│   │   ├── cache.py           # 第 21 章：KV Cache
│   │   ├── speculative.py     # 第 22 章
│   │   ├── quant.py           # 第 23 章
│   │   └── search.py          # 第 24 章：测试时搜索
│   ├── agent/
│   │   └── loop.py            # 第 26 章：最小工具调用循环
│   └── utils/
│       ├── shapes.py          # 形状断言
│       └── seed.py            # 可复现性
├── scripts/
│   ├── train_tokenizer.py
│   ├── train_pretrain.py
│   ├── train_sft.py
│   ├── train_dpo.py
│   ├── train_grpo.py
│   ├── generate.py
│   └── eval.py
└── tests/
    ├── test_tokenizer.py
    ├── test_shapes.py
    ├── test_gradcheck.py
    ├── test_attention_equivalence.py
    └── test_cache_equivalence.py
```

## F.2 接口契约

公共 API 只从 `tinylm/__init__.py` 导出。以下签名在全书范围内**冻结**。

### F.2.1 分词器

```python
from typing import Protocol, Sequence

class Tokenizer(Protocol):
    vocab_size: int

    def encode(self, text: str, *, add_special: bool = True) -> list[int]: ...
    def decode(self, ids: Sequence[int], *, skip_special: bool = True) -> str: ...
    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "Tokenizer": ...
```

约定：`encode` 不负责截断；`decode` 必须保证 `decode(encode(s)) == s`（对不含特殊 Token 的普通文本）。

### F.2.2 配置

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass(frozen=True)
class RoPEScalingConfig:
    method: Literal["none", "pi", "ntk", "yarn"] = "none"
    factor: float = 1.0

@dataclass(frozen=True)
class MoEConfig:
    n_experts: int = 8
    top_k: int = 2
    n_shared: int = 0
    d_ff_expert: int | None = None
    aux_loss_coef: float = 0.01

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int | None = None          # None 表示等于 n_heads
    d_ff: int | None = None                # None 表示 4 * d_model
    max_seq_len: int = 1024
    attention: Literal["mha", "mqa", "gqa", "mla"] = "mha"
    norm: Literal["layernorm", "rmsnorm"] = "layernorm"
    norm_eps: float = 1e-5
    ffn: Literal["mlp", "swiglu"] = "mlp"
    pos_encoding: Literal["learned", "rope", "nope"] = "learned"
    rope_theta: float = 10_000.0
    rope_scaling: RoPEScalingConfig = field(default_factory=RoPEScalingConfig)
    bias: bool = True
    tie_embeddings: bool = False
    dropout: float = 0.0
    moe: MoEConfig | None = None
```

{% hint style="warning" %}
`ModelConfig` 的默认值固定为**第 3 章的基线**（LayerNorm + MLP + 可学习绝对位置 + 偏置项）。第 5 章之后的"现代架构"通过 `configs/*.yaml` 切换，不修改 dataclass 默认值。这样第 3 章的实验永远可复现。
{% endhint %}

### F.2.3 模型前向

```python
@dataclass
class CausalLMOutput:
    logits: torch.Tensor | None      # (B, T, vocab)
    loss: torch.Tensor | None        # 标量
    aux_loss: torch.Tensor | None    # MoE 负载均衡损失
    past_kv: "KVCache | None"

class TinyLM(nn.Module):
    def forward(
        self,
        input_ids: torch.LongTensor,                    # (B, T)
        *,
        attention_mask: torch.Tensor | None = None,     # (B, T)
        position_ids: torch.LongTensor | None = None,   # (B, T)
        past_kv: "KVCache | None" = None,
        use_cache: bool = False,
        labels: torch.LongTensor | None = None,         # (B, T)
        logits_to_keep: int | None = None,
    ) -> CausalLMOutput: ...
```

`position_ids`、`past_kv`、`use_cache`、`logits_to_keep` 在第 4、21 章才真正生效；在此之前接受参数但忽略。`logits_to_keep` 只返回最后若干个位置的 logits，用于降低强化学习 rollout 的显存。

### F.2.4 生成

```python
@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None

@torch.no_grad()
def generate(
    self,
    input_ids: torch.LongTensor,
    *,
    max_new_tokens: int = 64,
    sampling: SamplingConfig | None = None,   # None 表示贪心
    eos_token_id: int | None = None,
    use_cache: bool = True,
) -> torch.LongTensor: ...
```

### F.2.5 冻结规则

1. 公共函数只允许**新增 keyword-only 参数且带默认值**；位置参数与返回值字段不得删除或改名。
2. 新增配置项必须有默认值，且默认值应保持之前章节的行为。
3. 所有公共张量必须使用 `(B, T, ...)` 布局；序列维永远是第 1 维。
4. 每个模块必须有不依赖数据的形状测试。
5. 随机性统一由 `tinylm.utils.seed.set_seed()` 控制。

## F.3 模型配置与规模

### F.3.1 参数量估算

对绑定词表、GQA、SwiGLU 的模型，参数量近似为：

$$
N \approx n_l \left( d_{model}^2 + 2 d_{model} d_{kv} + 2 d_{model} d_h + 3 d_{model} d_{ff} \right) + n_{vocab} d_{model}
$$

其中 $$d_{kv} = d_{model} n_{kv} / n_h$$。若 `tie_embeddings=True`，末项只计一次。

### F.3.2 三档配置

| 配置 | 用途 | $$d_{model}$$ | $$n_l$$ | $$n_h$$ | $$n_{kv}$$ | $$d_{ff}$$ | $$T$$ | $$n_{vocab}$$ | 参数量 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `tiny_cpu` | CPU 跑通全流程 | 128 | 4 | 4 | 4 | 512 | 256 | 8192 | ≈2M |
| `tiny_24g` | 单卡主实验 | 768 | 12 | 12 | 4 | 2048 | 1024 | 32000 | ≈100M |
| `scaling/s` | 缩放曲线 | 256 | 6 | 4 | 2 | 688 | 512 | 8192 | ≈7M |
| `scaling/m` | 缩放曲线 | 384 | 8 | 6 | 2 | 1024 | 512 | 8192 | ≈17M |
| `scaling/l` | 缩放曲线 | 512 | 10 | 8 | 2 | 1376 | 512 | 8192 | ≈36M |
| `scaling/xl` | 缩放曲线 | 640 | 12 | 10 | 2 | 1712 | 512 | 8192 | ≈64M |

{% hint style="info" %}
缩放实验的四个点采用近似等比的宽度增长，便于在对数坐标上拟合幂律。实际运行中如果显存紧张，可以先把 `scaling/xl` 的 `n_layers` 降到 8，再在附录中说明。
{% endhint %}

### F.3.3 显存与算力预算

训练算力的理论下界为：

$$
C \approx 6 N D
$$

在 `tiny_24g`（$$N \approx 10^8$$）上训练 10 亿 Token，需要约 $$6 \times 10^{17}$$ FLOPs。按消费级显卡有效算力 30–60 TFLOP/s 估算，纯计算时间约 **3–6 小时**；加上数据加载与评测，实际预留一个通宵更稳妥。这是**估算**，不是实测值，具体取决于显卡型号与实现效率。

`tiny_cpu` 的模型只有约 200 万参数，在几百万 Token 上训练一轮通常在**几分钟到半小时**量级，适合验证代码正确性而非追求效果。

## F.4 逐章演进路线

| 章节 | 新增/修改 | 新增配置项 | 验收测试 |
| --- | --- | --- | --- |
| 2 | `tokenizer/bpe.py` | — | 往返编解码；未知字节不崩溃 |
| 3 | `model/{norm,attention,ffn,block,model}.py` | 基线 `ModelConfig` | 输出形状；因果性；梯度可回传 |
| 4 | `model/rope.py` | `pos_encoding`、`rope_theta`、`rope_scaling` | 位置外推曲线；与手写 RoPE 数值一致 |
| 5 | `norm.py` 增 RMSNorm；`ffn.py` 增 SwiGLU | `norm`、`ffn`、`bias`、`tie_embeddings` | 同参数量下与基线对照 |
| 6 | `attention.py` 增 MQA/GQA/MLA；`kv_cache_bytes()` | `attention`、`n_kv_heads` | KV 显存公式核对；变体等价性（$$n_{kv}=n_h$$ 时退化为 MHA） |
| 7 | `model/moe.py` | `moe` | 负载均衡；容量溢出行为 |
| 8 | `model/ssm.py`（可选） | `layer_types` | 状态递推与并行扫描结果一致 |
| 9 | `train/loop.py`；`scaling/*.yaml` | `TrainConfig` | 损失单调下降；幂律拟合 |
| 10 | `data/mixture.py` | `DataConfig` | 去重率；配比采样比例 |
| 11 | `train/optim.py`、`schedule.py` | `OptimConfig` | 优化器单步与闭式解一致 |
| 12 | `data/collate.py`；多卡脚本 | `ParallelConfig` | 显存账本与实测误差小于 10% |
| 13 | RoPE 缩放预设；`eval/needle.py` | `rope_scaling` | 长度外推不失真 |
| 14 | `posttrain/sft.py`、`tokenizer/chat.py` | `SFTConfig` | 损失掩码只覆盖回答部分 |
| 15 | `posttrain/lora.py` | `LoraConfig` | 合并前后输出一致 |
| 16 | `posttrain/rm.py`、`ppo.py` | `RLConfig` | Bradley-Terry 损失单测 |
| 17 | `posttrain/dpo.py` | `DPOConfig` | 偏好对损失随间隔单调下降 |
| 18 | `posttrain/grpo.py` | `GRPOConfig` | 组内优势均值为零；KL 方向正确 |
| 19 | `posttrain/distill.py` | `DistillConfig` | 蒸馏损失随温度变化符合预期 |
| 20 | `inference/sample.py` | `SamplingConfig` | 贪心等于 `argmax` |
| 21 | `inference/cache.py` | `use_cache`、`past_kv` | 有/无缓存 logits 最大误差小于 $$10^{-4}$$ |
| 22 | `inference/speculative.py` | `SpecConfig` | 拒绝采样下分布与目标一致 |
| 23 | `inference/quant.py` | `QuantConfig` | 反量化误差在理论界内 |
| 24 | `inference/search.py` | `SearchConfig` | 预算耗尽时正确收束 |
| 26 | `agent/loop.py` | `AgentConfig` | 工具调用循环可终止 |

## F.5 复现步骤

```bash
# 1. 安装
cd tinylm
pip install -e ".[dev]"

# 2. 先跑测试（不依赖数据）
pytest tests/ -q

# 3. 训练分词器
python scripts/train_tokenizer.py --config configs/tiny_cpu.yaml

# 4. CPU 上跑通预训练
python scripts/train_pretrain.py --config configs/tiny_cpu.yaml

# 5. 生成
python scripts/generate.py --ckpt runs/cpu/last.pt --prompt "从前有一座山"

# 6. 单卡实验（需要 24GB 显存）
python scripts/train_pretrain.py --config configs/tiny_24g.yaml
```

{% hint style="warning" %}
代码会随章节逐步给出，以上命令描述的是**最终状态**。读到第 3 章时，只有分词器、模型与形状测试可用；训练脚本要到第 9 章才完整。
{% endhint %}

## F.6 验收标准

每个模块合并前必须通过：

1. **形状测试**：覆盖 `B=1`、`T=1`、`T=max_seq_len` 三个边界。
2. **数值测试**：存在闭式解的地方（如单头注意力、RMSNorm）与手写实现对比，误差小于 $$10^{-5}$$。
3. **梯度测试**：自定义算子通过 `torch.autograd.gradcheck`（float64）。
4. **等价性测试**：优化实现与朴素实现输出一致，例如分块注意力 vs 朴素注意力、有缓存 vs 无缓存。
5. **可复现测试**：固定种子连跑两次，损失曲线在小数点后 4 位一致。

{% hint style="success" %}
这五条测试是后续所有章节的地基。任何一章的代码如果让某条测试失败，优先修代码，而不是放宽测试。
{% endhint %}
