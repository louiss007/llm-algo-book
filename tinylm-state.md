# TinyLM 状态

本文件记录 TinyLM 的模块、接口与配置变更，供分会话写作交接。接口契约以[附录 F](appendix/f-tinylm.md) 为准。

## 当前状态

- **公共 API**：`tinylm/tokenizer/bpe.py` 的核心实现已在第 2 章给出，接口与附录 F 的 `Tokenizer` 协议一致；但代码目前以章节内嵌形式存在，尚未落盘到包目录。
- **模型**：第 3 章的 Transformer 基线已在章节内给出完整实现，接口与附录 F 对齐；同样尚未落盘。
- **位置编码**：第 4 章在基线上增加 `sine`、`rope`、`nope`、`alibi` 四种 `pos_encoding`，并接入 PI/NTK/YaRN 三种 RoPE 缩放；代码仍以章节内嵌形式存在。
- **现代架构**：第 5 章增加 `RMSNorm`、`SwiGLU`、无偏置、权重绑定与可选的并行块（`parallel_block`），并给出 `exact_params` 与 `flops_per_token` 两个账本函数；代码仍以章节内嵌形式存在。
- **注意力变体**：第 6 章增加 `VariantAttention`（MHA/MQA/GQA）与 `LatentAttention`（MLA），由 `attention`、`n_kv_heads`、`d_c`、`d_head_rope` 分派；新增 `attention_params` 与 `kv_cache_bytes` 两个账本函数；代码仍以章节内嵌形式存在。
- **混合专家**：第 7 章增加 `MoEConfig` 与 `MoE`（路由器、Top-K 分派、容量丢弃、共享专家、辅助损失），`TransformerBlock` 的前馈子层在稠密 FFN 与 MoE 之间分派；新增 `model_params(active=...)`、`moe_layer_params`、`expert_params` 账本函数；`CausalLMOutput.aux_loss` 开始生效；代码仍以章节内嵌形式存在。
- **替代架构**：第 8 章增加 `block_type`（`attention` / `linear_attention` / `ssm` / `hybrid`）、`ssm_d_state`、`ssm_expand`、`hybrid_attn_every`；新增 `SelectiveSSM`、`LinearAttention`、`scan_states`、`SSMBlock`、`LinearAttentionBlock` 与 `make_block`/`layer_kinds` 分派；新增 `ssm_layer_params`、`state_bytes`、`decode_flops` 账本函数；默认 `attention` 与第 7 章逐位一致；代码仍以章节内嵌形式存在。
- **数据工程**：第 10 章增加 `tinylm/data/filter.py`（分词、精确去重、shingle、Jaccard、MinHash、LSH、近似去重、质量过滤、污染检测与去污染）与 `tinylm/data/mixture.py`（温度采样、最大余数预算、按配额交错）；不修改模型模块。代码仍以章节内嵌形式存在。
- **训练优化**：第 11 章增加 `tinylm/train/optim.py`（`AdamW`、`make_param_groups`、`lr_factor`、`clip_grad_norm_`、`LossSpikeDetector`、Newton-Schulz 正交化与 `muon_update`）；优化器属于训练脚本选择，不进入 `ModelConfig`，模型接口未改动。代码仍以章节内嵌形式存在。
- **分布式训练**：第 12 章增加 `tinylm/train/parallel.py`（`ParallelConfig`、显存与通信账本、CPU 集合通信模拟、列/行并行原语与 `LossScaler`）；并行度、全局批与混合精度属于训练脚本参数，不进入 `ModelConfig`，模型接口未改动。代码仍以章节内嵌形式存在。
- **长上下文**：第 13 章增加 `tinylm/data/collate.py`（`PackedBatch`、块对角因果掩码、文档打包与 Token 利用率）、`tinylm/train/long_context.py`（算力/显存账本、长度课程、Token 预算与有效上下文）与 `tinylm/eval/needle.py`（单针检索构造、正确性与深度分桶）；注意力新增 4D 块对角掩码分支，2D padding 掩码与 `TinyLM.forward` 不变。代码仍以章节内嵌形式存在。
- **监督微调**：第 14 章增加 `tinylm/tokenizer/chat.py`（`ChatTemplate`、`encode_conversation`、`count_supervised`）与 `tinylm/posttrain/sft.py`（`SFTConfig`、掩码交叉熵、Token/序列平均、批拼接、`pack_sft_examples`、监督统计、标签噪声与拒绝采样公式）；复用第 13 章的 `block_causal_mask`；模型接口未改动。代码仍以章节内嵌形式存在。
- **参数高效微调**：第 15 章增加 `tinylm/posttrain/lora.py`（`LoraConfig`、`LoRALinear`、`apply_lora`、`lora_parameters`/`lora_state_dict`/`load_lora_state_dict`、`trainable_stats`、`lora_module_params`、`state_bytes`、NF4 量化/反量化与 `quantize_lora_bases`）；适配器通过替换线性层注入，不修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig`，未启用 LoRA 时默认路径与第 14 章逐位一致。代码仍以章节内嵌形式存在。
- **对齐训练**：第 16 章增加 `tinylm/posttrain/rm.py`（`ScalarModel`、`init_scalar_from_lm`、`bradley_terry_loss`、`pairwise_accuracy`、`preference_probability`、`reward_standardizer`）与 `tinylm/posttrain/ppo.py`（`RLConfig`、`Rollout`、`sample_responses`、`response_mask`、`token_logprobs`、`token_values`、`compute_gae`、`normalize_advantage`、`clipped_surrogate`、`kl_penalty`、`kl_estimator_k3`、`collect_rollout`、`ppo_update`）；奖励模型通过替换 `lm_head` 注入，PPO 配置只作训练脚本参数，不修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig`。代码仍以章节内嵌形式存在。
- **偏好优化**：第 17 章增加 `tinylm/posttrain/dpo.py`（`DPOConfig`、`completion_logprobs`、`log_ratio_margin`、`implicit_reward`、`dpo_loss`、`preference_accuracy`）；损失同时支持 DPO、cDPO 标签平滑与 IPO，参考模型为独立的冻结副本，配置只作训练脚本参数，不进入 `ModelConfig`。代码仍以章节内嵌形式存在。
- **接口契约**：未修改。`TinyLM.forward` 保持附录 F 的签名，新增字段只出现在 `ModelConfig` 与 `CausalLMOutput` 中。
- **配置**：`ModelConfig` 首次落地（见第 3 章变更）。`pos_encoding` 与 `rope_theta` 在第 4 章启用，新增 `rope_scaling`；第 6 章启用 `attention`、`n_kv_heads`、`d_c`、`d_head_rope`；第 7 章启用 `moe` 与 `MoEConfig`（含 `capacity_factor`）；部分字段仍为后续章节预留。

## 第 1 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `examples/ch01_lm_basics.py` | 独立示例 | n-gram、神经 n-gram、RNN 三种基线；不进入 `tinylm/` 包 |

该示例的定位：

- 建立交叉熵、困惑度、教师强制的计算流程；
- 第 3 章的 Transformer 会复用同样的损失计算与评测口径；
- 第 20 章的解码实验会复用其中的采样循环。

## 第 2 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/tokenizer/bpe.py` | 章节内嵌实现 | 字节级 BPE；训练、编解码、特殊 Token、JSON 保存/加载 |
| `tests/test_tokenizer.py` | 待补 | 往返编解码、任意 UTF-8、特殊 Token、保存/加载一致性 |
| `scripts/train_tokenizer.py` | 待补 | 从语料训练词表并保存为 JSON |

实现要点：

- 基础词表为 256 个字节，经 GPT-2 的 `bytes_to_unicode` 映射为可打印字符；
- 训练时反复合并最高频相邻对，频次低于 2 时提前停止；
- 编码时按合并 rank 从小到大贪心合并；
- 特殊 Token 在普通合并结束后追加，不参与 BPE 合并；
- 词表、合并序列、特殊 Token 一起保存，加载后编码结果不变。

## 第 3 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/config.py` | 章节内嵌实现（`ModelConfig`） | 冻结默认值；提供后续章节所需的开关 |
| `tinylm/model.py` | 章节内嵌实现 | `TinyLM`、`TransformerBlock`、`MultiHeadAttention`、`MLP`、`LayerNorm` |
| `tests/test_model.py` | 章节内嵌实现 | 形状、因果性、参数计数、前向/反向，6 个用例 |
| `examples/ch03_copy_task.py` | 章节内嵌实现 | 周期复制任务，含理论与实测对比 |

`ModelConfig` 字段：

| 字段 | 默认值 | 当前实现是否生效 |
| --- | --- | --- |
| `vocab_size` | 必填 | 是 |
| `d_model` | 512 | 是 |
| `n_layers` | 8 | 是 |
| `n_heads` | 8 | 是 |
| `n_kv_heads` | `None`（等于 `n_heads`） | 预留，第 6 章启用 |
| `d_ff` | `None`（等于 `4 * d_model`） | 是 |
| `max_seq_len` | 1024 | 是 |
| `attention` | `"mha"` | 仅 `"mha"`，第 6 章扩展 |
| `norm` | `"layernorm"` | 仅 `"layernorm"`，第 5 章扩展 |
| `norm_eps` | `1e-5` | 是 |
| `ffn` | `"mlp"` | 仅 `"mlp"`，第 5 章扩展 |
| `pos_encoding` | `"learned"` | 是；第 4 章起分派 `learned`/`sine`/`rope`/`nope`/`alibi` |
| `rope_theta` | `10000.0` | 是（仅 `rope` 模式）；第 4 章启用 |
| `bias` | `True` | 是 |
| `tie_embeddings` | `False` | 是 |
| `dropout` | `0.0` | 预留，暂不生效（小于 0.1 时对收敛无实质影响） |

实现要点：

- 结构为 Pre-LN：`x = x + Attn(LN(x))`，`x = x + MLP(LN(x))`，末端再过一个 `ln_f`；
- 注意力为多头，Q/K/V 各自独立投影，缩放因子 `1/sqrt(d_head)`，因果掩码以加性 `-inf` 注册为 buffer；
- 前馈为两层线性 + GELU，默认 `d_ff = 4 d_model`；
- 位置信息用可学习的绝对位置嵌入表 `P ∈ R^{max_seq_len × d_model}`；
- 嵌入与输出层不共享权重（`tie_embeddings=False`），与附录 F 的基线配置一致；
- 初始化：除残差输出投影按 `0.02 / sqrt(2 n_layers)` 缩放外，其余线性层与嵌入为标准正态 `std=0.02`；
- `forward` 接受 `attention_mask`、`position_ids`、`past_kv`、`use_cache`、`labels`、`logits_to_keep`；当前实现前两者与 `labels`，后三者（`past_kv`、`use_cache`、`logits_to_keep`）为后续章节预留；
- 损失按 Token 平均，`ignore_index=-100`，与第 1、2 章口径一致。

## 第 4 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/position.py` | 章节内嵌实现 | `sinusoidal_positions`、`rope_inv_freq`、`RoPE`、`alibi_slopes`、`ALiBi` |
| `tinylm/model.py` | 章节内嵌实现 | `PositionAwareAttention` 按 `pos_encoding` 分派；超表长时动态生成因果掩码 |
| `tinylm/config.py` | 章节内嵌实现 | 新增 `RoPEScalingConfig`；`ModelConfig` 增加 `rope_scaling` |
| `tests/test_position.py` | 章节内嵌实现 | 9 个用例：正弦平移、RoPE 相对性与保范数、缩放形状、ALiBi、超表长、越界、因果性 |
| `examples/ch04_length_extrapolation.py` | 章节内嵌实现 | 复制任务上的外推曲线与 RoPE 波长表 |

配置字段变化：

| 字段 | 默认值 | 第 4 章行为 |
| --- | --- | --- |
| `pos_encoding` | `"learned"` | 接受 `"learned"`、`"sine"`、`"rope"`、`"nope"`、`"alibi"` |
| `rope_theta` | `10000.0` | RoPE 基频 `b`，仅 `pos_encoding="rope"` 时生效 |
| `rope_scaling` | `RoPEScalingConfig(method="none", factor=1.0)` | `method` 接受 `"none"`、`"pi"`、`"ntk"`、`"yarn"` |

实现要点：

- `learned` 与 `sine` 把位置向量加到 Token 嵌入上，序列长度受 `max_seq_len` 限制；
- `rope` 在注意力内部旋转 Q/K，不修改嵌入；采用**分半式**约定（前一半与后一半配对），与 GPT-NeoX、Llama 一致；
- `alibi` 在打分上加 `-ν_h(i-j)`，斜率为固定缓冲区，不参与训练；
- `nope` 不注入任何位置信息，仅保留因果掩码；
- 注意力在 `T > max_seq_len` 时动态生成因果掩码，使 RoPE 可测试超表长序列；`learned` 与 `sine` 仍会断言失败；
- `position_ids` 从 `TinyLM.forward` 透传到每个注意力层；KV Cache 接入时复用同一路径；
- YaRN 的温度缩放通过把打分除以 `sqrt(t_y)` 实现，`t_y = 0.1 ln(factor) + 1`。

## 第 5 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/norm.py` | 章节内嵌实现 | `RMSNorm`；`make_norm` 按 `norm` 分派 |
| `tinylm/ffn.py` | 章节内嵌实现 | `SwiGLU` 三投影；`make_ffn` 按 `ffn` 分派 |
| `tinylm/block.py` | 章节内嵌实现 | `TransformerBlock` 支持 `parallel_block`：`x + Attn(LN(x)) + MLP(LN(x))` |
| `tinylm/model.py` | 章节内嵌实现 | 启用 `bias`、`tie_embeddings`；权重绑定在初始化后执行 |
| `tinylm/config.py` | 章节内嵌实现 | `ModelConfig` 新增 `parallel_block`；`d_ff=None` 时按 `ffn` 取 4 或 8/3 倍 |
| `tests/test_modern.py` | 章节内嵌实现 | 15 个用例：RMSNorm 公式/尺度不变/零均值等价/平移敏感、SwiGLU、绑定、精确参数、等预算 FLOPs、并行层、因果性 |
| `examples/ch05_ablation.py` | 章节内嵌实现 | 参数与算力账本、Llama 2 7B 复现、Llama 2→3 增量拆解、RMSNorm 行为演示、复制任务消融 |

配置字段变化：

| 字段 | 默认值 | 第 5 章行为 |
| --- | --- | --- |
| `norm` | `"layernorm"` | 接受 `"layernorm"`、`"rmsnorm"` |
| `ffn` | `"mlp"` | 接受 `"mlp"`、`"swiglu"` |
| `bias` | `True` | 设为 `False` 时所有线性层与 LayerNorm 去偏置 |
| `tie_embeddings` | `False` | 为 `True` 时 `lm_head.weight` 与 `tok_emb.weight` 共享存储 |
| `parallel_block` | `False` | 为 `True` 时注意力与前馈共享 `norm1`，并行相加 |
| `d_ff` | `None` | `None` 时 MLP 取 `4 d_model`，SwiGLU 取 `round(8 d_model / 3)` |

实现要点：

- `RMSNorm` 只除以均方根 `sqrt(mean(x²)+eps)`，不含均值中心化与偏置，参数只有 `gamma`；
- `SwiGLU` 的三个投影命名固定为 `w_gate`、`w_up`、`w_down`，前向为 `w_down(silu(w_gate(x)) * w_up(x))`；
- 权重绑定在 `apply(_init_weights)` 之后执行，避免共享矩阵被初始化两次；
- 残差输出投影（`out_proj.weight`、`fc_out.weight`、`w_down.weight`）按 `0.02/sqrt(2 n_layers)` 缩放，与第 3 章一致；
- `exact_params` 与 `flops_per_token` 是设计阶段账本，不属于 `TinyLM` 的公共接口；
- `TinyLM.forward` 签名未变，第 4 章所有 `pos_encoding` 分支仍可用。

## 第 6 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/attention.py` | 章节内嵌实现 | `VariantAttention`（MHA/MQA/GQA）、`LatentAttention`（MLA）、`make_attention` 分派 |
| `tinylm/model.py` | 章节内嵌实现 | `TransformerBlock` 与 `TinyLM` 改用 `make_attention`；`position_ids` 与因果掩码路径不变 |
| `tinylm/config.py` | 章节内嵌实现 | `ModelConfig` 新增 `attention`、`n_kv_heads`、`d_c`、`d_head_rope`；`__post_init__` 校验变体与整除 |
| `tests/test_attention_variants.py` | 章节内嵌实现 | 21 个用例：形状、MHA 退化一致性、参数/缓存公式、因果性、整除约束、吸收恒等式、TinyLM 前向 |
| `examples/ch06_kv_cache.py`（章内为 `ch06.py` 的 `ledger`/`ablation`） | 章节内嵌实现 | 变体账本、工业配置缓存、吸收等价性、复制任务消融 |

配置字段变化：

| 字段 | 默认值 | 第 6 章行为 |
| --- | --- | --- |
| `attention` | `"mha"` | 接受 `"mha"`、`"mqa"`、`"gqa"`、`"mla"` |
| `n_kv_heads` | `None` | `None` 时 MHA 取 `n_heads`、MQA 取 1；GQA 必须显式给出，且须整除 `n_heads` |
| `d_c` | 128 | MLA 的潜向量维度；需小于等于 `d_model` |
| `d_head_rope` | 32 | MLA 解耦 RoPE 的位置键维度 |

实现要点：

- `VariantAttention` 的 Q/输出投影始终是 `d_model → d_model`；K/V 投影是 `d_model → n_kv_heads * d_head`，`d_head = d_model / n_heads` 保持不变；
- RoPE 在复制 KV 之前施加，即先旋转 `n_kv` 份键，再 `repeat_interleave` 到 `n_heads`；这样缓存里存的是已旋转的键；
- `LatentAttention` 缓存的是潜向量 `c_t`（`d_c` 维）与共享位置键（`d_head_rope` 维），不缓存每头 K/V；
- 解耦 RoPE 的打分是内容分与位置分之和，位置键在所有头之间共享；
- `attention_params` 与 `kv_cache_bytes` 是账本函数，不属于 `TinyLM` 公共接口；MLA 的参数量含 Q RoPE 投影与共享 `k_rope` 投影；
- `TinyLM.forward` 签名未变，第 4、5 章所有配置仍可用；`n_kv_heads=None` 时 MHA 行为与第 5 章逐位一致。

## 第 7 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/config.py` | 章节内嵌实现 | 新增 `MoEConfig`；`ModelConfig` 增加 `moe` 字段，默认 `None` 保持稠密前馈 |
| `tinylm/moe.py` | 章节内嵌实现 | `Expert`、`MoE`、`load_balance_loss`、`make_ffn_or_moe`；路由、Top-K、容量、共享专家与辅助损失 |
| `tinylm/block.py` | 章节内嵌实现 | `TransformerBlock` 的前馈子层改用 `make_ffn_or_moe`；前向返回 `(x, aux)` |
| `tinylm/model.py` | 章节内嵌实现 | 汇总各层 `aux_loss` 到 `CausalLMOutput.aux_loss`；其余路径不变 |
| `tests/test_moe.py` | 章节内嵌实现 | 16 个用例：稠密退化、参数公式、路由、容量、辅助损失、因果性、前向 |
| `examples/ch07_moe.py`（章内为 `ch07.py` 的 `ledger`/`capacity_demo`/`ablation`） | 章节内嵌实现 | Mixtral/DeepSeek-V3 账本、容量演示、复制任务消融 |

配置字段变化（新增 `MoEConfig`）：

| 字段 | 默认值 | 第 7 章行为 |
| --- | --- | --- |
| `n_experts` | `8` | 普通专家的个数，必须 ≥ 1 |
| `top_k` | `2` | 每 Token 激活的专家数，必须满足 `1 <= top_k <= n_experts` |
| `n_shared` | `0` | 始终激活的共享专家数 |
| `d_ff_expert` | `None` | `None` 时沿用 `ModelConfig.d_ff` |
| `capacity_factor` | `1.25` | 每专家容量系数；`None` 表示不丢弃 |
| `aux_loss_coef` | `0.01` | 辅助损失系数；`0` 表示不计算 |

实现要点：

- 路由器是无偏置线性层，输出 `(B*T, E)` logits，softmax 后取 Top-K，再对选中的概率做截断归一化；
- 容量按 `ceil(CF * n_tokens * top_k / n_experts)` 计算；分派按 Token 位置先到先得，用 `cumsum` 实现，超出的分派计入 `last_dropped`；
- 分派用"逐专家掩码 + 索引"实现，每个专家只对分到它的 Token 做一次矩阵乘；生产实现应换成排序加分组 GEMM；
- 共享专家在路由之后对全部 Token 相加，不参与容量限制；
- 辅助损失只在 `self.training` 且 `aux_loss_coef > 0` 时计算，评估时 `aux_loss=None`；训练总损失由调用方使用 `loss + aux_loss`；
- `last_load`、`last_dropped` 是诊断属性，不进入 `state_dict`，不改变冻结接口；
- `moe=None` 时 `make_ffn_or_moe` 返回第 5 章的稠密 FFN，模型与第 6 章逐位一致（测试覆盖）。

## 本章固定的约定

1. 损失统一按 Token 平均，单位 nat；报告困惑度时同时给出 `exp(loss)`。
2. 评测集与训练集按 80/20 切分，随机种子固定。
3. 所有概率计算使用自然对数，比特数由 `loss / math.log(2)` 换算。
4. 采样使用全局 `random` 模块并固定种子，保证示例输出可复现。
5. 分词器的基础单元是 UTF-8 字节，不做 Unicode 规范化；`decode(encode(s)) == s` 必须严格成立。
6. 实际词表大小以训练结束后的 `len(vocab)` 为准，必须写入模型配置的 `vocab_size`。
7. 残差流宽度恒为 `d_model`，所有子层输出必须投影回 `d_model` 才能相加。
8. 因果掩码的语义统一为：允许处为 0，禁止处为 `-inf`；实现里用有限大负数只为数值稳妥，不改变语义。
9. 参数量报告同时给出精确公式值、简化估算值与 `sum(p.numel())` 实测值，三者差异要能解释。
10. 任何改动模型结构的实验，都必须先跑因果性测试（改第 10 个位置的输入，检查前 10 个位置的 logits 不变）。
11. `logits_to_keep` 在训练与预填充时忽略，只在第 18 章接入 KV cache 后生效。

## 第 8 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/config.py` | 章节内嵌实现 | `ModelConfig` 继承第 7 章，新增 `block_type`、`ssm_d_state`、`ssm_expand`、`hybrid_attn_every` |
| `tinylm/ssm.py` | 章节内嵌实现 | `scan_states`、`SelectiveSSM`、`LinearAttention` |
| `tinylm/block.py` | 章节内嵌实现 | `SSMBlock`、`LinearAttentionBlock`、`make_block`、`layer_kinds` |
| `tinylm/model.py` | 章节内嵌实现 | `TinyLM` 的模块列表改为 `make_block(cfg, i)` 分派，其余结构与第 7 章相同 |
| `tests/test_ssm.py` | 章节内嵌实现 | 22 个用例：默认等价、扫描/二次等价、因果性、状态大小、参数与状态账本 |
| `examples/ch08_alt_arch.py` | 本章以 `ch08.py` 的 `ledger`、`long_context_demo`、`ablation` 代替 | 状态/算力账本与复制任务消融 |

配置字段变化：

| 字段 | 默认值 | 第 8 章行为 |
| --- | --- | --- |
| `block_type` | `"attention"` | 接受 `"attention"`、`"linear_attention"`、`"ssm"`、`"hybrid"`；默认保持第 7 章逐位一致 |
| `ssm_d_state` | `16` | 每个内部通道的状态维度 $$N$$；仅 SSM 相关块使用 |
| `ssm_expand` | `2` | 内部宽度 $$I=e\,d_{model}$$ 的扩展倍数 |
| `hybrid_attn_every` | `4` | 混合架构中每隔多少层放一个注意力层，即 (8.20) 的 $$m$$ |

实现要点：

- `make_block` 按 `block_type` 返回 `TransformerBlock`、`SSMBlock`、`LinearAttentionBlock` 或混合序列；`layer_kinds` 是账本使用的同构函数，必须与 `make_block` 的分派一致；
- `SSMBlock` 与 `TransformerBlock` 同构：`norm1 + mixer + norm2 + FFN`，只在 `mixer` 处不同；`LinearAttentionBlock` 继承 `SSMBlock` 并替换 `mixer`，两者都返回 `(x, aux)` 以兼容第 7 章的 MoE 辅助损失；
- `SelectiveSSM` 的 $$\mathbf{A}=-\exp(\mathbf{A}_{\log})$$ 保证负实部；步长投影 `dt_proj` 始终带偏置，参数量按 (8.16) 计算；
- 深度因果卷积用移位切片手写，避免 CPU 上 `Conv1d` 反向的开销；生产实现应换成专用内核；
- `scan_states` 是 Hillis–Steele 扫描，复合算子见 (8.17)；要求 $$|a|\le1$$ 才能稳定；
- `last_state` 保存最后位置的状态并 `detach`，是诊断属性，不进入 `state_dict`；
- 线性注意力直接物化 $$\mathbf{S}_t$$（`cumsum`），只适合教学规模；生产实现必须分块；
- 账本函数全部解析计算，不为大配置实例化模型；`state_bytes` 与 `decode_flops` 区分注意力层与递推层。

## 第 9 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/scaling.py` | 章节内嵌实现 | `power_law_loss`、`optimal_exponents`、`compute_optimal`、`fit_scaling_law`、`effective_data`、`training_flops` |
| `tests/test_scaling.py` | 章节内嵌实现 | 9 个用例：幂律与算力手算、指数和为一、预算约束、平衡条件、合成数据参数恢复、有效数据饱和、损失随预算下降 |
| `examples/ch09_scaling.py`（章内为 `ch09.py`） | 章节内嵌实现 | 合成数据拟合、计算最优分配表、重复数据表、复制任务扫描、训练加推理账本 |

实现要点：

- 本章不新增模型模块，`ModelConfig`、`TinyLM`、`CausalLMOutput` 均未改动；`ch09.py` 直接 `from ch08 import ModelConfig, TinyLM, decode_flops, model_params`。
- `power_law_loss` 实现 (9.1)；`training_flops` 实现 (9.2)；`optimal_exponents` 返回 (9.5) 的两个指数；`compute_optimal` 在 $$\log_{10}N$$ 上做 200 轮黄金分割，约束由 $$D=C/(6N)$$ 自动满足。
- `fit_scaling_law` 在 log 损失上做 Huber 回归；$$L_\infty$$ 用 softplus、$$A,B$$ 用 exp、$$\alpha,\beta$$ 用 sigmoid 保证取值合法；默认 3 次重启，返回最优 RMSE 与参数。
- `effective_data` 实现 (9.8)，`R_STAR=15.4`；$$D\le U$$ 时直接返回 $$D$$，避免负的重复轮数。
- 训练加推理账本复用第 8 章的 `decode_flops` 与 `model_params`，没有另起一套 FLOPs 口径；玩具扫描复用第 8 章的复制任务（lag=8、T=32、V=16）。
- 账本函数全部解析计算，不为大配置实例化模型；70B/1.4T 的 $$g_0$$ 由 `scaling_allocation` 的默认参数 1.858 校准。

## 第 10 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/data/filter.py` | 章节内嵌实现 | `tokenize`、`exact_dedup`、`shingles`、`jaccard`、`minhash_signature`、`lsh_candidates`、`near_dedup`、`quality_score`、`quality_filter`、`contamination_hits`、`decontaminate` |
| `tinylm/data/mixture.py` | 章节内嵌实现 | `temperature_probs`、`allocate_counts`、`interleave` |
| `tests/test_data.py` | 章节内嵌实现 | 13 个用例：分词、精确去重、Jaccard 边界、MinHash 估计、近似去重、LSH 分桶、质量过滤、污染检测、去污染、温度采样、预算分配、交错配额、非法温度 |
| `examples/ch10_data.py`（章内为 `ch10.py`） | 章节内嵌实现 | 合成语料构造、漏斗、污染检测、3 种子训练对照、配比表与配比消融 |

实现要点：

- 本章不新增模型模块，`ModelConfig`、`TinyLM`、`CausalLMOutput` 均未改动；`ch10.py` 直接 `from ch08 import ModelConfig, TinyLM`。
- `tokenize` 是教学用混合分词：英文与代码按词，中文按空白分隔的连续汉字串，标点单独成 Token；它不是第 2 章的 BPE，只用于数据管道的集合运算与实验编码。
- `exact_dedup` 对规范化 Token 序列做 BLAKE2b-128 哈希，保留首次出现；`near_dedup` 先用 MinHash 签名与 LSH 分桶找候选，再用精确 Jaccard 复核，保留每簇第一个文档。
- `minhash_signature` 用 $$m$$ 个随机仿射哈希 $$(ax+b)\bmod(2^{61}-1)$$，空集合返回全 $$2^{61}-1$$；`lsh_candidates` 要求 `bands * rows <= num_perm`。
- `quality_score` 实现 (10.6)，对短文档与符号噪声加惩罚；`quality_filter` 默认 $$\tau=0.55$$。
- `contamination_hits` 计算每个评测文档被训练集 n-gram 覆盖的比例；`decontaminate` 反向计算每篇训练文档与评测集的覆盖率并删除超阈值文档，默认 $$n=8$$、阈值 0.5。
- `temperature_probs` 在 log 空间计算并减最大值，温度非正时报错；`allocate_counts` 用最大余数法保证计数之和等于预算；`interleave` 域内先打乱、按配额轮流取样，配额超过文档数时循环复用。
- 实验的清洗顺序固定为：质量过滤 → 精确去重 → 近似去重 → 防污染；统计口径是"本阶段删除"为相对上一级的文档数差。
- 训练对照统一用 `d_model=32`、2 层、4 头、GQA-2、RMSNorm、SwiGLU、RoPE、150 步、批 16、序列 32、3 个种子；配比消融用 2 个种子、120 步。
- 账本与漏斗均为纯函数，不依赖 GPU；所有随机源固定种子，输出可逐字复现。

## 第 11 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/train/optim.py` | 章节内嵌实现 | `AdamW`、`make_param_groups`、`lr_factor`、`clip_grad_norm_`、`LossSpikeDetector`、`zeropower_newtonschulz5`、`muon_update` |
| `tests/test_optim.py` | 章节内嵌实现 | 13 个用例：AdamW 闭式解、偏差修正、解耦 vs 耦合、参数分组、预热端点、余弦单调、WSD 平台与衰减、非法调度、全局裁剪、小梯度不变、尖峰检测、Newton-Schulz 条件数、Muon 二次下降 |
| `examples/ch11_optim.py`（章内为 `ch11.py`） | 章节内嵌实现 | 病态二次型对照、优化器/预热/调度 3 种子实验、注入梯度裁剪演示 |

实现要点：

- 本章不新增模型模块，`ModelConfig`、`TinyLM`、`CausalLMOutput` 均未改动；`ch11.py` 直接 `from ch08 import ModelConfig, TinyLM`，并复用第 10 章的语料、编码与评测函数。
- `AdamW` 继承 `torch.optim.Optimizer`，实现 (11.3)–(11.7)：一阶/二阶矩用 `mul_` + `add_`/`addcmul_` 原地更新，偏差修正后再做 `(1 - lr * wd)` 的解耦收缩。
- `make_param_groups` 只对 `ndim >= 2` 的参数施加权重衰减，偏置与归一化增益进入 `weight_decay=0` 组；嵌入层默认在衰减组，正文说明这一选择没有理论定论。
- `lr_factor(step, total, warmup, kind, min_ratio, decay_start, power)` 中 `step` 表示"已完成的更新次数"，取值 0..total；预热结束处乘子恰为 1。`kind` 支持 `constant`、`cosine`、`wsd`，未知调度抛 `ValueError`。
- `clip_grad_norm_` 把全部参数梯度视为一个向量，按全局 L2 范数缩放，返回裁剪前的范数；`max_norm <= 0` 报错。
- `LossSpikeDetector` 用长度为 `window` 的滑动窗口计算均值与标准差，先判定当前点再入队，避免尖峰抬高自己的基线；历史不足 `min_history` 时不报警。
- `zeropower_newtonschulz5` 只接受二维张量，按 (11.20) 做 5 次迭代，形状为"宽矩阵"时先转置以减小中间矩阵；`muon_update` 按 (11.19) 返回新动量，并按 `sqrt(max(1, n_out/n_in))` 缩放更新。
- 优化器状态、参数分组与调度都属于训练侧，不写入模型 `state_dict` 之外的结构；`TinyLM.forward` 签名未变。
- 训练实验统一用 `d_model=32`、2 层、4 头 GQA-2、RMSNorm、SwiGLU、RoPE、批 16、序列 32、3 个种子；裁剪演示用 2 维二次目标并人为在第 10 步把梯度放大 50 倍。
- 所有随机源（模型初始化、批采样）固定种子；输出中的耗时随机器波动，其余数字逐字可复现。

## 第 12 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/train/parallel.py` | 章节内嵌实现 | `PRECISION_BYTES`、`ParallelConfig`、`param_state_bytes`、`zero_state_bytes`、`memory_ledger`、`ring_all_reduce_bytes`、`dp_grad_bytes`、`tp_block_bytes`、`pp_bubble_fraction`、`ep_all_to_all_bytes`、`global_batch_size`、`scale_lr`；集合通信模拟 `all_reduce_sum`、`reduce_scatter_sum`、`all_gather_shards`、`shard_flat`、`merge_shards`、`shard_reduce_gather`；张量并行原语 `column_parallel_linear`、`row_parallel_linear`；`LossScaler` |
| `tests/test_parallel.py` | 章节内嵌实现 | 16 个用例：每参数状态字节、ZeRO 7B 四阶段与单调性、激活账本与重算、环形通信量、DP/TP 字节、气泡、全局批与学习率、集合通信往返与 padding、列/行并行等价、TP MLP 等价、缩放器溢出/增长/反缩放、分片往返、分片 AdamW 等价、非法配置 |
| `examples/ch12_parallel.py` | 章节内嵌实现 | 整批/梯度累积/DP/ZeRO-1 四模式 3 种子等价性、7B 显存账本、通信账本、精度表、损失缩放演示 |

配置与状态：

| 字段/属性 | 默认值 | 第 12 章行为 |
| --- | --- | --- |
| `ParallelConfig.dp` | `1` | 数据并行组大小，也是 ZeRO 分片组大小 |
| `ParallelConfig.tp` | `1` | 张量并行度；`d_model % tp == 0` |
| `ParallelConfig.pp` | `1` | 流水并行段数；`n_layers % pp == 0` |
| `ParallelConfig.sp` | `1` | 序列并行度，只允许 1 或等于 `tp` |
| `ParallelConfig.zero_stage` | `0` | 0/1/2/3，越界报错 |
| `ParallelConfig.micro_batch` | `1` | 每卡微批大小 |
| `ParallelConfig.grad_accum` | `1` | 梯度累积步数，也是流水线在途微批数 |
| `global_batch` | 属性 | `micro_batch * grad_accum * dp`，对应 (12.7) |
| `world_size` | 属性 | `dp * tp * pp` |

实现要点：

- 本章不新增模型模块，`ModelConfig`、`TinyLM`、`CausalLMOutput` 均未改动；`ch12_parallel.py` 复用第 8 章的 `ModelConfig`/`TinyLM`、第 10 章的语料与评测、第 11 章的 `AdamW` 与数据构造。
- `PRECISION_BYTES` 给出权重、梯度、主权重、单个 Adam 矩与激活的每参数字节；FP32/TF32 的 `master=0`，FP8 的权重与激活为 1 字节、梯度仍为 2 字节。
- `param_state_bytes` 实现 (12.1)；`zero_state_bytes` 实现 (12.4)，其中 `dp` 是数据并行组大小，不是总卡数；阶段 1/2/3 分别分片优化器状态、梯度、参数。
- `memory_ledger` 实现 (12.2)、(12.3)：激活按 `n_layers/pp` 与 `seq_len/sp` 缩放，默认系数 12；`recompute=True` 时除以 `sqrt(layers)`；只合计状态与激活，临时缓冲不在内。
- `ring_all_reduce_bytes` 实现 (12.5) 的 `2(N-1)/N` 系数；`dp_grad_bytes` 与 `tp_block_bytes` 分别按梯度张量与激活张量代入。
- 集合通信用"每 rank 一个张量"的列表模拟：`all_reduce_sum` 返回每卡相同的总和；`reduce_scatter_sum` 让 rank `r` 只保留总和的第 `r` 段；`shard_reduce_gather` 先散射再收集，并正确处理不能整除时的 padding；`shard_flat` 要求元素数能被 world 整除。
- `column_parallel_linear` 按输出维切分、前向无通信；`row_parallel_linear` 按输入维切分、用一次 all-reduce 求和；测试验证两者拼接/组合与稠密矩阵乘等价。
- `LossScaler` 实现 (12.12)：`scale_loss` 放大损失，`unscale_` 除回并返回是否全部有限；`update(False)` 时 scale 减半、`good_steps` 归零、记录跳过；连续 `growth_interval` 步正常后倍增。
- 实验统一用 `d_model=32`、2 层、4 头 GQA-2、序列 32、批 16、50 步、学习率 2e-3、3 个种子；整批训练是基准，其余三种按全局平均归一化后比对。
- 所有随机源（模型初始化、批采样）固定种子；输出中的耗时随机器波动，其余数字逐字可复现。
- 本章只执行数据并行与 ZeRO-1 的等价性；张量并行与流水并行验证原语与公式，不模拟完整层间调度；集合通信不测量真实带宽。

## 第 13 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/data/collate.py` | 章节内嵌实现 | `PackedBatch`、`block_causal_mask`、`pack_documents`、`token_utilization` |
| `tinylm/train/long_context.py` | 章节内嵌实现 | `attention_flops`、`linear_flops`、`attention_share`、`crossover_length`、`kv_cache_bytes`、`activation_bytes`、`length_schedule`、`token_budget_shares`、`effective_context` |
| `tinylm/eval/needle.py` | 章节内嵌实现 | `needle_positions`、`passkey_batch`、`retrieval_correct`、`bucket_accuracy` |
| `tests/test_long_context.py` | 章节内嵌实现 | 11 个用例：FLOPs 解析式与翻倍关系、交叉长度占比、KV/激活账本、课程表与有效上下文、块对角掩码、打包边界、打包等价与泄漏、2D 掩码兼容、Token 利用率、针位置、分桶统计 |
| `examples/ch13_long_context.py` | 章节内嵌实现 | 7B 账本、KV 表、打包等价性、NIAH 自检、固定/混合/课程三种长度策略的外推表 |

配置与状态：

| 字段/属性 | 默认值 | 第 13 章行为 |
| --- | --- | --- |
| `ModelConfig.max_seq_len` | 与第 3 章一致 | RoPE 路径允许前向长度超过它；注意力按当前长度动态重建因果掩码 |
| `ModelConfig.rope_scaling` | `None` | 复用第 4 章的 PI/NTK/YaRN 配置，本章实验用它做零样本外推对照 |
| `attention_mask`（前向参数） | `None` | 新增 4D 布尔掩码支持：`(B,1,T,T)` 按块对角文档掩码处理；2D `(B,T)` padding 掩码行为不变 |

实现要点：

- 本章不新增模型模块，`ModelConfig`、`TinyLM`、`CausalLMOutput` 均未改动；只在第 4 章 `PositionAwareAttention` 与第 6 章 `VariantAttention`/`LatentAttention` 中把 `attention_mask.dim() == 4` 的分支接上 `masked_fill`。
- `pack_documents` 把多篇文档拼成一条序列：输入是拼接后的前 `T-1` 个 Token，标签是后 `T-1` 个 Token；跨文档位置的标签置为 `-100`；`position_ids` 在每篇文档内从 0 开始；`attention_mask` 是 `(1,1,T,T)` 块对角因果掩码。
- `block_causal_mask` 实现 (13.8)：同文档且不越界；`token_utilization` 实现 (13.7)，返回 `{"pad", "packed"}`。
- `attention_flops`、`linear_flops` 实现 (13.1)、(13.2)；`crossover_length` 实现 (13.3)；`kv_cache_bytes` 与第 6 章 (6.6) 同形，本章按 (13.4) 复用；`activation_bytes` 实现 (13.5)，`materialize_scores=True` 时增加分数矩阵项。
- `length_schedule` 实现 (13.6) 的几何课程；`token_budget_shares` 按 $$1/L_i$$ 归一化；`effective_context` 实现 (13.10)。
- `needle_positions` 实现 (13.11)：末两位留给 value 与 QUERY，KEY 的位置范围是 `[0, T-3]`；`passkey_batch` 在指定位置放 KEY、下一位放随机 value、末位放 QUERY；`bucket_accuracy` 按深度分桶。
- 实验统一用 2 层、`d_model=32`、2 个头、`d_ff=128`、词表 17、400 步、学习率 3e-3、3 个种子；固定长度训练的 `max_seq_len=32`，混合长度与课程训练的 `max_seq_len=128`。
- 全部随机源固定种子；输出中的耗时随机器波动，其余数字逐字可复现。
- 本章的零样本外推结论只适用于这个玩具任务；正文明确说明不能外推到真实 LLM，PI/NTK/YaRN 的种子间双峰现象按原始结果报告。

## 第 14 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/tokenizer/chat.py` | 章节内嵌实现 | `ChatTemplate`（角色标记、EOS、`answer_only` 开关）、`encode_conversation`、`count_supervised` |
| `tinylm/posttrain/sft.py` | 章节内嵌实现 | `SFTConfig`、`masked_cross_entropy`、`token_mean_loss`/`sequence_mean_loss`、`collate_examples`、`PackedSFTBatch`/`pack_sft_examples`、`supervision_stats`、`label_noise_mixture`、拒绝采样三公式 |
| `tests/test_sft.py` | 章节内嵌实现 | 14 个用例：答案掩码、全序列监督、多轮监督、监督比例、掩码交叉熵、两种平均、padding、打包边界与位置、超长报错、打包等价、监督统计、均匀噪声、拒绝采样公式与模拟 |
| `examples/ch14_sft.py` | 章节内嵌实现 | 掩码对照、数据质量与数量、打包等价性、拒绝采样四组实验 |

配置与状态：

| 字段/属性 | 默认值 | 第 14 章行为 |
| --- | --- | --- |
| `SFTConfig.answer_only` | `True` | 只监督 assistant 内容与 EOS；`False` 时用户内容也进入损失 |
| `SFTConfig.lr` / `epochs` / `batch_size` | `2e-5` / `3.0` / `8` | 真实全参数微调的脚本默认值；玩具实验另用 `3e-3` 与 250–600 步 |
| `SFTConfig.replay_ratio` | `0.0` | 预留的回放比例字段，本章只给公式 (14.8)，实验未启用 |
| `TinyLM.forward` / `CausalLMOutput` | 不变 | 仍通过 `labels=` 触发交叉熵；掩码由 `-100` 表达 |

实现要点：

- 模板渲染返回与输入等长的 `labels`：`labels[:-1]` 由右移一位的输入按掩码填充，最后一位固定 `-100`；这样 EOS 是监督目标，打包时也不会丢边界信息。
- 掩码交叉熵实现 (14.1)；`token_mean_loss`/`sequence_mean_loss` 实现 (14.2)；`supervision_stats` 与 `count_supervised` 实现 (14.3)；`label_noise_mixture` 实现 (14.4)；`pack_sft_examples` 的边界处理对应 (14.6)；拒绝采样三函数对应 (14.7)。
- 打包复用第 13 章的 `block_causal_mask`；`position_ids` 在每条对话内从 0 重置；跨对话边界标签置 `-100`。
- 实验统一用 2 层、`d_model=32`、2 个头、`d_ff=128`、词表 16、3 个种子；掩码对照 250 步、质量对照 600 步；ID 评测长度 3–5，长度 OOD 为 7–8，模板 OOD 使用训练中未出现的角色标记 13/14；随机水平 0.125。
- 所有随机源固定种子；输出中的耗时随机器波动，其余数字逐字可复现。
- 正文明确标注 2 层 32 维、250–600 步的玩具结论不可外推；遗忘只给公式与工业经验，未单独跑实验。

## 第 15 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/posttrain/lora.py` | 章节内嵌实现 | `LoraConfig`；`LoRALinear`（冻结基座、`lora_A`/`lora_B`、`alpha/r` 缩放、`merge`/`unmerge`）；`apply_lora`、`lora_parameters`、`lora_state_dict`/`load_lora_state_dict`、`trainable_stats`、`lora_module_params`、`state_bytes`；NF4 码本、`nf4_quantize`/`nf4_dequantize`、`nf4_storage_bytes`、`quantize_lora_bases` |
| `tests/test_lora.py` | 章节内嵌实现 | 10 个用例：初始输出恒等、合并/撤销等价、仅适配器可训练、零初始化 B 的第一步梯度、适配器状态往返、目标模块选择、NF4 往返误差、存储比特数、QLoRA 冻结基座、适配器键校验 |
| `examples/ch15_peft.py` | 章节内嵌实现 | 「加二」预训练 + 「加一」SFT；全参数/LoRA 秩与目标模块/QLoRA/随机基座对照；静态状态与激活测量；合并等价性；NF4 误差与存储 |

配置与状态：

| 字段/属性 | 默认值 | 第 15 章行为 |
| --- | --- | --- |
| `LoraConfig.r` / `alpha` | `8` / `16` | 缩放为 `alpha/r`；实验中令 `alpha=2r` 以固定有效步长 |
| `LoraConfig.dropout` | `0.0` | 作用在低秩分支的输入上；合并前应关闭或切到推理模式 |
| `LoraConfig.target_modules` | 9 个线性层后缀 | 按完整名或末段名匹配；`lm_head` 不在默认目标内 |
| `LoraConfig.bias` | `"none"` | 可选 `all`/`lora_only`；默认只训练 `lora_A`/`lora_B` |
| `TinyLM.forward` / `CausalLMOutput` | 不变 | 适配器通过替换 `nn.Linear` 注入；未调用 `apply_lora` 的路径与第 14 章逐位一致 |
| `state_bytes` | fp32 记账 | 可训练 16 字节/参数、冻结 4 字节/参数；`_nf4` 存在时被量化部分按 4 比特记账 |

实现要点：

- `LoRALinear` 冻结 `base`，`lora_A` 用 Kaiming 初始化、`lora_B` 置零，因此初始前向与基座逐位相同；第一次反向时 `lora_A` 梯度为零，`lora_B` 先更新。
- `merge` 把 `(lora_B @ lora_A) * scaling` 加进基座权重并缓存 `_delta`，`unmerge` 减去同一增量；合并前后 logits 偏差约 `2e-6`。
- `apply_lora` 先冻结全部参数，再按目标名替换线性层；重复调用不会重复包装，`lm_head` 与嵌入层保持冻结。
- `quantize_lora_bases` 只做「NF4 量化—反量化」的数值模拟，并在 `model._nf4` 记录量化元素数与解析字节数；它不把张量真的压成 4 比特，正文已明确这一点。
- 实验用 5 个种子；预训练 1,200 步、SFT 600 步；全参数学习率 `3e-3`，LoRA/QLoRA 为 `1e-2`；评测 200 条长度 3–5。
- 所有随机源固定种子；输出中的耗时随机器波动，其余数字逐字可复现；正文明确标注玩具结论不可外推。

## 第 16 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/posttrain/rm.py` | 章节内嵌实现 | `ScalarModel`（复用 `TinyLM` 主干，把 `lm_head` 换成 `d_model → 1` 的标量头，`score` 取最后一个有效位置）；`init_scalar_from_lm`（从 SFT 权重初始化，丢弃 `lm_head.*`）；`bradley_terry_loss`、`preference_probability`、`pairwise_accuracy`、`reward_standardizer` |
| `tinylm/posttrain/ppo.py` | 章节内嵌实现 | `RLConfig`（冻结 dataclass，校验 `beta/clip/epochs/max_new`）；`Rollout`；`sample_responses`、`response_mask`、`token_logprobs`、`token_values`、`compute_gae`、`normalize_advantage`、`clipped_surrogate`、`kl_penalty`、`kl_estimator_k3`、`collect_rollout`、`ppo_update` |
| `tests/test_rlhf.py` | 章节内嵌实现 | 13 个用例：BT 零差值损失、损失随差值单调、偏好概率对称、标量模型形状与末位打分、从 LM 初始化保留主干、GAE 手算、GAE 忽略 padding、裁剪三分支、同策略比值为一、KL 估计量同模型为零、采样尊重期限并在 EOS 后补 padding、rollout 形状有限、`RLConfig` 校验 |
| `examples/ch16_rlhf.py` | 章节内嵌实现 | 4 主题玩具对话：SFT 冷启动 → 奖励模型训练 → PPO；三档 β 与 Best-of-8 对照；长度/正确性/干扰词三个奖励作弊探针；3 个种子 |

配置与状态：

| 字段/属性 | 默认值 | 第 16 章行为 |
| --- | --- | --- |
| `ScalarModel.backbone` | 复用 `TinyLM(cfg)` | 除 `lm_head` 外与输入 LM 的权重逐位一致；要求 `cfg.tie_embeddings == False` |
| `ScalarModel.forward` | 输出 `(B, T)` | 每个位置一个标量；`score` 用 `attention_mask.sum(-1)-1` 取末位 |
| `RLConfig.beta` | `0.5` | KL 正则系数；`0` 表示不约束，实验中取 0.10/0.50/2.00 |
| `RLConfig.clip` | `0.2` | PPO 裁剪半径 ε，必须满足 `0 < clip < 1` |
| `RLConfig.gamma` / `lam` | `1.0` / `0.95` | 语言模型回合取 γ=1；λ 控制 GAE 偏差—方差 |
| `RLConfig.epochs` / `minibatch` | `2` / `16` | 固定 rollout 上重复更新；策略与价值用独立优化器 |
| `RLConfig.max_new` | `6` | 响应最多生成的 Token 数（含 EOS）；无 EOS 时按有限期回合截断 |
| `RLConfig.normalize_reward` | `True` | 批内标准化 RM 分数；不改变最优策略，只稳定价值回归 |
| `TinyLM.forward` / `CausalLMOutput` / `ModelConfig` | 不变 | 奖励与价值通过替换 `lm_head` 注入；PPO 不进入模型配置 |

实现要点：

- `sample_responses` 不做词表截断、不强制 EOS、不做最小长度约束，保证「采样分布 = 计算 log 概率的分布」；EOS 之后的 Token 一律写成 `pad_id`。
- `token_logprobs` 返回对齐到「预测下一个 Token」的 log 概率，第 0 位与 padding 位填 0；`token_values` 用同样对齐，价值对应 `V(s_{t-1})`。
- `compute_gae` 从后往前扫，padding 与序列末尾的下一状态价值置零；`normalize_advantage` 只在 `mask` 上做零均值、单位方差。
- `collect_rollout` 先算 `-β(logπ−logπ_ref)` 的逐 Token 塑形奖励，再用 `scatter_add_` 在最后一个有效位置加 RM 分数；回报由 `adv + values` 得到并 detach。
- `ppo_update` 每个 minibatch 重新计算新策略 log 概率与价值，重要性比由 `logp - batch.logp` 指数化；策略损失按有效 Token 平均，价值损失为 MSE，两者分别反向、分别裁剪梯度、分别 step。
- 奖励模型与价值网络都从 SFT 权重初始化：`init_scalar_from_lm` 只丢 `lm_head.weight`，其余键严格加载；价值网络在实验中直接载入 RM 权重。
- 实验 3 个种子；RM 2,000 对训练、400 对评测、400 步；PPO 50 轮、批 48；评测 256 条；所有随机源固定种子，除耗时外逐字可复现。

## 第 17 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/posttrain/dpo.py` | 章节内嵌实现 | `DPOConfig`（冻结 dataclass，校验 `beta/lr/epochs/batch_size/loss_type/label_smoothing/max_grad_norm`）；`completion_logprobs`（左移掩码，支持长度归一化）；`log_ratio_margin`、`implicit_reward`、`dpo_loss`（DPO / cDPO / IPO）、`preference_accuracy` |
| `tests/test_dpo.py` | 章节内嵌实现 | 12 个用例：零间隔损失为 `log(2)`、损失随间隔单调下降、共同平移不变、标签平滑提高置信对损失、IPO 在 `1/(2β)` 处为零、隐式奖励尺度、梯度方向与对称、序列对数概率对齐手算、padding 不改变结果、长度归一化除以响应长度、平局计为失败、`DPOConfig` 校验 |
| `examples/ch17_dpo.py` | 章节内嵌实现 | 复用第 16 章玩具对话与 SFT 流程；偏置/去偏两套偏好数据；β=0.10/0.50/2.00、IPO β=0.50、去偏 DPO β=0.50 五组对照；3 个种子；报告任务正确率、干扰词、长度、逐 Token KL 与留出偏好对排序准确率/间隔 |

配置与状态：

| 字段/属性 | 默认值 | 第 17 章行为 |
| --- | --- | --- |
| `DPOConfig.beta` | `0.1` | 隐式奖励温度；实验中取 0.10/0.50/2.00，必须大于 0 |
| `DPOConfig.lr` | `5e-5` | DPO 学习率；实验中取 `5e-4`；与 β 耦合，报告时必须同时给出 |
| `DPOConfig.epochs` / `batch_size` | `1` / `8` | 固定离线偏好对上的训练轮数与批大小；实验为 300 步、批 32 |
| `DPOConfig.loss_type` | `"dpo"` | `dpo` 或 `ipo`；IPO 使用平方损失与目标间隔 `1/(2β)` |
| `DPOConfig.label_smoothing` | `0.0` | cDPO 平滑系数，必须满足 `0 ≤ ε < 0.5` |
| `DPOConfig.length_normalize` | `False` | 按响应 Token 数平均对数概率；改变 β 的单位 |
| `DPOConfig.max_grad_norm` | `1.0` | 梯度裁剪阈值，必须大于 0 |
| `TinyLM.forward` / `CausalLMOutput` / `ModelConfig` | 不变 | 参考模型为独立冻结副本；DPO 配置只作训练脚本参数 |

实现要点：

- `completion_logprobs` 用 `log_softmax` 加 `gather`，掩码左移一位（位置 `t` 的 logits 预测 `t+1`），并把 padding 与提示位置置零；不得先取 `exp` 再取对数。
- `dpo_loss` 先算不含 β 的对数比差，再乘 β 得到 logit；DPO 走 `logsigmoid`，cDPO 用 `logsigmoid(m)` 与 `logsigmoid(-m)` 的加权，IPO 直接对 `(Δ − 1/(2β))²` 求均值。
- 参考模型与策略严格分离：参考模型 `requires_grad_(False)` 并 `eval()`；示例中参考对数概率本可预计算，但为了与教学代码一致，训练循环内实时计算。
- 示例的偏好对按提示成对组织；评测回报任务正确率、干扰词比例、内容长度与逐 Token KL，避免把“偏好准确率”误当成任务质量。
- 实验 3 个种子；SFT 512 条、400 步；偏好对 2,000 条、评测 400 条；DPO 300 步、批 32；评测采样 256 条；所有随机源固定种子，除耗时外逐字可复现。

## 下一章待实现

第 18 章《可验证奖励强化学习与推理模型》在第 16–17 章之上展开，重点是 RLVR 与 GRPO。待定内容：

- 用规则可验证的奖励（数学答案、代码单测、工具返回）替代人工偏好与奖励模型；
- GRPO 的组内采样、组均值基线、优势标准化与去掉价值网络的推导；
- 与 PPO 的 (16.7)–(16.9)、DPO 的 (17.4) 对照：数据从哪来、奖励是否可验证、是否需要参考模型；
- 采样温度、组大小、长度惩罚与“全对/全错组”零梯度问题；
- 与第 15 章 LoRA、第 13 章长上下文、第 21 章推理引擎的接口关系。

第 18 章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有签名；RLVR/GRPO 的配置放在训练脚本或 `tinylm/posttrain/` 中，具体模块名与文件名在开工前按 CLAUDE.md、SUMMARY.md 与附录 F 确认。

## 验收门槛

- `BPETokenizer` 满足 `Tokenizer` 协议；
- `decode(encode(s)) == s` 对普通文本与任意 UTF-8 字节序列成立；
- 特殊 Token 不会被普通合并拆开；
- 保存/加载后的分词结果完全一致；
- 模型配置中的 `vocab_size` 与实际词表大小一致；
- 第 3 章的嵌入矩阵与输出层维度均能由 `vocab_size` 正确推导；
- 因果性测试通过：位置 $$t$$ 的 logits 只依赖前 $$t$$ 个 Token；
- 注意力权重每行和为 1，且上三角严格为 0；
- 精确参数公式 (3.22) 与实测参数量逐位一致；
- 切换位置编码实现时，`TinyLM.forward` 签名不得改变；
- RoPE 保范数：旋转前后 Q、K 的逐 Token 范数误差小于 `1e-5`；
- RoPE 相对性：同一对 Q/K 平移相同距离后，打分矩阵对应元素误差小于 `1e-4`；
- 位置显式由 `position_ids` 传入，缓存与打包序列复用同一路径；
- `rope="pi"` 只缩放位置，`"ntk"` 改基频，`"yarn"` 分频缩放并加温度；三者的频率表均与公式逐项一致；
- 任何位置编码改动都必须通过因果性测试：改动位置 `t` 的输入，`t` 之前的 logits 不变；
- RMSNorm 在零均值、无偏置输入上与 LayerNorm 的输出误差小于 `1e-5`；
- RMSNorm 尺度不变：输入放大 10 倍后输出误差小于 `1e-4`；
- SwiGLU 无偏置时参数量等于 `3 d_model d_ff`；绑定权重后 `lm_head.weight` 与 `tok_emb.weight` 指向同一存储；
- `exact_params` 在基线与现代配置的 6 种组合下与 `sum(p.numel())` 逐位一致；
- `d_ff=8d_model/3` 时 SwiGLU 与 `4d_model` MLP 的每 Token FLOPs 相对差小于 1%；
- 并行块输出等于 `x + Attn(LN(x)) + MLP(LN(x))`，误差小于 `1e-6`；
- 现代配置（RMSNorm + SwiGLU + 无偏置 + RoPE）仍通过因果性测试；
- `n_kv_heads=n_heads` 时 `VariantAttention` 与第 4 章 `PositionAwareAttention` 输出误差小于 `1e-6`；
- GQA/MQA 的 `d_head` 与 MHA 相同，只有 K/V 头数减少；K/V 投影宽度等于 `n_kv_heads * d_head`；
- `attention_params` 与 `make_attention(cfg).parameters()` 的实测值逐位一致，MLA 与 GQA 分别对应 (6.14) 与 (6.5)；
- `kv_cache_bytes` 与 (6.6)、(6.11) 逐位一致；同样配置下 MLA < GQA < MHA；
- `n_kv_heads` 不整除 `n_heads`（GQA）必须在 `ModelConfig.__post_init__` 中报错；
- MLA 权重吸收的内容通路输出误差小于 `1e-5`（FP32 随机权重）；
- MHA/MQA/GQA/MLA 四种变体均通过因果性测试：改动位置 `t` 的输入，`t` 之前的输出不变。
- `moe=None` 时 `TinyLM` 与第 6 章在相同权重下 logits 误差小于 `1e-6`，且 `aux_loss=None`；
- `model_params(cfg)` 与 `sum(p.numel())` 在含共享专家的 MoE 配置上逐位一致；`model_params(cfg, active=True)` 与 (7.13) 一致；
- Top-K 门控每行之和为 1（误差小于 `1e-6`）；所有专家权重相同时，MoE 输出等于单个专家；
- 容量因子生效时，每个专家接收的 Token 数不超过 `ceil(CF * N_tok * k / E)`，且接收数之和加丢弃数等于 `N_tok * k`；
- `capacity_factor=None` 时 `last_dropped == 0`，接收数之和等于 `N_tok * k`；
- 辅助损失在负载均匀时等于 `alpha`，失衡时大于 `alpha`；只在训练且系数大于 0 时返回，评估时为 `None`；
- `top_k > n_experts` 必须在 `MoEConfig.__post_init__` 中报错；
- MoE 层通过因果性测试：改动位置 `t` 的输入，`t` 之前的 logits 不变；
- `aux_loss` 不为空时，训练总损失采用 `loss + aux_loss`，且 `CausalLMOutput` 的其余字段语义不变。
- `block_type="attention"` 时 `TinyLM` 与第 7 章在相同权重下 logits 误差小于 `1e-6`，且 `aux_loss=None`；
- `LinearAttention` 的扫描形式与显式二次形式（带因果掩码）误差小于 `1e-5`；
- `scan_states` 与逐步递推在 T=1、2、17、32 上误差小于 `1e-6`；
- `SelectiveSSM.last_state` 形状为 `(B, expand*d_model, d_state)`，与序列长度无关；
- `ssm_layer_params(cfg)` 与 `SelectiveSSM(cfg)` 的 `sum(p.numel())` 逐位一致；
- `model_params(cfg)` 与 `sum(p.numel())` 在四种 `block_type` 上逐位一致；
- SSM、线性注意力与混合块均通过因果性测试：改动位置 `t` 的输入，`t` 之前的 logits 不变；
- T=512 时输出与状态全部有限，状态最大绝对值小于 `1e3`；
- `hybrid_attn_every=4`、`n_layers=4` 时 `layer_kinds(cfg) == ["ssm","ssm","ssm","attn"]`，且最后一个块是第 7 章的 `TransformerBlock`；
- `state_bytes` 对 SSM/线性注意力与序列长度无关，对注意力随长度线性增长；同配置下 SSM 状态小于线性注意力；
- `block_type` 不在四个合法值内时必须在 `ModelConfig.__post_init__` 中报错；
- 全零输入经过线性注意力不得产生 NaN（分母裁剪生效）。
- `power_law_loss` 在给定参数上与手算值逐位一致；`training_flops(N,D)` 等于 $$6ND$$；
- `optimal_exponents(0.34,0.28)` 返回 $$(0.28/0.62,0.34/0.62)$$，两指数之和为 1；
- `scaling_allocation(C)` 对任意 $$C$$ 满足 $$6ND=C$$（相对误差小于 `1e-12`）；
- `compute_optimal` 的解满足平衡条件 (9.4)（左右比值误差小于 `1e-3`），且损失随预算单调下降；
- 在 35 个带 1% 噪声的合成点上，`fit_scaling_law` 恢复 $$L_\infty$$ 的相对误差小于 5%、$$\alpha,\beta$$ 的绝对误差小于 0.05，log-RMSE 小于 0.02；
- `effective_data` 在 $$D\le U$$ 时等于 $$D$$，在 $$D\to\infty$$ 时趋于 $$U(1+R^*)$$，且对 $$D$$ 单调递增；
- 训练加推理账本必须复用第 8 章的 `decode_flops`，不得另写解码 FLOPs 公式；8B/15T 与 70B/1.4T 的盈亏平衡点由两者训练算力差与每 Token 解码差解出；
- 玩具扫描只允许用于验证实现方向，不得用其 9 个点拟合 (9.1) 的五个参数；正文必须说明算力跨度与贴近下界的限制。
- `exact_dedup` 保留首次出现，重复输入的第二份进入 `dropped`；规范化只做空白拼接，不做大小写或 Unicode 折叠；
- `jaccard` 在两边均为空集时返回 1.0；`minhash_signature` 对空集合返回全 `_MERSENNE`；`lsh_candidates` 对相同签名必须返回 `(0,1)`；
- `bands=8`、`rows=8` 时 `bands * rows = 64`，与默认 `num_perm=64` 一致；`bands * rows > num_perm` 必须触发断言；
- `near_dedup` 在长文档单字符改动上必须召回，在完全不同的文档上不得误删；阈值与 shingle 大小必须作为参数暴露；
- `quality_filter` 必须拒绝高重复、过短、纯符号噪声三类文档，保留正常文档；阈值默认 0.55，可通过参数覆盖；
- `contamination_hits` 对逐字泄漏返回 1.0，对干净文档返回小于 0.5；`decontaminate` 必须删除泄漏文档而保留干净文档；
- `temperature_probs` 对任意正温度返回和为 1 的分布，温度升高时分布更平坦，温度非正时抛出 `ValueError`；
- `allocate_counts` 的计数之和必须等于预算，且每个计数非负；`interleave` 必须满足各域配额，配额超过文档数时循环复用；
- 数据漏斗必须同时报告每一级的文档数与 Token 数，并给出清洗前后的总量与保留率；不得只报清洗后的损失；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有字段；实验只读取第 8 章的 `ModelConfig` 与 `TinyLM`；
- `ch10.py` 的输出必须可复现：固定语料种子、模型种子与批采样种子，重复运行逐字一致；
- 训练对照至少报告 3 个种子的均值与标准差；配比消融至少报告 2 个种子；
- 合成语料与玩具模型的结论必须标注不可外推；污染检测与清洗收益的相互作用不得被表述为普适结论。
- `AdamW` 第一步与闭式解一致：参数初值 1、梯度 2、`lr=0.1`、`betas=(0.9,0.95)`、`eps=0`、`weight_decay=0.1` 时，更新后参数为 `0.99 - 0.1*2/sqrt(4)`；
- 偏差修正使第一步更新幅度约等于学习率：梯度 `1e-3`、`lr=0.05`、无衰减时，参数变化约为 `-0.05`；
- 解耦衰减不被二阶矩放大：相同梯度与衰减系数下，AdamW 的收缩幅度与 `lr*wd` 一致，耦合 L2 的衰减约为其 `1/sqrt(v)` 倍；
- `make_param_groups` 只把二维以上参数放进衰减组，偏置与归一化增益进入不衰减组；
- `lr_factor` 的端点正确：预热起点为 0、中点为 0.5、预热结束为 1、末步为 `min_ratio`；余弦在预热后单调不增；WSD 在 `decay_start` 处平直、末步为 `min_ratio`；
- 未知 `kind` 必须抛出 `ValueError`；`total <= 0` 或 `step < 0` 必须报错；
- `clip_grad_norm_` 按全局 L2 范数缩放：两参数梯度为 `(3,0)` 与 `(4)` 时，返回 5.0，裁剪后范数为 1.0；小梯度不得被改动；
- `LossSpikeDetector` 在历史不足 `min_history` 时不报警；稳定序列后注入 3 倍均值必须触发尖峰，且随后的正常值不报警；
- `zeropower_newtonschulz5` 只接受二维张量；5 次迭代后条件数必须显著下降，谱范数接近 1（不要求精确正交）；
- `muon_update` 必须在二次目标上降低损失，返回与参数同形状的新动量；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig`；实验只读取第 8 章的 `ModelConfig` 与 `TinyLM`；
- `ch11.py` 的所有数字除耗时外必须逐字可复现；训练实验必须报告 3 个种子的均值与标准差；
- 玩具模型（2 层、`d_model=32`、150–300 步）的结论必须标注不可外推；Muon 的玩具实验若不占优，不得删去负面结果后只报正面结论。
- `param_state_bytes` 对 FP32/FP16/BF16 返回 16、对 FP8 返回 15；未知精度必须报错；
- `zero_state_bytes(7e9, 8, s, "bf16")` 在阶段 0/1/2/3 分别为 112.0/38.5/26.25/14.0 GB，且随阶段单调不增；阶段越界必须报错；
- `memory_ledger` 的合计等于状态加激活；激活按 `n_layers/pp`、`seq_len/sp`、微批与 `d_model` 缩放；`recompute=True` 时激活严格变小；
- `ring_all_reduce_bytes(S,1)==0`、`(100,4)==150`、`(100,64)==196.875`；
- `dp_grad_bytes` 与 `tp_block_bytes` 与 (12.5)、(12.9) 逐位一致；`pp_bubble_fraction(1,A)==0`，`(4,8)==3/11`，段数或微批非正时报错；
- `global_batch_size(4,2,8)==64`；`scale_lr` 在 `kappa=1` 时线性、`kappa=0.5` 时平方根；`kappa` 越界报错；
- `all_reduce_sum` 在每个 rank 返回相同的总和；`shard_reduce_gather` 在元素数不能被卡数整除时，padding 不得污染结果；
- `shard_flat` 要求元素数能被 world 整除，否则报错；`merge_shards` 能拼回原扁平张量；
- 列并行输出拼接后与稠密矩阵乘误差小于 `1e-6`；行并行与稠密矩阵乘误差小于 `1e-6`；列/行组合的 MLP 与稠密 MLP 误差小于 `1e-6`；
- 分片 AdamW 与稠密 AdamW 在相同梯度下逐元素误差小于 `1e-6`；
- `LossScaler` 连续 `growth_interval` 步正常后倍增；检测到非有限梯度时减半并跳过，`good_steps` 归零、`skipped_steps` 加一；`unscale_` 在梯度除回后能识别 `inf`；
- 四模式（整批、梯度累积、DP、ZeRO-1）在 3 个种子上验证损失一致、最大参数偏差小于 `1e-4`；每卡优化器状态比例约为 `1/dp`（含 padding）；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig`；并行度与混合精度只作为训练脚本参数；
- `ch12_parallel.py` 除耗时外必须逐字可复现；玩具模型的等价性结论不得外推为真实带宽或吞吐结论。
- `attention_flops`、`linear_flops` 与 (13.1)、(13.2) 逐位一致；`F_attn(2T)/F_attn(T)` 等于 `2(2T+1)/(T+1)`，`F_lin` 严格翻倍；
- `crossover_length` 实现 (13.3)；在该长度上 `attention_share` 等于 0.5，半长小于 0.5、倍长大于 0.5；
- `kv_cache_bytes(1, 131072, 32, 8, 128)` 等于 1.7179869184e10 字节（16 GiB），批大小线性放大；
- `activation_bytes` 实现 (13.5)；`materialize_scores=True` 时恰好多出 `B*n_layers*n_heads*T^2*b` 字节；
- `length_schedule(32,128,3)` 返回 `[32,64,128]` 且步数占比之和为 1；`token_budget_shares` 之和为 1 且随长度递减；`effective_context` 返回达标的最长长度，全不达标返回 0；
- `block_causal_mask` 的形状为 `(1,1,T,T)`，只在对角块的下三角为真；
- `pack_documents` 的 `input_ids` 与 `labels` 长度均为 `sum(len(doc))-1`；跨文档标签为 `-100`；位置在每篇文档内从 0 重置；总长超过 `max_len` 必须报错；
- 打包 + 块对角掩码的 logits、逐位置损失与逐条前向在 `1e-5` 内一致；仅保留因果掩码时必须出现可测偏差，证明文档隔离不是可选项；
- 2D padding 掩码“全 1”时与原前向逐位一致，`TinyLM.forward` 的签名与 2D 语义不得改变；
- `token_utilization` 在 6 篇文档、长度 9/12/7/15/11/8、`max_len=32` 时返回 padding 62/192、packed 62/64；
- `needle_positions(T,n)` 的首尾为 0 与 `T-3`、去重且有序；`T<3` 或 `n_probes<1` 必须报错；`passkey_batch` 在 KEY 位置写入 `key_id`、在末位写入 `query_id`，越界必须报错；
- `bucket_accuracy` 的样本数之和等于输入数，分桶边界左闭右开；`retrieval_correct` 只比较末位 argmax 与目标；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有字段；4D 掩码只作为 `attention_mask` 的新分支；
- `ch13_long_context.py` 除耗时外必须逐字可复现；零样本外推与课程/混合长度的结论必须标注为 2 层 32 维、400 步玩具实验，不得外推为真实 LLM 的普适结论。
- `ChatTemplate().render` 对 `user=[5,6]`、`assistant=[7,8]` 返回 `input_ids == [1,2,5,6,3,7,8,4]`、`labels == [-100,-100,-100,-100,7,8,4,-100]`，监督数 3；
- `answer_only=False` 时同一对话的监督数为 5；多轮对话中每个 assistant 内容与 EOS 都被监督，第二个 assistant 内容位于第 8 个 Token 之后；
- `count_supervised` 返回 `(3, 8, 0.375)`；`supervision_stats` 的 `ratio` 与逐条统计一致；
- `masked_cross_entropy` 与手写 `F.cross_entropy(..., ignore_index=-100, reduction="none")` 的掩码均值在浮点误差内一致；
- `token_mean_loss` 与 `sequence_mean_loss` 在长短样本混合时不等：给定逐 Token 损失 `[[1,1,9],[1,-100,-100]]`，前者为 3.0、后者为 `7/3`；
- `collate_examples` 补齐后 padding 位置的标签为 `-100`、注意力掩码为假，未 padding 样本的掩码全真；
- `pack_sft_examples` 的 `labels[n-1] == -100`（跨对话边界），`doc_ids` 为两段常量，`position_ids` 在每条对话内从 0 重置，`supervised` 等于两条对话监督数之和；
- 总长超过 `max_len` 必须报 `ValueError`，不得静默截断；
- 打包与逐条前向的 logits 与逐位置损失在 `1e-6` 内一致（TinyLM 随机初始化，三条对话）；
- `label_noise_mixture` 保持分布归一化与 argmax，且满足 (14.4) 的逐元素值；
- `acceptance_probability` 越界报错；`pass_at_n(0.25,8) == 1-0.75^8`；`expected_draws(0.25,100) == 400`；`pass_at_n(0, n) == 0`；
- 4,000 次独立采样模拟的 pass@8 与解析值相差小于 0.03；
- `ch14_sft.py` 除耗时外必须逐字可复现；掩码、噪声、打包与拒绝采样的结论必须标注为 2 层 32 维、250–600 步玩具实验，不得外推为真实 LLM 的 SFT 结论；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有字段；`SFTConfig` 只作为训练脚本参数。
- `LoRALinear` 在 `lora_B=0` 时的前向与基座逐位相同（`torch.equal`）；未调用 `apply_lora` 的模型路径不得改变；
- `merge` 后的 logits 与未合并前误差小于 `1e-6`，`unmerge` 后恢复；连续两次 `merge` 必须幂等；
- `apply_lora` 后只有 `lora_A`/`lora_B` 的 `requires_grad` 为真（默认 `bias="none"`）；`trainable_stats()["trainable"]` 等于 `lora_module_params` 的求和；
- 第一次反向时 `lora_B.grad` 非零而 `lora_A.grad` 全零，证明零初始化 B 的梯度阻断；
- `lora_state_dict`/`load_lora_state_dict` 往返后输出完全一致；缺键或多键必须报 `KeyError`；
- `LoraConfig` 的 `r>=1`、`alpha>0`、`0<=dropout<1`、`bias` 取值必须校验；
- `nf4_quantize` 的码字在 0–15，反量化形状与输入一致；随机正态权重的相对误差小于 0.2；
- `nf4_storage_bytes` 在 `b=64`、双重量化时约 4.127 比特/参数，不做双重量化时约 4.5；1,048,576 个元素的字节数与公式一致；
- `quantize_lora_bases` 后基座权重 `requires_grad=False`、适配器仍可训练，且 `model._nf4` 记录的元素数与目标线性层一致；
- `state_bytes` 对全参数、LoRA q/v、LoRA 全线性、QLoRA 的合计与正文表一致（530/149/225/113 KB，允许 1 KB 舍入）；
- `saved_activation_bytes` 的测量只用于相对比较，不得当作峰值显存；正文必须说明 CPU 分配器与张量复用会影响绝对值；
- `ch15_peft.py` 除耗时外必须逐字可复现；LoRA 秩/目标模块、QLoRA 与随机基座的结论必须标注为 2 层 32 维、600 步、5 种子玩具实验，适配器占比 14.8% 不得外推为真实模型比例；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有字段；`LoraConfig` 只作为适配器配置，QLoRA 的 NF4 模拟不得被表述为真实显存节省。
- `bradley_terry_loss` 在奖励差为零时返回 `log(2)`，奖励差增大时单调下降；`preference_probability(a,b) + preference_probability(b,a) == 1`；
- `ScalarModel` 要求 `cfg.tie_embeddings == False`；`forward` 返回 `(B, T)`，`score` 返回 `(B,)` 且等于每个样本最后一个有效位置的标量；
- `init_scalar_from_lm` 只丢弃 `lm_head.*`，其余主干权重与输入 LM 逐位一致；缺失键集合必须恰好是 `{"lm_head.weight"}`；
- `compute_gae` 在奖励 `[0,0,1]`、价值全零、`γ=λ=1` 时返回 `[1,1,1]`，`λ=0` 时返回 `[0,0,1]`；padding 位置的优点与回报必须为零；
- `clipped_surrogate` 在比值 0.5/1.0/2.0、优势 `[-1,1,1]`、`ε=0.2` 时返回 `[-0.8,1.0,1.2]`；比值落在区间内时等于未裁剪目标；
- `token_logprobs` 与 `token_values` 的位置对齐必须满足 `logp[:,t] = log π(seqs[t] | seqs[:t])`、价值对应 `V(s_{t-1})`；padding 位填 0，不得出现 `-inf`；
- 策略与参考模型同权重时，`kl_penalty` 与 `kl_estimator_k3` 在任意输入上都必须严格为零；
- `sample_responses` 必须使用全词表 softmax，不得截断词表、强制 EOS 或限制最小长度；EOS 之后的位置一律为 `pad_id`，生成长度不超过 `prompt_len + max_new`；
- `response_mask` 只标记提示之后且非 padding 的位置，EOS 计入响应；提示位置不得有 KL 塑形奖励；
- `RLConfig` 必须拒绝 `beta < 0`、`clip <= 0` 或 `clip >= 1`、`epochs < 1`、`max_new < 1`；
- `collect_rollout` 的 `rm_score` 形状为 `(B,)`，只在最后一个有效 Token 处加到奖励；`adv`、`returns`、`rewards`、`values`、`logp` 必须全部有限；
- `ppo_update` 必须重算新策略 log 概率，比值由 `exp(logp - 旧 logp)` 得到；策略与价值使用独立优化器与独立梯度裁剪；
- `ch16_rlhf.py` 除耗时外必须逐字可复现；3 个种子的 RM 准确率、三档 β 与 Best-of-8 的数字必须与正文表一致；
- 奖励作弊探针必须分别报告长度、正确性、干扰词三个维度；RM 在某个维度上系统性偏向时，不得只报成对准确率；
- PPO β=0.10/0.50/2.00 的正确率、长度与逐 Token log 比必须呈现「β 越小、奖励越高、偏离越大」的方向；若玩具实验不呈现，不得删去负面结果；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有字段；`ScalarModel` 只能替换 `lm_head`，`RLConfig` 只作训练脚本参数；玩具结论不得外推为真实 LLM 的 PPO 超参数。
- `dpo_loss` 在四个对数概率全相等时返回 `log(2)`；固定其他项、只增大赢家间隔时损失单调下降；策略与参考模型同时平移同一常数时损失不变；
- `label_smoothing > 0` 时，置信偏好对的损失必须大于未平滑版本；IPO 在 `Δ = 1/(2β)` 处损失为零，超过后损失回升；
- `dpo_loss` 对赢家的梯度必须为负、对输家的梯度必须为正，且两者在零间隔处绝对值相等；`implicit_reward` 等于 `β(logπ−logπ_ref)`；
- `completion_logprobs` 必须与逐位置手算的 `log_softmax` 对数概率之和一致；提示位置、padding 位置与 EOS 之后的 Token 不得计入；长度归一化后等于总和除以有效响应 Token 数；
- `preference_accuracy` 对间隔严格大于零的样本计为成功，平局与负间隔计为失败；
- `DPOConfig` 必须拒绝 `beta <= 0`、`lr <= 0`、`epochs < 1`、`batch_size < 1`、未知 `loss_type`、`label_smoothing >= 0.5` 与 `max_grad_norm <= 0`；
- `ch17_dpo.py` 除耗时外必须逐字可复现；3 个种子的五组对照数字必须与正文表一致；
- 偏置标签必须显示长度增加（4.768 对 3.977），去偏标签必须显示长度回落（4.357）；若玩具实验不呈现，不得删去负面结果；
- 报告必须同时包含任务正确率、干扰词、长度、逐 Token KL 与留出偏好对准确率；不得只报损失或只报偏好准确率；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有字段；`DPOConfig` 只作训练脚本参数；参考模型必须是独立冻结副本，玩具结论不得外推为真实 LLM 的 β 或学习率。
