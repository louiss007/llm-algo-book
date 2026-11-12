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
- **推理加速**：第 21 章增加 `tinylm/inference/kv_cache.py`（`KVCache`、`cache_bytes`、`cache_tokens_per_gib`、`weight_bytes`、`arithmetic_intensity`、`roofline_time`、`ttft`、`tpot`、`throughput`、`cached_attention`、`cached_forward`、`cached_prefill`、`cached_greedy_decode`）、`tinylm/inference/flash.py`（`dense_attention`、`flash_attention`）与 `tinylm/inference/paged.py`（`BlockTable`、`internal_fragmentation`、`padded_fragmentation`、`fragmentation_table`、`batch_capacity`）；缓存版本以同包 `cached_forward` 实现，不修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig`，只支持纯注意力块。代码仍以章节内嵌形式存在。
- **投机解码**：第 22 章增加 `tinylm/inference/speculative.py`（`SpecConfig`、`acceptance_rate`、`residual_probs`、`rejection_sample`、`speculative_sample`、`expected_output_tokens`、`speculative_speedup`、`truncate_cache`、`MTPHead`、`mtp_loss`、`tree_attention_mask`、`speculative_decode`）；以概率函数为接口，便于用玩具马尔可夫链验证，不修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig`。代码仍以章节内嵌形式存在。
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

## 第 18 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/posttrain/grpo.py` | 章节内嵌实现 | `GRPOConfig`（冻结 dataclass，校验 `group_size/lr/beta/clip_low/clip_high/epochs/minibatch/max_new/temperature/max_grad_norm`）；`group_advantages`（组均值、可选标准差、留一法）；`zero_variance_groups`、`mixed_group_probability`；`sample_groups`；`tempered_logprobs`；`grpo_loss`（逐 Token 比率、裁剪、k3 KL、Token/序列级归一化）；`answer_positions`、`verifiable_reward` |
| `tests/test_grpo.py` | 章节内嵌实现 | 17 个用例：`G=1` 零优势、留一法要求 `G≥2`、组均值优势求和为零、有限 G 收缩与 LOO 对比、标准差归一化、零方差组标记、混合组概率、单位比率损失、上下侧裁剪、Token/序列级差异、参考等于策略时 KL 为零、答案位置跳过 EOS/PAD、最终/任意位置奖励、温度对数概率手算、组采样形状与重复提示、配置校验 |
| `examples/ch18_grpo.py` | 章节内嵌实现 | 模 10 加 3 任务；弱 SFT 50 步；`G=1/4/8` 三档、3 个种子、300 轮；报告贪心正确率、pass@1、pass@8、奖励均值与零方差组比例 |

配置与状态：

| 字段/属性 | 默认值 | 第 18 章行为 |
| --- | --- | --- |
| `GRPOConfig.group_size` | `8` | 每个提示采样的回答数 $$G$$；`G=1` 时优势恒为零 |
| `GRPOConfig.lr` | `1e-3` | 策略学习率；实验中固定 `1e-3` |
| `GRPOConfig.beta` | `0.0` | KL 系数；`0` 表示不使用参考模型，实验中取 0 |
| `GRPOConfig.clip_low` / `clip_high` | `0.2` / `0.2` | 下侧与上侧裁剪半径；DAPO 风格可取 0.2/0.28 |
| `GRPOConfig.epochs` / `minibatch` | `2` / `64` | 固定 rollout 上重复更新；玩具实验取 1 个 epoch |
| `GRPOConfig.max_new` | `4` | 响应最多生成的 Token 数（含 EOS） |
| `GRPOConfig.temperature` | `1.0` | 采样温度；必须与 `tempered_logprobs` 一致 |
| `GRPOConfig.normalize_std` | `True` | 是否除以组内标准差；实验未做该消融 |
| `GRPOConfig.leave_one_out` | `False` | 是否使用留一基线；测试覆盖，实验未启用 |
| `GRPOConfig.token_level` | `True` | `True` 用批内 Token 平均，`False` 用逐序列平均 |
| `TinyLM.forward` / `CausalLMOutput` / `ModelConfig` | 不变 | GRPO 配置只作训练脚本参数；采样、奖励与损失均在 `tinylm/posttrain/grpo.py` |

实现要点：

- `group_advantages` 先把奖励 reshape 成 `(B, G)`，再按组减均值；`G=1` 直接返回零，避免出现“均值为自身、优势为零”的浮点路径；`leave_one_out=True` 时用 `(总和 − 自身)/(G−1)`，并显式拒绝 `G=1`。
- `sample_groups` 用 `repeat_interleave` 复制提示，逐 Token 按 `temperature` 采样；EOS 之后写 `pad_id`，返回 `attn` 掩码；采样分布与 `tempered_logprobs` 使用同一温度。
- `tempered_logprobs` 的 logits 先除以温度再 `log_softmax`，用 `gather` 取目标 Token；位置 `t` 的对数概率对应“在 `t` 之后预测 `t+1`”，因此左移一位并在首位补零，padding 位填零。
- `grpo_loss` 的比率用 `exp(logp - old_logp)`；裁剪用 `torch.clamp(ratio, 1-clip_low, 1+clip_high)` 后与未裁剪项取 `min`；Token 级归一化除以 `mask.sum()`，序列级先按每条回答平均再对回答平均。
- k3 KL 只在 `ref_logp` 存在且 `beta > 0` 时计算；`log_ratio` 裁剪到 `[-20, 20]` 防指数溢出；KL 加到损失上，因此 `loss = -pg + beta * kl`。
- `verifiable_reward` 的 `final` 模式取最后一个非 PAD、非 EOS 且位置 ≥ `prompt_len` 的 Token；`anywhere` 模式只要求答案出现在响应任意位置；空响应返回 0。
- 实验统一用 2 层、`d_model=32`、4 个头、`d_ff=128`、16 词表、3 个种子；SFT 50 步、30% 正确标签；GRPO 300 轮、批 16 提示、1 个 epoch、温度 1.0、`beta=0`；评测贪心与固定种子的 8 条采样。
- 所有随机源固定种子；输出中的耗时随机器波动，其余数字逐字可复现；正文明确标注玩具任务不能外推。

## 第 19 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/posttrain/distill.py` | 章节内嵌实现 | `DistillConfig`（冻结 dataclass，校验 `temperature/alpha/lr/max_new/max_grad_norm`）；`softmax_with_temperature`；`distill_loss`（$$\tau^2$$ 缩放 KL、可选硬标签 CE、诊断量）；`sft_labels`；`generate_greedy`；`teacher_targets`；`magnitude_prune_`；`apply_masks_`；`global_sparsity`；`parameter_count`；`compression_ratio` |
| `tests/test_distill.py` | 章节内嵌实现 | 21 个用例：温度归一化与熵单调、非法温度、相同 logits 零 KD、`alpha=1` 退化为 CE、`alpha=0` 等于 $$\tau^2$$ KL、常数平移不变、梯度方向、SFT 标签对齐两种长度、贪心生成形状与掩码、教师目标形状与无梯度、幅度剪枝稀疏度与嵌入保护、掩码在稠密更新后仍生效、剪枝比例校验、参数量与压缩比、形状不匹配报错、配置校验 |
| `examples/ch19_distill.py` | 章节内嵌实现 | 模 10 加法；40/100 提示带标签；教师 600 步、学生 800 步；SFT-40/Seq-KD/Logit-KD 三组对照；Logit-KD 学生 80% 幅度剪枝 + 200 步蒸馏恢复；3 个种子 |

配置与状态：

| 字段/属性 | 默认值 | 第 19 章行为 |
| --- | --- | --- |
| `DistillConfig.temperature` | `2.0` | 软标签温度 $$\tau$$；必须为正，实验中取 2.0 |
| `DistillConfig.alpha` | `0.5` | 硬标签权重；`0` 为纯蒸馏，`1` 为纯 CE；实验取 `0.0` |
| `DistillConfig.lr` | `1e-3` | 学生训练学习率；恢复阶段示例另用 `5e-4` |
| `DistillConfig.max_new` | `4` | 教师贪心生成的最大 Token 数 |
| `DistillConfig.max_grad_norm` | `1.0` | 梯度裁剪范数 |
| `DistillConfig.pad_id` / `eos_id` | `0` / `4` | 生成时的填充与结束 Token |
| `TinyLM.forward` / `CausalLMOutput` / `ModelConfig` | 不变 | 蒸馏配置只作训练脚本参数；生成、损失与剪枝均在 `tinylm/posttrain/distill.py` |

实现要点：

- `softmax_with_temperature` 显式拒绝 `temperature <= 0`；`distill_loss` 先检查学生与教师 logits 形状一致，再在 $$\tau$$ 下分别做 `log_softmax` 与 `softmax`，KL 用 `batchmean` 后乘 $$\tau^2$$。
- `distill_loss` 的 `alpha=1` 路径只剩硬标签 CE，`alpha=0` 路径完全跳过 CE；`labels=None` 时纯蒸馏仍可运行；诊断量包含 `kd`、`ce` 与教师熵。
- `sft_labels` 的监督从位置 `prompt_len - 1` 开始：该位置预测回答的第一个 Token，末尾位置预测 EOS，最后一个位置的标签保持 `-100`。
- `generate_greedy` 在 `torch.no_grad()` 下逐 Token 取 argmax，EOS 之后写 `pad_id`，返回布尔注意力掩码；批内全部结束后提前退出。
- `teacher_targets` 只取教师提示末位的 logits，并显式 `eval()` 与 `no_grad()`，避免教师进入学生优化图。
- `magnitude_prune_` 只处理二维以上参数并跳过 `tok_emb`/`lm_head`；用全局分位数阈值一次性置零，返回 `(threshold, total, masks)`；非结构化剪枝不改变张量形状。
- `apply_masks_` 必须在每次恢复优化器更新后调用，否则被置零的权重会重新长出来（原型中不调用时稀疏度从 0.80 掉到约 0.01）。
- 实验统一用 2 层、`d_model=32/64`、4 头、16 词表、3 个种子；教师与学生均只预测答案与 EOS；正文明确标注 40 条标注与 100 个提示的玩具设置不能外推。
- 所有随机源固定种子；输出中的耗时随机器波动，其余数字逐字可复现；SFT 与 KD 的差距同时含数据覆盖与软标签两个因素，正文已标注不可完全分离。

## 第 20 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/inference/sample.py` | 章节内嵌实现 | `SamplingConfig`（冻结 dataclass，校验温度、Top-k、Top-p、Min-p、重复惩罚）；`apply_repetition_penalty`；`top_k_mask`；`top_p_mask`；`min_p_mask`；`decode_mask`；`filtered_probs`；`sample_decode`；`greedy_decode`；`beam_search`；`sequence_logprob`；`pass_at_k`；`generate` |
| `tests/test_sample.py` | 章节内嵌实现 | 17 个用例：配置校验、Top-k 基数、Top-p 最小前缀、Min-p 相对阈值、温度熵与 argmax、重复惩罚分段与 padding 跳过、截断归一化、贪心路径、Top-k=1 等于贪心、束搜索全局更优、束宽 1 等于贪心、序列似然手算、提示/padding 跳过、pass@k 端点、固定种子复现与 EOS 填充、`generate(None)` 为贪心 |
| `examples/ch20_decode.py` | 章节内嵌实现 | 8 词表合成马尔可夫链；2 层 `d_model=32`、4 头、`d_ff=128`；700 步、批 64、3 个种子；比较贪心、束搜索（含长度归一化）、纯采样、Top-p 0.90/0.95、Min-p 0.05；统计 EOS 率、长度、distinct-2、唯一序列比例、平均对数似然与 pass@1/pass@8 |

配置与状态：

| 字段/属性 | 默认值 | 第 20 章行为 |
| --- | --- | --- |
| `SamplingConfig.temperature` | `1.0` | 采样温度 $$\tau$$；必须严格大于 0；不提供“温度 0”，贪心用 `sampling=None` |
| `SamplingConfig.top_k` | `None` | `None` 表示不截断；否则保留 logit 最大的 $$k$$ 个 |
| `SamplingConfig.top_p` | `None` | `None` 表示不截断；否则保留累积概率达到 $$p$$ 的最小前缀 |
| `SamplingConfig.min_p` | `0.0` | `0` 表示不截断；否则保留概率不低于最大概率 `min_p` 倍的 Token |
| `SamplingConfig.repetition_penalty` | `1.0` | `1.0` 表示不惩罚；大于 1 时对已出现 Token 分段缩放 logits |
| `SamplingConfig.seed` | `None` | 固定时用 `torch.Generator` 保证同设备可复现 |
| `generate(...).use_cache` | `True` | 第 20 章尚未实现 KV Cache，传入时发 `UserWarning` 并忽略；第 21 章接入 |
| `TinyLM.forward` / `CausalLMOutput` / `ModelConfig` | 不变 | 采样配置只作生成参数；解码、束搜索、似然与 pass@k 均在 `tinylm/inference/sample.py` |

实现要点：

- `filtered_probs` 的固定顺序是“重复惩罚 → 温度 → Top-k → Top-p → Min-p → 重新归一化”；顺序会改变结果，复现时必须记录。温度只在正 logits 上等比例缩放，重复惩罚才做正负分段。
- `top_p_mask` 先把概率降序排列并累加，`remove = cumulative > p` 右移一位、首位置 `False`，得到“累积概率首次达到 $$p$$”的最小前缀；不能简单用 `cumulative <= p`，否则会漏掉跨过阈值的那一项。
- `top_k_mask` 用 `topk` 的索引 `scatter` 成布尔掩码；`k` 先截到词表大小，避免越界。`decode_mask` 对多个截断取交集，最终由 `masked_fill(-inf)` 保证支撑集外概率为零。
- `apply_repetition_penalty` 用 `torch.unique(token_ids)` 取已出现集合，支持按批独立处理；`pad_id` 存在时从集合中剔除，避免惩罚 padding。正 logit 除以 $$\rho$$、非正 logit 乘以 $$\rho$$，$$\rho=1$$ 直接返回原张量。
- `sample_decode` 用 `torch.multinomial(probs, 1, generator=...)` 逐 Token 采样；EOS 之后写 `pad_id`，全批结束后提前退出；返回 `(seqs, seqs != pad_id)`。
- `greedy_decode` 与 `sample_decode` 共享相同的 EOS/padding 语义；`beam_search` 只接受 `(1, T)` 单条提示，按 (20.3) 累加对数似然，遇到 EOS 放入 `completed`，最后按 (20.5) 的长度归一化排序；不实现批内多提示并行。
- `sequence_logprob` 用 `seqs[:, :-1]` 预测 `seqs[:, 1:]`，把提示位置与 `pad_id` 目标置零；只用于比较不同策略的似然，不参与训练。
- `pass_at_k` 用 `math.comb` 实现 (20.13)，强制 `1 <= k <= n`；`c<=0` 返回 0，`n-c<k` 返回 1，避免组合数越界。
- 实验固定所有随机源；同一设备、同一种子逐位可复现；正文明确标注 8 词表、2 层 32 维、700 步的玩具任务不能外推为真实 LLM 的超参数。

## 第 21 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/inference/kv_cache.py` | 章节内嵌实现 | `KVCache`（按层预分配 `(B, n_kv_heads, max_len, d_head)`，共享 `length`）；`cache_bytes`；`cache_tokens_per_gib`；`weight_bytes`；`arithmetic_intensity`；`roofline_time`；`ttft`；`tpot`；`throughput`；`cached_attention`；`cached_forward`；`cached_prefill`；`cached_greedy_decode` |
| `tinylm/inference/flash.py` | 章节内嵌实现 | `dense_attention`（参考实现）；`flash_attention`（分块 + 在线 Softmax，支持因果掩码） |
| `tinylm/inference/paged.py` | 章节内嵌实现 | `BlockTable`（按 Token 数分配、扩容、释放、复用）；`internal_fragmentation`；`padded_fragmentation`；`fragmentation_table`；`batch_capacity` |
| `tests/test_kv_cache.py` | 章节内嵌实现 | 20 个用例：缓存公式与 GQA/MLA 缩放、形状、整段/分块预填充与逐 Token 解码等价、填充掩码、溢出断言、贪心步数、FlashAttention 与稠密一致及分块不变性、块表分配/扩容/释放、碎片率、算术强度与 roofline、TTFT/TPOT/吞吐 |
| `examples/ch21_inference.py` | 章节内嵌实现 | 缓存等价性、KV 账本、CPU 速度对照、FlashAttention 误差、分页碎片账本、7B roofline 扫描 |

接口与状态：

| 对象 | 第 21 章行为 |
| --- | --- |
| `KVCache.append(layer, k, v)` | 把新 Token 写到共享 `length` 处，不推进计数器；所有层写完由 `cached_forward` 统一 `advance(T)` |
| `KVCache.keys/values(layer)` | 返回 `[:, :, :length]` 的视图；`cached_attention` 内部用 `length + T` 直接切片，避免 append 后计数器尚未推进时读到空历史 |
| `cached_forward(model, ids, cache, ...)` | 与 `TinyLM.forward` 输入输出契约一致，读写缓存；只断言纯注意力块与 `TransformerBlock`；支持 GQA 的 `repeat_interleave`、RoPE/ALiBi 与 2D padding 掩码 |
| `cached_prefill` | 整段提示词一次写入缓存，只返回最后位置 logits |
| `cached_greedy_decode` | 预填充一次后每步只喂 1 个 Token；支持 EOS 与缓存溢出提前退出 |
| `generate(..., use_cache=True)` | 第 21 章起可接到 `cached_forward`；`use_cache` 仍不改变 `TinyLM.forward` 签名 |
| `flash_attention` | 实数运算下与 `dense_attention` 等价，差异只来自浮点累加顺序；初始 `row_max=-1e30`，避免全掩码块出现 `-inf - -inf` |
| `BlockTable` | `free` 为物理块队列，`mapping` 为序列→物理块表；`allocate` 向上取整，`ensure_capacity` 只补差量，`free_sequence` 立即归还 |
| `TinyLM.forward` / `CausalLMOutput` / `ModelConfig` | 不变；缓存、FlashAttention 与分页结构均在 `tinylm/inference/` 中，不进入模型签名 |

实现要点：

- `cache_bytes` 对 MLA 走 `n_l B T (d_c + d_head_rope) b_elem`；对 `hybrid` 只计实际含注意力的层；对非注意力块返回 0。
- `cached_attention` 只投影新 Token 的 Q/K/V；GQA 在拼接后按 `repeat_interleave` 展开 K/V；因果掩码用绝对位置 `position_ids`，与第 4 章的 RoPE 约定一致。
- `cached_forward` 不支持 SSM、线性注意力与混合架构，使用 `assert cfg.block_type == "attention"` 显式拒绝；并行块与普通块分别处理，MLP 返回 tuple 时取第一项。
- `flash_attention` 的块内因果跳过条件为 `j0 > start + i1 - 1`；`start = Tk - Tq`，支持查询是完整序列后缀的缓存场景。
- `internal_fragmentation` 按“已预留槽位”计分母；`padded_fragmentation` 按批内最长序列统一预留；`fragmentation_table` 同时返回利用率与浪费率。
- 实验固定随机源；缓存等价性中的分块与逐 Token 误差为 `1.49e-07`，FlashAttention 在 T=256 的误差为 `4.77e-07`；CPU 速度对照的绝对秒数随机器波动，正文已注明不能外推。

## 第 22 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/inference/speculative.py` | 章节内嵌实现 | `SpecConfig`；`acceptance_rate`；`residual_probs`；`rejection_sample`；`speculative_sample`；`expected_output_tokens`；`speculative_speedup`；`truncate_cache`；`MTPHead`；`mtp_loss`；`tree_attention_mask`；`speculative_decode` |
| `tests/test_speculative.py` | 章节内嵌实现 | 14 个用例：接受率恒等式、残差归一化与支撑集、单步无损性蒙特卡洛、`p=q` 全接受、多步接受数上界、期望 Token 端点与单调性、加速比端点、内部最优 $$\gamma$$、KV 回滚等价、树掩码祖先性、MTP 形状/损失/忽略填充、确定性解码、非法批次拒绝 |
| `examples/ch22_speculative.py` | 章节内嵌实现 | 20 万次拒绝采样分布核对、草稿质量与 $$\gamma$$ 账本、10 万轮接受数模拟、KV 回滚误差、MTP 损失、树掩码与玩具马尔可夫链 |

接口与状态：

| 对象 | 第 22 章行为 |
| --- | --- |
| `SpecConfig` | 冻结 dataclass，字段 `num_draft`（默认 4）与 `seed`（默认 `None`）；`num_draft >= 1` 在 `__post_init__` 断言；不进入 `ModelConfig` |
| `acceptance_rate(p, q)` | 在最后一维逐行计算 $$\sum\min(p,q)$$，返回形状 `(B,)`；要求 `p`、`q` 同形 |
| `residual_probs(p, q)` | 返回归一化的 $$\max(0,p-q)$$；当残差质量和为 0 时退回 `p`，避免对全零分布采样 |
| `rejection_sample(q, p, generator)` | `q`、`p` 形状 `(B,V)`；返回 `(token, accepted)`，各为 `(B,)`；含接受、残差回退两条分支 |
| `speculative_sample(q_seq, p_seq, generator)` | `q_seq` 形状 `(B,γ,V)`，`p_seq` 形状 `(B,γ+1,V)`；返回 `(drafts, n_accept, correction)`，形状 `(B,γ)`、`(B,)`、`(B,)`；用 `cumprod` 保证首次拒绝后不再接受 |
| `expected_output_tokens(alpha, gamma)` | 实现 (22.7)；`alpha >= 1` 时返回 `gamma+1`，避免除零 |
| `speculative_speedup(alpha, gamma, cost_ratio)` | 实现 (22.8)；只做账本计算，不测量真实延迟 |
| `truncate_cache(cache, length)` | 只修改 `cache.length`，断言 `0 <= length <= cache.length`；不重算已接受前缀 |
| `MTPHead` | 一组独立 `Linear(d_model, vocab_size, bias=False)`；`forward(hidden)` 返回长度为 `num_steps` 的 logits 列表 |
| `mtp_loss(logits_list, targets, weights, ignore_index)` | 第 `i` 个头预测 `targets[:, i:]`，与 `logits[:, :T-i]` 计算交叉熵；`weights=None` 时等权平均，否则按权重归一化 |
| `tree_attention_mask(parents)` | 输入父节点列表（根为 `-1`），返回 `(n,n)` 布尔掩码；节点只能看到自己与祖先 |
| `speculative_decode(...)` | 只接受 `(1,T)` 的提示；草稿自回归提议，目标每轮重算前向；返回 `(1, start+max_new_tokens)`；用于算法验证，不是高性能实现 |
| `TinyLM.forward` / `CausalLMOutput` / `ModelConfig` | 不变；投机解码、MTP 头与回滚工具均在 `tinylm/inference/speculative.py` 中 |

实现要点：

- `residual_probs` 的零质量回退是必要的：当 `p=q` 时残差全零，若不回退，`torch.multinomial` 会抛 “sum of probabilities <= 0”；回退到 `p` 不改变无损性，因为此时所有候选都应被接受。
- `speculative_sample` 的 `prefix_ok = cumprod(accepted)` 把“首次拒绝后全部作废”压缩成一次向量化操作；`n_accept` 为前缀真值之和。
- `truncate_cache` 与第 21 章的 `KVCache` 对齐：`append` 不推进计数器，所有层写完后统一 `advance`，因此回滚只改整数，不需要重算 K/V。
- `speculative_decode` 每轮重算目标前向，未复用 KV Cache；这样代码最短、验证最直接，但不能作为吞吐基准。
- 实验固定随机源；无损性误差与接受率、接受数模拟与理论值、回滚误差 `2.98e-08` 均写入正文，玩具结论明确标注不可外推。

## 第 23 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/inference/quant.py` | 章节内嵌实现 | `QuantConfig`（冻结 dataclass，校验比特数、粒度与组大小）；`QuantizedTensor`；`qrange`、`quant_params`、`quantize`/`dequantize`、`fake_quant`；`sqnr_db`、`theoretical_sqnr`、`quantization_error`；`storage_bits`、`compression_ratio`；`pack_int4`/`unpack_int4`；`quantized_linear`；`gptq_quantize`；`smooth_scale`/`smooth_quant`；`kv_quantize`、`attention_sqnr` |
| `tests/test_quant.py` | 章节内嵌实现 | 16 个用例：配置校验、整数范围、舍入上界、非对称端点、粒度单调、任意形状与填充、假量化往返、每比特约 6 dB、INT4 打包、存储公式、量化线性层、GPTQ 优于 RTN、SmoothQuant 降误差、KV 逐通道优于逐 Token、零张量安全、布局保持 |
| `examples/ch23_quant.py` | 章节内嵌实现 | 粒度与比特扫描、INT4 打包账本、GPTQ 对比 RTN、SmoothQuant 的 $$\alpha$$ 扫描、KV 粒度对比、TinyLM 权重仅量化、7B 带宽账本 |

配置与状态：

| 字段/属性 | 默认值 | 第 23 章行为 |
| --- | --- | --- |
| `QuantConfig.bits` | `8` | 允许 2–8；对称量化 `qmax=2^(b-1)-1`，非对称用满 `2^b` |
| `QuantConfig.granularity` | `"channel"` | `tensor`/`channel`/`group`；任意形状按最后两维整理成“行 × 组” |
| `QuantConfig.group_size` | `64` | 仅 `granularity="group"` 时生效，要求 ≥ 2；不能被整除时先右侧补零再截掉 |
| `QuantConfig.symmetric` | `True` | 对称时零点为 0、码点存 `int8`；非对称存 `uint8` 与零点 |
| `QuantConfig.eps` | `1e-8` | 全零组的下限，保证缩放因子非零 |
| `TinyLM.forward` / `CausalLMOutput` / `ModelConfig` | 不变 | 量化只操作张量；`QuantConfig` 不进入 `ModelConfig`；示例在模型外对权重做 `fake_quant` |

实现要点：

- `_group_view`/`_ungroup` 把任意形状的最后两维临时整理成 `(行, 组, 组内元素)`；逐张量与逐通道分别视为单组与“组大小等于输入维度”的逐组，反量化按同一元信息还原。
- 对称量化 `zero` 恒为零张量，反量化仍需减去它，以保持与 `quantize` 返回的三元组接口一致；非对称零点在 `qmin..qmax` 内取整。
- `pack_int4` 把 `[-8,7]` 两两塞进 `uint8`，奇数长度补一个零码点；`unpack_int4` 按 `numel` 截断，并做二补码还原。
- `gptq_quantize` 只支持逐通道对称量化；对 Hessian 的零对角做 dead-column 处理，加阻尼后 Cholesky 求逆，再逐列量化并按 `Hinv` 补偿剩余列；输入维度为 4096 时直接求逆不可行，工业实现需分块。
- `kv_quantize(per_channel=True)` 先转置最后两维，使“按通道”变成“按行”，量化后再转置回来；示例中 K 用逐通道、V 用逐 Token。
- `attention_sqnr` 用同一组 Q 比较 FP16 与量化后的注意力输出，直接测量误差进入 Softmax 后的影响。
- 实验固定随机源；除 TinyLM 训练耗时（示例中约 8.7 s）与 pytest 计时外，数字逐字可复现；带宽账本为理想下界，正文已标注不可当作实测加速比。

## 第 24 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/inference/search.py` | 章节内嵌实现 | `SearchConfig`（冻结 dataclass，校验样本数、Token 上限、温度、投票方式、最小样本数、停止阈值与 PUCT 系数）；`SearchResult`；`coverage_probability`、`pass_at_k`、`majority_accuracy`、`majority_vote`、`best_of_n`、`selection_accuracy`、`should_stop`、`self_consistency`、`token_cost`、`puct_score`；`TreeNode`、`tree_search` |
| `tests/test_search.py` | 章节内嵌实现 | 18 个用例：配置校验、覆盖率边界与蒙特卡洛、pass@k 端点、投票单调方向、平局、加权投票、Best-of-N 平局、选择率端点、提前停止、预算耗尽、PUCT 公式、预算为 0、陷阱树、终止节点 |
| `examples/ch24_search.py` | 章节内嵌实现 | 覆盖率、投票、验证器选择、陷阱树、TinyLM 链式算术、提前停止六组实验 |

配置与状态：

| 字段/属性 | 默认值 | 第 24 章行为 |
| --- | --- | --- |
| `SearchConfig.max_samples` | `16` | 单次请求最多采样条数；预算耗尽时收束 |
| `SearchConfig.max_tokens` | `512` | 每条样本的输出上限，用于成本记账 |
| `SearchConfig.temperature` | `1.0` | 采样温度；示例固定为 1.0 |
| `SearchConfig.vote` | `"majority"` | `majority` 等权投票；`weighted` 使用 `sample_fn` 返回的权重 |
| `SearchConfig.min_samples` | `2` | 提前停止前必须达到的样本下限 |
| `SearchConfig.stop_margin` | `0.0` | 0 表示不提前停止；否则要求最高分与次高分之差 ≥ 阈值 |
| `SearchConfig.c_puct` | `1.0` | PUCT 探索系数；示例陷阱树用 2.0 |
| `TinyLM.forward` / `CausalLMOutput` / `ModelConfig` | 不变 | 搜索逻辑全部在模型外；`SearchConfig` 不进入 `ModelConfig` |

实现要点：

- `margin` 用分数差而不是票数差，因为加权投票时“票数”不再等价于“分数”；平分时 `majority_vote` 取最早出现的答案，保证确定性。
- `majority_vote` 的 `counts` 保留原始出现次数，`scores` 保留加权和；两者都写入 `SearchResult`，便于示例与测试分别检查。
- `self_consistency` 每采一条就重新计票，因此提前停止只看当前结果；预算耗尽时返回最后一次计票，不额外采样。
- `selection_accuracy` 用成对判别近似：给定 $$k$$ 条正确候选，最高分正确的概率取 $$kq/(kq+(N-k)(1-q))$$；$$q=1$$ 退化为覆盖率，$$q=0.5$$ 退化为单次成功率。正文明确标注这是近似，不是精确的排序模型。
- `tree_search` 在展开内部节点后按先验下探一个子节点；预算只计 `value_fn` 调用，展开本身不计。最佳状态在所有被评估的节点上跟踪，而不是只看根的子节点。
- 树搜索没有复用 KV Cache；`state` 是不透明对象，示例用字符串。真实实现需要树形注意力掩码、按分支保留的 KV 块与第 22 章的回滚工具，本节只验证搜索策略本身。
- 实验固定随机源；除 TinyLM 训练耗时（示例约 5–12 s，随机器负载变化）与 pytest 计时外，输出逐字可复现。

## 第 25 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/multimodal.py` | 章节内嵌实现 | `VisionConfig`（冻结 dataclass，校验分辨率整除切块、维度整除头数、池化与投影类型）；`image_token_count`、`attention_multiplier`；`PatchEmbed`、`VisionEncoder`；`TokenResampler`、`Projector`；`MultimodalTinyLM`；`contrastive_loss`、`vq_quantize`、`vq_loss` |
| `tests/test_multimodal.py` | 章节内嵌实现 | 12 个用例：Token 计数与合并、注意力倍数、卷积切块等于线性分块、池化形状、三种投影、占位符数量校验、梯度路径、冻结视觉塔、答案位损失、对比损失排序、VQ 最近邻与损失 |
| `examples/ch25_multimodal.py` | 章节内嵌实现 | Token 预算表、InfoNCE 数值、VQ 码本扫描、12×12 合成视觉问答的五组配置与去图像对照 |

配置与状态：

| 字段/属性 | 默认值 | 第 25 章行为 |
| --- | --- | --- |
| `VisionConfig.image_size` | `12` | 合成图边长；必须被 `patch_size` 整除 |
| `VisionConfig.patch_size` | `3` | 切块边长；示例另跑 2、6 |
| `VisionConfig.in_channels` | `3` | RGB 通道数 |
| `VisionConfig.d_vision` | `32` | 视觉塔隐藏维度 |
| `VisionConfig.n_layers` | `2` | 视觉塔 Pre-LN 块数 |
| `VisionConfig.n_heads` | `4` | 视觉塔注意力头数；必须整除 `d_vision` |
| `VisionConfig.mlp_ratio` | `2.0` | 视觉塔 FFN 宽度倍率 |
| `VisionConfig.pool` | `"tokens"` | `tokens` 保留块序列；`cls` 只取 CLS |
| `VisionConfig.projector` | `"mlp"` | `linear` / `mlp` / `resampler` |
| `VisionConfig.n_queries` | `8` | 重采样器的可学习查询数，对应 (25.7) 的 $$M$$ |
| `MultimodalTinyLM.image_token_id` | 构造参数 | 文本序列中的图像占位符 ID，示例为 2 |
| `TinyLM.forward` / `CausalLMOutput` / `ModelConfig` | 不变 | 视觉分支全部在 `MultimodalTinyLM` 与 `VisionConfig`；`VisionConfig` 不进入 `ModelConfig` |

实现要点：

- `MultimodalTinyLM` 用一次性的 `tok_emb.register_forward_hook` 把占位符位置的嵌入替换为投影后的视觉特征，再调用原 `TinyLM.forward`；这是教学环境的兼容方案，正文明确说明生产实现应在嵌入层内完成拼接，避免依赖 hook 的注册顺序。
- 占位符数量必须逐行等于视觉 Token 数，否则抛 `ValueError`；不做静默截断或广播，防止图文错位被训练损失掩盖。
- `Projector` 在 `z.dim()==2` 时补一个 Token 维，使 `pool="cls"` 的 `(B, d_v)` 与 `pool="tokens"` 的 `(B, N_p, d_v)` 共用同一接口。
- `image_token_count` 同时负责 (25.1) 的切块计数与 (25.7) 的合并压缩；`merge` 必须整除块数，示例用 `merge=2` 演示 4 倍压缩。
- `attention_multiplier` 只记注意力打分与前后的二次项，不含 FFN 与投影；正文用它给预填充算力下界，不当作端到端加速比。
- `contrastive_loss` 返回对称 InfoNCE 的平均；温度必须为正，示例用 `τ=0.07` 与 `τ=1.0` 对照随机配对与对齐配对。
- `vq_quantize` 用展开的平方距离公式，不依赖 `cdist`；`vq_loss` 对码本项与承诺项分别 detach，系数为 `beta`。
- 冻结视觉塔通过 `requires_grad_(False)` 实现；示例中冻结的是随机初始化塔，正文已注明不得外推为“预训练视觉塔冻结”的结论。
- 训练与推理全部 CPU 单线程可跑；示例固定数据种子与模型种子，除训练耗时外输出逐字可复现。

## 第 26 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/agent/loop.py` | 章节内嵌实现 | `AgentConfig`（步数/上下文/重复/观测长度/错误即停）、`ToolSpec`、`ToolCall`、`FinalAction`、`AgentStep`、`AgentResult`；`validate_arguments`、`ToolRegistry`、`extract_json_object`、`parse_action`、`estimate_tokens`、`run_agent`；`trajectory_cost`、`result_cost`、`retry_success`、`task_success`、`value_of_information` |
| `tests/test_agent.py` | 章节内嵌实现 | 15 个用例：配置校验、带推理的 JSON 解析、歧义/畸形动作拒绝、未知工具、参数缺失/多余/类型错误、工具异常转观测、正常作答、步数上限、重复拦截、错误即停、上下文上限、观测截断、重试与任务公式端点、成本复算、VoI 与 Token 估算 |
| `examples/ch26_agent.py` | 章节内嵌实现 | 两步工具调用、重复拦截、参数校验、最小权限与过度授权的注入对照、2 万次蒙特卡洛核对 (26.8)/(26.9)、深度—成本表、VoI 对照 |

配置与状态：

| 字段/属性 | 默认值 | 第 26 章行为 |
| --- | --- | --- |
| `AgentConfig.max_steps` | `6` | 循环步数上限；达到后以 `max_steps` 停止 |
| `AgentConfig.max_context_tokens` | `4096` | 每步开始前检查；超限以 `context_limit` 停止，且不产生该步动作 |
| `AgentConfig.max_repeat` | `2` | 同一 `(工具名, 参数 JSON)` 的最大执行次数，超出返回错误观测 |
| `AgentConfig.max_observation_chars` | `512` | 观测 JSON 超过该长度时返回 `truncated` 预览 |
| `AgentConfig.stop_on_tool_error` | `False` | 为 `True` 时，解析错误或工具错误立即以 `tool_error` 停止 |
| `ToolSpec.parameters` | 无默认 | 参数名到 `str/int/float/bool` 的映射；执行前校验缺参、多参与类型 |
| `TinyLM.forward` / `CausalLMOutput` / `ModelConfig` | 不变 | 智能体模块不依赖模型；策略以 `Callable[[str], str]` 注入 |

实现要点：

- 循环只依赖 Python 标准库，单进程、无网络可跑；`policy` 与 `ToolRegistry` 完全解耦，脚本策略只验证控制流，不代表模型能力。
- 动作协议为“推理文本 + 单个 JSON 对象”：`{"tool": ..., "args": {...}}` 或 `{"answer": ...}`；`extract_json_object` 用 `JSONDecoder.raw_decode` 从第一个合法 `{` 开始解析，允许前后有自然语言。
- 解析失败、未知工具、参数非法与工具异常都转成 `{"ok": false, ...}` 观测回填，主循环不因单个工具崩溃；`stop_on_tool_error` 决定是否立刻终止。
- 重复调用检测以 `(name, sort_keys 后的 args JSON)` 为键；超过 `max_repeat` 时拒绝执行，但仍把错误观测写回上下文，让策略有机会换动作。
- 上下文按 (26.5) 追加 `action` 与 `observation`；每步记录 `context_tokens` 与 `action_tokens`，`result_cost` 用 (26.7) 复算，默认 `c_in=1.0, c_out=4.0`。
- `estimate_tokens` 按 UTF-8 字节数除以 4 估算，只用于相对账本；接入真实模型时应替换为模型分词器。
- `ToolRegistry.execute` 先校验参数再调用函数，避免非法参数触发副作用；工具返回值统一包成 `{"ok": true, "result": ...}` 并按字符数截断。
- 接 TinyLM 的方式是把 `model.generate` 与解码包成 `policy(context)`，不需要改 `TinyLM.forward`；系统提示应包含 `registry.describe()`、动作格式、数据/指令边界与最大生成长度。
- 示例中的 `delete_file` 是演示桩，不触碰文件系统；`safe_calc` 用 AST 白名单实现，不调用 `eval`。
- 最小权限演示：不注册 `delete_file` 时，注入触发的删除调用得到 `unknown tool`；注册后才执行演示桩。这是权限边界，不是模型层面的注入防御。

## 第 27 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/eval/metrics.py` | 章节内嵌实现 | `accuracy`、`macro_f1`、`pass_at_k`、`binomial_se`、`wilson_interval`、`paired_bootstrap_diff`、`mcnemar_test`、`expected_calibration_error`、`ngrams`、`contamination_rate`、`bradley_terry_prob`、`elo_update`、`cohen_kappa`、`cost_aware_utility` |
| `tests/test_eval.py` | 章节内嵌实现 | 12 个用例：准确率、macro-F1、pass@k、标准误、Wilson 区间、配对 bootstrap、McNemar、ECE、n-gram 污染、BT/Elo、kappa、成本效用及边界错误 |
| `examples/ch27_eval.py` | 章节内嵌实现 | 1000 题确定性配对比较、pass@k 公式与 2 万次模拟对照、n-gram 污染、裁判位置偏差、ECE、kappa、macro-F1 与成本效用 |

配置与状态：

| 字段/属性 | 默认值 | 第 27 章行为 |
| --- | --- | --- |
| 评测样本输入 | 逐样本序列 | 所有指标接收等长的预测/标签或逐次结果；空输入与长度不一致直接报错 |
| `paired_bootstrap_diff.n_boot` | `4000` | 重采样次数；示例 4000 次，测试可调小 |
| `paired_bootstrap_diff.level` | `0.95` | 百分位区间水平 |
| `paired_bootstrap_diff.seed` | `0` | 固定种子，保证区间可复现 |
| `expected_calibration_error.n_bins` | `10` | 置信度分箱数；等宽分箱 |
| `contamination_rate.n` | `13` | n-gram 长度；示例对照 5 与 13 |
| `elo_update.k` | `32.0` | 单场 Elo 更新幅度 |
| `TinyLM.forward` / `CausalLMOutput` / `ModelConfig` | 不变 | 评测模块不依赖模型，只接收结果 |

实现要点：

- 纯标准库；不依赖 NumPy，单进程 CPU 可跑，适合作为评测脚本的最小内核。
- `accuracy` 与 `macro_f1` 统一检查空输入与长度不一致；宏平均对类别等权，微平均留给上层按需实现。
- `pass_at_k` 用组合数实现无放回精确估计 (27.3)，不做 `1-(1-p)^k` 近似；`k`、`c` 越界报错。
- `wilson_interval` 把上下界截断到 `[0,1]`；`binomial_se` 在 `p=0` 或 `1` 时返回 0。
- `paired_bootstrap_diff` 对逐样本差值重采样，保留配对结构；返回值依次为观测差、下界、上界。
- `mcnemar_test` 用精确二项尾概率并截断到 1；只适用于二分类的配对结果。
- `expected_calibration_error` 采用等宽分箱；置信度必须落在 `[0,1]`，空桶不参与加权。
- `contamination_rate` 先把训练文档的 n-gram 并入集合，再统计评测文档命中比例；Token 序列必须可哈希。
- `bradley_terry_prob` 使用 Elo 尺度（400 分母），与 `elo_update` 保持一致；分数不是绝对能力。
- `cohen_kappa` 在机会一致率为 1 时返回 1；空输入报错。
- 示例中的配对数据是构造的确定性样本，不代表真实模型；位置偏差与 pass@k 是模拟，正文已注明。

## 第 28 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `tinylm/interpret/core.py` | 章节内嵌实现 | `logit_margin`、`linear_probe_accuracy`、`probe_selectivity`、`attention_maps`、`ablate_heads`、`activation_at`、`patched_output`、`recovery`、`SparseAutoencoder`、`train_sae`、`sae_metrics`、`feature_purity`、`project_out`、`ablated_direction` |
| `tests/test_interpret.py` | 章节内嵌实现 | 15 个用例：logit margin、线性探针、选择性、注意力归一化与因果性、MLA 拒绝、hook 恢复、激活形状、恢复率端点、SAE 形状/范数/训练、特征纯度、方向投影与方向消融 |
| `examples/ch28_interpret.py` | 章节内嵌实现 | 复制首 Token 任务、层 1 线性探针与打乱标签对照、位置交换因果测试、注意力图、单头/组消融、激活修补位点扫描、SAE 的 λ 网格与特征投影 |

配置与状态：

| 字段/属性 | 默认值 | 第 28 章行为 |
| --- | --- | --- |
| 解释模块位置 | `tinylm/interpret/` | 只读分析与临时 hook；不改变前向默认路径 |
| `linear_probe_accuracy.steps` | `300` | 线性探针训练步数；测试可调小 |
| `linear_probe_accuracy.lr` | `1e-2` | 探针学习率 |
| `SparseAutoencoder.dict_size` | `128` | 字典大小 $$K$$；示例用 128 |
| `train_sae.l1` | `0.02` | 稀疏系数 $$\lambda$$；示例扫描 0/0.005/0.02/0.05 |
| `train_sae.steps` | `1200` | SAE 训练步数 |
| `attention_maps.layer` | 无 | 指定层；要求模块具备 `q_proj`、`k_proj`、`n_heads`、`n_kv_heads`、`d_head`，MLA 直接抛 `NotImplementedError` |
| `ablate_heads(heads)` | 无 | 上下文管理器；退出时恢复原 forward，异常路径也不得残留 hook |
| `TinyLM.forward` / `CausalLMOutput` / `ModelConfig` | 不变 | 解释工具不改变签名与默认输出 |

实现要点：

- 纯 PyTorch 与标准库；单进程 CPU 可跑，不依赖 Transformers、SAE 库或可视化库。
- `logit_margin` 取目标 Token 与干扰 Token 的对数概率之差 (28.5)；目标与干扰相同时返回 0。
- `linear_probe_accuracy` 用随机置换切分训练/测试，避免按类别切分造成测试集缺类；训练目标含权重衰减 (28.1)。
- `probe_selectivity` 只做差值 (28.2)；显著性判断留给调用方，示例用打乱标签作为对照。
- `attention_maps` 显式检查所需属性后再取用，MLA 的 `LatentAttention` 没有 `k_proj`，必须得到 `NotImplementedError` 而不是 `AttributeError`。
- `ablate_heads`、`patched_output`、`ablated_direction` 均以 `contextmanager` 实现，在 `finally` 中恢复 hook 或激活。
- `recovery` 严格按 (28.6)；`S_clean == S_corrupt` 时报错，避免除零。
- `SparseAutoencoder` 训练后把解码器列归一化为单位长度；`train_sae` 返回的模型可直接用于 `sae_metrics` 与 `feature_purity`。
- `sae_metrics` 返回重构 MSE、平均 L0 (28.8) 与死特征数；`feature_purity` 按每个样本的最大激活特征归属类别，再取多数类比例 (28.9)。
- `project_out` 去掉指定方向上的分量；`ablated_direction` 只改指定位置的激活，用于特征级因果检查。
- 示例中的复制任务、探针、注意力、消融、修补与 SAE 数字全部固定种子；负结果（特征投影几乎不改变准确率）必须保留。

## 第 29 章变更

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `examples/ch29_frontier.py` | 独立示例 | 不进入 `tinylm/` 包；`continual_run`（顺序/EWC/回放）、`collapse_trace`（纯合成与 10% 真实锚定）、`sparse_rl`（终端奖励/逐步验证器/势函数塑形）三组数值实验 |

配置与状态：

| 字段/属性 | 默认值 | 第 29 章行为 |
| --- | --- | --- |
| `TinyLM` 包 | 不变 | 本章不新增模块，也不修改任何已有模块 |
| `TinyLM.forward` / `CausalLMOutput` / `ModelConfig` | 不变 | 签名与默认输出保持冻结 |
| `continual_run.steps` | `500` | 每个任务的训练步数；示例固定 500 |
| `continual_run.lam` | `1e4` | EWC 系数 $$\lambda$$ |
| `train.rho` | `0.5` | 回放比例 $$\rho$$ |
| `collapse_trace.N` | `100` | 每代样本数；方差收缩比为 $$(N-1)/N$$ |
| `collapse_trace.alpha` | `0.0` | 真实数据比例；示例对照 0.0 与 0.1 |
| `sparse_rl.steps` | `8` | 链长 $$K$$；每步独立伯努利 |
| `sparse_rl.mode` | `"terminal"` | `terminal` / `step` / `potential` 三种奖励信号 |

实现要点：

- 纯 PyTorch 与标准库；单进程 CPU 约 8 秒，不依赖 Transformers 或任何 RL 库。
- 持续学习用 2→64→64→1 的 Tanh MLP，任务标识作为第二维输入；先 A（$$\sin$$）后 B（$$\cos$$），比较顺序训练、EWC 与回放 (29.5)–(29.6)。
- `fisher_diag` 按样本累加梯度平方再除以样本数，得到 Fisher 对角近似；`train` 中的惩罚项按 $$0.5\lambda\sum_iF_i(\theta_i-\theta_i^*)^2$$ 实现 (29.5)。
- 回放按概率 $$\rho$$ 从旧任务缓冲采样，损失为混合形式 (29.6)；本实验直接复用 A 的全量数据，未实现缓冲淘汰。
- 合成数据实验递归做最大似然：先采样、再估计均值与方差、再用估计分布采样下一代；`var` 用除以 $$N$$ 的样本方差，递推为 (29.8)；`alpha>0` 时混入标准正态真实样本，对应 (29.9)。
- 稀疏奖励实验用独立伯努利策略，`terminal` 只给末端奖励 (29.1)–(29.2)；`step` 给每步 1 分；`potential` 以累计正确步数为势函数 $$\Phi$$，按 (29.4) 构造差分奖励，$$\gamma=1$$。
- 报告解析成功率与 2 万次采样的经验成功率；两者一起给，避免把策略参数乘积误当成真实环境成功率。
- 示例中的 MLP、一维高斯与 8 步链都是机制演示，不得外推为真实 LLM 的遗忘率、数据退化速度或 RL 样本效率；正文已写明玩具规模与边界。

## 全书完成后的状态

第 29 章是全书最后一章，没有第 30 章，也没有后续接口任务。全书 TinyLM 的冻结接口仍以 `appendix/f-tinylm.md` 为准：

- `ModelConfig`、`CausalLMOutput` 与 `TinyLM.forward` 自第 2 章冻结以来未破坏签名；后续章节只通过可选字段与配置分派扩展能力。
- 第 29 章不新增 TinyLM 模块，示例代码为独立脚本，可脱离 `tinylm/` 包单独运行。
- 全书所有实验均可由固定种子复现；第 29 章的三组实验与正文输出逐字一致。
- 若未来需要把前沿方向做成可运行模块，应另立新章节或附录，遵循同样的接口冻结与固定种子约束；时效性结论必须重新核对论文与发布材料。

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
- `group_advantages` 在 `group_size=1` 时必须返回全零；`leave_one_out=True` 且 `group_size=1` 时必须报错；
- 不标准化时，每个 `group_size` 组的优势之和在数值误差内为零；包含自身的组均值对单个样本的收缩必须等于 `(G-1)/G` 倍；
- 留一优势与组均值优势必须满足 `r_i - b_i^{(-i)} = G/(G-1)(r_i - mean)`；标准化后两者逐元素相等；
- `zero_variance_groups` 必须同时标记全对与全错组，混合组不得标记；`mixed_group_probability(0.5, 1) == 0` 且 `mixed_group_probability(0.5, 2) == 0.5`；
- `grpo_loss` 在比率为 1 时等于负优势；正优势且比率超过上界时目标被裁剪到 `(1+clip_high)A`；负优势且比率低于下界时目标被裁剪到 `(1-clip_low)A`；
- Token 级与序列级损失在回答长度不等且优势不同时必须给出不同结果；参考模型与策略相同时 k3 KL 必须严格为零；
- `verifiable_reward` 的 `final` 模式必须跳过 EOS 与 PAD，只看最后一个答案 Token；`anywhere` 模式必须能找到出现在响应任意位置的答案；空响应返回 0；
- `tempered_logprobs` 必须与同温度手算的 `log_softmax` 一致；`sample_groups` 必须按组重复提示，返回的 `attn` 与序列同形且为布尔；
- `ch18_grpo.py` 除耗时外必须逐字可复现；`G=1` 的贪心正确率必须与 SFT 逐位相同，零方差组比例必须为 1.00；`G=8` 的 3 种子均值必须高于 `G=4`；
- 报告必须同时包含贪心正确率、pass@1、pass@8、奖励均值与零方差组比例；不得只报奖励或只报一个种子；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有字段；`GRPOConfig` 只作训练脚本参数；采样温度必须与 `tempered_logprobs` 一致，玩具结论不得外推为真实 LLM 的组大小或学习率。
- `softmax_with_temperature` 的输出每行和为 1；同一 logits 下温度升高时熵严格增大；`temperature <= 0` 必须报 `ValueError`；
- `distill_loss` 在师生 logits 完全相同时 KD 为零；`alpha=1` 时与 `F.cross_entropy` 逐位一致；`alpha=0` 时等于手算的 $$\tau^2$$ KL；师生同时平移常数时损失不变；
- KD 梯度方向必须正确：教师偏好某类时学生该类 logit 的梯度为负、其余类别为正；
- `sft_labels(seqs, prompt_len)` 在位置 `prompt_len - 1` 放回答首 Token、在 `prompt_len` 放后续 Token，最后一个位置为 `-100`；两种回答长度均成立；
- `generate_greedy` 返回的序列与掩码同形，掩码为 `seqs != pad_id`；`teacher_targets` 形状为 `(B, V)` 且 `requires_grad == False`；
- `magnitude_prune_` 在 `fraction=0.5` 时 `global_sparsity` 落在 0.5±0.03；跳过的嵌入与输出头不得被置零；`fraction` 越界必须报错；
- 恢复阶段每次优化器更新后调用 `apply_masks_`，稀疏度必须保持目标值；不调用时稀疏度下降是可复现的负结果，正文不得删去；
- `parameter_count` 与 `compression_ratio` 的比值必须等于两模型参数量之比；
- `ch19_distill.py` 除耗时外必须逐字可复现；3 个种子的教师准确率、三种学生结果与剪枝前后准确率必须与正文表一致；
- 报告必须同时给出全量/留出准确率、参数量、压缩比与剪枝稀疏度；不得只报全量准确率或只报一个种子；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有字段；`DistillConfig` 只作训练脚本参数；教师必须冻结，玩具结论不得外推为真实 LLM 的温度、剪枝率或压缩比。
- `SamplingConfig` 必须拒绝 `temperature <= 0`、`top_k < 1`、`top_p` 不在 `(0,1]`、`min_p` 不在 `[0,1]`、`repetition_penalty <= 0`；
- `top_k_mask(logits, k)` 的保留数恰为 `min(k, V)`；`k >= V` 时全保留；`top_p_mask` 的最小前缀满足“累积概率 ≥ p，且去掉最后一项后 < p”；
- `min_p_mask` 只保留 `p_v >= min_p * max_v p_v` 的 Token；`min_p=0` 全保留，`min_p=1` 只保留 argmax；
- `filtered_probs` 输出的每一行和为 1（误差小于 `1e-6`），支撑集之外的零概率严格为 0；温度升高时熵增大且 argmax 不变；
- `apply_repetition_penalty` 对已出现且 logit 为正的 Token 除以 $$\rho$$、对非正 logit 乘以 $$\rho$$，未出现 Token 不变；`pad_id` 必须跳过；
- `greedy_decode` 与 `top_k=1` 的采样在同一种子下逐位一致；`beam_search(W=1)` 与贪心前几步一致；
- 束搜索必须在手工玩具模型上找到贪心错过的全局更优路径，且 `sequence_logprob` 对该路径给出更高值；
- `sequence_logprob` 必须跳过提示位置与 `pad_id`，与手算 `log_softmax` 之和误差小于 `1e-5`；
- `pass_at_k(n,c,k)` 在 `k=1` 时等于 `c/n`，在 `n-c<k` 时等于 1，在 `c=0` 时等于 0；`k>n` 必须报错；
- 固定 `seed` 的 `sample_decode` 两次运行逐位一致；EOS 之后的位置必须为 `pad_id`；
- `generate(sampling=None)` 必须等价于贪心；`use_cache=True` 在 chapter 20 只发警告、不得改变输出；
- `ch20_decode.py` 除耗时外必须逐字可复现；3 个种子的 7 组数字必须与正文表一致；
- 报告必须同时给出 EOS 率、平均长度、唯一序列比例、平均对数似然与 pass@1/pass@8；不得只报似然或只报一个种子；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有字段；`SamplingConfig` 只作生成参数；玩具结论不得外推为真实 LLM 的温度、Top-p 或束宽。
- `cache_bytes` 与 (21.1) 逐位一致；`attention="mla"` 时走 (21.2)，只计 `d_c + d_head_rope`；`block_type!="attention"` 时按实际注意力层数计，纯 SSM 返回 0；
- `KVCache` 的 `append` 不得推进 `length`；所有层 append 后 `cached_forward` 只 `advance(T)` 一次；`keys/values` 只返回已推进的历史；
- 整段预填充、分块预填充与逐 Token 解码的 logits 与 `TinyLM.forward` 误差小于 `1e-5`；缓存溢出必须触发断言；
- `cached_forward` 必须拒绝非注意力块；GQA 的 K/V 展开后头数等于查询头数；RoPE/ALiBi 下的位置必须由 `position_ids` 决定；
- `flash_attention` 与 `dense_attention` 的误差小于 `1e-5`，且改变块大小不改变结果（误差小于 `1e-6`）；全掩码块不得产生 NaN；
- `arithmetic_intensity` 与 `roofline_time` 与 (21.5)/(21.6) 逐位一致；批增大时 `roofline_time` 单调不增，且返回的 FLOPs/字节数与账本一致；
- 7B 配置在 B=1、T=4096 时算术强度约 1.1，`I_\infty` 约 31，低于 H100 的 295；该结论只用于账本，不得表述为实测加速比；
- `BlockTable.allocate` 按 `ceil(n/block_size)` 分配，`ensure_capacity` 只补差量，`free_sequence` 后块可立即复用；块不足必须抛 `MemoryError`；
- `internal_fragmentation` 在长度均为块大小整数倍时为 0；`padded_fragmentation` 不小于内部碎片率；`fragmentation_table` 的利用率与浪费率之和为 1；
- `ttft`、`tpot`、`throughput` 与 (21.17)–(21.19) 一致；`throughput(B, tpot)` 必须等于 `B / tpot`；
- `ch21_inference.py` 除耗时外必须逐字可复现；缓存等价性、FlashAttention 误差、KV 账本、碎片表与 roofline 数字必须与正文一致；
- 报告必须同时给出缓存误差、KV 账本、碎片利用率与 roofline 下界；不得只报加速比或只报单请求 token/s；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有字段；缓存以 `cached_forward` 实现，`use_cache` 走该路径；玩具 CPU 加速比不得外推为真实服务加速比。
- `acceptance_rate(p,q)` 必须等于 $$1-\mathrm{TV}(p,q)$$，在 `p=q` 时为 1、完全不相交时为 0；
- `residual_probs(p,q)` 每行和为 1，支撑集恰好是 `p>q` 的位置；`p=q` 时必须回退到 `p`，不得返回全零导致采样失败；
- 单步拒绝采样的经验分布与目标 $$p$$ 的最大偏差在 2 万次采样下小于 0.015；`p=q` 时所有候选都被接受；
- `speculative_sample` 的接受数在 `[0,γ]` 内，修正 Token 在词表内；首次拒绝之后的草稿不得被计入接受数；
- `expected_output_tokens(0,γ)==1`、`(1,γ)==γ+1`，对 $$\alpha$$ 单调不减；`speculative_speedup` 在 $$\alpha=0$$ 时等于 $$1/(1+c\gamma)$$，在 $$\alpha=1$$ 时等于 $$(\gamma+1)/(1+c\gamma)$$；
- 在固定 $$\alpha$$ 与 $$c$$ 下，加速比必须出现内部最优 $$\gamma$$，不得随 $$\gamma$$ 单调递增；
- `truncate_cache(cache, L+a)` 后的下一次前向必须与“用已接受前缀重新预填充”的 logits 一致（误差小于 `1e-6`），且 `cache.length` 只增加新 Token 数；
- `tree_attention_mask` 必须包含对角线且只包含祖先；兄弟分支互相不可见；
- `MTPHead` 的每个头输出形状为 `(B,T,V)`；`mtp_loss` 在 `weights=[1,0,0]` 时等于仅第一头的交叉熵；`ignore_index=-100` 的填充位置不得贡献损失；
- `speculative_decode` 在固定种子下两次运行逐位一致，返回长度恰为 `start+max_new_tokens`；批量不为 1 时必须抛 `ValueError`；
- `ch22_speculative.py` 除耗时外必须逐字可复现；无损性误差、接受率、加速比账本、接受数模拟、回滚误差与 MTP 损失必须与正文一致；
- 报告必须同时给出经验分布误差、接受率与理论值、加速比与最优 $$\gamma$$；不得只报加速比或只报一个负载的接受率；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有字段；`SpecConfig` 只作推理参数；玩具马尔可夫链与小模型的结论不得外推为真实模型的加速比。
- `SearchConfig` 必须拒绝 `max_samples < 1`、`max_tokens < 1`、`temperature <= 0`、非法 `vote`、`min_samples < 2`、`stop_margin < 0`、`c_puct <= 0`；
- `coverage_probability(p,1)==p`、`coverage_probability(p,n)` 对 $$n$$ 单调不减，且在 `p=0`/`p=1` 时分别为 0/1；
- `pass_at_k(n,c,k)` 在 `k=1` 时等于 `c/n`，在 `n-c<k` 时等于 1，在 `c=0` 时等于 0；`k>n` 必须报错；
- `majority_accuracy(p,n)` 在 `p=0.5` 时恒为 0.5；`p>0.5` 时随 $$n$$ 增大，`p<0.5` 时随 $$n$$ 减小；$$n$$ 必须为奇数；
- `majority_vote` 平分时取最早出现的答案；加权投票的 `scores` 等于权重和，`counts` 等于出现次数；
- `selection_accuracy(p,n,1)==coverage_probability(p,n)`、`selection_accuracy(p,n,0.5)==p`（误差小于 `1e-9`）；`q` 越界必须报错；
- `self_consistency` 在预算耗尽时返回 `n_samples==max_samples`；满足停止条件时 `stopped_early==True` 且 `n_samples<=max_samples`；
- `token_cost(cfg, prompt_tokens)` 必须等于 `max_samples*(prompt_tokens+max_tokens)`；
- `puct_score` 与 (24.7) 逐位一致；`TreeNode.q` 在零访问时返回 0，否则等于价值平均；
- `tree_search` 在 `budget=0` 时 `evaluations==0` 且不崩溃；在陷阱树上预算 12 必须找到价值 1.0 的节点；
- `ch24_search.py` 除 TinyLM 训练耗时外必须逐字可复现；覆盖率、投票、验证器、陷阱树、链式算术与提前停止的数字必须与正文一致；
- 报告必须同时给出覆盖率、选择率与预算（样本数/Token）；不得只报 pass@k 或只报最终准确率；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有字段；`SearchConfig` 只作推理参数；玩具结论不得外推为真实模型的温度、采样预算或验证器精度。
- `VisionConfig` 必须拒绝 `image_size` 不被 `patch_size` 整除、`d_vision` 不被 `n_heads` 整除、非法 `pool`、非法 `projector`，以及 `resampler` 下 `n_queries < 1`；
- `image_token_count(12,3)==16`、`image_token_count(12,3,2)==4`、`image_token_count(336,14)==576`、`image_token_count(448,14,2)==256`；合并数不整除块数必须报错；
- `attention_multiplier(0,T)==1`、`attention_multiplier(N,T)==(1+N/T)^2`；文本长度为 0 必须报错；
- `PatchEmbed` 的输出必须等于 `unfold` 后按 `patch_size` 展平再加权的线性投影（误差小于 `1e-6`）；
- `VisionEncoder` 在 `pool="tokens"` 时返回 `(B,N_p,d_v)`，在 `pool="cls"` 时返回 `(B,d_v)`；
- `Projector` 的 `linear`/`mlp` 保持 Token 数不变，`resampler` 把 Token 数压到 `n_queries`；
- `MultimodalTinyLM.forward` 在给定 `pixel_values` 时必须改变 logits；每行占位符数与视觉 Token 数不等时必须抛 `ValueError`；
- 反向传播必须到达 `projector` 与视觉塔；视觉塔参数 `requires_grad_(False)` 时其梯度必须为 `None`，而投影层仍有梯度；
- 只给最后一个位置标签时，`out.loss` 必须等于该位置 logits 的交叉熵（误差小于 `1e-6`）；
- `contrastive_loss` 必须对称，对齐配对的损失小于错位配对，非正温度必须报错；
- `vq_quantize` 的索引必须是平方欧氏距离的 argmin，`vq_loss(zq,zq)==0`，维度不匹配必须报错；
- `ch25_multimodal.py` 除训练耗时外必须逐字可复现；Token 预算、InfoNCE、VQ 与五组视觉问答的数字必须与正文一致；
- 报告必须同时给出图像 Token 数、注意力倍数、参数量与至少 3 个种子的均值/标准差；不得只报最终准确率；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有字段；`VisionConfig` 不进入 `ModelConfig`；12×12 合成任务与随机初始化视觉塔的结论不得外推为真实多模态模型的分辨率、投影层或冻结策略。
- `AgentConfig` 必须拒绝 `max_steps<1`、`max_context_tokens<1`、`max_repeat<1`、`max_observation_chars<1`；
- `ToolSpec` 必须拒绝非法工具名、不可调用函数与非 `str/int/float/bool` 的参数类型；
- `parse_action` 必须接受“推理 + 单个 JSON 对象”，拒绝无 JSON、同时含 `answer` 与 `tool`、`tool` 非字符串、`args` 非对象；
- `ToolRegistry.execute` 必须把未知工具、缺参、多参、类型错误与工具异常都转成 `{"ok": false, ...}`，且非法参数不得调用底层函数；
- 同一 `(工具名, 参数 JSON)` 超过 `max_repeat` 时必须返回 `repeated call rejected`；`max_steps`、`max_context_tokens` 与 `stop_on_tool_error` 必须分别以 `max_steps`、`context_limit`、`tool_error` 停止；
- 观测超过 `max_observation_chars` 时必须返回 `truncated` 预览，且原始内容不得整段进入上下文；
- `trajectory_cost` 必须等于 `c_in * ΣL_t + c_out * Σ|a_t|`；`result_cost` 必须与逐步记录一致；长度不等必须报错；
- `retry_success(p,b)` 必须等于 `1-(1-p)^b`，`task_success(p,b,K)` 必须等于其 `K` 次幂；`p` 越界与 `b<1`/`K<1` 必须报错；
- `value_of_information` 必须等于 `(u_after-u_before)/cost`，`cost<=0` 必须报错；
- `ch26_agent.py` 必须逐字可复现（蒙特卡洛固定种子）；重试/任务成功率、深度—成本表、VoI 与注入对照的数字必须与正文一致；
- 报告必须同时给出步数、上下文 Token、轨迹成本与停止原因；不得只报最终答案或成功率；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有字段；`AgentConfig` 只作循环参数，策略通过 `Callable[[str], str]` 注入；脚本策略与 2 万次玩具模拟的结论不得外推为真实模型的任务成功率。
- `accuracy` 与 `macro_f1` 必须拒绝空输入与长度不一致；macro-F1 在各类全对时为 1，在少数类全错时下降；
- `pass_at_k` 必须满足 `c=0 → 0`、`n-c<k → 1`，并与组合数定义逐位一致；`k` 越界必须报错；
- `binomial_se(p,n)` 必须等于 `sqrt(p(1-p)/n)`；`p∉[0,1]` 或 `n<1` 必须报错；
- `wilson_interval` 必须包含点估计，且在 0/10 与 10/10 时贴住 `[0,1]` 边界；越界计数必须报错；
- `paired_bootstrap_diff` 必须对逐样本差值重采样；全正差时区间退化为常数，长度不一致必须报错；
- `mcnemar_test(b,c)` 在 `b=c` 时为 1，在 `b=0,c=10` 时等于精确二项尾概率；负计数必须报错；
- `expected_calibration_error` 在完美校准时为 0，置信度越界必须报错；分箱数必须为正；
- `contamination_rate` 必须统计“至少命中一个 n-gram”的评测文档比例；空评测集必须报错；`n` 必须为正；
- `bradley_terry_prob` 在等分时为 0.5、分差 100 时约 0.640；`elo_update` 在等分对局时各变动 `K/2`；
- `cohen_kappa` 在完全一致时为 1、在 80% 一致与 50% 机会一致时为 0.6；空输入必须报错；
- `cost_aware_utility` 必须等于 `score - λ·cost`；负成本或负 λ 必须报错；
- `ch27_eval.py` 必须逐字可复现（bootstrap 与模拟固定种子）；配对差、pass@k、污染、位置偏差、ECE、kappa 与效用的数字必须与正文一致；
- 报告必须同时给出样本量、区间或 p 值、评测协议与污染检查；不得只报一个裸分；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有字段；评测函数只接收逐样本结果；构造的确定性数据与模拟结论不得外推为真实基准分数。
- `logit_margin` 必须等于目标与干扰 Token 的对数概率之差，形状为 `(B,)`；目标与干扰相同时必须为 0；
- `linear_probe_accuracy` 在可分数据上接近 1，在打乱标签上接近 `1/C`；训练/测试必须随机切分，不得按类别切分；
- `probe_selectivity` 必须等于探针准确率减对照准确率；输入越界必须报错；
- `attention_maps` 每头权重最后一维和为 1、上三角为 0；缺少 `q_proj`/`k_proj`/`n_heads`/`n_kv_heads`/`d_head` 时必须抛 `NotImplementedError`（覆盖 MLA）；
- `ablate_heads`、`patched_output`、`ablated_direction` 退出上下文后必须逐位恢复原输出；异常路径也不得留下 hook；
- `recovery` 在 `S_patch=S_clean` 时为 1、在 `S_patch=S_corrupt` 时为 0；`S_clean==S_corrupt` 必须报错；
- `SparseAutoencoder` 前向形状为 `(N,K)`，训练后解码器列范数接近 1；`train_sae` 增大 `l1` 时平均 L0 必须下降；
- `sae_metrics` 必须同时返回 MSE、平均 L0 与死特征数；`feature_purity` 必须按每个样本的最大激活特征归属类别并取多数类比例；
- `project_out` 后方向上的投影为零（误差小于 `1e-6`）；`ablated_direction` 必须只改指定位置的激活；
- `ch28_interpret.py` 除训练耗时外必须逐字可复现；探针、注意力、消融、修补与 SAE 的数字必须与正文一致；
- 报告必须同时给出干预位点、损坏方式、评价指标、样本量与对照；不得只报注意力图或单个恢复率；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有字段；解释工具只读或临时 hook；复制任务与 2 层玩具模型的结论不得外推为真实 LLM 的回路结构。
- `continual_run` 必须固定任务顺序（先 A 后 B）与每任务数据量；报告任务 A 的前/后 MSE、任务 B 的 MSE 与遗忘量；3 个种子必须给均值与标准差；
- EWC 惩罚必须为 `0.5 * lam * Σ F_i (θ_i-θ_i*)^2` (29.5)，`fisher_diag` 必须按样本累加梯度平方后再除以样本数；
- 回放必须按 `rho` 概率混入旧任务样本，损失为 `(1-rho)L_B + rho L_A` (29.6)；报告 `rho` 与存储假设；
- `collapse_trace` 的方差递推必须满足 `sigma_{t+1}^2 = ((N-1)/N) sigma_t^2` (29.8)；`alpha>0` 时必须收敛到 `sigma_real^2` (29.9)；报告必须说明是“方差均值的开方”，不是“标准差的均值”；
- `sparse_rl` 的终端奖励必须只在全部步骤成功时给 1 (29.1)–(29.2)；`potential` 模式的塑形奖励必须能写成 `gamma*Phi(s_{t+1})-Phi(s_t)` (29.4)，`gamma=1` 时与“每步 1 分”只差一个与动作无关的常数；
- 报告必须同时给出解析成功率 `prod p_t` 与经验成功率，并注明采样次数；不得只报策略参数的乘积；
- `ch29_frontier.py` 必须逐字可复现（固定种子），三组实验的输出必须与正文 `text` 块一致；
- 本章不得修改 `TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有字段；示例为独立脚本，不进入 `tinylm/` 包；玩具任务的结论不得外推为真实 LLM 的遗忘率、数据退化速度或 RL 样本效率。
