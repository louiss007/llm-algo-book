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

## 下一章待实现

第 9 章《缩放定律》在第 8 章基线上不需要新增模型模块，重点是实验与拟合：

| 文件 | 内容 |
| --- | --- |
| `tinylm/scaling.py` | 幂律拟合、计算最优分配、Chinchilla 风格的参数/数据配比 |
| `tests/test_scaling.py` | 幂律拟合在合成数据上的参数恢复、计算最优解的边界条件 |
| `examples/ch09_scaling.py` | 小模型扫描、损失-计算曲线、不同架构的每 Token FLOPs 口径对照 |

第 9 章不得修改 `TinyLM.forward` 与 `CausalLMOutput`；若引入架构相关的 FLOPs 口径，必须复用第 8 章的 `decode_flops` 与 `state_bytes`，不得另起一套账本。

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
