# TinyLM 状态

本文件记录 TinyLM 的模块、接口与配置变更，供分会话写作交接。接口契约以[附录 F](appendix/f-tinylm.md) 为准。

## 当前状态

- **公共 API**：`tinylm/tokenizer/bpe.py` 的核心实现已在第 2 章给出，接口与附录 F 的 `Tokenizer` 协议一致；但代码目前以章节内嵌形式存在，尚未落盘到包目录。
- **模型**：第 3 章的 Transformer 基线已在章节内给出完整实现，接口与附录 F 对齐；同样尚未落盘。
- **位置编码**：第 4 章在基线上增加 `sine`、`rope`、`nope`、`alibi` 四种 `pos_encoding`，并接入 PI/NTK/YaRN 三种 RoPE 缩放；代码仍以章节内嵌形式存在。
- **接口契约**：未修改。`TinyLM.forward` 保持附录 F 的签名，新增字段只出现在 `ModelConfig` 与 `CausalLMOutput` 中。
- **配置**：`ModelConfig` 首次落地（见第 3 章变更）。`pos_encoding` 与 `rope_theta` 在第 4 章启用，新增 `rope_scaling`；部分字段仍为后续章节预留。

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

## 下一章待实现

第 5 章《现代 LLM 的架构演进》要在第 3、4 章基线上增加：

| 文件 | 内容 |
| --- | --- |
| `tinylm/norm.py` | RMSNorm，替换 `LayerNorm`，由 `norm="rmsnorm"` 分派 |
| `tinylm/ffn.py` | SwiGLU 三个投影与 `d_ff` 重算，由 `ffn="swiglu"` 分派 |
| `tinylm/model.py` | 支持 `tie_embeddings`、可选偏置、并行注意力/前馈层的消融 |
| `tinylm/config.py` | 启用 `norm`、`ffn`、`dropout` 等预留字段 |
| `tests/test_modern.py` | RMSNorm 与 LayerNorm 的关系、SwiGLU 参数量、权重绑定、并行层等价性 |
| `examples/ch05_ablation.py` | 相同参数量与算力预算下的架构消融 |

第 5 章不得修改 `TinyLM.forward` 签名；`pos_encoding` 的既有分支必须保持可用。

## 验收门槛

- `BPETokenizer` 满足 `Tokenizer` 协议；
- `decode(encode(s)) == s` 对普通文本与任意 UTF-8 字节序列成立；
- 特殊 Token 不会被普通合并拆开；
- 保存/加载后的分词结果完全一致；
- 模型配置中的 `vocab_size` 与实际词表大小一致；
- 第 3 章的嵌入矩阵与输出层维度均能由 `vocab_size` 正确推导；
- 因果性测试通过：位置 $t$ 的 logits 只依赖前 $t$ 个 Token；
- 注意力权重每行和为 1，且上三角严格为 0；
- 精确参数公式 (3.22) 与实测参数量逐位一致；
- 切换位置编码实现时，`TinyLM.forward` 签名不得改变；
- RoPE 保范数：旋转前后 Q、K 的逐 Token 范数误差小于 `1e-5`；
- RoPE 相对性：同一对 Q/K 平移相同距离后，打分矩阵对应元素误差小于 `1e-4`；
- 位置显式由 `position_ids` 传入，缓存与打包序列复用同一路径；
- `rope="pi"` 只缩放位置，`"ntk"` 改基频，`"yarn"` 分频缩放并加温度；三者的频率表均与公式逐项一致；
- 任何位置编码改动都必须通过因果性测试：改动位置 `t` 的输入，`t` 之前的 logits 不变。
