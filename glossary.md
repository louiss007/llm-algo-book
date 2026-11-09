# 术语登记簿

权威版本为[附录 C](appendix/c-glossary.md)。本文件只登记每章新增术语，便于分会话交接。

## 第 1 章

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 词元 | Token | 模型处理文本的最小单位 | 第 1 章 |
| 自回归语言建模 | Autoregressive Language Modeling | 从左到右逐个预测下一个 Token | 第 1 章 |
| 因果语言模型 | Causal Language Model | 每个位置只能看到左侧上下文 | 第 1 章 |
| 交叉熵 | Cross-Entropy | 衡量模型分布与真实分布差异的损失 | 第 1 章 |
| 困惑度 | Perplexity | 交叉熵的指数，表示有效分支因子 | 第 1 章 |
| 压缩假说 | Compression Hypothesis | 下一 Token 预测等价于数据压缩，并与智能相关 | 第 1 章 |
| n 元模型 | n-gram Model | 用前 $$n-1$$ 个 Token 的计数预测下一个 | 第 1 章 |
| 教师强制 | Teacher Forcing | 训练时用真实前文作为输入 | 第 1 章 |
| 暴露偏差 | Exposure Bias | 训练与推理输入不一致导致的误差累积 | 第 1 章 |
| 马尔可夫假设 | Markov Assumption | 只依赖最近 $$n-1$$ 个 Token | 第 1 章 |
| 加性平滑 | Additive Smoothing | 给计数加常数以避免零概率 | 第 1 章 |
| 神经语言模型 | Neural Language Model | 用参数化函数估计条件概率 | 第 1 章 |
| 分布式表示 | Distributed Representation | 用低维稠密向量表示词并共享统计强度 | 第 1 章 |
| 循环神经网络 | Recurrent Neural Network (RNN) | 用隐状态逐步传递历史 | 第 1 章 |
| 填充中间 | Fill-in-the-Middle (FIM) | 根据前缀与后缀生成中间片段 | 第 1 章 |
| 多 Token 预测 | Multi-Token Prediction (MTP) | 一次前向预测未来多个 Token | 第 1 章 |

## 第 2 章

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 分词 | Tokenization | 把字符串切分为 Token 并映射为整数 ID 的过程 | 第 2 章 |
| 子词 | Subword | 介于字符与词之间的文本单元，是子词分词器的基本输出 | 第 2 章 |
| 词表 | Vocabulary | Token 到整数 ID 的完整映射表 | 第 2 章 |
| 特殊 Token | Special Token | 承担控制语义的保留 Token，如 BOS、EOS、UNK | 第 2 章 |
| 字节对编码 | Byte Pair Encoding (BPE) | 从字节或字符出发、反复合并最高频相邻对的分词算法 | 第 2 章 |
| 字节级 BPE | Byte-level BPE | 以 256 个字节为基础词表的 BPE，可覆盖任意 UTF-8 串 | 第 2 章 |
| WordPiece | WordPiece | 以点互信息近似分数为合并准则的子词分词算法，BERT 使用 | 第 2 章 |
| Unigram 语言模型 | Unigram Language Model | 从大候选词表出发、用 EM 训练并按概率剪枝的子词分词算法 | 第 2 章 |
| 点互信息 | Pointwise Mutual Information (PMI) | 衡量两个事件共同出现相对独立假设偏离程度的量 | 第 2 章 |
| 子词正则化 | Subword Regularization | 训练时对同一句子采样不同子词切分的数据增强方法 | 第 2 章 |
| 最长匹配 | Longest-Match | 从左到右每次匹配词表中最长子串的贪心编码方法 | 第 2 章 |
| 未登录词 | Out-of-Vocabulary (OOV) | 词表中没有、无法直接编码的文本单元 | 第 2 章 |
| 压缩比 | Compression Ratio | 字节数与 Token 数之比，衡量分词效率 | 第 2 章 |
| 生育率 | Fertility | 平均每个词被切成多少个 Token | 第 2 章 |
| 每字节比特数 | Bits per Byte | 交叉熵换算到每个字节的编码长度，可跨分词器比较 | 第 2 章 |
| 词表扩展 | Vocabulary Expansion | 在已有模型上加入新 Token 并继续预训练 | 第 2 章 |
| SentencePiece | SentencePiece | 直接在原始文本上训练、与语言无关的分词工具，支持 BPE 与 Unigram | 第 2 章 |

## 第 3 章

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| Transformer | Transformer | 以自注意力为核心、可并行处理整个序列的架构 | 第 3 章 |
| 自注意力 | Self-Attention | 序列内每个位置对其他位置加权聚合信息的机制 | 第 3 章 |
| 缩放点积注意力 | Scaled Dot-Product Attention | 用点积算相似度并除以 $$\sqrt{d_k}$$ 的注意力形式 | 第 3 章 |
| 因果掩码 | Causal Mask | 阻止位置看到未来 Token 的上三角掩码 | 第 3 章 |
| 多头注意力 | Multi-Head Attention (MHA) | 并行运行多组注意力并拼接输出的机制 | 第 3 章 |
| 头维度 | Head Dimension | 单个注意力头的特征维度 $$d_h$$ | 第 3 章 |
| 前馈网络 | Feed-Forward Network (FFN) | 对每个位置独立作用的两层 MLP | 第 3 章 |
| 残差连接 | Residual Connection | 把子层输出与输入相加的操作 | 第 3 章 |
| 残差流 | Residual Stream | 贯穿所有层、被各子层读写的共享表示通道 | 第 3 章 |
| 层归一化 | Layer Normalization (LN) | 对每个 Token 的隐藏向量沿特征维归一化 | 第 3 章 |
| 前置归一化 | Pre-LN | 把归一化放在子层之前的结构 | 第 3 章 |
| 后置归一化 | Post-LN | 把归一化放在残差相加之后的结构 | 第 3 章 |
| 排列等变性 | Permutation Equivariance | 输入排列后输出按同样方式排列的性质 | 第 3 章 |
| 键值记忆 | Key-Value Memory | 把前馈层解释为键模式与值输出的记忆视角 | 第 3 章 |
| 位置编码 | Positional Encoding | 向模型注入 Token 顺序信息的机制 | 第 3 章 |
| 绝对位置编码 | Absolute Positional Encoding | 为每个绝对位置分配一个编码向量 | 第 3 章 |

## 第 4 章

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 正弦位置编码 | Sinusoidal Positional Encoding | 用不同频率的正弦与余弦函数生成的无参数位置编码 | 第 4 章 |
| 相对位置编码 | Relative Positional Encoding | 只编码位置之间相对距离的位置编码 | 第 4 章 |
| 旋转位置编码 | Rotary Position Embedding (RoPE) | 通过旋转查询与键使内积只依赖相对位移的位置编码 | 第 4 章 |
| 相对位移性质 | Relative Shift Property | 同一对查询与键平移相同距离后内积不变的性质 | 第 4 章 |
| ALiBi | Attention with Linear Biases | 在注意力打分上按距离加线性惩罚的位置编码 | 第 4 章 |
| 无位置编码 | No Positional Encoding (NoPE) | 不显式注入位置、依赖因果结构隐式恢复顺序的方案 | 第 4 章 |
| 长度外推 | Length Extrapolation | 在比训练长度更长的序列上保持性能的能力 | 第 4 章 |
| 位置插值 | Position Interpolation (PI) | 把超长位置线性压缩回训练范围的扩展方法 | 第 4 章 |
| NTK-aware 缩放 | NTK-aware Scaling | 通过调整 RoPE 基频、按频率拉伸波长的扩展方法 | 第 4 章 |
| 分频插值 | NTK-by-Parts | 按波长比例分段决定哪些频率插值的 YaRN 组件 | 第 4 章 |
| YaRN | YaRN | 结合分频插值与温度缩放的长度外推方法 | 第 4 章 |
| 有效上下文 | Effective Context Length | 任务指标仍不低于阈值的最长序列长度 | 第 4 章 |

## 术语一致性约定

- 正文统一使用 "Token"，首次出现时注明"词元"；不混用"标记""词块"。
- "困惑度"统一对应 perplexity，不写作"复杂度"。
- "教师强制"与"暴露偏差"成对出现，分别对应训练与推理的差异。
- "分词器"统一对应 tokenizer，"分词"对应 tokenization；不混用"切词""编码器"（编码器专指 `Enc`）。
- "词表"对应 vocabulary；在分词语境不写作"词典"。
- "生育率"统一对应 fertility，不写作"词元字比"。
- "未登录词"对应 OOV；`<unk>` 是特殊 Token，不把 OOV 与 `<unk>` 混为一谈。
- "字节级 BPE"统一对应 byte-level BPE；首次出现附英文，之后用中文。
- "自注意力"指序列内部的注意力；"交叉注意力"专指解码器读取编码器输出的注意力，不混用。
- "残差连接"是操作，"残差流"是贯穿各层的张量；两者不互换。
- "多头注意力"统一对应 MHA；当 KV 头数少于查询头数时，按第 6 章写作 GQA 或 MQA，不笼统称为多头。
- "前馈网络"统一对应 FFN，不写作"全连接层"；中间维度统一用 $$d_{ff}$$。
- "位置编码"是机制，"位置嵌入"是可学习的查表实现；首次出现时注明二者关系。
- "RoPE" 固定为全大写，首次出现附中文"旋转位置编码"，之后可直接用缩写；不写作 "rope"。
- "ALiBi" 固定大小写，不展开为 "alibi"；首次出现附中文"带线性偏置的注意力"。
- "外推"指在训练长度之外测试，"内插"指把长位置映射回训练范围；"扩展上下文"是工程目标，不替代这两个术语。
- "有效上下文"统一对应 effective context length，不写作"实际上下文"或"真实窗口"。
- "位置插值"指 PI 方法，不泛指任何位置缩放；"分频插值"专指 YaRN 的 NTK-by-parts 组件。
- Pre-LN/Post-LN 首次出现附中文"前置/后置归一化"，之后可混用英文缩写。
