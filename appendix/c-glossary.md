# 附录 C 术语表（中英对照）

按主题分组，每组内按首次出现的章节顺序排列。"首次出现"指本书正文中第一次系统定义该术语的章节。

{% hint style="info" %}
同一个英文术语在不同文献中可能有不同译法（例如 token 也译作"词元"、context 也译作"语境"）。本书的选择以本表为准，正文首次出现时会附英文原文。
{% endhint %}

## C.1 语言建模与分词

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 词元 | Token | 模型处理文本的最小单位，通常是一个子词片段 | 第 1 章 |
| 自回归语言建模 | Autoregressive Language Modeling | 从左到右逐个预测下一个 Token 的建模方式 | 第 1 章 |
| 因果语言模型 | Causal Language Model | 每个位置只能看到左侧上下文的语言模型 | 第 1 章 |
| 交叉熵 | Cross-Entropy | 衡量模型分布与真实分布差异的损失函数 | 第 1 章 |
| 困惑度 | Perplexity (PPL) | 交叉熵的指数，可理解为每一步的平均候选数 | 第 1 章 |
| 压缩假说 | Compression Hypothesis | 把下一 Token 预测视为数据压缩，并认为压缩能力与智能相关 | 第 1 章 |
| n 元模型 | n-gram Model | 用前 $$n-1$$ 个 Token 的统计频率预测下一个 Token | 第 1 章 |
| 填充中间 | Fill-in-the-Middle (FIM) | 将文本切为前、中、后三段，训练模型补全中间部分 | 第 1 章 |
| 多 Token 预测 | Multi-Token Prediction (MTP) | 一次前向同时预测未来多个 Token 的训练目标 | 第 1 章 |
| 教师强制 | Teacher Forcing | 训练时用真实前文而非模型自己的输出作为输入 | 第 1 章 |
| 暴露偏差 | Exposure Bias | 训练与推理的输入分布不一致导致的误差累积 | 第 1 章 |
| 马尔可夫假设 | Markov Assumption | 假设下一个 Token 只依赖最近 $$n-1$$ 个 Token | 第 1 章 |
| 神经语言模型 | Neural Language Model | 用参数化函数而非计数表估计条件概率的语言模型 | 第 1 章 |
| 分布式表示 | Distributed Representation | 用低维稠密向量表示词，使相似词共享统计强度 | 第 1 章 |
| 循环神经网络 | Recurrent Neural Network (RNN) | 用隐状态逐步传递历史信息的序列模型 | 第 1 章 |
| 加性平滑 | Additive Smoothing | 给每个计数加一个常数以避免零概率 | 第 1 章 |
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

## C.2 Transformer 基础架构

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| Transformer | Transformer | 以自注意力为核心、可并行处理整个序列的架构 | 第 3 章 |
| 自注意力 | Self-Attention | 序列中每个位置对其他位置加权聚合信息的机制 | 第 3 章 |
| 缩放点积注意力 | Scaled Dot-Product Attention | 用点积计算相似度并除以 $$\sqrt{d_k}$$ 的注意力形式 | 第 3 章 |
| 因果掩码 | Causal Mask | 阻止位置看到未来 Token 的上三角掩码 | 第 3 章 |
| 多头注意力 | Multi-Head Attention (MHA) | 并行运行多组独立注意力的机制 | 第 3 章 |
| 前馈网络 | Feed-Forward Network (FFN) | 对每个位置独立作用的两层 MLP | 第 3 章 |
| 残差连接 | Residual Connection | 把子层输出与输入相加，缓解深层梯度消失 | 第 3 章 |
| 残差流 | Residual Stream | 贯穿所有层、被各子层读写的共享表示通道 | 第 3 章 |
| 层归一化 | Layer Normalization (LN) | 对每个 Token 的隐藏向量做归一化 | 第 3 章 |
| 前置归一化 | Pre-LN | 把归一化放在子层之前的结构 | 第 3 章 |
| 后置归一化 | Post-LN | 把归一化放在残差相加之后的结构 | 第 3 章 |
| 排列等变性 | Permutation Equivariance | 输入排列后输出按同样方式排列的性质 | 第 3 章 |
| 头维度 | Head Dimension | 单个注意力头的特征维度 $$d_h$$ | 第 3 章 |
| 键值记忆 | Key-Value Memory | 把前馈层解释为键模式与值输出的记忆视角 | 第 3 章 |
| 均方根归一化 | RMSNorm | 只除以均方根、不减均值的归一化 | 第 5 章 |
| SwiGLU | SwiGLU | 用 SiLU 门控的 GLU 前馈网络变体 | 第 5 章 |
| 权重绑定 | Weight Tying | 输入嵌入与输出投影共享同一矩阵 | 第 5 章 |
| 并行层 | Parallel Layers | 注意力与前馈并行计算而非串行 | 第 5 章 |

## C.3 位置编码

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 位置编码 | Positional Encoding | 向模型注入 Token 顺序信息的机制 | 第 3 章 |
| 绝对位置编码 | Absolute Positional Encoding | 为每个绝对位置分配一个编码向量 | 第 3 章 |
| 相对位置编码 | Relative Positional Encoding | 只编码位置之间的相对距离 | 第 4 章 |
| 旋转位置编码 | Rotary Position Embedding (RoPE) | 通过旋转查询与键来注入相对位置信息 | 第 4 章 |
| ALiBi | Attention with Linear Biases | 用与距离成正比的线性偏置惩罚远距离注意力 | 第 4 章 |
| 位置插值 | Position Interpolation (PI) | 把超长位置压缩到训练范围内以扩展上下文 | 第 4 章 |
| NTK-aware 缩放 | NTK-aware Scaling | 按频率分段调整 RoPE 基频的外推方法 | 第 4 章 |
| YaRN | YaRN | 结合温度缩放与分段频率调整的长度外推方法 | 第 4 章 |
| 无位置编码 | NoPE | 完全不显式编码位置，依赖因果结构隐式推断 | 第 4 章 |

## C.4 注意力变体与 KV Cache

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 多查询注意力 | Multi-Query Attention (MQA) | 所有查询头共享一组键值头 | 第 6 章 |
| 分组查询注意力 | Grouped-Query Attention (GQA) | 查询头分组、每组共享一组键值头 | 第 6 章 |
| 多头潜在注意力 | Multi-head Latent Attention (MLA) | 把键值压缩到低维潜在向量再投影的注意力 | 第 6 章 |
| KV 缓存 | KV Cache | 缓存历史键值以避免重复计算 | 第 6 章 |
| 滑动窗口注意力 | Sliding Window Attention | 每个位置只关注最近 $$W$$ 个位置 | 第 6 章 |
| 注意力汇 | Attention Sink | 模型倾向把大量注意力分配给首个 Token 的现象 | 第 6 章 |

## C.5 混合专家与替代架构

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 混合专家 | Mixture of Experts (MoE) | 用路由器为每个 Token 选择少数专家计算 | 第 7 章 |
| 路由器 | Router | 为每个 Token 计算专家权重的轻量网络 | 第 7 章 |
| Top-K 路由 | Top-K Routing | 只激活权重最高的 $$k$$ 个专家 | 第 7 章 |
| 负载均衡损失 | Load Balancing Loss | 惩罚专家负载不均的辅助损失 | 第 7 章 |
| 共享专家 | Shared Expert | 所有 Token 都会经过的公共专家 | 第 7 章 |
| 专家并行 | Expert Parallelism | 把不同专家分布到不同设备上 | 第 7 章 |
| 容量因子 | Capacity Factor | 每个专家可接收 Token 数的上限系数 | 第 7 章 |
| Token 丢弃 | Token Dropping | 专家超出容量时丢弃多余 Token | 第 7 章 |
| 线性注意力 | Linear Attention | 把注意力重写为线性复杂度的核方法 | 第 8 章 |
| 状态空间模型 | State Space Model (SSM) | 用线性递推状态描述序列的模型族 | 第 8 章 |
| S4 | Structured State Space Sequence Model | 用结构化矩阵参数化状态转移的 SSM | 第 8 章 |
| Mamba | Mamba | 引入输入相关选择机制的 SSM 架构 | 第 8 章 |
| 选择性扫描 | Selective Scan | Mamba 中按输入动态调整状态的并行扫描算法 | 第 8 章 |
| RWKV | Receptance Weighted Key Value | 结合 RNN 推理与 Transformer 训练并行的架构 | 第 8 章 |
| 混合架构 | Hybrid Architecture | 交错使用注意力层与 SSM/线性层的模型 | 第 8 章 |

## C.6 预训练

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 缩放定律 | Scaling Law | 损失随参数量、数据量、算力变化的经验规律 | 第 9 章 |
| Kaplan 定律 | Kaplan Scaling Law | 早期提出的幂律缩放关系 | 第 9 章 |
| Chinchilla 定律 | Chinchilla Scaling Law | 指出最优数据量应随参数量同比增长的缩放关系 | 第 9 章 |
| 计算最优 | Compute-Optimal | 在固定算力下使损失最小的参数/数据配比 | 第 9 章 |
| 推理最优 | Inference-Optimal | 考虑推理成本后选择的更小模型、更多数据配比 | 第 9 章 |
| 涌现能力 | Emergent Abilities | 规模超过阈值后突然出现的能力 | 第 9 章 |
| μP | Maximal Update Parametrization | 使超参数可跨宽度迁移的参数化方案 | 第 9 章 |
| 数据去重 | Data Deduplication | 移除重复或近重复样本来提升数据效率 | 第 10 章 |
| 质量分类器 | Quality Classifier | 给样本打分并过滤低质数据的模型 | 第 10 章 |
| 数据配比 | Data Mixture | 各来源数据在训练中的比例 | 第 10 章 |
| 数据课程 | Data Curriculum | 按难度或阶段安排数据顺序 | 第 10 章 |
| 数据退火 | Data Annealing | 训练末期切换到高质量数据以提升效果 | 第 10 章 |
| 合成数据 | Synthetic Data | 由模型生成而非人工采集的训练数据 | 第 10 章 |
| 数据污染 | Data Contamination | 评测数据出现在训练集中导致分数虚高 | 第 10 章 |
| AdamW | Adam with Decoupled Weight Decay | 把权重衰减与梯度更新解耦的 Adam | 第 11 章 |
| 学习率调度 | Learning Rate Schedule | 学习率随训练步数变化的规则 | 第 11 章 |
| 预热 | Warmup | 训练初期让学习率从很小线性上升 | 第 11 章 |
| 余弦退火 | Cosine Decay | 学习率按余弦曲线衰减到接近零 | 第 11 章 |
| WSD 调度 | Warmup-Stable-Decay | 先升、再恒定、最后快速衰减的调度 | 第 11 章 |
| Muon | Muon | 对动量矩阵做正交化的优化器 | 第 11 章 |
| 损失尖峰 | Loss Spike | 训练中损失突然大幅上升的现象 | 第 11 章 |
| 梯度裁剪 | Gradient Clipping | 限制梯度范数以防止更新过大 | 第 11 章 |
| 梯度累积 | Gradient Accumulation | 多次前向反向后再更新一次参数 | 第 12 章 |
| 序列打包 | Sequence Packing | 把多条短序列拼进一个定长样本以减少填充 | 第 12 章 |
| 数据并行 | Data Parallelism (DP) | 每卡持有完整模型、分摊不同数据 | 第 12 章 |
| 张量并行 | Tensor Parallelism (TP) | 把单个矩阵运算切分到多卡 | 第 12 章 |
| 流水线并行 | Pipeline Parallelism (PP) | 把不同层放到不同设备并按微批流水执行 | 第 12 章 |
| 序列并行 | Sequence Parallelism (SP) | 沿序列维度切分激活与计算 | 第 12 章 |
| ZeRO | Zero Redundancy Optimizer | 分片存储优化器状态、梯度与参数 | 第 12 章 |
| FSDP | Fully Sharded Data Parallel | PyTorch 的完全分片数据并行实现 | 第 12 章 |
| 激活检查点 | Activation Checkpointing | 只存部分激活、反向时重算以省显存 | 第 12 章 |
| 混合精度 | Mixed Precision | 同时使用低精度与高精度进行训练 | 第 12 章 |
| BF16 | BFloat16 | 指数范围与 FP32 相同、尾数更短的浮点格式 | 第 12 章 |
| FP8 | 8-bit Floating Point | 训练或推理用的 8 位浮点格式 | 第 12 章 |
| 长上下文 | Long Context | 上下文窗口显著超过训练常见长度的设定 | 第 13 章 |
| 长度外推 | Length Extrapolation | 在短序列上训练、在更长序列上保持性能 | 第 13 章 |
| 大海捞针 | Needle in a Haystack | 在长文本中检索特定信息的长上下文评测 | 第 13 章 |
| 检索增强生成 | Retrieval-Augmented Generation (RAG) | 先检索外部文档再让模型基于文档生成 | 第 13 章 |
| StreamingLLM | StreamingLLM | 保留注意力汇与近期窗口以支持无限流式输入 | 第 13 章 |

## C.7 后训练

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 监督微调 | Supervised Fine-Tuning (SFT) | 用人工标注的输入输出对继续训练模型 | 第 14 章 |
| 指令微调 | Instruction Tuning | 用指令-回答数据让模型学会遵循指令 | 第 14 章 |
| 对话模板 | Chat Template | 把多轮对话序列化为固定格式的规则 | 第 14 章 |
| 损失掩码 | Loss Masking | 只在目标回答部分计算损失 | 第 14 章 |
| 拒绝采样 | Rejection Sampling | 采样多条输出并只保留通过验证的样本 | 第 14 章 |
| 灾难性遗忘 | Catastrophic Forgetting | 学习新任务时丢失旧能力 | 第 14 章 |
| 参数高效微调 | Parameter-Efficient Fine-Tuning (PEFT) | 只训练少量新增参数完成适配 | 第 15 章 |
| LoRA | Low-Rank Adaptation | 用两个低秩矩阵的乘积近似权重更新 | 第 15 章 |
| 低秩分解 | Low-Rank Decomposition | 用秩远小于原矩阵的两个矩阵表示更新 | 第 15 章 |
| QLoRA | Quantized LoRA | 在 4 比特量化的基座上做 LoRA | 第 15 章 |
| DoRA | Weight-Decomposed Low-Rank Adaptation | 把权重分解为幅度与方向分别更新 | 第 15 章 |
| Adapter | Adapter | 插入 Transformer 层的可训练瓶颈模块 | 第 15 章 |
| 前缀微调 | Prefix Tuning | 在每层前拼接可训练的前缀向量 | 第 15 章 |
| 基于人类反馈的强化学习 | RLHF | 用人类偏好训练的奖励模型指导策略优化 | 第 16 章 |
| 奖励模型 | Reward Model | 给回答打分的模型，通常由偏好数据训练 | 第 16 章 |
| Bradley-Terry 模型 | Bradley-Terry Model | 用成对比较建模偏好概率的经典模型 | 第 16 章 |
| 策略梯度 | Policy Gradient | 用奖励的梯度估计直接优化策略 | 第 16 章 |
| TRPO | Trust Region Policy Optimization | 用信赖域约束策略更新幅度的算法 | 第 16 章 |
| PPO | Proximal Policy Optimization | 用裁剪替代硬约束的近似信赖域算法 | 第 16 章 |
| 优势函数 | Advantage Function | 动作相对平均水平好多少 | 第 16 章 |
| 广义优势估计 | Generalized Advantage Estimation (GAE) | 用多步时序差分估计优势的方法 | 第 16 章 |
| KL 惩罚 | KL Penalty | 惩罚策略偏离参考模型的项 | 第 16 章 |
| 奖励作弊 | Reward Hacking | 策略利用奖励模型漏洞获得高分 | 第 16 章 |
| AI 反馈强化学习 | RLAIF | 用 AI 而非人类提供偏好反馈 | 第 16 章 |
| 宪法 AI | Constitutional AI | 用一组原则让模型自我批评与修订 | 第 16 章 |
| 参考策略 | Reference Policy | 计算 KL 惩罚时固定的基准模型 | 第 16 章 |
| 直接偏好优化 | Direct Preference Optimization (DPO) | 直接在偏好数据上优化、无需显式奖励模型 | 第 17 章 |
| IPO | Identity Preference Optimization | 用平方损失替代 DPO 的 log-sigmoid 损失 | 第 17 章 |
| KTO | Kahneman-Tversky Optimization | 只需单条回答好坏的偏好优化方法 | 第 17 章 |
| SimPO | Simple Preference Optimization | 用长度归一化对数似然做偏好优化 | 第 17 章 |
| ORPO | Odds Ratio Preference Optimization | 把偏好项并入 SFT 损失的单阶段方法 | 第 17 章 |
| 隐式奖励 | Implicit Reward | 由策略与参考策略的对数比定义的奖励 | 第 17 章 |
| 可验证奖励强化学习 | RLVR | 用可自动验证的奖励（如答案正确性）做 RL | 第 18 章 |
| 组相对策略优化 | Group Relative Policy Optimization (GRPO) | 用同一问题多次采样的组内均值作基线 | 第 18 章 |
| 可验证奖励 | Verifiable Reward | 可由程序判定对错的奖励信号 | 第 18 章 |
| 思维链 | Chain-of-Thought (CoT) | 让模型在给出答案前先写出推理步骤 | 第 18 章 |
| 长思维链 | Long Chain-of-Thought | 长度达数千 Token、包含回溯与验证的推理 | 第 18 章 |
| 过程奖励 | Process Reward | 对推理中间步骤给出的奖励 | 第 18 章 |
| 结果奖励 | Outcome Reward | 只对最终答案给出的奖励 | 第 18 章 |
| 验证器 | Verifier | 判定答案或步骤正确性的程序或模型 | 第 18 章 |
| 知识蒸馏 | Knowledge Distillation | 用教师模型的输出指导学生学习 | 第 19 章 |
| Logit 蒸馏 | Logit Distillation | 让学生匹配教师的完整输出分布 | 第 19 章 |
| On-policy 蒸馏 | On-policy Distillation | 在学生自己生成的序列上做蒸馏 | 第 19 章 |
| 剪枝 | Pruning | 移除不重要权重、通道或层 | 第 19 章 |

## C.8 推理与效率

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 解码 | Decoding | 从模型输出分布中选出一个 Token 的过程 | 第 20 章 |
| 贪心解码 | Greedy Decoding | 每步选概率最大的 Token | 第 20 章 |
| 束搜索 | Beam Search | 同时维护多条候选序列并保留最优者 | 第 20 章 |
| 温度 | Temperature | 通过缩放 logits 调节分布尖锐程度的参数 | 第 20 章 |
| Top-k 采样 | Top-k Sampling | 只在概率最高的 k 个 Token 中采样 | 第 20 章 |
| Top-p 采样 | Nucleus Sampling | 在累积概率达到 p 的最小集合中采样 | 第 20 章 |
| Min-p 采样 | Min-p Sampling | 只保留概率不低于最大值 p 倍的后备集 | 第 20 章 |
| 重复惩罚 | Repetition Penalty | 降低已出现 Token 的概率以减少重复 | 第 20 章 |
| 约束解码 | Constrained Decoding | 通过掩码保证输出符合语法或格式 | 第 20 章 |
| Logit 处理器 | Logit Processor | 在采样前修改 logits 的插件 | 第 20 章 |
| FlashAttention | FlashAttention | 用分块与在线 Softmax 把注意力 IO 降到线性量级 | 第 21 章 |
| 在线 Softmax | Online Softmax | 分块累加时动态维护最大值与分母的算法 | 第 21 章 |
| PagedAttention | PagedAttention | 用分页管理 KV Cache 以消除碎片 | 第 21 章 |
| 连续批处理 | Continuous Batching | 每步动态加入与移除请求的批处理方式 | 第 21 章 |
| 分块预填充 | Chunked Prefill | 把长提示的预填充拆成多块与解码混合执行 | 第 21 章 |
| 预填充/解码分离 | Prefill-Decode Disaggregation (PD) | 把预填充与解码放到不同资源上执行 | 第 21 章 |
| 前缀缓存 | Prefix Caching | 复用相同前缀的 KV Cache | 第 21 章 |
| 首 Token 延迟 | Time to First Token (TTFT) | 从请求到输出第一个 Token 的时间 | 第 21 章 |
| 每 Token 延迟 | Time per Output Token (TPOT) | 解码阶段平均每个 Token 的耗时 | 第 21 章 |
| 投机解码 | Speculative Decoding | 用小模型起草、大模型并行验证 | 第 22 章 |
| 草稿模型 | Draft Model | 投机解码中负责快速起草的小模型 | 第 22 章 |
| 接受率 | Acceptance Rate | 草稿 Token 被目标模型接受的比例 | 第 22 章 |
| Medusa | Medusa | 用多个预测头并行生成候选 Token 的投机方案 | 第 22 章 |
| EAGLE | EAGLE | 在特征层做自回归起草的投机方案 | 第 22 章 |
| 量化 | Quantization | 用更低位宽表示权重或激活 | 第 23 章 |
| 训练后量化 | Post-Training Quantization (PTQ) | 训练完成后直接量化，不重新训练 | 第 23 章 |
| 量化感知训练 | Quantization-Aware Training (QAT) | 训练时模拟量化误差 | 第 23 章 |
| 对称/非对称量化 | Symmetric/Asymmetric Quantization | 量化区间是否关于零对称 | 第 23 章 |
| 逐通道/逐组量化 | Per-channel/Per-group Quantization | 每个通道或每组共享一个缩放因子 | 第 23 章 |
| GPTQ | GPTQ | 基于二阶信息的逐层权重量化方法 | 第 23 章 |
| AWQ | Activation-aware Weight Quantization | 根据激活重要性保护显著权重通道 | 第 23 章 |
| SmoothQuant | SmoothQuant | 把激活的量化难度迁移到权重上 | 第 23 章 |
| 离群值 | Outlier | 数值显著大于同组其他元素的激活或权重 | 第 23 章 |
| KV Cache 量化 | KV Cache Quantization | 对缓存的键值做低精度存储 | 第 23 章 |
| 测试时计算扩展 | Test-time Scaling | 在推理阶段投入更多计算提升质量 | 第 24 章 |
| Best-of-N | Best-of-N | 采样 N 条输出并选最优 | 第 24 章 |
| 自洽性 | Self-Consistency | 多次采样并对最终答案投票 | 第 24 章 |
| 树搜索 | Tree Search | 把推理展开为树并搜索高价值路径 | 第 24 章 |
| 蒙特卡洛树搜索 | Monte Carlo Tree Search (MCTS) | 用模拟估计节点价值并平衡探索与利用 | 第 24 章 |
| 预算控制 | Budget Forcing | 强制模型在给定 Token 预算内思考或收束 | 第 24 章 |
| 过度思考 | Overthinking | 简单问题也生成冗长推理，浪费计算 | 第 24 章 |

## C.9 多模态、智能体与前沿

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 多模态大模型 | Multimodal LLM | 同时处理文本与图像等模态的模型 | 第 25 章 |
| 视觉编码器 | Vision Encoder | 把图像编码为特征序列的网络 | 第 25 章 |
| 投影层 | Projector | 把视觉特征映射到语言模型空间的模块 | 第 25 章 |
| 视觉语言模型 | Vision-Language Model (VLM) | 能理解图像并生成文本的模型 | 第 25 章 |
| 原生多模态 | Native Multimodality | 从预训练起就在统一表示中混合多模态 | 第 25 章 |
| 图像 Token 化 | Image Tokenization | 把图像离散化为 Token 序列 | 第 25 章 |
| 智能体 | Agent | 能感知环境、调用工具并多步完成任务的系统 | 第 26 章 |
| 工具调用 | Tool Use / Function Calling | 模型输出结构化调用请求并使用返回结果 | 第 26 章 |
| ReAct | ReAct | 交替进行推理与行动的模式 | 第 26 章 |
| 轨迹 | Trajectory | 智能体一次任务中的完整思考与动作序列 | 第 26 章 |
| 环境 | Environment | 智能体可交互并给出反馈的外部系统 | 第 26 章 |
| 智能体强化学习 | Agentic RL | 在交互环境中用 RL 训练多步智能体 | 第 26 章 |
| 基准 | Benchmark | 用于横向比较模型能力的标准化测试集 | 第 27 章 |
| LLM 作为裁判 | LLM-as-Judge | 用强模型给其他模型的输出打分 | 第 27 章 |
| 竞技场 | Arena | 用人类两两投票排出模型名次 | 第 27 章 |
| 提示词敏感性 | Prompt Sensitivity | 分数随提示措辞大幅波动的现象 | 第 27 章 |
| 探针 | Probing | 用简单分类器读取模型内部表示 | 第 28 章 |
| 机制可解释性 | Mechanistic Interpretability | 逆向工程模型内部的计算回路 | 第 28 章 |
| 回路 | Circuit | 完成特定功能的神经元与注意力头子图 | 第 28 章 |
| 稀疏自编码器 | Sparse Autoencoder (SAE) | 把激活分解为稀疏特征的字典学习模型 | 第 28 章 |
| 特征 | Feature | 模型内部可解释的方向或概念 | 第 28 章 |
| 激活修补 | Activation Patching | 替换局部激活观察输出变化以定位因果作用 | 第 28 章 |
| 幻觉 | Hallucination | 模型生成看似合理但事实错误的内容 | 第 29 章 |
| 校准 | Calibration | 模型置信度与实际正确率的一致性 | 第 29 章 |
| 持续学习 | Continual Learning | 在不遗忘旧知识的前提下持续更新模型 | 第 29 章 |
