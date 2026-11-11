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
| 门控线性单元 | Gated Linear Unit (GLU) | 两路线性投影逐元素相乘的前馈结构 | 第 5 章 |
| SwiGLU | SwiGLU | 用 SiLU 门控的 GLU 前馈网络变体 | 第 5 章 |
| GeGLU | GeGLU | 用 GELU 门控的 GLU 前馈网络变体 | 第 5 章 |
| SiLU | SiLU (Swish) | 由 Sigmoid 自门控的激活 $$z\sigma(z)$$ | 第 5 章 |
| 权重绑定 | Weight Tying | 输入嵌入与输出投影共享同一矩阵 | 第 5 章 |
| 并行层 | Parallel Layers | 注意力与前馈共享归一化并并行相加的块结构 | 第 5 章 |
| 去偏置 | Bias-free | 去除线性层与归一化偏置的模型配置 | 第 5 章 |
| 等参数预算点 | Iso-parameter Budget | 替换前后参数量与 FLOPs 对齐的配置点 | 第 5 章 |

## C.3 位置编码

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 位置编码 | Positional Encoding | 向模型注入 Token 顺序信息的机制 | 第 3 章 |
| 绝对位置编码 | Absolute Positional Encoding | 为每个绝对位置分配一个编码向量 | 第 3 章 |
| 正弦位置编码 | Sinusoidal Positional Encoding | 用不同频率的正弦与余弦函数生成的无参数位置编码 | 第 4 章 |
| 相对位置编码 | Relative Positional Encoding | 只编码位置之间的相对距离 | 第 4 章 |
| 旋转位置编码 | Rotary Position Embedding (RoPE) | 通过旋转查询与键来注入相对位置信息 | 第 4 章 |
| 相对位移性质 | Relative Shift Property | 同一对查询与键平移相同距离后内积不变 | 第 4 章 |
| ALiBi | Attention with Linear Biases | 用与距离成正比的线性偏置惩罚远距离注意力 | 第 4 章 |
| 无位置编码 | NoPE | 完全不显式编码位置，依赖因果结构隐式推断 | 第 4 章 |
| 长度外推 | Length Extrapolation | 在比训练长度更长的序列上保持性能 | 第 4 章 |
| 位置插值 | Position Interpolation (PI) | 把超长位置线性压缩回训练范围以扩展上下文 | 第 4 章 |
| NTK-aware 缩放 | NTK-aware Scaling | 通过调整 RoPE 基频、按频率拉伸波长的外推方法 | 第 4 章 |
| 分频插值 | NTK-by-Parts | 按波长比例分段决定哪些频率插值的 YaRN 组件 | 第 4 章 |
| YaRN | YaRN | 结合分频插值与温度缩放的长度外推方法 | 第 4 章 |
| 有效上下文 | Effective Context Length | 任务指标仍不低于阈值的最长序列长度 | 第 4 章 |

## C.4 注意力变体与 KV Cache

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 多查询注意力 | Multi-Query Attention (MQA) | 所有查询头共享一组键值头 | 第 6 章 |
| 分组查询注意力 | Grouped-Query Attention (GQA) | 查询头分组、每组共享一组键值头 | 第 6 章 |
| 多头潜在注意力 | Multi-head Latent Attention (MLA) | 把键值压缩到低维潜在向量再投影的注意力 | 第 6 章 |
| KV 缓存 | KV Cache | 缓存历史键值以避免重复计算 | 第 6 章 |
| KV 头 | KV Head | 被若干查询头共享的一组键值投影 | 第 6 章 |
| 预填充 | Prefill | 处理完整提示、一次性建立 KV 缓存的阶段 | 第 6 章 |
| 解码 | Decode | 逐 Token 生成、每步读取 KV 缓存的阶段 | 第 6 章 |
| 访存受限 | Memory-bound | 性能由搬运字节数而非浮点运算数决定的负载 | 第 6 章 |
| 算术强度 | Arithmetic Intensity | 每搬运一字节所执行的浮点运算数 | 第 6 章 |
| 解耦 RoPE | Decoupled RoPE | 把位置键拆到独立小通路的 MLA 位置编码方案 | 第 6 章 |
| 权重吸收 | Weight Absorption | 把上投影合并进查询与输出投影的推理变换 | 第 6 章 |
| 分页注意力 | Paged Attention | 按固定大小块管理 KV 缓存、减少碎片的机制 | 第 6 章 |
| 前缀缓存 | Prefix Caching | 多个请求共享相同前缀 KV 的计算复用 | 第 6 章 |
| 继续训练转换 | Uptraining | 从 MHA 检查点均值池化 KV 头后少量继续训练 | 第 6 章 |
| 滑动窗口注意力 | Sliding Window Attention | 每个位置只关注最近 $$W$$ 个位置 | 第 6 章 |
| 注意力汇 | Attention Sink | 模型倾向把大量注意力分配给首个 Token 的现象 | 第 6 章 |

## C.5 混合专家与替代架构

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 混合专家 | Mixture of Experts (MoE) | 用路由器为每个 Token 选择少数专家计算 | 第 7 章 |
| 专家 | Expert | 参数独立、结构相同的前馈子网络 | 第 7 章 |
| 路由器 | Router | 为每个 Token 计算专家权重与门控的轻量网络 | 第 7 章 |
| 稀疏激活 | Sparse Activation | 每 Token 只计算全部参数中的一小部分 | 第 7 章 |
| Top-K 路由 | Top-K Routing | 只激活权重最高的 $$k$$ 个专家 | 第 7 章 |
| 负载均衡辅助损失 | Load Balancing Auxiliary Loss | 惩罚专家负载不均、调节路由概率的辅助损失 | 第 7 章 |
| 路由器 z-loss | Router z-loss | 约束路由 logits 幅度、提升稳定性的正则项 | 第 7 章 |
| 共享专家 | Shared Expert | 所有 Token 都会经过、不被稀疏化的公共专家 | 第 7 章 |
| 细粒度专家 | Fine-grained Expert | 把专家切窄并提高激活个数以保持计算预算 | 第 7 章 |
| Expert Choice | Expert Choice Routing | 由专家挑选固定数量 Token 的反向路由 | 第 7 章 |
| 无辅助损失均衡 | Auxiliary-loss-free Balancing | 用专家偏置反馈而非损失项控制负载 | 第 7 章 |
| 专家并行 | Expert Parallelism (EP) | 把不同专家分布到不同设备上 | 第 7 章 |
| 全对全通信 | All-to-All | 专家并行中 Token 与输出在设备间重排的集合通信 | 第 7 章 |
| 分组受限路由 | Group-limited Routing | 限制每个 Token 的专家落在少数设备组内 | 第 7 章 |
| 容量因子 | Capacity Factor (CF) | 每个专家可接收 Token 数的上限系数 | 第 7 章 |
| Token 丢弃 | Token Dropping | 专家超出容量时丢弃多余 Token | 第 7 章 |
| 无丢弃 | Dropless | 不设固定容量、按实际分派计算的实现 | 第 7 章 |
| 总参数 | Total Parameters | 全部专家与其余模块的参数之和 | 第 7 章 |
| 激活参数 | Activated Parameters | 每个 Token 实际参与计算的参数量 | 第 7 章 |
| 激活比 | Activation Ratio | 激活参数与总参数之比 | 第 7 章 |
| 线性注意力 | Linear Attention | 把注意力重写为线性复杂度的核方法 | 第 8 章 |
| 状态空间模型 | State Space Model (SSM) | 用线性递推状态描述序列的模型族 | 第 8 章 |
| S4 | Structured State Space Sequence Model | 用结构化矩阵参数化状态转移的 SSM | 第 8 章 |
| Mamba | Mamba | 引入输入相关选择机制的 SSM 架构 | 第 8 章 |
| 选择性扫描 | Selective Scan | Mamba 中按输入动态调整状态的并行扫描算法 | 第 8 章 |
| RWKV | Receptance Weighted Key Value | 结合 RNN 推理与 Transformer 训练并行的架构 | 第 8 章 |
| 混合架构 | Hybrid Architecture | 交错使用注意力层与 SSM/线性层的模型 | 第 8 章 |
| 核特征映射 | Kernel Feature Map | 把查询与键映射到可做内积的有限维空间的函数 $$\phi$$ | 第 8 章 |
| 前缀和 | Prefix Sum | 沿序列维累积外积、一次得到所有位置状态的并行操作 | 第 8 章 |
| 分块线性注意力 | Chunked Linear Attention | 块内二次、块间递推以控制中间显存的长序列实现 | 第 8 章 |
| 零阶保持 | Zero-Order Hold (ZOH) | 假设步长内输入不变、把连续系统离散化的方法 | 第 8 章 |
| 线性时不变 | Linear Time-Invariant (LTI) | 参数不随位置变化、可用固定卷积核描述的系统 | 第 8 章 |
| 并行扫描 | Parallel Scan | 利用结合律在 $$O(\log T)$$ 深度内计算前缀递推 | 第 8 章 |
| 门控线性递推单元 | Gated Linear Recurrent Unit | 以逐通道门控衰减更新状态的线性递推结构 | 第 8 章 |
| 闪电注意力 | Lightning Attention | 分块实现的分块线性注意力，用于超长上下文 | 第 8 章 |
| 门控 DeltaNet | Gated DeltaNet | 带门控衰减与 delta rule 更新的线性注意力变体 | 第 8 章 |
| 状态账本 | State Ledger | 按层统计缓存/状态字节数随序列长度变化的账目 | 第 8 章 |

## C.6 预训练

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 缩放定律 | Scaling Law | 损失随参数量、数据量、算力变化的经验规律 | 第 9 章 |
| 幂律 | Power Law | 两个量按固定指数成比例变化的函数关系 | 第 9 章 |
| 不可约损失 | Irreducible Loss | 模型与数据无限大时仍无法消除的损失下限 | 第 9 章 |
| Kaplan 定律 | Kaplan Scaling Law | 早期提出的幂律缩放关系 | 第 9 章 |
| Chinchilla 定律 | Chinchilla Scaling Law | 指出最优数据量应随参数量同比增长的缩放关系 | 第 9 章 |
| 计算最优 | Compute-Optimal | 在固定训练算力下使损失最小的参数/数据配比 | 第 9 章 |
| 推理最优 | Inference-Optimal | 把推理成本计入总成本后选择的更小模型、更多数据配比 | 第 9 章 |
| 过训练 | Overtraining | 训练 Token 数显著超过计算最优配比的做法 | 第 9 章 |
| 数据受限 | Data-Constrained | 唯一数据不足、必须重复训练的情形 | 第 9 章 |
| 唯一数据 | Unique Data | 去重后可用的不重复 Token 总量 | 第 9 章 |
| 有效数据量 | Effective Data | 重复数据折算成全新数据后的等效 Token 数 | 第 9 章 |
| 数据重复 | Data Repetition | 同一语料被训练多轮，用重复轮数度量 | 第 9 章 |
| 关键批大小 | Critical Batch Size | 继续增大批大小、收益开始明显递减的临界点 | 第 9 章 |
| 线性缩放规则 | Linear Scaling Rule | 批大小远小于关键值时，学习率与批大小成正比 | 第 9 章 |
| 平方根缩放规则 | Square-Root Scaling Rule | 批大小远大于关键值时，学习率与批大小的平方根成正比 | 第 9 章 |
| 最大更新参数化 | Maximal Update Parametrization (μP) | 使超参数可跨宽度、跨深度迁移的参数化方案 | 第 9 章 |
| 参数迁移 | Hyperparameter Transfer | 在小模型上调好超参数、直接用于大模型的做法 | 第 9 章 |
| 涌现能力 | Emergent Abilities | 规模超过某阈值后评测分数突然跳升的现象 | 第 9 章 |
| 相变 | Phase Transition | 系统宏观行为在参数越过临界点时发生质变 | 第 9 章 |
| 逆缩放 | Inverse Scaling | 规模增大反而使某项指标变差的现象 | 第 9 章 |
| IsoFLOP 曲线 | IsoFLOP Curve | 固定训练算力时，损失随参数/数据分配变化的曲线 | 第 9 章 |
| Token/参数比 | Tokens per Parameter | 训练 Token 数与参数量的比值 | 第 9 章 |
| 算力账本 | Compute Ledger | 把训练、推理各项 FLOPs 分项相加的核算表 | 第 9 章 |
| 数据管道 | Data Pipeline | 从原始抓取到训练 Token 的逐级处理流程 | 第 10 章 |
| 数据漏斗 | Data Funnel | 用逐级保留率描述管道删减的模型 | 第 10 章 |
| 正文抽取 | Content Extraction | 从网页 HTML 中剥离正文、去掉导航与广告 | 第 10 章 |
| 语种识别 | Language Identification | 判断文档语言并按目标语种过滤 | 第 10 章 |
| 精确去重 | Exact Deduplication | 对规范化文本做哈希，删除逐字相同的副本 | 第 10 章 |
| 近似去重 | Near Deduplication | 用相似度阈值删除改写、转载等近似副本 | 第 10 章 |
| Shingle | Shingle | 文档切成的连续 n 元组，用于集合相似度 | 第 10 章 |
| Jaccard 相似度 | Jaccard Similarity | 两个集合交集大小与并集大小之比 | 第 10 章 |
| MinHash | MinHash | 用随机哈希最小值估计 Jaccard 的签名方法 | 第 10 章 |
| 局部敏感哈希 | Locality-Sensitive Hashing (LSH) | 用分桶把可能相似的文档对召回为候选 | 第 10 章 |
| 质量过滤 | Quality Filtering | 按启发式分、分类器分或模型分删除低质文档 | 第 10 章 |
| 过滤产出率 | Yield | 保留文档数占输入文档数的比例 | 第 10 章 |
| 过滤精确率 | Filter Precision | 保留文档中真正高质量文档的比例 | 第 10 章 |
| 数据配比 | Data Mixture | 各数据来源在训练 Token 中的比例 | 第 10 章 |
| 温度采样 | Temperature Sampling | 用 $$p_i\propto w_i^{1/T}$$ 调节配比平滑程度 | 第 10 章 |
| 最大余数法 | Largest Remainder Method | 按小数部分分配剩余预算，使计数之和等于总量 | 第 10 章 |
| 配比定律 | Data Mixing Law | 领域损失随各领域配比呈指数变化的经验规律 | 第 10 章 |
| 分组分布鲁棒优化 | Group DRO | 按领域分组、优化最差领域损失的权重学习方法 | 第 10 章 |
| 数据课程 | Data Curriculum | 按训练阶段安排数据分布的做法 | 第 10 章 |
| 退火 | Annealing | 训练末段切换到高质量数据并衰减学习率 | 第 10 章 |
| WSD 调度 | Warmup-Stable-Decay | 预热、稳定、衰减三段式学习率调度 | 第 10 章 |
| 合成数据 | Synthetic Data | 由模型生成的教科书、问答、代码或改写文本 | 第 10 章 |
| 模型坍缩 | Model Collapse | 反复自训练导致分布尾部丢失、多样性下降 | 第 10 章 |
| 评测污染 | Benchmark Contamination | 评测样本或近似副本出现在训练集中 | 第 10 章 |
| 去污染 | Decontamination | 从训练集删除与评测集重叠超阈值的文档 | 第 10 章 |
| n-gram 覆盖率 | n-gram Coverage | 评测文档的 n-gram 中出现在训练集的比例 | 第 10 章 |
| 金丝雀串 | Canary String | 嵌入数据中的特殊标记，用于检测记忆与泄漏 | 第 10 章 |
| AdamW | Adam with Decoupled Weight Decay | 把权重衰减与梯度更新解耦的 Adam | 第 11 章 |
| 学习率调度 | Learning Rate Schedule | 学习率随训练步数变化的规则 | 第 11 章 |
| 预热 | Warmup | 训练初期让学习率从很小线性上升 | 第 11 章 |
| 余弦退火 | Cosine Decay | 学习率按余弦曲线衰减到接近零 | 第 11 章 |
| Muon | Muon | 对动量矩阵做正交化的优化器 | 第 11 章 |
| 损失尖峰 | Loss Spike | 训练中损失突然大幅上升的现象 | 第 11 章 |
| 梯度裁剪 | Gradient Clipping | 限制梯度范数以防止更新过大 | 第 11 章 |
| 优化器状态 | Optimizer State | Adam 为每个参数保存的一阶矩与二阶矩 | 第 11 章 |
| 一阶矩 | First Moment | 梯度的指数滑动平均，估计更新方向 | 第 11 章 |
| 二阶矩 | Second Moment | 梯度平方的指数滑动平均，估计逐坐标尺度 | 第 11 章 |
| 偏差修正 | Bias Correction | 用 $$1-\beta^t$$ 校正零初始化矩的偏小 | 第 11 章 |
| 耦合 L2 | Coupled L2 | 把 $$\lambda\theta$$ 加进梯度、再被二阶矩归一化的衰减 | 第 11 章 |
| 解耦权重衰减 | Decoupled Weight Decay | 直接按 $$(1-\eta\lambda)$$ 收缩参数的衰减 | 第 11 章 |
| 参数分组 | Parameter Groups | 按是否衰减把参数拆成不同超参数组 | 第 11 章 |
| 学习率乘子 | Learning Rate Multiplier | 相对峰值学习率的比例因子 | 第 11 章 |
| z 分数 | z-score | 损失相对滑动均值的标准差倍数 | 第 11 章 |
| 更新比例 | Update-to-Weight Ratio | 单步更新范数与参数范数之比 | 第 11 章 |
| 注意力熵 | Attention Entropy | 注意力权重分布的熵，用于诊断 softmax 饱和 | 第 11 章 |
| QK-norm | Query-Key Normalization | 对查询、键做归一化以限制打分幅度 | 第 11 章 |
| QK-Clip | Query-Key Clipping | 监控并缩放 query/key 的最大值 | 第 11 章 |
| z-loss | z-loss | 惩罚输出层配分函数对数的正则项 | 第 11 章 |
| 损失缩放 | Loss Scaling | 放大损失以避免低精度梯度下溢 | 第 11 章 |
| 主权重 | Master Weights | 在 FP32 中保存的参数副本 | 第 11 章 |
| Newton-Schulz 迭代 | Newton-Schulz Iteration | 只用矩阵乘逼近正交化的五次迭代 | 第 11 章 |
| 极分解 | Polar Decomposition | 把矩阵分解为正交因子与半正定因子的乘积 | 第 11 章 |
| 正交化 | Orthogonalization | 把矩阵奇异值推向 1 的变换 | 第 11 章 |
| 谱范数 | Spectral Norm | 矩阵最大奇异值，Muon 的约束范数 | 第 11 章 |
| 梯度累积 | Gradient Accumulation | 多个微批累加梯度后再更新一次；第 12 章并入全局批大小 | 第 11 章 |
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
| 分布式训练 | Distributed Training | 把训练任务切分到多设备协同完成 | 第 12 章 |
| 上下文并行 | Context Parallelism (CP) | 把长序列或 KV 块分到多卡并行计算 | 第 12 章 |
| 模型状态 | Model States | 权重、梯度、主权重与优化器状态的总和 | 第 12 章 |
| 微批 | Micro-batch | 一次前向反向处理的最小数据片 | 第 12 章 |
| 全局批 | Global Batch | 一步更新所覆盖的全部样本数 | 第 12 章 |
| 在途微批 | In-flight Micro-batches | 流水线中同时处于不同阶段的微批 | 第 12 章 |
| 全归约 | All-Reduce | 所有设备求和并把结果广播给每台设备 | 第 12 章 |
| 归约散射 | Reduce-Scatter | 先求和再按设备切分，各设备只留一段 | 第 12 章 |
| 全收集 | All-Gather | 各设备收集所有分片并拼接成完整张量 | 第 12 章 |
| 环形全归约 | Ring All-Reduce | 沿环传递分片完成归约与收集 | 第 12 章 |
| 列并行 | Column Parallel | 按输出维切分权重，前向无需通信 | 第 12 章 |
| 行并行 | Row Parallel | 按输入维切分权重，前向需一次全归约 | 第 12 章 |
| 词表并行 | Vocabulary Parallel | 按词表维切分输出投影与交叉熵 | 第 12 章 |
| 流水线气泡 | Pipeline Bubble | 填充与排空阶段设备空闲的比例 | 第 12 章 |
| 1F1B | One-Forward-One-Backward | 稳态交替执行前向与反向的流水调度 | 第 12 章 |
| 交错式调度 | Interleaved Schedule | 把每段切成虚拟段以进一步缩小气泡 | 第 12 章 |
| 动态损失缩放 | Dynamic Loss Scaling | 按溢出情况自动调整缩放系数 | 第 12 章 |
| 逐块缩放 | Per-Block Scaling | 每个量化块单独计算缩放系数 | 第 12 章 |
| 延迟缩放 | Delayed Scaling | 用历史最大值估计当前步的缩放系数 | 第 12 章 |
| 高精度累加 | High-Precision Accumulation | 低精度矩阵乘后用 FP32 累加结果 | 第 12 章 |
| DualPipe | DualPipe | 双向流水并重叠 all-to-all 的调度 | 第 12 章 |
| 长上下文 | Long Context | 上下文窗口显著超过训练常见长度的设定 | 第 13 章 |
| 渐进式长度扩展 | Progressive Length Extension | 从短长度分阶段加长到目标长度的训练策略 | 第 13 章 |
| 长度课程 | Length Curriculum | 按预设阶段分配不同序列长度的训练安排 | 第 13 章 |
| 混合长度训练 | Mixed-Length Training | 每一步从多个训练长度中轮流采样 | 第 13 章 |
| 短上下文遗忘 | Short-Context Forgetting | 长长度训练后短序列性能下降的现象 | 第 13 章 |
| 文档隔离 | Document Masking | 打包序列中阻止跨文档注意力的掩码 | 第 13 章 |
| 块对角因果掩码 | Block-Diagonal Causal Mask | 同文档且满足因果性的注意力掩码 | 第 13 章 |
| 环形注意力 | Ring Attention | KV 块在设备间轮转以分摊长序列注意力 | 第 13 章 |
| 大海捞针 | Needle in a Haystack (NIAH) | 在长文本指定深度检索关键信息的评测 | 第 13 章 |
| 单针检索 | Single-Needle Retrieval | 只放一条关键信息的 NIAH 变体 | 第 13 章 |
| 多针检索 | Multi-Needle Retrieval | 同时放多条关键信息的 NIAH 变体 | 第 13 章 |
| 迷失在中间 | Lost in the Middle | 关键信息位于长文中部时性能下降的现象 | 第 13 章 |
| RULER | RULER | 覆盖 13 项任务的长上下文评测套件 | 第 13 章 |
| 检索增强生成 | Retrieval-Augmented Generation (RAG) | 先检索外部文档再让模型基于文档生成 | 第 13 章 |

## C.7 后训练

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 监督微调 | Supervised Fine-Tuning (SFT) | 用人工标注的输入输出对继续训练模型 | 第 14 章 |
| 指令微调 | Instruction Tuning | 用指令-回答数据让模型学会遵循指令 | 第 14 章 |
| 对话模板 | Chat Template | 把多轮对话序列化为固定格式的规则 | 第 14 章 |
| 损失掩码 | Loss Masking | 只在目标回答部分计算损失 | 第 14 章 |
| 拒绝采样 | Rejection Sampling | 采样多条输出并只保留通过验证的样本 | 第 14 章 |
| 灾难性遗忘 | Catastrophic Forgetting | 学习新任务时丢失旧能力 | 第 14 章 |
| 监督比例 | Supervision Ratio | 监督 Token 占总 Token 的比例 | 第 14 章 |
| Token 平均 | Token Mean | 对整个批次的监督 Token 统一平均 | 第 14 章 |
| 序列平均 | Sequence Mean | 先对每条样本平均，再对样本平均 | 第 14 章 |
| 系统性标签噪声 | Systematic Label Noise | 错误与任务规则相关、不会被平均掉的标签噪声 | 第 14 章 |
| 多样性塌缩 | Diversity Collapse | 采样数据去重后种类极少、输出分布收窄 | 第 14 章 |
| 回放 | Replay | 在微调中混入预训练数据以保持旧能力 | 第 14 章 |
| 表面对齐 | Surface Alignment | SFT 主要学习回答格式与风格，而非注入新知识 | 第 14 章 |
| 参数高效微调 | Parameter-Efficient Fine-Tuning (PEFT) | 只训练少量新增参数完成适配 | 第 15 章 |
| LoRA | Low-Rank Adaptation | 用两个低秩矩阵的乘积近似权重更新 | 第 15 章 |
| 低秩分解 | Low-Rank Decomposition | 用秩远小于原矩阵的两个矩阵表示更新 | 第 15 章 |
| QLoRA | Quantized LoRA | 在 4 比特量化的基座上做 LoRA | 第 15 章 |
| DoRA | Weight-Decomposed Low-Rank Adaptation | 把权重分解为幅度与方向分别更新 | 第 15 章 |
| LoRA 秩 | LoRA Rank | 低秩矩阵的中间维度 $$r$$，控制适配容量 | 第 15 章 |
| 缩放系数 | Scaling Factor | LoRA 分支的 $$\alpha/r$$，决定有效步长 | 第 15 章 |
| 目标模块 | Target Modules | 被替换为 LoRA 层的线性层集合 | 第 15 章 |
| 合并 | Merging | 把低秩增量写回基座权重，推理时消除额外分支 | 第 15 章 |
| NF4 | 4-bit NormalFloat | 按正态分布分位数设计的 16 级量化码本 | 第 15 章 |
| 分块量化 | Block-wise Quantization | 每个小块独立按 absmax 缩放后量化 | 第 15 章 |
| 双重量化 | Double Quantization | 把第一层量化尺度再量化到 8 比特 | 第 15 章 |
| 分页优化器 | Paged Optimizer | 用统一内存把优化器状态换出以吸收显存尖峰 | 第 15 章 |
| 适配器 | Adapter | 插入 Transformer 块的瓶颈前馈模块 | 第 15 章 |
| 前缀微调 | Prefix-Tuning | 在每层键值前拼接可学习前缀 | 第 15 章 |
| 提示微调 | Prompt Tuning | 只在输入嵌入前拼接可学习软提示 | 第 15 章 |
| 多适配器服务 | Multi-LoRA Serving | 一份基座批量服务多个 LoRA 适配器 | 第 15 章 |
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
| 偏好数据 | Preference Data | 由提示与优胜/劣质回答三元组构成的数据集 | 第 16 章 |
| 成对偏好 | Pairwise Preference | 同一提示下两条回答的相对好坏 | 第 16 章 |
| 奖励校准 | Reward Calibration | 奖励模型预测的胜率与经验胜率一致 | 第 16 章 |
| 重要性比 | Importance Ratio | 新旧策略在同一动作上的概率之比 | 第 16 章 |
| 裁剪目标 | Clipped Objective | 把重要性比限制在 $$[1-\epsilon,1+\epsilon]$$ 并取较小值的目标 | 第 16 章 |
| 旧策略 | Behavior Policy | 产生当前 rollout 的策略快照 | 第 16 章 |
| KL 塑形奖励 | KL Shaping Reward | 逐 Token 的 $$-\beta(\log\pi_\theta-\log\pi_{\rm ref})$$ 奖励 | 第 16 章 |
| 价值网络 | Value Network | 估计状态价值的标量网络，与策略联合训练 | 第 16 章 |
| 长度偏置 | Length Bias | 奖励模型或策略系统性偏好更长回答 | 第 16 章 |
| Best-of-N | Best-of-N Sampling | 采样 N 条回答并选奖励最高的一条 | 第 16 章 |
| 拒绝采样微调 | Rejection Sampling Fine-Tuning | 用奖励筛选采样结果，再对通过者做 SFT | 第 16 章 |
| 在线偏好优化 | Online Preference Optimization | 训练中持续采样新回答并打分 | 第 16 章 |
| 离线偏好优化 | Offline Preference Optimization | 只在固定偏好数据上优化，不额外采样 | 第 16 章 |
| 奖励集成 | Reward Ensemble | 训练多个奖励模型并聚合分数以降低方差 | 第 16 章 |
| k1 估计量 | k1 Estimator | 用对数比直接估计 KL，无偏但方差大 | 第 16 章 |
| k3 估计量 | k3 Estimator | 用 $$r-\log r-1$$ 估计 KL，逐 Token 非负 | 第 16 章 |
| 直接偏好优化 | Direct Preference Optimization (DPO) | 直接在偏好数据上优化、无需显式奖励模型 | 第 17 章 |
| IPO | Identity Preference Optimization | 用平方损失替代 DPO 的 log-sigmoid 损失 | 第 17 章 |
| KTO | Kahneman-Tversky Optimization | 只需单条回答好坏的偏好优化方法 | 第 17 章 |
| SimPO | Simple Preference Optimization | 用长度归一化对数似然做偏好优化 | 第 17 章 |
| ORPO | Odds Ratio Preference Optimization | 把偏好项并入 SFT 损失的单阶段方法 | 第 17 章 |
| 隐式奖励 | Implicit Reward | 由策略与参考策略的对数比定义的奖励 | 第 17 章 |
| 对数比 | Log Ratio | $$\log\pi_\theta(y\mid x)-\log\pi_{\rm ref}(y\mid x)$$ | 第 17 章 |
| DPO 损失 | DPO Loss | 隐式奖励差上的负 log-sigmoid，等价于重参数化的 Bradley–Terry 似然 | 第 17 章 |
| 目标间隔 | Target Margin | IPO 等变体希望达到的隐式奖励差，通常为 $$1/(2\beta)$$ | 第 17 章 |
| 长度归一化 | Length Normalization | 用响应 Token 数平均序列对数概率，削弱长回答权重 | 第 17 章 |
| 似然位移 | Likelihood Displacement | 同时压低赢家与输家的绝对对数概率、只拉大相对间隔的现象 | 第 17 章 |
| 参考模型失配 | Reference Mismatch | 参考模型与产生偏好数据的策略不一致，使隐式奖励零点偏移 | 第 17 章 |
| 偏好对排序准确率 | Pairwise Preference Accuracy | 隐式奖励把赢家排在输家之前的比例，平局计为失败 | 第 17 章 |
| 迭代 DPO | Iterative DPO | 定期用当前策略重新生成回答并重新标注的离线—在线折中 | 第 17 章 |
| 保守 DPO | Conservative DPO (cDPO) | 对偏好标签做平滑，缓解标注噪声与过拟合 | 第 17 章 |
| 可验证奖励强化学习 | RLVR | 用可自动验证的奖励（如答案正确性）做 RL | 第 18 章 |
| 组相对策略优化 | Group Relative Policy Optimization (GRPO) | 用同一问题多次采样的组内均值作基线 | 第 18 章 |
| 可验证奖励 | Verifiable Reward | 可由程序判定对错的奖励信号 | 第 18 章 |
| 思维链 | Chain-of-Thought (CoT) | 让模型在给出答案前先写出推理步骤 | 第 18 章 |
| 长思维链 | Long Chain-of-Thought | 长度达数千 Token、包含回溯与验证的推理 | 第 18 章 |
| 过程奖励 | Process Reward | 对推理中间步骤给出的奖励 | 第 18 章 |
| 结果奖励 | Outcome Reward | 只对最终答案给出的奖励 | 第 18 章 |
| 验证器 | Verifier | 判定答案或步骤正确性的程序或模型 | 第 18 章 |
| 组大小 | Group Size | 每个提示采样的回答数 $$G$$，决定基线与混合组概率 | 第 18 章 |
| 组内均值基线 | Group-Mean Baseline | 用同一问题 $$G$$ 条回答的平均奖励构造的优势基线 | 第 18 章 |
| 组内优势 | Group Advantage | 回答奖励相对同题组均值的偏离，可再除以组内标准差 | 第 18 章 |
| 留一法基线 | Leave-One-Out Baseline (LOO) | 用除当前样本外其余样本的均值作基线，避免自身收缩 | 第 18 章 |
| 零方差组 | Zero-Variance Group | 组内奖励全相同（全对或全错），优势为零、无梯度 | 第 18 章 |
| 混合组 | Mixed Group | 组内既有正确又有错误回答，是 RLVR 唯一提供梯度的组型 | 第 18 章 |
| 动态采样 | Dynamic Sampling | 过滤零方差组并补采，保证每个 batch 都有有效梯度 | 第 18 章 |
| 裁剪上界放宽 | Clip-Higher | 令上侧裁剪半径大于下侧，缓解正优势 Token 的熵塌缩 | 第 18 章 |
| 超长奖励塑形 | Overlong Reward Shaping | 对超过长度上限的回答做软惩罚，抑制无意义的长推理 | 第 18 章 |
| 序列级重要性比 | Sequence-Level Importance Ratio | 用整条回答的似然比替代逐 Token 比率的稳定性方案 | 第 18 章 |
| 知识蒸馏 | Knowledge Distillation | 用教师模型的输出指导学生学习 | 第 19 章 |
| 教师模型 | Teacher Model | 提供监督信号的强模型，通常冻结 | 第 19 章 |
| 温度软标签 | Temperature-Softened Label | logits 除以温度 $$\tau$$ 后得到的概率分布 | 第 19 章 |
| 暗知识 | Dark Knowledge | 软标签中非目标类别的相对概率所携带的结构信息 | 第 19 章 |
| Logit 蒸馏 | Logit Distillation | 让学生匹配教师的完整输出分布 | 第 19 章 |
| 序列级蒸馏 | Sequence-Level Distillation | 用教师生成的序列作硬标签训练学生 | 第 19 章 |
| 曝光偏差 | Exposure Bias | 训练用教师前缀、推理用学生前缀导致的分步误差累积 | 第 19 章 |
| On-policy 蒸馏 | On-policy Distillation | 在学生自己生成的序列上做蒸馏 | 第 19 章 |
| 广义知识蒸馏 | Generalized Knowledge Distillation (GKD) | 在教师数据与学生数据间插值、用 JSD 统一 KL 方向的蒸馏框架 | 第 19 章 |
| 拒绝采样 | Rejection Sampling | 只保留验证正确的教师轨迹，兼顾正确性与多样性 | 第 19 章 |
| Top-k 蒸馏 | Top-k Distillation | 只在教师概率最高的 k 个类别上计算蒸馏损失 | 第 19 章 |
| 剪枝 | Pruning | 移除不重要权重、通道或层 | 第 19 章 |
| 幅度剪枝 | Magnitude Pruning | 按权重绝对值排序并删除最小的一批 | 第 19 章 |
| 非结构化剪枝 | Unstructured Pruning | 把任意单个权重置零，稀疏率高但不改变张量形状 | 第 19 章 |
| 结构化剪枝 | Structured Pruning | 删除整个头、神经元或通道，张量形状真实变小 | 第 19 章 |
| 恢复微调 | Recovery Fine-tuning | 剪枝后继续训练以恢复精度，常用蒸馏作监督 | 第 19 章 |
| 压缩比 | Compression Ratio | 教师参数量除以学生或剪枝后参数量 | 第 19 章 |
| long2short | Long-to-Short Distillation | 用长思维链教师教短回答学生，缩短推理长度 | 第 19 章 |

## C.8 推理与效率

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 解码 | Decoding | 从模型输出分布中选出一个 Token 的过程 | 第 20 章 |
| 屋顶线模型 | Roofline Model | 用峰值算力与带宽下界刻画性能上限的模型 | 第 21 章 |
| 分块注意力 | Tiled Attention | 把查询与键值分块读入片上存储、避免物化完整分数矩阵的注意力实现 | 第 21 章 |
| 块表 | Block Table | 逻辑 KV 块到物理块的映射表 | 第 21 章 |
| 内部碎片 | Internal Fragmentation | 已预留块中未被 Token 使用的槽位 | 第 21 章 |
| 外部碎片 | External Fragmentation | 释放后无法拼给新请求的空洞 | 第 21 章 |
| 迭代级调度 | Iteration-level Scheduling | 每一步重新决定批内序列的调度方式 | 第 21 章 |
| 抢占 | Preemption | 显存不足时让在途序列换出或丢弃 KV 的机制 | 第 21 章 |
| 无停顿调度 | Stall-free Scheduling | 用分块预填充保证每轮迭代时间有上界的调度 | 第 21 章 |
| 有效吞吐 | Goodput | 只统计满足 TTFT 与 TPOT 约束的请求吞吐 | 第 21 章 |
| 写时复制 | Copy-on-Write | 多个请求共享 KV 块，写时才复制的机制 | 第 21 章 |
| 贪心解码 | Greedy Decoding | 每步选概率最大的 Token | 第 20 章 |
| 束搜索 | Beam Search | 同时维护多条候选序列并保留最优者 | 第 20 章 |
| 温度 | Temperature | 通过缩放 logits 调节分布尖锐程度的参数 | 第 20 章 |
| Top-k 采样 | Top-k Sampling | 只在概率最高的 k 个 Token 中采样 | 第 20 章 |
| Top-p 采样 | Nucleus Sampling | 在累积概率达到 p 的最小集合中采样 | 第 20 章 |
| Min-p 采样 | Min-p Sampling | 只保留概率不低于最大值 p 倍的后备集 | 第 20 章 |
| 重复惩罚 | Repetition Penalty | 降低已出现 Token 的概率以减少重复 | 第 20 章 |
| 约束解码 | Constrained Decoding | 通过掩码保证输出符合语法或格式 | 第 20 章 |
| Logit 处理器 | Logit Processor | 在采样前修改 logits 的插件 | 第 20 章 |
| 解码分布 | Decoding Distribution | 模型分布经温度、截断、惩罚或掩码后实际用于选 Token 的分布 | 第 20 章 |
| 文本退化 | Neural Text Degeneration | 最大化序列似然时生成平淡、重复甚至循环文本的现象 | 第 20 章 |
| 长度惩罚 | Length Penalty | 束搜索中按长度归一化得分以比较不同长度候选 | 第 20 章 |
| 局部典型采样 | Locally Typical Sampling | 按信息量接近熵的程度排序并截断采样 | 第 20 章 |
| pass@k | Pass at k | 采样 n 条回答、随机取 k 条至少一条正确的概率估计 | 第 20 章 |
| 频率惩罚 | Frequency Penalty | 按 Token 出现次数线性降低其 logit | 第 20 章 |
| 存在惩罚 | Presence Penalty | 只要 Token 出现过就固定降低其 logit | 第 20 章 |
| 结构化输出 | Structured Output | 用语法或 schema 约束模型输出为可解析格式 | 第 20 章 |
| 语法约束解码 | Grammar-Constrained Decoding | 用有限状态机逐步限制允许 Token 的约束解码 | 第 20 章 |
| FlashAttention | FlashAttention | 用分块与在线 Softmax 把注意力 IO 降到线性量级 | 第 21 章 |
| 在线 Softmax | Online Softmax | 分块累加时动态维护最大值与分母的算法 | 第 21 章 |
| PagedAttention | PagedAttention | 用分页管理 KV Cache 以消除碎片 | 第 21 章 |
| 连续批处理 | Continuous Batching | 每步动态加入与移除请求的批处理方式 | 第 21 章 |
| 分块预填充 | Chunked Prefill | 把长提示的预填充拆成多块与解码混合执行 | 第 21 章 |
| 预填充/解码分离 | Prefill-Decode Disaggregation (PD) | 把预填充与解码放到不同资源上执行 | 第 21 章 |
| 首 Token 延迟 | Time to First Token (TTFT) | 从请求到输出第一个 Token 的时间 | 第 21 章 |
| 每 Token 延迟 | Time per Output Token (TPOT) | 解码阶段平均每个 Token 的耗时 | 第 21 章 |
| 投机解码 | Speculative Decoding | 用小模型起草、大模型并行验证 | 第 22 章 |
| 草稿模型 | Draft Model | 投机解码中负责快速起草的小模型 | 第 22 章 |
| 接受率 | Acceptance Rate | 草稿 Token 被目标模型接受的比例 | 第 22 章 |
| Medusa | Medusa | 用多个预测头并行生成候选 Token 的投机方案 | 第 22 章 |
| EAGLE | EAGLE | 在特征层做自回归起草的投机方案 | 第 22 章 |
| 目标模型 | Target Model | 投机解码中负责验证候选并给出最终分布的大模型 | 第 22 章 |
| 无损拒绝采样 | Lossless Rejection Sampling | 用接受概率与残差分布修正草稿、保持目标分布不变的规则 | 第 22 章 |
| 残差分布 | Residual Distribution | 拒绝后用于重采样的归一化正差额分布 | 第 22 章 |
| 全变差距离 | Total Variation Distance | 两个分布差额绝对值之和的一半 | 第 22 章 |
| 草稿长度 | Draft Length | 每轮投机解码提议的 Token 数 | 第 22 章 |
| 树状草稿 | Tree Speculative Decoding | 同时保留多条候选路径并用树状掩码一次验证 | 第 22 章 |
| 树状掩码 | Tree Attention Mask | 只允许节点注意自身与祖先的注意力掩码 | 第 22 章 |
| KV Cache 回滚 | KV Cache Rollback | 拒绝草稿后把缓存长度截回已接受前缀 | 第 22 章 |
| 量化 | Quantization | 用更低位宽表示权重或激活 | 第 23 章 |
| 训练后量化 | Post-Training Quantization (PTQ) | 训练完成后直接量化，不重新训练 | 第 23 章 |
| 量化感知训练 | Quantization-Aware Training (QAT) | 训练时模拟量化误差 | 第 23 章 |
| 对称/非对称量化 | Symmetric/Asymmetric Quantization | 量化区间是否关于零对称 | 第 23 章 |
| 逐通道/逐组量化 | Per-channel/Per-group Quantization | 每个通道或每组共享一个缩放因子 | 第 23 章 |
| GPTQ | GPTQ | 基于二阶信息的逐层权重量化方法 | 第 23 章 |
| AWQ | Activation-aware Weight Quantization | 根据激活重要性保护显著权重通道 | 第 23 章 |
| SmoothQuant | SmoothQuant | 把激活的量化难度迁移到权重上 | 第 23 章 |
| 离群值 | Outlier | 数值显著大于同组其他元素的激活或权重 | 第 23 章 |
| KV Cache 量化 | KV Cache Quantization | 对缓存的键值做低精度存储 | 第 21 章 |
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
