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

## 第 5 章

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 均方根归一化 | RMSNorm | 只除以均方根、不减均值的归一化 | 第 5 章 |
| 门控线性单元 | Gated Linear Unit (GLU) | 两路线性投影逐元素相乘的前馈结构 | 第 5 章 |
| SwiGLU | SwiGLU | 用 SiLU 做门控激活的 GLU 变体 | 第 5 章 |
| GeGLU | GeGLU | 用 GELU 做门控激活的 GLU 变体 | 第 5 章 |
| SiLU | SiLU (Swish) | 由 Sigmoid 自门控的激活 $$z\sigma(z)$$ | 第 5 章 |
| 权重绑定 | Weight Tying | 输入嵌入与输出投影共享同一矩阵 | 第 5 章 |
| 并行层 | Parallel Layers | 注意力与前馈共享归一化并并行相加的块结构 | 第 5 章 |
| 去偏置 | Bias-free | 去除线性层与归一化偏置的模型配置 | 第 5 章 |
| 等参数预算点 | Iso-parameter Budget | 替换前后参数量与 FLOPs 对齐的配置点 | 第 5 章 |

## 第 6 章

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 多查询注意力 | Multi-Query Attention (MQA) | 所有查询头共享一套 K/V 的注意力变体 | 第 6 章 |
| 分组查询注意力 | Grouped-Query Attention (GQA) | 查询头分组、每组共享一套 K/V 的注意力变体 | 第 6 章 |
| 多头潜在注意力 | Multi-head Latent Attention (MLA) | 把 K/V 压缩到低维潜向量再按头上投影的注意力 | 第 6 章 |
| KV 头 | KV Head | 被若干查询头共享的一组键值投影 | 第 6 章 |
| KV 缓存 | KV Cache | 缓存历史键值以避免解码时重复计算 | 第 6 章 |
| 预填充 | Prefill | 处理完整提示、一次性建立 KV 缓存的阶段 | 第 6 章 |
| 解码 | Decode | 逐 Token 生成、每步读取 KV 缓存的阶段 | 第 6 章 |
| 访存受限 | Memory-bound | 性能由搬运字节数而非浮点运算数决定的负载 | 第 6 章 |
| 算术强度 | Arithmetic Intensity | 每搬运一字节所执行的浮点运算数 | 第 6 章 |
| 解耦 RoPE | Decoupled RoPE | 把位置键拆到独立小通路的 MLA 位置编码方案 | 第 6 章 |
| 权重吸收 | Weight Absorption | 把上投影合并进查询与输出投影的推理变换 | 第 6 章 |
| 分页注意力 | Paged Attention | 按固定大小块管理 KV 缓存、减少碎片的机制 | 第 6 章 |
| 前缀缓存 | Prefix Caching | 多个请求共享相同前缀 KV 的计算复用 | 第 6 章 |
| 继续训练转换 | Uptraining | 从 MHA 检查点均值池化 KV 头后少量继续训练 | 第 6 章 |
| 滑动窗口注意力 | Sliding Window Attention | 每个位置只关注最近若干位置的注意力 | 第 6 章 |

## 第 7 章

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 混合专家 | Mixture of Experts (MoE) | 用路由器为每个 Token 选择少数专家计算的前馈结构 | 第 7 章 |
| 专家 | Expert | 参数独立、结构相同的前馈子网络 | 第 7 章 |
| 路由器 | Router | 为每个 Token 计算专家打分与门控的线性层 | 第 7 章 |
| 稀疏激活 | Sparse Activation | 每 Token 只计算全部参数中的一小部分 | 第 7 章 |
| Top-K 路由 | Top-K Routing | 只激活门控概率最高的 $$k$$ 个专家 | 第 7 章 |
| 容量因子 | Capacity Factor (CF) | 每个专家可用槽位相对均衡份额的倍数 | 第 7 章 |
| Token 丢弃 | Token Dropping | 专家超出容量时丢弃多余分派 | 第 7 章 |
| 无丢弃 | Dropless | 不设固定容量、按实际分派计算的实现 | 第 7 章 |
| 负载均衡辅助损失 | Load Balancing Auxiliary Loss | 惩罚专家负载不均、调节路由概率的辅助目标 | 第 7 章 |
| 路由器 z-loss | Router z-loss | 约束路由 logits 幅度、提升训练稳定性的正则项 | 第 7 章 |
| 共享专家 | Shared Expert | 所有 Token 都经过、不被稀疏化的公共专家 | 第 7 章 |
| 细粒度专家 | Fine-grained Expert | 把专家切窄、同时提高激活个数以保持计算预算的做法 | 第 7 章 |
| Expert Choice | Expert Choice Routing | 由专家挑选固定数量 Token 的反向路由 | 第 7 章 |
| 无辅助损失均衡 | Auxiliary-loss-free Balancing | 用专家偏置反馈而非损失项控制负载 | 第 7 章 |
| 专家并行 | Expert Parallelism (EP) | 把不同专家分布到不同设备、按路由通信 | 第 7 章 |
| 全对全通信 | All-to-All | 专家并行中 Token 与输出在设备间重排的集合通信 | 第 7 章 |
| 分组受限路由 | Group-limited Routing | 限制每个 Token 的专家落在少数设备组内 | 第 7 章 |
| 总参数 | Total Parameters | 全部专家与其余模块的参数之和 | 第 7 章 |
| 激活参数 | Activated Parameters | 每个 Token 实际参与计算的参数量 | 第 7 章 |
| 激活比 | Activation Ratio | 激活参数与总参数之比 | 第 7 章 |

## 第 8 章

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 线性注意力 | Linear Attention | 用可分解核替代 softmax、把历史压进矩阵状态的注意力 | 第 8 章 |
| 核特征映射 | Kernel Feature Map | 把查询与键映射到可做内积的有限维空间的函数 $$\phi$$ | 第 8 章 |
| 前缀和 | Prefix Sum | 沿序列维累积外积、一次得到所有位置状态的并行操作 | 第 8 章 |
| 分块线性注意力 | Chunked Linear Attention | 块内二次、块间递推以控制中间显存的长序列实现 | 第 8 章 |
| 状态空间模型 | State Space Model (SSM) | 用线性微分方程或递推描述隐藏状态演化的序列模型 | 第 8 章 |
| 零阶保持 | Zero-Order Hold (ZOH) | 假设步长内输入不变、把连续系统离散化的方法 | 第 8 章 |
| 线性时不变 | Linear Time-Invariant (LTI) | 参数不随位置变化、可用固定卷积核描述的系统 | 第 8 章 |
| 结构化状态空间 | Structured State Space (S4) | 用 HiPPO 初始化与结构化矩阵建模长序列的 SSM | 第 8 章 |
| 选择性状态空间 | Selective State Space / Mamba | 让步长与输入/输出矩阵依赖内容、用并行扫描训练的 SSM | 第 8 章 |
| 选择性 | Selectivity | 按输入内容控制状态写入、保留与读取的能力 | 第 8 章 |
| 并行扫描 | Parallel Scan | 利用结合律在 $$O(\log T)$$ 深度内计算前缀递推 | 第 8 章 |
| 门控线性递推单元 | Gated Linear Recurrent Unit | 以逐通道门控衰减更新状态的线性递推结构 | 第 8 章 |
| 混合架构 | Hybrid Architecture | 交替使用注意力层与线性/SSM 层的模型结构 | 第 8 章 |
| 闪电注意力 | Lightning Attention | 分块实现的分块线性注意力，用于超长上下文 | 第 8 章 |
| 门控 DeltaNet | Gated DeltaNet | 带门控衰减与 delta rule 更新的线性注意力变体 | 第 8 章 |
| 状态账本 | State Ledger | 按层统计缓存/状态字节数随序列长度变化的账目 | 第 8 章 |

## 第 9 章

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
| 缩放定律 | Scaling Law | 损失随参数量、数据量、算力变化的经验规律 | 第 9 章 |
| 幂律 | Power Law | 两个量按固定指数成比例变化的函数关系 | 第 9 章 |
| 不可约损失 | Irreducible Loss | 模型与数据无限大时仍无法消除的损失下限 $$L_\infty$$ | 第 9 章 |
| 计算最优 | Compute-Optimal | 在固定训练算力下使损失最小的参数/数据配比 | 第 9 章 |
| 推理最优 | Inference-Optimal | 把推理成本计入总成本后选择的更小模型、更多数据配比 | 第 9 章 |
| 过训练 | Overtraining | 训练 Token 数显著超过计算最优配比的做法 | 第 9 章 |
| 数据受限 | Data-Constrained | 唯一数据不足、必须重复训练的情形 | 第 9 章 |
| 唯一数据 | Unique Data | 去重后可用的不重复 Token 总量 $$U$$ | 第 9 章 |
| 有效数据量 | Effective Data | 重复数据折算成全新数据后的等效 Token 数 $$D_{\mathrm{eff}}$$ | 第 9 章 |
| 数据重复 | Data Repetition | 同一语料被训练多轮；用重复轮数 $$R=D/U-1$$ 度量 | 第 9 章 |
| 关键批大小 | Critical Batch Size | 继续增大批大小、收益开始明显递减的临界点 $$B_{\mathrm{crit}}$$ | 第 9 章 |
| 线性缩放规则 | Linear Scaling Rule | 批大小远小于关键值时，学习率与批大小成正比 | 第 9 章 |
| 平方根缩放规则 | Square-Root Scaling Rule | 批大小远大于关键值时，学习率与批大小的平方根成正比 | 第 9 章 |
| 最大更新参数化 | Maximal Update Parametrization (μP) | 使超参数可跨宽度、跨深度迁移的参数化方案 | 第 9 章 |
| 参数迁移 | Hyperparameter Transfer | 在小模型上调好超参数、直接用于大模型的做法 | 第 9 章 |
| 涌现能力 | Emergent Abilities | 规模超过某阈值后评测分数突然跳升的现象 | 第 9 章 |
| 相变 | Phase Transition | 系统宏观行为在参数越过临界点时发生质变 | 第 9 章 |
| 逆缩放 | Inverse Scaling | 规模增大反而使某项指标变差的现象 | 第 9 章 |
| IsoFLOP 曲线 | IsoFLOP Curve | 固定训练算力时，损失随参数/数据分配变化的曲线 | 第 9 章 |
| Token/参数比 | Tokens per Parameter | 训练 Token 数与参数量的比值，常用作数据充足度的粗指标 | 第 9 章 |
| 算力账本 | Compute Ledger | 把训练、推理各项 FLOPs 分项相加的核算表 | 第 9 章 |

## 第 10 章

| 中文 | 英文 | 一句话定义 | 首次出现 |
| --- | --- | --- | --- |
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

## 第 11 章

| 术语 | 英文 | 说明 | 首次出现 |
| --- | --- | --- | --- |
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
- "RMSNorm" 固定大小写，首次出现附中文"均方根归一化"，不写作 "RMS Norm"。
- "SwiGLU""GeGLU""SiLU" 固定大小写，首次出现附中文或公式说明；不写作 "swiglu"。
- "等预算"统一指参数与 FLOPs 同时对齐；只说参数对齐时写"等参数"。
- "门控"指乘性门控，不用于描述注意力掩码；掩码统一写"因果掩码"。
- "前馈宽度"统一指 $$d_{ff}$$，不写作"隐藏层大小"。
- "权重绑定"对应 weight tying，不与"参数共享"（跨层共享）混用。
- "并行层"特指注意力与前馈共享同一归一化的块结构，不泛指数据并行或张量并行。
- "MoE" 首次出现附中文"混合专家"，之后可直接用缩写；"专家"专指 MoE 中的前馈子网络，不用于泛指模型。
- 专家数统一写 $$E$$，每 Token 激活数写 $$k$$，共享专家数写 $$S_{\text{sh}}$$，专家容量写 $$C_{\text{cap}}$$；代码中的 `n_experts`、`top_k`、`n_shared` 只在代码语境出现。
- "激活参数"与"总参数"必须成对出现，不把"激活参数"简称为"参数量"；"激活比"固定指两者之比。
- "负载均衡"不写作"负载平衡"；"容量因子"不写作"容量系数"；"Token 丢弃"不写作"Token 溢出"。
- "专家并行"对应 expert parallelism，"全对全"对应 All-to-All；不混用"专家分片"。
- "Expert Choice" 与 "Top-K 路由" 并列时保留英文，首次出现附中文解释。
- "线性注意力"统一对应 linear attention，首次出现附英文；这里的"线性"指对序列长度线性，不指线性层。
- "状态空间模型"首次出现附 SSM；"选择性 SSM"与"Mamba"可互换，但指论文方法时写 Mamba。
- "固定状态"与"定长状态"择一使用，正文统一为"固定状态"；不写作"常数状态"。
- "混合架构"统一对应 hybrid architecture；具体比例写"注意力:线性层 = 1:m"或"每 m 层一个注意力层"。
- "衰减"描述状态按系数缩小，"遗忘"描述模型主动丢弃信息；两者不互换。
- "计算最优"固定对应 compute-optimal，不写作"最优计算"；"推理最优"对应 inference-optimal，二者不混用。
- "过训练"统一对应 overtraining；指训练 Token 超过计算最优配比，不含"训练过头导致过拟合"的贬义。
- "唯一数据"对应 unique data（去重后的语料），"有效数据量"对应 effective data（重复折算后的等效量）；两者不互换。
- "数据重复"指多轮训练的动作，用重复轮数 $$R=D/U-1$$ 度量；不写作"重复数据"。
- "关键批大小"对应 critical batch size，符号统一写 $$B_{\mathrm{crit}}$$；不写作"临界批量"。
- "线性缩放规则"与"平方根缩放规则"成对出现，分别对应批大小小于/大于关键值的两段；不统称为"批大小缩放"。
- "μP" 固定对应 maximal update parametrization，首次出现附中文"最大更新参数化"，之后可直接用缩写。
- "涌现能力"描述评测分数跳升的现象，"相变"描述系统行为的质变；不把两者当作同义词，讨论机制时才用"相变"。
- "逆缩放"对应 inverse scaling，指规模增大指标变差；不写作"负缩放"。
- "IsoFLOP 曲线"固定大小写，不写 "iso-flop"；"等算力曲线"可作中文同义表述。
- "Token/参数比"统一写"20 Token/参数"这类形式；不中英混排为 "20 tokens per parameter"。
- "算力账本"对应 compute ledger，与第 8 章的"状态账本"（state ledger）并列，不互换。
- "Chinchilla" 固定首字母大写，不写作 "chinchilla"；指论文方法时保留英文。
- "数据管道"对应 data pipeline，"数据漏斗"对应 data funnel；前者是流程，后者是保留率模型，不互换。
- "精确去重"与"近似去重"成对出现；不把后者写作"模糊去重"。
- "去重"删除重复副本，"去污染"删除与评测集重叠的样本；两者目标不同，不混用。
- "质量过滤"是清洗的一环，"数据清洗"是包含抽取、去重、去污染的上位概念；正文不把两者等同。
- "shingle" 首次出现附中文"连续 n 元组"，之后保留英文小写；不写作"切片"。
- "MinHash" 固定大小写，不写作 "minhash"；"LSH" 首次出现附中文"局部敏感哈希"。
- "配比"对应 data mixture，"混合"单独出现时可能指模型混合或数据混合；数据语境统一写"配比"。
- "温度采样"在本章专指配比采样，不指解码温度；解码温度按第 20 章写"采样温度"。
- "退火"是训练末段的数据切换，必须与学习率衰减同时出现；"数据课程"是更宽的阶段安排，不把两者等同。
- "合成数据"对应 synthetic data，不写作"生成数据"；"模型坍缩"对应 model collapse，不写作"模式崩溃"。
- "污染"指训练集含评测样本或近似副本，"泄漏"指具体的泄漏实例；正文先写"评测污染"，指某条样本时写"泄漏"。
- "n-gram 覆盖率"统一对应 n-gram coverage，符号写 $$c(d)$$；不写作"重叠率"。
- "金丝雀串"对应 canary string，首次出现附英文；不写作"水印"（水印是另一类技术）。
- "产出率"对应 yield，指保留文档比例；"保留率"在漏斗语境指每一级的保留比例，两者数值可能相同但语境不同，首次出现要写清。
- "优化器"指更新规则本身，"学习率调度"指乘子随步数变化；不把两者统称为"训练策略"。
- "一阶矩"与"二阶矩"成对出现；不写作"动量"与"方差"（后者只在统计语境使用）。
- "权重衰减"在 Adam 语境分"耦合 L2"与"解耦衰减"；单说"权重衰减"时默认指解耦写法，首次出现须说明。
- "预热"对应 warmup，不写作"热启动"；预热步数统一写 $$T_w$$。
- "余弦退火"是学习率调度，不指数据退火；数据退火按第 10 章写"退火"或"数据退火"。
- "WSD 调度"首次出现附全称 Warmup-Stable-Decay，之后可用缩写；不把"三段式调度"当作唯一名称。
- "关键批大小"与"批大小"区分：前者是噪声尺度的阈值，后者是单步样本数。
- "梯度裁剪"指全局范数裁剪，不指按值裁剪；后者须明确写"按值裁剪"。
- "损失尖峰"对应 loss spike；"尖峰后损失"是诊断量，不写作"损失爆炸"（爆炸指发散）。
- "注意力熵"是 softmax 权重的熵，不指模型输出的熵；符号写 $$H_t$$。
- "QK-norm"与"QK-Clip"大小写固定；前者每步归一化，后者按阈值缩放，不互换。
- "z-loss" 小写 z，保留连字符；不写作 "Z loss"。
- "Muon" 首字母大写；"Newton-Schulz 迭代"保留连字符；"正交化"指奇异值归一，不指 Gram-Schmidt。
- "主权重"对应 master weights，不写作"主副本"。
- "梯度累积"是多个微批后更新一次，不写作"梯度累加"；与"累积误差"无关。
