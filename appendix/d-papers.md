# 附录 D 必读论文清单

本附录按章节汇总全书正文“论文导读”部分的 145 篇论文，每章 5 篇。每条给出作者、年份、标题、出处与一句话贡献，供快速检索与复习使用。

作者一律缩写为“第一作者 等”；只有一到两位作者时保留全部姓氏。出处优先给会议或期刊，其次给 arXiv 编号；没有 arXiv 版本的技术报告与博客式论文注明载体。每篇论文“读哪几节”的详细建议在正文对应章节，本附录不重复。

{% hint style="info" %}
这是一份索引，不是阅读顺序。第一次读某篇论文时，先回到正文对应章节看“推荐阅读”的节号与注意事项，再打开原文；否则容易把论文的结论套到不匹配的模型规模或任务上。表中贡献是压缩后的一句话，省略了前提条件，不能替代原文的适用范围。
{% endhint %}

## D.1 基础篇（第 1–4 章）

| 章 | 作者（年份） | 论文 | 出处 | 一句话贡献 |
| --- | --- | --- | --- | --- |
| 第 1 章 | Shannon（1951） | Prediction and Entropy of Printed English | Bell System Technical Journal | 用猜字实验估计英文熵，建立“预测难度即信息量”的直觉 |
| 第 1 章 | Bengio 等（2003） | A Neural Probabilistic Language Model | JMLR | 首次用分布式表示做神经语言模型，正面攻击 n-gram 维度灾难 |
| 第 1 章 | Vaswani 等（2017） | Attention Is All You Need | NeurIPS；arXiv:1706.03762 | 提出 Transformer，用自注意力替代循环并实现序列维并行 |
| 第 1 章 | Delétang 等（2023） | Language Modeling Is Compression | arXiv:2309.10668 | 证明语言模型可直接作无损压缩器，量化压缩率与能力的关系 |
| 第 1 章 | Gloeckle 等（2024） | Better & Faster Large Language Models via Multi-token Prediction | arXiv:2404.19737 | 系统研究多 Token 预测对训练效率与推理速度的影响 |
| 第 2 章 | Gage（1994） | A New Algorithm for Data Compression | The C Users Journal | BPE 的源头，用合并高频字节对压缩数据，原本并非分词算法 |
| 第 2 章 | Sennrich 等（2016） | Neural Machine Translation of Rare Words with Subword Units | ACL；arXiv:1508.07909 | 把 BPE 引入神经机器翻译，用子词解决稀有词与未登录词 |
| 第 2 章 | Wu 等（2016） | Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation | arXiv:1609.08144 | GNMT 系统报告，给出 WordPiece 的训练与编码方法 |
| 第 2 章 | Kudo（2018） | Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates | ACL；arXiv:1804.10959 | 提出 Unigram 分词与子词正则化，用多种切分增强训练 |
| 第 2 章 | Radford 等（2019） | Language Models are Unsupervised Multitask Learners | OpenAI 技术报告 | GPT-2 采用字节级 BPE 与 50,257 词表，奠定现代分词器形态 |
| 第 3 章 | Vaswani 等（2017） | Attention Is All You Need | NeurIPS；arXiv:1706.03762 | 原始论文，给出缩放点积注意力、多头注意力与位置编码 |
| 第 3 章 | Radford 等（2019） | Language Models are Unsupervised Multitask Learners | OpenAI 技术报告 | GPT-2 确立仅解码器、Pre-LN、GELU 与残差缩放初始化基线 |
| 第 3 章 | Xiong 等（2020） | On Layer Normalization in the Transformer Architecture | ICML；arXiv:2002.04745 | 从梯度尺度解释 Pre-LN 免 warmup、Post-LN 需 warmup 的原因 |
| 第 3 章 | Kaplan 等（2020） | Scaling Laws for Neural Language Models | arXiv:2001.08361 | 给出损失随参数量、数据量与算力的幂律，并讨论形状分配 |
| 第 3 章 | Geva 等（2021） | Transformer Feed-Forward Layers Are Key-Value Memories | EMNLP；arXiv:2012.14913 | 把前馈层解释为键值记忆，说明其参数占比与作用 |
| 第 4 章 | Vaswani 等（2017） | Attention Is All You Need | NeurIPS；arXiv:1706.03762 | 正弦位置编码的出处，注意其编码器—解码器背景 |
| 第 4 章 | Su 等（2021） | RoFormer: Enhanced Transformer with Rotary Position Embedding | arXiv:2104.09864 | 提出 RoPE，用旋转使查询—键内积只依赖相对位移 |
| 第 4 章 | Press 等（2021） | Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation | arXiv:2108.12409 | 提出 ALiBi，用线性距离惩罚实现短训练、长测试 |
| 第 4 章 | Chen 等（2023） | Extending Context Window of Large Language Models via Positional Interpolation | arXiv:2306.15595 | 提出位置插值 PI，把超长位置压缩回训练范围 |
| 第 4 章 | Peng 等（2023） | YaRN: Efficient Context Window Extension of Large Language Models | arXiv:2309.00071 | 提出 YaRN 的 NTK-by-parts 分频插值与温度缩放 |

## D.2 现代架构篇（第 5–8 章）

| 章 | 作者（年份） | 论文 | 出处 | 一句话贡献 |
| --- | --- | --- | --- | --- |
| 第 5 章 | Zhang & Sennrich（2019） | Root Mean Square Layer Normalization | NeurIPS；arXiv:1910.07467 | 提出 RMSNorm，去掉均值中心化与偏置并给出不变性分析 |
| 第 5 章 | Shazeer（2020） | GLU Variants Improve Transformer | arXiv:2002.05202 | 系统比较 GLU 变体，SwiGLU 在同参数预算下表现突出 |
| 第 5 章 | Chowdhery 等（2022） | PaLM: Scaling Language Modeling with Pathways | arXiv:2204.02311 | PaLM 540B 采用 SwiGLU、并行层、无偏置与 RoPE 的完整配置 |
| 第 5 章 | Press & Wolf（2017） | Using the Output Embedding to Improve Language Models | EACL；arXiv:1608.05859 | 研究输入与输出嵌入权重绑定对困惑度和参数量的影响 |
| 第 5 章 | Dubey 等（2024） | The Llama 3 Herd of Models | arXiv:2407.21783 | 给出 Llama 3 系列完整架构与训练配置，是现代开源模型参照 |
| 第 6 章 | Shazeer（2019） | Fast Transformer Decoding: One Write-Head is All You Need | arXiv:1911.02150 | 提出 MQA，所有查询头共享一组 K/V，以表达换解码速度 |
| 第 6 章 | Ainslie 等（2023） | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | arXiv:2305.13245 | 提出 GQA 及从 MHA 检查点上训练到 GQA 的方法 |
| 第 6 章 | DeepSeek-AI（2024） | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | arXiv:2405.04434 | 提出 MLA，把 KV 压成低维潜向量并配合解耦 RoPE |
| 第 6 章 | Su 等（2021） | RoFormer: Enhanced Transformer with Rotary Position Embedding | arXiv:2104.09864 | RoPE 的旋转性质是 MLA 解耦位置键的基础 |
| 第 6 章 | Pope 等（2022） | Efficiently Scaling Transformer Inference | arXiv:2211.05102 | 系统建模推理访存瓶颈、KV 分区与批量策略 |
| 第 7 章 | Shazeer 等（2017） | Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer | arXiv:1701.06538 | 提出稀疏门控 MoE，用带噪 Top-K 路由做到 137B 参数 |
| 第 7 章 | Lepikhin 等（2020） | GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding | arXiv:2006.16668 | 把 Top-2 路由、容量因子与专家并行组合成可扩展系统 |
| 第 7 章 | Fedus 等（2021） | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | arXiv:2101.03961 | 用 Top-1 路由训练到 1.6T 参数并给出完整工程细节 |
| 第 7 章 | Zoph 等（2022） | ST-MoE: Designing Stable and Transferable Sparse Expert Models | arXiv:2202.08906 | 系统研究 MoE 稳定性，提出路由器 z-loss |
| 第 7 章 | Dai 等（2024） | DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models | arXiv:2401.06066 | 用细粒度专家与共享专家减少专家间的知识冗余 |
| 第 8 章 | Katharopoulos 等（2020） | Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention | arXiv:2006.16236 | 用核函数替换 softmax，把注意力写成线性递推 |
| 第 8 章 | Gu 等（2021） | Efficiently Modeling Long Sequences with Structured State Spaces | arXiv:2111.00396 | S4 用 HiPPO 初始化与结构化矩阵实现长程记忆和 O(T log T) 训练 |
| 第 8 章 | Gu & Dao（2023） | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | arXiv:2312.00752 | 让状态空间参数依赖输入，打破时不变性并支持并行扫描 |
| 第 8 章 | Peng 等（2023） | RWKV: Reinventing RNNs for the Transformer Era | arXiv:2305.13048 | 把注意力改写为指数衰减线性递推，可并行训练 |
| 第 8 章 | Lieber 等（2024） | Jamba: A Hybrid Transformer-Mamba Language Model | arXiv:2403.19887 | 组合 Transformer、Mamba 与 MoE，验证 256K 上下文 |

## D.3 预训练篇（第 9–13 章）

| 章 | 作者（年份） | 论文 | 出处 | 一句话贡献 |
| --- | --- | --- | --- | --- |
| 第 9 章 | Kaplan 等（2020） | Scaling Laws for Neural Language Models | arXiv:2001.08361 | 首次系统拟合语言模型缩放幂律，主张固定算力优先加参数 |
| 第 9 章 | Hoffmann 等（2022） | Training Compute-Optimal Large Language Models | arXiv:2203.15556 | Chinchilla 重新估计计算最优配比，70B/1.4T 击败 280B Gopher |
| 第 9 章 | Muennighoff 等（2023） | Scaling Data-Constrained Language Models | arXiv:2305.16264 | 研究数据受限下的重复训练，给出有效数据量与重复轮数上限 |
| 第 9 章 | Sardana 等（2024） | Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws | arXiv:2401.00448 | 把推理成本纳入缩放目标，推导推理感知的最优模型规模 |
| 第 9 章 | Schaeffer 等（2023） | Are Emergent Abilities of Large Language Models a Mirage? | arXiv:2304.15004 | 指出许多涌现是度量选择造成的假象，连续指标下曲线平滑 |
| 第 10 章 | Lee 等（2021） | Deduplicating Training Data Makes Language Models Better | arXiv:2107.06922 | 用后缀数组与 MinHash 做精确和近似去重，证明去重降低记忆化 |
| 第 10 章 | Xie 等（2023） | DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining | arXiv:2305.10429 | 用代理模型与分布鲁棒优化学习领域配比，加速预训练 |
| 第 10 章 | Penedo 等（2023） | The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only | arXiv:2306.01116 | 给出可复现网页清洗管道，证明高质量网页可媲美精选语料 |
| 第 10 章 | Ye 等（2024） | Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance | arXiv:2403.16952 | 发现领域损失随配比指数变化，提出数据混合定律 |
| 第 10 章 | Gunasekar 等（2023） | Textbooks Are All You Need | arXiv:2306.11644 | phi-1 用合成教科书数据证明数据密度可部分替代规模 |
| 第 11 章 | Kingma & Ba（2015） | Adam: A Method for Stochastic Optimization | arXiv:1412.6980 | 用一阶与二阶矩估计加偏差修正做自适应优化 |
| 第 11 章 | Loshchilov & Hutter（2019） | Decoupled Weight Decay Regularization | arXiv:1711.05101 | 解耦权重衰减与自适应梯度，提出 AdamW |
| 第 11 章 | Goyal 等（2017） | Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour | arXiv:1706.02677 | 给出大批量训练的线性缩放规则与 warmup 策略 |
| 第 11 章 | Wortsman 等（2024） | Small-scale proxies for large-scale Transformer training instabilities | arXiv:2309.14322 | 用小型代理模型预测大模型训练失稳，研究稳定化手段 |
| 第 11 章 | Liu 等（2025） | Muon is Scalable for LLM Training | arXiv:2502.16982 | 提出 Muon 优化器并验证其可扩展到 LLM 训练 |
| 第 12 章 | Rajbhandari 等（2020） | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | arXiv:1910.02054 | 分片优化器状态、梯度与参数，使万亿参数训练可行 |
| 第 12 章 | Shoeybi 等（2019） | Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism | arXiv:1909.08053 | 给出 Transformer 的列并行、行并行切分与全归约方案 |
| 第 12 章 | Huang 等（2019） | GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism | arXiv:1811.06965 | 用微批流水线隐藏气泡，提高设备利用率 |
| 第 12 章 | Narayanan 等（2021） | Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM | arXiv:2104.04473 | 系统比较 1F1B 与交错式流水线，统一分析气泡、显存与通信 |
| 第 12 章 | Micikevicius 等（2018） | Mixed Precision Training | arXiv:1710.03740 | 提出混合精度训练的主权重、损失缩放与溢出处理框架 |
| 第 13 章 | Chen 等（2023） | Extending Context Window of Large Language Models via Positional Interpolation | arXiv:2306.15595 | PI 把超长位置压缩回训练范围，用少量微调扩展上下文 |
| 第 13 章 | Peng 等（2023） | YaRN: Efficient Context Window Extension of Large Language Models | arXiv:2309.00071 | YaRN 用分频插值与温度缩放把窗口扩到 128K |
| 第 13 章 | Liu 等（2023） | Lost in the Middle: How Language Models Use Long Contexts | arXiv:2307.03172 | 揭示信息位置对长上下文利用的影响 |
| 第 13 章 | Hsieh 等（2024） | RULER: What's the Real Context Size of Your Long-Context Language Models? | arXiv:2404.06654 | 用 13 类任务区分标称窗口与有效上下文长度 |
| 第 13 章 | Liu 等（2023） | Ring Attention with Blockwise Transformers for Near-Infinite Context | arXiv:2310.01889 | 用块级注意力与环形通信支持近无限上下文 |

## D.4 后训练篇（第 14–19 章）

| 章 | 作者（年份） | 论文 | 出处 | 一句话贡献 |
| --- | --- | --- | --- | --- |
| 第 14 章 | Ouyang 等（2022） | Training Language Models to Follow Instructions with Human Feedback | arXiv:2203.02155 | 建立 SFT、奖励模型与 PPO 的三阶段对齐流程 |
| 第 14 章 | Zhou 等（2023） | LIMA: Less Is More for Alignment | arXiv:2305.11206 | 用 1,000 条精选数据提出“表面对齐”假说 |
| 第 14 章 | Wang 等（2023） | Self-Instruct: Aligning Language Models with Self-Generated Instructions | arXiv:2212.10560 | 用自生成指令扩充 SFT 数据 |
| 第 14 章 | Touvron 等（2023） | Llama 2: Open Foundation and Fine-Tuned Chat Models | arXiv:2307.09288 | 给出 SFT 数据规模、模板与拒绝采样加 PPO 的完整流程 |
| 第 14 章 | Wei 等（2022） | Finetuned Language Models Are Zero-Shot Learners | arXiv:2109.01652 | 证明指令微调能提升未见任务的零样本迁移 |
| 第 15 章 | Hu 等（2021） | LoRA: Low-Rank Adaptation of Large Language Models | arXiv:2106.09685 | 用低秩分解把微调参数量降低数个量级 |
| 第 15 章 | Dettmers 等（2023） | QLoRA: Efficient Finetuning of Quantized LLMs | arXiv:2305.14314 | 用 NF4、双重量化与分页优化器实现 4-bit 微调 |
| 第 15 章 | Houlsby 等（2019） | Parameter-Efficient Transfer Learning for NLP | arXiv:1902.00751 | 提出 Adapter 瓶颈结构，开启模块化参数高效微调 |
| 第 15 章 | Li & Liang（2021） | Prefix-Tuning: Optimizing Continuous Prompts for Generation | arXiv:2101.00190 | 优化每层连续键值前缀而非模型权重 |
| 第 15 章 | Liu 等（2024） | DoRA: Weight-Decomposed Low-Rank Adaptation | arXiv:2402.09353 | 把权重分解为幅度与方向，只对方向做低秩更新 |
| 第 16 章 | Christiano 等（2017） | Deep Reinforcement Learning from Human Preferences | arXiv:1706.03741 | 首次用人类成对比较训练奖励模型，替代手工奖励 |
| 第 16 章 | Stiennon 等（2020） | Learning to Summarize from Human Feedback | arXiv:2009.01325 | 把 RLHF 用于摘要生成，展示奖励模型规模扩展的收益 |
| 第 16 章 | Ouyang 等（2022） | Training Language Models to Follow Instructions with Human Feedback | arXiv:2203.02155 | 三阶段 RLHF 流程与人类评测的工业范本 |
| 第 16 章 | Bai 等（2022） | Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback | arXiv:2204.05862 | 把 helpful 与 harmless 作为独立目标训练 HH 助手 |
| 第 16 章 | Schulman 等（2017） | Proximal Policy Optimization Algorithms | arXiv:1707.06347 | PPO 的裁剪目标与 GAE，是 RLHF 的策略优化基础 |
| 第 17 章 | Rafailov 等（2023） | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | arXiv:2305.18290 | 消掉显式奖励模型，把偏好学习变成一次监督式训练 |
| 第 17 章 | Azar 等（2023） | A General Theoretical Paradigm to Understand Learning from Human Preferences | arXiv:2310.12036 | IPO 给出偏好学习的通用理论框架，解释 DPO 的过拟合风险 |
| 第 17 章 | Ethayarajh 等（2024） | KTO: Model Alignment as Prospect Theoretic Optimization | arXiv:2402.01306 | 用前景理论把二元反馈转成对齐目标 |
| 第 17 章 | Meng 等（2024） | SimPO: Simple Preference Optimization with a Reference-Free Reward | arXiv:2405.14734 | 用长度归一化与目标间隔实现无参考模型的偏好优化 |
| 第 17 章 | Hong 等（2024） | ORPO: Monolithic Preference Optimization without Reference Model | arXiv:2403.07691 | 把 odds ratio 并入 SFT 损失，实现单阶段对齐 |
| 第 18 章 | Shao 等（2024） | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | arXiv:2402.03300 | 提出 GRPO，用组内相对优势去掉价值网络 |
| 第 18 章 | DeepSeek-AI（2025） | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | arXiv:2501.12948 | 用纯 RL 激发推理，给出 R1-Zero 与多阶段训练流程 |
| 第 18 章 | Yu 等（2025） | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | arXiv:2503.14476 | 提出 clip-higher、动态采样、Token 级损失与超长奖励塑形 |
| 第 18 章 | Liu 等（2025） | Understanding R1-Zero-Like Training: A Critical Perspective | arXiv:2503.20783 | 分析 R1-Zero 类训练中长度与标准差归一化的偏差 |
| 第 18 章 | Zheng 等（2025） | Group Sequence Policy Optimization | arXiv:2507.18071 | 用序列级重要性比解决长思维链的方差问题 |
| 第 19 章 | Hinton 等（2015） | Distilling the Knowledge in a Neural Network | arXiv:1503.02531 | 蒸馏开篇，用温度 Softmax 与软目标传递暗知识 |
| 第 19 章 | Sanh 等（2019） | DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter | arXiv:1910.01108 | 组合输出蒸馏与隐藏层对齐来压缩 BERT |
| 第 19 章 | Agarwal 等（2024） | On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes | arXiv:2306.13649 | 用 JSD 统一前向与反向 KL，实现 on-policy 蒸馏 |
| 第 19 章 | DeepSeek-AI（2025） | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | arXiv:2501.12948 | 序列级蒸馏实验，把推理能力迁移到小模型 |
| 第 19 章 | Frantar & Alistarh（2023） | SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot | arXiv:2301.11093 | 用逐层重建一次性剪枝大语言模型 |

## D.5 推理与效率篇（第 20–24 章）

| 章 | 作者（年份） | 论文 | 出处 | 一句话贡献 |
| --- | --- | --- | --- | --- |
| 第 20 章 | Fan 等（2018） | Hierarchical Neural Story Generation | arXiv:1805.04833 | Top-k 采样的早期代表，用截断平衡跑题与无聊 |
| 第 20 章 | Holtzman 等（2019） | The Curious Case of Neural Text Degeneration | arXiv:1904.09751 | 揭示束搜索退化，提出 Top-p 核采样 |
| 第 20 章 | Chen 等（2021） | Evaluating Large Language Models Trained on Code | arXiv:2107.03374 | HumanEval 与 pass@k 无偏估计的出处 |
| 第 20 章 | Meister 等（2022） | Locally Typical Sampling | arXiv:2202.00666 | 用信息论典型集替代高概率前缀 |
| 第 20 章 | Nguyen 等（2024） | Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs | arXiv:2407.01082 | Min-p 用相对最大概率阈值替代 Top-p 的绝对累积质量 |
| 第 21 章 | Dao 等（2022） | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | arXiv:2205.14135 | 用分块与在线 Softmax 实现 IO 感知的精确注意力 |
| 第 21 章 | Dao（2023） | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | arXiv:2307.08691 | 改进并行划分，减少非矩阵乘 FLOPs 与通信 |
| 第 21 章 | Kwon 等（2023） | Efficient Memory Management for Large Language Model Serving with PagedAttention | arXiv:2309.06180 | 用块表与写时复制管理 KV Cache，支撑 vLLM |
| 第 21 章 | Agrawal 等（2024） | Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve | arXiv:2403.02310 | 用分块预填充与无停顿调度解耦迭代时间 |
| 第 21 章 | Zhong 等（2024） | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving | arXiv:2401.09670 | 分离预填充与解码，按 goodput 优化资源放置 |
| 第 22 章 | Leviathan 等（2022） | Fast Inference from Transformers via Speculative Decoding | arXiv:2211.17192 | 提出投机解码的接受—拒绝规则并证明输出无损 |
| 第 22 章 | Chen 等（2023） | Accelerating Large Language Model Decoding with Speculative Sampling | arXiv:2302.01318 | 用拒绝采样保持目标分布，给出分布式投机采样流程 |
| 第 22 章 | Cai 等（2024） | Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads | arXiv:2401.10774 | 用多头与树状候选做无草稿模型的投机解码 |
| 第 22 章 | Li 等（2024） | EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty | arXiv:2401.15077 | 在特征层自回归草稿，并用树验证提高接受率 |
| 第 22 章 | DeepSeek-AI（2024） | DeepSeek-V3 Technical Report | arXiv:2412.19437 | 把多 Token 预测写进预训练目标 |
| 第 23 章 | Jacob 等（2018） | Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference | arXiv:1712.05877 | 给出仿射量化、零点与整数-only 推理的标准公式 |
| 第 23 章 | Dettmers 等（2022） | LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale | arXiv:2208.07339 | 用混合精度分解处理激活离群值 |
| 第 23 章 | Frantar 等（2023） | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | arXiv:2210.17323 | 用逐列量化与 Hessian 补偿实现 3–4 bit 权重量化 |
| 第 23 章 | Xiao 等（2023） | SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models | arXiv:2211.10438 | 用通道缩放把激活难度搬到权重，实现 W8A8 |
| 第 23 章 | Lin 等（2024） | AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration | arXiv:2306.00978 | 用激活感知缩放保护显著权重，面向端侧部署 |
| 第 24 章 | Wang 等（2022） | Self-Consistency Improves Chain of Thought Reasoning in Language Models | arXiv:2203.11171 | 用多条思维链投票替代单条贪心解码 |
| 第 24 章 | Lightman 等（2023） | Let's Verify Step by Step | arXiv:2305.20050 | 过程监督与 PRM800K 证明逐步验证优于结果监督 |
| 第 24 章 | Yao 等（2023） | Tree of Thoughts: Deliberate Problem Solving with Large Language Models | arXiv:2305.10601 | 把推理组织成可搜索的树，并用评估函数剪枝 |
| 第 24 章 | Snell 等（2024） | Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters | arXiv:2408.03314 | 在等 FLOPs 下比较测试时算力与参数扩展，给出按难度分配策略 |
| 第 24 章 | Brown 等（2024） | Large Language Monkeys: Scaling Inference Compute with Repeated Sampling | arXiv:2407.21787 | 大规模重复采样展示覆盖率对数增长与验证器瓶颈 |

## D.6 拓展与前沿篇（第 25–29 章）

| 章 | 作者（年份） | 论文 | 出处 | 一句话贡献 |
| --- | --- | --- | --- | --- |
| 第 25 章 | Radford 等（2021） | Learning Transferable Visual Models From Natural Language Supervision | arXiv:2103.00020 | CLIP 用对比学习对齐图文，实现强零样本迁移 |
| 第 25 章 | Li 等（2023） | BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models | arXiv:2301.12597 | 用 Q-Former 与冻结双塔做轻量多模态对齐 |
| 第 25 章 | Liu 等（2023） | Visual Instruction Tuning | arXiv:2304.08485 | LLaVA 用线性投影连接视觉编码器与 LLM，并构造指令数据 |
| 第 25 章 | Chameleon Team（2024） | Chameleon: Mixed-Modal Early-Fusion Foundation Models | arXiv:2405.09818 | 原生离散多模态，用统一词表与早期融合训练 |
| 第 25 章 | Zhou 等（2024） | Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model | arXiv:2408.11039 | 在同一 Transformer 中结合自回归与扩散损失 |
| 第 26 章 | Yao 等（2022） | ReAct: Synergizing Reasoning and Acting in Language Models | arXiv:2210.03629 | 交替生成推理与动作，把思考落到可执行轨迹 |
| 第 26 章 | Schick 等（2023） | Toolformer: Language Models Can Teach Themselves to Use Tools | arXiv:2302.04761 | 用自监督学习何时调用 API 及如何填参 |
| 第 26 章 | Qin 等（2023） | ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs | arXiv:2307.16789 | 构造 16,000 以上 API 的工具学习数据与评测 |
| 第 26 章 | Yang 等（2024） | SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering | arXiv:2405.15793 | 研究智能体—计算机接口对能力的影响 |
| 第 26 章 | Liu 等（2023） | AgentBench: Evaluating LLMs as Agents | arXiv:2308.03688 | 用 8 类环境系统评测 LLM 智能体 |
| 第 27 章 | Liang 等（2022） | Holistic Evaluation of Language Models | arXiv:2211.09110 | 提出多场景、多指标的整体评测框架 |
| 第 27 章 | Hendrycks 等（2020） | Measuring Massive Multitask Language Understanding | arXiv:2009.03300 | 用 57 学科题目测量大规模多任务知识 |
| 第 27 章 | Chen 等（2021） | Evaluating Large Language Models Trained on Code | arXiv:2107.03374 | HumanEval 与 pass@k 的执行式代码评测 |
| 第 27 章 | Zheng 等（2023） | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | arXiv:2306.05685 | 研究模型裁判的能力与位置、冗长、自我偏好偏差 |
| 第 27 章 | Chiang 等（2024） | Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference | arXiv:2403.04132 | 用成对投票与 Bradley–Terry/Elo 排名 |
| 第 28 章 | Elhage 等（2021） | A Mathematical Framework for Transformer Circuits | Transformer Circuits Thread | 提出残差流、QK/OV 分解与头组合的回路框架 |
| 第 28 章 | Olsson 等（2022） | In-context Learning and Induction Heads | arXiv:2209.11895 | 发现归纳头及其相变，联系上下文学习 |
| 第 28 章 | Wang 等（2022） | Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small | arXiv:2211.00593 | 用消融与修补定位 GPT-2 small 的间接宾语识别回路 |
| 第 28 章 | Templeton 等（2024） | Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet | Transformer Circuits Thread | 用 SAE 从 Claude 3 Sonnet 提取可解释特征并做 steering |
| 第 28 章 | Gao 等（2024） | Scaling and Evaluating Sparse Autoencoders | arXiv:2406.04093 | 系统研究 SAE 的 L1/TopK 取舍与重构—稀疏缩放规律 |
| 第 29 章 | Kirkpatrick 等（2017） | Overcoming Catastrophic Forgetting in Neural Networks | PNAS 114(13)；arXiv:1612.00796 | 提出 EWC，用 Fisher 信息加二次惩罚缓解灾难性遗忘 |
| 第 29 章 | Shumailov 等（2024） | AI Models Collapse When Trained on Recursively Generated Data | Nature 631；arXiv:2305.17493 | 证明递归生成数据训练会导致模型崩溃与分布尾部消失 |
| 第 29 章 | DeepSeek-AI（2025） | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | arXiv:2501.12948 | 给出 RLVR、GRPO 与规则奖励的设计与失败模式 |
| 第 29 章 | Gu & Dao（2023） | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | arXiv:2312.00752 | 选择性状态空间是高效架构的代表路线 |
| 第 29 章 | Schaeffer 等（2023） | Are Emergent Abilities of Large Language Models a Mirage? | arXiv:2304.15004 | 质疑涌现，指出度量非线性制造假象 |

## D.7 按目标的阅读路线

如果不想按章节顺序读，可以按目标选一条路线。每条只列主干，分支回到对应章的完整五篇。

| 目标 | 建议路线 | 选这条路的理由 |
| --- | --- | --- |
| 建立全局框架 | 第 3 章 → 第 5 章 → 第 9 章 → 第 14 章 → 第 21 章 → 第 27 章 | 架构、现代组件、规模、对齐、推理系统、评测各取一篇代表作 |
| 做预训练 | 第 3 章 → 第 5 章 → 第 9 章 → 第 10 章 → 第 11 章 → 第 12 章 | 从模型结构走到数据、优化器与分布式系统 |
| 做后训练 | 第 14 章 → 第 16 章 → 第 17 章 → 第 18 章 → 第 19 章 | SFT、RLHF、DPO、可验证奖励 RL、蒸馏依次衔接 |
| 做推理部署 | 第 20 章 → 第 21 章 → 第 22 章 → 第 23 章 → 第 24 章 | 解码策略、KV 与调度、投机解码、量化、测试时算力 |
| 做前沿研究 | 第 8 章 → 第 13 章 → 第 18 章 → 第 24 章 → 第 28 章 → 第 29 章 | 高效架构、长上下文、推理 RL、测试时算力、可解释性、开放问题 |

{% hint style="warning" %}
论文里的加速比、质量提升与缩放指数都绑定特定的模型规模、数据分布、硬件与评测口径。跨论文比较数字前，先确认三件事：基线是否同一套实现，指标是否同一口径，硬件与批量是否可比。本附录的一句话贡献只保留结论，不携带这些条件。
{% endhint %}
