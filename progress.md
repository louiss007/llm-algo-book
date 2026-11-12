# 写作进度与分会话交接

## 已完成章节

| 章节 | 状态 | 正文中文字数 | 代码验证 | 备注 |
| --- | --- | --- | --- | --- |
| 第 1 章 | 完成 | 约 7100 | 已在本机运行 | 含可控马尔可夫链实验 |
| 第 2 章 | 完成 | 约 8100 | 已在本机运行 | 字节级 BPE 与词表扫描实验 |
| 第 3 章 | 完成 | 约 10300 | 已在本机运行 | Transformer 基线、因果性测试与复制任务 |
| 第 4 章 | 完成 | 7579 | 已在本机运行 | 位置编码分派、RoPE/ALiBi、长度外推受控实验 |
| 第 5 章 | 完成 | 7526 | 已在本机运行 | RMSNorm/SwiGLU/去偏置/绑定/并行层，参数账本与复制任务消融 |
| 第 6 章 | 完成 | 10516 | 已在本机运行 | MHA/MQA/GQA/MLA、KV Cache 账本、权重吸收等价性与复制任务消融 |
| 第 7 章 | 完成 | 10543 | 已在本机运行 | MoE 路由/容量/辅助损失、参数与激活账本、通信账本、复制任务消融 |
| 第 8 章 | 完成 | 7911 | 已在本机运行 | 线性注意力、S4/Mamba、RWKV、混合架构；状态/算力账本与复制任务消融 |
| 第 9 章 | 完成 | 7585 | 已在本机运行 | 双幂律拟合、计算最优分配、数据重复、训练+推理账本；玩具扫描与 9 项测试 |
| 第 10 章 | 完成 | 约 7853 | 已在本机运行 | 数据管道漏斗、MinHash/LSH 去重、质量过滤、配比与污染检测；13 项测试 |
| 第 11 章 | 完成 | 约 10368 | 已在本机运行 | AdamW/调度/裁剪/Muon；病态二次型与 3 种子对照；13 项测试 |
| 第 12 章 | 完成 | 11122 | 已在本机运行 | 显存/通信账本、ZeRO 四阶段、TP/PP/EP、损失缩放；4 种训练模式 3 种子等价；16 项测试 |
| 第 13 章 | 完成 | 8146 | 已在本机运行 | 长上下文算力/显存账本、长度课程与混合长度、位置外推、打包等价性、NIAH 评测；11 项测试 |
| 第 14 章 | 完成 | 8481 | 已在本机运行 | 对话模板与损失掩码、Token/序列平均、监督比例、随机/系统噪声、打包等价、拒绝采样；14 项测试 |
| 第 15 章 | 完成 | 7804 | 已在本机运行 | LoRA 低秩更新/合并、秩与目标模块消融、QLoRA 的 NF4/双重量化、参数与激活账本；5 种子对照，10 项测试 |
| 第 16 章 | 完成 | 11095 | 已在本机运行 | Bradley–Terry 奖励模型、KL 正则闭式解、PPO 裁剪/GAE/价值网络、k1/k3 KL；3 种子对照与奖励作弊探针，13 项测试 |
| 第 17 章 | 完成 | 10423 | 已在本机运行 | 隐式奖励与 DPO 损失/梯度、长度归一化、IPO/KTO/SimPO/ORPO；偏置/去偏偏好数据 3 种子对照，12 项测试 |
| 第 18 章 | 完成 | 10891 | 已在本机运行 | RLVR 与 GRPO、组内优势/裁剪/KL、零方差组；模运算任务 3 种子 G 消融，17 项测试 |
| 第 19 章 | 完成 | 7915 | 已在本机运行 | 温度软标签与 Logit 蒸馏、序列级/on-policy 蒸馏、幅度剪枝与恢复；模 10 加法 3 种子、40 标注/100 提示对照，21 项测试 |
| 第 20 章 | 完成 | 7516 | 已在本机运行 | 贪心、束搜索、温度、Top-k/Top-p/Min-p、重复惩罚、约束解码与 pass@k；合成马尔可夫链 3 种子 7 策略对照，17 项测试 |
| 第 21 章 | 完成 | 10442 | 已在本机运行 | KV Cache 账本、roofline、FlashAttention 在线 Softmax、PagedAttention、连续批/分块预填充/PD 分离；缓存等价性与碎片/延迟账本，20 项测试 |
| 第 22 章 | 完成 | 7736 | 已在本机运行 | 拒绝采样无损性、接受率与加速比账本、树状草稿、Medusa/EAGLE、MTP、KV 回滚；20 万次分布核对与 10 万轮接受数模拟，14 项测试 |
| 第 23 章 | 完成 | 10283 | 已在本机运行 | 对称/非对称与三种粒度、SQNR、GPTQ/AWQ/SmoothQuant、KV 量化、INT4 打包、FP8/NF4/FP4；7B 带宽账本与 TinyLM 端到端权重仅量化，16 项测试 |
| 第 24 章 | 完成 | 8579 | 已在本机运行 | 覆盖率/pass@k、自洽性与加权投票、Best-of-N 与验证器、PUCT 树搜索、预算与提前停止；陷阱树、TinyLM 链式算术与停止阈值对照，18 项测试 |
| 第 25 章 | 完成 | 7698 | 已在本机运行 | LLaVA 式图文拼接、ViT 与三种投影、InfoNCE、VQ；合成视觉问答 3 种子、Token 预算与注意力账本，12 项测试 |
| 第 26 章 | 完成 | 7869 | 已在本机运行 | POMDP 智能体、工具模式与 JSON 动作协议、上下文二次成本、重试与任务成功率、信息价值；最小权限拦截提示注入，2 万次蒙特卡洛核对公式，15 项测试 |
| 第 27 章 | 完成 | 7670 | 已在本机运行 | 准确率/F1/pass@k、Wilson 区间、配对 bootstrap、McNemar、ECE、n-gram 污染、Elo/BT、kappa、成本效用；确定性配对比较与 2 万次 pass@k 模拟，12 项测试 |
| 第 28 章 | 完成 | 7198 | 已在本机运行 | 线性探针与选择性对照、注意力/消融冗余、激活修补位点、SAE 重构—稀疏与特征纯度；复制任务端到端与 15 项测试 |
| 第 29 章 | 完成 | 7692 | 已在本机运行 | 长程稀疏奖励与势函数塑形、EWC/回放/参数隔离、合成数据方差递推与真实数据锚定、架构账本、涌现与损失—能力分离；3 组独立数值实验 |

## 第 1 章交接

### 本章摘要

本章从自回归分解出发，说明语言模型是精确的链式分解；推导出交叉熵等于数据熵加 KL 散度，困惑度等于交叉熵的指数、可解释为有效分支因子，并给出每 Token 比特数的换算。比较 n-gram 的极大似然、加性平滑与指数级参数量，神经语言模型的参数共享与 RNN 的隐状态。系统梳理自回归、掩码、填充中间、去噪、多 Token 预测、强化学习六类训练目标及其取舍。最后用二阶马尔可夫链的可控实验验证：n-gram、神经 n-gram、RNN 的测试交叉熵分别收敛到真实的一阶、二阶、三阶条件熵。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (1.1) | $$p(x_{1:T})=\prod_{t=1}^{T}p(x_t \mid x_{<t})$$ | 自回归分解，精确 |
| (1.2) | $$\log p(x_{1:T})=\sum_t \log p(x_t \mid x_{<t})$$ | 对数似然 |
| (1.3) | $$\mathcal{L}=-\frac{1}{T}\sum_t \log p_\theta(x_t \mid x_{<t})$$ | 负对数似然/交叉熵 |
| (1.4) | $$H(p_{\text{data}},p_\theta)=H(p_{\text{data}})+\mathrm{KL}(p_{\text{data}}\Vert p_\theta)$$ | 交叉熵分解 |
| (1.5) | $$\mathrm{PPL}=\exp(\mathcal{L})$$ | 困惑度定义 |
| (1.6) | $$\mathrm{PPL}=\left(\prod_t 1/p_t\right)^{1/T}$$ | 几何平均形式 |
| (1.7) | $$\text{bits/token}=\mathcal{L}/\ln 2$$ | 压缩长度换算 |
| (1.8) | $$p_{\text{MLE}}(x_t\mid \text{ctx})=c(\text{ctx},x_t)/c(\text{ctx})$$ | n-gram 极大似然 |
| (1.9) | $$p_{\text{add-}k}=(c+k)/(c(\text{ctx})+k n_{vocab})$$ | 加性平滑 |
| (1.10) | $$\lvert\theta_{n\text{-gram}}\rvert=\mathcal{O}(n_{vocab}^{\,n})$$ | n-gram 参数量 |
| (1.11) | $$\mathbf{e}_t=\mathbf{E}z_t$$ | 词嵌入 |
| (1.12) | $$\mathbf{h}_t=\tanh(\mathbf{W}_x\mathbf{e}_t+\mathbf{W}_h\mathbf{h}_{t-1}+\mathbf{b})$$ | RNN 隐状态 |
| (1.13) | $$\boldsymbol{\ell}_t=\mathbf{W}_o\mathbf{h}_t$$ | 输出 logits |
| (1.14) | $$p_\theta=\mathrm{softmax}(\boldsymbol{\ell}_t)$$ | 条件分布 |
| (1.15) | $$\mathcal{O}(T^2d_{model}+Td_{model}^2)$$ | 单层 Transformer 计算量 |
| (1.16) | $$\mathcal{L}_{\text{AR}}=-\sum_t\log p_\theta(x_t\mid x_{<t})$$ | 自回归目标 |
| (1.17) | $$\mathcal{L}_{\text{MLM}}=-\sum_{t\in\mathcal{M}}\log p_\theta(x_t\mid x_{\bar{\mathcal{M}}})$$ | 掩码目标 |
| (1.18) | $$\mathcal{L}_{\text{FIM}}=-\sum_{t\in\text{middle}}\log p_\theta(x_t\mid \text{prefix},\text{suffix},x_{<t})$$ | 填充中间 |
| (1.19) | $$\mathcal{L}_{\text{denoise}}=-\sum_{\text{span}}\log p_\theta(\text{span}\mid\text{corrupted})$$ | 去噪目标 |
| (1.20) | $$\mathcal{L}_{\text{MTP}}=-\sum_t\sum_{i=1}^{m}\log p_\theta(x_{t+i}\mid x_{\le t})$$ | 多 Token 预测 |
| (1.21) | $$J(\theta)=\mathbb{E}_{\pi_\theta}[r(y)]-\beta\,\mathrm{KL}(\pi_\theta\Vert\pi_{\text{ref}})$$ | 强化学习目标 |

### 留给后续章节的伏笔

- **第 2 章**：分词决定序列长度 $$T$$，也决定困惑度的分母；PPL 不可跨分词器比较的结论会在这里落地。
- **第 3 章**：用自注意力替代 RNN，实现 (1.15) 所述的序列维并行；残差流概念首次出现。
- **第 4 章**：位置编码决定模型能否区分顺序，直接影响 (1.1) 中条件分布的质量。
- **第 9 章**：缩放定律研究的是 (1.3) 的损失如何随参数量与数据量下降。
- **第 14、18 章**：(1.16)–(1.21) 的训练目标会在 SFT、RLHF、RLVR 中具体化。
- **第 20 章**：教师强制与曝光偏差会在解码策略中展开。
- **第 22 章**：(1.20) 的 MTP 头可复用为投机解码的草稿。
- **第 24 章**：推理时扩展计算，本质是在 (1.1) 的采样过程中投入更多算力。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、对比、代码、工业实践、论文、误区、小结、习题齐全。
- [x] 推导无跳步：链式法则、交叉熵分解、n-gram 极大似然、均匀分布 PPL 均有完整步骤。
- [x] 符号与附录 B 一致：$$T$$、$$n_{vocab}$$、$$D$$、$$z_t$$ 的首次出现已改为第 1 章，新增 $$n$$、$$d$$、$$k$$、$$m$$、$$H(p)$$。
- [x] 代码可运行：`examples/ch01_lm_basics.py` 已在本机 Python 3.13 + PyTorch 2.9 上运行，输出为真实结果。
- [x] 公式与代码对应：代码注释标注 (1.3)、(1.8)、(1.9)、(1.11)–(1.14)。
- [x] GitBook 语法：使用 hint、tabs、details、mermaid；代码块均标注语言；全文未使用单美元数学定界符。
- [x] 篇幅：正文约 7100 个中文字符，符合 6000–9000 字预算。
- [x] 论文信息核对：5 篇论文的作者、年份、标题与 arXiv 编号均无误。

## 第 2 章交接

### 本章摘要

本章形式化分词器与无损可逆条件，推导 BPE 的合并准则、词表规模关系与贪心性，比较 WordPiece 的点互信息准则、Unigram 的 EM 训练与 Viterbi 编码，分析字节级 BPE 的覆盖界与词表大小在参数、序列长度之间的权衡；用字节级 BPE 实验验证往返一致性、中英文成本差异、字母与数字切分、词表边际收益，并说明 PPL 不能跨分词器比较。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (2.1) | $$\mathbf{e}_t=\mathbf{E}z_t$$ | Token ID 查嵌入表 |
| (2.2) | $$\mathrm{Enc}:\Sigma^{*}\to\mathcal{V}^{*},\ \mathrm{Dec}:\mathcal{V}^{*}\to\Sigma^{*}$$ | 编码器与解码器 |
| (2.3) | $$\mathrm{Dec}(\mathrm{Enc}(s))=s$$ | 无损可逆条件 |
| (2.4) | $$r=B_{\text{byte}}/T$$ | 压缩比，字节/Token |
| (2.5) | $$f_t(a,b)=\sum_i\mathbf{1}[q_i^{(t)}=a,q_{i+1}^{(t)}=b]$$ | BPE 相邻对频次 |
| (2.6) | $$\lvert\mathcal{V}\rvert=\lvert\mathcal{A}\rvert+M+S$$ | 词表规模关系 |
| (2.7) | $$T_M=T_0-\sum_{t=1}^{M}n_t,\ n_t\le f_t(a_t,b_t)$$ | BPE 贪心长度缩减 |
| (2.8) | $$\mathrm{score}(a,b)=f(ab)/(f(a)f(b))$$ | WordPiece 合并得分 |
| (2.9) | $$\mathrm{score}(a,b)=\frac{1}{N_{\text{pair}}}e^{\mathrm{PMI}(a,b)}$$ | 得分与 PMI 的关系 |
| (2.10) | $$P(u)=\prod_{i=1}^{K}p(u_i)$$ | Unigram 切分概率 |
| (2.11) | $$\mathrm{best}[i]=\max_{j<i}(\mathrm{best}[j]+\log p(s_{j:i}))$$ | Viterbi 编码 |
| (2.12) | $$1\le\lvert\mathrm{Enc}(c)\rvert\le n_{\text{byte}}(c)$$ | 字符的 Token 数界 |
| (2.13) | $$N(n_{vocab})=N_0+2n_{vocab}d_{model}$$ | 参数量与词表大小 |
| (2.14) | $$C(n_{vocab})\approx6N(n_{vocab})B_{\text{byte}}/r(n_{vocab})$$ | 计算量与词表大小 |
| (2.15) | $$\mathrm{fertility}=T/n_{\text{word}}$$ | 生育率 |
| (2.16) | $$\rho=\mathcal{L}/(\ln2\cdot r)$$ | 每字节比特数 |
| (2.17) | $$\mathrm{PPL}_1=\mathrm{PPL}_2^{\,r_1/r_2}$$ | 跨分词器的 PPL 关系 |

### 留给后续章节的伏笔

- **第 3 章**：嵌入矩阵形状为 $$n_{vocab}\times d_{model}$$，输出层 logits 维度为 $$n_{vocab}$$；分词决定序列长度 $$T$$，直接影响注意力成本。
- **第 4 章**：位置编码作用在 Token 位置上，分词边界改变位置总数与长上下文成本。
- **第 6 章**：KV Cache 大小与 $$T$$ 成正比；分词压缩率直接影响显存占用。
- **第 9 章**：缩放定律以 Token 数 $$D$$ 计量；换分词器会改变同一字节语料对应的 $$D$$。
- **第 10 章**：数据清洗与去重通常以 Token 计量，分词器差异会影响数据配比与预算。
- **第 21 章**：推理吞吐以 Token/s 计量；分词器决定同一文本的实际推理成本。
- **第 23 章**：大词表的嵌入层与输出层是量化和显存优化的重点。
- **第 27 章**：PPL、Token 成本与污染统计都依赖分词器，评测时必须固定分词器版本。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、对比、代码、工业实践、论文、误区、小结、习题齐全。
- [x] 推导无跳步：分词器形式化、BPE 合并、WordPiece 的 PMI 关系、Unigram 的 EM/Viterbi、字节上界、词表权衡、跨分词器 PPL 关系均有完整步骤。
- [x] 符号与附录 B 一致：词表大小统一用 $$n_{vocab}$$，词表集合用 $$\mathcal{V}$$，基础符号集用 $$\mathcal{A}$$，文本字节数用 $$B_{\text{byte}}$$，避免与 $$\mathbf{V}$$、$$B$$、$$C$$、$$W$$、$$z_t$$ 冲突。
- [x] 代码可运行：从 `part1/chapter02.md` 直接提取代码块，在本机 Python 3.13 上运行，输出与正文完全一致。
- [x] 公式与代码对应：代码注释标注 (2.5)、(2.6)、(2.7)。
- [x] GitBook 语法：使用 info/warning/danger/success 四类 hint、mermaid、details；代码块均标注语言；全文未使用单美元数学定界符。
- [x] 篇幅：正文约 8100 个中文字符，符合 6000–9000 字预算。
- [x] 论文信息核对：Gage 1994、Sennrich et al. 2016、Wu et al. 2016、Kudo 2018、Radford et al. 2019 的作者、年份、标题与编号均无误。
- [x] 实验诚实报告：目标词表 768 实际 770（特殊 Token）、目标 1024 实际 1006（合并提前停止）均已说明；小语料结果的局限性已标注。

## 第 3 章交接

### 本章摘要

本章拆解 Transformer 解码器：以宽度不变的残差流为主线，说明注意力负责 Token 间通信、前馈网络负责逐位置计算；推导 LayerNorm 与 Pre-LN/Post-LN 的取舍，缩放点积注意力的方差论证、因果掩码、多头拆分与拼接、前馈参数占比；证明无位置编码时的排列等变性，并给出可学习绝对位置；推导精确参数量公式与每层 FLOPs，指出注意力与前馈的交叉点 $$T=d_{ff}-2d$$。代码实现 TinyLM 基线，验证参数计数与公式严格一致、因果性无泄漏、注意力行和为 1，并用固定偏移复制任务验证损失收敛到可解析的理论下界。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (3.1) | $$\mu_t=\frac{1}{d}\sum_i x_{t,i},\quad \sigma_t^2=\frac{1}{d}\sum_i(x_{t,i}-\mu_t)^2$$ | 逐位置均值与方差 |
| (3.2) | $$\mathrm{LN}(\mathbf{x}_t)=\boldsymbol{\gamma}\odot\frac{\mathbf{x}_t-\mu_t}{\sqrt{\sigma_t^2+\epsilon}}+\boldsymbol{\beta}$$ | 层归一化 |
| (3.3) | $$\mathbf{X}\leftarrow\mathrm{LN}(\mathbf{X}+\mathrm{Sublayer}(\mathbf{X}))$$ | Post-LN |
| (3.4) | $$\mathbf{X}\leftarrow\mathbf{X}+\mathrm{Sublayer}(\mathrm{LN}(\mathbf{X}))$$ | Pre-LN |
| (3.5) | $$\mathbf{Q}=\mathbf{X}\mathbf{W}^{Q},\ \mathbf{K}=\mathbf{X}\mathbf{W}^{K},\ \mathbf{V}=\mathbf{X}\mathbf{W}^{V}$$ | Q/K/V 投影 |
| (3.6) | $$s_{ij}=\mathbf{q}_i^{\top}\mathbf{k}_j/\sqrt{d_k}$$ | 缩放点积打分 |
| (3.7) | $$\mathbf{q}^{\top}\mathbf{k}=\sum_{l=1}^{d_k}q_lk_l$$ | 内积展开 |
| (3.8) | $$\mathrm{Var}(\mathbf{q}^{\top}\mathbf{k})=d_k$$ | 方差为 $$d_k$$ |
| (3.9) | $$\mathrm{softmax}(\mathbf{s})_j=\frac{\exp(s_j)}{\sum_{j'}\exp(s_{j'})}$$ | Softmax |
| (3.10) | $$M_{ij}=0\ (j\le i),\ -\infty\ (j>i)$$ | 因果掩码 |
| (3.11) | $$\mathbf{A}=\mathrm{softmax}(\mathbf{S}+\mathbf{M})$$ | 加掩码后的注意力 |
| (3.12) | $$\mathbf{O}=\mathbf{A}\mathbf{V}$$ | 加权求和 |
| (3.13) | $$d_h=d_{model}/n_h,\quad \mathbf{Q}^{(h)}=\mathbf{Q}_{:,\,(h-1)d_h+1:hd_h}$$ | 头拆分 |
| (3.14) | $$\mathbf{O}=\left[\mathbf{O}^{(1)};\cdots;\mathbf{O}^{(n_h)}\right]\mathbf{W}^{O}$$ | 拼接与输出投影 |
| (3.15) | $$\mathrm{FFN}(\mathbf{x})=\mathbf{W}_{\text{out}}\phi(\mathbf{W}_{\text{in}}\mathbf{x}+\mathbf{b}_{\text{in}})+\mathbf{b}_{\text{out}}$$ | 前馈网络 |
| (3.16) | $$\mathbf{S}'=\pi\mathbf{S}\pi^{\top}$$ | 排列后的打分 |
| (3.17) | $$f(\pi\mathbf{X})=\pi f(\mathbf{X})$$ | 排列等变 |
| (3.18) | $$\mathbf{H}^{(0)}_t=\mathbf{E}z_t+\mathbf{P}_t$$ | Token 嵌入 + 位置嵌入 |
| (3.19) | $$\boldsymbol{\ell}_t=\mathbf{W}_{\text{lm}}\mathrm{LN}(\mathbf{h}_t^{(n_l)})$$ | 输出 logits |
| (3.20) | $$\mathcal{L}=-\frac{1}{T}\sum_t\log\mathrm{softmax}(\boldsymbol{\ell}_t)_{x_{t+1}}$$ | 交叉熵损失 |
| (3.21) | $$N_{\text{layer}}=4d^2+2dd_{ff}+9d+d_{ff}$$ | 每层参数量 |
| (3.22) | $$N=n_l(4d^2+2dd_{ff}+9d+d_{ff})+2d+2n_{vocab}d+L_{ctx}d$$ | 精确参数量 |
| (3.23) | $$N\approx12n_ld^2+2n_{vocab}d$$ | 简化参数量 |
| (3.24) | $$\mathrm{FLOPs}_{\text{layer}}\approx8Td^2+4T^2d+4Tdd_{ff}$$ | 单层前向 |
| (3.25) | $$T=d_{ff}-2d$$ | 注意力/前馈交叉点 |
| (3.26) | $$\mathrm{FLOPs}_{\text{forward}}\approx2N\ \text{（每 Token）}$$ | 前向算力心算 |

### 留给后续章节的伏笔

- **第 4 章**：可学习绝对位置不能外推；RoPE、ALiBi、位置插值将替换 (3.18)，并处理 $$L_{ctx}$$ 之外的位置。
- **第 5 章**：RMSNorm 替换 LayerNorm；SwiGLU 替换 GELU MLP，$$d_{ff}$$ 按三矩阵重新计算；偏置与权重绑定做消融。
- **第 6 章**：MHA 泛化为 MQA/GQA/MLA；KV Cache 大小由 $$n_{kv}d_h$$ 与层数决定。
- **第 7 章**：MoE 把前馈层替换为多个专家，参数量与激活量分离；本章的 FFN 是专家网络的基线。
- **第 9 章**：用 (3.26) 的 $$2N$$ 与训练约 $$6N$$ 建立缩放定律；参数量与数据量的配比。
- **第 11 章**：Pre-LN 的稳定性、warmup、梯度裁剪；残差缩放初始化与损失尖峰。
- **第 12 章**：张量并行按头或按特征切分，要求 $$d_{model}$$、$$d_{ff}$$ 与头数满足整除约束。
- **第 13 章**：长上下文下 (3.24) 的 $$T^2$$ 项与注意力显存成为瓶颈。
- **第 18 章**：`logits_to_keep` 在 GRPO rollout 中降低 logits 显存。
- **第 21 章**：FlashAttention 避免显式保存 $$\mathbf{A}$$；KV Cache 复用 $$\mathbf{K},\mathbf{V}$$。
- **第 23 章**：嵌入层与输出层是量化与显存优化的重点。
- **第 28 章**：残差流是探针、激活修补与回路分析的主要对象。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、对比、代码、工业实践、论文、误区、小结、习题齐全。
- [x] 推导无跳步：LayerNorm、缩放方差、因果掩码、多头等价代价、排列等变、参数量、FLOPs 交叉点均有完整步骤。
- [x] 符号与附录 B 一致：$$d_{model}$$、$$n_l$$、$$n_h$$、$$d_h$$、$$d_{ff}$$、$$d_k$$、$$\mathbf{S}/\mathbf{A}/\mathbf{O}$$、$$\mathbf{W}_{\text{lm}}/\mathbf{W}_{\text{in}}/\mathbf{W}_{\text{out}}$$、$$\mathbf{P}$$、上标 $$(h)$$ 已登记；区分 $$\mathbf{h}_t$$（RNN）与 $$\mathbf{h}_t^{(l)}$$（残差流）。
- [x] 代码可运行：从 `part1/chapter03.md` 直接提取代码块，在本机 Python 3.13 + PyTorch 2.9 CPU 上运行，输出与正文一致。
- [x] 测试可运行：同一 Markdown 提取的 `tests/test_model.py` 通过 6 项测试，覆盖形状、因果性、参数计数、反向传播。
- [x] 公式与代码对应：代码注释标注 (3.1)、(3.2)、(3.4)、(3.5)、(3.6)、(3.10)–(3.12)、(3.14)、(3.15)、(3.18)–(3.20)、(3.22)、(3.23)。
- [x] GitBook 语法：使用 info/warning/danger/success 四类 hint、1 个 mermaid、1 个 details、2 张表；代码块均标注语言；全文未使用单美元数学定界符。
- [x] 篇幅：正文约 10300 个中文字符，符合核心章 10000–12000 字预算。
- [x] 论文信息核对：Vaswani 2017、Radford 2019、Xiong 2020（arXiv:2002.04745）、Kaplan 2020、Geva 2021（arXiv:2012.14913）的作者、年份、标题与编号均无误。
- [x] 实验诚实报告：复制任务的理论损失下界 0.6261、准确率上界 0.7883 与实测值均已给出；明确标注这是 10.6 万参数的受控实验，不能外推到 LLM 能力。

## 第 4 章交接

### 本章摘要

本章说明纯自注意力对输入排列等变、位置信息必须显式注入。系统比较两条注入路径：加在嵌入上的可学习绝对位置与正弦编码，以及改在注意力打分里的相对偏置、RoPE 与 ALiBi，并把 NoPE 作为对照。推导正弦编码的平移即旋转性质与内积只依赖位置差，推导 RoPE 的旋转矩阵、保范数与相对位移性质，给出 ALiBi 斜率公式。把长度外推拆成"有定义"与"能泛化"两件事，推导有效上下文的定义，比较 PI、NTK-aware、YaRN 分别改变哪些频率。代码上给 TinyLM 增加 `pos_encoding` 分派与 `rope_scaling`，用训练长度 32、lag=8 的复制任务在 32–128 上评测：RoPE 几乎无损外推，ALiBi 随长度改善，可学习绝对位置与正弦编码明显退化，NoPE 未学会规则。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (4.1) | $$\mathrm{PE}(t,2i)=\sin(t/b^{2i/d_{model}}),\ \mathrm{PE}(t,2i+1)=\cos(\cdot)$$ | 正弦位置编码 |
| (4.2) | $$\omega_i=b^{-2i/d_{model}}$$ | 正弦编码角频率 |
| (4.3) | $$\mathbf{p}_{t+\Delta,i}=\mathbf{R}(\Delta\omega_i)\mathbf{p}_{t,i}$$ | 平移等于旋转 |
| (4.4) | $$\mathbf{p}_m^{\top}\mathbf{p}_n=\sum_i\cos((m-n)\omega_i)$$ | 正弦内积只依赖位置差 |
| (4.5) | $$s_{ij}=\mathbf{q}_i^{\top}\mathbf{k}_j/\sqrt{d_k}+b_{\mathrm{clip}(j-i)}$$ | T5 相对位置偏置 |
| (4.6) | $$\theta_i=b^{-2i/d_h}$$ | RoPE 频率 |
| (4.7) | $$\mathbf{q}_t'=\mathbf{R}_t\mathbf{q}_t,\ \mathbf{k}_t'=\mathbf{R}_t\mathbf{k}_t$$ | RoPE 旋转查询与键 |
| (4.8) | $$\mathbf{R}_{t,i}=\begin{bmatrix}\cos t\theta_i & -\sin t\theta_i\\ \sin t\theta_i & \cos t\theta_i\end{bmatrix}$$ | 二维分块旋转矩阵 |
| (4.9) | $$(\mathbf{R}_m\mathbf{q})^{\top}(\mathbf{R}_n\mathbf{k})=\mathbf{q}^{\top}\mathbf{R}_{n-m}\mathbf{k}$$ | RoPE 相对位移性质 |
| (4.10) | 分量展开式，仅含 $$(m-n)\theta_i$$ | 打分只依赖位置差 |
| (4.11) | $$s_{ij}=\mathbf{q}_i^{\top}\mathbf{k}_j/\sqrt{d_k}-\nu_h(i-j)$$ | ALiBi 打分 |
| (4.12) | $$\nu_h=2^{-8(h+1)/n_h}$$ | ALiBi 几何斜率 |
| (4.13) | $$L_{\mathrm{eff}}(\epsilon)=\max\{L:\mathrm{metric}(L)\ge\epsilon\}$$ | 有效上下文定义 |
| (4.14) | $$t'=t/s,\ s=L_{\mathrm{test}}/L_{\mathrm{train}}$$ | 位置插值 PI |
| (4.15) | $$b'=b\,s^{d_h/(d_h-2)}$$ | NTK-aware 基频缩放 |
| (4.16) | $$w_i=\mathrm{clamp}((\beta-r_i)/(\beta-\alpha),0,1)$$ | YaRN 分频权重 |
| (4.17) | $$\theta_i'=\theta_i[(1-w_i)+w_i/s]$$ | YaRN 分段频率缩放 |
| (4.18) | $$t_y=0.1\ln s+1$$ | YaRN 温度缩放 |

### 留给后续章节的伏笔

- **第 5 章**：RMSNorm 与 SwiGLU 替换第 3 章结构与本章无关，但本章的因果性测试基线继续沿用；RoPE 与归一化顺序的组合在长上下文下会再次讨论。
- **第 6 章**：MQA/GQA/MLA 共享 KV 头，缓存里保存的是已旋转的键；`position_ids` 的传递方式与本章一致。
- **第 13 章**：长上下文训练、RULER 类评测与 (4.13) 的有效上下文会展开；本章只给定义与初步对照。
- **第 21 章**：KV Cache 复用只缓存已旋转的键，新查询用绝对位置旋转；`position_ids` 是接口。
- **第 22 章**：投机解码与多 Token 预测在位置上的对齐依赖同一套 `position_ids` 语义。
- **第 23 章**：RoPE 频率表与缓存可预计算，量化时要注意 cos/sin 的数值精度。
- **第 26、27 章**：长上下文能力与训练数据、位置缩放共同决定；评测时必须区分标称窗口与有效上下文。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、对比、代码、工业实践、论文、误区、小结、"本章的取舍"、习题齐全。
- [x] 推导无跳步：正弦平移旋转、内积形式、RoPE 旋转复合与分量展开、ALiBi 斜率、NTK 指数分析、YaRN 分段权重均有完整步骤。
- [x] 符号与附录 B 一致：新增 $$p$$、$$\mathbf{p}_t$$、$$\theta_i$$、$$b$$、$$s_{\text{rope}}$$、$$\Delta$$、$$\mathbf{R}_{m,i}$$、$$\lambda_i$$、$$\nu_h$$、$$L_{\text{train}}$$、$$L_{\text{test}}$$、$$L_{\text{eff}}$$、$$t_y$$、$$w_i$$、$$r_i$$ 已登记。
- [x] 代码可运行：从 `part1/chapter04.md` 直接提取代码块，在本机 Python 3.13 + PyTorch 2.9 CPU 上运行，输出与正文逐字一致。
- [x] 测试可运行：同一 Markdown 提取的 `tests/test_position.py` 通过 9 项测试，覆盖正弦平移、RoPE 相对性与保范数、缩放形状、ALiBi 斜率与偏置、超表长前向、可学习表越界、因果性。
- [x] 公式与代码对应：代码注释标注 (4.1)、(4.2)、(4.6)、(4.7)、(4.11)、(4.12)、(4.14)–(4.18)。
- [x] GitBook 语法：hint 仅用 `warning` 与 `success`，1 个 mermaid、1 个 details、2 张对比表；代码块均标注语言；数学全部用双美元符号。
- [x] 篇幅：正文 7579 个中文字符，符合 6000–9000 字预算。
- [x] 论文信息核对：Vaswani 2017（arXiv:1706.03762）、Su 2021（arXiv:2104.09864）、Press 2021（arXiv:2108.12409）、Chen 2023（arXiv:2306.15595）、Peng 2023（arXiv:2309.00071）的作者、年份、标题与编号均无误。
- [x] 实验诚实报告：明确标注这是 2 层、64 维、400 步的受控实验，不能外推到 LLM 长上下文能力；正弦编码"有定义但不保证外推"的反直觉结果已单独解释。

## 第 5 章交接

### 本章摘要

本章把现代 LLM 的架构演进拆成五个局部零件：RMSNorm 替换 LayerNorm、SwiGLU 替换 GELU MLP、去掉线性层偏置、选择是否绑定输入输出嵌入、块内串行或并行。推导 RMSNorm 的尺度不变性与零均值下与 LayerNorm 的等价条件；推导 SwiGLU 的等参数预算点 `d_ff=8d_model/3`，说明此时参数与 FLOPs 都与 `4d` MLP 相同，额外成本是约 4/3 的中间激活；给出无偏置开销 (5.9)、并行块定义 (5.10)、精确参数量 (5.11) 与每 Token FLOPs (5.12)。代码上给 TinyLM 增加 `RMSNorm`、`SwiGLU`、`bias`、`tie_embeddings`、`parallel_block` 与两个账本函数；实验包括四种配置的逐位参数核对、Llama 2 7B 的 6,738,415,616 精确复现、Llama 2→3 的 1.292B 增量拆解、RMSNorm 行为演示，以及 3 个种子的复制任务消融。消融结果四种配置都贴着理论下界，差异在一个标准差内；本章明确说明该实验验证实现与预算等价，不证明架构优劣。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (5.1) | $$\mathrm{RMS}(\mathbf{x}_t)=\sqrt{\frac{1}{d_{model}}\sum_i x_{t,i}^2}$$ | 均方根定义 |
| (5.2) | $$\mathrm{RMSNorm}(\mathbf{x}_t)=\boldsymbol{\gamma}\odot\mathbf{x}_t/\sqrt{\mathrm{RMS}^2+\epsilon}$$ | RMSNorm |
| (5.3) | $$\mu_t=0,\ \boldsymbol{\beta}=0\ \Longrightarrow\ \mathrm{RMSNorm}=\mathrm{LN}$$ | 零均值下的等价条件 |
| (5.4) | $$\mathrm{RMSNorm}(\alpha\mathbf{x}_t)=\mathrm{RMSNorm}(\mathbf{x}_t)$$ | 尺度不变性 |
| (5.5) | $$\mathrm{GLU}(\mathbf{x})=(\mathbf{W}_1\mathbf{x})\odot\sigma(\mathbf{W}_2\mathbf{x})$$ | 门控线性单元 |
| (5.6) | $$\mathrm{SiLU}(z)=z\sigma(z)$$ | SiLU/Swish |
| (5.7) | $$\mathrm{SwiGLU}(\mathbf{x})=\mathbf{W}_3(\mathrm{SiLU}(\mathbf{W}_1\mathbf{x})\odot\mathbf{W}_2\mathbf{x})$$ | SwiGLU |
| (5.8) | $$3d_{model}d_{ff}=8d_{model}^2\ \Longrightarrow\ d_{ff}=8d_{model}/3$$ | 等参数预算点 |
| (5.9) | $$4d_{model}+(2d_{ff}+d_{model})=5d_{model}+2d_{ff}$$ | 每层去掉的偏置数（SwiGLU + RMSNorm） |
| (5.10) | $$\mathbf{X}\leftarrow\mathbf{X}+\mathrm{Attn}(\mathrm{LN}(\mathbf{X}))+\mathrm{MLP}(\mathrm{LN}(\mathbf{X}))$$ | 并行块 |
| (5.11) | $$N=n_l(4d^2+3dd_{ff}+2d)+d+2n_{vocab}d$$ | 现代配置精确参数量（不绑定） |
| (5.12) | $$\mathrm{FLOPs}\approx8Td^2+6Tdd_{ff}+4T^2d$$ | 每 Token 前向 FLOPs |

### 实验与关键数字

- 小模型账本：基线与现代配置参数差 10,496（0.28%），FLOPs 差 2,048（0.03%，来自 `d_ff` 取整）；绑定省 262,144；并行层省 1,024。
- Llama 2 7B：(5.11) 得 6,738,415,616，与公开数字逐位一致；前馈占非嵌入参数 66.8%。
- Llama 2→3 8B：词表 +788,529,152；前馈 +1,308,622,848；GQA −805,306,368；净增 1,291,845,632。
- RMSNorm 演示：放大 10 倍差异 1.93e-05；平移 +5 差异 3.05（LayerNorm 为 7.15e-07）；零均值下差异 2.38e-07。
- 复制任务：理论损失下界 0.6261、准确率上界 0.7883；四种配置损失 0.6310–0.6325，准确率 0.7874–0.7894，差异均在一个标准差内。

### 留给后续章节的伏笔

- **第 6 章**：GQA/MQA/MLA 的 KV 头共享是本章参数增量中的 −0.805B 项；`n_kv_heads` 与 `d_kv` 将在那里正式定义并接入 KV Cache。
- **第 7 章**：前馈占非嵌入参数 66.8%，是 MoE 用稀疏专家替换稠密前馈的直接动机。
- **第 9 章**：本章账本给出参数与 FLOPs 的口径，缩放定律会把它与损失曲线相连。
- **第 12 章**：去偏置与归一化选择属于训练稳定性工程，会把本章的"稳定性理由"落到优化器与数据配置。
- **第 21 章**：并行层的收益来自通信合并，KV Cache 与推理吞吐会把"FLOPs 相等不等于速度相等"展开。
- **第 23 章**：RMSNorm 的平方和溢出与激活范围会再次出现在量化与低精度训练中。
- **第 26、27 章**：长上下文能力由位置编码、注意力和数据共同决定，本章的架构替换只是其中一项。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、对比、代码、工业实践、论文、误区、小结、"本章的取舍"、习题齐全。
- [x] 推导无跳步：RMSNorm 尺度不变性与零均值等价、SwiGLU 等参数与等 FLOPs、去偏置计数、精确参数式均有完整步骤。
- [x] 符号与附录 B 一致：新增 $$\mathrm{RMS}$$、$$\mathrm{RMSNorm}$$、$$\mathbf{W}_1/\mathbf{W}_2/\mathbf{W}_3$$、$$\mathrm{SiLU}$$、$$\sigma$$、$$N_{\text{modern}}$$、$$n_{kv}/d_{kv}$$ 已登记；$$\sigma(\cdot)$$（Sigmoid）与第 3 章 $$\sigma_t$$（标准差）已在易混点中区分。
- [x] 代码可运行：从 `part2/chapter05.md` 直接提取代码块，在本机 Python 3.13 + PyTorch 2.9 CPU 上运行，输出与正文逐字一致。
- [x] 测试可运行：同一 Markdown 提取的 `tests/test_modern.py` 通过 15 项测试，覆盖 RMSNorm 公式/尺度不变/零均值等价/平移敏感、SwiGLU、权重绑定、6 种配置的精确参数、等预算 FLOPs、并行层定义与现代配置因果性。
- [x] 公式与代码对应：代码注释标注 (5.1)、(5.2)、(5.7)、(5.8)、(5.10)–(5.12)。
- [x] GitBook 语法：hint 使用 warning/danger/info/success 四类，1 个 mermaid、1 个 details、4 张表；代码块均标注语言；全文无单美元数学定界符。
- [x] 篇幅：正文 7526 个中文字符，在其余章 6000–9000 字预算内，落在 7500–8500 目标区间。
- [x] 论文信息核对：Zhang & Sennrich 2019（arXiv:1910.07467）、Shazeer 2020（arXiv:2002.05202）、Chowdhery 2022（arXiv:2204.02311）、Press & Wolf 2017（arXiv:1608.05859）、Dubey 2024（arXiv:2407.21783）的作者、年份、标题与编号均无误。
- [x] 事实性数字可复算：Llama 2 7B 参数量由 (5.11) 逐位复现；Llama 3 增量由词表、前馈、GQA 三项分解；PaLM 540B 的 118 层、18432 维、48 头、73728 前馈、256000 词表、并行层、MQA 与权重绑定已核对。
- [x] 实验诚实报告：明确标注 10 万参数、2 层、64 维、300 步的玩具实验不能外推；四种配置差异在一个标准差内，结论限定为"实现正确、预算对齐、无回归"，不宣称架构优劣。

## 第 6 章交接

### 本章摘要

本章从解码阶段的访存瓶颈出发，说明 MHA 的 KV Cache 随上下文线性增长，且在长上下文、大批量下超过权重读取成为每步延迟的主项。把 MQA（$$n_{kv}=1$$）与 GQA（$$1<n_{kv}<n_h$$）统一为"减少 KV 头数、保持单头维度 $$d_h$$ 不变"的共享方案，推导参数账本 (6.5) 与缓存账本 (6.6)，并用 (6.7) 的访存模型说明变体省的是带宽而不是注意力 FLOPs。MLA 换一条路：把 K/V 压缩到 $$d_c$$ 维潜向量，用解耦 RoPE 把位置信息放进独立的 $$d_r$$ 维通路，缓存降到 (6.11)；再推导权重吸收恒等式 (6.12)–(6.13)，说明它把解压推迟到推理时并改变张量并行切分。最后给出参数量 (6.14) 与缓存比 (6.15) 的闭式，用代码逐位核对四种变体的账本，在复制任务上做 3 种子消融，并诚实说明玩具任务分辨不出质量差异。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (6.1) | $$\mathbf{q}_t^{(i)}=\mathbf{W}_Q^{(i)}\mathbf{x}_t,\ \mathbf{k}_t^{(i)}=\mathbf{W}_K^{(i)}\mathbf{x}_t,\ \mathbf{v}_t^{(i)}=\mathbf{W}_V^{(i)}\mathbf{x}_t$$ | MHA 单头投影 |
| (6.2) | $$g(i)=\lfloor i\,n_{kv}/n_h\rfloor,\ \mathbf{k}_t^{(i)}=\mathbf{k}_t^{\mathrm{kv},(g(i))}$$ | GQA 分组映射 |
| (6.3) | $$\mathbf{o}_t=\mathbf{W}_O[\sum_{j\le t}\mathrm{softmax}_j(\cdot)\mathbf{v}_j^{(g(i))}]_{i=1}^{n_h}$$ | GQA 输出 |
| (6.4) | $$d_{kv}=d_{model}n_{kv}/n_h$$ | KV 投影总宽度 |
| (6.5) | $$N_{attn}=2d_{model}^2+2d_{model}d_{kv}$$ | 注意力参数量 |
| (6.6) | $$M_{kv}=2bTLn_{kv}d_hb_{\mathrm{elem}}=2bTLd_{kv}b_{\mathrm{elem}}$$ | KV Cache 字节数 |
| (6.7) | $$t_{\mathrm{dec}}\approx(2Nb_{\mathrm{elem}}+2bLTd_{kv}b_{\mathrm{elem}})/\mathrm{BW}$$ | 解码步延迟的访存模型 |
| (6.8) | $$\mathbf{c}_t=\mathbf{W}_{DKV}\mathbf{x}_t,\ \mathbf{c}_t\in\mathbb{R}^{d_c}$$ | MLA 下投影 |
| (6.9) | $$\mathbf{k}_t^{(i)}=\mathbf{W}_{UK}^{(i)}\mathbf{c}_t,\ \mathbf{v}_t^{(i)}=\mathbf{W}_{UV}^{(i)}\mathbf{c}_t$$ | MLA 上投影 |
| (6.10) | $$s_{tj}^{(i)}=\mathbf{q}_t^{(i)\top}\mathbf{k}_j^{(i)}/\sqrt{d_h}+\mathbf{q}_{r,t}^{(i)\top}\mathbf{k}_{r,j}/\sqrt{d_r}$$ | 解耦 RoPE 打分 |
| (6.11) | $$M_{MLA}=bTL(d_c+d_r)b_{\mathrm{elem}}$$ | MLA 缓存字节数 |
| (6.12) | $$\mathbf{q}_t^{(i)\top}\mathbf{k}_j^{(i)}=(\mathbf{W}_{UK}^{(i)\top}\mathbf{W}_Q^{(i)}\mathbf{x}_t)^\top\mathbf{c}_j$$ | 查询吸收恒等式 |
| (6.13) | $$\mathbf{W}_{OV}^{(i)}=\mathbf{W}_O^{(i)}\mathbf{W}_{UV}^{(i)}$$ | 输出吸收与 $$\mathbf{W}_{OV}$$ |
| (6.14) | $$N_{MLA}=2d_{model}^2+3d_{model}d_c+d_{model}d_r(n_h+1)$$ | MLA 注意力参数量 |
| (6.15) | $$M_{MLA}/M_{MHA}=(d_c+d_r)/(2d_{model})$$ | 缓存比闭式 |

### 实验与关键数字

- 账本（$$d=4096$$、$$L=32$$、$$n_h=32$$、$$T=8192$$、FP16）：注意力参数 MHA 67,108,864、GQA-8 41,943,040、MQA 34,603,008、MLA 48,496,640；每 Token 缓存 512 / 128 / 16 / 36 KiB；8K 总缓存 4 GiB / 1 GiB / 128 MiB / 288 MiB。
- 工业配置：Llama 2 7B MHA 512 KiB/Token（4K 总 2 GiB）；Llama 3 8B GQA-8 128 KiB（8K 总 1 GiB）；PaLM 540B MQA 177 KiB（2K 总 354 MiB）；DeepSeek-V2 MLA 67.5 KiB（8K 总 540 MiB）。
- 权重吸收：内容通路吸收前后最大输出误差 1.49e-07。
- 复制任务消融（3 种子、600 步）：理论损失下界 0.6261、准确率参考值 0.7883；MHA 0.6384/0.7887、GQA-2 0.6616/0.7837、MQA 0.6478/0.7861、MLA 0.6481/0.7878，差异在一个标准差附近，任务天花板掩盖质量差异。
- 测试：21 项全部通过（1.08 s），覆盖 MHA 退化一致性、GQA/MQA 形状、参数与缓存公式、四种变体因果性、整除约束、吸收恒等式、缓存排序与 TinyLM 前向。

### 留给后续章节的伏笔

- **第 7 章**：MLA 的潜向量压缩与 MoE 的稀疏激活都在"用更少字节/算力承载更多容量"这条主线上；本章的账本口径可复用到专家路由的显存估算。
- **第 8 章**：线性注意力与 SSM 用固定大小的状态替代随 $$T$$ 增长的 KV Cache，本章 (6.6)、(6.11) 是它们的对照基线。
- **第 12 章**：GQA 按 KV 头切分、MLA 吸收后按潜维度切分，张量并行的通信模式不同；本章已指出这一点。
- **第 13 章**：长上下文下 (6.6) 的缓存项超过权重项，是位置外推与上下文扩展必须同时考虑显存的原因。
- **第 21 章**：KV Cache 的 `past_kv`、分页与前缀缓存会把本章的账本落到具体数据结构；(6.7) 的访存模型是吞吐估算的起点。
- **第 22 章**：投机解码一次验证多个 Token，缓存读写次数与本章的每 Token 字节数直接相关。
- **第 23 章**：FP8/INT8 量化把 (6.6)、(6.11) 中的 $$b_{\mathrm{elem}}$$ 减半；吸收矩阵的条件数与量化校准是新增风险点。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、对比、代码、工业实践、论文、误区、小结、"本章的取舍"、习题齐全。
- [x] 推导无跳步：MHA 复杂度、GQA 分组与输出、参数与缓存账本、访存延迟模型、MLA 下/上投影、解耦 RoPE、权重吸收、参数汇总与缓存比均有完整步骤。
- [x] 符号与附录 B 一致：新增 $$n_{kv}$$、$$d_{kv}$$、$$d_c$$、$$d_r$$、$$\mathbf{c}_t$$、$$\mathbf{W}_{DKV}$$、$$\mathbf{W}_{UK}$$、$$\mathbf{W}_{UV}$$、$$\mathbf{W}_{QR}$$、$$\mathbf{W}_{KR}$$、$$\mathbf{W}_{OV}$$、$$b_{\mathrm{elem}}$$、$$N_{attn}$$、$$N_{MLA}$$、$$M_{kv}$$、$$M_{MLA}$$、$$t_{\mathrm{dec}}$$、$$F_{\mathrm{prefill}}$$、$$F_{\mathrm{decode}}$$ 已登记；单头维度统一写 $$d_h$$，避免与第 4 章的 $$d_k$$ 混淆。
- [x] 代码可运行：从 `part2/chapter06.md` 直接提取代码块，在本机 Python 3.13 + PyTorch 2.9 CPU 上运行，输出与正文逐字一致。
- [x] 测试可运行：同一 Markdown 提取的 `tests/test_attention_variants.py` 通过 21 项测试，覆盖变体形状、MHA 退化一致性、参数/缓存公式、因果性、整除约束、吸收恒等式与 TinyLM 前向。
- [x] 公式与代码对应：代码注释标注 (6.2)、(6.4)–(6.6)、(6.8)–(6.15)。
- [x] GitBook 语法：hint 使用 info/warning/success 三类，1 个 mermaid、1 个 details、6 张表；代码块均标注语言；数学全部用双美元符号。
- [x] 篇幅：正文 10516 个中文字符，符合核心章 10000–12000 字预算。
- [x] 论文信息核对：Shazeer 2019（arXiv:1911.02150）、Ainslie 2023（arXiv:2305.13245）、DeepSeek-AI 2024（arXiv:2405.04434）、Su 2021（arXiv:2104.09864）、Pope 2022（arXiv:2211.05102）的作者、年份、标题与编号均无误。
- [x] 事实性数字可复算：Llama 2 7B、Llama 3 8B、PaLM 540B、DeepSeek-V2 的层数、宽度、头数、$$d_c$$、$$d_r$$ 与缓存字节数均由 (6.6)、(6.11) 复算；GQA 表 1 的指标已核对为"每样本每 TPUv4 芯片的推理时间"。
- [x] 实验诚实报告：明确标注 10 万参数、2 层、64 维、600 步的玩具实验不能外推；复制任务损失与准确率贴着理论下界，结论限定为"实现正确、账本对齐"，不宣称变体优劣。

## 第 7 章交接

### 本章摘要

本章从"容量的下一个来源"出发，把稠密前馈替换为稀疏激活的混合专家：路由器为每个 Token 打分并选出 Top-K 专家，输出是选中专家的门控加权和。推导了路由器与截断归一化、容量因子与 Token 丢弃、总参数与激活参数账本、负载均衡辅助损失及其梯度、z-loss、共享专家与细粒度专家、Expert Choice 与无辅助损失偏置反馈；补上了反向传播的门控/专家梯度与专家并行的通信账本。代码给 TinyLM 增加 `MoEConfig` 与 `MoE` 层，用逐专家掩码循环实现分派，`aux_loss` 只在训练时返回；实验包括 Mixtral 与 DeepSeek-V3 的参数账本复现、容量收紧演示、3 种子复制任务消融，测试 16 项全部通过。本章明确说明：玩具任务贴着天花板，实验验证的是实现与账本，不是 MoE 的质量优势。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (7.1) | $$\mathbf{y}_t=\sum_{e=1}^{E}g_{t,e}\mathrm{FFN}_e(\mathbf{x}_t)$$ | 稀疏加权输出 |
| (7.2) | $$\mathbf{u}_t=\mathbf{W}_r\mathbf{x}_t$$ | 路由 logits |
| (7.3) | $$\mathbf{p}_t=\mathrm{softmax}(\mathbf{u}_t)$$ | 路由概率 |
| (7.4) | $$\mathcal{T}_t=\mathrm{TopK}(\mathbf{p}_t,k)$$ | Top-K 选择 |
| (7.5) | $$\tilde{g}_{t,e}=p_{t,e}/\sum_{e'\in\mathcal{T}_t}p_{t,e'}$$ | 截断归一化 |
| (7.6) | $$\mathbf{y}_t=\sum_{e\in\mathcal{T}_t}\tilde{g}_{t,e}\mathrm{FFN}_e(\mathbf{x}_t)$$ | 最终输出 |
| (7.7) | $$C_{\text{cap}}=\lceil \mathrm{CF}\cdot N_{\text{tok}}k/E\rceil$$ | 每专家容量 |
| (7.8) | $$a_{t,e}=\mathbf{1}[\cdot]$$ | 分派指示变量 |
| (7.9) | $$\mathbf{y}_t=\sum_e a_{t,e}\tilde{g}_{t,e}\mathrm{FFN}_e(\mathbf{x}_t)$$ | 含丢弃的输出 |
| (7.10) | $$P_e=3d_{model}d_{ff}^{e}$$ | 单专家参数（SwiGLU） |
| (7.11) | $$P_{\mathrm{MoE}}=d_{model}E+(E+S_{\text{sh}})P_e$$ | 单层 MoE 参数 |
| (7.12) | $$N=N_0+n_lP_{\mathrm{MoE}}$$ | 总参数 |
| (7.13) | $$N_{\mathrm{act}}=N_0+n_l(d_{model}E+(k+S_{\text{sh}})P_e)$$ | 每 Token 激活参数 |
| (7.14) | $$\rho=N_{\mathrm{act}}/N$$ | 激活比 |
| (7.15) | $$\mathrm{FLOPs}\approx2N_{\mathrm{act}}$$ | 每 Token 前向 FLOPs |
| (7.16) | $$f_e=\frac{1}{N_{\text{tok}}k}\sum_t\sum_j\mathbf{1}[e\in\mathcal{T}_t]$$ | 专家实际份额 |
| (7.17) | $$P_e=\frac{1}{N_{\text{tok}}}\sum_t p_{t,e}$$ | 平均路由概率 |
| (7.18) | $$\mathcal{L}_{\text{aux}}=\alpha E\sum_e f_eP_e$$ | 负载均衡辅助损失 |
| (7.19) | $$\mathcal{L}_z=\frac{1}{N_{\text{tok}}}\sum_t(\log\sum_e e^{u_{t,e}})^2$$ | 路由器 z-loss |
| (7.20) | $$\mathcal{L}=\mathcal{L}_{\text{CE}}+\mathcal{L}_{\text{aux}}+\beta\mathcal{L}_z$$ | 总训练目标 |
| (7.21) | $$\mathbf{y}_t=\sum_{s=1}^{S_{\text{sh}}}\mathrm{FFN}_s(\mathbf{x}_t)+\sum_{e\in\mathcal{T}_t}\tilde{g}_{t,e}\mathrm{FFN}_e(\mathbf{x}_t)$$ | 共享专家 |
| (7.22) | $$k d_{ff}^{e}=k' d_{ff}/m=k d_{ff}$$ | 细粒度等计算条件 |
| (7.23) | $$\mathcal{S}_e=\mathrm{TopC}(\{u_{t,e}\}),\ \lvert\mathcal{S}_e\rvert=C_{\text{cap}}$$ | Expert Choice |
| (7.24) | $$b_e\leftarrow b_e+\gamma\,\mathrm{sign}(\mathrm{load}_e-N_{\text{tok}}k/E)$$ | 无辅助损失均衡 |
| (7.25) | $$\partial\tilde{g}_{t,e}/\partial p_{t,j}=\frac{1}{Z_t}(\delta_{ej}-\tilde{g}_{t,e})$$ | 门控对概率的偏导 |
| (7.26) | $$\partial\mathcal{L}/\partial p_{t,j}=\frac{1}{Z_t}(c_{t,j}-\sum_{e\in\mathcal{T}_t}\tilde{g}_{t,e}c_{t,e})$$ | 门控梯度 |
| (7.27) | $$\partial\mathcal{L}/\partial\mathbf{W}_e=\sum_{t:e\in\mathcal{T}_t}g_{t,e}\,\partial\mathcal{L}/\partial\mathbf{y}_t\,\partial\mathrm{FFN}_e/\partial\mathbf{W}_e$$ | 专家梯度 |
| (7.28) | $$V_{\text{comm}}=2kd_{model}b$$ | 每 Token 每层通信字节 |

### 实验与关键数字

- 账本（FP16 权重）：Mixtral 8x7B 总参数 46,702,792,704、激活 12,879,925,248（27.6%）、权重 87.0 GiB、每 Token 25.8 GFLOP；DeepSeek-V3 全 MoE 计 702,934,184,960、激活 36,694,160,384；扣除前 3 层稠密前馈后总参数 670,162,777,088；TinyLM MoE 141,861,376 / 66,363,904。
- 稠密对照：Mixtral 尺寸稠密前馈 7,241,732,096，MoE 用 1.78 倍激活参数承载 6.45 倍容量。
- 容量演示：CF 1.25→0.5、容量 32/专家，两层丢弃 553/1024 个分派，logits 最大变化 0.032。
- 复制任务消融（3 种子、600 步）：理论损失下界 0.6261、准确率参考值 0.7883；Dense 0.6313/0.7862、MoE-aux0 0.6323/0.7877、MoE-aux 0.6362/0.7876；负载熵 1.82→2.05（均匀值 2.079），最大专家份额 0.22→0.16。
- 测试：16 项全部通过，覆盖 moe=None 与第 6 章逐位一致、参数公式、门控行和、同权专家退化、容量上限、无容量不丢弃、辅助损失公式、训练/评估行为、因果性与 Top-K 越界。

### 留给后续章节的伏笔

- **第 8 章**：MoE 用"更多参数、固定计算"扩大容量；线性注意力与 SSM 则试图降低序列维计算。两者都在回答"Transformer 的哪个瓶颈可以先动"。
- **第 9 章**：总参数与激活参数的口径会影响缩放定律的横轴；MoE 的损失-计算曲线与稠密模型不可直接混用。
- **第 10 章**：专家分工依赖数据分布与配比；数据课程会影响路由是否塌缩。
- **第 12 章**：专家并行、All-to-All、ZeRO/FSDP 与容量补齐的显存账本会在分布式训练中展开；(7.28) 是通信估算的起点。
- **第 14、18 章**：SFT 与 RLVR 阶段仍要维护路由均衡；微调时专家偏置与辅助损失是否继续更新，是工程决策点。
- **第 21 章**：连续批处理对 MoE 尤其重要，因为小批量解码的 All-to-All 延迟无法被计算掩盖。
- **第 23 章**：MoE 的权重显存是量化的主要目标；专家粒度为逐专家量化提供天然的分组。
- **第 24 章**：Expert Choice 与非因果路由在测试时计算扩展中可能重新有用武之地。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、对比、代码、工业实践、论文、误区、小结、"本章的取舍"、习题齐全。
- [x] 推导无跳步：路由与归一化、容量与丢弃、参数/激活账本、辅助损失及其梯度、z-loss、共享/细粒度专家、Expert Choice、偏置反馈、门控与专家反向梯度、通信账本均有完整步骤。
- [x] 符号与附录 B 一致：新增 $$E$$、$$k$$、$$S_{\text{sh}}$$、$$d_{ff}^{e}$$、$$\mathbf{W}_r$$、$$\mathbf{u}_t$$、$$\mathbf{p}_t$$、$$\mathcal{T}_t$$、$$\tilde{g}_{t,e}$$、$$Z_t$$、$$C_{\text{cap}}$$、$$\mathrm{CF}$$、$$a_{t,e}$$、$$f_e$$、$$P_e$$、$$P_{\mathrm{MoE}}$$、$$N_{\text{tok}}$$、$$N_{\mathrm{act}}$$、$$\rho$$、$$\mathcal{L}_{\text{aux}}$$、$$\mathcal{L}_z$$、$$b_e$$、$$\gamma$$、$$G$$、$$V_{\text{comm}}$$ 已登记；容量用 $$C_{\text{cap}}$$ 避开第 9 章的算力 $$C$$，共享专家数用 $$S_{\text{sh}}$$ 避开第 2 章的特殊 Token 数 $$S$$，组数上限用 $$G$$ 避开第 2 章的 BPE 合并次数 $$M$$，批次 Token 数用 $$N_{\text{tok}}$$ 避开模型总参数 $$N$$。
- [x] 代码可运行：从 `part2/chapter07.md` 直接提取代码块，在本机 Python 3.13 + PyTorch 2.9 CPU 上运行，主程序输出与正文逐字一致。
- [x] 测试可运行：同一 Markdown 提取的 `tests/test_moe.py` 通过 16 项测试（约 1 s），覆盖稠密退化、参数公式、路由、容量、辅助损失、因果性。
- [x] 公式与代码对应：代码注释标注 (7.2)–(7.8)、(7.10)、(7.16)–(7.18)、(7.21)。
- [x] GitBook 语法：hint 使用 info/warning/success 三类，1 个 mermaid、1 个 details、3 张对比表；代码块均标注语言；数学全部用双美元符号。
- [x] 篇幅：正文 10,543 个中文字符，符合核心章 10,000–12,000 字预算。
- [x] 论文信息核对：Shazeer 2017（arXiv:1701.06538）、Lepikhin 2020（arXiv:2006.16668）、Fedus 2021（arXiv:2101.03961）、Zoph 2022（arXiv:2202.08906）、Dai 2024（arXiv:2401.06066）的作者、年份、标题与编号均无误。
- [x] 事实性数字可复算：Mixtral 46,702,792,704 / 12,879,925,248 由 (7.11)–(7.13) 逐位复现；DeepSeek-V3 全 MoE 702.9B 扣除前 3 层稠密后 670.16B；参数消融 135,104 → 435,008 与激活 136,256 均手算复核；2025 年 Qwen3 与 Kimi K2 配置标注为"截至写作时"并注明可能随版本微调。
- [x] 实验诚实报告：明确标注 2 层、64 维、600 步的玩具任务贴着天花板，三组损失/准确率差异不足以证明 MoE 质量优势；容量演示只验证丢弃机制，不说明质量影响；辅助损失提高负载熵但语言建模损失差异在标准差内，未宣称因果退化。

## 第 8 章交接

### 本章摘要

本章从长上下文的 KV Cache 瓶颈出发，系统梳理 Transformer 之外的三条路线：线性注意力、状态空间模型与 RWKV，以及把二者组合起来的混合架构。推导了线性注意力的核替换、矩阵状态与前缀和训练；从连续状态空间方程与零阶保持离散化出发，得到 SSM 的递推、卷积与 FFT 训练；解释了选择性 SSM 为什么打破时不变性、必须改用并行扫描；给出了 RWKV 的指数衰减递推、混合架构的层模式与状态/算力账本；最后讨论固定状态的表达上界与复制任务证据。代码给 TinyLM 增加 `block_type` 分派与 `tinylm/ssm.py`，默认 `attention` 与第 7 章逐位一致；账本覆盖注意力、线性注意力、纯 SSM 与 1:3 混合四种配置；受控实验在复制任务上复现了"纯 SSM 会失败、混合可恢复"的方向。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (8.1) | $$\mathbf{y}_t=\sum_{j\le t}\frac{\exp(\mathbf{q}_t^\top\mathbf{k}_j/\sqrt{d})}{\sum_{j'\le t}\exp(\mathbf{q}_t^\top\mathbf{k}_{j'}/\sqrt{d})}\mathbf{v}_j$$ | 标准因果注意力 |
| (8.2) | $$\exp(\mathbf{q}^\top\mathbf{k})\approx\phi(\mathbf{q})^\top\phi(\mathbf{k})$$ | 核函数替换 |
| (8.3) | $$\mathbf{y}_t=\frac{\sum_{j\le t}(\phi(\mathbf{q}_t)^\top\phi(\mathbf{k}_j))\mathbf{v}_j}{\sum_{j\le t}\phi(\mathbf{q}_t)^\top\phi(\mathbf{k}_j)}$$ | 核化注意力 |
| (8.4) | $$\mathbf{S}_t=\sum_{j\le t}\phi(\mathbf{k}_j)\mathbf{v}_j^\top,\ \mathbf{z}_t=\sum_{j\le t}\phi(\mathbf{k}_j)$$ | 状态与归一化向量的前缀和 |
| (8.5) | $$\mathbf{y}_t=\phi(\mathbf{q}_t)^\top\mathbf{S}_t/(\phi(\mathbf{q}_t)^\top\mathbf{z}_t)$$ | 由状态读出输出 |
| (8.6) | $$\mathbf{S}_t=\mathbf{S}_{t-1}+\phi(\mathbf{k}_t)\mathbf{v}_t^\top,\ \mathbf{z}_t=\mathbf{z}_{t-1}+\phi(\mathbf{k}_t)$$ | 线性注意力递推 |
| (8.7) | $$O(T^2d)\ \text{vs}\ O(Td^2),\ O(d^2)\ \text{state}$$ | 复杂度对照 |
| (8.8) | $$\mathbf{h}'=\mathbf{A}\mathbf{h}+\mathbf{B}\mathbf{x},\ \mathbf{y}=\mathbf{C}\mathbf{h}+\mathbf{D}\mathbf{x}$$ | 连续状态空间方程 |
| (8.9) | $$\bar{\mathbf{A}}=e^{\Delta\mathbf{A}},\ \bar{\mathbf{B}}=(\Delta\mathbf{A})^{-1}(e^{\Delta\mathbf{A}}-\mathbf{I})\Delta\mathbf{B}$$ | 零阶保持离散化 |
| (8.10) | $$h_{t,n}=\bar{a}_{t,n}h_{t-1,n}+\bar{b}_{t,n}x_t,\ y_t=\sum_n c_{t,n}h_{t,n}+D x_t$$ | 对角逐通道递推 |
| (8.11) | $$y_t=\sum_{j\le t}\mathbf{C}\bar{\mathbf{A}}^{t-j}\bar{\mathbf{B}}x_j+D x_t$$ | 展开成卷积 |
| (8.12) | $$K_i=\mathbf{C}\bar{\mathbf{A}}^i\bar{\mathbf{B}}$$ | 时不变卷积核 |
| (8.13) | $$\Delta_t=\mathrm{softplus}(\mathbf{W}_\Delta\mathbf{x}_t+\mathbf{b}_\Delta),\ \mathbf{B}_t=\mathbf{W}_B\mathbf{x}_t,\ \mathbf{C}_t=\mathbf{W}_C\mathbf{x}_t$$ | 输入相关的选择性参数 |
| (8.14) | $$\bar{\mathbf{A}}_t=e^{\Delta_t\mathbf{A}},\ \bar{\mathbf{B}}_t=(\Delta_t\mathbf{A})^{-1}(e^{\Delta_t\mathbf{A}}-\mathbf{I})\Delta_t\mathbf{B}_t$$ | 输入相关的离散化 |
| (8.15) | $$O(TIN),\ O(IN)$$ | 选择性 SSM 复杂度与状态 |
| (8.16) | $$P_{\text{ssm}}=3dI+I(3N+7)+[4I+d\ \text{or}\ I]$$ | 单层 SSM 参数量 |
| (8.17) | $$(a_1,b_1)\circ(a_2,b_2)=(a_2a_1,a_2b_1+b_2)$$ | 并行扫描的结合律 |
| (8.18) | $$\mathbf{n}_t=e^{-\mathbf{w}}\odot\mathbf{n}_{t-1}+e^{\mathbf{k}_t}\odot\mathbf{v}_t,\ \mathbf{z}_t=e^{-\mathbf{w}}\odot\mathbf{z}_{t-1}+e^{\mathbf{k}_t}$$ | RWKV 分子与分母 |
| (8.19) | $$\mathbf{y}_t=\sigma(\mathbf{r}_t)\odot\mathbf{n}_t/\mathbf{z}_t$$ | RWKV 输出 |
| (8.20) | $$\text{kind}(i)=\text{attn}\ \text{if}\ (i+1)\bmod m=0$$ | 混合层模式 |
| (8.21) | $$M_{\text{attn}}=2BTn_{kv}d_hb$$ | 注意力缓存 |
| (8.22) | $$M_{\text{ssm}}=BINb$$ | SSM 状态 |
| (8.23) | $$F_{\text{dec}}\approx2N_{\text{param}}+n_{\text{attn}}4Td+n_{\text{ssm}}4IN$$ | 解码计算近似 |

### 实验与关键数字

- 账本（$$d=4096$$、32 层、GQA-8、FP16 状态、B=1）：注意力 7,241,732,096 参数，状态 1.0 GiB @8K / 16.0 GiB @128K，解码 18.8 / 83.2 GFLOP；线性注意力 8,047,038,464，状态 1.0 GiB / 1.0 GiB，18.2 / 18.2 GFLOP；纯 SSM 9,135,460,352，状态 8.0 MiB / 8.0 MiB，18.3 / 18.3 GFLOP；1:3 混合 8,662,028,288，状态 262.0 MiB / 4.0 GiB，18.4 / 34.5 GFLOP。
- 每层状态：注意力 32.0 MiB @8K、512.0 MiB @128K；线性注意力 32.0 MiB 恒定；SSM 256.0 KiB 恒定；混合平均 8.2 MiB / 128.2 MiB。
- 解码算力交叉点：注意力与 SSM 约 7,300 Token；8K 时两者几乎持平，128K 时注意力是 SSM 的约 4.5 倍。
- 参数公式核对：单层 SSM 31,744，两层模型 164,160，与 `sum(p.numel())` 逐位一致。
- 长上下文稳定性：T=8/64/512 时状态形状恒为 `(1, 128, 16)`，范数 0.08/0.10/0.10，输出全部有限。
- 复制任务消融（3 种子、600 步，损失/准确率均值±标准差）：理论下界 0.6261、随机基线准确率约 0.788。注意力 135,104 参数，0.6313±0.0033 / 0.7862±0.0008；线性注意力 135,104，1.2705±0.0358 / 0.6169±0.0124，状态范数 16221.77；纯 SSM 166,208，2.0816±1.1965 / 0.3025±0.4136，状态范数 151.54；1:1 混合 150,656，0.6299±0.0012 / 0.7876±0.0009，状态范数 21.73。
- 测试：22 项全部通过，覆盖默认配置与第 7 章逐位一致、线性注意力扫描与二次形式等价、并行扫描与逐步递推一致、SSM 状态形状与序列长度解耦、参数公式、四类块型的账本、因果性、长序列稳定性、混合层型与状态账本。

### 留给后续章节的伏笔

- **第 9 章**：缩放定律的横轴是训练计算，混合架构的每 Token FLOPs 与上下文相关；比较不同架构的计算最优配置时，不能直接套用稠密 Transformer 的拟合结果。
- **第 10 章**：长上下文与检索任务的数据配比决定混合比例与线性层层数；纯 SSM 在复制类任务上的短板需要数据与评测共同兜住。
- **第 11 章**：SSM 的状态范数、衰减参数与门控是训练稳定性的新变量；(8.10) 的负实部约束是数值稳定的前提。
- **第 12 章**：并行扫描与序列并行是分布式的独立话题；混合架构的注意力层与递推层需要不同的切分策略。
- **第 13 章**：混合架构、分块线性注意力与位置外推是长上下文训练的三条并行路线，本章的账本可直接用于显存预算。
- **第 19 章**：把训练好的 Transformer 线性化或蒸馏成 SSM/线性注意力，是压缩长上下文推理成本的一条路线。
- **第 21 章**：KV Cache 的分页、量化与调度只作用于注意力层；混合架构让这部分显存按注意力占比缩小。
- **第 23 章**：SSM 状态对低精度更敏感，衰减乘积在 FP8/FP16 下的误差会累积；量化方案要单独评估状态与卷积核。
- **第 27 章**：长上下文评测（如 RULER）是检验混合架构是否真的保住检索能力的必要环节，不能只看平均损失。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、对比、代码、工业实践、论文、误区、小结、"本章的取舍"、习题齐全。
- [x] 推导无跳步：核替换与状态递推、前缀和、零阶保持（含积分推导）、卷积展开、选择性离散化、并行扫描结合律、RWKV 递推、状态与算力账本均有完整步骤。
- [x] 符号与附录 B 一致：新增 $$\phi$$、$$\mathbf{S}_t$$、$$\mathbf{z}_t$$、$$d_\phi$$、$$\bar{\mathbf{A}}$$、$$\bar{\mathbf{B}}$$、$$\Delta_t$$、$$N$$（状态维度）、$$I$$、$$e$$（扩展倍数）、$$\mathbf{w}$$、$$M_{\text{attn}}$$、$$M_{\text{ssm}}$$、$$F_{\text{dec}}$$ 已登记；状态维度用 $$N$$ 并注明与第 7 章模型总参数 $$N$$ 的冲突，读时按上下文区分；扩展倍数用 $$e$$ 并注明不与自然常数混淆。
- [x] 代码可运行：从 `part2/chapter08.md` 直接提取代码块，在本机 Python 3.13 + PyTorch 2.9 CPU 上运行，主程序输出与正文逐字一致。
- [x] 测试可运行：同一 Markdown 提取的 `tests/test_ssm.py` 通过 22 项测试（约 1 s），覆盖等价性、扫描、因果性、状态大小与账本。
- [x] 公式与代码对应：代码注释标注 (8.1)–(8.6)、(8.8)–(8.10)、(8.13)–(8.17)。
- [x] GitBook 语法：hint 使用 info/success 两类，1 个 mermaid、1 个 details、2 张表；代码块均标注语言；数学全部用双美元符号；公式 (8.1)–(8.23) 连续。
- [x] 篇幅：正文 7,893 个中文字符，符合其余章 6,000–9,000 字预算。
- [x] 论文信息核对：Katharopoulos 2020（arXiv:2006.16236）、Gu/Goel/Ré 2021（arXiv:2111.00396）、Gu/Dao 2023（arXiv:2312.00752）、Peng 2023（arXiv:2305.13048）、Lieber 2024（arXiv:2403.19887）的作者、年份、标题与编号均无误；正文另引 Jelassi 2024（arXiv:2402.01032）与 Merrill 2024（arXiv:2404.08819）。
- [x] 事实性数字可复算：4096 维、32 层、GQA-8 的状态与 FLOPs 均由 (8.21)–(8.23) 复算；SSM 参数 31,744 与 164,160 由 (8.16) 手算复核；Jamba 32 层（4 注意力 + 28 Mamba）、MiniMax-01 每 8 层 1 个 softmax 注意力、Qwen3-Next 48 层 3:1 均按公开模型卡/技术报告核对。
- [x] 实验诚实报告：明确标注 2 层、64 维、600 步的复制任务是玩具实验；纯 SSM 的高方差与失败种子如实报告，不宣称"SSM 全面劣于注意力"；明确说明 CPU 朴素扫描的训练速度不能外推到 GPU 内核。

## 第 9 章交接

### 本章摘要

本章把损失拆成不可约项、容量项与数据项之和，用双幂律描述损失随参数量与数据量的下降。从训练算力账本 $$C\approx6ND$$ 出发，用拉格朗日乘子推出计算最优配比的平衡条件与两个指数，说明 Chinchilla 的"参数与数据同速增长"从何而来，以及为什么"20 Token/参数"只是局部经验值。接着讨论数据受限时的有效数据量公式、重复训练的收益递减、关键批大小与学习率缩放规则，以及涌现能力与度量选择的争议。代码给出 `tinylm/scaling.py` 的纯函数实现、合成数据拟合验证、复制任务上的三规模三预算扫描，以及复用第 8 章 `decode_flops` 的训练加推理账本。实验表明：拟合器能恢复指数但常数不可辨识；最优 Token/参数比随算力从 10 缓慢升到 25；重复 32 遍只能拿到约 45% 的有效数据；玩具扫描贴近损失下界，不能用来估计指数。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (9.1) | $$L(N,D)=L_\infty+A/N^{\alpha}+B/D^{\beta}$$ | 双幂律损失 |
| (9.2) | $$C\approx6ND$$ | 训练算力账本 |
| (9.3) | $$\Lambda=L_\infty+A/N^\alpha+B/D^\beta+\lambda(6ND-C)$$ | 拉格朗日函数 |
| (9.4) | $$\alpha A N^{-\alpha}=\beta B D^{-\beta}$$ | 计算最优的平衡条件 |
| (9.5) | $$N_{\mathrm{opt}}\propto C^{\beta/(\alpha+\beta)},\ D_{\mathrm{opt}}\propto C^{\alpha/(\alpha+\beta)}$$ | 最优分配指数 |
| (9.6) | $$N_{\mathrm{opt}}=g_0(C/6)^a,\ D_{\mathrm{opt}}=g_0^{-1}(C/6)^b,\ a+b=1$$ | 参数化分配 |
| (9.7) | $$L_{\mathrm{opt}}-L_\infty\propto C^{-\alpha\beta/(\alpha+\beta)}$$ | 损失随算力的幂律 |
| (9.8) | $$D_{\mathrm{eff}}=U+UR^*(1-e^{-R/R^*}),\ R=D/U-1$$ | 有效数据量 |
| (9.9) | $$L(N,D)=L_\infty+A/N^\alpha+B/D_{\mathrm{eff}}^\beta$$ | 数据受限损失 |
| (9.10) | $$B_{\mathrm{crit}}\approx\operatorname{tr}(\boldsymbol{\Sigma})/\lVert\mathbf{g}\rVert^2$$ | 关键批大小 |
| (9.11) | $$\eta(B)=\eta_0 B/B_0\ (B\ll B_{\mathrm{crit}}),\ \eta_0\sqrt{B/B_0}\ (B\gg B_{\mathrm{crit}})$$ | 学习率缩放规则 |

### 实验与关键数字

- 合成数据拟合（35 点、1% 乘性噪声）：真值 $$E=1.7,A=B=400,\alpha=0.34,\beta=0.28$$；拟合 $$E=1.6879$$（0.71%）、$$A=553.4148$$（38.35%）、$$B=313.1785$$（21.71%）、$$\alpha=0.3621$$（6.51%）、$$\beta=0.2651$$（5.32%），对数损失 RMSE 0.00925。
- 计算最优分配（$$a=0.46,b=0.54,g_0=1.858$$）：$$C=10^{20}$$ 时 $$N=1.292\times10^9$$、$$D=1.290\times10^{10}$$、$$D/N=10.0$$、$$L=2.6205$$；$$C=10^{25}$$ 时 $$N=2.577\times10^{11}$$、$$D=6.468\times10^{12}$$、$$D/N=25.1$$、$$L=1.8499$$。
- 数据重复（$$U=10^{12}$$、$$R^*=15.4$$）：重复 1 遍有效比 0.984，3 遍 0.931，7 遍 0.828，15 遍 0.662，31 遍 0.448；上限 16.4U。
- 玩具扫描（复制任务，3 种子）：理论下界 0.6261、随机损失 2.7726；`s` 模型 10,336 参数，1,638,400 Token 时损失 0.8261、准确率 0.7410；`m` 模型 43,248 参数，409,600 Token 时 0.6394/0.7885；`l` 模型 76,096 参数，1,638,400 Token 时 0.6331/0.7883。`m` 在 400 步已优于 `l` 在 100 步，方向上符合平衡条件；算力跨度约 1.8 个数量级，不足以拟合五个参数。
- 训练 + 推理账本（8K 上下文、推理 $$10^{13}$$ Token）：8B/15T 参数 8,030,261,248，训练 $$7.23\times10^{23}$$，每 Token 解码 $$2.04\times10^{10}$$，推理 $$2.04\times10^{23}$$，合计 $$9.26\times10^{23}$$；70B/1.4T 参数 70,553,706,496，训练 $$5.93\times10^{23}$$，每 Token 解码 $$1.63\times10^{11}$$，推理 $$1.63\times10^{24}$$，合计 $$2.22\times10^{24}$$。总成本在推理 $$9.15\times10^{11}$$ Token 后相等。
- 工业数字：Chinchilla 70B/1.4T（$$C\approx5.9\times10^{23}$$）对比 Gopher 280B/300B（$$C\approx5.0\times10^{23}$$）；Llama 3 8B/70B >15T、405B 15.6T（$$C\approx3.8\times10^{25}$$）；DeepSeek-V3 671B 总 / 37B 激活 / 14.8T Token / 2.788M H800 GPU 小时，按激活参数约 400 Token/参数。
- 测试：9 项全部通过（2.82 s），覆盖幂律与算力手算、指数和为一、预算约束、平衡条件、合成数据参数恢复、有效数据饱和、损失随预算下降。

### 留给后续章节的伏笔

- **第 10 章**：数据配比、去重与质量过滤直接改变唯一数据量 $$U$$ 与有效数据量 $$D_{\mathrm{eff}}$$；数据课程与退火会让损失曲线偏离固定配比的幂律，缩放定律的常数需要按数据配方重新拟合。
- **第 11 章**：优化器、学习率与批大小决定能否到达 (9.1) 预测的损失；(9.10)、(9.11) 与 μP 会在那里展开。损失尖峰会让曲线短暂偏离幂律，不能当成缩放定律失效。
- **第 12 章**：分布式训练的实际算力要乘 MFU；(9.2) 是理论上界，通信、重计算与流水线气泡会让有效算力低于账面值。
- **第 13 章**：长上下文在 (9.2) 之外增加注意力 FLOPs，并把有效数据从"Token 数"变成"文档数与长度分布"的函数；长上下文缩放需要单独的账本。
- **第 14、16、18 章**：预训练缩放定律不覆盖 SFT、RLHF 与 RLVR 的算力；后训练的数据量与采样次数是另一套缩放问题。
- **第 19 章**：蒸馏可以把小模型的损失推向大模型的水平，等价于在参数轴上"抄近路"，会改变计算最优前沿。
- **第 24 章**：测试时计算扩展是第三条轴：固定参数与训练算力，用更多推理 Token 换能力；它与 (9.6) 的训练侧分配需要一起优化。
- **第 27 章**：涌现与逆缩放的争议说明评测指标会改变结论；缩放定律预测的是损失，不是能力，评测必须同时报告连续指标与任务指标。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、对比、代码、工业实践、论文、误区、小结、"本章的取舍"、习题齐全。
- [x] 推导无跳步：双幂律三项含义、6ND 账本、拉格朗日平衡条件、指数求解、损失-算力幂律、有效数据量性质、关键批大小与学习率规则均有完整步骤。
- [x] 符号与附录 B 一致：新增 $$A,B$$、$$C$$、$$a,b$$、$$g_0$$、$$L_{\mathrm{opt}}$$、$$U$$、$$D_{\mathrm{eff}}$$、$$R$$、$$R^*$$、$$B_{\mathrm{crit}}$$、$$\boldsymbol{\Sigma}$$、$$\mathbf{g}$$、$$N_c/D_c$$、$$\alpha_N/\alpha_D/\alpha_C$$ 已登记；$$\alpha,\beta$$ 与第 7 章辅助损失系数冲突、$$B$$ 与批大小冲突、$$\mathbf{g}$$ 与第 11 章梯度冲突、$$G$$ 与第 7 章组数冲突，均已在易混点中说明。
- [x] 代码可运行：从 `part3/chapter09.md` 直接提取两个 Python 代码块，与 `/tmp` 实测文件逐字一致；主程序输出与正文逐字一致。
- [x] 测试可运行：同一 Markdown 提取的 `tests/test_scaling.py` 通过 9 项测试（2.82 s），覆盖幂律、预算约束、平衡条件、拟合恢复与有效数据饱和。
- [x] 公式与代码对应：代码注释标注 (9.1)–(9.6)、(9.8)；测试注释标注 (9.1)–(9.6)、(9.8)。
- [x] GitBook 语法：hint 使用 info/warning/success 三类，1 个 mermaid、1 个 details、3 张表；代码块均标注语言；全文无落单 `$`；公式 (9.1)–(9.11) 连续。
- [x] 篇幅：正文 7,585 个中文字符，在其余章 6,000–9,000 字预算内，落在 7,500–8,500 目标区间。
- [x] 论文信息核对：Kaplan 2020（arXiv:2001.08361）、Hoffmann 2022（arXiv:2203.15556）、Muennighoff 2023（arXiv:2305.16264）、Sardana 2024（arXiv:2401.00448）、Schaeffer 2023（arXiv:2304.15004）的作者、年份、标题与编号均无误；正文另引 Wei 2022（arXiv:2206.10658）、McKenzie 2023（arXiv:2306.09479）。
- [x] 事实性数字可复算：Chinchilla $$5.9\times10^{23}$$、Gopher $$5.0\times10^{23}$$、Llama 3 8B $$7.2\times10^{23}$$、405B $$3.8\times10^{25}$$、DeepSeek-V3 $$3.3\times10^{24}$$ 均由 (9.2) 复算；70B/1.4T 锚点反解 $$g_0=1.858$$。
- [x] 实验诚实报告：明确标注玩具扫描算力跨度仅约 1.8 个数量级、部分配置贴近下界，不能用于估计指数；合成数据验证的是拟合器而非真实缩放定律；所有工业数字标注为公开报告值或按 (9.2) 复算。

## 第 10 章交接

### 本章摘要

本章把预训练数据工程拆成可复算的漏斗：原始抓取、正文抽取、语种识别、质量过滤、精确去重、近似去重、防污染、配比与打包。用 (10.1) 统一文档与 Token 的逐级保留率，说明清洗是"用算力换信息密度"的预算问题。去重部分推导精确哈希、shingle + Jaccard、MinHash 的无偏性与 LSH 的 S 曲线，给出 $$b=16,r=8$$ 的召回表与拐点。质量过滤给出启发式打分 (10.6) 与产出率/精确率权衡 (10.7)，强调按领域校准阈值。配比部分给出按比例 (10.8)、温度采样 (10.9)、最大余数预算 (10.10) 与配比定律 (10.11)，并讨论 DoReMi。课程与退火用 (10.12) 描述末段切换，合成数据讨论模型坍缩与真实数据锚点，污染检测给出覆盖率 (10.13)、去污染规则 (10.14) 与污染率 (10.15)。代码实现 `tinylm/data/filter.py`、`tinylm/data/mixture.py` 与 `ch10.py`，在 248 篇合成语料上跑完整漏斗、3 种子训练对照与配比消融；13 项测试全部通过。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (10.1) | $$N_k=N_0\prod_{j=1}^{k}r_j$$ | 数据管道逐级保留率 |
| (10.2) | $$h(x)=\mathrm{BLAKE2b}(\mathrm{norm}(x))$$ | 精确去重哈希 |
| (10.3) | $$J(A,B)=\lvert A\cap B\rvert/\lvert A\cup B\rvert$$ | Jaccard 相似度 |
| (10.4) | $$s_i(A)=\min_{g\in A}h_i(g),\ \Pr[s_i(A)=s_i(B)]=J(A,B)$$ | MinHash 无偏估计 |
| (10.5) | $$P(s)=1-(1-s^r)^b$$ | LSH 至少一段碰撞概率 |
| (10.6) | $$q(x)=0.30\ell+0.40u+0.15a+0.15(1-f)\ge\tau$$ | 启发式质量分与阈值 |
| (10.7) | $$\mathrm{yield}=\lvert K\rvert/\lvert D\rvert,\ \mathrm{precision}=\lvert K\cap G\rvert/\lvert K\rvert$$ | 过滤产出率与精确率 |
| (10.8) | $$p_i=w_i/\sum_j w_j$$ | 按来源比例配比 |
| (10.9) | $$p_i=w_i^{1/T}/\sum_j w_j^{1/T}$$ | 温度采样配比 |
| (10.10) | $$n_i=\lfloor Np_i\rfloor+\delta_i,\ \sum_i\delta_i=N-\sum_i\lfloor Np_i\rfloor$$ | 最大余数法预算 |
| (10.11) | $$L_i(\mathbf{p})=c_i+k_i\exp(\sum_j t_{ij}p_j)$$ | 数据配比定律 |
| (10.12) | $$\mathbf{p}(t)=\mathbf{p}_{\text{base}}\ (t\le(1-\rho)D),\ \mathbf{p}_{\text{anneal}}\ (t>(1-\rho)D)$$ | 退火/数据课程 |
| (10.13) | $$c(d)=\lvert\mathrm{grams}_n(d)\cap G_{\text{train}}\rvert/\lvert\mathrm{grams}_n(d)\rvert$$ | 评测文档的 n-gram 覆盖率 |
| (10.14) | $$\mathcal{D}_{\text{train}}^{\text{clean}}=\{x:c_{\text{eval}}(x)<\theta\}$$ | 去污染规则 |
| (10.15) | $$\mathrm{CR}=\frac{1}{M}\sum_m\mathbf{1}[c_m\ge\theta],\ \Delta=L_{\text{clean}}-L_{\text{leak}}$$ | 污染率与分数虚高 |

### 实验与关键数字

- 漏斗：248 篇 / 10473 Token → 质量过滤 228 / 9398 → 精确+近似去重 179 / 7283 → 防污染 171 / 6957；文档保留率 69.0%，Token 保留率 66.4%。质量过滤删 20 篇 / 1075 Token，去重删 49 篇 / 2115 Token，防污染删 8 篇 / 326 Token。
- 污染检测：8-gram、阈值 0.5；20 篇评测命中 8 篇（注入的 8 条泄漏全部命中），40 篇干净文档假阳性 0；命中文档覆盖率 0.64–1.0。
- 训练对照（3 种子、150 步，干净验证 / 泄漏评测）：原始 3.3896±0.023 / 3.3901±0.030；仅质量过滤 3.3353±0.015 / 3.3344±0.013；仅去重 3.3254±0.001 / 3.3259±0.004；质量+去重+防污染 3.2991±0.005 / 3.3236±0.015。
- 配比表（预算 400 篇）：按来源比例 201/98/63/38（最大/最小 5.3）；温度 T=2 为 149/104/83/64（2.3）；等权 100/100/100/100（1.0）。
- 配比消融（2 种子、120 步，按来源比例 vs 等权）：百科 3.3803 → 3.4849，代码 3.5280 → 3.5442，数学 3.6173 → 3.5328，对话 3.3297 → 3.0750。
- 测试：13 项全部通过（0.04 s），覆盖分词、精确去重、Jaccard 边界、MinHash 估计、近似去重、LSH 分桶、质量过滤、污染检测、去污染、温度采样、预算分配、交错配额与非法温度。

### 留给后续章节的伏笔

- **第 11 章**：优化器、学习率与批大小决定能否到达第 9 章缩放曲线预测的损失；数据质量会改变损失尖峰的频率与幅度，清洗后的语料通常更稳定。
- **第 12 章**：分布式训练的数据加载、分片与去重需要在全局视角完成；单机脚本里的 `interleave` 只是配比的语义原型。
- **第 13 章**：长上下文训练需要按长度分布配比数据，并把长文档的打包与截断纳入漏斗；本章的 Token 预算没有区分长度。
- **第 14、16、18 章**：SFT、RLHF 与 RLVR 的数据同样需要去重、质量过滤与防污染，但评测泄漏的形态从预训练 n-gram 变成提示—答案对。
- **第 19 章**：蒸馏与合成数据共用"教师生成、学生过滤"的流程，本章的合成数据风险讨论可直接复用。
- **第 24 章**：测试时计算扩展需要高质量的可验证数据；数据配比会改变采样预算的边际收益。
- **第 27 章**：污染检测与评测设计直接相关；本章的 n-gram 覆盖率只是下界，真实评测还要看改写、翻译与金丝雀串。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、对比、代码、工业实践、论文、误区、小结、"本章的取舍"、习题齐全。
- [x] 推导无跳步：漏斗保留率、MinHash 无偏性、LSH S 曲线、质量过滤产出率/精确率、温度采样、最大余数法、配比定律、退火、污染覆盖率均有完整步骤。
- [x] 符号与附录 B 一致：新增 $$N_k$$、$$r_j$$、$$r_j^{(T)}$$、$$c_{\text{filter}}$$、$$h(x)$$、$$A,B$$（shingle 集合）、$$s_i(A)$$、$$m$$（哈希个数）、$$b$$（段数）、$$r$$（每段行数）、$$P(s)$$、$$q(x)$$、$$\ell,u,a,f$$、$$\tau$$、$$K$$（领域数）、$$G$$、$$p_i$$、$$w_i$$、$$T$$（采样温度）、$$n_i$$、$$\delta_i$$、$$c_i,k_i,t_{ij}$$、$$\rho$$（退火比例）、$$c(d)$$、$$G_{\text{train}},G_{\text{eval}}$$、$$\theta$$（污染阈值）、$$\mathrm{CR}$$、$$\Delta$$ 已登记，并在附录 B 标注与前后章的冲突。
- [x] 代码可运行：从 `part3/chapter10.md` 提取 4 个 Python 代码块，与 `/tmp` 实测文件逐字一致；在 Python 3.13 + PyTorch 2.9 CPU 上运行，主程序输出与正文逐字一致。
- [x] 测试可运行：同一 Markdown 提取的 `tests/test_data.py` 通过 13 项测试（0.04 s）。
- [x] 公式与代码对应：代码注释标注 (10.2)–(10.6)、(10.9)、(10.10)、(10.13)、(10.14)。
- [x] GitBook 语法：hint 使用 info/warning/success 三类，1 个 mermaid、1 个 details、5 张表；代码块均标注语言；数学全部用双美元符号；公式 (10.1)–(10.15) 连续。
- [x] 篇幅：正文 7853 个中文字符，在其余章 6,000–9,000 字预算内，落在 7,500–8,500 目标区间。
- [x] 论文信息核对：Lee 2021（arXiv:2107.06922）、Xie 2023（arXiv:2305.10429）、Penedo 2023（arXiv:2306.01116）、Ye 2024（arXiv:2403.16952）、Gunasekar 2023（arXiv:2306.11644）的作者、年份、标题与编号均无误。
- [x] 事实性数字可复算：漏斗各级文档/Token 数与保留率由程序输出；LSH 概率表由 (10.5) 手算；配比表由 (10.9)、(10.10) 复算；RefinedWeb 5T→600B 保留率约 12%；Llama 3、DoReMi、Phi-3 的数字标注为"截至写作时"的公开报告值。
- [x] 实验诚实报告：明确标注合成语料与 2 层 32 维玩具模型不能外推；泄漏率仅 3.2% 时污染对平均损失的影响可能被清洗收益掩盖；配比消融是领域间的交换而非全面改进；所有数字附标准差或说明单次运行。

## 第 11 章交接

### 本章摘要

本章把训练稳定性拆成四层：优化器、学习率调度、批大小缩放与稳定性工具。从 SGD 与动量出发推导 Adam 的一阶/二阶矩与偏差修正 (11.1)–(11.5)，说明自适应步长近似逐坐标符号下降；比较耦合 L2 (11.6) 与解耦衰减 (11.7)，用近端优化视角解释 AdamW 的可迁移性，并讨论参数分组（偏置与归一化增益不衰减）。调度部分给出预热 (11.8)、余弦 (11.9)、WSD (11.10)，分析预热长度与"有效训练步数"；批大小部分连接第 9 章关键批大小，给出线性/平方根规则 (11.11)、(11.12) 与批大小—学习率扫描流程。稳定性部分给出尖峰检测 (11.13)、梯度裁剪 (11.14)、更新比例 (11.15)、QK-norm (11.16)、z-loss (11.17)、残差缩放初始化 (11.18) 与"诊断—恢复—记录"四步流程，并区分数据异常、注意力塌缩、二阶矩污染与混合精度溢出。Muon 部分用 Newton-Schulz (11.20) 近似极分解，给出更新 (11.19)、逐形状缩放与谱范数解释，并说明嵌入层/输出层仍需 AdamW。代码实现 `tinylm/train/optim.py`、`tests/test_optim.py` 与 `ch11.py`，在病态二次型、TinyLM 3 种子优化器对照、预热/调度对照与注入梯度裁剪演示上验证；13 项测试全部通过。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (11.1) | $$\theta_{t+1}=\theta_t-\eta_t\mathbf{g}_t$$ | SGD 更新 |
| (11.2) | $$\mathbf{m}_t=\beta_1\mathbf{m}_{t-1}+(1-\beta_1)\mathbf{g}_t$$ | 动量 |
| (11.3) | $$\mathbf{m}_t,\mathbf{v}_t$$ 的指数滑动平均 | Adam 一阶/二阶矩 |
| (11.4) | $$\hat{\mathbf{m}}_t=\mathbf{m}_t/(1-\beta_1^t),\ \hat{\mathbf{v}}_t=\mathbf{v}_t/(1-\beta_2^t)$$ | 偏差修正 |
| (11.5) | $$\theta_{t+1}=\theta_t-\eta_t\hat{\mathbf{m}}_t/(\sqrt{\hat{\mathbf{v}}_t}+\epsilon)$$ | Adam 更新 |
| (11.6) | $$\mathbf{g}_t^{\mathrm{L2}}=\mathbf{g}_t+\lambda\theta_t$$ | 耦合 L2 |
| (11.7) | $$\theta_{t+1}=(1-\eta_t\lambda)\theta_t-\eta_t\hat{\mathbf{m}}_t/(\sqrt{\hat{\mathbf{v}}_t}+\epsilon)$$ | AdamW 解耦衰减 |
| (11.8) | $$s_t=t/T_w$$ | 线性预热 |
| (11.9) | $$s_t=s_{\min}+(1-s_{\min})(1+\cos(\pi(t-T_w)/(T-T_w)))/2$$ | 余弦退火 |
| (11.10) | $$s_t=1\ (t\le T_s),\ s_{\min}+(1-s_{\min})((T-t)/(T-T_s))^p\ (t>T_s)$$ | WSD 调度 |
| (11.11) | $$\eta=\eta_0 B/B_0$$ | 线性缩放（$$B\ll B_{\mathrm{crit}}$$） |
| (11.12) | $$\eta=\eta_0\sqrt{B/B_0}$$ | 平方根缩放（$$B\gg B_{\mathrm{crit}}$$） |
| (11.13) | $$z_t=(L_t-\mu_t)/\sigma_t$$ | 损失尖峰 z 分数 |
| (11.14) | $$\mathbf{g}\leftarrow\mathbf{g}\min(1,\tau_{\mathrm{clip}}/\lVert\mathbf{g}\rVert_2)$$ | 全局梯度裁剪 |
| (11.15) | $$r_t=\lVert\theta_t-\theta_{t-1}\rVert_2/\lVert\theta_{t-1}\rVert_2$$ | 更新与权重比例 |
| (11.16) | $$s_{t,u}=\tilde{\mathbf{q}}_t^\top\tilde{\mathbf{k}}_u/\sqrt{d_h}$$ | QK-norm 后的打分 |
| (11.17) | $$\mathcal{L}_z=\beta_z(\log Z)^2$$ | z-loss |
| (11.18) | $$\mathrm{Var}(W_{ij})=2/(n_{\mathrm{in}}+n_{\mathrm{out}}),\ W_{\mathrm{res}}\leftarrow W_{\mathrm{res}}/\sqrt{2n_l}$$ | 初始化与残差缩放 |
| (11.19) | $$\theta_t=\theta_{t-1}-\eta_t\gamma\mathbf{O}_t$$ | Muon 更新 |
| (11.20) | $$\mathbf{A}_k=\mathbf{X}_k\mathbf{X}_k^\top,\ \mathbf{X}_{k+1}=a\mathbf{X}_k+(b\mathbf{A}_k+c\mathbf{A}_k^2)\mathbf{X}_k$$ | Newton-Schulz 正交化 |

### 实验与关键数字

- 病态二次型（条件数 1000，100 步）：SGD（lr=0.0015）在第 1/10/50/100 步的损失为 500.5 / 0.4886 / 0.4316 / 0.3714；Adam（lr=0.1）为 500.5 / 11.81 / 0.0214 / 0.0023。Adam 早期过冲，但 100 步后低两个数量级。
- 优化器对照（300 步、3 种子、验证损失）：SGD+动量 3.3158±0.0280；AdamW（解耦）3.2895±0.0069；Adam（L2 耦合）3.4351±0.0092。最后 10 步训练损失分别为 3.2532 / 3.1929 / 3.3738。
- 预热对照（峰值学习率 0.5、150 步、3 种子）：无预热 3.7122±0.1628、训练损失峰值 10.31；40 步预热 3.6222±0.0662、峰值 5.15。
- 调度对照（峰值 0.01、300 步、3 种子）：constant 3.5647±0.0252；cosine 3.4093±0.0319；wsd 3.5125±0.0104。固定预算下余弦最好，WSD 的稳定段占比过高导致末段衰减不足。
- 裁剪演示（SGD 动量，第 10 步梯度 ×50）：不裁剪时最大 $$|x|=10.79$$、尖峰后损失 81.78、最终 2.0301；阈值 1.0 时为 4.96 / 6.27 / 0.0292。
- 测试：13 项全部通过（1.69 s），覆盖 AdamW 闭式解、偏差修正、解耦 vs 耦合、参数分组、预热端点、余弦单调、WSD 平台与衰减、非法调度、全局裁剪、小梯度不变、尖峰检测、Newton-Schulz 条件数、Muon 二次下降。

### 留给后续章节的伏笔

- **第 12 章**：Adam 的两份优化器状态是显存的主要来源，ZeRO/FSDP 的分片对象就是参数、梯度与优化器状态；梯度累积与全局批大小决定第 11 章学习率缩放如何落地；混合精度的损失缩放与主权重在这里展开。
- **第 13 章**：长上下文训练需要激活重算与序列并行，学习率与预热要按有效批大小重新校准；注意力熵在长序列上更敏感，是尖峰诊断的重要指标。
- **第 14、16、18 章**：SFT、RLHF 与 RLVR 沿用本章优化器与调度，但有效批大小、KL 系数与奖励尺度会改变稳定区间；PPO 的比率裁剪与本章的梯度裁剪不是一回事，术语需区分。
- **第 19 章**：蒸馏常需要更小的学习率与更短预热；教师信号的尺度会改变有效步长，稳定性诊断要重做。
- **第 20 章**：解码的采样温度与本章调度的"温度"无关，术语不得混用。
- **第 24 章**：测试时计算扩展不改变训练优化器，但 FP8/FP4 推理会引入新的数值问题，需要与训练尖峰区分。
- **第 26、27 章**：MoE 与长上下文模型的稳定性工具（QK-Clip、z-loss、路由监控）在本章基础上展开；第 7 章的辅助损失会改变梯度尺度，需与本章的参数分组一起考虑。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、对比、代码、工业实践、论文、误区、小结、"本章的取舍"、习题齐全。
- [x] 推导无跳步：动量展开、偏差修正、符号下降、耦合 L2 的放大倍数、近端视角、预热/余弦/WSD、线性与平方根缩放、尖峰 z 分数、裁剪、QK-norm 上界、Newton-Schulz 迭代均有完整步骤。
- [x] 符号与附录 B 一致：新增 $$\eta_t$$、$$\beta_1,\beta_2$$、$$\epsilon$$、$$\lambda$$、$$\mathbf{m}_t,\mathbf{v}_t$$、$$\hat{\mathbf{m}},\hat{\mathbf{v}}$$、$$s_t$$、$$T_w$$、$$T_s$$、$$s_{\min}$$、$$p$$、$$\tau_{\mathrm{clip}}$$、$$r_t$$、$$z_t$$、$$\tilde{\mathbf{q}},\tilde{\mathbf{k}}$$、$$\beta_z$$、$$Z$$、$$\gamma$$、$$\mathbf{O}_t$$ 已登记；与第 9 章共享 $$B_{\mathrm{crit}},\boldsymbol{\Sigma},\mathbf{g}$$，与第 4/7/16 章的 $$\beta,\lambda,\epsilon$$ 冲突已在附录 B 标注。
- [x] 代码可运行：从 `part3/chapter11.md` 提取 3 个 Python 代码块，与 `/tmp` 实测文件逐字一致；在 Python 3.13 + PyTorch 2.9 CPU 上运行，主程序输出与正文逐字一致（耗时的秒数会随机器波动）。
- [x] 测试可运行：同一 Markdown 提取的 `tests/test_optim.py` 通过 13 项测试（1.69 s）。
- [x] 公式与代码对应：代码注释标注 (11.3)、(11.4)、(11.7)–(11.10)、(11.13)、(11.14)、(11.19)、(11.20)。
- [x] GitBook 语法：hint 使用 info/warning/danger 三类，1 个 mermaid、1 个 details、4 张表；代码块均标注语言；数学全部用双美元符号；公式 (11.1)–(11.20) 连续且按出现顺序编号。
- [x] 篇幅：正文 10368 个中文字符，在核心章 10,000–12,000 字预算内。
- [x] 论文信息核对：Kingma & Ba 2015（arXiv:1412.6980）、Loshchilov & Hutter 2019（arXiv:1711.05101）、Goyal et al. 2017（arXiv:1706.02677）、Wortsman et al. 2024（arXiv:2309.14322）、Liu et al. 2025（arXiv:2502.16982）的作者、年份、标题与编号均无误。
- [x] 事实性数字可复算：二次型轨迹、优化器/预热/调度/裁剪表均由程序输出；Llama 3、DeepSeek-V3、Kimi K2 的数字来自公开技术报告并标注"截至写作时"。
- [x] 实验诚实报告：明确标注 2 层 32 维玩具模型、150–300 步预算不能外推；WSD 在本章预算下不如余弦，正文说明原因是稳定段占 80%、末段衰减不足；Muon 的矩阵回归玩具实验显著差于 Adam，故只保留单元测试与理论/工业讨论，不写成普适优势。

## 第 12 章交接

### 本章摘要

本章把分布式训练拆成显存、通信与数值格式三条线。显存部分从模型状态 (12.1) 与激活 (12.2) 的分解出发，给出总显存 (12.3) 与 ZeRO 阶段 0–3 的每卡状态公式 (12.4)，说明分片组是数据并行组而非总卡数，并区分"省状态"与"省激活"两类问题。通信部分推导环形全归约 (12.5)、通信/计算时间 (12.6)、全局批大小 (12.7) 与学习率缩放 (12.8)，再给出张量并行每块四次集合通信 (12.9)、流水线气泡 (12.10) 与专家并行 all-to-all (12.11)，并讨论列/行并行、词表并行、1F1B、上下文并行与负载均衡。数值部分比较 FP32/FP16/BF16/FP8 的范围与尾数，推导动态损失缩放 (12.12) 与 FP8 逐块量化 (12.13)，把溢出与数据异常分开诊断。代码实现 `tinylm/train/parallel.py` 的账本函数、CPU 集合通信模拟、列/行并行原语与 `LossScaler`；`tests/test_parallel.py` 16 项测试覆盖账本、通信量、分片—聚合往返、张量并行等价与缩放器行为；`examples/ch12_parallel.py` 在 2 层 32 维 TinyLM 上验证整批、梯度累积、DP=4 与 ZeRO-1 四种模式在 3 个种子上等价，并打印 7B 显存与通信账本。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (12.1) | $$M_{\mathrm{state}}=N(c_w+c_g+c_m+2c_o)$$ | 模型状态字节数 |
| (12.2) | $$M_{\mathrm{act}}\approx c_a\frac{n_l}{N_p}B_m\frac{T}{N_s}d_{model}c_{\mathrm{act}}$$ | 激活显存估算 |
| (12.3) | $$M_{\mathrm{total}}=M_{\mathrm{state}}+M_{\mathrm{act}}+M_{\mathrm{temp}}$$ | 总显存 |
| (12.4) | $$M_{\mathrm{ZeRO}}$$ 分段：阶段 0–3 的每卡状态 | ZeRO 显存分解 |
| (12.5) | $$V_{\mathrm{DP}}=2\frac{N_d-1}{N_d}S$$ | 环形全归约每卡收发量 |
| (12.6) | $$T_{\mathrm{comm}}=\frac{V}{B_{\mathrm{bw}}},\ T_{\mathrm{comp}}=\frac{6ND_{\mathrm{tok}}}{N_{\mathrm{gpu}}F_{\mathrm{eff}}}$$ | 通信与计算时间 |
| (12.7) | $$B_{\mathrm{global}}=B_mAN_d$$ | 全局批大小 |
| (12.8) | $$\eta=\eta_0(B_{\mathrm{global}}/B_0)^\kappa$$ | 按全局批缩放学习率 |
| (12.9) | $$V_{\mathrm{TP}}=4\cdot2\frac{N_t-1}{N_t}B_mTd_{model}c_{\mathrm{act}}$$ | 每块张量并行通信量 |
| (12.10) | $$\mathrm{Bub}=\frac{N_p-1}{A+N_p-1}$$ | 流水线气泡占比 |
| (12.11) | $$V_{\mathrm{EP}}=2N_{\mathrm{tok}}kd_{model}c$$ | 每层专家并行通信量 |
| (12.12) | $$S_{\mathrm{ls}}\leftarrow S_{\mathrm{ls}}/2\ (\text{溢出}),\ 2S_{\mathrm{ls}}\ (\text{连续 }G\text{ 步正常})$$ | 动态损失缩放 |
| (12.13) | $$x_q=\mathrm{clamp}(x/S,-q_{\max},q_{\max}),\ \hat{x}=Sx_q$$ | FP8 逐块量化 |

### 实验与关键数字

- 等价性（50 步、3 种子）：整批 B=16、梯度累积 A=4、DP=4、ZeRO-1 四种模式的验证损失均为 3.8768 ± 0.0031；最大参数偏差 4.22e-06；每卡优化器状态占参数比例 0.250（4 卡分片，含 padding）。
- 7B 显存账本（BF16、8 卡、T=4096、d_model=4096、32 层）：ZeRO 阶段 0/1/2/3 的状态 112.0/38.5/26.2/14.0 GB，激活 12.9 GB，合计 124.9/51.4/39.1/26.9 GB；阶段 0 超过 80 GB，阶段 1 可放。
- 通信账本：DP=64 时梯度全归约每卡 27.56 GB，按 50 GB/s 约 0.55 s，整步计算 1.68e+17 FLOPs 约 6.56 s，通信/计算 8.4%；TP=8 每层 234.9 MB、32 层 7.52 GB/步；PP=16 时在途微批 16/64/256 对应气泡 48.4%/19.0%/5.5%；EP=64 每卡 4096 Token、Top-2、32 层 all-to-all 4.29 GB/步。
- 精度表：FP32/FP16/BF16 每参数状态 16 字节，FP8 为 15 字节；FP16 最大值 6.55e4、最小正规数 6.1e-5；BF16 与 FP32 范围相同；FP8 E4M3 最大值 448、最小正规数 0.0156。
- 损失缩放：梯度 1e-8 直接转 FP16 得 0，乘 1024 后得 1.025e-05，再除回 1.001e-08；scale 8→16（两步正常）→8（溢出一次，跳过 1 步）。
- 测试：16 项全部通过（1.62 s），覆盖每参数状态字节、ZeRO 7B 四阶段与单调性、激活账本与重算、环形通信量、DP/TP 字节、气泡、全局批与学习率、集合通信往返与 padding、列/行并行等价、TP MLP 等价、缩放器溢出/增长/反缩放、分片往返、分片 AdamW 等价与非法配置。

### 留给后续章节的伏笔

- **第 13 章**：长上下文训练把激活与 KV Cache 推到主项，序列并行、上下文并行与环形注意力是本章"序列维切分"的展开；激活重算的平方根规律与 (12.2) 直接相连。
- **第 14、16、18 章**：SFT、RLHF 与 RLVR 沿用本章的显存与混合精度工具，但强化学习的推理—训练分离、 rollout 显存与 KL 约束会引入新的账本项；PPO 的裁剪与本章梯度裁剪不是一回事。
- **第 19 章**：蒸馏与合成数据管线同样需要分布式训练与混合精度；教师前向的显存与吞吐要与学生训练一起记账。
- **第 20 章**：推理的 KV Cache、量化与批调度和训练的显存结构不同；本章的 FP8 逐块缩放可复用，但延迟指标不同。
- **第 24 章**：测试时计算扩展不改变训练并行，但 FP8/FP4 推理会引入新的数值问题，需要与训练溢出区分。
- **第 26、27 章**：MoE 的专家并行、容量因子与负载均衡是 (12.11) 的展开；长上下文并行会与本章的 TP/PP/DP 组合，重新决定拓扑。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、对比、代码、工业实践、论文、误区、小结、"本章的取舍"、习题齐全。
- [x] 推导无跳步：模型状态分解、激活估算、ZeRO 四阶段、环形全归约、通信/计算时间、全局批、学习率缩放、TP 四次通信、气泡、EP all-to-all、损失缩放、FP8 量化均有完整步骤。
- [x] 符号与附录 B 一致：新增 $$N_d,N_t,N_p,N_s,N_{\mathrm{gpu}},N_{\mathrm{tok}}$$、$$c_w,c_g,c_m,c_o,c_a,c_{\mathrm{act}}$$、$$M_{\mathrm{state}},M_{\mathrm{act}},M_{\mathrm{total}},M_{\mathrm{temp}},M_{\mathrm{ZeRO}}$$、$$B_m,B_{\mathrm{global}},B_{\mathrm{bw}},D_{\mathrm{tok}},F_{\mathrm{eff}}$$、$$V_{\mathrm{DP}},V_{\mathrm{TP}},V_{\mathrm{EP}},S$$、$$T_{\mathrm{comm}},T_{\mathrm{comp}}$$、$$A,\mathrm{Bub},k$$、$$S_{\mathrm{ls}},G,q_{\max},a_{\max},x_q$$ 已登记；与第 9 章的 $$A$$、第 10 章的 $$M$$、第 11 章的 $$B,T,p,\kappa$$ 冲突已在附录 B 易混点中说明。
- [x] 代码可运行：从 `part3/chapter12.md` 提取 3 个 Python 代码块，与 `/tmp/ch12run` 实测文件逐字一致；在 Python 3.13 + PyTorch 2.9 CPU 上运行，主程序输出与正文逐字一致（耗时秒数随机器波动）。
- [x] 测试可运行：同一 Markdown 提取的 `tests/test_parallel.py` 通过 16 项测试（1.62 s）。
- [x] 公式与代码对应：代码注释标注 (12.1)、(12.4)–(12.12)；`memory_ledger` 对应 (12.1)–(12.3)，`LossScaler` 对应 (12.12)，`ep_all_to_all_bytes` 对应 (12.11)。
- [x] GitBook 语法：hint 仅使用 warning 一类，1 个 mermaid、1 个 details、4 张表；代码块均标注语言；数学全部用双美元符号；公式 (12.1)–(12.13) 连续且按出现顺序编号。
- [x] 篇幅：正文 11122 个中文字符，在核心章 10,000–12,000 字预算内，接近 11,000 目标。
- [x] 论文信息核对：Rajbhandari 2020（arXiv:1910.02054）、Shoeybi 2019（arXiv:1909.08053）、Huang 2019（arXiv:1811.06965）、Narayanan 2021（arXiv:2104.04473）、Micikevicius 2018（arXiv:1710.03740）的作者、年份、标题与编号均无误。
- [x] 事实性数字可复算：四模式等价性、7B 显存表、通信账本、精度表与损失缩放均由程序输出；Llama 3 405B、DeepSeek-V3 的配置与规模来自公开技术报告并标注"截至写作时"；405B 的 DP 梯度量按 (12.5) 复算。
- [x] 实验诚实报告：明确标注 2 层 32 维玩具模型、CPU 列表集合通信不测真实带宽、激活系数 12 为估算；TP/PP 只验证原语与公式，未模拟完整层间调度；等价性偏差来源为浮点求和顺序。

## 第 13 章交接

### 本章摘要

本章讨论"窗口不等于能力"。先算三本账：因果注意力 FLOPs 按 $$T^2$$ 增长 (13.1)，线性项按 $$T$$ 增长 (13.2)，二者在 $$T^\star=4d_{model}+3d_{ff}-1$$ (13.3) 处交叉；推理 KV Cache (13.4) 与训练激活 (13.5) 分别按 $$T$$ 与 $$T^2$$ 增长。位置外推部分区分 PI/NTK/YaRN 的相位变换与模型学到的距离统计，指出零样本外推的种子间双峰现象。训练策略部分给出几何长度课程 (13.6)、混合长度训练与短上下文遗忘，并用 3 种子实验对照。数据部分推导打包的 Token 利用率 (13.7)，给出块对角因果掩码 (13.8) 与"掩码、边界标签、位置重置"三件套，证明打包与逐条前向的损失/梯度等价。工程部分推导上下文并行的 K/V 通信量 (13.9)，讨论环形注意力的因果负载不均。评测部分定义有效上下文 (13.10) 与单针放置 (13.11)，比较 NIAH、RULER 与"迷失在中间"的局限。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (13.1) | $$F_{\mathrm{attn}}=2\,B\,n_l\,d_{model}\,T(T+1)$$ | 因果注意力前向 FLOPs |
| (13.2) | $$F_{\mathrm{lin}}=2\,B\,T\,n_l(4d_{model}^2+3d_{model}d_{ff})$$ | 投影与 SwiGLU 前向 FLOPs |
| (13.3) | $$T^\star=4d_{model}+3d_{ff}-1$$ | 注意力与线性项交叉长度 |
| (13.4) | $$M_{\mathrm{kv}}=2\,B\,T\,n_l\,n_{kv}\,d_h\,b_{\mathrm{elem}}$$ | 推理 KV Cache 字节数 |
| (13.5) | $$M_{\mathrm{act}}\approx c_a B T n_l d_{model} b_{\mathrm{elem}}+\mathbf{1}_{\mathrm{scores}}B n_l n_h T^2 b_{\mathrm{elem}}$$ | 训练激活，含显式分数矩阵项 |
| (13.6) | $$L_i=L_0(L_N/L_0)^{i/(N_{\mathrm{st}}-1)}$$ | 几何长度课程 |
| (13.7) | $$\eta_{\mathrm{pad}}=\frac{\sum_i\ell_i}{N_{\mathrm{doc}}T_{\max}},\ \eta_{\mathrm{pack}}=\frac{\sum_i\ell_i}{\lceil\sum_i\ell_i/T_{\max}\rceil T_{\max}}$$ | 逐条 padding 与打包的 Token 利用率 |
| (13.8) | $$M_{ij}=\mathbf{1}[d_i=d_j]\cdot\mathbf{1}[j\le i]$$ | 块对角因果掩码 |
| (13.9) | $$V_{\mathrm{cp}}\approx 2\frac{N_{cp}-1}{N_{cp}}T d_{model}b_{\mathrm{elem}}$$ | 上下文并行每卡每层接收的 K/V 字节数 |
| (13.10) | $$T_{\mathrm{eff}}=\max\{T\in\mathcal{T}:\mathrm{metric}(T)\ge\epsilon\}$$ | 有效上下文长度 |
| (13.11) | $$u_i=\mathrm{round}\left(i(T-3)/(N_{\mathrm{probe}}-1)\right)$$ | 单针评测的等距放置深度 |

### 实验与关键数字

- 7B 账本（B=1、BF16、32 层、32 头、8 个 KV 头、$$d_h=128$$）：4K/32K/128K 的注意力占前向 FLOPs 为 6.5%/35.6%/68.8%；KV Cache 0.5/4.0/16.0 GiB；线性激活 12/96/384 GiB；显式分数矩阵 32/2048/32768 GiB；交叉长度 59,391。70B（80 层）128K 为 40.0 GiB，405B（126 层）为 63.0 GiB。
- 打包等价性（6 篇文档，长度 9/12/7/15/11/8，56 个预测位置）：打包 + 块对角掩码与逐条前向的平均损失同为 2.8151，logits 最大偏差 8.9e-08、梯度最大偏差 2.9e-06；仅因果掩码时损失 2.8174、logits 偏差 1.5e-01。Token 利用率 32.3% → 96.9%。
- 长度外推（2 层、32 维、400 步、3 种子；随机水平 0.062）：固定 32 训练时 none/pi/ntk/yarn 在 T=128 的达标种子为 0/1/1/1，均值 0.10/0.39/0.37/0.46，标准差最高 0.51；混合长度训练为 0.97/0.98/0.94（3/3 达标，有效长度 128）；课程训练在 32/64/128 为 0.28/0.30/1.00（短上下文遗忘），课程 + PI 为 0.99/1.00/1.00。
- NIAH 口径自检（随机初始化）：四个深度桶准确率 0.031/0.062/0.047/0.094，接近 1/16。
- 测试：11 项全部通过（1.08 s）；示例在 CPU 上约 124 s，除耗时外逐字可复现。

### 留给后续章节的伏笔

- **第 14 章（SFT）**：本章的打包与边界标签 `-100` 直接变成对话数据的损失掩码与多轮拼接；长上下文 SFT 仍需保留短数据回放。
- **第 15 章（PEFT）**：长上下文下的激活与 KV 显存决定 LoRA/QLoRA 的可行配置；位置缩放与低秩适配可以叠加。
- **第 18 章（RLVR）**：长思维链把生成长度推到数万 Token，有效上下文与训练长度的区分同样适用；rollout 的 KV Cache 是本账本的推理侧。
- **第 20–22 章**：解码阶段的 KV Cache (13.4)、FlashAttention 的 $$O(T)$$ 显存 (13.5) 与投机解码的上下文管理，将把本章的账本落到推理系统。
- **第 24 章（测试时计算）**：更长的思考过程需要更长上下文，训练长度与有效长度的差异直接限制推理时扩展的收益。
- **第 27 章（评测方法论）**：NIAH、RULER、位置分桶、随机水平与样本量，是本章评测部分的展开。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、对比、代码、工业实践、论文、误区、小结、"本章的取舍"、习题齐全。
- [x] 推导无跳步：两类 FLOPs、交叉长度、KV/激活、长度课程、Token 利用率、块对角掩码、上下文并行通信、有效上下文、单针放置均有完整步骤。
- [x] 符号与附录 B 一致：新增 $$T^\star,\mathbf{1}_{\mathrm{scores}},L_0,L_N,L_i,N_{\mathrm{st}},T_{\max},N_{\mathrm{doc}},\eta_{\mathrm{pad}},\eta_{\mathrm{pack}},d_i,N_{cp},V_{\mathrm{cp}},T_{\mathrm{eff}},\mathcal{T},\epsilon,N_{\mathrm{probe}},u_i$$ 已登记；复用 $$b_{\mathrm{elem}},M_{\mathrm{kv}},c_a,M_{\mathrm{act}},M_{ij},\ell_i$$ 并标注首次出现；$$N_{\mathrm{st}}$$ 与 $$N$$ 参数量、$$N_{cp}$$ 与 $$N_p$$ 流水段数、$$N_{\mathrm{doc}}$$ 与 $$N_d$$ 数据并行度的易混点已在附录 B 说明。
- [x] 代码可运行：从 `part3/chapter13.md` 提取 6 个 Python 代码块，其中 3 个模块、1 个示例、1 个测试文件与 `/tmp/ch13run` 实测文件逐字一致，另 1 个为 4D 掩码片段；在 Python 3.13 + PyTorch 2.9 CPU 上运行，主程序输出与正文一致（耗时秒数随机器波动）。
- [x] 测试可运行：同一 Markdown 提取的 `tests/test_long_context.py` 通过 11 项测试（1.08 s）。
- [x] 公式与代码对应：代码注释标注 (13.1)–(13.8)、(13.10)、(13.11)；`block_causal_mask` 对应 (13.8)，`token_utilization` 对应 (13.7)，`length_schedule` 对应 (13.6)，`needle_positions` 对应 (13.11)。
- [x] GitBook 语法：hint 使用 info/warning 两类，1 个 mermaid、3 张表、1 个 details、5 篇论文、5 条误区；代码块均标注语言；公式 (13.1)–(13.11) 按出现顺序连续编号。
- [x] 篇幅：正文 8146 个中文字符，在其余章 6,000–9,000 字预算内，处于 7,500–8,500 目标区间。
- [x] 论文信息核对：Chen 2023（arXiv:2306.15595）、Peng 2023（arXiv:2309.00071）、Liu 2023（arXiv:2307.03172）、Hsieh 2024（arXiv:2404.06654）、Liu 2023（arXiv:2310.01889）的作者、年份、标题与编号均无误。
- [x] 事实性数字可复算：三张账本表、打包等价性、NIAH 自检与外推表均由程序输出；Llama 3.1 的六阶段 8K→128K、约 800B Token 与 RULER 的 13 项任务、有效长度结论来自公开报告并标注"截至写作时"。
- [x] 实验诚实报告：明确标注 2 层 32 维、400 步、单针检索的玩具实验不能外推到真实 LLM；PI/NTK/YaRN 的种子间双峰现象如实报告，不写成某一方法的普适优势；耗时随机器波动。

## 第 14 章交接

### 本章摘要

本章把 SFT 拆成掩码、数据、序列组织与合成四件事。损失上，掩码交叉熵 (14.1) 只监督助手回答与 EOS，Token 平均与序列平均 (14.2) 对长短样本的加权不同，监督比例 (14.3) 决定有效数据量。数据上，均匀标签噪声 (14.4) 只压平分布、由 (14.5) 保持 argmax；系统性噪声会改写条件分布，这解释了「随机加噪几乎不掉分、系统性错误直接崩掉」的对照。序列组织上，打包 (14.6) 用块对角掩码、边界 `-100` 与位置重置，与逐条前向精确等价。合成数据上，拒绝采样的接受率、pass@n 与期望采样量由 (14.7) 给出；提示异质性使经验 pass@n 显著低于平均率公式，弱模型还会带来多样性塌缩。遗忘控制用回放比例 (14.8) 描述，本章只给公式与工业经验，未单独跑实验。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (14.1) | $$\mathcal{L}_{\mathrm{SFT}}=-\frac{1}{\sum_t m_t}\sum_t m_t\log p_\theta(x_{t+1}\mid x_{\le t})$$ | 掩码交叉熵，只对监督 Token 求平均 |
| (14.2) | $$\mathcal{L}_{\mathrm{tok}}=\frac{\sum_{i,t}m_{i,t}\ell_{i,t}}{\sum_{i,t}m_{i,t}},\ \mathcal{L}_{\mathrm{seq}}=\frac{1}{B}\sum_i\frac{\sum_t m_{i,t}\ell_{i,t}}{\sum_t m_{i,t}}$$ | Token 平均与序列平均 |
| (14.3) | $$\rho_{\mathrm{sup}}=\frac{\sum_i n_i^{\mathrm{sup}}}{\sum_i T_i}$$ | 监督 Token 占比 |
| (14.4) | $$\tilde p(y\mid x)=(1-\epsilon)p(y\mid x)+\frac{\epsilon}{\lvert\mathcal{V}\rvert}$$ | 均匀标签噪声下的学得分布 |
| (14.5) | $$\tilde p(a\mid x)-\tilde p(j\mid x)=(1-\epsilon)[p(a\mid x)-p(j\mid x)]$$ | 均匀噪声保持排序、argmax 不变 |
| (14.6) | $$y_t^{\mathrm{pack}}=x_{t+1}\ (d_t=d_{t+1}),\ -100\ (d_t\neq d_{t+1})$$ | 打包的跨对话边界标签 |
| (14.7) | $$p_{\mathrm{acc}}=p,\ \mathrm{pass@}n=1-(1-p)^n,\ \mathbb{E}[\text{draws}]=k/p$$ | 拒绝采样的接受率、通过率与期望采样量 |
| (14.8) | $$r_{\mathrm{replay}}=\frac{N_{\mathrm{PT}}}{N_{\mathrm{PT}}+N_{\mathrm{SFT}}},\ \lambda=\frac{r_{\mathrm{replay}}}{1-r_{\mathrm{replay}}}$$ | 回放比例与损失权重 |

### 实验与关键数字

- 掩码对照（256 条干净数据，250 步，3 种子；随机水平 0.125）：只算回答的 ID 精确 0.494±0.152、ID Token 0.864±0.050、长度 OOD Token 0.134±0.006、模板 OOD Token 0.470±0.058；全序列为 0.433±0.079、0.859±0.026、0.114±0.051、0.439±0.032。只算回答方向性更好但差异未超出种子间波动。
- 数据质量与数量（600 步，3 种子）：clean-64 的 ID 精确 0.179±0.023、clean-256 为 0.760±0.233、rand-50%-1024 为 0.798±0.204、sys-50%-1024 为 0.159±0.054、sys-25%-1024 为 0.554±0.193。随机噪声与干净 256 条打平，系统性噪声使精确匹配崩溃。
- 打包等价性（6 条对话）：逐条与打包损失同为 82.2033，监督 Token 同为 30，logits 最大偏差 6.0e-08；监督比例 0.417，平均回答长度 5.0。
- 拒绝采样（弱模型，64 个提示 × 8 条）：平均接受率 0.176，经验 pass@8 为 0.281，平均率公式给出 0.787；接受 90 条、去重后 17 条；凑够 64 条期望采样 364 条。
- 测试：14 项全部通过（约 1 s）；示例在 CPU 上约 75 s，除耗时外逐字可复现。

### 留给后续章节的伏笔

- **第 15 章（PEFT）**：本章的模板、掩码、打包与评测口径可直接复用；LoRA/QLoRA 只更新少量参数，能降低全参数 SFT 的显存与遗忘。
- **第 16–17 章（RLHF/DPO）**：SFT 是对齐流程的第一阶段；本章的拒绝采样与偏好数据构造、KL 约束下的策略优化直接衔接。
- **第 18 章（RLVR）**：拒绝采样中的验证器在可验证奖励任务上推广为程序化奖励；SFT 是 RL 冷启动的常见初始化。
- **第 19 章（蒸馏）**：合成数据与拒绝采样是蒸馏的数据侧；本章的多样性塌缩与选择偏差提醒蒸馏数据同样会收窄分布。
- **第 20 章（解码）**：EOS 监督决定模型能否停止；温度采样与 pass@n 的评测口径在本章已出现。
- **第 27 章（评测）**：ID/OOD、随机水平、人工评测与 LLM-as-judge 是本章评测部分的展开。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、对比、代码、工业实践、论文、误区、小结、"本章的取舍"、习题齐全。
- [x] 字数：正文 8481 个中文字符（剔除代码块与公式），在其余章 6,000–9,000 字预算内，处于 7,500–8,500 目标区间。
- [x] 公式编号 (14.1)–(14.8) 连续；推导无跳步：掩码损失、两种平均、监督比例、噪声混合与 argmax、打包边界、拒绝采样与 Jensen、回放权重均有完整步骤。
- [x] 至少 1 张对比表（表 14-1，6 个决策）、工业案例（InstructGPT、Llama 2、LIMA、Self-Instruct/STaR）、5 篇论文、5 条误区。
- [x] 代码可运行：从 `part4/chapter14.md` 提取 4 个 Python 代码块，与 `/tmp/ch14run` 实测文件逐字一致；在 Python 3.13 + PyTorch 2.9 CPU 上运行，主程序输出与正文一致（耗时秒数随机器波动）。
- [x] 测试可运行：同一 Markdown 提取的 `tests/test_sft.py` 通过 14 项测试（约 1 s）。
- [x] 公式与代码对应：注释标注 (14.1)、(14.2)、(14.3)、(14.4)、(14.6)、(14.7)；`masked_cross_entropy` 对应 (14.1)，两种平均对应 (14.2)，`supervision_stats`/`count_supervised` 对应 (14.3)，`label_noise_mixture` 对应 (14.4)，`pack_sft_examples` 边界对应 (14.6)，拒绝采样三函数对应 (14.7)。
- [x] 新符号已登记 `notation.md` 与附录 B：$$m_t,\mathcal{V},\rho_{\mathrm{sup}},n_i^{\mathrm{sup}},\epsilon,\tilde p,p_{\mathrm{acc}},n,k,N_{\mathrm{PT}},N_{\mathrm{SFT}},r_{\mathrm{replay}},\lambda,y_t^{\mathrm{pack}}$$ 等，并说明与既有符号的冲突。
- [x] 新术语已登记 `glossary.md` 与附录 C：监督比例、Token 平均、序列平均、系统性标签噪声、多样性塌缩、回放、表面对齐等。
- [x] GitBook 语法：hint 使用 info/warning 两类；1 个 mermaid、1 张对比表、1 个 details、5 篇论文、5 条误区；代码块均标注语言；无落单 `$`。
- [x] 事实性数字可复算：四组实验数字均由程序输出；InstructGPT 的 13,000 条 SFT 提示、Llama 2 的约 27,540 条标注与 LIMA 的约 1,000 条示例来自公开论文并标注"截至写作时"。
- [x] 实验诚实报告：明确标注 2 层 32 维、250–600 步玩具任务不能外推到真实 LLM；掩码差异未超出种子波动、长度 OOD 接近随机、遗忘未单独实验等边界均如实写明。
- [x] 接口未破坏：`TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有签名未改；`SFTConfig` 仅作训练脚本参数。

## 第 15 章交接

### 本章摘要

本章把参数高效微调拆成低秩更新、目标模块、显存账本与量化四件事。LoRA 冻结基座 $$W_0$$，用 $$(\alpha/r)BA$$ 参数化增量，(15.1) 是前向，(15.2) 是合并，(15.3) 是单模块参数量。Eckart–Young (15.4) 只说明「最优低秩逼近权重」的误差，不能直接证明 LoRA 有效；真正要问的是任务更新 $$\Delta W$$ 是否近似低秩。缩放系数 $$\alpha/r$$ 与零初始化 $$B$$ 决定有效步长和训练起点；第一次反向时 $$\partial\mathcal{L}/\partial A=0$$，训练先移动 $$B$$。显存上，静态状态 (15.5) 把可训练参数按 16 字节、冻结参数按 4 字节记账；QLoRA 把冻结权重压到 4 比特，用 (15.6) 算得块大小 64、双重量化下约 4.127 比特。Adapter、Prefix、Prompt 的参数量由 (15.7) 给出，分别以串行层、上下文前缀与软提示换取模块化。实验在「加二」预训练基座上适配「加一」，5 个种子对照全参数、LoRA 秩与目标模块、QLoRA 及随机基座边界。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (15.1) | $$h=W_0x+\frac{\alpha}{r}BAx$$ | LoRA 前向，$$A\in\mathbb{R}^{r\times d_{in}}$$、$$B\in\mathbb{R}^{d_{out}\times r}$$ |
| (15.2) | $$W=W_0+\frac{\alpha}{r}BA$$ | 合并后的等价权重 |
| (15.3) | $$P_{\mathrm{LoRA}}=r(d_{in}+d_{out})$$ | 单模块适配器参数量 |
| (15.4) | $$\min_{\mathrm{rank}(\hat W)\le r}\lVert W-\hat W\rVert_F^2=\sum_{i=r+1}^{\min(d_{out},d_{in})}\sigma_i^2$$ | Eckart–Young 低秩近似误差 |
| (15.5) | $$M_{\mathrm{state}}=16P_{\mathrm{train}}+4P_{\mathrm{frozen}}$$ | fp32 AdamW 静态状态字节数 |
| (15.6) | $$\text{bits/param}=4+\frac{8}{b}+\frac{32}{256b}$$ | NF4 + 双重量化的每参数比特数 |
| (15.7) | $$P_{\mathrm{adapter}}=2dm+m+d,\ P_{\mathrm{prefix}}=2L\ell d,\ P_{\mathrm{prompt}}=\ell d$$ | Adapter/Prefix/Prompt 参数量 |

### 实验与关键数字

- 设计：预训练 1,024 条「加二」对话、全序列损失、1,200 步；SFT 256 条「加一」对话、只算回答、600 步；5 个模型种子；全参数学习率 $$3\times10^{-3}$$，LoRA/QLoRA 为 $$10^{-2}$$，$$\alpha=2r$$；评测 200 条长度 3–5，随机水平 0.125。
- 零样本基座（会加二、不会加一）：ID 精确 0.000±0.000，Token 0.209±0.010。
- 全参数：33,952 可训练（100%）、静态 530 KB、ID 精确 0.868±0.119、Token 0.972±0.025。
- LoRA q/v：$$r=1,2,4,8$$ 的 ID 精确依次为 0.192±0.222、0.510±0.393、0.747±0.252、0.775±0.282；可训练参数 256/512/1,024/2,048，静态 137/141/149/165 KB。
- LoRA 全线性 $$r=4$$：5,888（14.8%，分母含适配器）、静态 225 KB、ID 精确 0.911±0.070、Token 0.982±0.015；QLoRA 全线性 $$r=4$$：5,888、113 KB、0.869±0.120、0.972±0.025。
- 随机基座：全参数 0.632±0.104；LoRA 全线性 $$r=4$$ 仅 0.226±0.042，说明 PEFT 不能替代预训练。
- 激活（32 条、长度 3–5，反向保存张量）：全参数 3,355 KB、q/v 2,473 KB、全线性 3,289 KB、QLoRA 3,289 KB；QLoRA 不省激活，缩小适配范围才会因冻结 MLP 少保存输入激活。
- 合并：前后 logits 最大偏差 2.0e-06，撤销后 1.9e-06；14 个线性层被替换，适配器 5,888 参数。
- NF4：码本 16 级、块 64；基座权重相对误差 0.0971，logits 相对偏差 0.0635，存储 4.139 比特/参数。
- 测试：10 项全部通过（1.20 s）；示例 CPU 约 225 s，除耗时外逐字可复现。

### 留给后续章节的伏笔

- **第 16–17 章（RLHF/DPO）**：策略模型可用 LoRA 适配，参考模型保持冻结；第 15 章的适配器保存/切换与显存账本直接复用，尤其是「一份基座 + 多个策略」的服务方式。
- **第 18 章（RLVR）**：可验证奖励训练常从 SFT 模型出发，LoRA 可降低多轮实验的切换成本；但低秩容量是否足够支撑推理能力提升，需要用本章的随机基座边界来警惕。
- **第 19 章（蒸馏）**：适配器可作为轻量学生或任务头；本章的参数账本可用于比较「小模型全参数」与「大模型 + 适配器」的部署成本。
- **第 21 章（推理加速）**：多 LoRA 服务、合并与未合并分支的吞吐差异是调度问题；本章的合并等价性与多适配器讨论是前置。
- **第 23 章（量化）**：NF4 是训练侧的 4 比特方案，推理侧还有 GPTQ/AWQ 等权重量化；本章的「存储 4 比特、计算 bf16」区分必须延续。
- **第 13 章（长上下文）**：本章实测说明 PEFT 主要省静态状态，不省激活；长上下文训练的显存瓶颈仍应回到批大小、序列长度与并行策略。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、对比、代码、工业实践、论文、误区、小结、"本章的取舍"、习题齐全。
- [x] 字数：正文 7799 个中文字符（剔除代码块与公式），在其余章 6,000–9,000 字预算内，处于 7,500–8,500 目标区间。
- [x] 公式编号 (15.1)–(15.7) 连续；推导无跳步：LoRA 前向与合并、参数量、Eckart–Young、零初始化梯度、静态状态、NF4 比特数、Adapter/Prefix/Prompt 参数量均有完整步骤。
- [x] 至少 1 张对比表（表 15-1，6 种方法）、工业案例（LoRA GPT-3 175B、QLoRA 65B、vLLM 多 LoRA）、5 篇论文、5 条误区。
- [x] 代码可运行：从 `part4/chapter15.md` 提取 3 个 Python 代码块，与 `/tmp/ch15run` 实测文件逐字一致；在 `/tmp/ch15verify` 干净目录重跑，主程序输出与正文一致（耗时秒数随机器波动）。
- [x] 测试可运行：同一 Markdown 提取的 `tests/test_lora.py` 通过 10 项测试（1.20 s）。
- [x] 公式与代码对应：注释标注 (15.1)、(15.2)、(15.3)、(15.6)；`LoRALinear.forward` 对应 (15.1)，`merge` 对应 (15.2)，`lora_module_params` 对应 (15.3)，`nf4_storage_bytes` 对应 (15.6)。
- [x] 新符号已登记 `notation.md` 与附录 B：$$W_0,\Delta W,A,B,r,\alpha,d_{in},d_{out},P_{\mathrm{LoRA}},P_{\mathrm{train}},P_{\mathrm{frozen}},M_{\mathrm{state}},b,\sigma_i,\ell,m$$ 等，并说明与既有符号的冲突。
- [x] 新术语已登记 `glossary.md` 与附录 C：参数高效微调、低秩适配、秩、缩放系数、合并、QLoRA、NF4、双重量化、分页优化器、Adapter、Prefix-Tuning、Prompt Tuning、DoRA 等。
- [x] GitBook 语法：hint 使用 info/warning 两类；1 个 mermaid、1 张对比表、1 个 details、5 篇论文、5 条误区；代码块均标注语言；无落单 `$`。
- [x] 事实性数字可复算：全部实验数字由程序输出；LoRA 的 10,000× 参数缩减、3× 显存下降、25% 加速与 QLoRA 的 65B/48GB、99.3% 均来自原论文并标注"截至写作时"。
- [x] 实验诚实报告：明确标注 2 层 32 维、600 步、5 种子玩具任务不能外推；适配器占比 14.8% 远高于真实模型约 1%，QLoRA 模拟只复现数值误差不省真实显存，均如实写明。
- [x] 接口未破坏：`TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有签名未改；`LoraConfig` 只作适配器配置，默认路径与第 14 章逐位一致。

## 第 16 章交接

### 本章摘要

本章把 RLHF 拆成偏好建模、带 KL 约束的策略优化与工程实现三部分。奖励模型用 Bradley–Terry 把成对比较变成 logistic 回归，(16.1) 是偏好概率，(16.2) 是负对数似然；其梯度说明 RM 只在「奖励差」上学习，绝对尺度不可识别，因此需要标准化与校准。KL 正则目标 (16.3) 有闭式最优解 (16.4)：参考策略按奖励的指数重新加权；把 (16.4) 反解并代回 (16.2) 即得 DPO 的出发点，而「完成平方」改写说明最大化 (16.3) 等价于向 $$\pi^\star$$ 投影。PPO 用逐 Token 塑形奖励 (16.5) 实现 KL 约束，用重要性比 (16.6) 与裁剪目标 (16.7) 控制步长，用 GAE (16.8) 估计优势、用 (16.9) 联合训练价值网络；k1 (16.10) 与 k3 (16.11) 是 KL 的两种估计量，前者无偏、方差大且适合做塑形奖励，后者逐 Token 非负、适合监控。实验在 4 主题、16 词表的玩具对话上跑通 SFT→RM→PPO，用三档 $$\beta$$ 与 Best-of-8 对照，并用长度、正确性、干扰词三个探针测量奖励作弊。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (16.1) | $$P(y_w\succ y_l\mid x)=\sigma(r(x,y_w)-r(x,y_l))$$ | Bradley–Terry 偏好概率 |
| (16.2) | $$\mathcal{L}_{\rm RM}(\phi)=-\mathbb{E}[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))]$$ | 奖励模型负对数似然 |
| (16.3) | $$\max_{\pi_\theta}\mathbb{E}_{x}\big[\mathbb{E}_{y\sim\pi_\theta}r_\phi-\beta\,\mathrm{KL}(\pi_\theta\parallel\pi_{\rm ref})\big]$$ | KL 正则 RLHF 目标 |
| (16.4) | $$\pi^\star(y\mid x)=\frac{1}{Z(x)}\pi_{\rm ref}(y\mid x)\exp\!\big(r_\phi(x,y)/\beta\big)$$ | 闭式最优策略 |
| (16.5) | $$r_t=-\beta(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\rm ref}(a_t\mid s_t))$$ | 逐 Token KL 塑形奖励 |
| (16.6) | $$\rho_t=\pi_\theta(a_t\mid s_t)/\pi_{\theta_{\rm old}}(a_t\mid s_t)$$ | PPO 重要性比 |
| (16.7) | $$\mathcal{L}^{\rm CLIP}_t=\min(\rho_tA_t,\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)$$ | 裁剪代理目标 |
| (16.8) | $$A_t=\sum_l(\gamma\lambda)^l\delta_{t+l},\ \delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$$ | 广义优势估计 |
| (16.9) | $$\mathcal{L}=-\frac{1}{\sum m_t}\sum m_t\mathcal{L}^{\rm CLIP}_t+c_1\frac{1}{\sum m_t}\sum m_t(V_\phi-\hat R_t)^2$$ | 策略与价值联合损失 |
| (16.10) | $$\widehat{\mathrm{KL}}_1=\frac{1}{\sum m_t}\sum m_t(\log\pi_\theta-\log\pi_{\rm ref})$$ | k1 KL 估计量 |
| (16.11) | $$\widehat{\mathrm{KL}}_3=\frac{1}{\sum m_t}\sum m_t(\rho_t^{\rm ref}-\log\rho_t^{\rm ref}-1)$$ | k3 KL 估计量 |

### 实验与关键数字

- 设计：16 词表玩具对话，PAD/BOS/USER/ASSISTANT/EOS = 0/1/2/3/4；主题 5–8，目标词 8–11，形近干扰词 12–15；SFT 行为策略 55% 目标词、10% 干扰词、35% 均匀采样，回答长度 3–5；RM 用 2,000 对训练、400 对评测，400 步、批 64、lr $$10^{-3}$$；PPO 50 轮、批 48、2 epoch、minibatch 16，3 个种子；评测 256 条。
- 「人类」奖励 = 目标词比例 + 0.30×干扰词比例 + 0.35×(长度−3)/3 + 高斯噪声；它故意把干扰词与长度当成优点，用来观察 RM 与 PPO 是否放大偏置。
- RM：成对偏好准确率 0.897/0.890/0.915（种子 0/1/2）；长度探针 1.000、正确性探针 1.000、干扰词探针 0.000，即 RM 学会了长度与正确性，但没有学走干扰词偏好。
- SFT 策略：目标命中 0.591±0.015、干扰词 0.142±0.006、长度 3.853±0.096、RM 分数 −0.442±0.880。
- PPO $$\beta=0.10$$：0.972±0.020 / 0.004±0.003 / 4.987±0.008 / 7.027±0.703；$$\beta=0.50$$：0.949±0.018 / 0.022±0.008 / 4.438±0.091 / 6.591±0.833；$$\beta=2.00$$：0.684±0.008 / 0.119±0.010 / 3.997±0.106 / 1.604±0.759。
- Best-of-8：0.905±0.001 / 0.049±0.005 / 4.181±0.102 / 5.777±0.757；说明在线 PPO 在三档 β 下都优于同预算的纯推理时筛选，但 β 过小会把长度推到上限 5。
- 逐 Token log 比（k1 批平均）：β=0.10 为 0.742±0.110，β=0.50 为 0.372±0.051，β=2.00 为 0.033±0.011；与长度、正确率的单调关系构成本章 KL–奖励前沿。
- 测试：13 项全部通过（1.11 s）；示例 CPU 约 65 s，除耗时外逐字可复现。

### 留给后续章节的伏笔

- **第 17 章（DPO）**：把 (16.4) 反解 $$r_\phi=\beta\log(\pi^\star/\pi_{\rm ref})+\beta\log Z$$ 并代入 (16.2)，即可消掉奖励模型与在线采样；本章的推导与「Z(x) 只依赖 x、比较时抵消」是 DPO 的直接前置。DPO 的 β 与本章的 β 是同一个符号、同一个含义。
- **第 18 章（RLVR）**：把可验证奖励（答案对错、单测通过）替换本章的 RM 分数，rollout/GAE/裁剪管线不变；GRPO 去掉价值网络、用组内均值作基线，是本章 (16.9) 的变体。第 15 章的 LoRA 可用于策略侧，参考模型保持冻结。
- **第 19 章（蒸馏）**：RLHF 后的策略常作为教师，把「对齐后」的分布蒸馏回小模型；本章的 KL 与重要性比可用于分析教师—学生差距。
- **第 13 章（长上下文）**：RM 只读最后一个 Token、PPO 的 rollout 长度与显存、参考模型的额外前向，都是长上下文推理成本的来源；本章的四模型账本需要与第 13 章的 KV Cache 账本合看。
- **第 15 章（LoRA）**：策略侧可用 LoRA，参考模型冻结共享基座；「一份基座 + 两份适配器」是工业降低 RLHF 显存的标准做法。
- **第 21 章（推理加速）**：rollout 吞吐是 PPO 的主要瓶颈，vLLM/PagedAttention 与本章的采样一致性约束（不截断词表、不强制 EOS）必须同时满足，否则 log 概率与采样分布不一致。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、对比、代码、工业实践、论文、误区、小结、「本章的取舍」、习题齐全。
- [x] 字数：正文 11090 个中文字符（剔除代码块与公式），在核心章 10,000–12,000 预算内。
- [x] 公式编号 (16.1)–(16.11) 连续；推导无跳步：BT 梯度、闭式最优与完成平方、重要性采样恒等式、GAE 手算、k1/k3 性质均有完整步骤。
- [x] 至少 1 张对比表（表 16-1，PPO/Best-of-N/拒绝采样微调/DPO 四行）、工业案例（InstructGPT、Anthropic HH、Llama 2）、5 篇论文、5 条误区。
- [x] 代码可运行：从 `part4/chapter16.md` 提取 4 个 Python 代码块，与 `/tmp/ch16verify` 实测文件逐字一致；输出与 `ch16_output.txt` 一致（耗时秒数随机器波动）。
- [x] 测试可运行：同一 Markdown 提取的 `tests/test_rlhf.py` 通过 13 项测试（1.11 s）。
- [x] 公式与代码对应：注释标注 (16.1)、(16.6)、(16.7)、(16.9)、(16.10)、(16.11)；`bradley_terry_loss` 对应 (16.1)/(16.2)，`clipped_surrogate` 对应 (16.7)，`compute_gae` 对应 (16.8)，`kl_penalty`/`kl_estimator_k3` 对应 (16.10)/(16.11)。
- [x] 新符号已登记 `notation.md` 与附录 B：$$x,y_w,y_l,r_\phi,\phi,\pi_\theta,\pi_{\rm ref},\pi_{\rm old},\pi^\star,\beta,Z(x),\rho_t,\epsilon,A_t,\delta_t,V_\phi,\hat R_t,c_1,\widehat{\mathrm{KL}}_1,\widehat{\mathrm{KL}}_3,m_t$$ 等，并说明与第 7 章 $$\alpha,\beta$$ 的冲突。
- [x] 新术语已登记 `glossary.md` 与附录 C：RLHF、奖励模型、Bradley–Terry、偏好数据、策略梯度、优势、GAE、PPO、重要性比、裁剪目标、参考模型、KL 塑形奖励、价值网络、奖励作弊、长度偏置、校准、Best-of-N、拒绝采样微调、k1/k3 估计量、奖励集成等。
- [x] GitBook 语法：hint 使用 info/warning/danger 三类；1 个 mermaid、1 张对比表、1 个 details、5 篇论文、5 条误区；代码块均标注语言；无落单 `$`。
- [x] 事实性数字可复算：全部实验数字由程序输出；InstructGPT 的 33k 偏好对、Llama 2 的约 100 万条偏好数据等来自公开论文并标注「截至写作时」。
- [x] 实验诚实报告：明确标注 2 层 32 维、16 词表、长度上限 5、50 轮 PPO 的玩具任务不能外推；RM 未学走干扰词偏置、Best-of-8 在特定预算下的比较边界均如实写明。
- [x] 接口未破坏：`TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有签名未改；`ScalarModel` 通过替换 `lm_head` 注入，`RLConfig` 只作训练脚本参数。

## 第 17 章交接

### 本章摘要

本章从第 16 章式 (16.4) 的 KL 正则闭式解出发，把奖励反解为“策略与参考模型的对数比乘以 β 再加上提示级常数” (17.1)；在成对比较中该常数抵消，于是定义隐式奖励 (17.2)。把隐式奖励代回 Bradley–Terry 偏好概率得到 (17.3)，其负对数似然就是 DPO 损失 (17.4)。对损失求导得到 (17.5)：梯度权重为 $$\sigma(-m)$$，只推高赢家、压低输家，参考模型只通过基准项影响权重。随后讨论 β 同时缩放 logit 与梯度带来的有效步长耦合、长度归一化 (17.6) 对奖励单位的改变，以及 IPO (17.7)、KTO、SimPO、ORPO 在“目标间隔、参考模型、长度归一化、数据形式”四个维度上的取舍。工程部分说明参考对数概率预计算、LoRA DPO 共享基座、偏好对按长度分批，以及标签去偏、长度匹配、位置随机化、标注者校准与去污染。实验用偏置与去偏两套偏好数据做 3 种子对照，验证“标签质量比损失函数更重要”，并展示 β 与 KL 的非单调关系、IPO 的保守性与偏好准确率不等于任务质量。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (17.1) | $$r(x,y)=\beta\log\frac{\pi^\star(y\mid x)}{\pi_{\rm ref}(y\mid x)}+\beta\log Z(x)$$ | 由闭式最优策略反解奖励 |
| (17.2) | $$r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\rm ref}(y\mid x)}$$ | 隐式奖励，省略提示级常数 |
| (17.3) | $$P_\theta(y_w\succ y_l\mid x)=\sigma\!\big(\beta[\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\rm ref}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\rm ref}(y_l\mid x)}]\big)$$ | 隐式奖励下的 Bradley–Terry 概率 |
| (17.4) | $$\mathcal{L}_{\rm DPO}(\theta)=-\mathbb{E}_{D_{\rm pref}}\big[\log\sigma(\beta\Delta_\theta(x,y_w,y_l))\big]$$ | DPO 成对损失 |
| (17.5) | $$\nabla_\theta\mathcal{L}_{\rm DPO}=-\beta\sigma(-m)\big[\nabla_\theta\log\pi_\theta(y_w\mid x)-\nabla_\theta\log\pi_\theta(y_l\mid x)\big]$$ | DPO 梯度，$$m=\beta\Delta_\theta$$ |
| (17.6) | $$\Delta^{\rm norm}_\theta=\frac{1}{\lvert y_w\rvert}\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\rm ref}(y_w\mid x)}-\frac{1}{\lvert y_l\rvert}\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\rm ref}(y_l\mid x)}$$ | 长度归一化对数比 |
| (17.7) | $$\mathcal{L}_{\rm IPO}(\theta)=\mathbb{E}\big[(\Delta_\theta(x,y_w,y_l)-\frac{1}{2\beta})^2\big]$$ | IPO 的平方损失与目标间隔 |

### 实验与关键数字

- 设计：沿用第 16 章的 16 词表玩具对话；提示 4 Token（BOS/USER/主题/ASSISTANT），回答 3–5 Token；4 个主题各有目标词 8–11 与形近干扰词 12–15；SFT 行为策略 55% 目标词、10% 干扰词、35% 均匀采样；SFT 512 条、400 步；偏好对 2,000 条、评测 400 条；DPO 300 步、批 32、lr $$5\times10^{-4}$$、3 个种子；评测采样 256 条。
- 偏置奖励 = 目标词比例 + 0.30×干扰词比例 + 0.35×(长度−3)/3 + 噪声；去偏奖励只取目标词比例 + 噪声。两者用同一回答生成器，唯一差别是标签。
- SFT 基线：正确率 0.581±0.019、干扰词 0.143±0.007、长度 3.977±0.118、KL 0。
- DPO β=0.10：0.858±0.097 / 0.007±0.003 / 4.438±0.257 / 1.221±0.638；β=0.50：0.957±0.009 / 0.016±0.002 / 4.768±0.108 / 0.574±0.049；β=2.00：0.929±0.004 / 0.034±0.003 / 4.686±0.033 / 0.359±0.032。
- IPO β=0.50：0.696±0.020 / 0.115±0.009 / 4.086±0.127 / 0.027±0.005；去偏 DPO β=0.50：0.980±0.001 / 0.004±0.001 / 4.357±0.118 / 0.558±0.072。
- 留出偏好对（偏置标签）排序准确率/间隔：β=0.10 为 0.849/1.044；β=0.50 为 0.895/3.018；β=2.00 为 0.912/6.540；IPO 为 0.873/0.299；去偏 DPO 为 0.839/2.729。去偏标签上，去偏 DPO 为 0.923/3.045。
- 关键观察：偏置标签把长度从 3.977 推到 4.768，去偏标签只到 4.357；去偏 DPO 的任务正确率更高（0.980 对 0.957），但在偏置评测集上的排序准确率更低（0.839 对 0.895）。KL 不随 β 单调（β=2.00 的 0.359 低于 β=0.50 的 0.574）。
- 测试：12 项全部通过（约 1.2 s）；示例在 CPU 上约 102 s，除耗时外逐字可复现。

### 留给后续章节的伏笔

- **第 18 章（RLVR/GRPO）**：DPO 需要成对偏好数据；当奖励可由规则验证（数学答案、单测、工具返回）时，可去掉人工偏好，直接用可验证奖励做在线优化。本章的“离线数据决定终点”边界正是 RLVR 要突破的地方；GRPO 的组内基线可看作去掉价值网络后的优势估计，与第 16 章 (16.8)/(16.9) 对照阅读。
- **第 15 章（LoRA）**：DPO 的策略与参考模型可共享一份冻结基座，各挂一份适配器；本章“一份基座 + 两份适配器”的显存账本直接复用第 15 章结论。参考对数概率预计算与适配器切换结合，可在单卡上做多组 β 扫描。
- **第 19 章（蒸馏）**：DPO 后的策略常作为教师；隐式奖励与对数比可用于诊断教师—学生分布差距，但蒸馏的目标是匹配分布而不是排序，两者不能混用。
- **第 21 章（推理加速）**：DPO 训练不需要 rollout，但评测需要采样；采样实现必须与训练时的对数概率一致，否则离线偏好与在线指标会出现系统性偏差。
- **第 13 章（长上下文）**：序列对数概率随长度增长，长度归一化与批内长度对齐是长回答 DPO 的必要工程；长上下文下参考模型前向的显存与 KV Cache 账本需与本章的“四次前向”合看。
- **第 16 章（PPO）**：DPO 与 PPO 共享同一个 KL 正则闭式解；本章的 β 与第 16 章的 β 同符号同含义，但 DPO 的 β 还进入梯度尺度，比较两章的实验数字时必须同时看学习率。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、对比、代码、工业实践、论文、误区、小结、「本章的取舍」、习题齐全。
- [x] 字数：正文 10412 个中文字符（剔除代码块与公式），在核心章 10,000–12,000 预算内。
- [x] 公式编号 (17.1)–(17.7) 连续；推导无跳步：闭式解反解、配分函数抵消、BT 代入、DPO 梯度、长度归一化、IPO 目标均有完整步骤，并指出“把奖励代回原目标”的循环代入错误。
- [x] 至少 1 张对比表（表 17-1，PPO/DPO/Best-of-N/拒绝采样/迭代 DPO 五行）、工业案例（Zephyr-7B、Tulu 2、Llama 3）、5 篇主论文 + 2 篇延伸、5 条误区。
- [x] 代码可运行：从 `part4/chapter17.md` 提取 3 个 Python 代码块，与 `/tmp/ch17verify` 实测文件逐字一致；输出与 `ch17_output.txt` 一致（耗时秒数随机器波动）。
- [x] 测试可运行：同一 Markdown 提取的 `tests/test_dpo.py` 通过 12 项测试（约 1.2 s）。
- [x] 公式与代码对应：`implicit_reward` 标注 (17.2)，`log_ratio_margin` 标注 (17.3)，`dpo_loss` 标注 (17.4)，长度归一化标注 (17.6)，IPO 分支标注 (17.7)。
- [x] 新符号已登记 `notation.md` 与附录 B：$$x,y_w,y_l,D_{\rm pref},r_\theta,\pi_\theta,\pi_{\rm ref},\beta,\Delta_\theta,m,Z(x),|y|,\Delta^{\rm norm}_\theta,\eta$$ 等，并说明与第 14–16 章符号的关系。
- [x] 新术语已登记 `glossary.md` 与附录 C：直接偏好优化、隐式奖励、DPO 损失、IPO、KTO、SimPO、ORPO、cDPO、似然位移、参考模型失配、长度归一化、离线/在线偏好优化、迭代 DPO、拒绝采样微调等。
- [x] GitBook 语法：1 个 mermaid、1 张对比表、1 个 details、5 篇论文、5 条误区；代码块均标注语言；无落单 `$`。
- [x] 事实性数字可复算：全部实验数字由程序输出；Zephyr-7B 的 β=0.1、lr $$5\times10^{-7}$$、MT-Bench 7.34，Tulu 2 的数据规模与 Llama 3 的 SFT→拒绝采样→DPO 顺序来自公开报告并标注「截至写作时」。
- [x] 实验诚实报告：明确标注 2 层 32 维、16 词表、长度上限 5、手工奖励生成偏好的玩具任务不能外推；偏好准确率与任务质量的冲突、β 与 KL 的非单调关系均如实写明。
- [x] 接口未破坏：`TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有签名未改；`DPOConfig` 只作训练脚本参数，不进入 `ModelConfig`；参考模型仍为独立冻结副本。

## 第 18 章交接

### 本章摘要

本章把第 16–17 章的偏好奖励换成规则可验证奖励，先说明 RLVR 的数据链——只需题目，回答在线采样，奖励由答案匹配、单元测试或工具返回即时判定；再推导 GRPO。由带基线的策略梯度 (18.3) 出发，用同题组内均值作基线得到优势 (18.4)，指出包含自身的均值带来 $$(1-1/G)$$ 收缩 (18.5)，留一基线 (18.6) 无偏。把 PPO 的重要性比与裁剪用于组采样，得到逐 Token 代理目标 (18.8)、完整目标 (18.9) 与 k3 KL (18.10)；比较 Token 级与序列级归一化 (18.11)，并用混合组概率 (18.12) 解释全对/全错组的零梯度与 $$G=1$$ 的结构性失效。工程部分拆解 DeepSeekMath、DeepSeek-R1-Zero/R1、DAPO、Kimi k1.5 与 Tulu 3，讨论长思维链、格式奖励、动态采样、长度塑形与验证器风险。实验在模 10 加 3 任务上做 3 种子 G 消融：$$G=1$$ 与 SFT 逐位相同，$$G=8$$ 贪心正确率均值 0.83，$$G=4$$ 为 0.33。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (18.1) | $$r_i=\mathbb{1}[\operatorname{Verify}(q,o_i,y^\star)=1]$$ | 可验证奖励 |
| (18.2) | $$\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\rm old}}(\cdot\mid q)$$ | 每提示采样 G 条回答 |
| (18.3) | $$\nabla_\theta J=\mathbb{E}[(r(q,o)-b(q))\nabla_\theta\log\pi_\theta(o\mid q)]$$ | 带基线的策略梯度 |
| (18.4) | $$\hat A_i=(r_i-\bar r)/(\sigma_r+\epsilon)$$ | 组内优势与可选标准化 |
| (18.5) | $$\mathbb{E}[r_i-\bar r_i]=(1-1/G)(r_i-\mu)$$ | 包含自身的组均值收缩 |
| (18.6) | $$b_i^{(-i)}=\frac{1}{G-1}\sum_{j\ne i}r_j$$ | 留一基线，无偏 |
| (18.7) | $$\rho_{i,t}=\pi_\theta(o_{i,t}\mid q,o_{i,<t})/\pi_{\theta_{\rm old}}(o_{i,t}\mid q,o_{i,<t})$$ | 逐 Token 重要性比 |
| (18.8) | $$\ell_{i,t}=\min(\rho_{i,t}\hat A_i,\operatorname{clip}(\rho_{i,t},1-\epsilon_l,1+\epsilon_h)\hat A_i)$$ | 裁剪代理目标 |
| (18.9) | $$J_{\rm GRPO}=\mathbb{E}\big[\frac1G\sum_i\frac{1}{\lvert o_i\rvert}\sum_t(\ell_{i,t}-\beta\hat D_{i,t})\big]$$ | GRPO 完整目标 |
| (18.10) | $$\hat D_{i,t}=x-\log x-1,\ x=\pi_{\rm ref}/\pi_\theta$$ | k3 KL 估计量 |
| (18.11) | $$J_{\rm tok}=\frac{1}{\sum_i\lvert o_i\rvert}\sum_i\sum_t(\ell_{i,t}-\beta\hat D_{i,t})$$ | Token 级归一化 |
| (18.12) | $$P_{\rm mixed}=1-p^G-(1-p)^G$$ | 混合组概率 |

### 实验与关键数字

- 设计：16 词表、`d_model=32`、2 层、4 头、RoPE/RMSNorm/SwiGLU；提示 4 Token（BOS/Q/数字/SEP），回答 2 Token（答案/EOS）；任务为 `(a+3) mod 10`，随机猜中率 0.10；SFT 仅 50 步、30% 正确标签，故意留出 RL 空间；GRPO 300 轮、批 16 提示、lr $$10^{-3}$$、1 个 epoch、温度 1.0、$$\beta=0$$。
- SFT 基线：贪心 0.10/0.10/0.10，pass@1 均值 0.05，pass@8 均值 0.33。
- $$G=1$$：贪心 0.10、0.10、0.10；pass@1 均值 0.05；pass@8 均值 0.33；奖励均值 0.09；零方差组比例 1.00。与 SFT 逐位相同，参数未更新。
- $$G=4$$：贪心 0.70/0.10/0.20，均值 0.33；pass@1 均值 0.27；pass@8 均值 0.67；奖励均值 0.14；零方差组比例约 0.61。
- $$G=8$$：贪心 0.90/0.60/1.00，均值 0.83；pass@1 均值 0.80；pass@8 均值 0.87；奖励均值 0.41；零方差组比例约 0.45。
- 关键观察：$$G=1$$ 的零梯度是结构性的；$$G$$ 增大提高混合组概率并降低基线方差，但采样成本线性增长；3 个种子方差很大，必须逐种子报告；pass@8 高于 pass@1，RL 先提高采样命中率；$$G=8$$ 的平均零方差组比例约 0.45，$$G=4$$ 约 0.61，$$G=1$$ 恒为 1.00。
- 测试：17 项全部通过（约 1.1 s）；示例在 CPU 上约 55 s，除耗时外逐字可复现。

### 留给后续章节的伏笔

- **第 19 章（蒸馏）**：RLVR 训练出的长思维链模型是蒸馏的主要教师；序列级蒸馏、logit 蒸馏与 on-policy 蒸馏的取舍，要与本章的“RL 提高采样效率还是扩展能力边界”一起看。 
- **第 21 章（推理加速）**：GRPO 的成本主要在 rollout；连续批处理、PagedAttention 与 PD 分离直接决定组采样吞吐。第 6 章的 KV Cache 账本要与本章的 $$B\cdot G\cdot L$$ 采样量合并估算。
- **第 20 章（解码策略）**：训练采样温度必须与旧策略对数概率的温度一致；评测温度、top-p 与贪心解码的差异，会改变 pass@1 与 pass@k 的读数。
- **第 13 章（长上下文）**：长思维链把回答长度推到数千 Token，位置外推、KV Cache 显存与长度课程成为 RLVR 的硬约束；DAPO 的超长奖励塑形是这里的工程补丁。
- **第 15 章（LoRA）**：策略与参考模型共享冻结基座、各挂 LoRA，是单卡跑 GRPO 的常见配置，但组采样仍需为每份适配器维护独立的 rollout。
- **第 16 章（PPO）**：GRPO 复用 (16.7) 的裁剪思路但去掉价值网络；比较两章时必须区分“优势来自价值估计”与“优势来自组内均值”，否则会把 GRPO 的方差问题误判为裁剪问题。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、方案对比、代码、工业实践、论文导读、常见误区、「本章的取舍」、习题齐全。
- [x] 字数：正文 10880 个中文字符（剔除代码块与公式），在核心章 10,000–12,000 预算内。
- [x] 公式编号 (18.1)–(18.12) 连续；推导无跳步：带基线策略梯度、有限 G 收缩、留一基线、裁剪目标、k3 KL、Token/序列归一化、混合组概率均有完整步骤。
- [x] 至少 1 张对比表（表 18-2，PPO/GRPO/DPO/RLOO 四列）、1 张变体表（表 18-3）、1 张奖励来源表（表 18-1）；工业案例含 DeepSeekMath、R1-Zero/R1、DAPO、Kimi k1.5、Tulu 3；论文导读 5 篇；误区 5 条。
- [x] 代码可运行：`tinylm/posttrain/grpo.py`、`examples/ch18_grpo.py`、`tests/test_grpo.py` 在 `/tmp/ch18verify` 实测通过；章内输出与 `example_out.txt` 逐字一致（耗时秒数随机器波动）。
- [x] 测试可运行：17 项全部通过（约 1.1 s）。
- [x] 公式与代码对应：`verifiable_reward` 标注 (18.1)，`sample_groups` 标注 (18.2)，`group_advantages` 标注 (18.4)–(18.6)，`tempered_logprobs` 标注 (18.7)，`grpo_loss` 标注 (18.7)–(18.11)，零方差函数标注 (18.12)。
- [x] 新符号已登记 `notation.md` 与附录 B：$$q,o_i,G,r_i,\bar r,\sigma_r,\hat A_i,b_i^{(-i)},\rho_{i,t},\epsilon_l,\epsilon_h,\hat D_{i,t},P_{\rm mixed}$$ 等，并说明与第 16 章 PPO 符号的关系。
- [x] 新术语已登记 `glossary.md` 与附录 C：RLVR、GRPO、可验证奖励、结果奖励、过程奖励、验证器、组内优势、零方差组、混合组、动态采样、clip-higher、超长奖励塑形、序列级比率等。
- [x] GitBook 语法：hint 使用 info/warning/danger/success 四类；1 个 mermaid、3 张表（奖励来源、方法对照、变体对照）、1 个 details、5 篇论文、5 条误区；代码块均标注语言；无落单 `$`。
- [x] 事实性数字可复算：全部实验数字由程序输出；DeepSeek-R1-Zero 的 AIME 2024 15.6%→71.0%、DAPO 的 32B 基座与 AIME 2024 50 分、DeepSeekMath 每组 64 条等来自公开论文；不确定的工程细节已避免写死。
- [x] 实验诚实报告：明确标注 16 词表、32 维、2 层、10 道题、300 轮的玩具任务不能外推；3 个种子方差很大、$$G$$ 的成本线性增长、验证器风险均如实写明。
- [x] 接口未破坏：`TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有签名未改；`GRPOConfig` 只作训练脚本参数，不进入 `ModelConfig`；采样与损失均在 `tinylm/posttrain/grpo.py`。

## 第 19 章交接

### 本章摘要

本章从“教师会而学生不会”的差距出发，把蒸馏分成 logit、序列级与在策略三条路线。先定义温度软标签 (19.1)，用高温展开 (19.2) 说明“高温看结构、低温看答案”；推导带 $$\tau^2$$ 缩放的 KL 蒸馏 (19.3)、硬软标签组合 (19.4)、梯度 (19.5) 与高温极限 (19.6)，解释 $$\tau^2$$ 为什么让不同温度下的学习率可比。序列级蒸馏 (19.7) 只用教师的生成结果，把长思维链的格式一起教给学生，但监督稀疏并带来曝光偏差；在策略蒸馏 (19.8) 把学生自己的前缀交给教师打分，直接消除分布偏移，代价是教师前向次数大增；GKD 用 JSD 在两者之间插值。压缩部分给出剪枝的基数约束目标 (19.9)、幅度剪枝与恢复微调，区分结构化与非结构化剪枝的“稀疏率不等于加速”，并强调掩码防止权重长回。最后用 (19.10)、(19.11) 估算参数量与压缩比，指出参数压缩比不等于延迟收益。实验在模 10 加法上对照 SFT-40、Seq-KD 与 Logit-KD，并对 Logit-KD 学生做 80% 幅度剪枝与恢复。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (19.1) | $$p_i^{\tau}=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}$$ | 温度软标签 |
| (19.2) | $$p_i^{\tau}\approx \frac1V+\frac{z_i-\bar z}{V\tau}$$ | 高温一阶展开 |
| (19.3) | $$\mathcal L_{\rm KD}=\tau^2\,\mathrm{KL}\left(p_t^{\tau}\,\parallel\,p_s^{\tau}\right)$$ | Logit 蒸馏损失 |
| (19.4) | $$\mathcal L=\alpha\,\mathcal L_{\rm CE}+(1-\alpha)\,\mathcal L_{\rm KD}$$ | 硬标签与软标签组合 |
| (19.5) | $$\frac{\partial \mathcal L_{\rm KD}}{\partial z_{s,i}}=\tau\left(p_{s,i}^{\tau}-p_{t,i}^{\tau}\right)$$ | 蒸馏对学生 logits 的梯度 |
| (19.6) | $$\frac{\partial \mathcal L_{\rm KD}}{\partial z_{s,i}}\approx\frac{z_{s,i}-z_{t,i}}{V}$$ | 高温梯度极限，近似匹配 logit 差 |
| (19.7) | $$\mathcal L_{\rm seq}=-\sum_t\log\pi_s(y_t\mid x,y_{<t}),\ y\sim\pi_t$$ | 序列级蒸馏 |
| (19.8) | $$\mathcal L_{\rm on}=\mathbb E_x\mathbb E_{y\sim\pi_s}\sum_t\mathrm{KL}\left(\pi_t\,\parallel\,\pi_s\right)$$ | 在策略蒸馏 |
| (19.9) | $$\min_{m\in\{0,1\}^d}\mathcal L(\theta\odot m)\ \text{s.t.}\ \lVert m\rVert_0\le k$$ | 剪枝的基数约束目标 |
| (19.10) | $$N\approx Vd+L\left(4d^2+2d\,d_{\rm ff}\right)$$ | Transformer 参数量估计 |
| (19.11) | $$r=N_t/N_s$$ | 参数压缩比 |

### 实验与关键数字

- 设计：模 10 加法 `(a+b) mod 10`，100 个提示，提示 5 Token（BOS/Q/数字/数字/SEP），回答 2 Token（答案/EOS）；仅 40 个提示带标签，其余 60 个只在蒸馏时可作为无标签输入；随机猜中率 0.10。
- 教师：2 层、`d_model=64`、4 头、`d_ff=256`，133,440 参数；600 步、批 64、lr $$2\times10^{-3}$$；3 个种子全部达到 1.00 准确率，输出熵 0.01–0.02 nats。
- 学生：2 层、`d_model=32`、4 头、`d_ff=128`，33,952 参数；800 步、批 64、lr $$10^{-3}$$；`DistillConfig(temperature=2.0, alpha=0.0)`。参数压缩比 3.93×。
- SFT-40（仅 40 条标注）：全量准确率 0.60/0.64/0.64，均值 0.63；留出准确率 0.33/0.40/0.40，均值 0.38。
- Seq-KD（教师贪心序列做硬标签）：3 个种子全量、留出均为 1.00。
- Logit-KD（教师末位软标签）：全量 0.97/1.00/1.00，均值 0.99；留出 0.97/1.00/1.00，均值 0.99。
- 剪枝：对 Logit-KD 学生做 80% 全局幅度剪枝，剪后准确率 0.19/0.43/0.32，均值 0.31；200 步蒸馏恢复并每步重施掩码后为 0.91/1.00/1.00，均值 0.97；稀疏度保持 0.80。
- 关键观察：SFT 与 KD 的差距同时来自“多出的 60 个无标签提示”和“软标签信号”，本章设计无法把两者完全分离，正文已明确标注；Seq-KD 与 Logit-KD 打平，是因为教师几乎确定（熵约 0.01 nats），暗知识本来就少；剪枝后不恢复只剩 0.31，说明幅度准则只是局部近似。
- 测试：21 项全部通过（约 1.17 s）；示例在 CPU 上约 102 s，除耗时外逐字可复现。

### 留给后续章节的伏笔

- **第 20 章（解码策略）**：训练时的蒸馏温度 $$\tau$$ 与推理时的采样温度不是同一个参数，但都缩放 logits；贪心、top-k、top-p 会改变师生分布的重叠程度，比较蒸馏效果时必须固定解码配置。
- **第 21 章（推理加速）**：非结构化剪枝只省存储不省时间，必须与稀疏内核、量化或结构化剪枝配合才能兑现延迟收益；本章的压缩比不能直接当作加速比。
- **第 22 章（投机解码）**：蒸馏出的小模型天然是草稿模型候选；接受率取决于小模型与大模型分布的接近程度，本章的 KL 方向讨论可直接复用。
- **第 23 章（量化）**：剪枝与量化是正交的压缩轴，通常先剪枝再量化，并分别评估恢复后的精度；低精度会放大被剪枝后更小的权重误差。
- **第 14 章（SFT）**：序列级蒸馏就是“用教师生成的数据做 SFT”，掩码、打包与拒绝采样直接沿用；第 15 章的 LoRA 可用作恢复微调的低成本适配器。
- **第 18 章（RLVR）**：长思维链教师是序列级蒸馏的主要来源；教师的能力上限与验证器质量决定学生能学到什么，蒸馏不能凭空创造教师没有的能力。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、方案对比、代码、工业实践、论文导读、常见误区、「本章的取舍」、习题齐全。
- [x] 字数：正文 7896 个中文字符（剔除代码块与公式），在其余章 6,000–9,000 预算、7,500–8,500 目标内。
- [x] 公式编号 (19.1)–(19.11) 连续；温度软标签、高温展开、KL 梯度、曝光偏差、剪枝目标、参数估计均有完整推导或说明。
- [x] 至少 1 张对比表（表 19-1，三种蒸馏与剪枝的成本/数据要求对照）；工业案例含 DistilBERT、DeepSeek-R1 蒸馏、Kimi k1.5 long2short、SparseGPT/Wanda；论文导读 5 篇；误区 5 条。
- [x] 代码可运行：`tinylm/posttrain/distill.py`、`examples/ch19_distill.py`、`tests/test_distill.py` 在 `/tmp/ch19verify` 实测通过；章内代码块与示例输出和源文件逐字一致（耗时秒数随机器波动）。
- [x] 测试可运行：21 项全部通过（约 1.17 s）。
- [x] 公式与代码对应：`softmax_with_temperature` 标注 (19.1)，`distill_loss` 标注 (19.3)–(19.4)，`sft_labels`/`generate_greedy` 标注 (19.7)，`magnitude_prune_`/`global_sparsity` 标注 (19.9)，`parameter_count`/`compression_ratio` 标注 (19.11)。
- [x] 新符号已登记 `notation.md` 与附录 B：$$\tau,\alpha,z_t,z_s,p_t^{\tau},p_s^{\tau},\mathcal L_{\rm KD},\mathcal L_{\rm seq},\mathcal L_{\rm on},m,k,\theta\odot m,N,r$$ 等，并说明与第 15 章 LoRA 秩、第 16 章奖励、第 18 章组大小的重名冲突。
- [x] 新术语已登记 `glossary.md` 与附录 C：温度软标签、暗知识、序列级蒸馏、曝光偏差、在策略蒸馏、GKD、幅度剪枝、结构化/非结构化剪枝、恢复微调、压缩比、long2short 等；附录 C 已有的四个第 19 章词条未重复添加。
- [x] GitBook 语法：hint 使用 info/warning/success；1 个 Mermaid、1 张表、1 个 `<details>`；代码块均标注语言；无落单 `$`。
- [x] 事实性数字可复算：全部实验数字由程序输出；DistilBERT 66M/110M、DeepSeek-R1 蒸馏约 80 万条与 Qwen-7B/32B 的 AIME、MATH-500 数字来自公开论文；推测与时效性内容已标注。
- [x] 实验诚实报告：明确标注 100 提示、2 层、32/64 维、40 条标注的玩具任务不能外推；SFT 与 KD 的数据覆盖混杂、教师近确定导致两种蒸馏打平、剪枝不恢复的崩溃均如实写明。
- [x] 接口未破坏：`TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有签名未改；`DistillConfig` 只作训练脚本参数，不进入 `ModelConfig`；蒸馏、生成与剪枝均在 `tinylm/posttrain/distill.py`。

## 第 20 章交接

### 本章摘要

本章把解码统一写成“把模型分布 $$p_\theta$$ 重写为解码分布 $$q$$”。先给出贪心 (20.2) 与束搜索 (20.3)–(20.5)，用“单步最大不等于序列最大”的反例说明贪心为何错过全局更优路径，并指出束搜索的剪枝、长度偏置与线性增长的代价。温度 (20.6) 只改变分布形状、不改变排序；Top-k (20.7)、Top-p (20.8) 与 Min-p (20.9) 分别用固定基数、累积质量与相对最大概率定义支撑集，截断后按 (20.10) 重新归一化。重复惩罚 (20.11) 与约束解码 (20.12) 是两类“外挂”：前者压低已出现 Token，后者用掩码保证语法合法，二者都把 $$q$$ 推离 $$p$$。随后从似然最大化与典型文本的目标错位解释神经文本退化，最后给出 pass@k 的无偏估计 (20.13) 及其期望 (20.14)，强调温度、$$n$$、$$k$$ 与截断配置必须一起报告。实验在合成马尔可夫链上比较 7 种策略：贪心与束搜索全部陷入 16 Token 循环，纯采样以低约 2.9 nats 的似然换来 56% 结束率与 pass@8=1.00，Top-p=0.9 在尖分布上剪掉结束符，Min-p 的相对阈值更稳健。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (20.1) | $$p_\theta(x_{t+1}\mid x_{\le t})=\operatorname{softmax}(\boldsymbol{\ell}_t)$$ | 自回归条件分布 |
| (20.2) | $$x_{t+1}=\arg\max_{v\in\mathcal{V}}p_\theta(v\mid x_{\le t})$$ | 贪心解码 |
| (20.3) | $$S(y_{1:t})=\sum_{i=1}^{t}\log p_\theta(y_i\mid y_{<i})$$ | 序列对数似然 |
| (20.4) | $$\mathcal{B}_t=\operatorname{top-}W_{y_{1:t-1},v}[S(y_{1:t-1})+\log p_\theta(v\mid y_{1:t-1})]$$ | 束搜索更新 |
| (20.5) | $$\tilde S(y)=S(y)/\lvert y\rvert^{\alpha}$$ | 长度归一化 |
| (20.6) | $$p_i^{\tau}=\frac{\exp(\ell_i/\tau)}{\sum_j\exp(\ell_j/\tau)}$$ | 温度缩放 |
| (20.7) | $$\mathcal{S}_k=\operatorname{TopK}_v(\ell_v,k)$$ | Top-k 支撑集 |
| (20.8) | $$\mathcal{S}_p=\min\{\mathcal{S}:\sum_{v\in\mathcal{S}}p_v\ge p\}$$ | Top-p 最小前缀 |
| (20.9) | $$\mathcal{S}_{\min p}=\{v:p_v\ge p_{\min}\max_j p_j\}$$ | Min-p 支撑集 |
| (20.10) | $$q_v=\frac{p_v}{\sum_{u\in\mathcal{S}}p_u}\mathbb{1}[v\in\mathcal{S}]$$ | 截断后重新归一化 |
| (20.11) | $$\ell_i\leftarrow\ell_i/\rho\ (\ell_i>0),\ \ell_i\rho\ (\ell_i\le 0),\ i\in\mathcal{C}$$ | 重复惩罚 |
| (20.12) | $$q_v\propto p_v m_v,\ m_v\in\{0,1\}$$ | 约束解码掩码 |
| (20.13) | $$\widehat{\mathrm{pass@}}k=1-\binom{n-c}{k}/\binom{n}{k}$$ | pass@k 无偏估计 |
| (20.14) | $$\mathbb{E}[\widehat{\mathrm{pass@}k}]=1-(1-p_{\rm succ})^k$$ | 独立同分布下的期望 |

### 实验与关键数字

- 设计：8 词表合成马尔可夫链，`BOS` 后进入 A–B 或 C–D 循环；循环内继续概率 0.9，结束与切换各 0.05；2 层、`d_model=32`、4 头、`d_ff=128`，700 步、批 64、lr $$3\times10^{-3}$$，3 个种子，每条策略 128 条样本、最大 16 Token。
- 贪心：结束率 0.00，平均长度 16.00，唯一序列比例 0.01，平均对数似然 −2.28；束宽 4 为 −2.14，长度归一化后不变，仍是 0 结束率。三种确定性策略都停在循环里。
- 纯采样 T=1：结束率 0.56，平均长度 11.45，唯一序列比例 0.54，平均对数似然 −5.04，pass@1 0.56、pass@8 1.00。
- Top-p：`T=0.8, top_p=0.9` 结束率 0.00、pass@8 0.00；放宽到 0.95 后结束率 0.17、pass@8 0.74。尖分布下单 Token 超过阈值会剪掉结束符。
- Min-p：`T=1.0, min_p=0.05` 结束率 0.45、唯一序列比例 0.36、平均对数似然 −4.08、pass@1 0.45、pass@8 0.97。
- 测试：17 项全部通过（约 0.9 s）；示例在 CPU 上约 42 s，除耗时外逐字可复现。
- 关键观察：似然最高的路径不等于最自然的路径；温度只改形状不改排序；截断改变支撑集，阈值必须结合分布形状解释；单种子方差明显（Top-p 0.95 的结束率在 0.08–0.27 之间），正文已如实报告。

### 留给后续章节的伏笔

- **第 21 章（推理加速）**：本章每步重算完整前向，`generate` 的 `use_cache` 被显式忽略；第 21 章的 KV Cache、PagedAttention 与连续批处理要接在这个冻结入口上，采样配置必须随请求走。
- **第 22 章（投机解码）**：草稿模型的接受率依赖目标模型的解码分布；温度与截断会改变验证阶段的有效分布，比较接受率时必须固定采样配置。
- **第 23 章（量化）**：低精度 logits 会放大温度与截断的边界效应；Top-p 的最小前缀在数值误差下可能抖动，评测要固定精度与内核。
- **第 24 章（测试时计算）**：Best-of-N、自洽性与树搜索都以采样为入口；本章的 pass@k 与采样预算直接决定这些方法的成本与收益。
- **第 18、19 章**：采样温度、蒸馏温度与强化学习 rollouts 的温度都缩放 logits，但不是同一个超参数；跨章比较必须确认参数出现在训练还是推理段落。
- **第 6 章（KV Cache）**：解码循环的复杂度账本要与第 6 章的缓存分析对齐，本章的玩具实现不承担吞吐结论。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、方案对比、代码、工业实践、论文导读、常见误区、「本章的取舍」、习题齐全。
- [x] 字数：正文 7511 个中文字符（剔除代码块与公式），在其余章 6,000–9,000 预算内；偏向下限，未注水。
- [x] 公式编号 (20.1)–(20.14) 连续；贪心反例、束搜索更新、温度熵变、三种截断几何、重复惩罚分段、pass@k 无偏性均有推导或说明。
- [x] 至少 1 张对比表（表 20-1，9 种策略的目标/随机性/多样性/场景/风险）；工业案例含 Hugging Face `generate`、vLLM `SamplingParams`、Llama 3 模型卡、GNMT 与 guided decoding；论文导读 5 篇；误区 6 条。
- [x] 代码可运行：从章内 Markdown 重新提取 `tinylm/inference/sample.py` 与 `examples/ch20_decode.py` 到干净目录，`/tmp/ch20final` 实测 17 项测试通过、示例输出与章内数字逐字一致（耗时秒数随机器波动）。
- [x] 测试可运行：17 项全部通过（约 0.9 s）。
- [x] 公式与代码对应：`filtered_probs` 标注 (20.6)/(20.10)，`top_k_mask`/`top_p_mask`/`min_p_mask` 标注 (20.7)–(20.9)，`apply_repetition_penalty` 标注 (20.11)，`beam_search` 标注 (20.3)–(20.5)，`pass_at_k` 标注 (20.13)。
- [x] 新符号已登记 `notation.md` 与附录 B：$$q,q_t,S(y),W,\tau,p,p_{\min},p_{\rm succ},k,\rho,\mathcal{S}_k,\mathcal{S}_p,\mathcal{S}_{\min p},n,c$$ 等，并说明与第 19 章蒸馏温度、第 6 章序列长度、第 3 章词表等重名冲突。
- [x] 新术语已登记 `glossary.md` 与附录 C：文本退化、长度惩罚、局部典型采样、pass@k、频率惩罚、存在惩罚、结构化输出、语法约束解码；附录 C 已有的贪心/束搜索/温度/Top-k/Top-p/Min-p/重复惩罚/约束解码/Logit 处理器未重复添加。
- [x] GitBook 语法：hint 使用 info/success/warning；1 个 Mermaid、1 张表、1 个 `<details>`；代码块均标注语言；无落单 `$`。
- [x] 事实性数字可复算：全部实验数字由程序输出；Llama 3 的 `temperature=0.6, top_p=0.9`、Codex 的 $$n=200$$ 与 GNMT 束宽 4–8 来自公开资料并标注「截至写作时」或「通常」；推测内容已标注。
- [x] 实验诚实报告：明确标注 8 词表、2 层 32 维、700 步的玩具任务不能外推；长度归一化无效、Top-p 0.9 失效、三种种子方差偏大等负结果均保留。
- [x] 接口未破坏：`TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有签名未改；`SamplingConfig` 只作生成参数，不进入 `ModelConfig`；所有解码逻辑在 `tinylm/inference/sample.py`。

## 第 21 章交接

### 本章摘要

本章把一次生成拆成预填充与解码两段，用同一本账解释它们为什么一个吃算力、一个吃带宽。先给出 KV Cache 的显存公式 (21.1) 与 MLA 的压缩形式 (21.2)，说明 GQA/MQA/MLA、量化与驱逐如何缩小每 Token 的 KV；再推导解码的 FLOPs (21.3)、每步搬运字节 (21.4)、算术强度 (21.5) 与 roofline 下界 (21.6)，用 7B 在 4K 下的数字说明解码的算术强度只有约 1.1，远低于 H100 的 295 FLOP/Byte。批处理能摊薄权重却摊不薄 KV，因此渐近强度 (21.7) 只有约 31，仅靠加批无法让解码变成算力受限。FlashAttention 部分从 $$T\times T$$ 中间矩阵的显存爆炸出发，推导分块在线 Softmax 的 (21.8)–(21.11)，证明运行量更新的不变量，并给出 HBM 访问复杂度 (21.12)；强调它是精确注意力，省的是 IO 不是 FLOPs。PagedAttention 把 KV 当虚拟内存，用块表消除内外碎片，给出碎片率 (21.13)；调度部分依次讨论连续批处理、抢占、分块预填充 (21.14)、PD 分离与前缀缓存 (21.15)。最后用容量 (21.16)、TTFT (21.17)、TPOT 与端到端延迟 (21.18)、吞吐与利特尔法则 (21.19) 组成延迟–吞吐账本，把 goodput 作为真正的优化目标。代码在 TinyLM 上新增 `KVCache`、`cached_forward`/`cached_greedy_decode`、`flash_attention` 与 `BlockTable`，不改冻结的 `TinyLM.forward`。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (21.1) | $$M_{\mathrm{kv}}=2n_lBTn_{kv}d_hb_{\mathrm{elem}}$$ | KV Cache 显存 |
| (21.2) | $$M_{\mathrm{kv}}^{\mathrm{MLA}}=n_lBT(d_c+d_{\mathrm{rope}})b_{\mathrm{elem}}$$ | MLA 压缩缓存 |
| (21.3) | $$F_{\mathrm{dec}}=2N+4n_lTd_{\mathrm{model}}$$ | 解码一步 FLOPs |
| (21.4) | $$M_{\mathrm{step}}=Nb_{\mathrm{elem}}+M_{\mathrm{kv}}(B)$$ | 解码一步搬运字节 |
| (21.5) | $$I_{\mathrm{arith}}=\frac{BF_{\mathrm{dec}}}{Nb_{\mathrm{elem}}+M_{\mathrm{kv}}(B)}$$ | 算术强度 |
| (21.6) | $$t_{\mathrm{step}}=\max\left(\frac{BF_{\mathrm{dec}}}{F_{\max}},\frac{M_{\mathrm{step}}}{B_{\mathrm{bw}}}\right)$$ | roofline 下界 |
| (21.7) | $$I_\infty=\frac{2N+4n_lTd_{\mathrm{model}}}{2n_lTn_{kv}d_hb_{\mathrm{elem}}}$$ | 批趋于无穷的渐近强度 |
| (21.8) | $$m_{\mathrm{new}}=\max\left(m,\max_j s_j\right)$$ | 在线最大值 |
| (21.9) | $$\ell_{\mathrm{new}}=\alpha\ell+\sum_j e^{s_j-m_{\mathrm{new}}}$$ | 在线指数和 |
| (21.10) | $$\mathbf{O}_{\mathrm{new}}=\alpha\mathbf{O}+\sum_j e^{s_j-m_{\mathrm{new}}}\mathbf{v}_j$$ | 在线输出累积 |
| (21.11) | $$\mathbf{O}=\mathbf{O}/\ell$$ | 最终归一化 |
| (21.12) | $$\Theta\left(T^2d^2/M+Td\right)$$ | 分块注意力的 HBM 访问 |
| (21.13) | $$w_{\mathrm{frag}}=1-\frac{\sum_i n_i}{\sum_i\lceil n_i/s\rceil s}$$ | 块内碎片率 |
| (21.14) | $$T_{\mathrm{chunk}}\le R_{\mathrm{pre}}\left(t_{\mathrm{budget}}-t_{\mathrm{dec}}(B)\right)$$ | 分块大小上界 |
| (21.15) | $$s_{\mathrm{prefix}}=T_{\mathrm{shared}}/(T_{\mathrm{shared}}+T_{\mathrm{new}})$$ | 前缀缓存节省比例 |
| (21.16) | $$B_{\max}=\left\lfloor\frac{M_{\mathrm{total}}-M_{\mathrm{weights}}-M_{\mathrm{act}}}{M_{\mathrm{kv,seq}}}\right\rfloor$$ | 显存容量上限 |
| (21.17) | $$\mathrm{TTFT}=W_q+T_{\mathrm{prompt}}/R_{\mathrm{pre}}$$ | 首 Token 延迟 |
| (21.18) | $$\mathrm{TPOT}=t_{\mathrm{step}},\quad W=\mathrm{TTFT}+(n_{\mathrm{out}}-1)\mathrm{TPOT}$$ | 每 Token 时间与端到端延迟 |
| (21.19) | $$R_{\mathrm{tok}}=B/\mathrm{TPOT},\quad N_{\mathrm{inflight}}=r_{\mathrm{arr}}\bar W$$ | 吞吐与利特尔法则 |

### 实验与关键数字

- 设计：TinyLM 小模型（`vocab_size=64`、`d_model=64`、3 层、8 个查询头、2 个 KV 头）验证缓存等价；速度对照用 `d_model=128`、4 层、提示 32 Token、生成 96 Token；roofline 用 7B GQA（32 层、8 个 KV 头、`d_h=128`、4K 上下文）与 H100 SXM 的 989 TFLOPS / 3.35 TB/s。
- 缓存等价性（最大绝对误差）：整段预填充 `0.00e+00`，分块预填充 `1.49e-07`，逐 Token 解码 `1.49e-07`。
- FlashAttention 与稠密注意力在 T=256 时最大绝对误差 `4.77e-07`，输出形状 `(2, 8, 256, 64)`。
- KV 账本（FP16，单序列）：7B GQA 4K 为 512.0 MiB、32K 为 4.0 GiB，每 GiB 约 8,192 Token，24 GB 卡最多约 40 条；13B MHA 4K 为 3.1 GiB、32K 为 25.0 GiB，每 GiB 约 1,311 Token，最多约 6 条；MLA（V2 规模）4K 为 270.0 MiB、32K 为 2.1 GiB，每 GiB 约 15,534 Token，最多约 75 条。
- 纯 PyTorch CPU 实测：每步重算全序列 0.186 s，KV Cache 增量 0.101 s，加速比 1.85×；该数字只在 32 Token 提示、96 Token 生成的玩具设置下成立，不能外推为真实服务加速比。
- 分页 vs 朴素预分配（`block=16`，长度 `[2048,17,4000,256,1,512]`）：有效 Token 6,834；分页预留 6,864 槽位、利用率 99.563%；朴素预留 24,000 槽位、利用率 28.475%。
- 解码 roofline（7B，T=4096）：B=1 时算术强度 1.1 F/B、搬运 14.0 GB、下界 4.5 ms、吞吐 223 tok/s；B=8 时 7.1 F/B、17.5 GB、5.6 ms、1,427 tok/s；B=32 时 16.8 F/B、29.5 GB、9.5 ms、3,386 tok/s；B=64 时 21.8 F/B、45.5 GB、14.6 ms、4,390 tok/s。
- 关键观察：$$I_\infty\approx31$$ 远低于 H100 的 295，批大小无法把解码变成算力受限；要让 $$I_\infty$$ 达到 295，上下文需短到约 380 Token；KV 读取与批同阶增长，是吞吐的最终约束。
- 测试：20 项全部通过（约 1 s）；示例除耗时外逐字可复现，速度对照行的绝对秒数随机器波动，正文已注明。

### 留给后续章节的伏笔

- **第 22 章（投机解码）**：本章的 roofline 账本解释了为什么一次验证多个 Token 能提高算术强度；草稿模型与目标模型的接受率决定实际节省，KV Cache 必须能回滚被拒绝的 Token。
- **第 23 章（量化）**：KV 量化直接缩小 (21.1) 的 $$b_{\mathrm{elem}}$$，同时改变解码的带宽下界；量化误差在长上下文与远距离键上更敏感，需与本章的容量账本一起评估。
- **第 24 章（测试时计算）**：Best-of-N、自洽性与树搜索会成倍放大解码步数与 KV 占用；本章的吞吐与利特尔法则决定这些方法的服务成本。
- **第 12 章（并行训练）**：张量并行与流水并行在推理侧的通信、KV 分片与 PD 分离的资源放置，直接复用本章的账本；PD 分离本质上是把训练中的阶段划分搬到服务侧。
- **第 6 章（注意力变体）**：GQA/MQA/MLA 的 KV 头数与压缩维度决定 (21.1)、(21.2) 的系数，本章不再重复推导结构，只消费其结论。
- **第 20 章（解码策略）**：`generate` 的 `use_cache` 从第 21 章起真正生效；采样配置必须随请求进入调度器，连续批处理中不同请求的温度、Top-p 不能互相污染。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、方案对比、代码、工业实践、论文导读、常见误区、「本章的取舍」、习题齐全。
- [x] 字数：正文 10437 个中文字符（剔除代码块与公式），在核心章 10,000–12,000 预算内，接近 11,000 目标。
- [x] 公式编号 (21.1)–(21.19) 连续；KV 显存、解码 FLOPs、算术强度、roofline、在线 Softmax、IO 复杂度、碎片率、分块上界、前缀节省、容量、TTFT/TPOT 与利特尔法则均有推导或说明。
- [x] 至少 1 张对比表（表 21-1 四种调度方案，表 21-2 五条 KV 压缩路线）；工业案例含 vLLM、FlashAttention-2/3、TensorRT-LLM、SGLang、Orca、Sarathi-Serve、DistServe；论文导读 5 篇；误区 6 条。
- [x] 代码可运行：从章内 Markdown 重新提取四个代码块到 `/tmp/ch21final`，与源文件 `diff` 一致；20 项测试通过，示例的确定性输出与章内逐字一致（耗时秒数随机器波动）。
- [x] 测试可运行：20 项全部通过（约 1 s）。
- [x] 公式与代码对应：`cache_bytes` 标注 (21.1)/(21.2)，`arithmetic_intensity`/`roofline_time` 标注 (21.5)/(21.6)，`flash_attention` 标注 (21.8)–(21.11)，`internal_fragmentation` 标注 (21.13)，`ttft`/`tpot`/`throughput` 标注 (21.17)–(21.19)。
- [x] 新符号已登记 `notation.md` 与附录 B：$$M_{\mathrm{kv}},n_l,n_{kv},d_h,b_{\mathrm{elem}},F_{\mathrm{dec}},M_{\mathrm{step}},I_{\mathrm{arith}},F_{\max},B_{\mathrm{bw}},I_\infty,m,\ell,\mathbf{O},T_{\mathrm{chunk}},R_{\mathrm{pre}},s_{\mathrm{prefix}},B_{\max},\mathrm{TTFT},\mathrm{TPOT},W_q,n_{\mathrm{out}},N_{\mathrm{inflight}},r_{\mathrm{arr}},\bar W$$ 等，并说明与第 9 章 $$N$$、第 6 章 $$n_{kv}$$、第 13 章 $$T$$ 的重名冲突。
- [x] 新术语已登记 `glossary.md` 与附录 C：预填充、解码、算术强度、屋顶线模型、在线 Softmax、分块注意力、分页注意力、块表、内部/外部碎片、连续批处理、迭代级调度、抢占、分块预填充、无停顿调度、预填充-解码分离、前缀缓存、TTFT、TPOT、有效吞吐、写时复制、KV 量化。
- [x] GitBook 语法：hint 使用 info/success/warning/danger；1 个 Mermaid、2 张表、1 个 `<details>`；代码块均标注语言；无落单 `$`。
- [x] 事实性数字可复算：实验数字全部由程序输出；vLLM 的 60%–80% 浪费与 2–4×、FlashAttention-2 的 50%–73% 与 2×、Orca 36.9×、Sarathi 2.6×、DistServe 7.4× 来自公开论文，正文标注了模型、硬件与 SLO 条件；推测与时效性内容已标注。
- [x] 实验诚实报告：明确标注小模型、CPU、短序列的加速比不能外推；L2 命中率、CPU 分配器与固定开销会影响绝对耗时；批增大后吞吐收益递减的负结果保留。
- [x] 接口未破坏：`TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有签名未改；缓存版本以同包 `cached_forward` 实现，`use_cache=True` 走该路径，`past_kv` 承载缓存对象；纯注意力块之外的架构被显式断言排除。

## 第 22 章交接

### 本章摘要

本章回答“能不能让便宜的小模型先猜几个 Token，再让大模型一次验收”，并证明这种接受–拒绝规则不改变输出分布。先从第 21 章的 roofline 账本出发：解码受带宽限制，验证 $$\gamma+1$$ 个位置共享一次权重读取，因此“多算”比“多读”便宜。数学部分给出单步拒绝采样 (22.1)–(22.4)：接受概率 $$a(x)=\min(1,p/q)$$，拒绝后从残差 $$r(y)\propto\max(0,p-q)$$ 重采样，两项相加恰为 $$p$$；接受率 $$\alpha=\sum\min(p,q)=1-\mathrm{TV}$$ (22.5)。序列级用归纳证明每个位置的条件分布都等于目标分布，并补充贪心解码下的确定性接受规则。收益部分推导期望接受数 (22.6)、期望产出 Token 数 (22.7) 与加速比 (22.8)，说明 $$\gamma$$ 的边际收益为何指数衰减、最优 $$\gamma$$ 为何存在；按位置接受率给出更细的 (22.9)。方案部分对比外部草稿、Medusa、EAGLE 与 MTP，讲清“Token 层猜”“特征层猜”“训练时附带”的差别；树状草稿用祖先掩码 (22.10) 一次验证多条路径。MTP 部分给出损失 (22.11)–(22.12)，强调它首先是训练目标，推理时仍须验证。最后讨论 KV Cache 回滚 (22.13) 与位置对齐。代码在 TinyLM 中新增 `tinylm/inference/speculative.py`，以概率函数为接口，便于用玩具马尔可夫链精确验证。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (22.1) | $$a(x)=\min\left(1,\frac{p(x)}{q(x)}\right)$$ | 接受概率 |
| (22.2) | $$r(y)=\frac{\max(0,p(y)-q(y))}{Z}$$ | 残差分布 |
| (22.3) | $$\Pr(\text{reject})=Z=\sum_x\max(0,q(x)-p(x))$$ | 拒绝概率 |
| (22.4) | $$\Pr(Y=y)=\min(p(y),q(y))+\max(0,p(y)-q(y))=p(y)$$ | 无损性 |
| (22.5) | $$\alpha=\sum_x\min(p(x),q(x))=1-\mathrm{TV}(p,q)$$ | 接受率 |
| (22.6) | $$\mathbb{E}[A]=\frac{\alpha(1-\alpha^\gamma)}{1-\alpha}$$ | 期望接受草稿数 |
| (22.7) | $$\mathbb{E}[K]=\frac{1-\alpha^{\gamma+1}}{1-\alpha}$$ | 每轮期望产出 Token |
| (22.8) | $$S=\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(1+c\gamma)}$$ | 加速比 |
| (22.9) | $$\mathbb{E}[K]=1+\alpha_1+\alpha_1\alpha_2+\cdots+\prod_{i=1}^{\gamma}\alpha_i$$ | 按位置接受率 |
| (22.10) | $$M_{ij}=0\ \text{若}\ j\in\mathcal{A}(i),\ -\infty\ \text{否则}$$ | 树状祖先掩码 |
| (22.11) | $$\mathcal{L}_{\mathrm{MTP}}=-\frac1k\sum_{i=1}^{k}\log p_i(x_{t+i}\mid\mathbf{h}_t)$$ | MTP 损失 |
| (22.12) | $$\mathcal{L}=\mathcal{L}_{\mathrm{CE}}+\lambda\mathcal{L}_{\mathrm{MTP}}$$ | 总损失 |
| (22.13) | $$L'=L+a$$ | 回滚后的缓存长度 |

### 实验与关键数字

- 设计：无损性与接受数用 5 词表分布做 20 万次/10 万轮蒙特卡洛；KV 回滚用 TinyLM（`vocab_size=32`、`d_model=64`、2 层、8 个查询头、2 个 KV 头）对比“截断缓存”与“重新预填充”；MTP 用 3 个线性头、`d_model=16`、词表 11；树掩码用 5 节点树。
- 无损性：20 万次拒绝采样的经验分布与目标 $$p$$ 最大误差 0.001；实测接受率 0.5016，理论值 0.5000。
- 草稿质量账本（$$\gamma=4$$、$$c=0.1$$）：混合系数 0.2/0.4/0.6/0.8 对应 $$\alpha=0.890/0.780/0.670/0.560$$，加速比 2.867/2.309/1.872/1.534。
- $$\gamma$$ 的边际收益（$$\alpha=0.8$$、$$c=0.1$$）：$$\gamma=1,4,6,8$$ 的加速比分别为 1.636、2.401、2.470、2.405；最优在 $$\gamma=6$$ 附近。
- 接受数模拟（10 万轮，$$\alpha=0.50$$、$$\gamma=4$$）：实测平均接受 0.9365，理论 0.9375；每轮平均输出 1.9365 Token。
- KV 回滚（接受 2/4 个草稿）：回滚后缓存长度 11；与重新预填充的 logits 最大误差 $$2.98\times10^{-8}$$。
- MTP：每步 logits 形状 `(1, 6, 11)` × 3；三步平均损失 2.7020，仅第一步损失 2.3890。
- 玩具马尔可夫模型：平均每步接受率 0.700，$$\gamma=4$$ 的期望产出 2.773 Token。
- 测试：14 项全部通过（约 1 s），覆盖接受率恒等式、残差归一化、无损性蒙特卡洛、$$p=q$$ 全接受、接受数上界、端点与单调性、内部最优 $$\gamma$$、KV 回滚等价、树掩码祖先性、MTP 形状/损失/忽略填充、确定性解码与非法批次拒绝。

### 留给后续章节的伏笔

- **第 23 章（量化）**：草稿模型量化会改变其分布 $$q$$，直接降低接受率；目标模型量化也会改变验证分布。量化误差对 $$\alpha$$ 的影响必须用真实负载测量，不能只看模型困惑度。
- **第 24 章（测试时计算）**：Best-of-N、自洽性与树搜索增加的是采样次数而不是单次解码步数；投机解码可与这些方法叠加，但 KV 占用与调度开销要一起算。
- **第 20 章（解码策略）**：温度、Top-k、Top-p 改变 $$p$$ 与 $$q$$，接受率随之变化；贪心与采样两条路径的无损规则不同，监控必须分开。
- **第 21 章（推理加速）**：KV Cache 回滚依赖 `length` 指针与分页块归还；树状分支需要写时复制或按节点缓存，否则分支互相污染。
- **第 19 章（蒸馏）**：蒸馏出的小模型是天然的草稿模型候选；师生分布的接近程度直接对应接受率。
- **第 1 章（训练目标）**：MTP 作为训练目标的动机与本章的推理用法相互印证；训练时的多步监督与推理时的多 Token 验证不是同一件事。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、方案对比、代码、工业实践、论文导读、常见误区、「本章的取舍」、习题齐全。
- [x] 字数：正文 7731 个中文字符（剔除代码块与公式），在其余章 6,000–9,000 预算内，落在 7,500–8,500 目标区间。
- [x] 公式编号 (22.1)–(22.13) 连续；单步无损性、残差归一化、序列归纳、接受率恒等式、期望收益、加速比、按位置接受率、树掩码、MTP 损失与回滚长度均有推导或说明。
- [x] 至少 1 张对比表（表 22-1 四条路线）；工业案例含 Leviathan/Chen、Medusa、EAGLE、DeepSeek-V3、vLLM/TensorRT-LLM；论文导读 5 篇；误区 6 条。
- [x] 代码可运行：从章内 Markdown 重新提取 `tinylm/inference/speculative.py` 与 `examples/ch22_speculative.py` 到 `/tmp/ch22final`，14 项测试通过，示例输出与章内逐字一致。
- [x] 测试可运行：14 项全部通过（约 1 s）。
- [x] 公式与代码对应：`acceptance_rate` 标注 (22.5)，`residual_probs` 标注 (22.2)，`rejection_sample`/`speculative_sample` 标注 (22.1)–(22.3)，`expected_output_tokens` 标注 (22.7)，`speculative_speedup` 标注 (22.8)，`truncate_cache` 标注 (22.13)，`MTPHead` 标注 (22.11)，`mtp_loss` 标注 (22.11)–(22.12)，`tree_attention_mask` 标注 (22.10)。
- [x] 新符号已登记 `notation.md` 与附录 B：$$p,q,a(x),r(y),Z,\alpha,\mathrm{TV},\gamma,c,A,K,S,\alpha_i,\mathbf{h}_t,p_i,M_{ij},\mathcal{A}(i),\lambda,k,L'$$ 等，并说明与第 20 章采样概率、第 21 章缓存长度、第 13 章序列长度的重名冲突。
- [x] 新术语已登记 `glossary.md` 与附录 C：投机解码、草稿模型、目标模型、拒绝采样、残差分布、接受率、全变差距离、草稿长度、树状草稿、树状掩码、Medusa、EAGLE、多 Token 预测、KV Cache 回滚；附录 C 已有的贪心/温度/Top-p/蒸馏/KV Cache 未重复添加。
- [x] GitBook 语法：hint 使用 warning；1 个 Mermaid、1 张表、1 个 `<details>`；代码块均标注语言；无落单 `$`。
- [x] 事实性数字可复算：实验数字全部由程序输出；Medusa 2.2–3.6×、EAGLE 约 2–3×、Leviathan 2–3×、DeepSeek-V3 的 671B/37B 来自公开论文并在正文标注模型与负载条件；推测与时效性内容已标注。
- [x] 实验诚实报告：明确标注 5 词表、2 层小模型与玩具马尔可夫链的加速比不能外推；接受率随负载漂移、长上下文降低收益等负结果均保留。
- [x] 接口未破坏：`TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有签名未改；`SpecConfig` 只作推理参数，不进入 `ModelConfig`；所有新逻辑在 `tinylm/inference/speculative.py`。


## 第 23 章交接

### 本章摘要

本章回答“为什么权重能压到 4 比特而激活不行”，并把量化的误差、存储与延迟收益拆开记账。直觉部分先建立“有限格子”模型：缩放因子决定格距、零点决定偏移，误差分舍入与截断两类；然后用 4 比特步长的具体数字说明离群值如何撑粗整组格子，并解释残差流与 RMSNorm 为何在深层制造系统性离群通道。数学部分给出仿射量化 (23.1)–(23.5)、舍入上界 (23.6)、均匀噪声方差 (23.7)、SQNR 定义 (23.8) 与满量程正弦波的 (23.9)；再给出含元数据的平均比特 (23.10) 与压缩比 (23.11)。权重量化从逐层重建目标 (23.12) 出发，推导 GPTQ 的逐列 Hessian 补偿 (23.13)，并对比 AWQ 的激活感知缩放 (23.14) 与 SmoothQuant 的离群值搬移 (23.15)。激活部分讨论逐 Token 动态定标 (23.16)、最小均方误差定标 (23.17) 与直通估计器 (23.18)。KV 部分把第 21 章的容量公式换成量化字节数 (23.19)，说明键按通道、值按 Token 的粒度差异。格式部分给出 FP8 的数值表示 (23.20)、NF4 码本与 MXFP4 块浮点。性能部分给出权重与 KV 同时量化后的理想加速比 (23.21) 与最大并发 (23.22)，并用 7B 账本说明长上下文与大批量会把瓶颈搬到 KV。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (23.1) | $$q_{\min}=-(2^{b_q-1}-1),\ q_{\max}=2^{b_q-1}-1$$ | 对称量化整数范围 |
| (23.2) | $$x_{\mathrm{int}}=\mathrm{clamp}(\mathrm{round}(x/s+z),q_{\min},q_{\max})$$ | 量化 |
| (23.3) | $$\hat{x}=s(x_{\mathrm{int}}-z)$$ | 反量化 |
| (23.4) | $$s=\max\lvert x\rvert/q_{\max},\ z=0$$ | 对称定标 |
| (23.5) | $$s=(x_{\max}-x_{\min})/(q_{\max}-q_{\min}),\ z=\mathrm{round}(q_{\min}-x_{\min}/s)$$ | 非对称定标 |
| (23.6) | $$\lvert x-\hat{x}\rvert\le s/2$$ | 舍入误差上界 |
| (23.7) | $$\mathbb{E}[(x-\hat{x})^2]=s^2/12$$ | 均匀量化噪声方差 |
| (23.8) | $$\mathrm{SQNR}=10\log_{10}\frac{\mathbb{E}[x^2]}{\mathbb{E}[(x-\hat{x})^2]}$$ | 信噪比定义 |
| (23.9) | $$\mathrm{SQNR}=6.02b_q+1.76\ \text{dB}$$ | 满量程正弦波上界 |
| (23.10) | $$b_{\mathrm{eff}}=b_q+\frac{16+8\cdot\mathbf{1}[\text{非对称}]}{\min(G,d_{\mathrm{in}})}$$ | 平均比特/权重 |
| (23.11) | $$\rho=16/b_{\mathrm{eff}}$$ | 相对 FP16 压缩比 |
| (23.12) | $$\mathcal{L}=\lVert\mathbf{W}\mathbf{X}^\top-\widehat{\mathbf{W}}\mathbf{X}^\top\rVert_F^2=\mathrm{tr}((\mathbf{W}-\widehat{\mathbf{W}})\mathbf{H}(\mathbf{W}-\widehat{\mathbf{W}})^\top)$$ | 逐层重建目标 |
| (23.13) | $$\mathbf{W}_{:,i+1:}\leftarrow\mathbf{W}_{:,i+1:}-\frac{\mathbf{w}_i-\hat{\mathbf{w}}_i}{[\mathbf{H}^{-1}]_{ii}}[\mathbf{H}^{-1}]_{i,i+1:}$$ | GPTQ 误差补偿 |
| (23.14) | $$s_j=\left(\frac1n\sum_t\lvert X_{t,j}\rvert\right)^{\alpha},\ \mathbf{W}\leftarrow\mathbf{W}\,\mathrm{diag}(\mathbf{s})$$ | AWQ 激活感知缩放 |
| (23.15) | $$s_j=\frac{\max_t\lvert X_{t,j}\rvert^{\alpha}}{\max_i\lvert W_{i,j}\rvert^{1-\alpha}}$$ | SmoothQuant 通道搬移 |
| (23.16) | $$s_t=\max_j\lvert X_{t,j}\rvert/q_{\max}$$ | 逐 Token 动态定标 |
| (23.17) | $$s^\star=\arg\min_s\mathbb{E}[(x-\mathrm{clamp}(\mathrm{round}(x/s)s))^2]$$ | 最小 MSE 定标 |
| (23.18) | $$\partial\mathcal{L}/\partial x\approx\partial\mathcal{L}/\partial\hat{x}$$ | 直通估计器 |
| (23.19) | $$M_{\mathrm{kv}}=2n_lBTn_{kv}d_hb_{\mathrm{kv}}$$ | 量化后 KV 显存 |
| (23.20) | $$x=(-1)^{\sigma}2^{e-\mathrm{bias}}(1+m/2^{M_{\mathrm{fp}}})$$ | FP8 数值表示 |
| (23.21) | $$S=\frac{2N+2M_{\mathrm{kv}}}{b_wN+b_{\mathrm{kv}}M_{\mathrm{kv}}}$$ | 理想解码加速比 |
| (23.22) | $$B_{\max}=\left\lfloor\frac{M_{\mathrm{total}}-Nb_w-M_{\mathrm{act}}}{2n_lTn_{kv}d_hb_{\mathrm{kv}}}\right\rfloor$$ | 量化后最大并发 |

### 实验与关键数字

- 设计：粒度与比特扫描用合成权重（行尺度差 + 行内离群值）；GPTQ 用 1024 条相关输入校准 64×128 线性层；SmoothQuant 用 4 个 30 倍离群通道的 2048×128 激活；KV 用 2×4×64×16 张量、前两个键通道放大 20 倍；TinyLM 为 `d_model=64`、2 层、4 查询头、2 KV 头、词表 128，训练 400 步；带宽账本按 7B 参数、4K 上下文、H100 3.35 TB/s 的理想下界。
- 粒度（4 比特，64×128）：逐张量 8.92 dB，逐通道 11.45 dB，逐组 128 同逐通道，逐组 64 13.84 dB，逐组 32 16.73 dB；平均比特 4.125/4.250/4.500。
- 比特扫描（逐组 64，256×1024）：2 比特 2.88 dB（7.11×），3 比特 12.00 dB（4.92×），4 比特 19.35 dB（3.76×），6 比特 32.30 dB（2.56×），8 比特 44.53 dB（1.94×）。
- INT4 打包：4096×4096 权重 FP16 32.0 MiB、INT4 码点 8.0 MiB，含缩放因子 4.250 比特/权重，压缩 3.76×。
- GPTQ：RTN 输出 MSE 0.012894、18.67 dB；GPTQ 0.001805、27.21 dB，误差约降至 1/7，信噪比高 8.5 dB。
- SmoothQuant：不平滑 0.004089/34.29 dB；$$\alpha=0.3/0.5/0.7$$ 为 41.62/43.76/41.82 dB，$$\alpha=0.5$$ 时 MSE 0.000385。
- KV：INT8 逐 Token 28.00 dB、逐通道 34.74 dB；INT4 逐 Token 11.55 dB、逐通道 16.42 dB。
- TinyLM：FP16 验证损失 6.2076；W8 逐通道 6.2082（+0.0007）；W4 逐通道 6.2096（+0.0021）；W4 逐组 32 6.2180（+0.0105）；W2 逐通道 6.3988（+0.1913，困惑度 1.211×）。
- 7B 账本（批 1）：FP16 14.01 GB、239.1 Token/s；INT8 7.01 GB、478.1 Token/s；INT4 3.50 GB、956.2 Token/s。批 32：FP16 30.66 GB、109.3 Token/s；W4 20.55 GB、163.0 Token/s；W4+KV4 7.66 GB、437.1 Token/s。
- 测试：16 项全部通过（约 0.7 s），覆盖配置校验、整数范围、舍入上界、非对称端点、粒度单调性、任意形状与填充、假量化往返、每比特约 6 dB、INT4 打包、存储公式、量化线性层、GPTQ 优于 RTN、SmoothQuant 降误差、KV 逐通道优于逐 Token、零张量安全与布局保持。

### 留给后续章节的伏笔

- **第 24 章（测试时计算）**：Best-of-N、自洽性与树搜索都要多次采样或多次打分；量化会同时改变采样分布与奖励/验证信号，低比特模型上的测试时扩展收益需要重新测量，不能直接套用 FP16 的曲线。
- **第 22 章（投机解码）**：量化草稿改变 $$q$$、量化目标改变 $$p$$，接受率 $$\alpha=\sum\min(p,q)$$ 会下降；两者叠加的加速比必须在真实负载上重测。
- **第 21 章（推理加速）**：带宽账本 (23.21) 是第 21 章 roofline 的直接延伸；KV 量化的尺度表必须与分页缓存、连续批处理兼容。
- **第 20 章（解码策略）**：温度、Top-k/Top-p 改变激活与输出分布，量化误差对不同采样策略的敏感度不同，评测要固定解码配置。
- **第 15 章（参数高效微调）**：NF4 与双重量化是 QLoRA 的存储基础，本章从推理侧解释了同一条“4 比特存储、16 比特计算”路线。
- **第 19 章（蒸馏）**：量化可看作一种受限容量下的压缩；蒸馏出的小模型与量化后的大模型在质量—成本前沿上可以相互替代，选型时要一起比较。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、方案对比、代码、工业实践、论文导读、常见误区、「本章的取舍」、习题齐全。
- [x] 字数：正文 10276 个中文字符（剔除代码块与公式），在核心章 10,000–12,000 预算内。
- [x] 公式编号 (23.1)–(23.22) 连续；量化/反量化、误差界、噪声方差、SQNR、存储、GPTQ、AWQ、SmoothQuant、动态定标、STE、KV 容量、FP8、加速比与最大并发均有推导或说明。
- [x] 至少 1 张对比表（表 23-1 粒度、表 23-2 格式、表 23-3 五条路线）；工业案例含 LLM.int8()、GPTQ、AWQ、SmoothQuant、DeepSeek-V3 FP8、llama.cpp/GGUF、QLoRA/NF4；论文导读 5 篇；误区 6 条。
- [x] 代码可运行：从章内 Markdown 重新提取 `tinylm/inference/quant.py` 与 `examples/ch23_quant.py`，与已验证源文件逐字一致；在 `/tmp/ch23final` 重跑，输出一致，16 项测试通过。
- [x] 公式与代码对应：`qrange` 标注 (23.1)，`quantize` 标注 (23.2)，`dequantize` 标注 (23.3)，`quant_params` 标注 (23.4)–(23.5)，`sqnr_db` 标注 (23.8)，`theoretical_sqnr` 标注 (23.9)，`storage_bits` 标注 (23.10)，`compression_ratio` 标注 (23.11)，`gptq_quantize` 标注 (23.13)，`smooth_scale`/`smooth_quant` 标注 (23.15)。
- [x] 新符号已登记 `notation.md` 与附录 B：$$x_{\mathrm{int}},\hat{x},q_{\min},q_{\max},b_{\mathrm{eff}},\rho,\mathbf{H},\mathbf{X},\widehat{\mathbf{W}},b_{\mathrm{kv}},b_w,\alpha,\sigma,e,m,M,S,B_{\max}$$ 等，并说明与第 1、3、6、11、15、21、22 章的重名冲突。
- [x] 新术语已登记 `glossary.md` 与附录 C：反量化、缩放因子、零点、逐张量/逐通道/逐组量化、PTQ、QAT、直通估计器、量化校准、SQNR、量化误差、块浮点、微缩放格式、W8A8/W4A16、GPTQ、AWQ、SmoothQuant、KV Cache 量化；附录 C 已有的量化、NF4、离群值未重复添加。
- [x] GitBook 语法：hint 使用 danger/warning/success；1 个 Mermaid、3 张表、1 个 `<details>`；代码块均标注语言；无落单 `$`，无外部图片。
- [x] 事实性数字可复算：实验数字全部由固定种子程序输出；FP8 E4M3/E5M2 范围、LLM.int8() 的 0.1% 离群特征、GPTQ/AWQ 的论文加速倍数、DeepSeek-V3 的块缩放配置均来自公开论文并在正文标注条件；FP4 生态等时效性内容标注“截至写作时”。
- [x] 实验诚实报告：明确标注 TinyLM 规模小、合成任务可约损失低、带宽账本为理想下界、困惑度倍数为同验证集上的损失差指数；负结果（$$\alpha=0.7$$ 回落、W2 明显退化）保留。
- [x] 接口未破坏：`TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有签名未改；`QuantConfig` 只作推理参数，不进入 `ModelConfig`；所有新逻辑在 `tinylm/inference/quant.py`。

## 第 24 章交接

### 本章摘要

本章把测试时计算扩展拆成“覆盖率”与“选择”两个瓶颈。直觉部分先说明三条路线对应三种信号强度：答案可验证用 Best-of-N，答案可比较用自洽性，过程可打分用树搜索；再把成本拆成采样、打分与搜索开销，并指出样本数、思考长度、模型规模是三条不同的加码轴。数学部分给出覆盖率 (24.1)、达到目标覆盖率的样本下界 (24.2) 与 pass@k 无偏估计 (24.3)；多数投票 (24.4) 及其在 $$p<0.5$$、错误相关时的失效，加权投票 (24.5)。验证器部分给出选择成功率 (24.6)，用 $$q=1$$ 与 $$q=0.5$$ 两个端点说明“选择瓶颈经常比生成瓶颈更紧”，并讨论 ORM、PRM、可执行验证器、模型裁判与奖励作弊。树搜索部分给出 PUCT (24.7) 与节点价值 (24.8)，用陷阱树说明小预算下搜索可能不如随机采样。预算部分给出总成本 (24.9)、边际分配 (24.10)、提前停止 (24.11) 与效用/过度思考 (24.12)。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (24.1) | $$P_{\mathrm{cov}}(N)=1-(1-p)^N$$ | 覆盖率 |
| (24.2) | $$N\ge\ln\delta/\ln(1-p)\approx-\ln\delta/p$$ | 达到目标覆盖率的样本下界 |
| (24.3) | $$\mathrm{pass@}k=1-\binom{n-c}{k}/\binom{n}{k}$$ | pass@k 无偏估计 |
| (24.4) | $$A_{\mathrm{vote}}(p,N)=\sum_{i=\lfloor N/2\rfloor+1}^{N}\binom{N}{i}p^i(1-p)^{N-i}$$ | 多数投票准确率 |
| (24.5) | $$\hat a=\arg\max_a\sum_i w_i\mathbf{1}[a_i=a]$$ | 加权投票 |
| (24.6) | $$A_{\mathrm{sel}}=\sum_k\binom{N}{k}p^k(1-p)^{N-k}\frac{kq}{kq+(N-k)(1-q)}$$ | 验证器选择成功率 |
| (24.7) | $$U(s,a)=Q(s,a)+c_{\mathrm{puct}}P(s,a)\frac{\sqrt{N(s)+1}}{1+N(s,a)}$$ | PUCT 选择分数 |
| (24.8) | $$Q(s,a)=\frac{1}{N(s,a)}\sum_{i=1}^{N(s,a)}v_i$$ | 节点平均价值 |
| (24.9) | $$C=N(T_{\mathrm{prompt}}+T_{\mathrm{out}})+N_{\mathrm{verify}}T_{\mathrm{verify}}$$ | 测试时总成本 |
| (24.10) | $$a^\star=\arg\max_a\Delta A_a/\Delta C_a$$ | 边际预算分配 |
| (24.11) | $$D_n\ge\delta,\ n\ge n_{\min}$$ | 提前停止 |
| (24.12) | $$U(n)=A(n)-\lambda C(n)$$ | 准确率—成本效用 |

### 实验与关键数字

- 设计：覆盖率与投票用解析公式加 5000 次蒙特卡洛；验证器选择用 20000 次分数模拟；树搜索用 7 叶陷阱树、2000 次随机叶对照；TinyLM 用 29 符号两跳算术（841 题）、`d_model=64`、2 层、4 查询头、2 KV 头，训练 600/1000 步、批 128；提前停止用二选一、$$p=0.65$$、1000 题。
- 覆盖率：$$p=0.10$$ 时 $$N=1,4,16,64$$ 为 0.100/0.344/0.815/0.999；$$p=0.05$$ 时 $$N=64$$ 为 0.962。
- 多数投票：$$p=0.65$$、$$N=15$$ 为 0.887；$$p=0.35$$、$$N=15$$ 为 0.113；$$p=0.50$$ 恒为 0.500；模拟单一错误答案 0.891、多个错误答案 0.999。
- 验证器：$$p=0.10$$、$$N=64$$，$$q=1.0/0.9/0.7/0.5$$ 的选择成功率为 0.999/0.482/0.203/0.100；干净分数模拟 0.984，给错误候选加 1.5 分后 0.705。
- 陷阱树：贪心 0.800/0.000；随机 8 叶 0.942/0.716；PUCT 预算 4、8 为 0.800/0.000，预算 12、16 为 1.000/1.000（“平均最佳值/找到最优”）。
- TinyLM：600 步贪心 0.848，投票@1/4/8/16 为 0.559/0.654/0.731/0.775；1000 步贪心 0.918，投票为 0.799/0.851/0.888/0.901；每样本 5 个 Token，训练耗时约 5.4 s 与 9.0 s。
- 提前停止：不停止 32.0 样本/0.965；$$\delta=2$$ 为 5.0/0.811；$$\delta=3$$ 为 7.9/0.879；$$\delta=5$$ 为 14.6/0.942。
- 测试：18 项全部通过（约 0.9 s），覆盖配置校验、覆盖率边界与蒙特卡洛、pass@k 端点、投票单调方向、平局、加权投票、Best-of-N 平局、选择率端点、提前停止、预算耗尽、PUCT 公式、预算为 0、陷阱树与终止节点。

### 留给后续章节的伏笔

- **第 25 章（多模态）**：图像/视频 Token 的采样成本更高，验证器要跨模态；覆盖率与选择率的分解仍然成立，但 $$p$$ 与 $$q$$ 的测量方式要重做。
- **第 26 章（智能体）**：工具调用提供天然的可执行验证器，适合 Best-of-N 与树搜索；长程任务的中间状态更多、回滚更贵，预算分配与提前停止更关键。
- **第 27 章（评测）**：pass@k 是 oracle 指标，不能当作部署准确率；报告测试时扩展结果时必须同时给出采样预算、温度、验证器与选择率，否则数字不可比。
- **第 23 章（量化）**：低比特模型改变 $$p$$、样本多样性与验证器校准；INT4 上的覆盖率与最优预算必须重测，不能按 FP16 曲线缩放。
- **第 20 章（解码策略）**：温度、Top-k/Top-p 决定单次成功率与样本相关性；测试时扩展的“最优温度”是任务相关量，应在验证集上扫描。
- **第 21、22 章（推理加速）**：$$N$$ 条样本使 KV 占用与调度开销按倍增长；分页缓存、连续批处理与投机解码可以摊薄单条成本，但树搜索的树形注意力与回滚会带来额外复杂度。
- **第 18 章（RLVR）**：可验证奖励既能训练长思维链，也能充当推理时的验证器；训练时扩展与测试时扩展是互补而非替代关系。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、方案对比、代码、工业实践、论文导读、常见误区、「本章的取舍」、习题齐全。
- [x] 字数：正文 8560 个中文字符（剔除代码块与公式），在其余章 6,000–9,000 预算内，未超过作者放宽的 10,000。
- [x] 公式编号 (24.1)–(24.12) 连续；覆盖率、样本下界、pass@k、多数/加权投票、选择率、PUCT、节点价值、成本、边际分配、提前停止与效用均有推导或说明。
- [x] 至少 1 张对比表（表 24-1 六条路线）；工业案例含 AlphaCode、Large Language Monkeys、OpenAI o1、DeepSeek-R1、PRM800K、vLLM/SGLang；论文导读 5 篇；误区 6 条。
- [x] 代码可运行：从章内 Markdown 重新提取 `tinylm/inference/search.py` 与 `examples/ch24_search.py`，与已验证源文件逐字一致；在 `/tmp/ch24final` 重跑，除训练耗时外输出一致，18 项测试通过。
- [x] 公式与代码对应：`coverage_probability` 标注 (24.1)，`pass_at_k` 标注 (24.3)，`majority_accuracy` 标注 (24.4)，`majority_vote` 标注 (24.5)，`selection_accuracy` 标注 (24.6)，`puct_score` 标注 (24.7)，`TreeNode.q` 标注 (24.8)，`token_cost` 标注 (24.9)，`should_stop` 标注 (24.11)。
- [x] 新符号已登记 `notation.md` 与附录 B：$$P_{\mathrm{cov}},p,N,\delta,k,c,q,A_{\mathrm{vote}},A_{\mathrm{sel}},w_i,D_n,n_{\min},c_{\mathrm{puct}},P(s,a),N(s),N(s,a),Q(s,a),T_{\mathrm{prompt}},T_{\mathrm{out}},T_{\mathrm{verify}},\lambda$$ 等，并说明与第 11、12、18、20、21 章的重名冲突。
- [x] 新术语已登记 `glossary.md` 与附录 C：测试时计算扩展、覆盖率、选择、自洽性、加权投票、Best-of-N、结果/过程奖励模型、验证器、奖励作弊、PUCT、思维树、MCTS、过度思考、提前停止、pass@k；附录 C 已有的测试时计算、Best-of-N、自洽性、树搜索、MCTS、预算控制、过度思考未重复添加。
- [x] GitBook 语法：hint 使用 warning/success；1 个 Mermaid、1 张表、1 个 `<details>`；代码块均标注语言；无落单 `$`，无外部图片。
- [x] 事实性数字可复算：解析公式与模拟数字由固定种子程序输出；AlphaCode、Large Language Monkeys、o1/R1、PRM800K 的数字来自公开论文/技术报告并在正文标注条件；时效性内容写“截至写作时”。
- [x] 实验诚实报告：明确标注 TinyLM 规模小、合成任务可约、训练耗时随机器变化；保留负结果（投票@16 不如强贪心、PUCT 小预算不如随机采样、提前停止降准确率）；选择率公式标注为成对判别近似。
- [x] 接口未破坏：`TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有签名未改；`SearchConfig` 只作推理参数，不进入 `ModelConfig`；所有新逻辑在 `tinylm/inference/search.py`。

## 第 25 章交接

### 本章摘要

本章把多模态大模型拆成视觉编码、模态对齐与语言建模三段。图像按 (25.1) 切成块，序列长度 (25.2) 与注意力倍数 (25.3) 决定预填充成本；ViT 用 (25.4) 的块序列编码图像，CLIP 用 (25.5) 的对称 InfoNCE 做图文对比预训练；投影层用线性层、MLP 或 (25.6) 的重采样器完成维度与尺度翻译，压缩比由 (25.7) 给出；语言模型在 (25.8) 的下一 Token 目标下联合训练。理解与生成的统一靠 (25.9)–(25.10) 的离散 VQ Token 化，或 Transfusion 式混合扩散。正文区分晚期融合与早期融合，讨论高分辨率切块、位置编码、标签掩码、幻觉与能力边界，并用 12×12 合成视觉问答核对“去掉图像会退化”与“Token 数不是唯一变量”。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (25.1) | $$N_p=HW/p^2$$ | 图像块数 |
| (25.2) | $$L=N_p+T_{\mathrm{text}}$$ | 拼接后的序列长度 |
| (25.3) | $$R_{\mathrm{attn}}=(1+N_p/T_{\mathrm{text}})^2$$ | 相对纯文本的注意力算力倍数 |
| (25.4) | $$\mathbf{z}_0=[\mathbf{x}_{\mathrm{cls}};\mathbf{E}\mathbf{x}_p^1;\dots]+\mathbf{E}_{\mathrm{pos}}$$ | ViT 输入序列 |
| (25.5) | $$\mathcal{L}_{\mathrm{CLIP}}$$ 对称 InfoNCE | 图文对比预训练 |
| (25.6) | $$\mathbf{H}=\mathrm{softmax}(\mathbf{Q}\mathbf{K}^{\top}/\sqrt{d})\mathbf{V}$$ | 重采样器交叉注意力 |
| (25.7) | $$\rho_{\mathrm{vis}}=N_p/M$$ | 视觉 Token 压缩比 |
| (25.8) | $$\mathcal{L}_{\mathrm{VLM}}=-\sum_t\log p_\theta(y_t\mid y_{<t},\mathrm{Proj}(\mathrm{ViT}(\mathbf{X})))$$ | 多模态语言建模损失 |
| (25.9) | $$k^\star=\arg\min_k\lVert\mathbf{z}-\mathbf{e}_k\rVert_2^2$$ | VQ 最近邻量化 |
| (25.10) | $$\mathcal{L}_{\mathrm{VQ}}=\lVert\mathbf{z}-\mathrm{sg}[\mathbf{z}_q]\rVert_2^2+\beta\lVert\mathrm{sg}[\mathbf{z}]-\mathbf{z}_q\rVert_2^2$$ | 码本损失与承诺损失 |

### 实验与关键数字

- 设计：12×12 合成图，3 形状 × 3 颜色，问题为形状或颜色，6 类答案；4096 训练 / 800 测试，300 步、批 64、3 个种子；`d_model=32`、2 层文本 Transformer、视觉塔 32 维 2 层。
- Token 预算（文本 128 Token）：336/14 未合并 576 Token、注意力 30.25 倍；2×2 合并 144 Token、4.52 倍；448/14 未合并 1024 Token、81 倍；合并 256 Token、9 倍；896/14 合并 1024 Token、81 倍。
- InfoNCE（batch 64）：独立随机配对 4.184，对齐配对 0.001，错位配对 14.148，随机水平 $$\ln 64=4.159$$。
- VQ（2048 个 8 维特征，16 个真实簇）：码本 4/8/16/32 的 MSE 为 2.3056/1.5658/0.5973/0.0816；码字范围 0–14。
- 合成 VQA：线性投影 0.912±0.016；MLP 投影 0.884±0.046；4 Token 0.907±0.033；36 Token 0.892±0.084；冻结随机视觉塔 0.838±0.064；去掉图像 0.159（六分类随机水平 1/6）。
- 训练耗时（CPU 单线程，随机器变化）：线性 12.4 s、MLP 12.7 s、4 Token 6.0 s、36 Token 29.7 s、冻结 8.9 s。
- 测试：12 项全部通过（约 1 s），覆盖 Token 计数与合并、注意力倍数、卷积/线性切块等价、池化形状、三种投影、占位符校验、梯度路径、冻结参数、答案位损失、对比损失、VQ 最近邻。

### 留给后续章节的伏笔

- **第 26 章（智能体）**：图像、截图与工具返回值都是环境观测；多模态观测会放大轨迹长度与上下文成本，工具调用的 Token 预算要按图像 Token 重算。
- **第 27 章（评测）**：VQA、OCR、图表与幻觉要分维度报告；必须写清分辨率、图像 Token 数、是否使用 OCR 与评测集版本，否则分数不可比。
- **第 28 章（可解释性）**：图像 Token 与文本 Token 共享残差流，跨模态注意力与视觉特征回路是机制可解释性的新对象。
- **第 29 章（前沿）**：原生多模态、统一理解与生成、长视频与动态分辨率仍是开放问题；离散 Token 化的细节损失与训练稳定性尚无定论。
- **第 23 章（量化）**：视觉塔、投影层与图像 Token 的 KV Cache 都可以量化；低比特对细粒度 OCR 的影响需要单独测量。
- **第 21 章（推理加速）**：图像 Token 进入预填充与 KV 账本；视觉特征可按图像哈希缓存，分块预填充与连续批处理要处理变长图像序列。
- **第 24 章（测试时扩展）**：高分辨率裁切、多视角与“放大再看”可以看作多模态的测试时计算；验证器要跨模态，覆盖率与选择率的分解仍成立。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、方案对比、代码、工业实践、论文导读、常见误区、「本章的取舍」、习题齐全。
- [x] 字数：正文 7700 个中文字符（剔除代码块与公式），在前沿章 7,000–9,000 预算内。
- [x] 公式编号 (25.1)–(25.10) 连续；切块、序列长度、注意力倍数、ViT、InfoNCE、重采样器、压缩比、VLM 损失、VQ 量化与损失均有推导或说明。
- [x] 至少 1 张对比表（表 25-1 六条路线）；工业案例含 CLIP、Flamingo、BLIP-2、LLaVA、Qwen-VL、InternVL、Chameleon、Emu3、Transfusion；论文导读 5 篇；误区 5 条。
- [x] 代码可运行：从章内 Markdown 重新提取 `tinylm/multimodal.py` 与 `examples/ch25_multimodal.py`，在 `/tmp/ch25final` 运行，除训练耗时外输出一致；12 项测试通过。
- [x] 公式与代码对应：`image_token_count` 标注 (25.1)/(25.7)，`attention_multiplier` 标注 (25.3)，`PatchEmbed`/`VisionEncoder` 标注 (25.4)，`contrastive_loss` 标注 (25.5)，`TokenResampler` 标注 (25.6)，拼接标注 (25.8)，`vq_quantize` 标注 (25.9)，`vq_loss` 标注 (25.10)。
- [x] 新符号已登记 `notation.md` 与附录 B：$$N_p,H,W,p,C,d_v,T_{\mathrm{text}},L,R_{\mathrm{attn}},z_0,\mathbf{E},\mathbf{E}_{\mathrm{pos}},z_l,\tau,L_{\mathrm{CLIP}},K,Z_v,M,m,\rho_{\mathrm{vis}},L_{\mathrm{VLM}},e_k,z_q,\beta,\mathrm{sg},L_{\mathrm{VQ}}$$ 等，并说明与第 3、6、14、16、21、23、24 章的重名冲突。
- [x] 新术语已登记 `glossary.md` 与附录 C：晚期融合、早期融合、模态对齐、视觉塔、连接器、图像占位符、视觉指令微调、重采样器、动态分辨率、码本、向量量化、视觉问答、码本坍塌；附录 C 已有的多模态大模型、视觉编码器、投影层、视觉语言模型、原生多模态、图像 Token 化未重复添加。
- [x] GitBook 语法：hint 使用 info/danger/success；1 个 Mermaid、1 张表、1 个 `<details>`；代码块均标注语言；无落单 `$`（代码块内的 shell 提示符除外），无外部图片。
- [x] 事实性数字有来源或可复算：Token 预算、InfoNCE、VQ 与 VQA 数字由固定种子程序输出；CLIP、LLaVA、BLIP-2、Qwen-VL、InternVL、Chameleon、Emu3、Transfusion 的配置来自公开论文/技术报告并在正文标注条件；GPT-4V/Gemini 未公开细节已标注为推测。
- [x] 实验诚实报告：明确标注 12×12、4096 样本、300 步的玩具规模；保留负结果（MLP 不优于线性、4 Token 不劣于 16 Token、36 Token 方差大）；冻结视觉塔一行明确是随机初始化冻结，不得外推为预训练冻结策略。
- [x] 接口未破坏：`TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 的现有签名未改；`VisionConfig` 只作视觉分支配置，不进入 `ModelConfig`；所有新逻辑在 `tinylm/multimodal.py`。

## 第 26 章交接

### 本章摘要

本章把智能体写成“策略—工具—观测”的闭环。工具由 (26.1) 的名字、参数模式与函数三元组给出；模型看到的是观测历史 (26.2)，对应部分可观测决策过程；轨迹概率与期望回报由 (26.3)–(26.4) 定义，与第 18 章策略梯度同形，但动作空间是自然语言与结构化调用的混合。成本侧，上下文按 (26.5) 逐步增长，前 $$T$$ 步输入 Token 总数由 (26.6) 给出，成本 (26.7) 因而近似随步数平方增长；可靠性侧，重试成功率 (26.8) 与多步任务成功率 (26.9) 解释“任务越长越脆弱”，并强调失败相关性与副作用对独立假设的破坏；工具选择用单位成本期望收益 (26.10) 排序。训练上，SFT 只对动作 Token 计损 (26.11)，RL 用轨迹回报加 KL 约束 (26.12)。正文对比六条路线，给出参数模式、最小权限、沙箱超时、幂等重试、输出截断、注入防线与可观测性等上线约束，并用脚本策略验证控制流。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (26.1) | $$\mathcal{T}=\{(n_j,\sigma_j,f_j)\}_{j=1}^{J},\ f_j:\mathcal{A}_j\to\mathcal{O}_j\cup\{\bot\}$$ | 工具三元组：名字、模式、函数 |
| (26.2) | $$h_t=(q,a_1,o_1,\dots,a_{t-1},o_{t-1})$$ | 观测历史 |
| (26.3) | $$p_\theta(\tau\mid q)=\prod_{t=1}^{T}\pi_\theta(a_t\mid h_t)P_{\mathrm{env}}(o_t\mid h_t,a_t)$$ | 轨迹概率 |
| (26.4) | $$J(\theta)=\mathbb{E}_{\tau\sim p_\theta}[R(\tau)]$$ | 期望回报 |
| (26.5) | $$L_{t+1}=L_t+\lvert a_t\rvert+\lvert o_t\rvert$$ | 上下文递推 |
| (26.6) | $$\sum_{t=1}^{T}L_t=T L_0+\sum_{t=1}^{T}(T-t)(\lvert a_t\rvert+\lvert o_t\rvert)$$ | 输入 Token 总数 |
| (26.7) | $$C=c_{\mathrm{in}}\sum_t L_t+c_{\mathrm{out}}\sum_t\lvert a_t\rvert$$ | 轨迹成本 |
| (26.8) | $$P_{\mathrm{retry}}(p,b)=1-(1-p)^b$$ | 重试后单步成功率 |
| (26.9) | $$P_{\mathrm{task}}(p,b,K)=[1-(1-p)^b]^K$$ | 多步任务成功率 |
| (26.10) | $$\mathrm{VoI}(j)=(\mathbb{E}[u(o)\mid a_j]-u_0)/C_j$$ | 单位成本期望收益 |
| (26.11) | $$\mathcal{L}_{\mathrm{SFT}}=-\sum_t m_t\log\pi_\theta(a_t\mid h_t)$$ | 动作掩码监督损失 |
| (26.12) | $$J_{\mathrm{RL}}(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]-\beta\,\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}})$$ | 带 KL 的轨迹级 RL |

### 实验与关键数字

- 设计：纯标准库实现 `tinylm/agent/loop.py`；参数校验、重复调用拦截、上下文预算、步数上限、可选“工具错误即停”；成本按 UTF-8 字节数除以 4 估算 Token，仅用于相对账本。
- 两步工具调用：17×6=102，再乘 1.08 得 110.16，最终答案 110；3 步，轨迹成本 250.0（`c_in=1, c_out=4`）。
- 重复调用：`max_steps=4, max_repeat=1`，第 2–4 步均返回 `repeated call rejected: calc`，以 `max_steps` 停止。
- 参数校验：传入 `expression: 42`（int）被拒，错误作为观测回填，模型下一步给出“无法计算”。
- 注入演示：最小权限注册表把 `delete_file` 变成 `unknown tool`；过度授权注册表执行了演示桩，返回 `simulated delete: /tmp/report`（不触碰文件系统）。
- 蒙特卡洛：每格 20,000 次、固定种子；`p∈{0.5,0.7,0.9}`、`b∈{1,2,3}`、`K=3`，经验值与 (26.8)–(26.9) 的偏差不超过约 0.004。例：`p=0.9,b=1,K=3` 经验 0.729 / 公式 0.729；`p=0.7,b=2` 的 `K=1..4` 为 0.910/0.830/0.756/0.688，公式为 0.910/0.828/0.754/0.686。
- 成本随深度增长：`K=1→3` 时步数 2→4，上下文 Token 34→162，轨迹成本 98.0→322.0，验证 (26.6)/(26.7) 的二次效应。
- VoI：工具 A（+0.3/100 Token）为 0.0030，工具 B（+0.4/1000 Token）为 0.0004。
- 重试阈值：`p=0.9` 时达到 0.99 成功率至少需 `b=2`。
- 测试：15 项全部通过（约 0.03 s），覆盖配置校验、带推理的 JSON 解析、歧义/畸形动作拒绝、未知工具、缺参/多参/类型错误、工具异常转观测、正常作答、步数上限、重复拦截、错误即停、上下文上限、观测截断、重试与任务公式端点、成本复算、VoI 与 Token 估算。

### 留给后续章节的伏笔

- **第 27 章（评测方法论）**：智能体要按轨迹评测，报告成功率、步数分布、成本分布、工具错误率与副作用次数；固定任务集、工具版本与初始状态，否则分数不可比。本章的 `AgentResult` 与逐步日志正是轨迹级指标的最小载体。
- **第 21 章（推理加速）**：前缀缓存只能降低重复计算的算力，不能消除 (26.6) 的输入 Token 账本；连续批处理要处理智能体轨迹的变长与暂停/恢复。
- **第 24 章（测试时计算）**：Best-of-N 与树搜索可作用在“每一步动作”或“整条轨迹”上；搜索宽度与深度直接放大 (26.7) 的成本，验证器质量决定收益。
- **第 18 章（RLVR/GRPO）**：轨迹级回报稀疏且延迟，信用分配比单轮生成更难；可执行验证器是智能体 RL 的关键基础设施。
- **第 16–17 章（RLHF/DPO）**：动作格式与安全行为可用偏好数据修正，但工具调用的正确性更适合用可执行反馈而非人工偏好判断。
- **第 14 章（SFT）**：工具输出必须做损失掩码，否则模型会学会复述观测；对话模板要能容纳 `tool_use`/`tool_result` 这类结构化块。
- **第 25 章（多模态）**：截图、图像与视频都是环境观测，图像 Token 会放大 (26.5)–(26.7) 的成本；多模态智能体的轨迹预算要按视觉 Token 重算。
- **第 19 章（蒸馏）**：用小模型执行高频、低风险的子步骤（如动作解析、工具选择）可以降低单步成本，但错误会沿轨迹累积。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、方案对比、代码、工业实践、论文导读、常见误区、「本章的取舍」、习题齐全。
- [x] 字数：正文 7871 个中文字符（剔除代码块与公式），在前沿章 7,000–9,000 预算内。
- [x] 公式编号 (26.1)–(26.12) 连续，习题答案另有 (26.13)–(26.15)；工具定义、历史、轨迹概率、回报、上下文递推、Token 总数、成本、重试、任务成功率、VoI、SFT 与 RL 目标均有推导或说明。
- [x] 至少 1 张对比表（表 26-1 六条路线）；工业案例含 OpenAI function calling/tools、Anthropic tool use、MCP、ReAct、Toolformer、ToolLLM、SWE-agent、AgentBench；论文导读 5 篇；误区 5 条。
- [x] 代码可运行：从章内 Markdown 重新提取 `tinylm/agent/loop.py` 与 `examples/ch26_agent.py`，在 `/tmp/ch26verify` 运行；15 项测试通过，示例输出与正文逐字一致（`diff` 为 0）。
- [x] 公式与代码对应：`ToolSpec` 标注 (26.1)，上下文初值与追加标注 (26.2)/(26.5)，`trajectory_cost`/`result_cost` 标注 (26.7)，`retry_success` 标注 (26.8)，`task_success` 标注 (26.9)，`value_of_information` 标注 (26.10)。
- [x] 新符号已登记 `notation.md` 与附录 B：$$q,a_t,o_t,h_t,s_t,\pi_\theta,P_{\mathrm{env}},\tau,R(\tau),L_t,C,c_{\mathrm{in}},c_{\mathrm{out}},p,b,K,u,\mathrm{VoI},m_t,\beta$$ 等，并说明与 top-k、bias、VQ/KL、采样概率、温度的冲突。
- [x] 新术语已登记 `glossary.md` 与附录 C：智能体、工具调用、ReAct、轨迹、环境、智能体强化学习、提示注入、最小权限、幂等、沙箱、人工确认、计划—执行、动作空间、观测、停止条件、信息价值、工具模式；附录 C 已有的智能体相关词条未重复添加。
- [x] GitBook 语法：hint 使用 info/warning/success；1 个 Mermaid、1 张表、1 个 `<details>`；代码块均标注语言；无落单 `$`，无外部图片。
- [x] 事实性数字有来源或可复算：Retry/成本/VoI 数字由固定种子程序输出；OpenAI、Anthropic、MCP、ReAct、Toolformer、ToolLLM、SWE-agent、AgentBench 的配置与结论来自公开文档/论文并在正文标注条件；API 时效性已用 warning 提示。
- [x] 实验诚实报告：脚本策略验证的是控制流而非模型能力；蒙特卡洛固定种子、2 万次；独立性假设、失败相关性、幂等与副作用限制均明确讨论；`delete_file` 是演示桩，不触碰文件系统。
- [x] 接口未破坏：`TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 未改；智能体模块独立于模型，策略以 `Callable[[str], str]` 注入，接 TinyLM 只需包装 `generate`。

## 第 27 章交接

### 本章摘要

本章把评测写成“带噪声的测量”。指标侧，准确率 (27.1)、macro-F1 (27.2) 与 pass@k (27.3) 分别对应逐样本、按类别与多次采样；不确定性侧，标准误 (27.4)、Wilson 区间 (27.5)、配对 bootstrap (27.6) 与 McNemar (27.7) 回答“差距是否真实”，Bonferroni (27.8) 与样本量公式 (27.9) 处理多重比较与功效；数据侧，n-gram 污染率 (27.10) 只是第一道检查，还需扰动对照、时间切分与私有留出集；校准侧，ECE (27.11) 与覆盖率—风险曲线描述置信度质量；偏好侧，Bradley–Terry (27.12)、Elo (27.13) 与 Cohen kappa (27.14) 处理排名与标注一致性；决策侧，成本感知效用 (27.15) 把质量与代价放在同一尺度。智能体评测强调 pass^k (27.16) 与聚类有效样本量 (27.17)，而不是只看 pass@k。正文给出可复现报告的字段清单与常见误区。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (27.1) | $$\mathrm{Acc}=\frac{1}{n}\sum_i\mathbf{1}[\hat y_i=y_i]$$ | 准确率 |
| (27.2) | $$\mathrm{F1}_{\mathrm{macro}}=\frac{1}{C}\sum_c\frac{2\mathrm{TP}_c}{2\mathrm{TP}_c+\mathrm{FP}_c+\mathrm{FN}_c}$$ | 宏平均 F1 |
| (27.3) | $$\mathrm{pass@}k=1-\binom{n-c}{k}/\binom{n}{k}$$ | pass@k 无偏估计 |
| (27.4) | $$\mathrm{SE}(\hat p)=\sqrt{\hat p(1-\hat p)/n}$$ | 二项比例标准误 |
| (27.5) | $$\frac{\hat p+\frac{z^2}{2n}\pm z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}}$$ | Wilson 置信区间 |
| (27.6) | $$\Delta^{*}=\frac{1}{n}\sum_i(s_i^{A*}-s_i^{B*})$$ | 配对 bootstrap 差值 |
| (27.7) | $$p=2\sum_{i=0}^{\min(b,c)}\binom{b+c}{i}2^{-(b+c)}$$ | McNemar 精确双侧 p |
| (27.8) | $$\alpha'=\alpha/m$$ | Bonferroni 阈值 |
| (27.9) | $$n\approx2(z_{1-\alpha/2}+z_{1-\beta})^2p(1-p)/\Delta^2$$ | 每组样本量估算 |
| (27.10) | $$r_n=\frac{1}{M}\sum_i\mathbf{1}[\mathcal{G}_n(x_i^{\mathrm{eval}})\cap\mathcal{G}_n(\mathcal{D}_{\mathrm{train}})\neq\varnothing]$$ | n-gram 污染率 |
| (27.11) | $$\mathrm{ECE}=\sum_b\frac{n_b}{n}\lvert\mathrm{acc}(b)-\mathrm{conf}(b)\rvert$$ | 期望校准误差 |
| (27.12) | $$P(i\succ j)=\frac{1}{1+10^{-(R_i-R_j)/400}}$$ | Bradley–Terry 胜率 |
| (27.13) | $$R_i'=R_i+K(1-E_i),\ E_i=\frac{1}{1+10^{(R_j-R_i)/400}}$$ | Elo 更新 |
| (27.14) | $$\kappa=\frac{p_o-p_e}{1-p_e}$$ | Cohen kappa |
| (27.15) | $$U=\mathrm{score}-\lambda\cdot\mathrm{cost}$$ | 成本感知效用 |
| (27.16) | $$\mathrm{pass}^k=p_{\mathrm{task}}^{\,k}$$ | 每次尝试都成功 |
| (27.17) | $$n_{\mathrm{eff}}=n/\mathrm{DEFF},\ \mathrm{DEFF}=1+(m-1)\rho$$ | 聚类设计效应 |
| (27.18) | $$\mathrm{SE}=\sqrt{0.7\times0.3/1000}\approx0.0145$$ | 习题：标准误实算 |

### 实验与关键数字

- 设计：纯标准库实现 `tinylm/eval/metrics.py`；逐样本输入、固定种子；bootstrap 4000 次；pass@k 模拟 20000 次。
- 配对比较（n=1000）：A 准确率 0.700，标准误 0.0145，Wilson [0.671, 0.728]；B 0.720，Wilson [0.691, 0.747]；配对差 B-A=-0.020，bootstrap 95% 区间 [-0.035, -0.005]；不一致对 b=20、c=40，McNemar 精确 p=0.0135。
- pass@k（n=10, c=2）：k=1..5 的公式值 0.2000/0.3778/0.5333/0.6667/0.7778；2 万次模拟值 0.2034/0.3735/0.5332/0.6679/0.7790。
- 污染检测：3 条评测文档，n=5 命中 0.667，n=13 命中 0.333。
- 裁判位置偏差：只按 A 在前时 A 胜率 0.603；交换顺序后 0.503。Bradley–Terry 分差 0/100/200 的概率 0.500/0.640/0.760；1500 战胜 1700 后 Elo 为 1524.3/1675.7。
- 校准：构造每箱准确率比置信度低 0.05，ECE=0.050。
- 一致性：一致率 0.800、机会一致率 0.500 时 kappa=0.600；macro-F1 0.778 对应准确率 0.750。
- 成本效用（λ=0.02）：A 0.70，B 0.68。
- 测试：12 项全部通过（约 0.4 s，耗时随机器变化，章内用 `sed` 去掉耗时以便逐字复现），覆盖每个指标的端点、边界与错误输入。

### 留给后续章节的伏笔

- **第 28 章（可解释性）**：探针与因果干预同样需要置信区间、对照与显著性；“某个头重要”必须有可复现的干预实验，而不是只看注意力权重。
- **第 29 章（前沿与开放问题）**：持续学习、合成数据与智能体 RL 的进展都要用本章的评测框架衡量；基准饱和与 Goodhart 效应会直接影响“前沿”判断。
- **第 26 章（智能体）**：`AgentResult` 的逐步记录是轨迹级评测的数据源；成功率之外还要报步数、成本、工具错误率与副作用次数。
- **第 24 章（测试时计算）**：pass@k 与预算控制共享同一套抽样统计；比较不同采样预算时必须固定 k、n 与成本口径。
- **第 10 章（数据工程）**：MinHash/LSH 去重可直接用于训练—评测污染检测；n-gram 只是最便宜的初筛。
- **第 18 章（RLVR/GRPO）**：可执行验证器既是训练信号也是评测器；验证器偏差会被策略利用，需用独立留出集与对抗样本检查。
- **第 19 章（蒸馏）**：学生模型的评测要同时看能力、成本与校准，不能只比平均分。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、方案对比、代码、工业实践、论文导读、常见误区、「本章的取舍」、习题齐全。
- [x] 字数：正文 7664 个中文字符（剔除代码块与公式），在前沿章 7,000–9,000 预算内。
- [x] 公式编号 (27.1)–(27.18) 按正文出现顺序连续；指标、区间、检验、污染、校准、排名、智能体统计均有推导或说明。
- [x] 至少 1 张对比表（表 27-1 七类路线）；工业案例含 HELM、MMLU/MMLU-Pro、GSM8K/MATH、HumanEval/MBPP、MT-Bench、Chatbot Arena、SWE-bench/AgentBench/WebArena、LiveBench；论文导读 5 篇；误区 5 条。
- [x] 代码可运行：从章内 Markdown 重新提取 `tinylm/eval/metrics.py` 与 `examples/ch27_eval.py`，在 `/tmp/ch27verify` 运行；12 项测试通过，示例输出与正文逐字一致。
- [x] 公式与代码对应：`accuracy` (27.1)、`macro_f1` (27.2)、`pass_at_k` (27.3)、`binomial_se` (27.4)、`wilson_interval` (27.5)、`paired_bootstrap_diff` (27.6)、`mcnemar_test` (27.7)、`contamination_rate` (27.10)、`expected_calibration_error` (27.11)、`bradley_terry_prob` (27.12)、`elo_update` (27.13)、`cohen_kappa` (27.14)、`cost_aware_utility` (27.15)。
- [x] 新符号已登记 `notation.md` 与附录 B：$$n,M,C,\mathrm{TP}_c,\mathrm{FP}_c,\mathrm{FN}_c,\hat p,p,k,z,\Delta,b,c,\alpha,m,\mathcal{G}_n,r_n,B,n_b,\mathrm{ECE},R_i,K,E_i,\kappa,p_o,p_e,U,\lambda,\rho,\mathrm{DEFF},n_{\mathrm{eff}}$$ 等，并说明与 top-k、专家数、KL 系数、切块边长、尝试次数的冲突。
- [x] 新术语已登记 `glossary.md` 与附录 C：基准、评测协议、数据污染、提示词敏感性、配对比较、Bradley–Terry、Elo、Cohen kappa、期望校准误差、选择性预测、聚类 bootstrap、效应量、基准饱和、系统卡、标准误、Wilson 区间；附录 C 已有的 LLM 作为裁判、竞技场未重复添加。
- [x] GitBook 语法：hint 使用 info/warning/success；1 个 Mermaid、1 张表、1 个 `<details>`；代码块均标注语言；无落单 `$`，无外部图片。
- [x] 事实性数字有来源或可复算：配对、pass@k、污染、校准、kappa 与成本效用数字由固定种子程序输出；MMLU/MMLU-Pro、GSM8K/MATH、HumanEval/MBPP、MT-Bench、Chatbot Arena、SWE-bench/AgentBench/WebArena、HELM 的配置与结论来自公开论文/文档并在正文标注条件；排行榜时效性已加 warning。
- [x] 实验诚实报告：配对数据是构造的确定性样本，不是真实模型输出；pass@k 与位置偏差是模拟；污染示例只有 3 条文档，明确为演示；未把玩具数字外推为真实基准分数。
- [x] 接口未破坏：`TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 未改；评测模块独立于模型，只接收逐样本结果。

## 第 28 章交接

### 本章摘要

本章把可解释性写成“带干预的假设检验”。探针侧，线性分类器 (28.1) 只回答某属性是否可线性解码，必须用打乱标签或随机特征构造对照，并以选择性 (28.2) 扣除分类器自身学到的部分。注意力侧，(28.3) 表明权重只是值向量的混合系数，实际贡献还取决于值向量与输出投影；单头消融 (28.4) 给出因果影响的直接度量，但单头零效应可能来自其他头补偿，必须做组消融。激活修补侧，以对数几率差 (28.5) 为平滑指标，恢复率 (28.6) 量化反事实干预；损坏方式、位点、位置与指标这四个选择都会改变结论，选错位点会产生假阴性。SAE 侧，重构加 L1 的目标 (28.7) 把叠加激活分解为稀疏特征，用 L0 (28.8) 与特征纯度 (28.9) 评价，但特征可读不等于因果有效。最后强调模型版本、损坏分布、种子、样本量与多重比较校正等可复现要求。玩具复制任务展示了三个关键现象：可解码的信息可能不被使用、注意力权重高的头可能完全冗余、修补位点选错会产生零结果。

### 关键公式

| 编号 | 公式 | 说明 |
| --- | --- | --- |
| (28.1) | $$\min_{W}\ \frac{1}{n}\sum_{i=1}^{n}\mathrm{CE}\left(\mathrm{softmax}(Wh_i),y_i\right)+\lambda\lVert W\rVert_2^2$$ | 线性探针目标 |
| (28.2) | $$\mathrm{Sel}=\mathrm{Acc}_{\mathrm{probe}}-\mathrm{Acc}_{\mathrm{control}}$$ | 探针选择性 |
| (28.3) | $$\mathrm{Attn}_h(x)_t=\sum_{j\le t}\alpha_{tj}^{(h)}v_j^{(h)},\ \alpha_{tj}^{(h)}=\mathrm{softmax}_j\left(\frac{q_t^\top k_j}{\sqrt{d_h}}\right)$$ | 注意力输出分解 |
| (28.4) | $$\Delta_h=\mathcal{L}(x)-\mathcal{L}\left(x;\ \text{head }h\leftarrow 0\right)$$ | 单头消融的指标变化 |
| (28.5) | $$\mathrm{LD}(x)=\log p(y_{\mathrm{target}}\mid x)-\log p(y_{\mathrm{distractor}}\mid x)$$ | 对数几率差 |
| (28.6) | $$R=\frac{S_{\mathrm{patch}}-S_{\mathrm{corrupt}}}{S_{\mathrm{clean}}-S_{\mathrm{corrupt}}}$$ | 激活修补恢复率 |
| (28.7) | $$\mathcal{L}_{\mathrm{SAE}}=\left\lVert x-\left(fW_d+b_d\right)\right\rVert_2^2+\lambda\lVert f\rVert_1,\ f=\mathrm{ReLU}\left(W_e x+b_e\right)$$ | SAE 重构加稀疏目标 |
| (28.8) | $$\mathrm{L0}=\frac{1}{n}\sum_{i=1}^{n}\left\lVert f_i\right\rVert_0$$ | 平均非零特征数 |
| (28.9) | $$P_k=\frac{\max_c\lvert\{i:\arg\max_j f_{ij}=k,\ y_i=c\}\rvert}{\lvert\{i:\arg\max_j f_{ij}=k\}\rvert}$$ | 特征纯度 |
| (28.10) | $$R_A=\frac{1.0-(-2.0)}{4.0-(-2.0)}=\frac{3.0}{6.0}=0.5$$ | 习题：恢复率实算 |

### 实验与关键数字

- 设计：`tinylm/interpret/core.py` 提供探针、注意力、消融、修补、SAE 与方向投影工具；`examples/ch28_interpret.py` 在 2 层、4 头、`d_model=64`、词表 16 的 TinyLM 上训练复制首 Token 任务（200 步、批 128、AdamW lr 3e-3、wd 0.01、seed 0），1000 条留出样本（seed 99），准确率 1.000。
- 探针（层 1，训练 800 / 测试 200，300 步）：位置 1/2/5 准确率 1.000/0.975/1.000；打乱标签对照（位置 10）0.060，选择性 +0.940。
- 因果交换：改位置 5 填充词，仍复制原答案 1.000；改位置 1 被复制词，改为复制新词 1.000。说明位置 5 的信息可解码但不被读取。
- 最后一层对位置 1 的平均注意力：层 0 四个头均 1.000；层 1 为 0.103/0.162/0.120/0.128。
- 单头消融：层 0 四个头的平均 margin 变化约 -0.002/-0.001/-0.001/-0.001；层 1 均约 -0.000；层 0 四头同时消融准确率 0.136、margin -4.103，显示高度冗余。
- 激活修补恢复率：`tok_emb`@pos1 +1.000；`attn[0]`@pos10 +0.999；`attn[0]`@pos1 +0.017；`block[0]`@pos1 +0.019；`block[0]`@pos2 +0.032；`block[0]`@pos10 +0.999；`block[1]`@pos10 +1.000。错误位点给出零结果，作为假阴性示例。
- SAE（4000 条末位激活，字典 128）：λ=0/0.005/0.020/0.050 时 MSE 1.73e-4/1.43e-4/1.56e-4/2.22e-4，L0 56.05/55.89/43.60/27.24，死特征 0/0/0/1；λ=0.020 的特征纯度 1.000。
- 特征消融（负结果）：Token 4 最强特征 #57 平均激活 2.951；投影掉后 margin -7.853（原 -8.598，随机方向 -8.466），准确率 0.984（随机方向 1.000），特征可读但因果作用与随机方向接近。
- 测试：15 项全部通过，覆盖 margin、探针、注意力归一化与因果性、MLA 拒绝、hook 恢复、激活形状、恢复率端点、SAE 形状/范数/训练、特征纯度、方向投影与方向消融。

### 留给后续章节的伏笔

- **第 29 章（前沿与开放问题）**：持续学习、智能体强化学习与合成数据的“进展”都需要可解释性审计；解释工具能指出能力来自哪些部件，但不能替代评测与因果验证。
- **第 27 章（评测方法论）**：探针准确率、修补恢复率与特征效应的区间估计、配对比较与多重比较校正直接复用其 bootstrap 与效应量框架；只报点估计不足以支撑机制结论。
- **第 3、6 章（Transformer 与注意力变体）**：注意力的头粒度决定消融与修补的单位；GQA/MQA/MLA 的 K/V 共享会改变“头”的独立性，解释工具需按实际参数化调整。
- **第 21、23 章（推理加速与量化）**：KV Cache 量化与权重仅量化会改变激活分布，解释结论必须在目标推理配置下复现；量化误差可能放大或掩盖小效应。
- **第 18 章（RLVR/GRPO）**：策略更新会改变回路权重；训练前后同一探针或修补结论是否稳定，是检验解释是否抓住算法而非伪相关的关键。
- **第 26 章（智能体）**：工具选择与多步轨迹的信用分配可用激活修补与路径修补定位；轨迹级解释必须区分“某步读取了信息”与“整条轨迹因此成功”。
- **第 25 章（多模态）**：视觉 Token 与文本 Token 的跨模态回路可用同样的探针与修补工具研究；图像 Token 数会放大干预成本。
- **第 19 章（蒸馏）**：学生是否继承教师的回路是一个可检验问题；仅比较输出分布不足以说明机制被迁移。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、方案对比、代码、工业实践、论文导读、常见误区、「本章的取舍」、习题齐全。
- [x] 字数：正文 7195 个中文字符（剔除代码块与公式），在前沿章 7,000–9,000 预算内。
- [x] 公式编号 (28.1)–(28.10) 按正文出现顺序连续；探针、选择性、注意力分解、消融、对数几率差、恢复率、SAE、L0、纯度与习题实算均有推导或说明。
- [x] 至少 1 张对比表（表 28-1 七条路线）；工业案例含 Transformer Circuits、归纳头、IOI 回路、单层 SAE、Claude 3 Sonnet 的 Scaling Monosemanticity、SAE 缩放规律、拒绝方向与 ROME；论文导读 5 篇；误区 5 条。
- [x] 代码可运行：从章内 Markdown 重新提取 `tinylm/interpret/core.py` 与 `examples/ch28_interpret.py`，在 `/tmp/ch28verify` 运行；15 项测试通过，示例输出与正文逐字一致。
- [x] 公式与代码对应：`linear_probe_accuracy` (28.1)、`probe_selectivity` (28.2)、`attention_maps` (28.3)、`ablate_heads` (28.4)、`logit_margin` (28.5)、`recovery` (28.6)、`SparseAutoencoder`/`train_sae` (28.7)、`sae_metrics` (28.8)、`feature_purity` (28.9)、习题 (28.10)。
- [x] 新符号已登记 `notation.md` 与附录 B：$$h_i,W,C,\mathrm{Sel},\alpha_{tj},v_j,q_t,k_j,d_h,\Delta_h,\mathrm{LD},S_{\mathrm{clean/corrupt/patch}},R,W_e,W_d,b_e,b_d,K,\lambda,f,\mathrm{L0},P_k$$ 等，并说明与第 3、6、7、15、16、17、18、21、23、25、27 章的重名冲突。
- [x] 新术语已登记 `glossary.md` 与附录 C：线性探针、探针选择性、注意力可视化、头消融、组消融、路径修补、回路发现、因果擦除、叠加、特征纯度、死特征、特征吸收、特征分裂、拒绝方向、归纳头、IOI 回路、特征钳制、分布外修补；附录 C 已有的探针、机制可解释性、回路、稀疏自编码器、特征、激活修补未重复添加。
- [x] GitBook 语法：hint 使用 info/warning/success；1 个 Mermaid、1 张表、1 个 `<details>`；代码块均标注语言；无落单 `$`，无外部图片。
- [x] 事实性数字有来源或可复算：探针、注意力、消融、修补、SAE 与特征消融数字由固定种子程序输出；Elhage、Olsson、Wang、Bricken、Templeton、Gao、Arditi、Meng、Hase 的配置与结论来自公开论文/技术报告并在正文标注条件；机制可解释性缺少统一基准的时效性用 warning 提示。
- [x] 实验诚实报告：明确标注 2 层 64 维、200 步、1000 条留出样本的玩具规模；保留特征消融不显著的负结果；错误位点修补产生的假阴性被写成教学点；不得把复制任务的结论外推为真实 LLM 的回路结构。
- [x] 接口未破坏：`TinyLM.forward`、`CausalLMOutput` 与 `ModelConfig` 未改；解释逻辑全部在 `tinylm/interpret/`，以 hook 与只读分析实现，不改变模型输出。

## 第 29 章交接

### 本章摘要

本章作为全书收束章，把前沿问题放回“能力 = 架构 × 数据 × 算力 × 训练目标”的四因子框架，说明前沿不是第五个因子，而是四项同时逼近边界时的取舍。智能体强化学习部分把训练目标从下一个 Token 拉长到整条轨迹，推导长度 $$K$$、单步成功率 $$p$$ 下的轨迹成功率 $$p^K$$ 与看到一次成功所需样本数 $$n\approx\log(1/\delta)/p^K$$，指出稀疏奖励的采样成本随轨迹长度指数增长；比较终端奖励、过程奖励与势函数塑形 (29.4)，说明只有势函数差分形式能保证最优策略不变。持续学习部分区分稳定性与可塑性，推导 EWC 的 Fisher 二次惩罚 (29.5)，比较回放 (29.6) 与参数隔离，给出 ACC 与 BWT 两个评价指标 (29.7)。合成数据部分用一维高斯的递归最大似然推出方差收缩 $$((N-1)/N)^t$$ (29.8)，说明模型崩溃是长尾被逐代削掉；加入真实数据锚定后不动点回到真实方差 (29.9)，并讨论验证器过滤的分布倾斜 (29.10)。高效架构部分用算力账本比较全注意力、线性注意力/SSM 与混合架构 (29.11)，并说明架构会改变缩放定律常数与指数 (29.12)。理论部分用阈值型指标 (29.13) 说明部分“涌现”可能是度量非线性造成的幻觉，用能力定义 (29.14) 说明损失与能力分离，最后讨论可解释性边界、基准饱和与古德哈特定律。

### 关键公式

| 编号 | 公式 | 含义 |
| --- | --- | --- |
| (29.1) | $$G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k$$ | 折扣回报 |
| (29.2) | $$P(\mathrm{success})=\prod_{t=1}^{K}p_t=p^{K}$$ | 稀疏奖励下的轨迹成功率 |
| (29.3) | $$n\ge\frac{\log\delta}{\log(1-p^{K})}\approx\frac{\log(1/\delta)}{p^{K}}$$ | 看到一次成功所需样本数 |
| (29.4) | $$r'_t=r_t+\gamma\Phi(s_{t+1})-\Phi(s_t)$$ | 势函数塑形，最优策略不变 |
| (29.5) | $$\mathcal{L}_{\mathrm{EWC}}=\mathcal{L}_{B}+\frac{\lambda}{2}\sum_iF_i(\theta_i-\theta_{A,i}^{*})^2$$ | EWC 二次惩罚 |
| (29.6) | $$\mathcal{L}_{\mathrm{replay}}=(1-\rho)\mathcal{L}_{B}+\rho\mathcal{L}_{A}$$ | 回放损失 |
| (29.7) | $$\mathrm{ACC}=\frac{1}{T}\sum_i R_{T,i},\quad \mathrm{BWT}=\frac{1}{T-1}\sum_i(R_{T,i}-R_{i,i})$$ | 持续学习评价指标 |
| (29.8) | $$\sigma_{t+1}^{2}=\frac{N-1}{N}\sigma_t^{2}$$ | 纯合成自举的方差收缩 |
| (29.9) | $$\sigma_{t+1}^{2}=(1-\alpha)\sigma_t^{2}+\alpha\sigma_{\mathrm{real}}^{2}$$ | 真实数据锚定的不动点 |
| (29.10) | $$p_{\mathrm{filter}}(x)=\frac{p(x)q(x)}{Z},\quad Z=\mathbb{E}_p[q]$$ | 验证器过滤后的分布 |
| (29.11) | $$F_{\mathrm{hybrid}}(T)=aF_{\mathrm{attn}}(T)+(1-a)F_{\mathrm{state}}(T)$$ | 混合架构算力 |
| (29.12) | $$L(N,D)=L_{\infty}+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$$ | 缩放定律经验形式 |
| (29.13) | $$M(L)=\sigma\left(\frac{L^{\star}-L}{\tau}\right)$$ | 阈值型指标制造“涌现” |
| (29.14) | $$C(\theta)=\mathbb{E}_{x}\left[\mathbf{1}[m(x,\theta)\ge\tau]\right]$$ | 任务能力定义 |

### 实验与关键数字

- 设计：本章不新增 TinyLM 模块，接口保持冻结；`examples/ch29_frontier.py` 为独立数值实验，只依赖 PyTorch，CPU 约 8 秒，全部由固定种子控制。
- 持续学习：2→64→64→1 的 Tanh MLP，先学 $$y=\sin(2\pi x)$$ 再学 $$y=\cos(2\pi x)$$，任务标识作为第二维输入，每任务 200 点、各 500 步 Adam lr 1e-2，3 个种子。任务 A 的 MSE：顺序训练 0.0018→0.8741（遗忘 0.8722±0.1535），EWC $$\lambda=10^4$$ 后 0.1010（遗忘 0.0991±0.0869），回放 $$\rho=0.5$$ 后 0.0004（遗忘 -0.0014±0.0025）；任务 B 的 MSE 分别为 0.0001/0.0004/0.0004。
- 合成数据自举：一维标准高斯，$$N=100$$，200 代，200 次模拟；报告的是每次模拟方差的均值再开方。纯合成的标准差为 1.0000/0.8068/0.6416/0.3682（第 0/50/100/200 代），理论值 $$(0.99)^{t/2}$$ 为 1.0000/0.7778/0.6050/0.3660；加入 10% 真实数据后稳定在 1.0000/0.9640/0.9843/0.9594，不再收缩到零。
- 稀疏奖励：8 步独立伯努利链，4000 条轨迹/次，200 次更新，lr 0.5，3 个种子；报告解析成功率 $$\prod_t p_t$$ 与 2 万次采样的经验成功率。终端奖励 0.0129±0.0000 / 0.0129±0.0005（几乎没学动）；逐步验证器 0.9171 / 0.9181±0.0017；势函数塑形 0.9171 / 0.9181±0.0017。逐步奖励与势函数塑形在本任务中重合，因为“每步正确得 1 分”正好等于以累计正确步数为势函数的差分。
- 复现：从章内 Markdown 提取 Python 块到 `/tmp/ch29final/ch29_frontier.py` 运行，退出码 0，输出与正文 `text` 块逐字一致；提取脚本另核对正文 18 行输出块与实跑输出 `match=True`。

### 全书收束与待验证清单

- 本章是全书最后一章，没有“下一章”接口任务；`tinylm-state.md` 只登记独立性说明与不变量，不新增模块。
- **四因子复查**：第 1–7 章给架构与训练目标，第 8–13 章给算力与长上下文账本，第 9–10、14–19 章给数据与后训练目标，第 20–24 章给推理侧的算力交换，第 25–28 章把能力扩展到多模态、智能体、评测与可解释性，第 29 章把四项放回同一张取舍表。
- **待验证假设（发布前必须补证据）**：智能体 RL 的验证器偏差与跨环境泛化；持续学习的长期在线更新与可验证遗忘度量；合成数据的自举上限与可证明防坍塌；高效架构的长程检索与统一评价协议；涌现真实性、损失—能力映射与可控性边界。
- **时效性**：DeepSeek-R1、Phi-3、Constitutional AI、Mamba-2、Jamba、Griffin、DeepSeek-V3 的规模、上下文与分数只代表论文发布时的配置，正文已用 warning 提示截至写作时。
- **可复用工具**：第 27 章的 Wilson 区间、配对 bootstrap、McNemar、污染检测，与第 28 章的探针、修补、SAE 审计，是判断任何“前沿进展”是否成立的最低工具箱。

### 自检清单

- [x] 模板结构完整：导读、难度、目标、直觉、数学、方案对比、代码、工业实践、论文导读、常见误区、「本章的取舍」、习题齐全；五个前沿方向各有一节，收束到四因子框架。
- [x] 字数：正文 7712 个中文字符（剔除代码块与公式），在前沿章 7,000–9,000 预算内；未为凑字数注水。
- [x] 公式编号 (29.1)–(29.14) 按正文出现顺序连续；无落单编号；无编号的 `aligned` 推导按规范独立成行，其余随文行内。
- [x] 至少 1 张对比表：表 29-1 架构账本（五种架构的算力、状态、检索与风险），表 29-2 五个方向的综合对照；工业案例覆盖 DeepSeek-R1、过程监督、Phi-3/Constitutional AI、Mamba-2/Jamba/Griffin/DeepSeek-V3 与评测工程；论文导读 5 篇；误区 5 条。
- [x] 代码可运行：从最终 Markdown 提取 `examples/ch29_frontier.py`，在 `/tmp/ch29final` 运行，退出码 0，CPU 约 8 秒；三组实验的输出与正文逐字一致。
- [x] 公式与代码对应：终端奖励注释 (29.1)–(29.2)、EWC 注释 (29.5)、回放注释 (29.6)、方差递推注释 (29.8)–(29.9)、势函数塑形注释 (29.4)。
- [x] 新符号已登记 `notation.md` 与附录 B：$$G_t,p,K,\delta,n,\Phi,r',F_i,\theta^{*},\lambda,\rho,R_{t,i},\mathrm{ACC},\mathrm{BWT},\sigma_t^2,\alpha,q(x),Z,F_{\mathrm{attn}},F_{\mathrm{state}},F_{\mathrm{hybrid}},a,L_{\infty},A,B,M(L),\tau,C(\theta)$$ 等，并说明与第 9、13、16–19、26–28 章的重名冲突。
- [x] 新术语已登记 `glossary.md` 与附录 C：智能体强化学习、长程信用分配、稀疏奖励、过程奖励模型、势函数塑形、奖励作弊、灾难性遗忘、弹性权重固化、回放、参数隔离、后向迁移、模型崩溃、真实数据锚定、验证器偏差、混合架构、选择性状态空间、古德哈特定律、损失—能力分离、基准饱和等；已有术语不重复添加。
- [x] GitBook 语法：hint 使用 warning/success；1 个 Mermaid、2 张表、1 个 `<details>`；代码块均标注语言；无落单 `$`，无外部图片；每个文件一个 H1。
- [x] 事实性数字有来源或可复算：三组实验数字由固定种子程序输出；R1 的 AIME 15.6%→71.0%、Jamba 52B/12B/256K、DeepSeek-V3 671B/37B、EWC/Shumailov/Schaeffer 的结论来自公开论文并在正文标注条件；时效性内容已标注“截至写作时”。
- [x] 实验诚实报告：明确标注玩具规模（2 层 64 维 MLP、一维高斯、8 步独立链）与不能外推的边界；保留 EWC 高方差、逐步奖励与势函数塑形重合等细节；负结果与巧合都写进正文。
- [x] 接口未破坏：未改动 `TinyLM.forward`、`CausalLMOutput` 与已有 `ModelConfig` 字段；本章代码为独立脚本，不进入 `tinylm/` 包。

## 附录 A 交接

### 本附录摘要

《数学基础速览》按“定义—常用形式—成立条件—正文落点”组织，覆盖线性代数与分解、矩阵微积分与反向传播、概率与统计、信息论、优化、数值稳定与浮点、常用近似与不等式七个主题，最后给出与正文的对应关系表。全书正文的公式引用均有落点：交叉熵分解对应第 1 章式 (1.4)，Adam 与解耦权重衰减对应第 11 章式 (11.6)–(11.7)，方差递推对应第 29 章式 (29.8)。

### 结构与自检

- 一级标题 1 个，分节 A.1–A.8；导语 + `info` 提示开头，与附录 B、C 的组织方式一致。
- 共 9 张速查表；关键推导（交叉熵分解）用 `aligned` 独立成行，其余公式随表或随文行内。
- 正文 4470 个中文字符（剔除公式）；无落单 `$`；hint 使用 info/warning/danger/success。
- 新增算子已登记 `notation.md` 与附录 B.22：$$\operatorname{tr},\det,\operatorname{rank},\sigma_i,\lambda_i,\kappa,\mathbf{H},\operatorname{Cov},\operatorname{Var},\operatorname{Bias},\bar b,\nu_i$$，并说明与第 11、15、23、27 章的重名冲突。
- 事实性内容：SVD/Eckart–Young、Jacobian/VJP、MLE 的 $$N$$ 与 $$N-1$$、交叉熵分解、KL 非负性、GD 收敛率、Adam/AdamW、log-sum-exp、FP16/BF16/FP8 的范围均按标准结果给出；近似公式标注了成立条件。
- 本附录不涉及 TinyLM 代码与接口，`TinyLM.forward`、`CausalLMOutput`、`ModelConfig` 未改动。

## 附录 D 交接

### 本附录摘要

《必读论文清单》按部分与章节组织，汇总正文“论文导读”的 145 篇论文，每章 5 篇，共 D.1–D.7 七节。D.1–D.6 对应全书六个部分，表列为“章、作者（年份）、论文、出处、一句话贡献”；D.7 给出五条按目标的阅读路线。作者统一缩写为“第一作者 等”，出处优先会议或期刊，其次 arXiv；没有 arXiv 的技术报告注明载体。

### 结构与自检

- 一级标题 1 个；D.1 20 篇、D.2 20 篇、D.3 25 篇、D.4 30 篇、D.5 25 篇、D.6 25 篇，合计 145 篇；每章恰好 5 篇，29 章全部覆盖。
- 6 张论文表与 1 张阅读路线表：前六节各 1 张 5 列表，D.7 为 1 张 3 列表；行内无未转义 `|`，无 `$`，无 HTML 标签。
- 重复论文按章节分别列出，贡献按该章的阅读角度改写，不简单复制：Vaswani（第 1/3/4 章）、Kaplan（第 3/9 章）、RoFormer（第 4/6 章）、PI 与 YaRN（第 4/13 章）、Mamba（第 8/29 章）、Schaeffer（第 9/29 章）、InstructGPT（第 14/16 章）、R1（第 18/19/29 章）、HumanEval（第 20/27 章）。
- 标题与出处逐条回查 29 章原文：145 条的标题词元全部命中原文；arXiv 编号与原文一致，唯一例外是第 1 章 Vaswani 原文未给编号，按第 3/4 章同一论文补为 arXiv:1706.03762。
- GitBook 语法：仅使用 `info` 与 `warning` 两种 hint；无外部图片；表格单元格内无换行。
- 事实性：作者、年份、标题、会议与 arXiv 编号均取自正文，不新增未在正文出现的论文；一句话贡献不引入新的数字或结论。
- 本附录不涉及 TinyLM 代码与接口，`TinyLM.forward`、`CausalLMOutput`、`ModelConfig` 未改动。

## 附录 E 交接

### 本附录摘要

《主流开源模型架构参数对照表》把第 3–7 章的架构分析落到具体配置，共 E.1–E.6 六节。E.1 列 8 个稠密模型（Llama 2 7B/13B/70B、Llama 3 8B/70B、Llama 3.1 405B、Mistral 7B、PaLM 540B†），字段为层数、$$d_{model}$$、Q/KV 头、$$d_{ff}$$、词表、上下文、参数量；E.2 列 5 个 MoE 模型（Mixtral 8x7B、DeepSeek-V2/V3、Qwen3-235B-A22B、Kimi K2），增加总参数、激活参数、激活比、专家配置与专家宽度；E.3 按归一化、门控前馈、位置编码、注意力、偏置、权重绑定、块结构、长上下文扩展八个维度横纵对照九类模型；E.4 给出参数、权重显存与 KV Cache 的闭式估算，并用六个代表模型实算；E.5 列出五条读表注意事项；E.6 汇总 11 份技术报告与 arXiv 编号。

### 结构与自检

- 一级标题 1 个；二级标题 6 个（E.1–E.6）；表格 5 张（稠密 8 行、MoE 5 行、结构对照 9 行、预算账本 6 行、技术报告 11 行），列数分别为 8/10/9/4/3，逐行管道符计数一致。
- GitBook 语法：2 个 hint（`info`、`warning`），配平；行内公式 44 处，`$$` 成对，无落单 `$`；无代码块、无外部图片；表格单元格内无换行与未转义 `|`。
- 事实性来源：稠密与 MoE 的逐层几何参数取自第 5–7 章已核对表格与技术报告；PaLM 非开源，用 `†` 标注且仅作架构参照；Qwen3-235B-A22B 与 Kimi K2 的层数、$$d_{model}$$、专家宽度在正文与公开材料中无同口径数据，一律填“—”，不推测。
- 数值复算：Llama 2 7B 参数逐位加总为 6,738,415,616，四项拆分为注意力 2.147B、前馈 4.329B、归一化 0.266M、嵌入与输出 0.262B，前馈占非嵌入参数 66.8%；Llama 3 8B 相对增量 1.292B 可由前馈 +1.309B、词表 +0.789B、GQA −0.805B 对上。
- 推理账本：8K、FP16、批量 1 下，Llama 2 7B 4 GiB、Llama 3 8B 1 GiB、Mistral 1 GiB、Mixtral 1 GiB、DeepSeek-V2 540 MiB、DeepSeek-V3 约 549 MiB；权重显存 12.6 GiB/15.0 GiB/13.5 GiB/87 GiB/440 GiB/1.22 TiB，均按总参数而非激活参数计。
- 口径说明：DeepSeek-V3 激活比 5.5% 以 671B 总参数为分母，若以未扣前 3 层稠密的 702.9B 为分母则为 5.2%，文中两口径并列。
- 符号与术语：本附录未引入新符号或新术语，`notation.md`、`glossary.md`、附录 B、附录 C 无需改动。
- TinyLM：无代码与接口变更，`TinyLM.forward`、`CausalLMOutput`、`ModelConfig` 未改动。
- 中文正文字符数（剔除代码块与 `$$` 公式）约 2,022，符合附录体量。

## 全书统筹记录（2026-11-12）

### 契约与账本

- 附录 F.1 的 TinyLM 目录树与会话交接、各章 `tinylm/*.py` 的实际路径对齐：模型模块改为平铺的 `config.py`、`norm.py`、`position.py`、`attention.py`、`ffn.py`、`moe.py`、`block.py`、`ssm.py`、`model.py`、`scaling.py`，补齐 `data/filter.py`、`data/mixture.py`、`data/collate.py`、`train/parallel.py`、`train/long_context.py`、`inference/kv_cache.py`、`inference/flash.py`、`inference/paged.py`；F.4 演进表同步改到这些文件。
- 附录 F.3.1 的参数量公式由 `d² + 2d·d_kv + 2d·d_h + 3d·d_ff` 修正为 `2d² + 2d·d_kv + 3d·d_ff`，与第 6 章 `attention_params` 及第 5 章式 (5.11) 一致；F.3.2 表格与 F.3.3 的 FLOPs 估算按绑定词表重算为 tiny_cpu≈2M、s≈6.5M、m≈16M、l≈32M、xl≈56M、tiny_24g≈100M（不绑定时约 125M），并注明绑定假设。
- 附录 F.2.2 补入第 8 章已实现的架构开关 `block_type`、`ssm_d_state`、`ssm_expand`、`hybrid_attn_every`，默认值保持既有行为，冻结接口未改。

### 术语与符号

- 统一“曝光偏差”（原 `progress.md` 与 `glossary.md` 的“暴露偏差”）。
- 统一“多样性坍缩”（第 14 章正文与术语表的“多样性塌缩”）。
- 统一 `PagedAttention` 拼写（原术语表“Paged Attention”）。
- 术语表去重：合并第 10/14 章重复的“多样性坍缩”、第 20/27 章重复的 `pass@k`、第 21/23 章重复的 KV Cache 与 KV Cache 量化、第 21 章内部的预填充/分页注意力/前缀缓存重复项。
- 区分专家路由 `Top-K` 与采样 `Top-k`，全书保持该大小写约定。

### 排版与引用

- `progress.md` 公式索引表中 17 行的未转义 `|` 改为 `\lvert/\rvert`、`\lVert/\rVert`、`\parallel`，表格列数恢复一致；全书表格逐行管道符计数 0 异常。
- 章节难度标签统一为 `⭐ 等级 ｜ 前置/数学要求 ｜ 代码要求` 形式；第 3 章 `⭐⭐⭐ 核心` 改为 `⭐⭐⭐ 进阶`，其余章的 `；` 与句号分隔统一为 ` ｜ `。
- `preface.md` 的 12 处 ASCII 双引号改为中文弯引号。
- `README.md` 的阅读时间口径改为“每章约 1–1.5 小时阅读，再加动手时间”，TinyLM 覆盖面补上“线性注意力与 SSM 块”。

### 复核结果

- 29 章均含 12 部分与“本章的取舍”；本章导读均 ≤200 字；每章 1 个 `<details>`；论文导读每章 5 篇；习题三类型齐全。
- 公式编号逐章从 1 连续、不重号；`第 X 章式 (X.Y)`、`第 X.Y 节`、附录 A–F 小节引用全部可解析。
- GitBook 语法：hint 类型合法且配平，代码围栏成对，tabs 与 details 配平，无落单 `$`；`SUMMARY.md` 37 条链接全部指向存在的文件，无未登记章节。
- 编辑项已按作者确认处理：`CLAUDE.md` 把第 11、17 章并入核心章（10,000–12,000），误区规范放宽为 3–6 条；第 11–29 章论文导读统一为“加粗编号段落”，第 15–29 章习题统一为 `### 习题 N（推导/编程/开放思考）`，原有题干逐条回填并复核无丢失。
- 全书统一后复算：15 个改造章的中文字符数各增加 5 个，全部来自习题小标题（`习题 N（类型）` 替代行内粗体标签），正文题干无增删；论文条目仍为每章 5 篇，编号 1–5 连续。
- 第 24–29 章“本章小结”改为与第 1–23 章一致的结构：先列 5–6 条要点，再用 `{% hint style="success" %}` 承载 `**本章的取舍**` 段落；原 `### 本章的取舍` 小标题与独立提示块合并进该 hint，原有收束句保留为 hint 的第二段。29 章小结结构现全部一致。
