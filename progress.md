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
- **第 20 章**：教师强制与暴露偏差会在解码策略中展开。
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
| (2.6) | $$|\mathcal{V}|=|\mathcal{A}|+M+S$$ | 词表规模关系 |
| (2.7) | $$T_M=T_0-\sum_{t=1}^{M}n_t,\ n_t\le f_t(a_t,b_t)$$ | BPE 贪心长度缩减 |
| (2.8) | $$\mathrm{score}(a,b)=f(ab)/(f(a)f(b))$$ | WordPiece 合并得分 |
| (2.9) | $$\mathrm{score}(a,b)=\frac{1}{N_{\text{pair}}}e^{\mathrm{PMI}(a,b)}$$ | 得分与 PMI 的关系 |
| (2.10) | $$P(u)=\prod_{i=1}^{K}p(u_i)$$ | Unigram 切分概率 |
| (2.11) | $$\mathrm{best}[i]=\max_{j<i}(\mathrm{best}[j]+\log p(s_{j:i}))$$ | Viterbi 编码 |
| (2.12) | $$1\le|\mathrm{Enc}(c)|\le n_{\text{byte}}(c)$$ | 字符的 Token 数界 |
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
