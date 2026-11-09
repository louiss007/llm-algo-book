# 写作进度与分会话交接

## 已完成章节

| 章节 | 状态 | 正文中文字数 | 代码验证 | 备注 |
| --- | --- | --- | --- | --- |
| 第 1 章 | 完成 | 约 7100 | 已在本机运行 | 含可控马尔可夫链实验 |
| 第 2 章 | 完成 | 约 8100 | 已在本机运行 | 字节级 BPE 与词表扫描实验 |
| 第 3 章 | 完成 | 约 10300 | 已在本机运行 | Transformer 基线、因果性测试与复制任务 |

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
