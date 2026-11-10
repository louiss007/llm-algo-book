# 符号登记簿

权威版本为[附录 B](appendix/b-notation.md)。本文件只登记每章新增或修改的符号，便于分会话交接。

## 第 1 章

| 符号 | 含义 | 首次出现 |
| --- | --- | --- |
| $$T$$ | 序列长度 | 第 1 章 |
| $$n_{vocab}$$ | 词表大小 | 第 1 章 |
| $$n$$ | n-gram 的阶数 | 第 1 章 |
| $$d$$ | 通用表示维度（非 Transformer 语境） | 第 1 章 |
| $$D$$ | 数据集 Token 总数 | 第 1 章 |
| $$x_t$$ | 第 $$t$$ 个位置的 Token | 第 1 章 |
| $$z_t$$ | 第 $$t$$ 个 Token 的整数 ID | 第 1 章 |
| $$\mathbf{e}_t$$ | 第 $$t$$ 个 Token 的嵌入向量 | 第 1 章 |
| $$\mathbf{E}$$ | Token 嵌入矩阵 | 第 1 章 |
| $$\mathbf{h}_t$$ | 循环网络的隐状态 | 第 1 章 |
| $$p_\theta(x_t \mid x_{<t})$$ | 模型给出的下一 Token 概率 | 第 1 章 |
| $$\mathcal{L}$$ | 训练损失（按 Token 平均的负对数似然） | 第 1 章 |
| $$H(p)$$ | 信息熵 | 第 1 章 |
| $$H(p,q)$$ | 交叉熵 | 第 1 章 |
| $$\mathrm{PPL}$$ | 困惑度 | 第 1 章 |
| $$k$$ | n-gram 加性平滑常数 | 第 1 章 |
| $$m$$ | 多 Token 预测的未来步数 | 第 1 章 |
| $$\boldsymbol{\ell}_t$$ | 位置 $$t$$ 的输出 logits | 第 1 章 |
| $$\mathbf{W}_x, \mathbf{W}_h, \mathbf{W}_o$$ | RNN 的输入、循环、输出权重 | 第 1 章 |

{% hint style="warning" %}
$$k$$ 在第 1 章指加性平滑常数，在第 7、20 章指"前 k 个"；$$d$$ 在第 1 章指通用维度，第 3 章起统一使用 $$d_{model}$$。读后续章节时按上下文区分。
{% endhint %}

## 第 2 章

$$n_{vocab}$$ 已在第 1 章登记；本章新增关系 $$n_{vocab}=|\mathcal{V}|$$，即词表大小等于词表集合的基数。

| 符号 | 含义 | 首次出现 |
| --- | --- | --- |
| $$\mathcal{V}$$ | 词表集合（Token 的集合） | 第 2 章 |
| $$\mathcal{A}$$ | 基础符号集（字节级 BPE 为 256 个字节） | 第 2 章 |
| $$M$$ | BPE 的合并次数 | 第 2 章 |
| $$S$$ | 特殊 Token 数 | 第 2 章 |
| $$q^{(t)}=(q_1,\dots,q_{T_t})$$ | 第 $$t$$ 轮合并后的符号序列 | 第 2 章 |
| $$T_0,\ T_t$$ | 初始符号序列长度、第 $$t$$ 轮符号序列长度 | 第 2 章 |
| $$f_t(a,b)$$ | 第 $$t$$ 轮相邻符号对 $$(a,b)$$ 的频次 | 第 2 章 |
| $$N_{\text{pair}}$$ | 语料中相邻对位置的总数 | 第 2 章 |
| $$n_{\text{byte}}(c)$$ | 字符 $$c$$ 的 UTF-8 字节数 | 第 2 章 |
| $$r$$ | 压缩比，字节/Token | 第 2 章 |
| $$B_{\text{byte}}$$ | 文本的 UTF-8 总字节数 | 第 2 章 |
| $$n_{\text{char}},\ n_{\text{word}}$$ | 字符数、词数 | 第 2 章 |
| $$u=(u_1,\dots,u_K)$$ | Unigram 模型中的 Token 序列 | 第 2 章 |
| $$p(u)$$ | Unigram 模型中 Token $$u$$ 的概率 | 第 2 章 |
| $$\mathrm{best}[i]$$ | 前缀 $$s_{0:i}$$ 的 Viterbi 最优对数概率 | 第 2 章 |
| $$L_{\max}$$ | 词表中最长 Token 的长度 | 第 2 章 |
| $$\rho$$ | 每字节比特数 | 第 2 章 |
| $$\mathrm{fertility}$$ | 生育率，平均每个词的 Token 数 | 第 2 章 |
| $$\mathrm{Enc},\ \mathrm{Dec}$$ | 分词器的编码器与解码器 | 第 2 章 |
| $$N_0$$ | 除嵌入层和输出层外的参数量 | 第 2 章 |
| $$N(n_{vocab})$$ | 模型参数量随词表大小的变化 | 第 2 章 |

{% hint style="warning" %}
第 2 章涉及多个易混符号：词表大小统一用 $$n_{vocab}$$，词表集合用 $$\mathcal{V}$$，注意力的值矩阵是粗体 $$\mathbf{V}$$；Token ID 仍是 $$z_t$$，BPE 的符号序列用 $$q$$，Unigram 的 Token 用 $$u$$；$$B$$ 是批大小，文本字节数用 $$B_{\text{byte}}$$；$$C$$ 是训练算力，字符数用 $$n_{\text{char}}$$；$$W$$ 是滑动窗口大小，词数用 $$n_{\text{word}}$$；$$M_{ij}$$ 是因果掩码，$$M$$ 是 BPE 合并次数；$$s_{ij}$$ 是注意力打分，$$s_{j:i}$$ 是子串。
{% endhint %}

## 第 3 章

| 符号 | 含义 | 首次出现 |
| --- | --- | --- |
| $$d_{model}$$ | 隐藏层维度、残差流宽度 | 第 3 章 |
| $$n_l$$ | Transformer 层数 | 第 3 章 |
| $$n_h$$ | 注意力头数 | 第 3 章 |
| $$d_h$$ | 单头维度，$$d_h=d_{model}/n_h$$ | 第 3 章 |
| $$d_{ff}$$ | 前馈网络中间维度 | 第 3 章 |
| $$d_k$$ | 打分缩放维度，通常等于 $$d_h$$ | 第 3 章 |
| $$L_{ctx}$$ | 上下文窗口长度、位置嵌入表大小 | 第 3 章 |
| $$B$$ | 批大小 | 第 3 章 |
| $$\mathbf{X}^{(l)},\ \mathbf{H}^{(l)}$$ | 第 $$l$$ 层输入、残差流隐藏状态 | 第 3 章 |
| $$\mathbf{W}^{Q},\mathbf{W}^{K},\mathbf{W}^{V}$$ | 查询、键、值投影 | 第 3 章 |
| $$\mathbf{Q},\mathbf{K},\mathbf{V}$$ | 查询、键、值矩阵 | 第 3 章 |
| $$\mathbf{S}$$ | 注意力打分矩阵 | 第 3 章 |
| $$\mathbf{A}$$ | 注意力权重矩阵 | 第 3 章 |
| $$M_{ij}$$ | 因果掩码，允许处为 0，禁止处为 $$-\infty$$ | 第 3 章 |
| $$\mathbf{O},\ \mathbf{O}^{(h)}$$ | 注意力输出、第 $$h$$ 个头的输出 | 第 3 章 |
| $$\mathbf{W}^{O}$$ | 注意力输出投影 | 第 3 章 |
| $$\mathrm{LN}(\cdot)$$ | 层归一化 | 第 3 章 |
| $$\boldsymbol{\gamma},\ \boldsymbol{\beta},\ \epsilon$$ | 归一化增益、偏置、数值稳定项 | 第 3 章 |
| $$\phi(\cdot)$$ | 逐元素激活函数 | 第 3 章 |
| $$\mathbf{W}_{\text{in}},\mathbf{W}_{\text{out}}$$ | 前馈网络输入、输出投影 | 第 3 章 |
| $$\mathbf{W}_{\text{lm}}$$ | 输出层词表投影 | 第 3 章 |
| $$\mathbf{P},\ \mathbf{P}_t$$ | 位置嵌入表及其第 $$t$$ 行 | 第 3 章 |
| $$\pi$$ | 排列（permutation） | 第 3 章 |
| $$\mathrm{FLOPs}$$ | 浮点运算次数 | 第 3 章 |
| $$\nabla_\theta$$ | 对参数 $$\theta$$ 求梯度 | 第 3 章 |

{% hint style="warning" %}
第 3 章的易混点：$$(h)$$ 是注意力头的上标，不是幂次；$$\mathbf{h}_t$$ 无上标时是第 1 章的 RNN 隐状态，带上标 $$\mathbf{h}_t^{(l)}$$ 时是第 $$l$$ 层残差流的位置表示。$$\mathbf{S}$$ 是打分矩阵，$$s_{ij}$$ 是它的元素；$$\mathbf{O}$$ 是注意力输出，$$\mathbf{o}_i$$ 是它的第 $$i$$ 行。$$N$$ 是参数量，$$N_{\text{pair}}$$ 是第 2 章的相邻对位置数。$$L_{ctx}$$ 是位置表大小，$$L_{\max}$$ 是第 2 章的最长 Token 长度。$$M$$ 是 BPE 合并次数，$$M_{ij}$$ 是因果掩码。$$d$$ 在第 1 章是通用维度，第 3 章起统一写 $$d_{model}$$。
{% endhint %}

## 第 4 章

| 符号 | 含义 | 首次出现 |
| --- | --- | --- |
| $$p$$ | 位置索引，与 $$t$$ 同义 | 第 4 章 |
| $$\mathbf{p}_t$$ | 位置 $$t$$ 的编码向量 | 第 4 章 |
| $$\omega_i$$ | 正弦编码第 $$i$$ 对维度的角频率，$$\omega_i=b^{-2i/d_{model}}$$ | 第 4 章 |
| $$\theta_i$$ | RoPE 第 $$i$$ 个旋转频率，$$\theta_i=b^{-2i/d_h}$$ | 第 4 章 |
| $$b$$ | 位置编码基频，通常取 10000 | 第 4 章 |
| $$\lambda_i$$ | 第 $$i$$ 个频率的波长，$$\lambda_i=2\pi/\omega_i$$ 或 $$2\pi/\theta_i$$ | 第 4 章 |
| $$\Delta$$ | 位置平移量 | 第 4 章 |
| $$\mathbf{R}_{t,i},\ \mathbf{R}_t$$ | 第 $$i$$ 个二维块的旋转矩阵、整体分块对角旋转矩阵 | 第 4 章 |
| $$b_{\mathrm{clip}(j-i)}$$ | T5 相对位置偏置，按截断距离索引 | 第 4 章 |
| $$\nu_h$$ | ALiBi 第 $$h$$ 个头的距离惩罚斜率 | 第 4 章 |
| $$L_{\mathrm{train}},\ L_{\mathrm{test}}$$ | 训练长度、测试长度 | 第 4 章 |
| $$s_{\mathrm{rope}}$$ | RoPE 缩放因子，$$s_{\mathrm{rope}}=L_{\mathrm{test}}/L_{\mathrm{train}}$$ | 第 4 章 |
| $$L_{\mathrm{eff}}(\epsilon)$$ | 阈值为 $$\epsilon$$ 时的有效上下文长度 | 第 4 章 |
| $$r_i$$ | YaRN 中训练长度与第 $$i$$ 个波长的比值，$$r_i=L_{\mathrm{train}}/\lambda_i$$ | 第 4 章 |
| $$w_i$$ | YaRN 第 $$i$$ 个频率的分段插值权重 | 第 4 章 |
| $$t_y$$ | YaRN 温度缩放因子 | 第 4 章 |

{% hint style="warning" %}
第 4 章的易混点：$$b$$ 单独出现时是位置编码基频，$$b_{\mathrm{clip}(j-i)}$$ 是 T5 的相对位置偏置；$$\omega_i$$ 是正弦编码的角频率，$$\theta_i$$ 是 RoPE 的旋转频率，二者形式相同但定义域不同；$$s_{\mathrm{rope}}$$ 是位置缩放因子，$$s_{ij}$$ 仍是第 3 章的注意力打分；$$r_i$$ 是 YaRN 的波长比，$$r$$ 在第 2 章是压缩比；$$t_y$$ 是温度因子，$$T$$ 是序列长度。
{% endhint %}

## 第 5 章

| 符号 | 含义 | 首次出现 |
| --- | --- | --- |
| $$\mathrm{RMS}(\mathbf{x})$$ | 均方根，$$\sqrt{\frac{1}{d}\sum_i x_i^2}$$ | 第 5 章 |
| $$\mathrm{RMSNorm}(\cdot)$$ | 均方根归一化，只缩放不减均值 | 第 5 章 |
| $$\mathrm{GLU}(\cdot)$$ | 门控线性单元 | 第 5 章 |
| $$\mathrm{SiLU}(\cdot)$$ | SiLU/Swish 激活，$$z\sigma(z)$$ | 第 5 章 |
| $$\sigma(\cdot)$$ | Sigmoid 函数 | 第 5 章 |
| $$\mathbf{W}_1,\ \mathbf{W}_2,\ \mathbf{W}_3$$ | SwiGLU 的门、值与输出投影 | 第 5 章 |
| $$N_{\mathrm{modern}}$$ | 现代配置的精确参数量 | 第 5 章 |
| $$n_{kv},\ d_{kv}$$ | 键/值头数、键值投影总宽度 | 第 5 章（第 6 章展开） |

{% hint style="warning" %}
第 5 章的易混点：$$\sigma(\cdot)$$ 是 Sigmoid，$$\sigma_t$$ 是第 3 章层归一化中的标准差，二者不是同一个符号；$$N_{\mathrm{modern}}$$ 是参数量公式，$$N$$ 在第 2 章是 BPE 词表大小；$$\mathbf{W}_1,\mathbf{W}_2,\mathbf{W}_3$$ 专指 SwiGLU 的三个投影，与第 3 章的 $$\mathbf{W}_{\text{in}},\mathbf{W}_{\text{out}}$$ 对应但不混用；$$d_{ff}$$ 是前馈中间维度，$$d_k$$ 是注意力打分维度；$$d_{kv}$$ 是键值投影的总宽度，不是单个 KV 头的维度。
{% endhint %}

## 第 6 章

| 符号 | 含义 | 首次出现 |
| --- | --- | --- |
| $$n_{kv}$$ | KV 头数，$$1\le n_{kv}\le n_h$$；MHA 取 $$n_h$$，MQA 取 1 | 第 6 章（第 5 章预留） |
| $$d_{kv}$$ | K/V 投影总宽度，$$d_{kv}=d_{model}n_{kv}/n_h$$ | 第 6 章（第 5 章预留） |
| $$d_h$$ | 单头维度，减少 KV 头时保持不变 | 第 3 章 |
| $$d_c$$ | MLA 潜向量维度，$$d_c\ll d_{model}$$ | 第 6 章 |
| $$d_r$$ | MLA 解耦 RoPE 的位置键维度 | 第 6 章 |
| $$\mathbf{c}_t$$ | 位置 $$t$$ 的 MLA 压缩潜向量 | 第 6 章 |
| $$\mathbf{W}_{DKV}$$ | MLA 下投影，$$d_{model}\to d_c$$ | 第 6 章 |
| $$\mathbf{W}_{UK}^{(i)},\ \mathbf{W}_{UV}^{(i)}$$ | 第 $$i$$ 个头的键、值上投影，$$d_c\to d_h$$ | 第 6 章 |
| $$\mathbf{W}_{QR}^{(i)},\ \mathbf{W}_{KR}$$ | 查询位置投影、共享位置键投影 | 第 6 章 |
| $$\mathbf{W}_{OV}^{(i)}$$ | 吸收后的输出权重，$$\mathbf{W}_O^{(i)}\mathbf{W}_{UV}^{(i)}$$ | 第 6 章 |
| $$\tilde{\mathbf{q}}_t^{(i)}$$ | 吸收后的查询，位于 $$d_c$$ 维潜空间 | 第 6 章 |
| $$b_{\mathrm{elem}}$$ | 每个缓存元素的字节数（FP16 为 2） | 第 6 章 |
| $$N_{attn}$$ | 单层注意力参数量，$$2d_{model}^2+2d_{model}d_{kv}$$ | 第 6 章 |
| $$N_{MLA}$$ | 单层 MLA 注意力参数量 | 第 6 章 |
| $$M_{kv}$$ | KV Cache 字节数 | 第 6 章 |
| $$M_{MLA}$$ | MLA Cache 字节数 | 第 6 章 |
| $$t_{\mathrm{dec}}$$ | 解码一步的近似延迟 | 第 6 章 |
| $$\mathrm{BW}$$ | 显存带宽，字节/秒 | 第 6 章 |
| $$F_{\mathrm{prefill}},\ F_{\mathrm{decode}}$$ | 预填充、解码一步的浮点运算次数 | 第 6 章 |

{% hint style="warning" %}
第 6 章的易混点：$$b$$ 是批大小，$$b_{\mathrm{elem}}$$ 是每元素字节数，$$b$$ 在第 4 章还曾是 RoPE 基频；$$d_{kv}$$ 是 K/V 投影总宽度，$$d_h$$ 是单个头的维度，减少 KV 头时 $$d_h$$ 不变、$$d_{kv}$$ 按 (6.4) 缩小；$$d_c$$ 是 MLA 潜维度，$$d_k$$ 是第 3 章的打分缩放维度；$$\mathbf{W}_O$$ 是输出投影，$$\mathbf{W}_{OV}$$ 是吸收后的合并权重；$$M_{kv}$$ 是缓存字节数，$$M_{ij}$$ 是因果掩码，$$M$$ 是第 2 章的 BPE 合并次数；$$N_{attn}$$、$$N_{MLA}$$ 是单层注意力参数量，$$N$$ 是模型总参数量；$$t_{\mathrm{dec}}$$ 是解码延迟，$$t_y$$ 是第 4 章 YaRN 的温度因子。
{% endhint %}
