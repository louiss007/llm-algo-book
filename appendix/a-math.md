# 附录 A 数学基础速览

本附录把全书反复用到的数学工具压缩成一份可查的速查表。它不是线性代数、概率论或优化理论的教材：每个条目只给定义、常用形式、成立条件与在正文中的落点，推导只保留关键一步。遇到不熟悉的主题，先在这里建立索引，再回到对应章节看完整推导。

{% hint style="info" %}
记号以附录 B 为准：粗体大写是矩阵，粗体小写是向量，普通斜体是标量。本附录额外使用 $$\operatorname{tr}$$（迹）、$$\det$$（行列式）、$$\operatorname{rank}$$（秩）、$$\sigma_i$$（奇异值）、$$\lambda_i$$（特征值）与 $$\kappa$$（条件数）；这些符号在正文或本附录中均有定义。
{% endhint %}

## A.1 向量、矩阵与分解

| 概念 | 定义或公式 | 在正文中的用途 |
| --- | --- | --- |
| 内积 | $$\langle\mathbf{x},\mathbf{y}\rangle=\mathbf{x}^{\top}\mathbf{y}=\sum_i x_i y_i$$ | 相似度、投影、能量 |
| $$L_2$$ 范数 | $$\lVert\mathbf{x}\rVert_2=\sqrt{\mathbf{x}^{\top}\mathbf{x}}$$ | 梯度裁剪、权重衰减 |
| 余弦相似度 | $$\cos\theta=\frac{\mathbf{x}^{\top}\mathbf{y}}{\lVert\mathbf{x}\rVert_2\lVert\mathbf{y}\rVert_2}$$ | 嵌入检索、对比学习 |
| Frobenius 范数 | $$\lVert\mathbf{A}\rVert_F=\sqrt{\sum_{ij}A_{ij}^2}$$ | 低秩近似误差、量化误差 |
| 矩阵乘法 | $$(\mathbf{A}\mathbf{B})_{ij}=\sum_k A_{ik}B_{kj}$$，复杂度 $$O(mnk)$$ | 注意力、前馈、投影 |
| 转置与逆 | $$(\mathbf{A}^{\top})_{ij}=A_{ji}$$，$$\mathbf{A}\mathbf{A}^{-1}=\mathbf{I}$$ | 最小二乘、闭式解 |
| 秩 | $$\operatorname{rank}(\mathbf{A})$$ 等于线性无关的行（或列）数 | 低秩适配、表示瓶颈 |
| 迹 | $$\operatorname{tr}(\mathbf{A})=\sum_i A_{ii}=\sum_i\lambda_i$$ | 矩阵导数、谱范数 |
| 行列式 | $$\det(\mathbf{A})=\prod_i\lambda_i$$ | 高斯归一化常数 |
| 特征分解 | $$\mathbf{A}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^{-1}$$；实对称时 $$\mathbf{Q}$$ 正交 | 曲率分析、条件数 |
| 奇异值分解 | $$\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}$$ | 低秩近似、条件数 |
| 条件数 | $$\kappa(\mathbf{A})=\sigma_{\max}/\sigma_{\min}$$（正定时为 $$\lambda_{\max}/\lambda_{\min}$$） | 优化难度、数值稳定性 |
| 正定性 | 对所有 $$\mathbf{x}\neq\mathbf{0}$$ 有 $$\mathbf{x}^{\top}\mathbf{A}\mathbf{x}>0$$ | Hessian、协方差矩阵 |
| 正交投影 | $$\mathbf{P}=\mathbf{A}(\mathbf{A}^{\top}\mathbf{A})^{-1}\mathbf{A}^{\top}$$（$$\mathbf{A}$$ 列满秩） | 最小二乘、特征投影 |

奇异值分解是把任意矩阵写成“旋转—缩放—旋转”的标准工具。截断到前 $$r$$ 个奇异值，得到秩为 $$r$$ 的近似 $$\mathbf{A}_r=\sum_{i=1}^{r}\sigma_i\mathbf{u}_i\mathbf{v}_i^{\top}$$。Eckart–Young 定理给出它的最优性：在所有秩不超过 $$r$$ 的矩阵中，$$\mathbf{A}_r$$ 的 Frobenius 误差最小，且误差平方恰好等于被丢弃奇异值的平方和，即 $$\lVert\mathbf{A}-\mathbf{A}_r\rVert_F^2=\sum_{i=r+1}^{\min(m,n)}\sigma_i^2$$。第 15 章的 LoRA 就是在这个意义上假设权重更新是低秩的；第 6 章的 MLA 也利用低秩压缩 KV 表示。低秩假设是否成立，取决于奇异值谱衰减得多快，而不是参数量本身。

条件数衡量“输入的小扰动会被放大多少”。对线性系统 $$\mathbf{A}\mathbf{x}=\mathbf{b}$$，相对误差的上界约为 $$\kappa(\mathbf{A})$$ 乘以 $$\mathbf{b}$$ 的相对误差；在优化中，Hessian 的条件数决定梯度下降在最坏方向上的收敛速度。第 11 章的病态二次型实验把条件数设为 1000，正是为了让不同优化器的差异显性化。

深度学习代码里的矩阵很少是二维的。一个批次的激活通常写成 $$(B,T,d)$$ 的三维张量，注意力分数写成 $$(B,n_h,T,T)$$；批量矩阵乘把前两维当作批维度，只对最后两维做矩阵乘。广播规则允许形状不完全相同的张量逐元素运算：从右往左对齐维度，缺失的维度视为 1，大小为 1 的维度自动扩展。它省去了显式复制，但也容易把“本应逐样本”的操作误写成跨样本广播。写代码前先写出目标形状，是避免这类错误最便宜的办法。

{% hint style="warning" %}
特征分解只对方阵有定义，且一般矩阵的特征向量不一定正交；SVD 对任意矩阵都存在，奇异值总为非负实数。不要把“特征值”和“奇异值”混用：对称正定矩阵两者相同，非对称矩阵则不同。条件数用奇异值定义，不要只算特征值的比值。
{% endhint %}

## A.2 矩阵微积分与反向传播

| 函数 | 导数 | 备注 |
| --- | --- | --- |
| $$\mathbf{a}^{\top}\mathbf{x}$$ | $$\mathbf{a}$$ | 对 $$\mathbf{x}$$ 求梯度 |
| $$\mathbf{x}^{\top}\mathbf{A}\mathbf{x}$$ | $$(\mathbf{A}+\mathbf{A}^{\top})\mathbf{x}$$ | $$\mathbf{A}$$ 对称时为 $$2\mathbf{A}\mathbf{x}$$ |
| $$\lVert\mathbf{x}\rVert_2^2$$ | $$2\mathbf{x}$$ | 权重衰减的梯度 |
| $$\mathbf{A}\mathbf{x}$$ | $$\mathbf{A}$$ | Jacobian 为 $$\mathbf{A}$$ |
| $$\operatorname{tr}(\mathbf{A}\mathbf{B})$$ | $$\mathbf{B}^{\top}$$ | 对 $$\mathbf{A}$$ 求导 |
| $$\log\det(\mathbf{A})$$ | $$\mathbf{A}^{-\top}$$ | 高斯分布的对数密度 |
| $$\sigma(z)$$ | $$\sigma(z)\bigl(1-\sigma(z)\bigr)$$ | Sigmoid 的导数 |
| $$\operatorname{softmax}(\mathbf{z})$$ | $$\operatorname{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^{\top}$$ | $$\mathbf{p}=\operatorname{softmax}(\mathbf{z})$$ |
| 交叉熵（对 logits） | $$\mathbf{p}-\mathbf{y}$$ | $$\mathbf{y}$$ 为 one-hot 标签 |
| $$\max_i z_i$$ | $$\mathbf{e}_j$$（$$j=\arg\max_i z_i$$） | 次梯度，几乎处处可导 |

反向传播是链式法则在计算图上的动态规划。若 $$\mathbf{y}=f(\mathbf{x})$$、$$z=g(\mathbf{y})$$，则 $$\partial z/\partial\mathbf{x}=(\partial\mathbf{y}/\partial\mathbf{x})^{\top}\partial z/\partial\mathbf{y}$$。深度学习框架不显式构造 Jacobian，而是计算向量—Jacobian 乘积（VJP）：给定上游梯度，一次反向即可得到所有输入与参数的梯度。一次前向加一次反向的计算量约为前向的 2–3 倍，显存则要保存中间激活；第 12 章的激活重计算就是用计算换显存。

以一个线性层 $$\mathbf{y}=\mathbf{x}\mathbf{W}+\mathbf{b}$$ 为例，设上游梯度为 $$\mathbf{g}=\partial z/\partial\mathbf{y}$$。则 $$\partial z/\partial\mathbf{W}=\mathbf{x}^{\top}\mathbf{g}$$、$$\partial z/\partial\mathbf{b}=\mathbf{g}$$、$$\partial z/\partial\mathbf{x}=\mathbf{g}\mathbf{W}^{\top}$$。三个式子都没有出现 Jacobian：它们只是把上游梯度按不同方向投影回去。这就是反向传播的全部机制；复杂的网络只是把这个过程沿计算图重复多次。

Softmax 加交叉熵的梯度值得单独记住。设 logits 为 $$\mathbf{z}$$、概率为 $$\mathbf{p}=\operatorname{softmax}(\mathbf{z})$$，标签为 one-hot 向量 $$\mathbf{y}$$，则交叉熵 $$\mathcal{L}=-\sum_i y_i\log p_i$$ 对 logits 的梯度是 $$\partial\mathcal{L}/\partial\mathbf{z}=\mathbf{p}-\mathbf{y}$$。这个结果把“预测概率减真实标签”直接变成梯度，也是第 1 章交叉熵损失与第 3 章输出层能端到端训练的原因。若自己实现，应当用 `log_softmax` 而不是先 `softmax` 再取对数，理由见 A.6。

{% hint style="warning" %}
矩阵求导有两种常见布局：分子布局（梯度与自变量同形）与分母布局（梯度是自变量的转置形）。本书统一使用“梯度与参数同形”的约定，因此 $$\partial(\mathbf{a}^{\top}\mathbf{x})/\partial\mathbf{x}=\mathbf{a}$$。换用其他资料时先确认布局，否则会差一个转置。另一个常见错误是把 softmax 的 Jacobian 当成逐元素运算：它是 $$n\times n$$ 矩阵，只有在计算 VJP 时才无需显式构造。
{% endhint %}

## A.3 概率与统计

| 分布 | 概率质量或密度 | 均值 | 方差 | 在正文中的用途 |
| --- | --- | --- | --- | --- |
| Bernoulli $$(p)$$ | $$P(X=1)=p$$ | $$p$$ | $$p(1-p)$$ | 单步成功、采样 |
| Categorical $$(\mathbf{p})$$ | $$P(X=k)=p_k$$ | 向量 $$\mathbf{p}$$ | $$p_k(1-p_k)$$；协方差 $$-p_jp_k$$ | 下一 Token 分布 |
| 高斯 $$\mathcal{N}(\mu,\sigma^2)$$ | $$\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/(2\sigma^2)}$$ | $$\mu$$ | $$\sigma^2$$ | 初始化、扩散、噪声 |
| 均匀 $$U(a,b)$$ | $$\frac{1}{b-a}$$ | $$\frac{a+b}{2}$$ | $$\frac{(b-a)^2}{12}$$ | 随机初始化、量化 |
| Dirichlet $$(\boldsymbol{\alpha})$$ | $$\frac{\Gamma(\alpha_0)}{\prod_i\Gamma(\alpha_i)}\prod_i x_i^{\alpha_i-1}$$ | $$\alpha_i/\alpha_0$$ | $$\frac{\alpha_i(\alpha_0-\alpha_i)}{\alpha_0^2(\alpha_0+1)}$$ | 数据配比、主题分布 |

期望是线性的：$$\mathbb{E}[aX+bY]=a\mathbb{E}[X]+b\mathbb{E}[Y]$$，不需要独立性。方差不是线性的：$$\operatorname{Var}(X+Y)=\operatorname{Var}(X)+\operatorname{Var}(Y)+2\operatorname{Cov}(X,Y)$$，只有独立（或至少不相关）时协方差项才消失。条件期望满足全期望公式 $$\mathbb{E}[X]=\mathbb{E}\bigl[\mathbb{E}[X\mid Y]\bigr]$$；全方差公式为 $$\operatorname{Var}(X)=\mathbb{E}\bigl[\operatorname{Var}(X\mid Y)\bigr]+\operatorname{Var}\bigl(\mathbb{E}[X\mid Y]\bigr)$$。第 27 章讨论聚类抽样与设计效应时，用的就是后者。

条件概率定义为 $$P(A\mid B)=P(A,B)/P(B)$$（要求 $$P(B)>0$$），由此得到贝叶斯公式 $$P(A\mid B)=P(B\mid A)P(A)/P(B)$$。两个事件独立当且仅当 $$P(A,B)=P(A)P(B)$$；条件独立则要求 $$P(A,B\mid C)=P(A\mid C)P(B\mid C)$$。自回归分解用的是条件概率的链式法则，朴素贝叶斯假设特征在给定类别下条件独立，分层抽样则依赖按层分别建模的分布。协方差矩阵 $$\boldsymbol{\Sigma}=\mathbb{E}[(\mathbf{X}-\boldsymbol{\mu})(\mathbf{X}-\boldsymbol{\mu})^{\top}]$$ 把各分量的方差与两两协方差放在一起；它总是半正定的，正定时才有逆。

中心极限定理说，在温和条件下，独立同分布随机变量之和减去均值、再除以标准差，依分布收敛到标准正态；对样本均值而言，这个标准差就是标准误 $$\sigma/\sqrt{n}$$。它是给出 $$\pm1.96\,\mathrm{SE}$$ 这类区间的依据。但正态近似需要样本量足够大、分布不太偏；比例接近 0 或 1 时，正态区间会越出 $$[0,1]$$，第 27 章因此改用 Wilson 区间。区间估计回答“估计量有多不稳”，不回答“两个模型谁更好”；后者要配对比较与假设检验。

最大似然估计把“让数据概率最大”变成“让负对数似然最小”。对类别分布，最大似然解是观测频率，负对数似然就是交叉熵；对高斯分布，$$\hat\mu=\frac{1}{N}\sum_i x_i$$，而 $$\hat\sigma^2=\frac{1}{N}\sum_i(x_i-\hat\mu)^2$$ 是有偏的，无偏版本分母为 $$N-1$$。第 29 章式 (29.8) 的方差递推正是用了除以 $$N$$ 的版本，才得到收缩因子 $$(N-1)/N$$；换成无偏估计，递推会多出一个修正项。这是“公式里的 $$N$$ 与 $$N-1$$ 不能随手互换”的具体例子。

评估估计量时，平方误差可以分解为偏差、方差与不可约噪声：$$\mathbb{E}[(y-\hat y)^2]=\operatorname{Bias}(\hat y)^2+\operatorname{Var}(\hat y)+\sigma_\varepsilon^2$$。偏差描述系统性偏离，方差描述对训练集的敏感度；增大模型容量通常降低偏差、提高方差。第 27 章的置信区间估计的是“估计量的抽样波动”，标准误为 $$\sigma/\sqrt{n}$$（比例的版本是 $$\sqrt{p(1-p)/n}$$）。样本量增大时标准误按 $$1/\sqrt{n}$$ 下降，这也是“多采样几条轨迹”收益递减的来源。

{% hint style="warning" %}
标准差描述单个观测的离散程度，标准误描述估计量的波动；报告 $$\pm$$ 时必须说明是哪一种。方差的期望不等于标准差的期望：$$\mathbb{E}[\sqrt{V}]\neq\sqrt{\mathbb{E}[V]}$$，第 29 章因此报告“方差均值的开方”，而不是“标准差的均值”。另外，$$1-(1-p)^n$$ 这类非线性函数不能把平均概率直接代入，除非各提示的 $$p$$ 相同；第 14 章用 Jensen 不等式说明了这种高估。
{% endhint %}

## A.4 信息论

| 量 | 定义 | 单位与说明 |
| --- | --- | --- |
| 熵 | $$H(p)=-\sum_x p(x)\log p(x)$$ | 自然对数时为 nat，以 2 为底时为 bit |
| 联合熵 | $$H(X,Y)=-\sum_{x,y}p(x,y)\log p(x,y)$$ | 两个变量的总不确定性 |
| 条件熵 | $$H(X\mid Y)=H(X,Y)-H(Y)$$ | 已知 $$Y$$ 后 $$X$$ 的剩余不确定性 |
| 交叉熵 | $$H(p,q)=-\sum_x p(x)\log q(x)$$ | 用 $$q$$ 编码 $$p$$ 的平均长度 |
| KL 散度 | $$D_{\mathrm{KL}}(p\,\Vert\,q)=\sum_x p(x)\log\frac{p(x)}{q(x)}$$ | 非负，不对称 |
| 互信息 | $$I(X;Y)=H(X)-H(X\mid Y)$$ | 也等于 $$D_{\mathrm{KL}}\bigl(p(x,y)\,\Vert\,p(x)p(y)\bigr)$$ |
| 困惑度 | $$\mathrm{PPL}=\exp(H(p,q))$$ | 交叉熵以 nat 计时；以 bit 计则底数为 2 |
| 每字节比特数 | $$\mathrm{BPB}=\frac{H(p,q)}{\ln 2\cdot\bar b}$$ | $$\bar b$$ 是每 Token 平均字节数；跨分词器比较用这个量 |

交叉熵可以拆成“数据本身的熵”加“模型与数据的 KL 散度”，这是第 1 章式 (1.4) 的核心：

$$
\begin{aligned}
H(p,q)
&=-\sum_x p(x)\log q(x)\\
&=-\sum_x p(x)\log p(x)+\sum_x p(x)\log\frac{p(x)}{q(x)}\\
&=H(p)+D_{\mathrm{KL}}(p\,\Vert\,q)
\end{aligned}
$$

第一项与模型无关，因此最小化交叉熵等价于最小化 KL 散度。KL 的非负性来自 Jensen 不等式：$$-\log$$ 是凸函数，所以 $$D_{\mathrm{KL}}(p\,\Vert\,q)=-\mathbb{E}_p[\log(q/p)]\ge-\log\mathbb{E}_p[q/p]=0$$，等号仅在 $$p=q$$ 几乎处处成立时取到。

KL 不对称，方向决定优化行为。前向 KL $$D_{\mathrm{KL}}(p\,\Vert\,q)$$ 惩罚“$$p$$ 有质量而 $$q$$ 接近零”的区域，倾向于覆盖全部支撑集；反向 KL 允许 $$q$$ 忽略 $$p$$ 的长尾，倾向于聚焦少数模式。第 19 章的蒸馏、第 17 章的 DPO 与第 16 章的 KL 正则，都在选择方向与强度。困惑度只在同一分词器、同一评测集上可比；不同分词器的 Token 边界不同，直接比较 PPL 没有意义，应换算成每字节比特数。

熵满足链式法则 $$H(X,Y)=H(X)+H(Y\mid X)$$，因此条件熵可写成两个联合熵之差，且非负。互信息度量“知道 $$Y$$ 后 $$X$$ 的不确定性减少了多少”；若随机变量构成马尔可夫链 $$X\to Y\to Z$$，则数据处理不等式给出 $$I(X;Z)\le I(X;Y)$$：对 $$Y$$ 再做任何确定性或随机处理，都不会增加关于 $$X$$ 的信息。第 28 章的线性探针就是在检验中间表示 $$Y$$ 中还剩多少关于标签 $$X$$ 的信息；Fano 不等式把解码误差与 $$I(X;Y)$$ 联系起来，探针能做到多好受这个上界约束。这也是“探测到某属性”不等于“模型使用了该属性”的信息论原因。

{% hint style="warning" %}
KL 散度不是距离：它不满足对称性与三角不等式，不要写“两个分布的 KL 距离”。交叉熵不小于熵，差值正是 KL；不要把交叉熵与困惑度混为一谈，后者是指数变换。以 2 为底与以 $$e$$ 为底的熵相差一个 $$\ln 2$$ 因子，报告数字时必须注明底数。
{% endhint %}

## A.5 优化

| 方法 | 更新或目标 | 适用条件与代价 |
| --- | --- | --- |
| 梯度下降 | $$\theta_{t+1}=\theta_t-\eta\nabla\mathcal{L}(\theta_t)$$ | 每步一次全量梯度；条件数大时收敛慢 |
| 随机梯度下降 | 用一小批样本估计 $$\mathbf{g}_t$$ | 无偏估计，方差随批大小下降 |
| 动量 | $$\mathbf{m}_t=\beta\mathbf{m}_{t-1}+(1-\beta)\mathbf{g}_t$$ | 平滑噪声、加速一致方向 |
| Adam | 一阶矩与二阶矩的偏差修正更新 | 坐标自适应步长，内存为参数量的 3 倍 |
| 牛顿法 | $$\theta_{t+1}=\theta_t-\mathbf{H}^{-1}\nabla\mathcal{L}$$ | 局部二次收敛；构造与求逆代价高 |
| L2 正则 | $$\mathcal{L}+\frac{\lambda}{2}\lVert\theta\rVert_2^2$$ | 抑制大权重；与解耦权重衰减不同 |
| 拉格朗日法 | $$\mathcal{L}+\sum_i\nu_i g_i(\theta)$$ | $$\nu_i$$ 为乘子；约束优化由 KKT 条件给出最优性 |

梯度下降来自一阶泰勒展开：$$\mathcal{L}(\theta+\Delta)\approx\mathcal{L}(\theta)+\nabla\mathcal{L}^{\top}\Delta$$，取 $$\Delta=-\eta\nabla\mathcal{L}$$ 使线性项下降。牛顿法保留二阶项，即 $$\mathcal{L}(\theta+\Delta)\approx\mathcal{L}(\theta)+\nabla\mathcal{L}^{\top}\Delta+\frac{1}{2}\Delta^{\top}\mathbf{H}\Delta$$；对 $$\Delta$$ 求导并令其为零，得到 $$\Delta=-\mathbf{H}^{-1}\nabla\mathcal{L}$$。二次型上的梯度下降收敛速度由条件数决定：最优步长下误差每步约按 $$(\kappa-1)/(\kappa+1)$$ 收缩，$$\kappa$$ 越大越慢。第 11 章的实验用条件数 1000 的二次型展示这一点，并说明 Adam 的坐标缩放为什么能缓解病态方向。动量表中的更新采用 $$\mathbf{m}_t=\beta\mathbf{m}_{t-1}+(1-\beta)\mathbf{g}_t$$ 的约定；PyTorch 的 `SGD(momentum=...)` 少一个 $$(1-\beta)$$ 因子，等价于放大学习率，这一点第 11 章已说明。

随机梯度的关键性质是无偏：$$\mathbb{E}[\mathbf{g}_t]=\nabla\mathcal{L}(\theta_t)$$。噪声方差随批大小增大而下降，但收益按 $$1/B$$ 递减，因此大批量训练要靠学习率缩放与更长的预热来补偿。正则化可以从两个角度理解：在损失上加 $$\frac{\lambda}{2}\lVert\theta\rVert^2$$ 等价于高斯先验下的最大后验估计；在每步更新中直接把参数乘以 $$(1-\eta\lambda)$$ 则是解耦权重衰减。第 11 章式 (11.6)–(11.7) 说明，二者在 SGD 下近似等价，在 Adam 下并不等价。

凸函数满足 $$f(\lambda\mathbf{x}+(1-\lambda)\mathbf{y})\le\lambda f(\mathbf{x})+(1-\lambda)f(\mathbf{y})$$。凸优化的任何局部最优都是全局最优，且一阶条件 $$\nabla f(\mathbf{x}^*)=0$$ 充分；深度学习的目标通常非凸，这个保证不成立。但在过参数化网络中，大量局部最优的损失值彼此接近，真正拖慢训练的是病态曲率、鞍点与噪声，而不是“掉进一个很差的局部谷底”。

带约束的问题写成 $$\min_{\theta}f(\theta)\ \text{s.t.}\ g_i(\theta)\le0$$ 时，拉格朗日函数为 $$L=f+\sum_i\nu_i g_i$$，最优解满足 KKT 条件：对 $$\theta$$ 的偏导为零、乘子非负、互补松弛 $$\nu_i g_i=0$$。第 17 章的 DPO 就是一个例子：在“最大化奖励”与“不要偏离参考策略太远”之间做 KL 正则约束，先写出带约束的目标，再对策略求闭式最优解，最后把奖励消去，得到只依赖偏好数据的损失。看懂这一步，比记住 DPO 的最终公式更有用。

{% hint style="warning" %}
Adam 不是二阶方法：它只估计每个坐标的梯度尺度，不建模参数之间的曲率耦合，因此不能替代牛顿法或自然梯度。耦合 L2 与解耦权重衰减在自适应优化器下会产生不同的有效衰减，迁移超参数前先确认实现。深度网络训练的主要障碍通常不是“局部最优”，而是鞍点、平坦方向与条件数；把一切训练失败都归因于局部最优，会错过真正的原因。
{% endhint %}

## A.6 数值稳定性与浮点

| 操作 | 稳定写法 | 解决的问题 |
| --- | --- | --- |
| Softmax | $$p_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}$$，$$m=\max_j z_j$$ | 指数溢出 |
| Log-sum-exp | $$\log\sum_i e^{x_i}=m+\log\sum_i e^{x_i-m}$$ | 上溢与下溢 |
| Log-softmax | $$\log p_i=z_i-\log\sum_j e^{z_j}$$ | 先取概率再取对数造成的下溢 |
| $$\log(1+x)$$ | `log1p(x)` | $$x$$ 很小时的有效数字丢失 |
| $$e^x-1$$ | `expm1(x)` | $$x$$ 很小时的有效数字丢失 |
| 方差 | 先减均值再平方，或用 Welford 单遍更新 | 大均值下的灾难性抵消 |
| 除法与开方 | 分母或根号内加 $$\epsilon$$，如 $$\sqrt{v+\epsilon}$$ | 除零与负零 |

Softmax 的稳定性来自一个恒等式：分子分母同乘 $$e^{-m}$$ 不改变结果，但让最大的指数变成 $$e^0=1$$，其余项都不超过 1，求和不会溢出。Log-sum-exp 用同一技巧；如果先算 $$\sum e^{x_i}$$ 再取对数，$$x_i$$ 稍大就会得到 `inf`，稍小则全部下溢为零。第 21 章的 FlashAttention 在线 Softmax 把这个恒等式做成流式更新，每处理一个块就修正一次最大值与归一化因子，因此无需保存完整的注意力矩阵。

浮点格式决定“能表示多大”和“能表示多细”。

| 格式 | 符号/指数/尾数 | 典型范围 | 有效精度 | 常见用途 |
| --- | --- | --- | --- | --- |
| FP32 | 1/8/23 | 约 $$10^{-38}$$ 到 $$3\times10^{38}$$ | 约 7 位十进制 | 优化器状态、损失累加 |
| FP16 | 1/5/10 | 约 $$6\times10^{-8}$$ 到 $$6.5\times10^4$$ | 约 3–4 位十进制 | 混合精度前向反向 |
| BF16 | 1/8/7 | 与 FP32 同量级 | 约 2–3 位十进制 | 大模型训练 |
| FP8 E4M3 | 1/4/3 | 约 $$2\times10^{-3}$$ 到 448 | 很低 | 推理、部分训练 |
| FP8 E5M2 | 1/5/2 | 约 $$1.5\times10^{-5}$$ 到 $$5.7\times10^4$$ | 很低 | 梯度、极端值 |

FP16 的指数位少，最大值只有约 $$6.5\times10^4$$，小梯度容易下溢；BF16 保留 8 位指数，动态范围与 FP32 接近，但尾数只有 7 位，精度更低。混合精度训练通常用 BF16/FP16 做矩阵乘、用 FP32 做损失累加与优化器状态，原因就在这里。第 23 章的量化把这套取舍推到 INT8/FP8：范围、精度与硬件吞吐三者不能同时最优。

{% hint style="danger" %}
不要先 `exp` 再归一化，也不要在未减最大值的情况下计算 softmax；前者会溢出，后者在长序列上会得到全零或 `nan`。用 BF16 累加长序列的损失时，求和顺序会影响结果，必要时用 FP32 累加。方差的单遍公式 $$\mathbb{E}[X^2]-\mathbb{E}[X]^2$$ 在均值远大于标准差时会严重抵消，优先用两遍法或 Welford。
{% endhint %}

## A.7 常用近似与不等式

| 近似或不等式 | 形式 | 成立条件与用途 |
| --- | --- | --- |
| 一阶对数近似 | $$\log(1-x)\approx-x$$ | $$\lvert x\rvert\ll1$$；采样成本估计 |
| 二阶对数近似 | $$\log(1+x)\approx x-\frac{x^2}{2}$$ | $$\lvert x\rvert\ll1$$；小概率修正 |
| 指数近似 | $$e^x\approx1+x$$ | $$\lvert x\rvert\ll1$$；学习率分析 |
| 复利近似 | $$(1-x)^n\approx e^{-nx}$$ | $$n$$ 大、$$x$$ 小；成功概率衰减 |
| 小概率并集 | $$1-(1-p)^n\approx np$$ | $$np\ll1$$；pass@k 与稀疏奖励 |
| Stirling 公式 | $$n!\approx\sqrt{2\pi n}\,(n/e)^n$$ | $$n$$ 很大；组合数估计 |
| Cauchy–Schwarz | $$\lvert\langle\mathbf{x},\mathbf{y}\rangle\rvert\le\lVert\mathbf{x}\rVert_2\lVert\mathbf{y}\rVert_2$$ | 任意内积空间 |
| Jensen 不等式 | 凸 $$f$$：$$f(\mathbb{E}[X])\le\mathbb{E}[f(X)]$$ | 凹函数方向相反 |
| 并集界 | $$P(\bigcup_i A_i)\le\sum_i P(A_i)$$ | 任意事件；保守上界 |
| Markov 不等式 | $$P(X\ge a)\le\mathbb{E}[X]/a$$ | $$X\ge0$$、$$a>0$$ |
| Chebyshev 不等式 | $$P(\lvert X-\mu\rvert\ge k\sigma)\le1/k^2$$ | 方差已知；分布无关 |

这些近似在正文中都有具体落点。第 29 章用 $$\log(1-p^K)\approx-p^K$$ 把“看到一次成功所需的轨迹数”化简为 $$\log(1/\delta)/p^K$$；第 14、20 章用 $$1-(1-p)^n\approx np$$ 解释小样本下的 pass@k；第 29 章的方差递推则反复使用 $$(1-1/N)^t\approx e^{-t/N}$$。使用前先检查条件：当 $$p$$ 不接近零、$$n$$ 不够大时，近似会系统性高估或低估，应改用精确公式或数值计算。

{% hint style="warning" %}
近似只在给定区间内成立，不能因为形式上“更像闭式解”就无条件使用。Jensen 不等式的方向取决于凸凹性：对 $$\log$$ 这类凹函数，$$f(\mathbb{E}[X])\ge\mathbb{E}[f(X)]$$，与凸函数相反。Markov 与 Chebyshev 给出的是上界，不是等号；把它们当成精确概率是常见错误。
{% endhint %}

## A.8 与正文的对应关系

| 主题 | 建议先读 | 后续主要用到 |
| --- | --- | --- |
| 向量、矩阵与分解 | 本附录 A.1 | 第 3、5、6、15、23 章 |
| 矩阵微积分与反向传播 | 本附录 A.2 | 第 3、11、15 章 |
| 概率、估计与区间 | 本附录 A.3 | 第 1、9、16–18、27、29 章 |
| 信息论 | 本附录 A.4 | 第 1、16–19、27 章 |
| 优化与正则化 | 本附录 A.5 | 第 11、14–18 章 |
| 数值稳定与浮点 | 本附录 A.6 | 第 11、21、23 章 |
| 近似与不等式 | 本附录 A.7 | 第 14、18、20、29 章 |

{% hint style="success" %}
读正文遇到不熟悉的数学记号时，先查附录 B 确认符号语义，再回到本附录看该工具的定义与条件，最后读对应章节的完整推导。附录 A 只负责“够用”，不替代教材；如果某个主题反复卡住，说明需要补的是那门课的基础，而不是继续在本书里找捷径。
{% endhint %}
