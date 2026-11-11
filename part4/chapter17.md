# 第 17 章 直接偏好优化与相关变体

## 本章导读

第 16 章的 PPO 需要奖励模型、参考模型、价值网络和在线采样四套组件。能不能只用一份偏好数据、两个模型，就把策略调整到人类更喜欢的方向？DPO 的答案是：把奖励模型从目标里“解”出来，让策略自己充当奖励模型。代价是，它再也无法通过在线探索发现偏好数据之外的问题。

## 难度标签

⭐⭐⭐ 进阶

- 数学：Bradley–Terry 模型、KL 正则、对数似然与梯度推导；
- 代码：PyTorch，理解第 14 章 SFT 与第 16 章 RLHF 的损失与采样流程；
- 数据：成对偏好数据的构造、偏置与评测。

## 学习目标

1. 能从 KL 正则的 RLHF 目标反解出隐式奖励，并说明配分函数为什么在成对比较中抵消。
2. 能写出 DPO 损失及其梯度，解释每一项在训练中起什么作用。
3. 能实现序列对数概率、成对批处理、长度归一化与标签平滑，并跑通一个三 seed 对照实验。
4. 能区分 DPO、IPO、KTO、SimPO、ORPO 的目标差异与适用条件。
5. 能识别 DPO 的主要失败模式：偏好标签偏置、分布外样本、长度偏置与 β—学习率耦合。

## 17.1 为什么还要 DPO

第 16 章的 RLHF 管线可以概括为：先用偏好数据训练奖励模型，再用 PPO 在奖励模型上做在线强化学习。它有效，但工程负担很重。一次完整的 PPO 训练要同时维护四份模型：策略、参考、奖励、价值。参考模型和奖励模型都要冻结，价值网络要和策略一起更新；每轮还要从策略采样大量回答，再用奖励模型打分。采样吞吐、奖励作弊、价值估计误差、KL 系数调节，任何一个环节出问题，训练都会崩。

DPO（Direct Preference Optimization，直接偏好优化）换了一个问法：既然 RLHF 的最优策略有闭式形式，能不能把这个闭式形式代回偏好模型，消掉奖励模型？如果可以，偏好学习就变成一次监督式训练：不需要奖励模型，不需要在线采样，只需要策略模型、冻结的参考模型和成对数据。

这个想法之所以成立，关键在于两点。第一，Bradley–Terry 模型只关心两条回答的奖励差，不关心奖励的绝对数值；而 RLHF 闭式解给出的奖励，恰好可以写成“策略与参考模型的对数比”加上一个只依赖提示的常数。第二，两条回答共享同一个提示，这个常数在相减时抵消。于是，奖励模型被“吸收”进策略的对数概率里，偏好概率可以直接用策略和参考模型算出来。

DPO 不是把 PPO 原样简化，而是换了一种等价参数化。它放弃的是在线探索：PPO 可以用当前策略不断生成新回答、让奖励模型发现新的高分区；DPO 只能反复看同一批离线偏好对。如果偏好数据没有覆盖某个行为，DPO 没有机制主动去发现它。这个代价在后面的实验里会以“长度偏置被原样学走”的形式出现。

在完整的后训练流水线里，DPO 通常站在 SFT 之后。第 14 章的 SFT 负责让模型学会指令格式与基本回答能力，DPO 负责在参考模型附近调整偏好。若跳过 SFT 直接对基座做 DPO，参考模型的对数概率还没有“会回答问题”的结构，隐式奖励的零点没有意义，策略容易在低质量区域里做对比。工业流程有时还会在 SFT 与 DPO 之间插入拒绝采样：先用奖励筛选出高质量回答做一次 SFT，再用偏好数据做 DPO。第 18 章的 RLVR 则是在 DPO 之后，用可验证奖励继续提升推理能力。把这几个阶段分清，才能判断一份“DPO 提升”的报告到底提升的是格式、偏好还是推理。

DPO 也不是 PPO 的唯一替代。它和 PPO 的关系更像“离线策略优化”与“在线策略优化”的关系：前者用固定数据做监督式拟合，后者用当前策略采样做强化学习。两者共享同一个 KL 正则目标与同一个闭式最优解，区别在于数据从哪里来、奖励由谁提供、优化是否闭环。

## 17.2 直觉：把奖励模型变成策略的“影子”

先不看公式，用一个类比。假设你有一把尺子（奖励模型），可以给任意回答打分。PPO 的做法是：让选手（策略）不断试跳，用尺子量，再按分数调整动作。DPO 的做法是：不造尺子，而是规定“选手相对于训练起点的进步幅度”就是分数。谁比起点进步得多，谁就赢。于是，分数不再是一个外部模型，而是策略与参考模型的对数概率之比。

这个类比有两个直接推论。

第一，参考模型定义了“零分”。同一份策略，换一个参考模型，隐式奖励就完全不同。参考模型通常是 SFT 模型，它决定了“哪些行为算进步、哪些算退步”。这也是为什么 DPO 必须冻结参考模型：如果参考模型跟着策略一起变，“进步”的基准就漂移了，损失会退化成没有锚点的对比学习。

第二，DPO 只能比较“同一提示下的两条回答”。隐式奖励里有一个只依赖提示的常数，单看一条回答无法确定它的绝对分数；只有两条回答共享提示时，常数才抵消。这解释了为什么 DPO 的数据必须是成对的，也解释了为什么跨提示比较隐式奖励没有意义。

把这两个推论放进一个具体数字里。设参考模型对赢家、输家的对数概率相同，所以两者的参考对数比都是 0。训练初期，策略把赢家的对数概率提高了 0.2、把输家降低了 0.1，于是 $$\Delta=0.2-(-0.1)=0.3$$。取 β=0.5，margin 为 0.15，偏好概率是 $$\sigma(0.15)\approx0.537$$，损失约 0.62。若继续训练，把赢家再提高 0.4、输家再降低 0.2，margin 变成 0.45，偏好概率升到 0.61，损失降到 0.50。注意，赢家的绝对对数概率只提高了 0.6，但损失已经明显下降；DPO 关心的是相对间隔，不是赢家本身有多好。这也解释了为什么“损失下降”不能直接当作“回答质量上升”。

```mermaid
flowchart LR
    A[偏好对 x, y_w, y_l] --> B[策略 πθ 与参考 π_ref]
    B --> C[四个序列对数概率]
    C --> D[隐式奖励差 Δ]
    D --> E[σ(βΔ) 与 DPO 损失]
    E --> F[只更新策略]
    G[冻结参考模型] --> C
```

图 17-1：DPO 的前向与反向。没有奖励模型，没有 rollout，参考模型只提供固定的对数概率基准。

从优化角度看，DPO 损失像一个带“锚”的成对排序损失。普通的成对排序只要求赢家的分数高于输家；DPO 还要求这个分数差来自“策略相对参考模型的概率变化”，偏离参考模型越远，隐式奖励越极端。β 控制这个锚的松紧：β 小，隐式奖励对概率比不敏感，策略可以大幅偏离参考模型才获得同样的奖励差；β 大，同样的概率比会产生更大的奖励差，损失对偏离更敏感。

但 β 的作用不止于此。它同时出现在损失函数的 logit 里和梯度的前面，因此也改变有效学习率。后文会用实验说明：β 不是单一方向的“正则旋钮”，调它时必须同时看学习率和实际 KL。

## 17.3 数学原理

### 17.3.1 从 RLHF 目标反解奖励

第 16 章式 (16.4) 给出了 KL 正则目标的闭式最优策略：

$$
\pi^\star(y\mid x)
= \frac{1}{Z(x)}\,\pi_{\rm ref}(y\mid x)\,
\exp\!\left(\frac{r(x,y)}{\beta}\right),
\qquad
Z(x)=\sum_{y'}\pi_{\rm ref}(y'\mid x)\,
\exp\!\left(\frac{r(x,y')}{\beta}\right)
$$

这里 $$r$$ 是真实的（或奖励模型给出的）奖励，$$\pi_{\rm ref}$$ 是冻结的参考策略，$$\beta$$ 是 KL 系数。把上式两边取对数并移项，可以把奖励写成策略与参考模型的对数比：

$$
r(x,y)
= \beta\log\frac{\pi^\star(y\mid x)}{\pi_{\rm ref}(y\mid x)}
+ \beta\log Z(x)
\qquad \text{(17.1)}
$$

这个式子是 DPO 的起点。它说明：在最优策略处，奖励（差一个只依赖 $$x$$ 的常数）等于“策略相对参考模型的对数概率比”乘以 β。换句话说，奖励模型并不携带独立信息；一旦知道了最优策略和参考策略，奖励就被确定了。

在实践中我们并不知道 $$\pi^\star$$。DPO 的做法是用参数化策略 $$\pi_\theta$$ 代替它，并定义隐式奖励

$$
r_\theta(x,y)
= \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\rm ref}(y\mid x)}
\qquad \text{(17.2)}
$$

注意，(17.2) 省略了 (17.1) 中的 $$\beta\log Z(x)$$。这不是近似，而是因为后面的成对比较只用到同一提示下的奖励差，而 $$\beta\log Z(x)$$ 只依赖 $$x$$，在相减时严格抵消。这也是 DPO 不需要计算配分函数的根本原因：配分函数是“提示级”的常数，不是“回答级”的信号。

### 17.3.2 代入 Bradley–Terry 得到 DPO 损失

第 16 章式 (16.1) 的 Bradley–Terry 模型假设

$$
P(y_w\succ y_l\mid x)
= \sigma\big(r(x,y_w)-r(x,y_l)\big)
$$

把 (17.2) 的隐式奖励代进去，得到

$$
P_\theta(y_w\succ y_l\mid x)
= \sigma\!\left(
\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\rm ref}(y_w\mid x)}
-
\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\rm ref}(y_l\mid x)}
\right]
\right)
\qquad \text{(17.3)}
$$

括号里的量记为 $$\Delta_\theta(x,y_w,y_l)$$。它只涉及四个序列对数概率：策略在赢家、输家上的对数概率，参考模型在赢家、输家上的对数概率。四个数都可以通过一次前向得到，不需要奖励模型，也不需要采样。

给定偏好数据集 $$D_{\rm pref}=\{(x^{(i)},y_w^{(i)},y_l^{(i)})\}$$，DPO 的负对数似然是

$$
\mathcal{L}_{\rm DPO}(\theta)
= -\mathbb{E}_{(x,y_w,y_l)\sim D_{\rm pref}}
\left[
\log\sigma\big(\beta\,\Delta_\theta(x,y_w,y_l)\big)
\right]
\qquad \text{(17.4)}
$$

这就是代码里的 `dpo_loss`。它和 (16.2) 的奖励模型损失形式相同，区别只在“奖励”从 $$r_\phi$$ 换成了隐式奖励 $$r_\theta$$。因此，DPO 可以看作“把奖励模型和策略合并成一个网络”的 Bradley–Terry 回归。

这个等价关系需要说清楚边界。在**模型假设成立、数据覆盖充分、优化到全局最优**的条件下，最小化 (17.4) 得到的策略与 (16.3) 的 KL 正则最优策略一致。它不是“任意数据、任意优化轨迹下都与 PPO 等价”。数据没有覆盖的行为，DPO 没有信号；优化没有收敛时，两种方法的解也不同。把 DPO 说成“RLHF 的无损替代”是常见误解。

这里还有一个容易写错的“证明”。有人把 (17.2) 的隐式奖励直接代回 (16.3)，得到

$$
\mathbb{E}_{\pi_\theta}\left[\beta\log\frac{\pi_\theta}{\pi_{\rm ref}}\right]
-\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\rm ref})=0
$$

然后声称 DPO 目标恒为零，或者声称“RLHF 目标在 DPO 参数化下自动最优”。这个式子确实恒为零，但它不是 DPO 的等价关系，而是一个循环代入：被优化的奖励和 KL 项用了同一个策略。正确的等价关系是 (17.4)：它把 Bradley–Terry 似然中的奖励模型替换成策略的对数比，目标是让参数化策略**成为**偏好数据对应的最优策略。读 DPO 论文时，要区分“重参数化 BT 似然”和“把奖励代回原目标”这两件事。

### 17.3.3 梯度：谁错得多，谁被推得多

对 (17.4) 求梯度。记 $$m=\beta\Delta_\theta$$，参考模型不参与求导，因此

$$
\nabla_\theta\mathcal{L}_{\rm DPO}
= -\beta\,\sigma(-m)\,
\left[
\nabla_\theta\log\pi_\theta(y_w\mid x)
-
\nabla_\theta\log\pi_\theta(y_l\mid x)
\right]
\qquad \text{(17.5)}
$$

这个梯度有三个值得注意的性质。

第一，权重是 $$\sigma(-m)$$。当模型已经把赢家排在输家前面（$$m$$ 很大）时，权重接近零，这对数据几乎不再提供梯度；当模型排反了（$$m<0$$）时，权重接近一，梯度最大。这和 logistic 回归完全一致：DPO 自动把训练重点放在“当前还分错”的偏好对上。

第二，梯度只提高赢家的对数概率、降低输家的对数概率，参考模型只通过 $$m$$ 里的基准项影响权重。因此，DPO 不需要显式奖励，也不需要价值网络；它把“奖励”完全编码进了策略与参考模型的对数比。

第三，梯度没有上界裁剪。若某个偏好对极其难分，或者标签本身有噪声，模型会被持续推向“记住这一对”，而不是学到可泛化的偏好。DPO 对标签噪声的鲁棒性来自数据量和正则，而不是损失本身。第 14 章讨论过的系统性标签噪声，在 DPO 里会直接变成策略的行为偏置。

还要注意长度的影响。序列对数概率是逐 Token 对数概率之和，回答越长，和的绝对值通常越大，梯度也越大。若不做长度归一化，DPO 会天然地给长回答更大的权重；若偏好标签又恰好偏好长回答，两个效应会叠加。后面的实验会测量这一点。

### 17.3.4 β、长度归一化与有效步长

β 在 (17.4) 里出现两次：一次作为 logit 的缩放，一次通过梯度 (17.5) 作为整体步长。这带来一个容易忽略的耦合：固定学习率、只改 β，并不是“只改正则强度”，而是在同时改损失曲率和有效步长。

举一个极端例子。β 从 0.5 降到 0.1，梯度幅度大约变成原来的五分之一；如果学习率不变，训练会变慢、噪声占比变大，模型可能在 300 步内还没收敛，表现为不同 seed 的结果差异很大。反过来，β 从 0.5 升到 2.0，单对样本的梯度更大，但 $$\sigma(-m)$$ 饱和得更快，容易分的对被迅速压到零梯度，更新集中在少数难分对上。这两个效应方向相反，所以“β 越大越保守、KL 越小”并不是必然成立的。本章实验中 β=2.0 的 KL 反而低于 β=0.5，就是一个具体反例。

若要让 β 只承担“正则强度”的角色，需要同时调整学习率，使 $$\beta\cdot\eta$$ 大致不变。工业实现常报告“β 与 lr 的组合”，而不是单独报告 β，原因就在这里。

长度归一化是另一个改变目标的实现选择。设赢家、输家的回答长度分别为 $$|y_w|$$、$$|y_l|$$，长度归一化的对数比定义为

$$
\Delta^{\rm norm}_\theta
= \frac{1}{|y_w|}\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\rm ref}(y_w\mid x)}
- \frac{1}{|y_l|}\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\rm ref}(y_l\mid x)}
\qquad \text{(17.6)}
$$

它把“整条回答的对数概率”换成“平均每 Token 的对数概率”，削弱了长回答的权重。代价是隐式奖励的单位变了：β 的含义不再是“整条序列的奖励温度”，而是“每 Token 的奖励温度”。不同实现之间比较 β 时，必须先确认是否做了长度归一化。SimPO 就是在这个方向上走得更远的代表：它去掉参考模型，直接用长度归一化的平均对数似然加一个目标间隔。

### 17.3.5 变体：IPO、KTO、SimPO 与 ORPO

DPO 的损失不是唯一选择。围绕“如何把偏好变成目标”，出现了几类有代表性的变体。

**IPO（Identity Preference Optimization）** 认为 DPO 的 log-sigmoid 会把赢家与输家的间隔无限拉大，在确定性偏好下容易过拟合。它把目标改成一个带目标间隔的平方损失：

$$
\mathcal{L}_{\rm IPO}(\theta)
= \mathbb{E}\left[
\left(\Delta_\theta(x,y_w,y_l)-\frac{1}{2\beta}\right)^2
\right]
\qquad \text{(17.7)}
$$

当对数比差达到 $$1/(2\beta)$$ 时损失为零，超过也不再奖励。这个目标更保守，KL 通常更小；代价是它可能学得不够，尤其在偏好信号较弱时。本章实验中 IPO 的目标正确率明显低于 DPO，但 KL 只有 DPO 的几十分之一。

**KTO（Kahneman–Tversky Optimization）** 不要求成对数据，只需要“这条回答好/不好”的二元标签。它借用前景理论的价值函数，对收益和损失采用不对称的权重。优势是数据门槛低、可利用单边反馈；劣势是丢失了“同一提示下两条回答直接比较”的信息，学习信号更弱。

**SimPO（Simple Preference Optimization）** 去掉参考模型，使用长度归一化的平均对数似然，并在损失里加一个目标间隔。它只需要一份模型，训练和部署都更简单；代价是没有参考模型锚定，策略更容易漂移，尤其在数据分布窄或训练步数多时。

**ORPO（Odds Ratio Preference Optimization）** 把偏好项并入 SFT 损失，用 odds ratio 代替对数比，单阶段完成“指令微调 + 偏好对齐”。它省掉了单独的参考模型和偏好训练阶段；代价是两个目标的权重需要仔细调，且 SFT 数据与偏好数据必须匹配。

此外还有 cDPO（label smoothing）、RPO、CPO 等针对标签噪声或计算效率的变体。它们大多在“目标间隔、参考模型、长度归一化、数据形式”这四个维度上做取舍。读论文时先问这四个问题，比记损失长什么样更有用。

### 17.3.6 训练动态与数值实现

DPO 的损失是成对 log-sigmoid，训练动态比 SFT 的交叉熵更敏感。下面四个现象在实现和调参时经常遇到。

**似然位移。** 由于梯度只关心赢家与输家的相对差，模型可能同时降低两者的对数概率，只要输家降得更多，损失仍然下降。极端情况下，策略会把赢家和输家都推向参考模型分布之外，生成不自然但“相对更好”的回答。参考模型的 KL 锚可以缓解，但不能消除；实践中要监控赢家回答的绝对对数概率，而不只是相对间隔。若赢家的对数概率持续下降，说明模型在“靠压低输家”而不是“靠提高赢家”来降低损失。

**参考模型失配。** DPO 的隐式奖励以参考模型为零点。如果参考模型不是产生偏好数据时的策略，隐式奖励的基准就错了。比如用 Llama 2 的对数概率做参考、却用 Llama 3 的数据训练，模型会把“版本差异”误当成偏好信号。工业实现通常要求参考模型与 SFT 模型逐位一致，或者至少同源同规模。

**分布外样本。** 离线偏好对的回答来自旧策略，而训练中的策略会逐渐偏离。偏离越大，策略在这些旧回答上的对数概率越极端，梯度越容易饱和或爆炸。DPO 没有 PPO 的在线重采样来纠正这一点；迭代 DPO 通过定期用新策略生成回答来缓解，但标注成本随之上升。

**数值稳定。** 序列对数概率通常是很大的负数，两个对数概率相减可能损失精度。实现时应使用 `log_softmax` 加 `gather`，不要先算概率再取对数；损失用 `logsigmoid`，不要手写 `log(1+exp(-x))`。混合精度训练时，log-softmax 与求和最好留在 fp32；否则在长序列上容易出现 `-inf` 与 NaN。批内还要按有效 Token 掩码，避免 padding 位置的零对数概率干扰求和。

## 17.4 方案对比

表 17-1 把 DPO 放回第 16 章的坐标系里。关键不是“谁更强”，而是“哪一份组件可以省、省掉之后失去什么”。

| 方法 | 需要奖励模型 | 需要参考模型 | 在线采样 | 每步前向 | 主要风险 | 适用条件 |
| --- | --- | --- | --- | --- | --- | --- |
| PPO/RLHF | 是 | 是 | 是 | 策略+参考+奖励+价值 | 奖励作弊、训练不稳 | 算力充足、需要在线探索 |
| DPO | 否 | 是 | 否 | 策略+参考各两次 | 偏好标签偏置、分布外 | 有成对数据、算力有限 |
| Best-of-N | 是 | 否 | 推理时 | 策略+奖励 | 推理成本随 N 增长 | 只需推理时提分 |
| 拒绝采样微调 | 是 | 否 | 离线采样 | 策略+奖励 | 多样性塌缩 | 有可验证或可靠奖励 |
| 迭代 DPO | 否 | 是 | 是（重新标注） | 策略+参考各两次 | 标注成本、标签漂移 | 有持续人工/AI 反馈 |

DPO 的“每步前向”看起来少，但它对每个偏好对要做四次前向：策略在赢家、输家上各一次，参考模型在赢家、输家上各一次。参考模型的前向可以预计算并缓存，因为参考模型冻结、数据固定；这是 DPO 工程实现里最值得做的一项优化。PPO 的每步前向更多，但它的样本可以复用多个 epoch，二者不能只按单步成本比较。

把账算细一点。设单条序列长度为 $$T$$、模型参数量为 $$N$$。一次前向约需 $$2NT$$ FLOPs，一次前向加反向约需 $$6NT$$。DPO 每个偏好对需要策略的两条前向加反向（约 $$12NT_w+12NT_l$$），参考模型可以预计算，所以稳态每步主要花在策略上；PPO 每步则要支付 rollout 生成、奖励模型打分、价值网络前向，以及策略与价值的反向。DPO 的优势不在“单次前向更便宜”，而在“没有生成、没有奖励模型、没有价值网络”，并且参考项可以离线摊销。

显存上，DPO 至少需要两份完整权重（策略与参考）。若参考权重用全精度、策略用 LoRA，静态状态可以接近“一份基座 + 两份小适配器”；若两份都全参数训练，静态状态约是 SFT 的两倍。激活方面，DPO 的批里同时有赢家与输家两条序列，最大长度由两者中较长者决定；按长度排序、把相近长度的对放在同一批，可以显著减少 padding 浪费。这些工程细节不改变算法，但决定了同样的 β 与步数能否在单卡上跑完。

从数据效率看，DPO 把“偏好”直接变成策略的监督信号，通常比 PPO 更样本高效；但它对数据覆盖的要求更高。PPO 可以用在线采样探索数据之外的行为，DPO 只能在已有偏好对上反复拟合。若偏好数据只覆盖“短回答”，DPO 不会主动发现“长回答可能更好”；若数据里长回答被标为更好，DPO 会毫不犹豫地学走这个偏置。

这也意味着两种方法的失败模式是互补的。PPO 的风险主要落在优化侧：奖励作弊、价值估计偏差、KL 系数与裁剪半径失配；DPO 的风险主要落在数据侧：标签偏置、覆盖不足、参考模型失配。选型时先判断瓶颈在哪一侧。若已有较干净的成对数据、但工程资源不足以支撑在线 RL，DPO 的风险更容易通过数据审计控制；若奖励可以自动验证、且需要模型探索新行为，PPO 或第 18 章的 RLVR 更合适。把两者对立成“新旧之争”没有意义，它们优化的目标相同，只是一份风险被搬到了不同的位置。

## 17.5 代码实现

### 17.5.1 实验设计

实验沿用第 16 章的 16 词表玩具对话。提示固定为四个 Token（BOS、USER、主题、ASSISTANT），回答长度 3–5。四个主题各有目标词和形近干扰词；SFT 行为策略以 55% 概率输出目标词、10% 概率输出干扰词、其余均匀采样。SFT 用 512 条对话训练 400 步，得到冷启动策略。

偏好对由两部分构成：

- **偏置偏好**：奖励 = 目标词比例 + 0.30 × 干扰词比例 + 0.35 ×（长度 − 3）/3 + 噪声。它模拟“标注者偏好更长、偶尔把形近词当成正确答案”的真实偏置。
- **去偏偏好**：奖励只取目标词比例 + 噪声，长度与干扰词不进入标签。

两个数据集各 2,000 对，用同一批提示和回答生成器；评测用另外 400 对。DPO 训练 300 步、批 32、学习率 $$5\times10^{-4}$$、3 个种子。评测时从策略采样 256 条回答，报告目标词命中率、干扰词比例、内容长度，以及策略相对参考模型的逐 Token KL。留出偏好对上报告隐式奖励的排序准确率与平均间隔。

四个指标的读法要明确。目标词命中率只统计回答内容（去掉提示与结尾的 EOS），并且按内容 Token 数平均，否则长回答会因分母不同而不可比。干扰词比例回答的是“模型把形近错误答案当成正确答案的频率”，它是目标词命中率的反向探针：如果一个方法同时提高命中率、降低干扰词比例，才说明它学到了主题映射，而不是靠排除法。内容长度用于检查长度偏置，必须与任务指标并排看。逐 Token KL 是策略与参考模型在回答位置上的平均 $$\mathrm{KL}(\pi_\theta\|\pi_{\rm ref})$$，它只反映偏离幅度，不反映偏离方向。偏好对准确率则是在留出对上看隐式奖励是否把赢家排在输家前面，平局按失败计。任何一项单独看都可能被误导，这也是下一节把四类指标并列成表的原因。

### 17.5.2 完整代码

`tinylm/posttrain/dpo.py`：

```python
"""第 17 章：直接偏好优化（tinylm/posttrain/dpo.py）。"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DPOConfig:
    """DPO 训练脚本参数，不进入 ``ModelConfig``。"""

    beta: float = 0.1               # 隐式奖励的温度，也控制偏离参考模型的力度
    lr: float = 5e-5
    epochs: int = 1
    batch_size: int = 8
    loss_type: str = "dpo"          # dpo | ipo
    label_smoothing: float = 0.0    # cDPO 的平滑系数
    length_normalize: bool = False  # 按响应 Token 数平均对数概率
    max_grad_norm: float = 1.0

    def __post_init__(self):
        assert self.beta > 0.0
        assert self.lr > 0.0
        assert self.epochs >= 1
        assert self.batch_size >= 1
        assert self.loss_type in ("dpo", "ipo")
        assert 0.0 <= self.label_smoothing < 0.5
        assert self.max_grad_norm > 0.0


def completion_logprobs(model, input_ids, attention_mask, completion_mask,
                        length_normalize=False):
    """返回每条样本完成部分的对数概率，形状 ``(B,)``。

    ``completion_mask[:, t]`` 标记位置 ``t`` 的 Token 属于回答（含 EOS）。
    第 ``t`` 位的 logits 预测 Token ``t+1``，因此参与求和的掩码要左移一位。
    """
    logits = model(input_ids, attention_mask=attention_mask).logits[:, :-1]
    logp = F.log_softmax(logits, dim=-1)
    target = input_ids[:, 1:]
    token_logp = logp.gather(-1, target[..., None]).squeeze(-1)
    valid = completion_mask[:, 1:] & attention_mask[:, 1:]
    token_logp = torch.where(valid, token_logp, torch.zeros_like(token_logp))
    total = token_logp.sum(-1)
    if length_normalize:
        # (17.6) 按响应 Token 数平均
        total = total / valid.sum(-1).clamp_min(1).to(total.dtype)
    return total


def log_ratio_margin(policy_chosen, policy_rejected, ref_chosen, ref_rejected):
    """(17.3) 括号内不含 β 的项：赢家与输家的对数比之差。"""
    win = policy_chosen - ref_chosen
    lose = policy_rejected - ref_rejected
    return win - lose


def implicit_reward(logp, ref_logp, beta):
    """(17.2)：隐式奖励，策略与参考模型的对数比乘以 β。"""
    return beta * (logp - ref_logp)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected,
             beta=0.1, label_smoothing=0.0, loss_type="dpo"):
    """(17.4)：直接偏好优化的成对损失，返回批平均标量。"""
    ratio_margin = log_ratio_margin(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    margin = beta * ratio_margin
    if loss_type == "ipo":
        # (17.7) IPO 把 log 比拉向 1/(2β)，而不是无限拉大
        return ((ratio_margin - 0.5 / beta) ** 2).mean()
    if label_smoothing > 0.0:
        eps = label_smoothing
        return -(F.logsigmoid(margin) * (1.0 - eps)
                 + F.logsigmoid(-margin) * eps).mean()
    return -F.logsigmoid(margin).mean()


def preference_accuracy(chosen_margin):
    """隐式奖励把赢家排在输家之前的比例；平局按失败计。"""
    return (chosen_margin > 0).float().mean()
```

`examples/ch17_dpo.py`：

```python
"""第 17 章：直接偏好优化（examples/ch17_dpo.py）。"""
import copy
import random
import time

import torch

from ch08 import ModelConfig, TinyLM
from tinylm.posttrain import dpo as D
from tinylm.posttrain import ppo as P
from tinylm.posttrain import sft
from tinylm.tokenizer.chat import ChatTemplate

PAD, BOS, EOS, USER, ASSISTANT = 0, 1, 4, 2, 3
TOPICS = (5, 6, 7, 8)
RESP_IDS = tuple(range(8, 16))
TARGET = {t: 8 + (t - 5) for t in TOPICS}
DECOY = {t: 12 + (t - 5) for t in TOPICS}
VOCAB = 16
PROMPT_LEN = 4
SEEDS = (0, 1, 2)
TEMPLATE = ChatTemplate()


def make_model(seed):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2,
                      d_ff=128, max_seq_len=32, pos_encoding="rope",
                      norm="rmsnorm", ffn="swiglu", bias=False)
    return TinyLM(cfg), cfg


def prompt_ids(topic):
    return [BOS, USER, int(topic), ASSISTANT]


def make_response(gen, topic, p_target=0.55, p_decoy=0.10, lo=3, hi=5):
    """行为策略：55% 目标词、10% 形近干扰词、其余从响应词表均匀采样。"""
    length = int(torch.randint(lo, hi + 1, (1,), generator=gen))
    out = []
    for _ in range(length):
        u = float(torch.rand(1, generator=gen))
        if u < p_target:
            out.append(TARGET[topic])
        elif u < p_target + p_decoy:
            out.append(DECOY[topic])
        else:
            out.append(RESP_IDS[int(torch.randint(0, len(RESP_IDS), (1,), generator=gen))])
    return out


def make_example(topic, ys):
    return TEMPLATE.render([("user", [int(topic)]), ("assistant", ys)])


def make_sft_data(seed, size):
    gen = torch.Generator().manual_seed(seed)
    data = []
    for _ in range(size):
        topic = TOPICS[int(torch.randint(0, len(TOPICS), (1,), generator=gen))]
        data.append(make_example(topic, make_response(gen, topic)))
    return data


def train_sft(model, data, seed, steps=400, lr=3e-3, batch_size=32):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(1000 + seed)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(data), (batch_size,), generator=gen)
        batch = sft.collate_examples([data[int(i)] for i in idx])
        out = model(batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    return model


def biased_reward(topic, ys, noise=0.0, rng=None):
    """标注者的真实偏好：目标词 + 形近词 + 长度，后两项是偏置。"""
    correct = sum(y == TARGET[topic] for y in ys) / len(ys)
    decoy = sum(y == DECOY[topic] for y in ys) / len(ys)
    length_bonus = 0.35 * (len(ys) - 3) / 3.0
    noise = rng.gauss(0.0, noise) if (noise and rng) else 0.0
    return correct + 0.30 * decoy + length_bonus + noise


def debiased_reward(topic, ys, noise=0.0, rng=None):
    """去掉长度与形近词偏置，只看目标词比例。"""
    correct = sum(y == TARGET[topic] for y in ys) / len(ys)
    noise = rng.gauss(0.0, noise) if (noise and rng) else 0.0
    return correct + noise


def make_pairs(seed, size, reward_fn, noise=0.03):
    gen = torch.Generator().manual_seed(seed + 77)
    rng = random.Random(seed + 77)
    pairs = []
    for _ in range(size):
        topic = TOPICS[int(torch.randint(0, len(TOPICS), (1,), generator=gen))]
        y1 = make_response(gen, topic)
        y2 = make_response(gen, topic)
        r1 = reward_fn(topic, y1, noise, rng)
        r2 = reward_fn(topic, y2, noise, rng)
        if abs(r1 - r2) < 1e-6:
            continue
        win, lose = (y1, y2) if r1 > r2 else (y2, y1)
        pairs.append({"topic": topic,
                      "win": make_example(topic, win)["input_ids"],
                      "lose": make_example(topic, lose)["input_ids"]})
    return pairs


def pad_batch(rows, pad=PAD, prompt_len=PROMPT_LEN):
    length = max(r.numel() for r in rows)
    ids = torch.full((len(rows), length), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), length), dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, :r.numel()] = r
        mask[i, :r.numel()] = True
    positions = torch.arange(length)[None, :]
    completion = mask & (positions >= prompt_len)
    return ids, mask, completion


def pair_tensors(batch):
    win_ids, win_mask, win_comp = pad_batch([b["win"] for b in batch])
    lose_ids, lose_mask, lose_comp = pad_batch([b["lose"] for b in batch])
    return win_ids, win_mask, win_comp, lose_ids, lose_mask, lose_comp


@torch.no_grad()
def pair_stats(policy, ref, pairs, beta):
    win_ids, win_mask, win_comp, lose_ids, lose_mask, lose_comp = pair_tensors(pairs)
    pc = D.completion_logprobs(policy, win_ids, win_mask, win_comp)
    pr = D.completion_logprobs(policy, lose_ids, lose_mask, lose_comp)
    rc = D.completion_logprobs(ref, win_ids, win_mask, win_comp)
    rr = D.completion_logprobs(ref, lose_ids, lose_mask, lose_comp)
    margin = beta * D.log_ratio_margin(pc, pr, rc, rr)
    return float(D.preference_accuracy(margin)), float(margin.mean())


def train_dpo(sft_state, cfg, pairs, dcfg, seed, steps=300, log_every=0):
    policy = TinyLM(cfg)
    policy.load_state_dict(sft_state)
    ref = TinyLM(cfg)
    ref.load_state_dict(sft_state)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    opt = torch.optim.AdamW(policy.parameters(), lr=dcfg.lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(4000 + seed)
    history = []
    done = 0
    for _ in range(dcfg.epochs):
        for _ in range(steps):
            idx = torch.randint(0, len(pairs), (dcfg.batch_size,), generator=gen)
            batch_data = [pairs[int(i)] for i in idx]
            win_ids, win_mask, win_comp, lose_ids, lose_mask, lose_comp = \
                pair_tensors(batch_data)
            pc = D.completion_logprobs(policy, win_ids, win_mask, win_comp,
                                       dcfg.length_normalize)
            pr = D.completion_logprobs(policy, lose_ids, lose_mask, lose_comp,
                                       dcfg.length_normalize)
            with torch.no_grad():
                rc = D.completion_logprobs(ref, win_ids, win_mask, win_comp,
                                           dcfg.length_normalize)
                rr = D.completion_logprobs(ref, lose_ids, lose_mask, lose_comp,
                                           dcfg.length_normalize)
            loss = D.dpo_loss(pc, pr, rc, rr, beta=dcfg.beta,
                              label_smoothing=dcfg.label_smoothing,
                              loss_type=dcfg.loss_type)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), dcfg.max_grad_norm)
            opt.step()
            done += 1
            if log_every and done % log_every == 0:
                history.append((done, float(loss.detach())))
    return policy, ref, history


@torch.no_grad()
def sample_for_topics(policy, topics, cfg, generator, batch=256):
    policy.eval()
    rows = []
    for start in range(0, len(topics), batch):
        chunk = topics[start:start + batch]
        prompts = torch.tensor([prompt_ids(t) for t in chunk], dtype=torch.long)
        rows.append(P.sample_responses(policy, prompts, cfg, generator))
    return torch.cat(rows)


@torch.no_grad()
def evaluate(policy, ref, seed, n=256):
    topics = [TOPICS[i % len(TOPICS)] for i in range(n)]
    gen = torch.Generator().manual_seed(5000 + seed)
    seqs = sample_for_topics(policy, topics, P.RLConfig(), gen)
    attn = seqs != PAD
    mask = P.response_mask(seqs, PROMPT_LEN, PAD)
    last = attn.long().sum(-1) - 1
    content = mask.clone()
    content[torch.arange(seqs.size(0)), last] = False
    target = torch.tensor([TARGET[int(t)] for t in topics])
    decoy = torch.tensor([DECOY[int(t)] for t in topics])
    denom = content.sum(1).clamp_min(1)
    true = ((seqs == target[:, None]) & content).sum(1) / denom
    fake = ((seqs == decoy[:, None]) & content).sum(1) / denom
    length = content.sum(1).float()
    logp = P.token_logprobs(policy, seqs, attn)
    ref_logp = P.token_logprobs(ref, seqs, attn)
    kl = P.kl_penalty(logp, ref_logp, mask)
    return {"true": true.mean().item(), "fake": fake.mean().item(),
            "len": length.mean().item(), "kl": float(kl)}


def fmt(values):
    t = torch.tensor(values, dtype=torch.float64)
    return f"{t.mean():.3f}±{t.std(unbiased=False):.3f}"


def main():
    t0 = time.time()
    keys = ("sft", "b01", "b05", "b20", "ipo", "deb05")
    rows = {k: [] for k in keys}
    pair_rows = {k: [] for k in keys}
    deb_rows = []
    traj = None
    for seed in SEEDS:
        model, cfg = make_model(seed)
        train_sft(model, make_sft_data(seed, 512), seed)
        sft_state = copy.deepcopy(model.state_dict())

        train_pairs = make_pairs(seed + 100, 2000, biased_reward)
        debiased_pairs = make_pairs(seed + 100, 2000, debiased_reward)
        eval_pairs = make_pairs(seed + 900, 400, biased_reward)
        eval_debiased = make_pairs(seed + 900, 400, debiased_reward)

        ref = TinyLM(cfg)
        ref.load_state_dict(sft_state)
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
        rows["sft"].append(evaluate(model, ref, seed))

        configs = (
            ("b01", train_pairs, D.DPOConfig(beta=0.10, lr=5e-4, batch_size=32)),
            ("b05", train_pairs, D.DPOConfig(beta=0.50, lr=5e-4, batch_size=32)),
            ("b20", train_pairs, D.DPOConfig(beta=2.00, lr=5e-4, batch_size=32)),
            ("ipo", train_pairs, D.DPOConfig(beta=0.50, lr=5e-4, batch_size=32,
                                             loss_type="ipo")),
            ("deb05", debiased_pairs, D.DPOConfig(beta=0.50, lr=5e-4, batch_size=32)),
        )
        line = f"seed {seed}: SFT 正确 {rows['sft'][-1]['true']:.3f}"
        for key, pairs, dcfg in configs:
            policy, ref2, history = train_dpo(
                sft_state, cfg, pairs, dcfg, seed,
                log_every=50 if (seed == 0 and key == "b01") else 0)
            result = evaluate(policy, ref2, seed)
            rows[key].append(result)
            pair_rows[key].append(pair_stats(policy, ref2, eval_pairs, dcfg.beta))
            if key == "deb05":
                deb_rows.append(pair_stats(policy, ref2, eval_debiased, dcfg.beta))
            line += (f" | {key} 正确 {result['true']:.3f} 长度 {result['len']:.2f}")
            if history:
                traj = history
        print(line)

    print("\nDPO 训练轨迹（seed 0，偏置偏好，β=0.10，每 50 步）")
    print("步数   批损失")
    for step, loss in traj:
        print(f"{step:>3d}   {loss:.4f}")

    print("\n方法                 正确率              干扰词              长度               逐 Token KL")
    labels = (("sft", "SFT 策略"), ("b01", "DPO β=0.10"),
              ("b05", "DPO β=0.50"), ("b20", "DPO β=2.00"),
              ("ipo", "IPO β=0.50"), ("deb05", "DPO 去偏 β=0.50"))
    for key, name in labels:
        r = rows[key]
        print(f"{name:<18}   {fmt([x['true'] for x in r])}   "
              f"{fmt([x['fake'] for x in r])}   {fmt([x['len'] for x in r])}   "
              f"{fmt([x['kl'] for x in r])}")

    print("\n留出偏好对：隐式奖励排序准确率 / 平均间隔（偏置标签）")
    for key, name in labels[1:]:
        acc = pair_rows[key]
        print(f"{name:<18}   准确率 {fmt([a for a, _ in acc])}   "
              f"间隔 {fmt([m for _, m in acc])}")
    print("\n去偏标签上的隐式奖励（DPO 去偏训练）")
    print(f"{'DPO 去偏':<18}   准确率 {fmt([a for a, _ in deb_rows])}   "
          f"间隔 {fmt([m for _, m in deb_rows])}")
    print(f"\n总耗时 {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
```

`tests/test_dpo.py`：

```python
"""第 17 章测试：DPO 损失、序列对数概率与隐式奖励。"""
import math

import pytest
import torch

from ch08 import ModelConfig, TinyLM
from tinylm.posttrain import dpo as D


def tiny_cfg():
    return ModelConfig(vocab_size=16, d_model=32, n_layers=2, n_heads=2,
                       d_ff=128, max_seq_len=32, pos_encoding="rope",
                       norm="rmsnorm", ffn="swiglu", bias=False)


def model(seed=0):
    torch.manual_seed(seed)
    return TinyLM(tiny_cfg())


def test_dpo_loss_zero_margin_is_log_two():
    z = torch.zeros(4)
    assert torch.allclose(D.dpo_loss(z, z, z, z, beta=0.1),
                          torch.tensor(math.log(2.0)), atol=1e-6)


def test_dpo_loss_decreases_with_margin():
    margins = torch.tensor([0.0, 0.5, 1.0, 2.0])
    zeros = torch.zeros(4)
    losses = [float(D.dpo_loss(margins[:i + 1], zeros[:i + 1], zeros[:i + 1],
                               zeros[:i + 1], beta=1.0))
              for i in range(4)]
    assert losses[0] > losses[1] > losses[2] > losses[3]


def test_dpo_loss_ignores_common_shift():
    pc, pr = torch.tensor([0.3, -1.0]), torch.tensor([0.1, 0.5])
    rc, rr = torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])
    a = D.dpo_loss(pc, pr, rc, rr)
    b = D.dpo_loss(pc + 3.0, pr + 3.0, rc, rr)
    assert torch.allclose(a, b)


def test_label_smoothing_increases_loss_for_confident_pair():
    pc, pr, rc, rr = (torch.tensor([2.0]), torch.tensor([0.0]),
                      torch.tensor([0.0]), torch.tensor([0.0]))
    hard = D.dpo_loss(pc, pr, rc, rr, beta=1.0)
    smooth = D.dpo_loss(pc, pr, rc, rr, beta=1.0, label_smoothing=0.1)
    assert smooth > hard


def test_ipo_has_zero_at_target_margin():
    beta = 0.5
    h = 1.0 / (2.0 * beta)
    pc = torch.tensor([h])
    zeros = torch.zeros(1)
    assert torch.allclose(D.dpo_loss(pc, zeros, zeros, zeros, beta=beta,
                                     loss_type="ipo"), torch.tensor(0.0))


def test_implicit_reward_scale():
    logp = torch.tensor([0.2, -0.4])
    ref = torch.tensor([0.1, 0.0])
    reward = D.implicit_reward(logp, ref, beta=0.5)
    assert torch.allclose(reward, torch.tensor([0.05, -0.2]))


def test_dpo_gradient_pushes_chosen_up_and_rejected_down():
    pc = torch.tensor([0.5], requires_grad=True)
    pr = torch.tensor([0.0], requires_grad=True)
    zero = torch.zeros(1)
    loss = D.dpo_loss(pc, pr, zero, zero, beta=0.1)
    loss.backward()
    assert pc.grad < 0 and pr.grad > 0
    assert torch.allclose(pc.grad, -pr.grad)


def test_completion_logprobs_matches_manual():
    m = model()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    attn = torch.ones(1, 5, dtype=torch.bool)
    completion = torch.tensor([[False, False, True, True, True]])
    got = D.completion_logprobs(m, ids, attn, completion)
    with torch.no_grad():
        logits = m(ids, attention_mask=attn).logits[:, :-1]
        logp = torch.log_softmax(logits, dim=-1)
        manual = sum(logp[0, t, ids[0, t + 1]] for t in (1, 2, 3))
    assert torch.allclose(got[0], manual)


def test_completion_logprobs_excludes_prompt_and_padding():
    m = model()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    attn = torch.ones(1, 5, dtype=torch.bool)
    completion = torch.tensor([[False, False, True, True, True]])
    base = D.completion_logprobs(m, ids, attn, completion)
    padded_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0]])
    padded_attn = torch.tensor([[True] * 5 + [False] * 2])
    padded_completion = torch.tensor([[False, False, True, True, True, False, False]])
    padded = D.completion_logprobs(m, padded_ids, padded_attn, padded_completion)
    assert torch.allclose(base, padded)


def test_length_normalize_divides_by_completion_length():
    m = model()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    attn = torch.ones(1, 5, dtype=torch.bool)
    completion = torch.tensor([[False, False, True, True, True]])
    total = D.completion_logprobs(m, ids, attn, completion)
    mean = D.completion_logprobs(m, ids, attn, completion, length_normalize=True)
    assert torch.allclose(mean, total / 3.0)


def test_preference_accuracy_ties_fail():
    margin = torch.tensor([1.0, -1.0, 0.0, 2.0])
    assert float(D.preference_accuracy(margin)) == 0.5


def test_dpo_config_validation():
    with pytest.raises(AssertionError):
        D.DPOConfig(beta=0.0)
    with pytest.raises(AssertionError):
        D.DPOConfig(loss_type="orpo")
    with pytest.raises(AssertionError):
        D.DPOConfig(label_smoothing=0.5)
    with pytest.raises(AssertionError):
        D.DPOConfig(epochs=0)
```

### 17.5.3 运行结果

```text
seed 0: SFT 正确 0.572 | b01 正确 0.857 长度 4.09 | b05 正确 0.945 长度 4.62 | b20 正确 0.926 长度 4.72 | ipo 正确 0.691 长度 4.11 | deb05 正确 0.982 长度 4.42
seed 1: SFT 正确 0.607 | b01 正确 0.739 长度 4.70 | b05 正确 0.958 长度 4.88 | b20 正确 0.936 长度 4.64 | ipo 正确 0.723 长度 3.92 | deb05 正确 0.978 长度 4.19
seed 2: SFT 正确 0.565 | b01 正确 0.977 长度 4.53 | b05 正确 0.966 长度 4.81 | b20 正确 0.927 长度 4.70 | ipo 正确 0.674 长度 4.23 | deb05 正确 0.980 长度 4.46

DPO 训练轨迹（seed 0，偏置偏好，β=0.10，每 50 步）
步数   批损失
 50   0.5982
100   0.4608
150   0.4052
200   0.4702
250   0.3571
300   0.3535

方法                 正确率              干扰词              长度               逐 Token KL
SFT 策略               0.581±0.019   0.143±0.007   3.977±0.118   0.000±0.000
DPO β=0.10           0.858±0.097   0.007±0.003   4.438±0.257   1.221±0.638
DPO β=0.50           0.957±0.009   0.016±0.002   4.768±0.108   0.574±0.049
DPO β=2.00           0.929±0.004   0.034±0.003   4.686±0.033   0.359±0.032
IPO β=0.50           0.696±0.020   0.115±0.009   4.086±0.127   0.027±0.005
DPO 去偏 β=0.50        0.980±0.001   0.004±0.001   4.357±0.118   0.558±0.072

留出偏好对：隐式奖励排序准确率 / 平均间隔（偏置标签）
DPO β=0.10           准确率 0.849±0.011   间隔 1.044±0.054
DPO β=0.50           准确率 0.895±0.007   间隔 3.018±0.092
DPO β=2.00           准确率 0.912±0.019   间隔 6.540±0.172
IPO β=0.50           准确率 0.873±0.024   间隔 0.299±0.027
DPO 去偏 β=0.50        准确率 0.839±0.008   间隔 2.729±0.067

去偏标签上的隐式奖励（DPO 去偏训练）
DPO 去偏               准确率 0.923±0.009   间隔 3.045±0.092

总耗时 102.0 s
```

### 17.5.4 结果解读

先看主表。SFT 策略的目标命中率是 0.581±0.019，干扰词 0.143，长度 3.977。DPO（偏置标签，β=0.50）把目标命中率提高到 0.957±0.009，干扰词降到 0.016；去偏标签的 DPO（β=0.50）进一步把目标命中率提高到 0.980±0.001，干扰词降到 0.004。这说明在同样的模型、同样的训练步数下，**标签质量比损失函数更重要**：去偏数据的收益（+2.3 个百分点、干扰词再降 1.2 个百分点）超过了多数超参数调整。

再看长度。偏置标签的 DPO 把平均内容长度从 3.977 推到 4.768，接近评测上限 5；去偏标签的 DPO 只到 4.357。两者相差 0.411 个 Token，而评测的最大可能长度只有 2 个 Token 的余量。偏好标签里“长度每多一档加 0.117”的微小权重，被 DPO 完整地学走，并放大成了策略层面的长度偏置。这正是第 16 章奖励作弊的离线版本：没有奖励模型可以被“钻空子”，但标签本身就是那个空子。

β 的行为也值得注意。β=0.10 的平均正确率只有 0.858±0.097，种子间方差很大，KL 高达 1.221±0.638；β=0.50 的正确率 0.957±0.009、KL 0.574±0.049；β=2.00 的正确率 0.929±0.004、KL 0.359±0.032。KL 没有随 β 单调变化：β=2.00 的 KL 反而低于 β=0.50。原因在 17.3.4 已经分析：β 同时改变 logit 缩放、梯度尺度和饱和速度。β=0.10 时梯度太小，300 步内不同种子的收敛程度差异很大；β=2.00 时容易分的对迅速饱和，更新集中到少数难分对，整体偏离反而更小。

IPO 的结果更保守：目标命中率 0.696±0.020，干扰词 0.115，长度 4.086，KL 仅 0.027±0.005。它确实避免了过度偏离参考模型，但也几乎没有把偏好学进去。目标间隔 $$1/(2\beta)=1$$ 对这个小模型来说太大，大部分偏好对在 300 步内达不到这个间隔，损失持续把策略往“稍微好一点”的方向推，而不是“明确分开”。

一个反直觉的细节是：偏置 DPO 与去偏 DPO 在 β=0.50 时的 KL 很接近（0.574 对 0.558），但长度与任务正确率差别很大。KL 只度量“偏离参考模型多远”，不度量“往哪个方向偏”。去偏 DPO 用几乎相同的偏离预算，把概率质量放到了目标词上；偏置 DPO 则把一部分预算花在了长度上。这说明 KL 是预算指标，不是质量指标；报告 KL 时必须同时报告任务指标与长度，否则“KL 更小”可能只是“学得更少”。

训练轨迹也解释了 β=0.10 的种子方差。偏置 DPO（β=0.10，seed 0）的批损失从 0.598 降到 0.353，但中途在 0.405 与 0.470 之间来回波动；这是小批量、低梯度尺度下的正常噪声。三个种子的最终正确率分别是 0.857、0.739、0.977，差异远大于 β=0.50 的 0.945、0.958、0.966。低 β 并不“更保守”，它只是把有效步长压小，让优化结果更依赖随机种子与步数。

留出偏好对上的数字揭示了另一个问题。偏置标签下，DPO β=2.00 的隐式奖励排序准确率最高（0.912±0.019），平均间隔 6.540；但它的真实目标命中率（0.929）低于去偏 DPO（0.980）。去偏 DPO 在偏置评测集上的排序准确率只有 0.839，低于偏置 DPO 的 0.895，却在真实任务上更好。这说明**在偏好对上的准确率不等于任务质量**：去偏模型故意不拟合“长度 = 好”的标签，因此在偏置评测集上看起来更差，但它学到的偏好更接近任务目标。模型可以非常擅长复现标注者的排序，同时把标注者的长度偏置一起学走。评测必须同时看任务指标、长度、KL 和偏好准确率，缺一不可。

最后是边界。这个实验只有 2 层、32 维、16 词表，回答长度上限 5，偏好由手工奖励生成。它能验证机制和诊断方法，不能给出真实 LLM 的 β、学习率或数据规模。真实系统里，长度偏置可能表现为冗长、谄媚、过度格式化和过度拒答；测量方法相同：构造一个与任务目标无关的反事实维度，看策略是否沿着它偏移。

### 17.5.5 复现步骤

在装有 PyTorch 的 CPU 环境里，把本章三个代码块分别保存为 `tinylm/posttrain/dpo.py`、`examples/ch17_dpo.py`、`tests/test_dpo.py`，并复用第 16 章的 `ch08.py` 与 `tinylm` 包：

```bash
export PYTHONPATH=.
python3 -m pytest -q tests/test_dpo.py
python3 examples/ch17_dpo.py
```

测试应输出 12 项全部通过；主程序约 1–2 分钟，除耗时外逐字可复现。所有随机源固定种子：模型种子、SFT 数据种子、偏好对种子、DPO 批采样种子、评测采样种子。

## 17.6 工业实践

DPO 在工业界的流行，主要因为它把“对齐”从强化学习工程变成了监督学习工程。下面三个公开案例能说明它实际怎么用。

**Zephyr-7B**（Tunstall 等，2023）是 DPO 早期最有影响力的开源复现之一。它先在同一基座上做 SFT（UltraChat），再用 UltraFeedback 的成对偏好做 DPO。公开配置为 β=0.1、学习率 $$5\times10^{-7}$$、训练 1 个 epoch、批大小 32 左右；DPO 之后没有再做 PPO。报告称其在 MT-Bench 上达到 7.34 分，超过同规模的部分 RLHF 模型。这个案例的关键不是分数，而是流程：一份 SFT 模型、一份偏好数据、一次 DPO，就能得到可用的对齐模型。

**Tulu 2**（Ivison 等，2023）把 DPO 放在 SFT 之后，用 UltraFeedback 的偏好数据做第二轮对齐，公开配置同样是 β=0.1、学习率 $$5\times10^{-7}$$ 量级，训练 1–2 个 epoch。Tulu 2 的报告强调数据混合与评测口径：DPO 带来的提升在多数任务上为正，但在某些知识密集型任务上会回退，需要靠数据配比与 SFT 回放来抵消。这与本章“标签质量优先”的结论一致：偏好数据覆盖不到的能力，DPO 不会自动补上。

**Llama 3**（Meta，2024）的后训练流程是 SFT → 拒绝采样 → DPO。它先用 SFT 和拒绝采样得到高质量回答，再用偏好数据做 DPO；论文没有公开完整的 β 与学习率组合，但明确把 DPO 作为“在 SFT 之后进一步对齐人类偏好”的阶段，而不是替代 SFT。这个顺序很重要：DPO 的目标是“在参考模型附近调整偏好”，参考模型本身必须已经具备基本能力。若直接从基座做 DPO，参考模型的对数概率没有意义，策略会在低质量区域里做对比。

数据规模上，公开案例给出的是“万到百万”量级的偏好比较：Zephyr 使用 UltraFeedback 约 6 万条提示衍生的成对数据，Tulu 2 在同一量级的数据上做 1–2 个 epoch，Llama 3 的技术报告则提到百万量级的人类偏好比较。这个量级比 SFT 数据小得多，说明 DPO 并不是“数据越多越好”的堆量游戏；偏好数据的覆盖、去偏与一致性，比单纯增加对数更重要。一个常见的工业做法是：先用强模型或奖励模型对大量候选做初筛，再对少量高价值、高分歧的样本做人工标注，把标注预算集中在“模型还分不清”的偏好对上。

工程实现上有几个共性问题。

第一，参考模型的对数概率可以预计算。参考模型冻结、偏好数据固定，因此“参考模型在每条回答上的对数概率”只需算一次，之后每个 epoch 复用。这样每步只剩策略的两次前向，显存和算力都接近普通 SFT。若做迭代 DPO，新采样的回答需要重新算参考对数概率，但旧数据仍可缓存。

第二，参考模型与策略可以共享基座。第 15 章的 LoRA 在这里非常合适：一份冻结基座加两份适配器，一份给策略、一份给参考，或者参考直接用 SFT 权重。全参数 DPO 需要两份完整权重；LoRA DPO 的静态状态接近一份基座加两份小适配器，显存压力小得多。

第三，偏好对的构造比损失更重要。工业数据通常要做长度匹配、位置随机化、标注者校准与去污染。长度匹配是指让赢家和输家的长度分布接近，避免模型把“长”当成“好”；位置随机化是指随机交换左右顺序，抵消位置偏置；标注者校准是指按标注者一致性加权，避免某个标注者的极端偏好主导训练。这些步骤不改变 DPO 的公式，但对最终策略的影响往往大于 β。

第四，评测要分层。至少要报告四类指标：留出偏好对上的排序准确率、任务指标（正确率、通过率、人工偏好）、长度与格式统计、相对参考模型的 KL。只看损失或只看偏好准确率，都会漏掉本章实验里“偏好准确率上升、任务质量下降”的情况。

第五，在线与离线的选择。离线 DPO 只能利用固定偏好数据；迭代 DPO 会让当前策略生成新回答，再用人工或 AI 重新标注，然后继续训练。迭代 DPO 保留了部分在线探索能力，但标注成本高，且容易陷入“模型自己生成、自己偏好”的循环。工业上常见的折中是：用强模型或奖励模型做初筛，再用人标注少量高价值样本。

## 17.7 论文导读

1. **Rafailov, Sharma, Mitchell, Ermon, Manning, Finn. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. 2023. arXiv:2305.18290.** 读第 3 节看从 RLHF 目标到 DPO 损失的完整推导，第 4 节看与 PPO 的实验对照，第 5 节看理论分析。本章 (17.1)–(17.5) 主要依据这篇。

2. **Azar, Rowland, Piot, Guo, Calandriello, Valko, Munos. A General Theoretical Paradigm to Understand Learning from Human Preferences. 2023. arXiv:2310.12036.** 读第 3–4 节看 IPO 的目标间隔与 Ψ-PO 框架。它解释了为什么 DPO 在确定性偏好下会过度拉大间隔。

3. **Ethayarajh, Xu, Muennighoff, Jurafsky, Kiela. KTO: Model Alignment as Prospect Theoretic Optimization. 2024. arXiv:2402.01306.** 读第 2–3 节看二元反馈下的价值函数，第 4 节看与 DPO 的数据效率对照。适合只有“好/不好”标签、没有成对数据的场景。

4. **Meng, Xia, Chen. SimPO: Simple Preference Optimization with a Reference-Free Reward. 2024. arXiv:2405.14734.** 读第 3 节看长度归一化与目标间隔的设计，第 4 节看无参考模型的实验。它把本章 (17.6) 的方向推到了极致。

5. **Hong, Lee, Thorne. ORPO: Monolithic Preference Optimization without Reference Model. 2024. arXiv:2403.07691.** 读第 3 节看 odds ratio 如何并入 SFT 损失，第 4 节看单阶段训练的结果。它代表“把对齐压进 SFT”的路线。

延伸：Tunstall 等（2023，arXiv:2310.16944）的 Zephyr 报告给出了 DPO 的完整开源配方；Ivison 等（2023，arXiv:2310.12379）的 Tulu 2 讨论了数据混合与 DPO 的交互。读这两篇时，把超参数表与本章 17.5 的实验设计并排看，会更容易判断哪些结论能迁移到自己的数据。特别要核对三件事：参考模型是否就是产生偏好数据时的 SFT 模型；DPO 数据与 SFT 数据之间是否存在重叠；报告的长度是“输入加输出”还是只算输出。这三点不统一，表格里的数字就不能横向比较。

## 17.8 常见误区

**误区一：DPO 不需要参考模型，或者参考模型越强越好。** DPO 确实不需要奖励模型，但必须有冻结的参考模型。参考模型定义了隐式奖励的零点；没有它，策略的对数概率没有基准，损失会退化成无锚点的成对排序。SimPO 是真正去掉参考模型的变体，代价是更容易漂移。反过来，参考模型也不是越强越好：它的作用是提供与偏好数据同源的零点。如果参考模型与产生数据的策略不一致，例如用新版权重给旧数据算对数概率，隐式奖励里就会混入“版本差异”，模型学到的是两个模型之间的差异，而不是人类偏好。正确做法是让参考模型与 SFT 模型逐位一致；确实要更换时，应重新生成偏好数据或重新校准基线。

**误区二：DPO 与 RLHF 完全等价。** 只有在 Bradley–Terry 假设成立、偏好数据覆盖充分、优化到全局最优时，两者的最优解才一致。DPO 没有在线探索，无法发现数据之外的高分行为；数据没有覆盖的偏好，它学不到。把 DPO 当作“更便宜的 PPO”可以，当作“无损替代”不行。

**误区三：β 越小越保守，长度归一化只是实现细节。** β 同时缩放 logit 和梯度。固定学习率时，减小 β 会让梯度变小、收敛变慢、种子方差变大；增大 β 会让容易分的对迅速饱和。本章实验里 β=2.0 的 KL 低于 β=0.5，就是反例。调 β 必须同时报告学习率、训练步数与实际 KL。长度归一化也不是纯实现细节：它改变隐式奖励的单位与长度激励。不做归一化时，长回答的对数概率和更大、梯度也更大；做了归一化，β 的含义变成“每 Token 的温度”。跨实现比较 β 之前，先确认归一化方式。

**误区四：DPO 不会奖励作弊。** DPO 没有奖励模型，但偏好标签就是奖励。标注者偏好长回答，DPO 就学长度；标注者把形近词当正确答案，DPO 就学形近词。本章实验里，偏置标签让长度从 3.977 推到 4.768，而去偏标签只到 4.357。奖励作弊从“钻奖励模型的空子”变成了“钻标签的空子”。

**误区五：损失下降就说明模型变好。** DPO 损失可以通过三种方式下降：提高赢家的对数概率、降低输家的对数概率、或者两者同时发生但相对间隔变大。第一种会让模型更像数据里的赢家，第二种会让模型远离输家——后者在极端情况下会把策略推到一个“什么都不像”的区域。必须同时看任务指标、长度、KL 和生成样本。

## 17.9 本章小结

- DPO 用隐式奖励与对数比值的闭式解绕开奖励模型和 PPO，只需两个模型、一次监督式训练，优化更稳。
- IPO、KTO、SimPO、ORPO 各自替换 DPO 中的一项约束：参考模型、成对结构或长度归一化。
- β 与学习率、饱和速度耦合；工程上固定 $$\beta\cdot\eta$$ 做扫描，并同时报告 KL 与任务指标。
- 偏好数据的偏置会被原样学走：偏置标签与去偏标签的长度 4.768/4.357、任务正确率 0.957/0.980。
- DPO 的甜点区是已有质量尚可的成对偏好数据、算力有限、能接受不探索数据之外行为。

{% hint style="success" %}
**本章的取舍**：DPO 用“离线成对监督”换掉了 PPO 的“在线强化学习”。它放弃了奖励模型与在线探索，换来两个模型、一次监督式训练和更稳定的优化；代价是只能利用已有偏好数据，标签里的偏置会被原样学走。本章实验里，偏置标签与去偏标签的差距（长度 4.768 对 4.357，任务正确率 0.957 对 0.980）比多数超参数调整都大，这不是 DPO 的缺陷，而是“离线学习”的必然结果。

β 是第二个取舍。小 β 给策略更大的自由度，能更充分地利用偏好信号，但也更容易过拟合标签噪声、在种子间不稳定；大 β 把策略拴在参考模型附近，KL 小、行为保守，但可能学不动。本章的 β=0.10/0.50/2.00 三档没有给出单调前沿，反而说明 β 与学习率、饱和速度耦合。工程上更可靠的做法是固定 $$\beta\cdot\eta$$ 做扫描，并同时报告 KL 与任务指标。

参考模型是第三个取舍。它提供了锚点，也带来了额外的一份权重与前向。LoRA DPO 用“一份基座 + 两份适配器”缓解显存；预计算参考对数概率缓解算力；SimPO 直接去掉参考模型，用长度归一化与目标间隔替代。每一次简化都放弃了某种约束，选型时要问“我失去的那份约束，当前数据能否替代”。

什么情况下不该用 DPO？如果奖励可以自动验证（数学答案、代码单测、工具调用结果），第 18 章的 RLVR 与 GRPO 能提供比离线偏好更直接的信号，而且不需要人工标注成对数据。如果系统需要持续在线探索、或者已有可靠的奖励模型可以复用，PPO 的闭环仍然更强。如果只是想在推理时提分、不想改模型，Best-of-N 或拒绝采样更便宜。DPO 的甜点区是：有质量尚可的成对偏好数据、算力有限、希望一次监督式训练完成对齐、且能接受“不探索数据之外行为”的边界。

最后是算法与数据的关系。DPO 的论文标题说“你的语言模型 secretly 是一个奖励模型”，但本章的实验说明：模型是不是奖励模型，取决于你喂给它什么偏好。损失函数决定了优化的方向，数据决定了方向的终点。工业上把大量精力放在偏好数据的去偏、长度匹配、标注者校准与去污染上，原因就在这里。

{% endhint %}

## 17.10 习题

**推导题**：从第 16 章式 (16.4) 出发，推导 (17.1)，并说明配分函数为什么在同一提示的成对比较中抵消，使 (17.3) 不必计算它。若两条回答来自不同提示，这个抵消还成立吗？

<details>
<summary>参考答案</summary>

由 (16.4)，

$$
\pi^\star(y\mid x)=\frac{1}{Z(x)}\pi_{\rm ref}(y\mid x)\exp\!\left(\frac{r(x,y)}{\beta}\right)
$$

两边取对数：

$$
\log\pi^\star(y\mid x)
=\log\pi_{\rm ref}(y\mid x)+\frac{r(x,y)}{\beta}-\log Z(x)
$$

移项并乘以 β，即得 (17.1)。对同一提示 $$x$$ 的两条回答 $$y_w,y_l$$，奖励差为

$$
r(x,y_w)-r(x,y_l)
=\beta\left[
\log\frac{\pi^\star(y_w\mid x)}{\pi_{\rm ref}(y_w\mid x)}
-\log\frac{\pi^\star(y_l\mid x)}{\pi_{\rm ref}(y_l\mid x)}
\right]
+\beta\big(\log Z(x)-\log Z(x)\big)
$$

后一项严格为零，因此不需要计算 $$Z(x)$$。若两条回答来自不同提示 $$x_w\ne x_l$$，两个配分函数一般不相等，$$\beta\log Z(x_w)-\beta\log Z(x_l)$$ 不会抵消，隐式奖励的绝对尺度就重新变得不可识别。这也是 DPO 数据必须按提示成对组织的原因。

</details>

**编程题**：在 `dpo_loss` 中加入长度归一化选项（按回答 Token 数平均对数概率），在玩具实验上比较归一化前后的平均长度与 KL。归一化是否消除了长度偏置？如果没有，为什么？

**开放题**：设计一套诊断 DPO 标签偏置的实验。要求至少包含：长度匹配的偏好对、与任务目标无关的反事实维度、留出任务指标，以及在偏好准确率上升时识别“任务质量下降”的判据。讨论你的方案在真实人工标注数据上可能遇到的困难。
