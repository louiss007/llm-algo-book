# 第 18 章 可验证奖励强化学习与推理模型

## 本章导读

数学题的答案对不对，程序比人更清楚。既然一条规则就能给出 0 或 1 的奖励，为什么还要先训练一个奖励模型，再让模型去猜人类偏好？把奖励换成“答案匹配、单元测试、工具返回”，强化学习的目标突然变得清晰，却也暴露出一个新问题：一组回答全对或全错时，模型该向谁学习？

## 难度标签

⭐⭐⭐⭐ 研究向

- 数学：策略梯度、基线、重要性采样、PPO 裁剪、KL 估计与有限样本偏差；
- 代码：PyTorch，理解第 16 章 PPO 的 rollout 与裁剪目标、第 17 章 DPO 的离线偏好训练；
- 系统：大规模 rollout 的吞吐、显存与推理引擎接口。

## 学习目标

1. 能说明可验证奖励（RLVR）与人类偏好奖励（RLHF）在数据、奖励和优化闭环上的差异，并判断一个任务是否适合 RLVR。
2. 能从带基线的策略梯度推导 GRPO 的组均值基线、优势标准化与裁剪目标，解释它为什么不需要价值网络。
3. 能分析组大小 $$G$$、采样温度、全对/全错组与长度归一化对梯度的影响，并写出零方差组的概率。
4. 能实现一个可运行的 GRPO 训练循环，在规则可验证的数学任务上复现“$$G=1$$ 不学习、$$G=8$$ 明显提升”的现象。
5. 能拆解 DeepSeek-R1、DAPO 等系统的训练流程，说明长思维链（Long CoT）涌现与过程奖励、结果奖励之间的取舍。

## 18.1 奖励为什么可以不学

第 16 章的 RLHF 用人类偏好训练奖励模型，再用 PPO 在奖励模型上优化策略。这条路线有一个根本的弱点：奖励模型只是人类判断的近似，策略会找到它的漏洞。回答变得更长、更自信、更符合奖励模型的表面特征，却不一定更正确。奖励作弊（reward hacking）不是训练 bug，而是“用一个可优化的代理去逼近不可优化的目标”时必然出现的现象。

可在数学、代码、工具调用这类任务里，正确性有独立于模型的判定方式。数学题有标准答案，代码有单元测试，工具调用有真实的返回结果。于是我们可以绕过奖励模型，直接定义

$$
r_i=\mathbb{1}\big[\operatorname{Verify}(q,o_i,y^\star)=1\big]
\qquad \text{(18.1)}
$$

其中 $$q$$ 是提示，$$o_i$$ 是策略采样出的第 $$i$$ 条回答，$$y^\star$$ 是标准答案或可执行的检查器。这个奖励只有 0 和 1 两个取值，不光滑、不稠密，但它不依赖人类标注者的即时判断，也不会被策略“说服”。这类设定称为可验证奖励强化学习（Reinforcement Learning with Verifiable Rewards，RLVR）。

RLVR 与 RLHF 的差别不只是“奖励从哪来”，而是整条数据链都变了。RLHF 需要成对或打分的偏好数据，偏好数据的规模受标注预算限制；RLVR 只需要题目，回答由当前策略在线生成，奖励由程序即时计算。于是训练从“拟合一份静态偏好”变成“不断出题、答题、判卷、改进”的闭环。第 17 章的 DPO 只能利用已有偏好对，无法主动发现数据之外的行为；RLVR 的在线采样恰好补上了这个缺口。

表 18-1：三类常见可验证奖励

| 任务 | 验证器 | 奖励形式 | 常见失效 |
| --- | --- | --- | --- |
| 数学 | 答案规范化后精确匹配、符号计算 | 0/1 | 等价形式未归一化、单位与精度差异 |
| 代码 | 编译、单元测试、运行结果 | 通过率或 0/1 | 测试覆盖不足、读取测试文件、超时 |
| 工具调用 | 真实环境返回、状态检查 | 0/1 或分步 | 缓存污染、权限差异、非确定性 |

但“可验证”不等于“正确”。验证器本身是代码，代码会有漏洞。数学答案的字符串匹配可能把 $$0.5$$ 和 $$\frac{1}{2}$$ 判成不同答案；单元测试可能只覆盖少数样例，策略可以针对测试过拟合；工具返回可能被环境污染。工业系统通常要做答案规范化、测试集隔离、沙箱执行与污染检测。验证器越接近任务的真实目标，RLVR 越可靠；如果验证器只是目标的代理，策略就会学会骗这个代理，而不是学会完成任务。验证器决定了奖励的上限，策略只会在验证器划定的范围内变强。

可验证奖励也有粒度问题。最常用的是结果奖励（outcome reward）：只看最终答案对错。另一种是过程奖励（process reward）：对推理的每一步打分。过程奖励更稠密，能缓解稀疏奖励，但需要逐步标注或训练过程奖励模型，成本高，而且“这一步是否合理”本身很难用规则定义。DeepSeek-R1 明确选择了结果奖励加格式奖励，没有采用过程奖励，原因正是过程奖励难以通用、容易引入新的奖励作弊。本章的主线因此是结果级 RLVR；过程奖励只在方案对比与论文导读中出现。

从系统角度看，RLVR 把瓶颈从“标注”移到了“采样”。一个提示要采 $$G$$ 条回答，每条回答可能长达数千 Token；训练吞吐取决于推理引擎能多快地把这些 Token 生成出来。第 21 章的 KV Cache、连续批处理与预填充/解码分离，在 RLVR 里不是可选的性能优化，而是决定训练能否跑起来的基础设施。第 15 章的 LoRA 则决定了在有限显存下能否同时放下策略、参考模型与优化器状态。

```mermaid
flowchart LR
    A[题目 q] --> B[策略 πθ 采样 G 条回答]
    B --> C[验证器：答案/单测/工具]
    C --> D[奖励 r_i ∈ {0,1}]
    D --> E[组内均值与标准差]
    E --> F[优势 Â_i]
    F --> G[裁剪代理目标]
    G --> H[更新策略]
    I[参考模型 π_ref] -.可选 KL.-> G
```

图 18-1：RLVR 的闭环。奖励由规则给出，基线由同一提示的组内样本给出，价值网络被完全去掉。

## 18.2 直觉：一场同题竞赛

把 GRPO 想成一场同题竞赛。对每道题，让 $$G$$ 个“选手”各答一次，然后统一判卷，得到 $$G$$ 个 0/1 分数。我们不关心绝对分数，只关心谁高于这场竞赛的平均分：高于平均的回答被鼓励，低于平均的被抑制。平均分同时充当了难度校准——难题大家都不对，平均分是 0，没有人被鼓励；简单题大家都对，平均分是 1，也没有人被鼓励。

这个类比解释了 GRPO 最重要的结构：它不需要价值网络（critic）。PPO 需要一个价值网络来预测“当前状态未来能拿多少奖励”，用预测值构造优势；GRPO 直接用同一提示下已经观测到的组内平均奖励作为基线。基线的作用是降低方差，不改变梯度的期望。组内平均是一个蒙特卡洛估计，虽然粗糙，但不需要额外训练一个网络，也不会引入价值网络的估计偏差。

组内样本在给定提示后是独立采样的，但它们共享同一个提示，因此组内均值只消除“题目难度”这一项共同波动，不会消除“这道题本身很难，所有回答都错”带来的整体偏差。跨组之间仍然存在题目差异，这也是为什么训练时要打乱提示、按难度分层，而不是让模型连续做同一类题。组内基线是“同题比较”，不是“跨题校准”；把两者混为一谈，会错误地期待它解决课程学习的问题。

它同时解释了 GRPO 的失效边界。如果 $$G=1$$，选手只能和自己比，平均分就是自己的分数，优势恒为零，参数一步都不会更新。如果一组回答全对或全错，组内没有差异，平均分等于每个人的分数，同样没有梯度。只有“有的对、有的错”的混合组，才提供学习信号。组大小、题目难度与采样温度，本质上都在调节“混合组出现的概率”。

为什么不用整批样本的均值作基线？因为不同题目的难度差异太大。一道简单题的正确率可能是 0.9，一道难题可能是 0.05；把两者的奖励放在一起减同一个全局均值，简单题的正确回答会被当成“理所当然”，难题的正确回答会被过度奖励。组内基线只比较同一道题的不同回答，天然消除了题目难度这个混杂因素。这也是 GRPO 的“相对”二字的含义：优势是相对于同题组的，而不是相对于全局的。

从统计角度看，组内均值是一个关于题目难度的蒙特卡洛估计。策略梯度估计的方差大致正比于 $$\mathbb{E}[(r-b)^2\|\nabla\log\pi\|^2]$$，当 $$b$$ 接近 $$\mathbb{E}[r]$$ 时方差最小。用 $$G$$ 个样本估计这个期望，基线本身的标准误是 $$\sigma_r/\sqrt{G}$$；$$G$$ 越大，基线越准，优势的噪声越小。但基线误差只是总方差的一部分，采样带来的奖励噪声仍然存在，所以增大 $$G$$ 的收益会递减。这也解释了为什么工业系统不会无限加大 $$G$$：从 8 增到 16 的方差改善，往往抵不上翻倍的 rollout 成本。

温度是另一个容易被忽视的旋钮。采样温度太低，$$G$$ 条回答几乎一样，组内方差接近零，优势也接近零；温度太高，回答质量下降，验证器给出的奖励变得稀疏而嘈杂。训练时通常用接近 1 的温度保持探索，评测时再用较低温度取贪心或低熵样本。采样时使用的温度必须与计算旧策略对数概率时的温度一致，否则重要性比的分母就不是真正的行为策略，裁剪目标会系统性偏移。

长思维链是这场竞赛的副产品。当奖励只看最终答案时，模型可以通过写更长的推理来提高答对概率；训练过程中回答长度往往先上升，再在奖励饱和后趋于稳定。长度增长本身不是目标，它可能是“更多计算换更高正确率”的合理策略，也可能是长度偏置造成的假象。后文会看到，按回答长度归一化会改变梯度权重，Dr. GRPO 等工作正是从这里指出 GRPO 的隐含偏差。

## 18.3 数学原理

### 18.3.1 问题设定与组采样

设提示为 $$q$$，策略为 $$\pi_\theta$$，一条完整回答是 Token 序列 $$o=(o_1,\dots,o_{|o|})$$。对每个提示，从旧策略采样 $$G$$ 条回答：

$$
\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\rm old}}(\cdot\mid q),
\qquad i=1,\dots,G
\qquad \text{(18.2)}
$$

每条回答由 (18.1) 得到一个二值奖励 $$r_i$$。训练目标是最大化期望奖励，同时（可选地）用 KL 惩罚把策略拴在参考模型附近：

$$
J(\theta)=\mathbb{E}_{q,\,\{o_i\}\sim\pi_{\theta_{\rm old}}}
\left[\frac{1}{G}\sum_{i=1}^{G}r_i
-\beta\,\mathrm{KL}\big(\pi_\theta(\cdot\mid q)\,\|\,\pi_{\rm ref}(\cdot\mid q)\big)\right]
$$

这里的期望关于提示分布与旧策略的采样。策略更新后，旧策略被刷新，下一轮重新采样。这个“采样—更新—刷新”的循环与 PPO 相同，区别只在优势如何估计、是否需要价值网络。旧策略必须与采样时的分布一致，否则 (18.7) 的比率就不再是“新策略相对行为策略”的修正；同一批 rollout 上更新太多轮，策略偏离旧策略越远，裁剪会拦下大部分 Token，实际有效样本数骤降。这是“多 epoch”在 RLVR 里比在 SFT 里更危险的原因。

### 18.3.2 策略梯度与基线

对单个提示，策略梯度的经典形式是

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{o\sim\pi_\theta}
\left[r(q,o)\,\nabla_\theta\log\pi_\theta(o\mid q)\right]
$$

这个估计的方差很大，因为奖励的绝对尺度直接乘在得分函数上。引入只依赖提示的基线 $$b(q)$$ 不改变期望：

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{o\sim\pi_\theta}
\left[\big(r(q,o)-b(q)\big)\,\nabla_\theta\log\pi_\theta(o\mid q)\right]
\qquad \text{(18.3)}
$$

证明只用到一个事实：$$\mathbb{E}_{o\sim\pi_\theta}[\nabla_\theta\log\pi_\theta(o\mid q)]=\nabla_\theta\sum_o\pi_\theta(o\mid q)=0$$。因此减去 $$b(q)$$ 后，多出的项 $$-b(q)\mathbb{E}[\nabla\log\pi]$$ 严格为零。基线不引入偏差，只改变方差；最优 baseline 大致是奖励的期望，组内均值正是它的一个无需训练的估计。

```info
基线是“控制变量”而不是“奖励塑形”。它不改变梯度的期望，所以不会改变最优策略；它只改变估计的方差与有效步长。把基线当成额外的奖励信号，是理解策略梯度时最常见的错误之一。
```

### 18.3.3 组内优势与有限样本偏差

GRPO 用同一提示的组内统计量构造优势：

$$
\hat A_i=\frac{r_i-\bar r}{\sigma_r+\epsilon},
\qquad
\bar r=\frac{1}{G}\sum_{j=1}^{G}r_j,
\qquad
\sigma_r=\sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j-\bar r)^2}
\qquad \text{(18.4)}
$$

其中 $$\epsilon$$ 是防止除零的小常数。分母的标准化是可选的：去掉它就是“奖励减组均值”，保留它则让不同组、不同题目难度的优势尺度大致一致。二值奖励下，若单条回答正确率为 $$p$$，则 $$\sigma_r=\sqrt{p(1-p)}$$；当 $$p$$ 接近 0 或 1 时标准差很小，除以它会放大噪声，这是标准化在简单题和极难题上不稳定的原因之一。

把数字代进去会更清楚。设某组的中心化优势 $$r_i-\bar r=0.1$$：当 $$p=0.5$$ 时 $$\sigma_r=0.5$$，标准化后是 0.2；当 $$p=0.9$$ 时 $$\sigma_r\approx0.3$$，标准化后是 0.333；当 $$p=0.99$$ 时 $$\sigma_r\approx0.0995$$，标准化后约为 1.0。同样大小的“高于组均值”，在简单题上被放大了五倍。如果简单题的数量又很多，训练就会被这些低方差组主导。这正是 Dr. GRPO 主张去掉标准差归一化的原因：它把一个“让尺度一致”的技巧，变成了“按题目方差重新分配权重”的隐式目标。

组均值基线有一个容易被忽略的细节：它把当前样本自己也放进了均值。设组内奖励独立同分布、均值为 $$\mu$$，则

$$
\mathbb{E}\big[\bar r_i\big]
=\frac{1}{G}r_i+\frac{G-1}{G}\mu,
\qquad
\mathbb{E}\big[r_i-\bar r_i\big]
=\left(1-\frac{1}{G}\right)(r_i-\mu)
\qquad \text{(18.5)}
$$

期望是对组内其他样本取的。也就是说，包含自身的组均值会把优势系统性地缩小为原来的 $$1-1/G$$。这个收缩不改变最优策略（所有优势同乘一个正常数，梯度方向不变），但会改变有效步长，并与裁剪、学习率、标准化相互作用。$$G=1$$ 时收缩因子为零，优势恒为零，这不是数值巧合，而是结构性的：没有其他样本，就没有“相对”可言。

去掉自身可以得到无偏基线。留一法（leave-one-out）定义

$$
b_i^{(-i)}=\frac{1}{G-1}\sum_{j\ne i}r_j,
\qquad
\mathbb{E}\big[b_i^{(-i)}\big]=\mu
\qquad \text{(18.6)}
$$

它要求 $$G\ge 2$$，并且在组内样本独立时严格无偏。RLOO（REINFORCE Leave-One-Out）就是这一思路的代表。GRPO 的原始实现使用包含自身的组均值，工程上更简单；如果发现小 $$G$$ 下学习偏慢，可以检查是不是这个收缩在起作用，而不是先怀疑学习率。

```warning
$$G=1$$ 时 GRPO 完全失效，不是“效果差”，而是梯度严格为零。用 $$G=1$$ 做消融得到的“无提升”不能证明奖励无效，只能证明组相对基线没有定义。做 $$G$$ 的消融时，至少从 2 开始。
```

### 18.3.4 裁剪目标与 KL 惩罚

同一条 rollout 通常要更新多次，因此需要重要性采样把旧策略的样本用于新策略。逐 Token 的重要性比为

$$
\rho_{i,t}(\theta)
=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
{\pi_{\theta_{\rm old}}(o_{i,t}\mid q,o_{i,<t})}
\qquad \text{(18.7)}
$$

在更新开始时 $$\rho_{i,t}=1$$。PPO 的裁剪代理目标对每个 Token 取

$$
\ell_{i,t}(\theta)
=\min\Big(\rho_{i,t}\hat A_i,\;
\operatorname{clip}(\rho_{i,t},1-\epsilon_l,1+\epsilon_h)\,\hat A_i\Big)
\qquad \text{(18.8)}
$$

上界与下界可以不对称：DAPO 取 $$\epsilon_l=0.2$$、$$\epsilon_h=0.28$$，故意给上侧更大的空间。原因是正优势的 Token 概率过低时，梯度会把它推得很猛，过早把概率推到 1 会导致熵塌缩；放宽上侧裁剪可以保留探索。负优势的 Token 则相反，过早压到 0 也会让模型失去纠错机会，但风险小于正优势的过早饱和。

裁剪的梯度行为可以分三种情况。比率落在 $$[1-\epsilon_l,1+\epsilon_h]$$ 内时，目标就是标准的策略梯度，梯度正比于 $$\hat A_i\nabla\log\pi_\theta$$；正优势且比率超过上界时，目标被钉在 $$\hat A_i(1+\epsilon_h)$$，对参数的梯度为零，策略不再继续推高这个 Token；负优势且比率低于下界时同理，目标被钉在下界，梯度也为零。裁剪并不会永久冻结某个 Token：一旦策略更新使比率回到区间内，梯度就恢复。它限制的是单次更新能走多远，而不是永久禁止某个方向。

这也解释了为什么裁剪系数与学习率、epoch 数必须一起调。epoch 越多，策略偏离旧策略越远，落在裁剪区外的 Token 越多，实际获得梯度的 Token 越少；学习率越大，同样几步就能把比率推到边界。工业训练中常把裁剪比例（`clip_frac`）作为健康指标：如果大部分 Token 都被裁剪，说明更新过猛或旧策略太旧，应该减小学习率、减少 epoch，或者重新采样。

完整的 GRPO 目标把逐 Token 的代理项、KL 惩罚与长度归一化放在一起：

$$
J_{\rm GRPO}(\theta)
=\mathbb{E}\left[
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\Big(\ell_{i,t}(\theta)-\beta\,\hat D_{i,t}\Big)
\right]
\qquad \text{(18.9)}
$$

KL 项通常用 k3 估计量，它对每个 Token 非负，方差比 k1 小：

$$
\hat D_{i,t}
=\frac{\pi_{\rm ref}(o_{i,t}\mid q,o_{i,<t})}
{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
-\log\frac{\pi_{\rm ref}(o_{i,t}\mid q,o_{i,<t})}
{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
-1
\qquad \text{(18.10)}
$$

记 $$x=\pi_{\rm ref}/\pi_\theta$$，则 $$\hat D=x-\log x-1\ge 0$$，等号只在 $$x=1$$ 时成立。第 16 章式 (16.11) 已经推导过这个估计量；在这里它的作用是防止策略在追求二值奖励时把语言分布彻底破坏。$$\beta=0$$ 时参考模型可以完全去掉，DeepSeek-R1-Zero 就采用了这种“纯 RL”设定；$$\beta>0$$ 时则需要额外一份冻结模型与一次前向，成本更高，但训练更保守。

### 18.3.5 长度归一化与零方差组

(18.9) 先对每条回答按长度平均，再对组内回答平均。另一种实现是直接把批内所有响应 Token 放在一起平均：

$$
J_{\rm tok}(\theta)
=\frac{1}{\sum_{i=1}^{G}|o_i|}
\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}
\Big(\ell_{i,t}(\theta)-\beta\,\hat D_{i,t}\Big)
\qquad \text{(18.11)}
$$

两者的差别在权重。序列级平均让每条回答权重相同，短回答的每个 Token 权重更大；Token 级平均让每个 Token 权重相同，长回答整体权重更大。DAPO 采用 Token 级归一化，理由是长回答中的推理过程不应该被长度平均稀释；Dr. GRPO 则更进一步，指出按长度归一化会让“答对且更短”与“答对且更长”获得不同的梯度，建议用与长度无关的常数归一化，并去掉标准差归一化。两种主张针对的是不同失效模式：前者担心长推理被压制，后者担心长度偏置被奖励。

零方差组是另一个结构性问题。若组内所有奖励相同，则 $$r_i-\bar r=0$$，优势为零，无论是否除以标准差都没有梯度。设单条回答正确率为 $$p$$，组内回答独立，则全对概率为 $$p^G$$，全错概率为 $$(1-p)^G$$，混合组概率为

$$
P_{\rm mixed}(p,G)=1-p^G-(1-p)^G
\qquad \text{(18.12)}
$$

当 $$p=0.5$$、$$G=8$$ 时，混合组概率约为 $$1-2/256\approx0.992$$；当 $$p=0.9$$、$$G=8$$ 时，约 $$1-0.43-10^{-8}\approx0.57$$。题目太简单或太难，大部分组都是零方差组，采样算力被浪费。$$G=1$$ 时 $$P_{\rm mixed}=0$$，再次得到同一个结论。

动态采样（dynamic sampling）是 DAPO 给出的工程解法：过滤掉全对与全错的组，再补采直到批大小满足要求。它提高的是“每个梯度步的有效样本比例”，不是改变目标函数。另一种解法是课程学习：根据当前正确率调整题目难度，让 $$p$$ 保持在 0.2–0.8 的区间。两者都在对抗同一个事实——RLVR 的学习信号来自组内的差异，而不是奖励的绝对高低。

### 18.3.6 复杂度与显存

设批内提示数为 $$B$$，组大小为 $$G$$，平均回答长度为 $$L$$，模型隐藏维度为 $$d$$。每轮 rollout 需要生成 $$B\cdot G\cdot L$$ 个 Token；每个 Token 的一次前向约为 $$O(Ld+L^2d)$$（计入 KV Cache 后，解码阶段每步只处理一个新 Token，但训练阶段的反向仍要覆盖整条序列）。训练阶段对同样多的 Token 做前向与反向，激活显存约为 $$O(B_{\rm mb}GLd)$$，其中 $$B_{\rm mb}$$ 是 minibatch 内的提示数。

举个具体数字：若 $$B=16$$、$$G=8$$、平均回答长度 $$L=512$$，一轮 rollout 要生成约 $$6.6\times10^4$$ 个 Token；训练阶段还要对同样多的 Token 做前向与反向，总算力约为十几万 Token 的前向等价。注意这里的 batch 维度是 $$B\cdot G$$，激活显存也按这个维度放大；梯度检查点（gradient checkpointing）与序列并行可以换回显存，但会增加计算或通信。

与 PPO 相比，GRPO 省掉了一个与策略同规模的价值网络：参数、优化器状态、前向与反向都少一份。但它没有省掉参考模型（当 $$\beta>0$$ 时），也没有省掉 rollout。实际上，GRPO 把 PPO 的“价值网络显存”换成了“更大的组采样”。$$G$$ 从 1 增到 8，采样 Token 数变成 8 倍，rollout 成为主要成本。这也是为什么 RLVR 的系统优化重点在推理吞吐，而不是反向传播。

```danger
组内标准差可能为零或极小。除以 $$\sigma_r+\epsilon$$ 时，若 $$\epsilon$$ 太小，二值奖励在极端正确率下会产生巨大的优势；若 $$\epsilon$$ 太大，又会把正常组的优势压扁。工程上通常取 $$10^{-4}$$ 到 $$10^{-6}$$，并在日志中监控零方差组比例与优势绝对值。KL 项的 log 比也要裁剪，避免指数溢出。
```

## 18.4 方案对比

GRPO 不是孤立的算法，而是“在线策略优化 + 组内基线 + 可验证奖励”这一族方法中的一个点。把它与 PPO、DPO、RLOO 放在同一张表里，才能看清各自的取舍。

表 18-2：四种后训练目标的对照

| 维度 | PPO（第 16 章） | GRPO | DPO（第 17 章） | RLOO |
| --- | --- | --- | --- | --- |
| 数据来源 | 当前策略在线采样 | 当前策略在线采样，每提示 $$G$$ 条 | 固定成对偏好数据 | 当前策略在线采样 |
| 奖励来源 | 学习的奖励模型 | 可验证规则或奖励模型 | 偏好对中的隐式奖励 | 可验证规则或奖励模型 |
| 价值网络 | 需要 | 不需要 | 不需要 | 不需要 |
| 基线 | 价值网络估计 | 组均值 | 无（成对比较） | 留一均值 |
| 参考模型 | 需要（KL） | 可选（KL） | 需要 | 可选 |
| 主要偏差 | 价值估计偏差 | 有限 $$G$$ 收缩、长度/标准差偏置 | 离线数据覆盖不足 | 组内样本相关性 |
| 适用场景 | 通用偏好对齐、稠密奖励 | 数学/代码/工具等可验证任务 | 有成对偏好、算力有限 | 可验证任务、小 $$G$$ |

这张表最重要的不是“谁更好”，而是“谁需要什么”。PPO 需要可靠的奖励模型和价值网络；DPO 需要高质量的成对偏好；GRPO 需要可验证奖励和足够的组内多样性。把 GRPO 用在没有验证器的开放域对话上，奖励仍然要靠奖励模型，这时它相对 PPO 的优势就只剩“少一个价值网络”，而组内采样会显著增加成本。

RLOO 与 GRPO 的差别比表面看起来小。对固定的一组奖励，留一优势与组均值优势只差一个常数倍，$$r_i-b_i^{(-i)}=\frac{G}{G-1}(r_i-\bar r)$$。也就是说，不除以标准差时，RLOO 只是把 GRPO 的优势放大了 $$G/(G-1)$$ 倍；除以标准差后，这个常数被约掉，两者的归一化优势完全相同。RLOO 的“无偏”主要修的是尺度，而不是方向；它不会改变最优策略，但会改变有效步长，进而与裁剪、学习率相互作用。理解这一点，就不会把 RLOO 当成一个与 GRPO 完全不同的算法；它更像是“换了一个基线尺度”的 GRPO。

GRPO 之后出现了一系列改进，可以按“它们修的是哪个偏差”来记：

表 18-3：GRPO 的主要变体

| 方法 | 关键改动 | 针对的问题 |
| --- | --- | --- |
| RLOO | 用留一均值替代包含自身的组均值 | 有限 $$G$$ 的 $$(1-1/G)$$ 收缩 |
| DAPO | clip-higher（$$\epsilon_h=0.28$$）、动态采样、Token 级损失、超长奖励塑形 | 熵塌缩、零方差组浪费、长回答被稀释 |
| Dr. GRPO | 去掉标准差归一化与长度归一化 | 难度偏置与长度偏置 |
| GSPO | 用序列级重要性比与序列级裁剪替代 Token 级 | 长序列中 Token 级比率方差爆炸 |

DAPO 的四项改动值得展开。clip-higher 放宽上侧裁剪，避免正优势 Token 过早饱和；动态采样过滤零方差组并补采，保证每个 batch 都有梯度；Token 级损失让长回答的每个 Token 都被计入，而不是被长度平均压小；超长奖励塑形对超过长度上限的回答做软惩罚，防止模型用无限长的重复推理骗取正确率。四项改动各自独立，但共同指向一个目标：让稀疏的二值奖励更稳定地转化为有效的策略梯度。

Dr. GRPO 的批评更根本。它指出，按长度归一化会让不同长度的回答获得不同的梯度尺度，按标准差归一化会让低方差（过难或过易）的题目获得更大的权重；两者叠加，会把优化目标从“提高正确率”悄悄改成“偏好某些长度与难度”。它的修正很简洁：去掉这两个归一化，用一个与长度无关的常数做分母。这个例子说明，RLVR 的很多“技巧”其实是在补偿目标函数里的隐含偏置，而不是在增加新信息。

GSPO 处理的是长思维链带来的数值问题。Token 级重要性比是每个 Token 比率的连乘，序列越长，比率的方差越大，裁剪也越难稳定。GSPO 把整条回答的似然比作为一个序列级比率，并在序列级裁剪。代价是粒度变粗，无法逐 Token 判断哪一步该被强化；收益是长序列训练更稳定。选型时要问：当前瓶颈是“信用分配不够细”，还是“长序列比率不稳定”？前者倾向 Token 级，后者倾向序列级。

```success
选型可以按三步走：先问奖励是否可验证——可验证才优先考虑 GRPO/RLOO；再问回答是否很长——很长时优先考虑序列级比率或 GSPO 类方法；最后问算力与显存——组采样是线性成本，$$G$$ 每翻一倍，rollout 成本就翻一倍。不要在没有验证器的任务上为了“省一个价值网络”而选 GRPO。
```

## 18.5 代码实现

### 18.5.1 模块与实验设计

第 18 章的增量代码放在 `tinylm/posttrain/grpo.py`，配置类 `GRPOConfig` 只作为训练脚本参数，不进入 `ModelConfig`，`TinyLM.forward` 与 `CausalLMOutput` 的签名保持不变。模块提供六组函数：

- `verifiable_reward`：实现 (18.1)，检查回答中最后一个非 EOS Token 是否等于目标答案；
- `sample_groups`：实现 (18.2)，把每个提示重复 $$G$$ 次后按温度采样；
- `group_advantages`：实现 (18.4)–(18.6)，支持组均值、留一法与可选标准差归一化；
- `tempered_logprobs`：实现 (18.7) 的分母，保证采样与对数概率使用同一温度；
- `grpo_loss`：实现 (18.7)–(18.11)，返回裁剪损失、比率、裁剪比例与 KL；
- `zero_variance_groups` 与 `mixed_group_probability`：实现 (18.12)，用于诊断。

实验任务是一个刻意简化的模运算：提示是 `[BOS, Q, digit(a), SEP]`，要求回答 `(a+3) mod 10` 对应的数字 Token，随后输出 EOS。词表 16 个 Token，数字占 10 个，随机猜中的概率是 0.1。SFT 数据只有 30% 正确，训练 50 步，故意让 SFT 模型停在“会格式、不会算”的状态，为 RL 留出提升空间。若 SFT 已经全对，RL 就没有可学习的差异，实验会退化成“确认零梯度”。

### 18.5.2 完整代码

`tinylm/posttrain/grpo.py`。函数注释标注了对应公式；`group_advantages` 在 `group_size == 1` 时直接返回零，而不是让均值除法产生一个恒为零的浮点结果。`grpo_loss` 返回损失与诊断量，诊断量包含策略梯度、平均比率与裁剪比例。

```python
"""第 18 章：可验证奖励与组相对策略优化（tinylm/posttrain/grpo.py）。"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tinylm.posttrain import ppo as P


@dataclass(frozen=True)
class GRPOConfig:
    """GRPO 训练脚本参数，不进入 ``ModelConfig``。"""

    group_size: int = 8             # 每个提示采样的回答数 G
    lr: float = 1e-3
    beta: float = 0.0               # KL 系数；0 表示不使用参考模型
    clip_low: float = 0.2           # 下侧裁剪半径
    clip_high: float = 0.2          # 上侧裁剪半径，DAPO 取 0.28
    epochs: int = 2                 # 同一组 rollout 重复更新次数
    minibatch: int = 64
    max_new: int = 4                # 响应最多生成多少个 Token（含 EOS）
    temperature: float = 1.0
    normalize_std: bool = True      # 是否除以组内标准差
    leave_one_out: bool = False     # 基线是否排除当前样本
    token_level: bool = True        # 按批内 Token 平均，而非逐序列平均
    max_grad_norm: float = 1.0
    pad_id: int = 0
    eos_id: int = 4

    def __post_init__(self):
        assert self.group_size >= 1
        assert self.lr > 0.0
        assert self.beta >= 0.0
        assert 0.0 <= self.clip_low < 1.0
        assert 0.0 <= self.clip_high < 1.0
        assert self.epochs >= 1
        assert self.minibatch >= 1
        assert self.max_new >= 1
        assert self.temperature > 0.0
        assert self.max_grad_norm > 0.0


def group_advantages(rewards, group_size, normalize_std=True,
                     leave_one_out=False, eps=1e-4):
    """(18.4)–(18.6)：组内均值基线，可选除以组内标准差。

    ``rewards`` 形状 ``(B * G,)``，组内连续排列。G=1 且使用组均值基线时，
    每个样本的基线就是它自己，优势恒为零，这是 GRPO 至少需要两个样本的
    直接原因。``leave_one_out=True`` 用其余 G-1 个样本的均值作基线，可以
    消除有限 G 带来的 (1 - 1/G) 收缩。
    """
    groups = rewards.view(-1, group_size)
    if group_size == 1:
        if leave_one_out:
            raise ValueError("leave-one-out 基线要求 group_size >= 2")
        return torch.zeros_like(groups).view(-1)
    if leave_one_out:
        total = groups.sum(dim=1, keepdim=True)
        baseline = (total - groups) / (group_size - 1)
    else:
        baseline = groups.mean(dim=1, keepdim=True)
    centered = groups - baseline
    if normalize_std:
        scale = groups.std(dim=1, unbiased=False, keepdim=True)
        centered = centered / (scale + eps)
    return centered.view(-1)


def zero_variance_groups(rewards, group_size, eps=1e-6):
    """(18.12)：返回 ``(B,)`` 布尔张量，标记全对或全错的组。"""
    groups = rewards.view(-1, group_size)
    return groups.std(dim=1, unbiased=False) < eps


def mixed_group_probability(p, group_size):
    """(18.12)：组内既有正确又有错误回答的概率。

    单条回答正确率为 ``p`` 时，全对概率为 ``p^G``，全错为 ``(1-p)^G``，
    两者都给出零优势。只有混合组提供策略梯度。
    """
    return 1.0 - p ** group_size - (1.0 - p) ** group_size


@torch.no_grad()
def sample_groups(policy, prompts, cfg, generator):
    """(18.2)：把每个提示重复 G 次后采样，返回 ``(B*G, T)`` 的序列与掩码。

    采样与日志概率使用同一个温度，保证重要性比的分母就是行为策略。
    """
    policy.eval()
    seqs = prompts.repeat_interleave(cfg.group_size, dim=0)
    done = torch.zeros(seqs.size(0), dtype=torch.bool)
    for _ in range(cfg.max_new):
        logits = policy(seqs).logits[:, -1] / cfg.temperature
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, cfg.pad_id), nxt)
        seqs = torch.cat([seqs, nxt[:, None]], dim=1)
        done = done | (nxt == cfg.eos_id)
        if bool(done.all()):
            break
    return seqs, seqs != cfg.pad_id


def tempered_logprobs(model, seqs, attn, temperature=1.0):
    """(18.7)：与采样温度一致的逐 Token 对数概率，形状 ``(B, T)``。"""
    logits = model(seqs, attention_mask=attn).logits[:, :-1] / temperature
    logp = F.log_softmax(logits, dim=-1)
    target = seqs[:, 1:]
    out = logp.gather(-1, target[..., None]).squeeze(-1)
    out = torch.where(attn[:, 1:], out, torch.zeros_like(out))
    return F.pad(out, (1, 0))


def grpo_loss(logp, old_logp, adv, mask, ref_logp=None, beta=0.0,
              clip_low=0.2, clip_high=0.2, token_level=True):
    """(18.7)–(18.11)：GRPO 的裁剪代理目标与 KL 惩罚。

    ``logp``、``old_logp``、``mask`` 形状 ``(N, T)``，``adv`` 形状 ``(N,)``。
    返回损失与诊断量；损失已经取负，可以直接反向。
    """
    # (18.7) 重要性比
    ratio = torch.exp(logp - old_logp)
    # (18.8) 裁剪代理目标
    surrogate = torch.minimum(
        ratio * adv[:, None],
        torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * adv[:, None],
    )
    if token_level:
        # (18.11) 左式：批内所有响应 Token 一起平均
        pg = (surrogate * mask).sum() / mask.sum().clamp_min(1)
    else:
        # (18.11) 右式：先对每条回答平均，再对回答平均
        lengths = mask.sum(1).clamp_min(1)
        pg = ((surrogate * mask).sum(1) / lengths).mean()
    loss = -pg
    diagnostics = {
        "pg": float(pg.detach()),
        "ratio": float(((ratio * mask).sum() / mask.sum().clamp_min(1)).detach()),
        "clip_frac": float((((ratio < 1.0 - clip_low) |
                             (ratio > 1.0 + clip_high)) * mask).sum()
                           / mask.sum().clamp_min(1)),
    }
    if ref_logp is not None and beta > 0.0:
        # (18.10) k3 估计量：r - log r - 1，其中 r = π_ref / π_θ
        log_ratio = torch.clamp(ref_logp - logp, -20.0, 20.0)
        k3 = torch.exp(log_ratio) - log_ratio - 1.0
        kl = (k3 * mask).sum() / mask.sum().clamp_min(1)
        loss = loss + beta * kl
        diagnostics["kl"] = float(kl.detach())
    return loss, diagnostics


def answer_positions(seqs, prompt_len, pad_id=0, eos_id=4):
    """返回每条响应中最后一个非 padding、非 EOS 的 Token 位置。"""
    positions = torch.arange(seqs.size(1), device=seqs.device)[None, :]
    valid = (seqs != pad_id) & (seqs != eos_id) & (positions >= prompt_len)
    has = valid.any(dim=1)
    reverse = valid.long().flip(-1).argmax(dim=-1)
    idx = seqs.size(1) - 1 - reverse
    return torch.where(has, idx, torch.full_like(idx, -1))


def verifiable_reward(seqs, prompt_len, targets, mode="final",
                      pad_id=0, eos_id=4):
    """(18.1)：规则可验证奖励。``final`` 检查最后一个答案 Token，
    ``anywhere`` 只要求答案出现在响应中的任意位置。"""
    mask = (seqs != pad_id) & (seqs != eos_id)
    if mode == "anywhere":
        return ((seqs == targets[:, None]) & mask).any(dim=1).float()
    if mode != "final":
        raise ValueError(f"未知奖励模式 {mode!r}")
    idx = answer_positions(seqs, prompt_len, pad_id, eos_id)
    safe = idx.clamp_min(0)
    token = seqs.gather(1, safe[:, None]).squeeze(1)
    return ((token == targets) & (idx >= 0)).float()
```

`examples/ch18_grpo.py`。训练循环把一次 rollout 组织成“采样 → 奖励 → 优势 → 多轮小批量更新”。旧策略对数概率在更新前算一次，之后固定；每个 epoch 重新打乱样本，用 (18.7) 的比率修正策略变化。为了让玩具实验在 CPU 上可复现，示例只跑 1 个 epoch、300 轮、批 16 个提示；生产训练通常跑 2–4 个 epoch，并用更大的批与组。

```python
"""第 18 章：可验证奖励与组相对策略优化（examples/ch18_grpo.py）。"""
import time

import torch

from ch08 import ModelConfig, TinyLM
from tinylm.posttrain import grpo as G

PAD, BOS, Q, SEP, EOS = 0, 1, 2, 3, 4
DIGIT0 = 5
PROMPT_LEN = 4
N_PROMPTS = 10
SEEDS = (0, 1, 2)


def prompt(a):
    return [BOS, Q, DIGIT0 + a, SEP]


def answer(a):
    return DIGIT0 + ((a + 3) % 10)


def sft_example(a, gen, p_correct=0.3):
    """弱监督数据：只有 30% 的答案正确，其余从数字词表随机抽取。"""
    tok = answer(a) if float(torch.rand(1, generator=gen)) < p_correct else \
        int(torch.randint(DIGIT0, DIGIT0 + 10, (1,), generator=gen))
    ids = prompt(a) + [tok, EOS]
    labels = [-100] * len(ids)
    labels[-3] = tok          # 第 4 个位置（SEP）预测答案
    labels[-2] = EOS          # 第 5 个位置（答案）预测 EOS
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
            "attention_mask": torch.ones(len(ids), dtype=torch.bool)}


def collate(examples):
    n = max(e["input_ids"].numel() for e in examples)
    ids = torch.zeros(len(examples), n, dtype=torch.long)
    labels = torch.full((len(examples), n), -100, dtype=torch.long)
    mask = torch.zeros(len(examples), n, dtype=torch.bool)
    for i, e in enumerate(examples):
        k = e["input_ids"].numel()
        ids[i, :k] = e["input_ids"]
        labels[i, :k] = e["labels"]
        mask[i, :k] = True
    return ids, labels, mask


def make_model(seed):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=16, d_model=32, n_layers=2, n_heads=4,
                      d_ff=128, max_seq_len=32, pos_encoding="rope",
                      norm="rmsnorm", ffn="swiglu", bias=False)
    return TinyLM(cfg), cfg


def train_sft(model, seed, steps=50, p_correct=0.3):
    gen = torch.Generator().manual_seed(1000 + seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, N_PROMPTS, (32,), generator=gen)
        ids, labels, mask = collate([sft_example(int(i), gen, p_correct)
                                     for i in idx])
        out = model(ids, attention_mask=mask, labels=labels)
        opt.zero_grad()
        out.loss.backward()
        opt.step()


@torch.no_grad()
def greedy_accuracy(model, prompts=None):
    model.eval()
    a = list(range(N_PROMPTS)) if prompts is None else prompts
    seqs = torch.tensor([prompt(x) for x in a])
    done = torch.zeros(len(a), dtype=torch.bool)
    for _ in range(4):
        nxt = model(seqs).logits[:, -1].argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        seqs = torch.cat([seqs, nxt[:, None]], dim=1)
        done = done | (nxt == EOS)
        if bool(done.all()):
            break
    goals = torch.tensor([answer(x) for x in a])
    return float(G.verifiable_reward(seqs, PROMPT_LEN, goals).mean())


@torch.no_grad()
def sampled_accuracy(model, seed, group_size=8):
    """用固定评测种子采 ``group_size`` 条，返回 pass@1 与 pass@G。"""
    gen = torch.Generator().manual_seed(9000 + seed)
    prompts = torch.tensor([prompt(a) for a in range(N_PROMPTS)])
    seqs, attn = G.sample_groups(
        model, prompts, G.GRPOConfig(group_size=group_size, max_new=4), gen)
    goals = torch.tensor([answer(a) for a in range(N_PROMPTS)])
    rew = G.verifiable_reward(seqs, PROMPT_LEN, goals.repeat_interleave(group_size))
    rew = rew.view(N_PROMPTS, group_size)
    return float(rew.mean()), float(rew.any(1).float().mean())


def train_grpo(sft_state, cfg, gcfg, seed, rounds=300, batch=16):
    policy = TinyLM(cfg)
    policy.load_state_dict(sft_state)
    opt = torch.optim.AdamW(policy.parameters(), lr=gcfg.lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(7000 + seed)
    rew_hist, zero_hist = [], []
    for _ in range(rounds):
        idx = torch.randint(0, N_PROMPTS, (batch,), generator=gen)
        prompts = torch.tensor([prompt(int(i)) for i in idx])
        seqs, attn = G.sample_groups(policy, prompts, gcfg, gen)
        goals = torch.tensor([answer(int(i)) for i in idx])
        goals = goals.repeat_interleave(gcfg.group_size)
        rew = G.verifiable_reward(seqs, PROMPT_LEN, goals)
        adv = G.group_advantages(rew, gcfg.group_size, gcfg.normalize_std,
                                 gcfg.leave_one_out)
        zero = G.zero_variance_groups(rew, gcfg.group_size)
        rew_hist.append(float(rew.mean()))
        zero_hist.append(float(zero.float().mean()))

        mask = (seqs != PAD) & (
            torch.arange(seqs.size(1))[None, :] >= PROMPT_LEN)
        with torch.no_grad():
            old = G.tempered_logprobs(policy, seqs, attn, gcfg.temperature)
        policy.train()
        for _ in range(gcfg.epochs):
            perm = torch.randperm(seqs.size(0), generator=gen)
            for start in range(0, seqs.size(0), gcfg.minibatch):
                mb = perm[start:start + gcfg.minibatch]
                logp = G.tempered_logprobs(policy, seqs[mb], attn[mb],
                                           gcfg.temperature)
                loss, _ = G.grpo_loss(logp, old[mb], adv[mb], mask[mb], None,
                                      beta=gcfg.beta, clip_low=gcfg.clip_low,
                                      clip_high=gcfg.clip_high,
                                      token_level=gcfg.token_level)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                               gcfg.max_grad_norm)
                opt.step()
    return policy, sum(rew_hist) / len(rew_hist), sum(zero_hist) / len(zero_hist)


VARIANTS = (
    ("G=1", dict(group_size=1)),
    ("G=4", dict(group_size=4)),
    ("G=8", dict(group_size=8)),
)


def main():
    t0 = time.time()
    rows = {"SFT": ([], [], [])}
    for name, _ in VARIANTS:
        rows[name] = ([], [], [])
    for seed in SEEDS:
        model, cfg = make_model(seed)
        train_sft(model, seed)
        sft_state = {k: v.clone() for k, v in model.state_dict().items()}
        p1, p8 = sampled_accuracy(model, seed)
        g3 = greedy_accuracy(model)
        rows["SFT"][0].append(g3)
        rows["SFT"][1].append(p1)
        rows["SFT"][2].append(p8)
        print(f"seed {seed} SFT: greedy {g3:.2f} "
              f"pass@1 {p1:.2f} pass@8 {p8:.2f}")
        for name, kw in VARIANTS:
            gcfg = G.GRPOConfig(lr=1e-3, epochs=1, minibatch=64, max_new=4,
                                beta=0.0, **kw)
            ts = time.time()
            policy, rew, zero = train_grpo(sft_state, cfg, gcfg, seed)
            p1, p8 = sampled_accuracy(policy, seed)
            g3 = greedy_accuracy(policy)
            rows[name][0].append(g3)
            rows[name][1].append(p1)
            rows[name][2].append(p8)
            print(f"  {name:<11} greedy {g3:.2f} "
                  f"pass@1 {p1:.2f} pass@8 {p8:.2f} "
                  f"reward {rew:.2f} zero {zero:.2f} {time.time()-ts:.1f}s")
    print("\n3-seed means")
    print(f"{'':<11}{'greedy':>7}{'pass@1':>8}{'pass@8':>8}")
    for name, (g3, p1, p8) in rows.items():
        print(f"{name:<11}{sum(g3)/len(g3):>7.2f}"
              f"{sum(p1)/len(p1):>8.2f}{sum(p8)/len(p8):>8.2f}")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
```

### 18.5.3 运行结果

在 CPU 上运行 `python examples/ch18_grpo.py`，3 个种子、每种子 300 轮，结果如下（原样输出）：

```text
seed 0 SFT: greedy 0.10 pass@1 0.00 pass@8 0.00
  G=1         greedy 0.10 pass@1 0.00 pass@8 0.00 reward 0.09 zero 1.00 2.6s
  G=4         greedy 0.70 pass@1 0.50 pass@8 0.80 reward 0.23 zero 0.48 5.5s
  G=8         greedy 0.90 pass@1 0.90 pass@8 0.90 reward 0.61 zero 0.52 9.6s
seed 1 SFT: greedy 0.10 pass@1 0.11 pass@8 0.60
  G=1         greedy 0.10 pass@1 0.11 pass@8 0.60 reward 0.09 zero 1.00 2.6s
  G=4         greedy 0.10 pass@1 0.15 pass@8 0.50 reward 0.09 zero 0.68 5.5s
  G=8         greedy 0.60 pass@1 0.51 pass@8 0.70 reward 0.24 zero 0.45 9.7s
seed 2 SFT: greedy 0.10 pass@1 0.05 pass@8 0.40
  G=1         greedy 0.10 pass@1 0.05 pass@8 0.40 reward 0.09 zero 1.00 2.6s
  G=4         greedy 0.20 pass@1 0.16 pass@8 0.70 reward 0.10 zero 0.67 5.6s
  G=8         greedy 1.00 pass@1 0.99 pass@8 1.00 reward 0.39 zero 0.37 9.6s

3-seed means
            greedy  pass@1  pass@8
SFT           0.10    0.05    0.33
G=1           0.10    0.05    0.33
G=4           0.33    0.27    0.67
G=8           0.83    0.80    0.87
total 54.8s
```

### 18.5.4 结果解读

第一，随机水平是 0.1，SFT 的贪心正确率正好是 0.10。这不是巧合：弱监督 SFT 没有学到“加 3”的规则，只学到了输出格式；它在 10 道题上的贪心答案几乎是同一个数字。pass@1 的均值 0.05 低于随机水平，说明模型的采样分布并不均匀，而是集中在少数几个错误答案上。

第二，$$G=1$$ 的三行与 SFT 逐位相同，奖励均值 0.09，零方差组比例 1.00。由于组内只有一个样本，优势恒为零，反向传播的梯度为零，优化器虽然被调用，参数却没有变化。这个结果验证了 (18.5)：$$G=1$$ 时收缩因子 $$1-1/G=0$$，不是“效果不好”，而是“没有信号”。

第三，$$G=8$$ 的三个种子分别达到 0.90、0.60、1.00，均值 0.83；$$G=4$$ 只有 0.33。组越大，组内越可能出现“有的对、有的错”的混合组，优势估计也越稳定。这个趋势与 (18.12) 一致：固定正确率下，$$G$$ 增大会提高混合组概率，同时降低组均值的方差。代价是采样成本线性增长：$$G=8$$ 的每轮耗时约为 $$G=1$$ 的 3.7 倍，而不是 8 倍，因为小模型的前向固定开销占了一部分。

第四，种子方差很大。$$G=8$$ 在种子 1 上只到 0.60，说明 10 道题、300 轮的玩具任务仍然是一个高方差实验。本章报告 3 个种子的逐种子结果与均值，而不是只报最好的一次；读者复现时应当预期同一配置在不同种子间有明显波动。如果只报种子 0 或种子 2，会得到“GRPO 几乎解决任务”的结论；只看种子 1，又会得到“GRPO 提升有限”的结论。两者都是真实数据的一部分，单种子报告没有统计意义。

第五，pass@8 的均值 0.87 高于 pass@1 的 0.80，说明模型的采样分布里已经包含正确答案，但贪心解码还没有完全收敛到它；这也是 RLVR 常见的现象——先提高采样命中率，再提高贪心稳定性。更值得注意的是，SFT 的 pass@8 均值已经有 0.33，种子 1 甚至达到 0.60：基座模型偶尔就能采到正确答案，RL 的主要作用是把这种“偶尔”变成“稳定”。这与 18.8 中 Yue et al. 的观察一致：RL 首先改善的是采样效率，至于是否扩展了能力边界，需要更大的 $$k$$ 与更严格的留出评测来判断。

第六，零方差组比例是训练健康度的直接指标。$$G=8$$ 三个种子的平均比例为 0.37–0.52，$$G=4$$ 约为 0.61，$$G=1$$ 恒为 1.00。比例越高，采样算力中真正贡献梯度的部分越少。比例过高时，可能是题目太简单（全对）、太难（全错），也可能是温度太低导致组内回答缺乏多样性；需要结合奖励均值与回答熵一起判断，不能只看比例。当大多数组都全对时，模型已经接近饱和；真正危险的是“全对与全错各占一半、混合组很少”的难度分布，这时需要课程学习或动态采样。

### 18.5.5 测试与复现

`tests/test_grpo.py` 共 17 项测试，覆盖：$$G=1$$ 优势为零、组均值优势求和为零、留一法无偏、标准差归一化的缩放、零方差组标记、混合组概率、裁剪上下界、Token 级与序列级损失的差异、参考模型等于策略时 KL 为零、答案位置跳过 EOS/PAD、最终答案与任意位置奖励的差异、温度对数概率与手算 softmax 一致、组采样形状与重复提示、非法配置拒绝。全部通过约 1.1 s。

复现步骤：

```bash
python -m pytest tests/test_grpo.py -q     # 17 passed
python examples/ch18_grpo.py               # CPU 约 55 s
```

```text
17 passed in 1.11s
```

## 18.6 工业实践

### 18.6.1 DeepSeekMath 与 GRPO 的起点

GRPO 由 DeepSeekMath 提出，最初用于数学推理。它面对的是一个很实际的问题：PPO 的价值网络与策略同规模，在数学任务上要同时维护策略、参考、奖励、价值四份模型，显存和调参成本都很高。GRPO 用同一道题的组内平均奖励替代价值网络，把“学习一个 critic”变成“多采样几条回答”，在数学这种奖励可验证、采样相对便宜的任务上非常划算。原论文对每个问题采样 64 条回答，用答案正确性作为奖励，在 GSM8K 与 MATH 上超过了当时的 PPO 基线。

这个案例的启示是：算法选择要和任务的成本结构匹配。数学题的验证几乎免费，多采样 64 条的代价主要是推理算力；如果换成开放式写作，验证需要奖励模型，多采样 64 条既贵又未必有区分度，GRPO 的性价比就会下降。

### 18.6.2 DeepSeek-R1：纯 RL 与长思维链

DeepSeek-R1-Zero 是“纯 RL”路线的代表：从基座模型出发，不做 SFT，直接用 GRPO 训练，奖励由两部分组成——答案正确性奖励与格式奖励（要求把推理过程放在指定标签内）。论文报告 AIME 2024 的 pass@1 从 15.6% 提升到 71.0%，多数投票可达 86.7%。训练过程中回答长度持续增长，模型自发出现回看、验算、换思路等行为，论文称之为“顿悟时刻”（aha moment）。

但 R1-Zero 也暴露了纯 RL 的问题：回答可读性差、语言混杂、格式不稳定。DeepSeek-R1 因此改成多阶段流程：先用少量冷启动数据做 SFT，让模型学会可读的长思维链格式；再做推理导向的 RL；然后用拒绝采样筛选高质量回答，混合通用数据做一次 SFT；最后做覆盖推理与通用能力的第二轮 RL。这个流程说明，RLVR 不是“不要 SFT”，而是“用 RL 提升能力上限，用 SFT 稳定输出形式”。

R1 还做了一个重要的工程选择：不采用过程奖励模型。论文给出的理由是，过程奖励的“每一步是否正确”很难用统一规则定义，容易引入主观判断与奖励作弊，而且需要额外的过程标注或过程奖励模型训练。结果奖励加格式奖励虽然稀疏，但定义清晰、可扩展。这个选择与本章的主线一致：RLVR 的力量来自“验证器足够客观”，而不是“奖励足够稠密”。

### 18.6.3 DAPO：把稳定性做成系统

DAPO 是 ByteDance 提出的开源大规模 RL 系统，基于 Qwen2.5-32B 基座，在 AIME 2024 上达到 50 分，并报告比 DeepSeek-R1-Zero 的 Qwen-32B 版本少用约 50% 的训练步数。它的贡献不只是某个损失项，而是把四个稳定性问题一起处理：

- clip-higher：把上侧裁剪从 0.2 放宽到 0.28，缓解正优势 Token 的熵塌缩；
- 动态采样：过滤全对/全错组并补采，保证 batch 内始终有梯度；
- Token 级策略梯度：用 (18.11) 的左式，避免长回答被长度平均稀释；
- 超长奖励塑形：对超过长度上限的回答做软惩罚，抑制用无限长推理刷正确率。

这四项分别对应本章讨论过的零方差组、长度归一化与熵塌缩。DAPO 的价值在于证明：GRPO 的原始形式在大规模训练中会遇到系统性问题，必须把“采样、归一化、裁剪、长度控制”作为一整套设计，而不是只调学习率。

### 18.6.4 长上下文与推理系统

Kimi k1.5 把 RL 的训练上下文扩展到 128k，使用部分 rollout（partial rollout）避免长序列采样阻塞训练，并用“长到短”（long2short）的蒸馏把长思维链的能力迁移到短回答模型（Kimi Team, 2025, arXiv:2501.12599）。它与第 13 章的长上下文训练直接相关：长思维链让序列长度成为新的瓶颈，位置编码的外推、KV Cache 的显存、采样吞吐都会限制可训练的回答长度。

在工程上，RLVR 的训练系统通常包含四个角色：推理引擎负责高吞吐 rollout；验证器服务负责答案匹配、单测执行或工具调用；训练进程负责前向反向与参数更新；调度器负责把 rollout、验证、训练流水线化。任何一个环节的吞吐不匹配，都会让 GPU 在等待中空转。第 21 章的连续批处理与 PD 分离，在这里直接决定组采样能否喂饱训练。

开放后训练也把 RLVR 纳入了标准配方。Tulu 3（Lambert et al., 2024, arXiv:2411.15124）在 SFT、DPO 之后加入 RLVR 阶段：数学题用可验证奖励，其他任务用奖励模型，再用 PPO 统一优化。它的意义不在于刷新某个榜单，而在于证明 RLVR 不是推理模型的专属技术，可以作为一种通用的后训练阶段，与已有的偏好数据、奖励模型和评测体系拼接。代价是管线更长、需要维护的奖励来源更多，训练稳定性也更难归因。

工业上的奖励很少只有一个 0/1。常见的组合是“正确性奖励 + 格式奖励 + 长度惩罚 + 重复惩罚”：正确性奖励保证目标，格式奖励让输出可解析，长度惩罚抑制无意义的重复推理，重复惩罚降低退化风险。DeepSeek-R1-Zero 用正确性加格式奖励；DAPO 用超长奖励塑形；许多系统还会对空思考、循环重复、提前终止单独扣分。奖励项越多，越需要监控每一项的激活比例，否则策略会找到“只拿格式分、不拿正确性分”的捷径。

```warning
可验证奖励的工业风险主要不在优化，而在验证器。数学答案要做等价形式归一化；代码单测要在沙箱里跑，并防止策略读取测试文件；工具调用要防止缓存与环境污染；评测集要做去污染。验证器的一个漏洞，会被 RL 放大成系统性的奖励作弊。
```

## 18.7 论文导读

1. **Shao et al., 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.** arXiv:2402.03300。读第 4 节，重点是 GRPO 的目标函数、组内优势与“去掉价值网络”的动机；第 5 节的数学实验结果可以对照本章的玩具实验。

2. **DeepSeek-AI, 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.** arXiv:2501.12948。读第 2.2–2.3 节与图 2–3：R1-Zero 的纯 RL 流程、AIME 曲线、回答长度增长与“顿悟时刻”；第 2.4 节的多阶段流程解释了为什么最终系统仍然需要 SFT。

3. **Yu et al., 2025. DAPO: An Open-Source LLM Reinforcement Learning System at Scale.** arXiv:2503.14476。读第 3 节的四项技术（clip-higher、动态采样、Token 级损失、超长奖励塑形）与第 4 节的消融；注意它如何把本章的 (18.11) 与 (18.12) 变成工程约束。

4. **Liu et al., 2025. Understanding R1-Zero-Like Training: A Critical Perspective.** arXiv:2503.20783（Dr. GRPO）。读第 3 节对长度归一化与标准差归一化的偏差分析；这篇论文适合与本章 18.3.5 对照阅读，理解“看似中性的归一化”如何改变优化目标。

5. **Zheng et al., 2025. Group Sequence Policy Optimization.** arXiv:2507.18071（GSPO）。读第 2–3 节的序列级重要性比与裁剪；它解决的是长思维链下 Token 级比率的方差问题，适合与 DAPO 的 Token 级方案对比。


## 18.8 常见误区

**误区一：GRPO 不需要价值网络，所以不需要基线。** 纠正：组均值就是基线。去掉价值网络不等于去掉基线；如果没有基线，策略梯度的方差会大到无法训练。GRPO 的创新是“用组内蒙特卡洛均值替代学习的 critic”，不是“取消基线”。

**误区二：组越大越好。** 纠正：组大小是成本与信号质量的折中。$$G$$ 增大提高混合组概率、降低基线方差，但采样成本线性增长；当正确率接近 1 时，全对组占比按 $$p^G$$ 增长，增加 $$G$$ 反而浪费算力。工业上常见的 $$G$$ 是 8–64，要结合题目难度分布选择，而不是盲目加大。

**误区三：全对/全错组没有梯度是代码 bug。** 纠正：这是组相对目标的定义决定的。组内没有差异，优势就为零。正确的处理是动态采样、课程学习或难度过滤，而不是给损失加一个常数“强行制造梯度”。后者会引入与正确性无关的偏置。

**误区四：标准差归一化总能让训练更稳。** 纠正：二值奖励下标准差是 $$\sqrt{p(1-p)}$$，在 $$p$$ 接近 0 或 1 时很小，除以它会放大噪声；Dr. GRPO 还指出它会让低方差题目获得更大权重。归一化是否有利，取决于题目难度分布与是否使用动态采样，不能一概而论。

**误区五：RLVR 一定让模型学会新的推理能力。** 纠正：RL 可能主要是在重加权基座模型已有的能力。Yue et al. (2025, arXiv:2504.13837) 用 pass@k 发现，RL 训练后的模型在 $$k$$ 很大时会被基座模型追平，说明 RL 更多是提高采样效率，而不是扩展能力边界。判断“学会”还是“激发”，要同时报告 pass@1、pass@k 与基座对照，并控制格式与长度。这不是说 RL 没有价值：即使只是重加权，能把能力稳定地取出来，本身就有巨大的工程意义。

## 18.9 本章小结

- RLVR 用规则验证替代人类偏好，GRPO 用组内均值基线替代价值网络。
- 组内相对优势带来更少的模型与更低的显存，代价是组采样成本线性增长、小 $$G$$ 下基线有 $$(1-1/G)$$ 的收缩。
- 零方差组与长度偏置是主要工程问题；$$G=1$$ 的零梯度是该取舍的极端情形，不是实现缺陷。
- RLVR 适用于“会一点但不稳定、答案可判定、采样相对便宜”的任务，外推到开放式任务是常见的过度推广。
- 结果奖励客观但稀疏，过程奖励稠密但难规则化；DAPO 的动态采样与长度塑形是在两者之间折中。

{% hint style="success" %}
**本章的取舍**：GRPO 用“组内采样”换掉了“价值网络”。它放弃了对每个状态未来回报的精细估计，换来更少的模型、更简单的训练循环和更低的显存；代价是组采样成本线性增长，且小 $$G$$ 下基线有 $$(1-1/G)$$ 的收缩。$$G=1$$ 的零梯度不是实现缺陷，而是这个取舍的极端情形。工程上如果显存紧张、题目验证便宜，GRPO 很划算；如果每条回答很长、采样极贵，PPO 的价值网络反而可能更省。

RLVR 用“规则验证”换掉了“人类偏好”。它放弃了奖励的通用性，换来可扩展、可复现、难以被语言表面特征欺骗的奖励；代价是它只适用于答案可判定的任务。数学、代码、工具调用是它的甜点区；开放式写作、主观对话仍然需要奖励模型或人类反馈。把 RLVR 的成功外推到所有任务，是当前推理模型叙事里最常见的过度推广。

结果奖励用“稀疏但客观”换掉了“稠密但主观”。过程奖励能提供更细的信用分配，但需要逐步标注或过程奖励模型，且“这一步是否合理”很难规则化。DeepSeek-R1 选择结果奖励加格式奖励，DAPO 用动态采样与长度塑形缓解稀疏性，都是在“客观性”与“信号密度”之间做取舍。若任务本身有可靠的步骤验证器（例如形式化证明的每一步），过程奖励值得重新考虑。

在线 RL 用“探索”换掉了“离线效率”。DPO 一次监督式训练就能利用全部偏好数据，GRPO 需要不断采样、验证、更新，训练成本高得多；但只有在线采样才能发现数据之外的高奖励行为。选型时要问：我的瓶颈是数据不够，还是探索不够？前者优先 DPO/SFT，后者才需要 RLVR。

反过来，什么时候不该用 RLVR？如果任务没有可靠的验证器，或者验证器只能覆盖答案的一小部分，RLVR 会把策略变成“验证器漏洞利用器”；如果每条回答极长、采样成本远高于一次 SFT，组采样的开销可能得不偿失；如果基座模型在任务上的正确率已经接近零，组内几乎全是零方差组，RL 没有可学习的差异，应该先做 SFT 或课程预热；如果目标只是让输出更符合格式或风格，SFT/DPO 更直接。RLVR 适合的是“会一点但不稳定、答案可判定、采样相对便宜”的区域，这个区域比宣传中的“所有推理任务”窄得多。

最后是长度与难度的取舍。长思维链能提高正确率，也会带来长度偏置、采样成本与评测不公平；动态采样能提高有效梯度比例，也会引入实现复杂度与分布偏移。本章的玩具实验只展示了“组大小”这一条主线，真实系统的成败往往取决于验证器质量、题目难度分布与 rollout 吞吐这些不显眼的工程细节。

{% endhint %}

## 18.10 习题

**推导题**：设组内奖励独立同分布、均值为 $$\mu$$。证明包含自身的组均值基线满足 (18.5)，并证明留一法基线 (18.6) 无偏。进一步证明：若所有奖励相同，则 (18.4) 的优势为零；并利用 (18.12) 说明 $$G=1$$ 时混合组概率为零。

<details>
<summary>参考答案</summary>

记组内奖励为 $$r_1,\dots,r_G$$，固定当前样本 $$r_i$$，对其余样本取期望。包含自身的组均值为

$$
\bar r_i=\frac{1}{G}\left(r_i+\sum_{j\ne i}r_j\right)
$$

对 $$j\ne i$$ 有 $$\mathbb{E}[r_j]=\mu$$，因此

$$
\mathbb{E}[\bar r_i]=\frac{1}{G}r_i+\frac{G-1}{G}\mu
$$

于是

$$
\mathbb{E}[r_i-\bar r_i]
=r_i-\frac{1}{G}r_i-\frac{G-1}{G}\mu
=\frac{G-1}{G}(r_i-\mu)
=\left(1-\frac{1}{G}\right)(r_i-\mu)
$$

即 (18.5)。留一法基线为

$$
b_i^{(-i)}=\frac{1}{G-1}\sum_{j\ne i}r_j
$$

由于每个 $$r_j$$ 的期望都是 $$\mu$$，所以 $$\mathbb{E}[b_i^{(-i)}]=\mu$$，优势 $$r_i-b_i^{(-i)}$$ 的期望为 $$r_i-\mu$$，无偏。

若所有奖励相同，设为 $$c$$，则 $$\bar r=c$$，$$r_i-\bar r=0$$，无论是否除以标准差，优势都为零。混合组要求组内既有正确（奖励 1）又有错误（奖励 0）。单条回答正确率为 $$p$$ 时，全对概率为 $$p^G$$，全错概率为 $$(1-p)^G$$，所以

$$
P_{\rm mixed}=1-p^G-(1-p)^G
$$

当 $$G=1$$ 时，$$P_{\rm mixed}=1-p-(1-p)=0$$，即单样本组永远不可能混合，优势恒为零。

</details>

**编程题**：在 `examples/ch18_grpo.py` 中加入动态采样：过滤 `zero_variance_groups` 标记的组，并重新采样提示直到凑满批大小；比较开启前后每个梯度步的有效样本数、零方差组比例与最终贪心正确率。注意“过滤后丢弃”与“过滤后补采”是两种不同实现，前者会减少更新次数，后者才保持批大小不变。

**开放题**：设计一个实验区分“RLVR 教会了模型新推理”与“RLVR 只是重加权了基座模型已有的能力”。要求至少包含：基座模型与 RL 模型在 pass@1 与 pass@k（$$k$$ 取 1、8、64）上的对照；对回答长度与格式的匹配控制；一个与训练分布不同的留出任务；以及对验证器覆盖范围的讨论。说明你的实验在什么结果下支持“新能力”，在什么结果下支持“重加权”。
