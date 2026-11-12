# 第 2 章 分词：模型看到的世界

## 本章导读

同样一句话，在不同模型里可能被切成数量相差很大的 Token；API 按 Token 计费，上下文窗口按 Token 计量，模型数不清 “strawberry” 里有几个 r，也与分词有关。本章讨论这个常被当作预处理、却决定了模型输入空间与经济成本的环节。

## 难度标签

⭐⭐ 基础 ｜ 数学要求：概率论、信息论初步 ｜ 代码要求：Python 基础

## 学习目标

读完本章你将能够：

- 形式化分词器，并说明“无损可逆”究竟要求什么；
- 推导 BPE 的合并准则、词表规模关系与贪心性质；
- 比较 BPE、WordPiece、Unigram 与字节级 BPE 的取舍；
- 分析词表大小在参数量、序列长度与计算量之间的权衡；
- 实现字节级 BPE，并用字节/Token 等指标评估分词器。

## 2.1 模型看不到文字

把一个字符串送进语言模型，实际发生的是这样一条流水线：

```mermaid
flowchart LR
    A["原始文本<br/>字符串"] --> B["UTF-8 字节"]
    B --> C["分词器<br/>encode"]
    C --> D["Token ID 序列"]
    D --> E["嵌入矩阵<br/>查表"]
    E --> F["Transformer"]
    F --> G["输出层<br/>词表上的分布"]
    G --> H["下一个 Token ID"]
    H --> I["分词器<br/>decode"]
    I --> J["文本"]
```

模型的第一层是一个查表操作：第 $$t$$ 个 Token 的 ID 为 $$z_t$$，它的向量是嵌入矩阵 $$\mathbf{E}$$ 的第 $$z_t$$ 行。

$$
\mathbf{e}_t = \mathbf{E} z_t
\qquad \text{(2.1)}
$$

这里的 $$z_t$$ 只是一个整数，它的大小没有语义，顺序也没有含义。Token 5 和 Token 6 之间不存在“更接近”的关系；所有语义都来自训练中学会的嵌入向量。换句话说，**模型看到的不是文字，而是一串被分词器（tokenizer）编好号的符号**。

这带来三个直接后果。

第一，**输入粒度被固定了**。模型能区分的最小单位是 Token，不是字符。如果 “strawberry” 被切成 `s | tra | w | b | er | r | y`，模型看到的是 7 个符号；字母 r 藏在 `tra` 和 `er` 内部，不会单独出现。模型当然可以通过嵌入和注意力间接推断拼写，但这不是它的原生视角。

第二，**Token 成了计费与容量的单位**。上下文窗口 128K 指的是 128K 个 Token，不是 128K 个字符或单词。同样的内容，分词效率高的模型占用的窗口更少、推理更便宜；分词效率低的模型可能提前撞上长度上限。

第三，**分词器是模型身份的一部分**。嵌入矩阵的行数等于词表（vocabulary）大小，行的顺序由 Token ID 决定。换一个分词器，Token ID 的含义就变了，原来的嵌入和输出层全部失效。所以分词器一旦训练完成就被冻结，和模型权重一起发布、一起版本管理。

可以把分词器想成一副固定分辨率的眼镜：它决定了模型能看清多细的纹理，也决定了同样一片风景会占用多少视野。换眼镜不是不能做，但需要重新训练模型的眼睛。

{% hint style="info" %}
“分词”这个词容易让人以为单位是“词”。现代大模型的分词单位通常是子词（subword）甚至字节（byte），一个 Token 可能是一个词、半个词、一个汉字、几个字节，或者一个标点。后文统一用“Token”，不把它等同于“词”。
{% endhint %}

## 2.2 数学原理

### 2.2.1 分词器的形式化

设字符集为 $$\Sigma$$，通常是 UTF-8 字节的集合。词表（vocabulary）$$\mathcal{V}$$ 是一个有限的字符串集合，其中的元素称为 Token，每个 Token 是若干字节组成的字符串。分词器由编码器（encoder）$$\mathrm{Enc}$$ 和解码器（decoder）$$\mathrm{Dec}$$ 组成。

$$
\mathrm{Enc}:\Sigma^{*}\to \mathcal{V}^{*},\qquad \mathrm{Dec}:\mathcal{V}^{*}\to\Sigma^{*}
\qquad \text{(2.2)}
$$

其中 $$\Sigma^{*}$$ 表示任意长度的字节串，$$\mathcal{V}^{*}$$ 表示任意长度的 Token 序列。**无损可逆**要求对任意输入串 $$s$$ 都有

$$
\mathrm{Dec}(\mathrm{Enc}(s))=s
\qquad \text{(2.3)}
$$

这个条件看起来理所当然，实际很容易被破坏。很多分词器会先做 Unicode 规范化（例如 NFKC）、转小写、压缩连续空白；这些操作让编码更“干净”，却让解码无法还原原始字节。字节级 BPE（byte-level BPE）不做规范化，因此可以严格满足 (2.3)。

另一个容易忽略的性质是**非组合性**：把两个字符串拼起来再分词，结果不一定等于分别分词后拼接。例如 “a” 和 “b” 单独编码时可能是两个 Token，拼成 “ab” 后可能被合并成一个 Token。这意味着不能假设 Token 边界在拼接下保持不变；在做提示拼接、约束解码或缓存前缀时，必须对完整字符串重新分词。

分词器的压缩效果可以用压缩比（compression ratio）衡量。

$$
r=\frac{B_{\text{byte}}}{T}\quad\text{（字节/Token）}
\qquad \text{(2.4)}
$$

其中 $$B_{\text{byte}}$$ 是输入文本的 UTF-8 字节数，$$T$$ 是编码后的 Token 数。$$r$$ 越大，同样长度的文本占用的 Token 越少。另一个常用指标是每个字符的 Token 数 $$T/n_{\text{char}}$$，其中 $$n_{\text{char}}$$ 是字符数；它和 $$r$$ 一起，能刻画不同语言之间的成本差异。

### 2.2.2 BPE：从字符对到子词

字节对编码（Byte Pair Encoding，BPE）最初是一种数据压缩算法，1994 年由 Gage 提出；2016 年 Sennrich 等人把它引入神经机器翻译，用来解决稀有词和未登录词（out-of-vocabulary，OOV）问题。

算法从一个基础符号表开始。字节级 BPE 的基础表是 256 个字节；字符级 BPE 的基础表是语料中出现过的字符。把训练语料表示成符号序列 $$q^{(0)}=(q_1,\dots,q_{T_0})$$，然后重复以下三步：

1. 统计所有相邻符号对的频次。

   $$
   f_t(a,b)=\sum_{i=1}^{T_t-1}\mathbf{1}\left[q_i^{(t)}=a,\ q_{i+1}^{(t)}=b\right]
   \qquad \text{(2.5)}
   $$

2. 选择频次最高的对 $$(a^{*},b^{*})=\arg\max_{a,b} f_t(a,b)$$，把新符号 $$a^{*}b^{*}$$ 加入词表；
3. 从左到右、不重叠地把序列中所有 $$(a^{*},b^{*})$$ 替换成新符号。

重复 $$M$$ 次后，词表规模为

$$
|\mathcal{V}|=|\mathcal{A}|+M+S
\qquad \text{(2.6)}
$$

其中 $$|\mathcal{A}|$$ 是基础符号数（字节级为 256），$$S$$ 是特殊 Token（special token）数。注意这是**实际**词表规模；如果训练目标设为 768、特殊 Token 有 2 个，最终可能是 770。如果语料太小，某个时刻所有相邻对的频次都小于 2，合并会提前停止，实际词表也会小于目标值。第 2.4 节的实验会出现这两种情况。

BPE 的训练目标可以写成最小化最终序列长度。

$$
T_M=T_0-\sum_{t=1}^{M}n_t,\qquad n_t\le f_t(a_t,b_t)
\qquad \text{(2.7)}
$$

其中 $$n_t$$ 是第 $$t$$ 步实际合并的不重叠出现次数。由于 $$n_t$$ 最多等于频次 $$f_t$$，每一步选择最高频的对，是在贪心地最大化当前的长度缩减。**贪心不保证全局最优**：某个高频合并可能阻断后续更有利的组合，但精确求解最优合并序列的代价很高，实践中没有人这样做。BPE 的价值不在于最优，而在于简单、确定、可增量执行。

编码时不需要重新统计频次，只需要记住合并的顺序。给每个合并对一个优先级（越早合并优先级越高），然后反复合并当前序列中优先级最高的相邻对，直到没有可合并的对。这个过程的直观含义是“按训练时学到的顺序还原压缩过程”。朴素实现每轮扫描整个序列，最坏复杂度为 $$\mathcal{O}(T^2)$$；生产实现会用优先队列、双向链表和按词缓存，把常见词的编码降到接近线性。

BPE 还有一个重要细节：**词边界**。字符级 BPE 通常在词尾加一个结束符（如 `</w>`），防止跨词合并；字节级 BPE 不需要，因为空格本身就是一个字节，会被当作普通符号参与合并。GPT-2 的词表中经常看到带 `Ġ` 的 Token，`Ġ` 就是空格字节经过映射后的可打印形式。

### 2.2.3 WordPiece：用点互信息挑合并

WordPiece 的外观和 BPE 很像：也是从基础符号开始，不断合并相邻对。区别在**选择准则**。BPE 选频次最高的对，WordPiece 选得分最高的对，得分为

$$
\mathrm{score}(a,b)=\frac{f(ab)}{f(a)\,f(b)}
\qquad \text{(2.8)}
$$

其中 $$f(ab)$$ 是符号串 $$ab$$ 作为一个整体的出现次数，$$f(a)$$、$$f(b)$$ 分别是 $$a$$、$$b$$ 单独出现的次数。设语料中共有 $$N_{\text{pair}}$$ 个相邻对位置，则 $$f(ab)=N_{\text{pair}}\,P(ab)$$、$$f(a)=N_{\text{pair}}\,P(a)$$、$$f(b)=N_{\text{pair}}\,P(b)$$，于是

$$
\mathrm{score}(a,b)=\frac{1}{N_{\text{pair}}}\cdot\frac{P(ab)}{P(a)P(b)}=\frac{1}{N_{\text{pair}}}e^{\mathrm{PMI}(a,b)}
\qquad \text{(2.9)}
$$

其中 $$\mathrm{PMI}(a,b)=\ln\frac{P(ab)}{P(a)P(b)}$$ 是点互信息（pointwise mutual information，PMI）。因此，**最大化 WordPiece 得分等价于最大化两个符号的点互信息**：它奖励“共同出现得比偶然预期更频繁”的对，而不只是奖励高频对。

这个区别在自然语言里很实际。字母 `t` 和 `h` 都很常见，`th` 也很常见，但 `th` 的高频可能只是因为 `t`、`h` 本身高频；PMI 会把这种“由边缘频率带来的虚假高频”除掉，更倾向于发现真正的搭配。反过来，BPE 的纯频次准则会优先合并最常见的字符组合，这在压缩上合理，在语言学上未必最有意义。

WordPiece 的编码通常采用**最长匹配**（longest-match）：从左到右，每次尝试匹配词表中最长的子串；非词首的子词加 `##` 前缀（例如 `playing` 切成 `play` 和 `##ing`）。BERT 使用的就是 30K 词表的 WordPiece。最长匹配是贪心的，也不保证概率意义上的最优切分。

### 2.2.4 Unigram：先假设一个概率模型

BPE 和 WordPiece 都是自底向上的合并；Unigram 语言模型（Unigram Language Model）反过来，自顶向下地删减。

它假设每个 Token $$u$$ 有一个概率 $$p(u)$$，且所有 Token 概率之和为 1。一个字符串被切成 Token 序列 $$u=(u_1,\dots,u_K)$$ 的概率是

$$
P(u)=\prod_{i=1}^{K}p(u_i)
\qquad \text{(2.10)}
$$

训练目标是最大化语料在所有可能切分下的边际似然。由于切分是隐变量，通常用期望最大化（expectation-maximization，EM）算法：E 步用前向-后向算法计算每个 Token 的期望出现次数，M 步用期望频次重新估计 $$p(u)$$。训练完一轮后，按照“移除该 Token 后似然下降多少”给 Token 排序，删掉最不重要的一批，再继续训练。如此反复，直到词表缩到目标大小。

编码时，给定一个字符串 $$s$$，要找概率最大的切分。设 $$\mathrm{best}[i]$$ 是前缀 $$s_{0:i}$$ 的最优对数概率，则

$$
\mathrm{best}[i]=\max_{0\le j<i,\ s_{j:i}\in \mathcal{V}}\left(\mathrm{best}[j]+\log p(s_{j:i})\right)
\qquad \text{(2.11)}
$$

这就是 Viterbi 算法，复杂度为 $$\mathcal{O}(T\cdot L_{\max})$$，其中 $$L_{\max}$$ 是最长 Token 的长度。与 BPE 的确定性编码不同，Unigram 给出的是一个概率分布，因此可以在训练时对切分采样，这就是 Kudo 提出的**子词正则化**（subword regularization）：同一个句子每次以不同方式切分，相当于一种数据增强。

SentencePiece 同时支持 BPE 和 Unigram 两种算法；T5、ALBERT 等模型使用 SentencePiece 词表，具体采用哪种算法以各自发布的分词器配置为准。Unigram 的优势是概率解释清晰、天然支持多切分；代价是训练比 BPE 复杂，编码也需要动态规划。

### 2.2.5 字节级 BPE：用 256 个字节兜底

字符级分词有一个绕不开的问题：没见过的字符怎么办？传统做法是引入 `<unk>`，但 `<unk>` 丢失了信息，而且不同语言的未登录率差异很大。字节级 BPE 的做法很彻底：**基础词表就是 256 个字节，任何 UTF-8 字符串都能表示，不存在未登录**。

对任意一个字符 $$c$$，设它的 UTF-8 编码占 $$n_{\text{byte}}(c)$$ 个字节。单独编码时，字节级 BPE 的 Token 数满足

$$
1\le |\mathrm{Enc}(c)|\le n_{\text{byte}}(c)
\qquad \text{(2.12)}
$$

下界是因为非空字符至少需要一个 Token；上界是因为最坏情况下每个字节各占一个 Token，而合并只会减少数量。ASCII 字符的 $$n_{\text{byte}}(c)=1$$，所以恰好是一个 Token；常见汉字的 $$n_{\text{byte}}(c)=3$$，可能被合并成一个 Token，也可能保持两三个；四字节的 emoji 同理。这个界说明了一件事：**字节级 BPE 的覆盖率是完备的，但非拉丁文字天然更“费 Token”**，除非词表里专门为这些字符学习了合并。

GPT-2 是字节级 BPE 的代表。它先把 256 个字节映射到 256 个可打印的 Unicode 字符（`bytes_to_unicode`），再在这个字符空间上做 BPE。这个映射是一个双射，不改变信息，只是让 Token 在调试和可视化时更可读；空格字节被映射成 `Ġ`，换行等控制字符也有对应的可见形式。因为映射是双射，解码时可以逐字节还原，严格满足 (2.3)。

SentencePiece 走了另一条工程路线：它直接把原始文本（包括空格）当作输入流，用 `▁` 表示空格，不依赖预先的字符切分。这样分词器与语言无关，中文、日文、泰文都不需要先做分词。SentencePiece 可以训练 BPE，也可以训练 Unigram，Llama 2 使用的是基于 SentencePiece 的 BPE。

### 2.2.6 词表大小的权衡

词表大小 $$n_{vocab}$$ 是分词器最重要的超参数。它同时影响参数、计算和数据效率，而且方向相反。

**参数侧**：输入嵌入矩阵 $$\mathbf{E}\in\mathbb{R}^{n_{vocab}\times d_{model}}$$ 有 $$n_{vocab}d_{model}$$ 个参数；如果输出层不共享权重，还有一个同样大小的矩阵，于是

$$
N(n_{vocab})=N_0+2n_{vocab}d_{model}
\qquad \text{(2.13)}
$$

其中 $$N_0$$ 是其余参数。以 $$n_{vocab}=128\text{K}$$、$$d_{model}=4096$$ 为例，单个矩阵约 5.24 亿参数，两个约 10.5 亿。对一个 8B 模型，这已经超过总参数的 13%；而 AdamW 的优化器状态（一阶矩、二阶矩和 fp32 主权重）按每个参数十几字节计算，仅嵌入层的参数与优化器状态就可能超过 15GB。这就是为什么大词表模型的嵌入层在分布式训练中往往要单独切分。

**计算侧**：设语料总字节数为 $$B_{\text{byte}}$$，平均每个 Token 覆盖 $$r(n_{vocab})$$ 个字节，则 Token 总数为 $$D=B_{\text{byte}}/r(n_{vocab})$$。训练计算量近似为

$$
C(n_{vocab})\approx 6N(n_{vocab})\frac{B_{\text{byte}}}{r(n_{vocab})}
\qquad \text{(2.14)}
$$

词表越大，$$r(n_{vocab})$$ 通常越大，Token 数越少，注意力与全连接的计算量随之下降；但 $$N(n_{vocab})$$ 线性增长，输出层 softmax 的代价也随 $$n_{vocab}$$ 线性增长。两条曲线的斜率决定了最优点，而这个最优点取决于语料的语言分布、模型大小、上下文长度和推理预算，没有一个通用答案。

**数据侧**：词表越大，长尾 Token 越多。很多 Token 在训练语料中只出现几次，嵌入得不到充分训练；它们占用了参数，却没有学到稳定的表示。这也是“词表越大越好”不成立的原因之一。

经验上，2023 年前后的英文中心模型常用 32K 词表（如 Llama 2）；截至写作时（2026 年 11 月），面向多语言和代码的模型普遍转向 128K 甚至更大（Llama 3 为 128K，Qwen 系列约 15 万，DeepSeek-V3 为 128K）。这个迁移的主要驱动力不是英文，而是中文、日文、代码和符号的压缩效率：更大的词表能让这些内容少占 Token，从而在同样的上下文窗口里放下更多信息。

### 2.2.7 如何评价一个分词器

单看词表大小没有意义，需要看它在真实文本上的表现。常用指标有四个。

**每个 Token 的字节数**，即 (2.4) 的 $$r$$。它是最直接的压缩效率指标。

**生育率**（fertility）指每个词被切成多少个 Token。

$$
\mathrm{fertility}=\frac{T}{n_{\text{word}}}
\qquad \text{(2.15)}
$$

其中 $$n_{\text{word}}$$ 是词数。英语的 fertility 通常在 1.2 到 1.5 之间，中文按“词”统计时差异更大，因为中文的词边界本身就有歧义。跨语言比较时，更稳妥的指标是每个字符的 Token 数。

**每字节比特数**（bits per byte）。第 1 章说过，交叉熵 $$\mathcal{L}$$ 是每个 Token 的平均负对数似然（单位 nat）。把它换算成每字节的编码长度。

$$
\rho=\frac{\mathcal{L}}{\ln 2\cdot r}
\qquad \text{(2.16)}
$$

其中 $$r$$ 是平均每个 Token 的字节数，即 (2.4) 定义的压缩比。$$\rho$$ 才是可以跨分词器比较的压缩质量指标。

**未登录率**，即 `<unk>` 占所有 Token 的比例。字节级分词器的未登录率恒为 0；词级或字符级分词器在跨领域、跨语言时可能很高。

这里有一个必须记住的结论：**困惑度不能跨分词器比较**。假设分词器 1 平均每个 Token 覆盖 $$r_1$$ 字节，分词器 2 覆盖 $$r_2$$ 字节，两者在同一文本上的每字节比特数相同，记为 $$\rho$$，则 $$\mathcal{L}_1=\ln 2\cdot r_1\rho$$，$$\mathcal{L}_2=\ln 2\cdot r_2\rho$$。

于是 $$\mathrm{PPL}_1=e^{\mathcal{L}_1}=e^{\ln 2\cdot r_1\rho}=2^{r_1\rho}$$，$$\mathrm{PPL}_2=2^{r_2\rho}$$，由此得到

$$
\mathrm{PPL}_1=\mathrm{PPL}_2^{\,r_1/r_2}
\qquad \text{(2.17)}
$$

如果分词器 1 的 Token 平均长度是分词器 2 的两倍，即 $$r_1=2r_2$$，那么 $$\mathrm{PPL}_1=\mathrm{PPL}_2^2$$。也就是说，**同样质量的两个模型，Token 更长的那个困惑度会呈指数级更高**。看到“某模型 PPL 只有 5”时，必须先问它用的是什么分词器；否则这个数字没有可比性。

{% hint style="danger" %}
跨分词器比较 PPL 是文献和博客里最常见的错误之一。若要比较两个分词器不同的模型，请把每 Token 交叉熵换算成每字节比特数 (2.16)，并在同一份文本上计算；否则 Token 更长的模型会显得“困惑度更高”，Token 更短的模型会显得“困惑度更低”，结论可能完全反过来。
{% endhint %}

用第 2.4 节的数据做一个算术示例：中文的平均 Token 长度是 2.38 字节，英文是 1.83 字节。如果两者每字节比特数相同，那么中文的 PPL 约是英文的 $$\mathrm{PPL}_{en}^{\,2.38/1.83}\approx \mathrm{PPL}_{en}^{1.30}$$，即在指数上高出约 30%。这不是“中文更难”，而是“中文的每个 Token 装了更多字节”。

## 2.3 方案对比

| 方法 | 基础单元 | 选择准则 | 编码方式 | 未登录 | 典型词表 | 代表 |
| --- | --- | --- | --- | --- | --- | --- |
| 词级 | 词 | 词典匹配 | 最长匹配 | 有 | 几十万 | 早期统计机器翻译 |
| 字符级 | Unicode 字符 | 无 | 直接映射 | 有（未见字符） | 几万 | 早期中文模型 |
| BPE | 字符 | 最高频相邻对 | 按合并顺序贪心 | 有 | 32K | 早期 NMT |
| WordPiece | 字符 | 最高 PMI 近似分数 | 最长匹配 | 有 | 30K | BERT |
| Unigram | 子词 | 最大化似然后剪枝 | Viterbi 动态规划 | 有 | 32K | T5、ALBERT |
| 字节级 BPE | 字节 | 最高频相邻对 | 按合并顺序贪心 | 无 | 50K–200K | GPT 系列、Llama 3 |

这张表里没有“全面最优”的一行。词级分词最符合人的直觉，但词表爆炸、未登录严重；字符级没有词表爆炸，但序列太长；BPE 和 WordPiece 在压缩与泛化之间取平衡；Unigram 有概率解释，但训练和编码更复杂；字节级 BPE 用“序列变长”换来了“永不未登录”。

选型的核心是三个问题：**目标语言是什么、模型有多大、推理成本有多敏感**。英文为主、模型较小、追求简单，32K 的 BPE 足够；多语言、代码、长上下文，128K 以上的字节级 BPE 更合适；需要概率化切分或子词正则化，Unigram 更自然。

## 2.4 代码实现：字节级 BPE

### 2.4.1 实现选择

下面实现一个完整的字节级 BPE，接口与附录 F 的 `Tokenizer` 协议一致：

- 基础词表是 256 个字节，经 GPT-2 的 `bytes_to_unicode` 映射成可打印字符；
- 训练时反复统计相邻对、合并最高频对，频次低于 2 时停止；
- 编码时按合并优先级（rank）从小到大贪心合并；
- 支持 `<bos>`、`<eos>` 等特殊 Token，特殊 Token 不参与普通合并；
- 用 JSON 保存词表、合并序列和特殊 Token，加载后分词结果完全一致。

代码放在 `tinylm/tokenizer/bpe.py`。为了让本章可以独立运行，文件末尾附带一个小型语料上的演示；正式使用时只导入 `BPETokenizer` 类。代码中的局部变量 `V` 表示目标词表大小，对应正文的 $$n_{vocab}$$；数学记号里的粗体 V 专指注意力的值矩阵，不用于词表。

### 2.4.2 完整代码

```python
"""第 2 章：字节级 BPE 分词器（tinylm/tokenizer/bpe.py 的核心实现）。"""
import json
import re
from collections import Counter


def bytes_to_unicode():
    """把 256 个字节映射到可打印的 Unicode 字符（GPT-2 的做法）。"""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}


class BPETokenizer:
    def __init__(self, vocab=None, merges=None, special_tokens=None):
        self.vocab = vocab or {BYTE_ENCODER[b]: b for b in range(256)}
        self.merges = merges or []
        self.special_tokens = special_tokens or []
        self._build_ranks()

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _build_ranks(self):
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.id_to_token = {i: t for t, i in self.vocab.items()}

    @staticmethod
    def _merge(symbols, pair, merged):
        """从左到右、不重叠地替换一个相邻对。"""
        out, i = [], 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return out

    def train(self, text, vocab_size, special_tokens=None):
        assert vocab_size >= 256
        symbols = [BYTE_ENCODER[b] for b in text.encode("utf-8")]
        self.vocab = {BYTE_ENCODER[b]: b for b in range(256)}
        self.merges = []
        while len(self.vocab) < vocab_size:
            # 公式 (2.5)：统计相邻对频次
            counts = Counter(zip(symbols, symbols[1:]))
            if not counts:
                break
            # 公式 (2.7)：贪心选择当前最高频的对
            pair = max(counts, key=lambda p: (counts[p], p))
            if counts[pair] < 2:
                break
            merged = pair[0] + pair[1]
            self.vocab[merged] = len(self.vocab)          # 公式 (2.6)
            self.merges.append(pair)
            symbols = self._merge(symbols, pair, merged)
        self.special_tokens = special_tokens or []
        for tok in self.special_tokens:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        self._build_ranks()
        return self

    def _encode_ordinary(self, text):
        symbols = [BYTE_ENCODER[b] for b in text.encode("utf-8")]
        while len(symbols) >= 2:
            # 按训练时的合并顺序，优先执行 rank 最小的对
            pairs = set(zip(symbols, symbols[1:]))
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            symbols = self._merge(symbols, best, best[0] + best[1])
        return symbols

    def encode(self, text, *, add_special=True):
        ids = []
        if add_special and "<bos>" in self.vocab:
            ids.append(self.vocab["<bos>"])
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.vocab[part])
            elif part:
                ids.extend(self.vocab[s] for s in self._encode_ordinary(part))
        return ids

    def decode(self, ids, *, skip_special=True):
        pieces = []
        for i in ids:
            tok = self.id_to_token[i]
            if skip_special and tok in self.special_tokens:
                continue
            pieces.append(tok)
        data = bytes(BYTE_DECODER[ch] for ch in "".join(pieces))
        return data.decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab,
                       "merges": [list(p) for p in self.merges],
                       "special_tokens": self.special_tokens},
                      f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        tok = cls(vocab=d["vocab"],
                  merges=[tuple(p) for p in d["merges"]],
                  special_tokens=d["special_tokens"])
        tok._build_ranks()
        return tok


CORPUS = """
语言模型通过预测下一个词元来学习文本的统计规律。分词器把字符串切分成词元序列，是模型看到文本的第一步。
中文没有空格，分词器需要自己决定一个汉字或一个词应该占几个词元。英文单词之间用空格分隔，但标点和大小写仍然需要处理。
代码中的缩进、变量名和运算符对分词器提出了额外的要求。数字串经常被切成不规则的片段，这会影响算术能力。
Natural language processing studies how computers understand and generate human language. Tokenization is the first step of the pipeline.
Large language models are trained on trillions of tokens. The vocabulary size controls the trade-off between sequence length and embedding parameters.
Tokenizer design affects cost, context length, and even arithmetic. A Chinese character is three bytes in UTF-8, so byte-level BPE may need several merges to represent it.
def add(a, b):
    return a + b

for i in range(10):
    print(i, i ** 2)
2024年，模型的参数量从70亿增长到6710亿。上下文窗口从8千扩展到12.8万词元。
""" * 4


def main():
    tok = BPETokenizer().train(
        CORPUS, vocab_size=768, special_tokens=["<bos>", "<eos>"]
    )
    print(f"训练后: vocab_size={tok.vocab_size}  merges={len(tok.merges)}")
    print("往返一致:", tok.decode(tok.encode(CORPUS, add_special=False)) == CORPUS)

    test_zh = "大语言模型的分词器决定了文本被切成多少个词元。"
    test_en = "Tokenization determines how many tokens the model sees."
    for name, s in [("中文", test_zh), ("英文", test_en)]:
        ids = tok.encode(s, add_special=False)
        b = len(s.encode("utf-8"))
        print(f"{name}: chars={len(s)} bytes={b} tokens={len(ids)} "
              f"bytes/token={b / len(ids):.2f}")

    for s in ["strawberry", "1234567890", "def add(a, b):"]:
        ids = tok.encode(s, add_special=False)
        pieces = [tok.id_to_token[i] for i in ids]
        print(f"{s!r}: tokens={len(ids)} pieces={pieces}")

    print("\n词表大小对 Token 数的影响:")
    for V in [256, 384, 512, 768, 1024]:
        t = BPETokenizer().train(CORPUS, vocab_size=V)
        n_zh = len(t.encode(test_zh, add_special=False))
        n_en = len(t.encode(test_en, add_special=False))
        print(f"V_target={V:4d} V_actual={t.vocab_size:4d} "
              f"merges={len(t.merges):4d} zh={n_zh:3d} en={n_en:3d}")


if __name__ == "__main__":
    main()
```

### 2.4.3 实验结果

在普通 CPU 上用 Python 3.13 运行，输出为：

```text
训练后: vocab_size=770  merges=512
往返一致: True
中文: chars=23 bytes=69 tokens=29 bytes/token=2.38
英文: chars=55 bytes=55 tokens=30 bytes/token=1.83
'strawberry': tokens=7 pieces=['s', 'tra', 'w', 'b', 'er', 'r', 'y']
'1234567890': tokens=10 pieces=['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
'def add(a, b):': tokens=11 pieces=['de', 'f', 'Ġa', 'd', 'd', '(', 'a', ',', 'Ġb', ')', ':']

词表大小对 Token 数的影响:
V_target= 256 V_actual= 256 merges=   0 zh= 69 en= 55
V_target= 384 V_actual= 384 merges= 128 zh= 33 en= 31
V_target= 512 V_actual= 512 merges= 256 zh= 29 en= 30
V_target= 768 V_actual= 768 merges= 512 zh= 29 en= 30
V_target=1024 V_actual=1006 merges= 750 zh= 29 en= 30
```

整理关键指标：

| 指标 | 中文 | 英文 |
| --- | --- | --- |
| 字符数 | 23 | 55 |
| UTF-8 字节数 | 69 | 55 |
| Token 数 | 29 | 30 |
| 字节/Token | 2.38 | 1.83 |
| Token/字符 | 1.26 | 0.55 |

### 2.4.4 结果说明了什么

**第一，往返完全一致。** 字节级 BPE 没有未登录问题，`decode(encode(s)) == s` 对这段混合了中文、英文和代码的语料成立。这不是因为 BPE 聪明，而是因为 256 个字节的基础词表保证了任何 UTF-8 串都有表示，合并只是改变了分组方式。

**第二，中文和英文的成本差异不是简单的“谁更贵”。** 在这份小语料上，中文每个字符 1.26 个 Token，英文每个字符 0.55 个 Token；但按字节算，中文每个 Token 覆盖 2.38 字节，英文只有 1.83 字节。原因是常用汉字的 UTF-8 编码通常占 3 字节，词表里又学到了不少常用汉字的合并；英文虽然每字符 1 字节，但这份语料里的英文单词很多没有被充分合并，所以每 Token 只覆盖 1.83 字节。**绝对数字依赖于语料和词表大小，不能直接外推到生产分词器**；能外推的是“成本由分词器和语料共同决定”这个结论。

**第三，`strawberry` 被切成 7 个 Token。** 词表里没有这个单词，BPE 只能用它学过的片段拼出来：`s | tra | w | b | er | r | y`。模型看到的是这 7 个 ID，字母 r 并不以独立符号出现；要数清 r 的个数，模型必须从 Token 的内部拼写中推断。这是“模型数不清字母”的一个结构性原因，但不是唯一原因——大模型可以通过推理弥补，只是要多花一些计算。

**第四，数字被切得很碎。** `1234567890` 变成 10 个单字符 Token，因为这份语料里的数字太少，没有形成稳定的数字合并。数字切分的不一致会伤害算术：同一个数在不同上下文里可能被切成不同片段，模型很难学到“位”的概念。Llama 3 的技术报告明确说，它把数字串拆成单个数字，就是为了让切分一致；代价是数字更费 Token。

**第五，词表增大有边际递减。** 从 256 增到 384，中文从 69 个 Token 降到 33 个，收益很大；384 到 512 只降到 29；512 之后在这份语料上不再下降。原因很简单：语料太小，能支持的高频合并已经用完了。目标 1024 时实际只到 1006，是因为剩下的相邻对频次都小于 2，训练提前停止。这也说明，**词表大小的收益是数据相关的**，不是把 $$n_{vocab}$$ 调大就自动变好。

**第六，目标词表和实际词表可能不一致。** 目标 768、实际 770，是因为两个特殊 Token 在合并结束后被追加；目标 1024、实际 1006，是因为合并提前停止。工程上必须记录**实际**词表大小，并让模型的嵌入矩阵与之一致，否则加载时会维度不匹配。

{% hint style="warning" %}
这段代码是为了展示算法而写的教学实现，不是生产级分词器。它没有做 Unicode 规范化、没有并行化、没有按词缓存，训练语料也只有几千字节。生产分词器会在数百 GB 的语料上训练，并用 Rust 或 C++ 实现编码器；但核心的合并规则、rank 编码和字节兜底逻辑是一样的。
{% endhint %}

复现方式：把代码保存为 `bpe.py`，运行 `python bpe.py`。若想观察词表大小的影响，修改 `main()` 里的 `vocab_size`；若想验证保存/加载，调用 `tok.save("bpe.json")` 后用 `BPETokenizer.load("bpe.json")` 重新编码同一段文本，比较 ID 序列。

## 2.5 工业实践

### 2.5.1 从 GPT-2 到 GPT-4

GPT-2（Radford et al.，2019）使用 50,257 词表的字节级 BPE，让“永不未登录”成为大模型分词的重要设计目标。它的词表里既有完整的英文单词，也有常见的词片段和单个字节；`Ġ` 前缀表示空格，因此解码时可以精确还原空白。

GPT-4 系列常用的 `cl100k_base` 词表约 10 万，GPT-4o 使用的 `o200k_base` 约 20 万。词表扩大的主要收益不是英文，而是多语言和代码：更大的词表能为中文、日文、Python 关键字、常见运算符分配专门的 Token，减少这些内容的 Token 数。OpenAI 通过 `tiktoken` 库公开了这些分词器的实现和词表，但不同模型的 Token 计数仍然不能混用。

### 2.5.2 Llama 的两次选择

Llama 2 使用 32K 词表的 SentencePiece BPE，训练语料以英文为主。Llama 3 把词表扩大到 128K，技术报告给出的估计是：在英文上比 Llama 2 的分词器最多可少用约 15% 的 Token。这个数字看起来不大，但在万亿 Token 的预训练规模上，意味着同样的计算预算可以覆盖更多文本，或者在同样文本上少做 15% 的计算。

Llama 3 还做了一个有意的设计：把数字串拆成单个数字。这样“123”和“124”的切分方式一致，模型更容易学习按位运算；代价是数字串变长。这个例子说明，分词器的目标不只是“压缩率最高”，还要考虑下游任务需要什么样的不变性。

### 2.5.3 多语言与词表扩展

Qwen 系列使用约 15 万词表，DeepSeek-V3 使用 128K 词表，都是为了让中文、代码和多语言内容获得更好的压缩率。对于以中文为主的模型，词表里必须包含足够多的常用汉字和中文词片段；否则一个汉字要占两三个 Token，上下文窗口和推理成本都会吃亏。

在领域适配时，常见做法是**词表扩展**（vocabulary expansion）：在原有词表上加入领域专有 Token（例如基因序列、医学缩写、新的编程语言关键字），然后把模型的嵌入矩阵和输出层扩大相应的行数。新行通常用旧行的均值或小随机数初始化，再继续预训练；训练时可以冻结旧行、只训练新行，也可以用较小的学习率微调全部参数。风险是灾难性遗忘和旧 Token 表示的漂移，因此必须用领域数据和通用数据混合训练，并重新评测通用能力。

词表扩展还有一个工程约束：**分词器和模型必须一起换**。只换分词器而不扩嵌入矩阵，模型会遇到越界的 Token ID；只扩嵌入矩阵而不换分词器，新行永远得不到训练。发布时要把分词器文件、词表大小、特殊 Token 列表和模型权重绑定成一个版本，避免出现“权重和分词器不匹配”的隐蔽错误。

### 2.5.4 分词器是 API 契约

在商业 API 里，分词器不只是预处理代码，而是计费、限流和缓存的契约。Token 数决定请求价格、上下文占用和输出长度上限；提示缓存（prompt caching）按 Token 前缀命中，分词边界一变，缓存就可能失效。因此主流厂商不会随意更换已发布模型的分词器，新分词器通常伴随新模型版本出现。

这也解释了为什么“同一个问题，不同模型的 Token 数不同”不是 bug，而是设计。比较价格时，必须用各自的分词器计算 Token 数，再把单价换算成“每百万字符”或“每百万字节”的成本，否则会得出错误结论。

## 2.6 论文导读

**1. Gage, P. (1994). A New Algorithm for Data Compression. *The C Users Journal*.**

核心贡献：提出字节对编码，用反复合并高频相邻字节对的方式压缩数据。推荐阅读：算法的合并流程和替换规则。这是 BPE 的源头；读的时候注意它原本是压缩算法，不是分词算法。

**2. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. *ACL*. arXiv:1508.07909.**

核心贡献：把 BPE 引入神经机器翻译，用子词单元解决稀有词和未登录词。推荐阅读：第 3 节的 BPE 训练与编码细节，以及词表大小对翻译质量的影响实验。这是“子词分词”成为主流的起点。

**3. Wu, Y., et al. (2016). Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation. arXiv:1609.08144.**

核心贡献：系统描述 Google 神经机器翻译系统，其中包含 WordPiece 的训练和编码方法。推荐阅读：分词部分对“为什么用子词而不是词”的讨论，以及词表规模的工程权衡。

**4. Kudo, T. (2018). Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. *ACL*. arXiv:1804.10959.**

核心贡献：提出 Unigram 语言模型分词和子词正则化，用多种切分增强训练。推荐阅读：第 2 节的 Unigram 模型与 EM 训练，第 3 节的 Viterbi 编码和采样切分。对照 BPE 读，能看清“确定性切分”和“概率化切分”的差别。

**5. Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners.**

核心贡献：GPT-2 技术报告，采用字节级 BPE，把词表扩大到 50,257，并证明字节级表示可以避免未登录词。推荐阅读：第 2 节的分词器描述和字节到 Unicode 的映射。它是现代大模型分词器的直接祖先。

## 2.7 常见误区

**误区一：一个 Token 就是一个词或一个汉字。**

Token 是分词器定义的符号，可能是词、词片段、单个字符、多个字节，甚至一个标点。中文的一个汉字可能是一个 Token，也可能是两三个；英文的一个单词可能是一个 Token，也可能是四五个。把 Token 当成词，会高估上下文窗口能容纳的内容，也会误判模型的语言能力。

**误区二：困惑度可以跨分词器比较。**

由 (2.17)，同样压缩质量下，Token 更长的分词器 PPL 呈指数级更高。分词器 1 的 Token 平均长度是分词器 2 的两倍时，$$\mathrm{PPL}_1=\mathrm{PPL}_2^2$$。所以“某模型 PPL 5.0，另一个 10.0”不能直接说明前者更好；必须换算成每字节比特数，或者在同一个分词器下比较。

**误区三：词表越大越好。**

大词表降低序列长度，却增加嵌入和输出层参数、softmax 计算量和长尾 Token 的稀疏性。一个 128K 词表的 1B 小模型，可能有 20% 以上的参数花在嵌入上，而这些参数在少量数据上训练不充分。词表大小必须和模型规模、数据量、语言分布匹配；小模型用超大词表通常是浪费。

**误区四：字节级 BPE 没有未登录，所以全面更优。**

字节级 BPE 的覆盖率确实完备，但代价是非拉丁文字的序列更长。一个三字节的汉字最多可能占三个 Token；如果词表没有为它学习合并，成本就是字符级分词的三倍。字节级 BPE 解决的是“能不能表示”，不是“表示得省不省”；后者要靠词表设计。

**误区五：模型看到的是字母，所以拼写任务应该很容易。**

模型看到的是 Token ID，字母藏在 Token 内部。`strawberry` 被切成 `s | tra | w | b | er | r | y` 后，“r 出现了几次”需要模型从多个 Token 的内部结构推断，而不是逐字符扫描。分词不是拼写困难的唯一原因，但它是结构性的原因之一。类似的还有押韵、字符级替换、精确计数等任务。

## 2.8 本章小结

- 分词器把字符串映射成 Token 序列，模型看到的是 Token ID，不是文字；分词器一旦训练完成就被冻结，是模型身份的一部分。
- 无损可逆要求 $$\mathrm{Dec}(\mathrm{Enc}(s))=s$$；Unicode 规范化、转小写等预处理会破坏这个性质，字节级 BPE 不做规范化，因此可以严格还原。
- BPE 反复合并最高频相邻对，词表规模为 $$|\mathcal{V}|=|\mathcal{A}|+M+S$$；训练是贪心的，不保证全局最优，但简单、确定、可增量执行。
- WordPiece 用 $$f(ab)/(f(a)f(b))$$ 选择合并，等价于最大化点互信息；Unigram 用概率模型和 EM 训练，用 Viterbi 编码，支持多种切分。
- 字节级 BPE 用 256 个字节兜底，任意 UTF-8 串都不会未登录；代价是非拉丁文字的 Token 数可能接近字节数。
- 词表大小在嵌入参数量、序列长度、注意力计算和长尾稀疏性之间权衡，没有通用最优值；多语言和代码场景推动了 128K 以上词表的普及。
- 困惑度不能跨分词器比较；应换算成每字节比特数 (2.16)。同样质量下，Token 更长的分词器 PPL 呈指数级更高 (2.17)。

{% hint style="success" %}
**本章的取舍**：字节级 BPE 用“非拉丁文字序列变长”换来了“永不未登录、可严格还原”；大词表用“嵌入参数和 softmax 开销”换来了“更短的序列和更好的多语言压缩”；WordPiece 用“PMI 准则”换来了“更有语义的合并”，代价是编码要做最长匹配；Unigram 用“EM 训练和动态规划编码”换来了“概率化切分和子词正则化”。分词器没有全局最优，只有和目标语言、模型规模、推理预算匹配的选择。
{% endhint %}

## 2.9 习题

### 习题 1（推导）

设分词器 1 的平均 Token 长度为 $$r_1$$ 字节，分词器 2 为 $$r_2$$ 字节。两者在同一文本上的每字节比特数相同，记为 $$\rho$$。证明 $$\mathrm{PPL}_1=\mathrm{PPL}_2^{\,r_1/r_2}$$，并回答：若 $$r_1=2r_2$$，两者 PPL 相差多少倍？如果有人报告分词器 1 的 PPL 为 20、分词器 2 的 PPL 为 5，能否据此判断分词器 2 更好？

<details>

<summary>展开参考答案</summary>

设每个 Token 的平均交叉熵为 $$\mathcal{L}$$（单位 nat），每字节比特数为 $$\rho=\frac{\mathcal{L}}{\ln 2\cdot r}$$。两边乘以 $$\ln 2\cdot r$$，得 $$\mathcal{L}=\ln 2\cdot r\rho$$。

对分词器 1、2 分别写出 $$\mathcal{L}_1=\ln 2\cdot r_1\rho$$、$$\mathcal{L}_2=\ln 2\cdot r_2\rho$$。由困惑度定义 $$\mathrm{PPL}=\exp(\mathcal{L})$$，得 $$\mathrm{PPL}_1=e^{\ln 2\cdot r_1\rho}=2^{r_1\rho}$$、$$\mathrm{PPL}_2=2^{r_2\rho}$$。

两式相除得 $$\frac{\mathrm{PPL}_1}{\mathrm{PPL}_2}=2^{(r_1-r_2)\rho}$$。

又因为 $$\mathrm{PPL}_2=2^{r_2\rho}$$，所以 $$2^{\rho}=\mathrm{PPL}_2^{1/r_2}$$，代入得 $$\mathrm{PPL}_1=\mathrm{PPL}_2\cdot \mathrm{PPL}_2^{(r_1-r_2)/r_2}=\mathrm{PPL}_2^{\,r_1/r_2}$$。

若 $$r_1=2r_2$$，则 $$\mathrm{PPL}_1=\mathrm{PPL}_2^2$$。例如分词器 2 的 PPL 为 5，分词器 1 在同样压缩质量下应为 25。

因此，报告分词器 1 为 20、分词器 2 为 5 时，不能判断分词器 2 更好。若两者每字节比特数相同，分词器 1 的期望 PPL 是 25；实际只有 20，反而说明分词器 1 的压缩质量更好。要比较两个模型，必须先把 PPL 换算成每字节比特数，或者在同一个分词器下评测。

</details>

### 习题 2（编程）

把本章 BPE 的合并准则从“最高频相邻对”改成 WordPiece 得分 $$\mathrm{score}(a,b)=\frac{f(ab)}{f(a)f(b)}$$。

在同一份语料上训练 512 次合并，要求：

1. 用对数计算得分，避免浮点溢出；对零频次做平滑；
2. 固定并列时的选择规则，保证结果可复现；
3. 比较两种准则得到的合并序列、中文/英文 Token 数、往返一致性；
4. 实现 `tokens_per_char`、`tokens_per_word`、`bytes_per_token` 三个指标，并报告结果；
5. 思考：为什么 WordPiece 在高频字符对上可能反而不如 BPE 激进？

### 习题 3（开放思考）

同一段语义相同的内容，用不同语言表达时，Token 数可能相差很大。这会导致 API 计费、上下文占用和推理延迟在不同语言之间不平等。

请设计一个实验，在固定模型和语义对齐的多语言文本上，测量至少 5 种语言的 Token/字符、字节/Token 和每字节比特数；然后讨论：

1. 扩大词表能在多大程度上缩小差距？代价是什么？
2. 如果词表无法完全消除差距，按语言调整价格、按字节计费、或者提供更大的上下文窗口，哪种机制更合理？
3. 分词器的设计是否只是一个工程问题，还是也涉及公平性？
