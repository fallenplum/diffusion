# Phase23 CDEI 前作与数学边界审计

日期：2026-10-18。创新状态：**`UNKNOWN_NOT_ESTABLISHED`**。本文件为文献与算子审计，不包含新增模型训练或生存结果。

## 1. 审计范围与结论

本轮实际尝试打开 Google 的 Energy-based OOD、Hopfield Networks、Energy Transformer 搜索页，三页均由检索工具返回不可访问；随后使用可用网页搜索发现来源，再核验原论文、会议论文集、作者论文页及由论文指向的官方代码。OpenReview 的部分页面触发浏览器校验，相关公式改从原作者 arXiv 全文或正式会议 PDF 核验。没有把二手博客、同名第三方 GitHub 复现或搜索摘要作为算子同一性的最终证据。

**已知组件足以排除“首次从 softmax 以外读取 logit 强度”“首次使用 LSE 聚合跨模态匹配”“首次使用 null 去除单模态先验”等宽泛创新表述。** 当前仍可检验的差异，是在固定强主效应与相同兼容核下，训练分区参考校准的额外分数读出，是否提供患者配对及 Clinical 条件所特有的、可泛化的 DSS 条件增量。本次检索没有建立这一差异的首创性，也没有验证其有效性。

## 2. 必须承认的原始前作

|原始前作|本轮核验到的相关内容|CDEI 不能据此声称的新颖点|目前仍不同的部分|
|---|---|---|---|
|[JEM，ICLR 2020](https://research.google/pubs/your-classifier-is-secretly-an-energy-based-model-and-you-should-treat-it-like-one/)，[原文](https://arxiv.org/abs/1912.03263)，[官方代码](https://github.com/wgrathwohl/JEM)|同一个判别分类器可同时给出条件类别概率与未归一化输入能量。|用分类分数中未被条件概率识别的自由度读取额外信息。|不是 Clinical 条件的 P–O 兼容核，也不是本项目的小样本条件生存增量检验。|
|[Energy-based Out-of-distribution Detection，NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html)，[全文 §2–3](https://arxiv.org/html/2010.03759)，[官方代码](https://github.com/weitliu/energy_ood)|以 `−T log Σ exp(f_k/T)` 读取分类 logits；比较其与 softmax confidence 的差别。|LogSumExp 能量本身，或“softmax confidence 不充分保留原分数信息”的一般动机。|原任务是 OOD；不能把其密度解释直接转移到任意生存兼容分数。|
|[Hopfield Networks is All You Need，ICLR 2021](https://arxiv.org/html/2008.02217)，[官方代码](https://github.com/ml-jku/hopfield-layers)|§2 给出含 LSE、`log N` 常数及二次项的能量；相应更新为 `X softmax(β Xᵀξ)`。|将归一化 attention 与 LSE 能量联系起来；按候选数消除确定性 `log K` 项也不构成新数学。|CDEI 是双读出风险候选，不自动继承 Hopfield 收敛、记忆容量或能量下降保证。|
|[Energy Transformer，NeurIPS 2023 原文](https://arxiv.org/html/2302.07253)，[原文指向的官方 JAX 代码](https://github.com/bhoov/energy-transformer-jax)|§2 对 Q–K 相容度构造 LSE 能量，并由整体能量梯度导出 token 动态。|“attention 之外具有能量/强度路径”的广义概念。|ET 对架构和对称性有明确约束；把一个 LSE 标量接入 Cox head 并不是 ET，也无权借用其理论保证。|
|[Theory, Analysis, and Best Practices for Sigmoid Self-Attention，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a43b459e9fab3a703148ba0c83b8a442-Abstract-Conference.html)，[正式全文](https://proceedings.iclr.cc/paper_files/paper/2025/file/a43b459e9fab3a703148ba0c83b8a442-Paper-Conference.pdf)，[官方代码](https://github.com/apple-aiml-research/ml-sigmoid-attention)|§2 使用逐元素 sigmoid 代替行 softmax；研究初始范数、QK normalization 与约 `−log n` 偏置的稳定性。|第一次使用对共同分数偏移敏感的 attention，或认为去掉行概率归一化天然稳定。|不是固定相对路径加独立参考能量路径。可作为有关分数尺度及候选数的近邻控制；本文件不自动增加正式实验臂。|

上表只概括本问题相关内容，未复述上述方法的性能数字。数学组件的前作归属置信度：**高**；CDEI 整体设计是否已有完全相同实现：**未确定**。

## 3. 比医学应用更接近的匹配读出前作

### 3.1 SCAN：相对注意力后，再读跨模态匹配强度

[Stacked Cross Attention for Image-Text Matching，ECCV 2018 正文 §3](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Kuang-Huei_Lee_Stacked_Cross_Attention_ECCV_2018_paper.pdf) 使用跨注意力获得局部对齐，并对词与 attended image vector 的余弦相关性做 AVG 或 LSE 汇总得到图文相似度；[官方代码](https://github.com/kuanghuei/SCAN)。因此，**“位置分配之外读取整体跨模态匹配”已存在**。但它的 LSE 输入为 attention 后的局部相似度，并非此处带训练供体参考的原始临床条件兼容核，不能把两者写成完全相同算子。

### 3.2 Discffusion：attention map 经 LSE 产生匹配分数

[Discffusion，TMLR 2024，原文 §4.3–4.4](https://arxiv.org/html/2305.10722) 从 Stable Diffusion 的跨注意力中提取图文关联，经 LSE 与汇总得到匹配分数；[官方项目](https://github.com/UCSB-AI/Discffusion)。**LSE 读取 attention 相关匹配分数不能作为 CDEI 的新贡献。** 原文 `A` 的写法同时涉及归一化注意力与 attention 输出，不能据文字 `attention score` 就把它等同于 CDEI 的明确 pre-softmax logits。它也未在这里提出 Clinical 条件、训练供体 null 与固定生存主效应组合。

### 3.3 MASS：null 校准以抑制单模态先验

[MASS: Overcoming Language Bias in Image-Text Matching，AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/download/32262/34417)，[全文 §3](https://arxiv.org/html/2501.11469)，用视觉条件 token log likelihood 减去语言侧参考，实际以黑色 null image 近似参考；[官方代码](https://github.com/JiwanChung/mass)。这直接排除“首次用 null 校准跨模态关联、区别单模态先验”的宽泛表述。MASS 基于语言概率；CDEI 的训练错配兼容分数不具备该概率模型，**不能直接把 CDEI 的 E 称为 PMI、似然比或医学无关联概率**。

## 4. 医学多模态领域的近邻

|方法与原始来源|已经覆盖的方向|与当前候选的具体区分|
|---|---|---|
|[MCAT，ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Multimodal_Co-Attention_Transformer_for_Survival_Prediction_in_Gigapixel_Whole_Slide_ICCV_2021_paper.pdf)|WSI 与组学 embedding 的 dense co-attention，用于生存。|P–O token 交互不新；原文不提供本候选所定义的 Clinical 条件与训练参考双证据算子。|
|[MGCT，BIBM 2023 原文 §II-C](https://arxiv.org/html/2311.11659)，[官方代码](https://github.com/lmxmercy/MGCT)|病理与组学双向、互相引导的跨模态 Transformer。|双向读取不新；需要实际比较共同核与风险读出，不能仅以加入 Clinical 或改名宣称更强机制。|
|[MRePath，IJCAI 2025 正式论文 §3.3](https://www.ijcai.org/proceedings/2025/0201.pdf)，[会议页](https://www.ijcai.org/proceedings/2025/201)|由单模态 mono-confidence 与基于两模态 confidence 的 holo-confidence 形成动态权重，再做 interactive alignment fusion。|病例自适应可信度与跨模态重加权已做过；holo-confidence 并非当前逐 Patch–Pathway 原始 logit 的训练 null LSE。|
|[M2EF-NNs，Scientific Reports 2025，式8–15](https://www.nature.com/articles/s41598-025-93770-3)|co-attention 加非负证据、Dirichlet/subjective logic、Dempster–Shafer 融合，用于生存风险。|“evidence-guided survival fusion”不是新概念。其 evidence 有特定参数化；CDEI E 是可正可负的参考分数，不能混用为已校准不确定性或可靠性。|

本轮没有把医学论文中的关联热图当组织学真值，也未审计这些论文的全部代码、数据泄露控制或报告性能在本队列的可复现性。上述区分是已核文本的范围判断，**不是不存在同构前作的证明**。

## 5. 从第一性原理得到的约束

以下是直接代数推导及本项目的实验设计分析，和论文中已验证的医学结论分开。

### 5.1 softmax 精确丢掉的是共同加性偏移

对有效 token 行，`softmax(s + a·1)=softmax(s)`；一般 `softmax(αs)≠softmax(s)`。因此不能笼统写成“softmax 丢掉一切绝对尺度”。若同温度相对概率为 `A_k=exp(s_k/τ)/Z`，则 `s_k=τ log A_k + τ log Z`。保留 LSE 等于为概率分布补回一个逐行标量自由度，**它是否包含独立预后信息必须另证**。

`logmeanexp` 去掉的是固定候选数引起的 `τ log K`。它不能保证不同 K、缺失模式、score variance 或强离群值时完全可比。对残差分数 `r_k`，小波动展开为 `E≈mean(r)+Var(r)/(2τ)+…`，故 E 上升也可能是方差/尾部变化，不等于广泛更强的匹配。需同时报告行均值、标准差、范数、有效 K 和极值敏感性。

### 5.2 “固定 null”必须说清固定什么

若参考为同一核、同一 Patch 与 Clinical 对训练供体 Omics 的平均：

\[
b_{ik}^{\rm ref}(\theta)=\frac1M\sum_{j\in\mathcal R_{train}}s_\theta(p_i,o_{jk},C_i),
\]

则任何与 Omics 无关的公共项 `h(p_i,C_i)` 会在 `s−b` 中精确抵消。这有利于控制 P/C 主效应，却意味着候选并不保留所有原始公共方向。

如果预先算好并冻结 `b(θ₀)` 数值，再继续训练兼容核 `θ`，后来产生的 `h_θ−h_θ₀` 不会自动消失，绝对通道可变成普通风险偏置。需要明确采用“固定参考输入、随当前核计算”还是“冻结核后冻结参考分数”，以及是否传播参考分支梯度；不能把不同实现混称同一校准。此处不是要求未授权地改变既定网络，而是正式训练前必须锁定的可识别性合同。

另有 `LME(s−b)` 一般不等于 `LME(s)−LME(b)`。候选公式应逐式实现，不能用更容易写的另一式替代。

### 5.3 参考不是无关联医学真值

训练供体错配只构造经验参考，其分布可能混合癌种、RNA 平台、缺失区块或阶段。reference 必须限于当前 inner-train / outer-train；inner-val、outer-test 不能参与参考均值、温度/方差校准或样本选择。错误患者对照同样可能引入平台变化，需要保存错配约束、有效覆盖及匹配残余差异。

### 5.4 历史动机的范围

本轮实际读取 [Phase21 最终机制复核](../phase21_relative_attention/FINAL_MECHANISM_REVIEW.md)。其中已经明确：DIRECT 和 S16 都删除行公共偏移，本轮没有检验保留公共偏移；较大的公共能量份额没有证明后续消息发生普遍抵消。故可以从“公共偏移未被独立读出”提出新假设，**不能把“归一化导致风险消失”改写成已证实的完整病因链**。

## 6. 可保留的候选贡献与需要的证据

当前可写成待检验问题：**在同一 Clinical 条件 P–O 核及强 C/P/O 主效应下，训练参考校准的 logit-level 读出，能否比仅相对 attention 提供额外、患者身份相关的条件预后信息？**

至少需要同时具备以下证据，才有资格讨论比组件组合更具体的贡献：

1. 数学/实现资格：有效 mask、token 数、共同移位、分数缩放、稳定 logsumexp、参考折谱系全部正确；只说明技术资格。
2. 容量归因：relative-only、relative+absolute、等容量额外主效应控制；保持相同基础特征、预处理、核容量、损失、训练选择规则与可比较预算。
3. 三模态归因：无 Clinical 条件、正确/错误患者 Omics 与 Clinical 条件；错误配对效果需和主效应丢失区分。
4. 预测归因：同 checkpoint 关/开 absolute 的风险差异与患者对修复/损伤，以及独立训练模型间的完整差异；不能只报分数能量非零。
5. 泛化边界：严格嵌套选择、同折配对区间、seed 稳定性；研究反复使用的同队列结果不替代独立外部验证。

**优势：**假设可用少量明确控制隔离，能直接检验此前未读出的逐行自由度。**劣势/风险：**自由度可能主要是主效应、norm 或平台偏置，null 可能抵消目标信息，65 个 DSS 事件也可能不足以稳定选择新增通道。文献不会替代这些实验。

## 7. 决策与未完成边界

- `novelty_status: UNKNOWN_NOT_ESTABLISHED`
- `mathematical_components_have_prior_art: true`
- `complete_literature_exclusion: false`
- `new_survival_experiments_in_this_document: 0`
- `evidence_for_DSS_conditional_gain: NOT_TESTED_HERE`

可进入受约束的机制资格与锁定实验准备，不能据本报告宣称论文创新成立，也不能自动扩大模型矩阵。后续若出现有效机制证据，再针对“Clinical 条件的参考兼容能量 + 主效应隔离”做系统前作追踪；目前没有满足首创性判定的证据。
