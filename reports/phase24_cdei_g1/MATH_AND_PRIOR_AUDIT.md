# CDEI-G1：数学边界、训练参考与前作审阅

审阅日期：2026-10-18。性质：独立协议审阅；本文件未读取患者级数组、未训练模型、未评价生存标签。实验进度与结果以同目录实际运行回执为准。

## 1. 本轮真正需要回答的问题

**当前核与局部读出联合训练后，训练参考校准的、保留正负号的 Patch–Pathway 消息，能否在保留相同 C/P/O 主效应的前提下，比普通相对注意力及同容量条件消息提供额外的 held-out DSS 排序信息？**

这有两个相互关联但不能混同的改动：核与读出的联合优化；由患者级两个标量改为关系层面的消息。四臂设计可以分别提供相关证据，但并非所有两臂差异都只有一个原因。新候选仍属于待检验架构；本文件不把 signed gating、双通道、参考相减或低秩三线性核列为新数学。

历史边界已经直接核对 [Phase21 最终独立机制复核](../phase21_relative_attention/FINAL_MECHANISM_REVIEW.md)：公共修正能量较大不等于已证明消息普遍抵消，更不等于已证明公共方向含有独立预后信息。G1 不能把该动机写成已成立病因。

## 2. 精确算子与方向

为避免患者与 Patch 都写成 i，以下用 n 表患者、i 表 Patch、k 表 Omics token。

\[
s_{nik}=q(p_{ni})^\top k(o_{nk})/\sqrt d+
\sum_r a_r(p_{ni})b_r(o_{nk})c_r(C_n),\qquad
A_{nik}=\operatorname{softmax}_{k\in\mathcal V_n}(s_{nik}).
\]

\[
D_{nik}=s_\theta(p_{ni},o_{nk},C_n)-s_\theta(p_{ni},\bar o^{\rm ref}_{nk}(\theta),C_n),
\qquad
M_{ni}=\sum_{k\in\mathcal V_n}A_{nik}\tanh(D_{nik}/\tau)V_\theta(o_{nk}).
\]

关键实施合同：

- A 必须来自本人原始核 s，在 K 方向做 masked softmax；不是 softmax(D)，也不是对 `A*tanh(D/tau)` 再归一化。
- 本轮局部证据消息对应 Patch→Omics 方向。可以保留原有双向 Relative 消息，但不能不经预注册就增加反向局部证据分支。
- 无有效 token 的行必须返回零消息及明确 availability；padding 中 NaN 不能经 `0*NaN` 进入前向或反向。
- `tau` 必须为事先锁定的正数。不能以 test 表现选择温度、阈值、核缩放或新的正则。
- 患者内多张切片沿用既定等 WSI 聚合；不能在 reference 中按 WSI 数重复患者，也不能在风险中按切片重复计事件或重复加入 Clinical anchor。

### 2.1 `tanh` 实际保留与限制的内容

令 \(w_k=A_k\tanh(D_k/\tau)\)。对有效行有

\[
\sum_k|w_k|\le 1,\quad
-1\le\sum_k w_k\le 1,\quad
\|M_i\|\le\sum_k A_{ik}\|V(o_k)\|\le\max_k\|V(o_k)\|.
\]

这是**相对于当前 Value 范数的界**。它不是固定的消息范数界，更不是最终风险修正的固定界：Value 投影、后续 head 和输入范数仍可扩大输出。若风险头另有 `tanh` 与 anchor SD 缩放，应按该真实公式单独声明，不能将其归功于局部 gate。

`w` 可正可负，不能称为概率分布、可靠性概率或注意力置信度；D<0 只表示低于经验参考，不是医学“无关联”，也不自动表示保护性风险。正负项可能在 token、Patch 或 WSI 聚合时相互抵消，因此 gate 非零并不保证有最终风险贡献。

若一行 D 恰好为共同常数 d，则 \(M_i=\tanh(d/\tau)\sum_k A_{ik}V(o_k)\)。这证明该通道可以保留原本不影响 A 的幅度，但同时说明它可能退化为相对消息的患者级幅度门控。出现这种情况不能直接宣称发现了新的局部语义关系。

### 2.2 Softmax 与共同偏移的精确边界

单方向有 `softmax(s+a*1)=softmax(s)`；通常 `softmax(alpha*s)` 会改变分布。因此“softmax 丢掉一切绝对尺度”不准确。若共享矩阵同时在两个方向归一化，某方向的逐行偏移可能影响反方向，不能称所有这些信息在整个双向网络中都消失。

G1 在 gate 后不再做 softmax，故总 signed mass 不被强制为 1；但 A 本身仍是归一化分配。该候选是“相对分配加参考残差的有界调制”，不是任意 unnormalized attention，也不是恢复了原始兼容度的全部自由度。

## 3. 动态训练参考：固定输入与同步核

对当前训练患者集合 \(\mathcal T\)，每槽可用性为 \(m_{jk}\)，定义

\[
\bar o^{\rm ref}_{nk}(\theta)=
\frac{\sum_{j\in\mathcal T\setminus\mathcal E_n}m_{jk}f_\theta(x_j)_k}
{\sum_{j\in\mathcal T\setminus\mathcal E_n}m_{jk}}.
\]

这里固定的是原始参考患者、输入和 mask；编码表示及 score 随当前参数变化。

1. 内层的参考只能来自 inner-train，外层只来自 outer-train。held-out 特征不能参与参考均值、校准或统计估计，更不能使用 held-out 标签决定参考患者。
2. 训练 query 排除自身；held-out query 使用相应 fit 集。均值按患者等权、按槽可用数作分母，不能把缺失 token 当成零值观测。
3. 每次参数更新后的新前向必须重新编码参考并重新算参考分数。将 `b(theta_old)` 或旧 encoded bank 冻结后训练新核不符合本轮假设。
4. `mean(encode(raw patient))` 一般不等于 `encode(mean(raw patient))`。不能用后者节约计算而称为同一参考。
5. 已核对现有 `RankConditionedAttention`：q/k/a/b/c 都是无 bias 线性投影；在固定 P、C 下，score 对 encoded Omics token 为线性。因此“先按槽均值 encoded token 再计算 score”与“对相同供体逐一算 score 后按槽均值”精确等价。该等价不延伸到非线性关系核，也不延伸到先分别算 tanh 再平均。
6. 若目标是精确优化 \(D_\theta=s_\theta-s^{ref}_\theta\)，当前参考编码、参考核分数均应保留 autograd。`detach(reference)` 会更改参考编码器的梯度，是另一个优化算子；必须如实披露，不能只因前向值相同就称完全等价。原始数据和分区索引本身不需要梯度。
7. memory checkpoint、microbatch 或梯度累积的优化不能导致同一 optimizer step 内混用多个版本的 kernel/reference。所有 fit 患者在完整 Efron 风险集中贡献一次；不能把 microbatch Cox 当成同一个目标。

### 3.1 动态参考仍未解决的可识别性

若同一当前核出现与 Omics 无关、对本人及参考相同的加性项 \(h(P,C,k)\)，它会在 D 中抵消。这是参考减法的优点，也意味着局部通道并不保留所有公共方向。

但以下捷径仍可能存在：Omics 单独主效应、O×C 项、输入/Value 范数、缺失与平台来源、核权重尺度及 head 权重之间的补偿。放大核既改变 A 又改变 D；固定 tau 下 tanh 还可能饱和。**动态参考避免旧核基准漂移，不保证尺度已经唯一确定，也不保证 D 是纯三模态关联。**

参考分布是当前 fit 样本的经验比较点，不是医学无关联群体、PMI、似然比、校准风险概率或因果反事实。

## 4. 四臂控制的可解释范围

|臂|输入读出|能回答|不能单独回答|
|---|---|---|---|
|G1-REL|原双向 Relative 64D|新协议的端到端共同基线|和旧 G0 的差异不一定只来自核解冻|
|G1-POOL|Relative 64D + 原定义的两个 Absolute 标量|相同新协议下池化证据是否可用|若训练器/head/schedule 也变，不能把与 G0 的差异全归因于联合训练|
|G1-LOCAL|Relative 64D + 局部证据消息聚合32D|候选完整效果|比 POOL 多30个读出维度，不能称二者容量完全匹配|
|G1-CTRL|Relative 64D + 普通条件消息32D|同输入维度、同 head 参数预算下，参考 signed evidence 是否优于普通 gate|只排除该具体 gate；不能证明优于所有门控或所有额外主效应|

初轮 CTRL 采用已锁定形式

\[
M_i^{ctrl}=\sum_k A_{ik}\sigma(s_{ik}^{context}/\tau)V(o_k),
\]

无参考相减，与 LOCAL 共享同类型核、Value 维度及相同读出参数量；它本身也对 raw-score 公共偏移敏感，不是 relative-only 控制。不能看到 CTRL 有效就称 Absolute Evidence 无效，也不能看到 LOCAL>CTRL 就分解成“必由 signedness”或“必由参考校准”产生：两者同时不同。本轮不为分解这两点追加 sigmoid/tanh 等完整矩阵。

四臂应采用相同数据分区、输入、预处理、强主效应、初始化配对、优化与选择规则。参数量相同不等于优化难度相同，LOCAL/reference 的计算量也高于普通 gate，需报告真实耗时和参数。

### 4.1 冻结主效应的边界

共同 J0 必须来自当前角色的 fit-only 训练，held-out 风险可逐元素回放验证。它固定了风险主效应的共同基准，有利于比较新增分支。

**当前 fit 患者上的 J0 是 in-sample 预测；不是 cross-fitted teacher。** 在相同 fit 标签上训练残差分支不等于 held-out 泄露，但 anchor 的训练拟合偏差可能影响剩余监督与可学梯度。因此结果适用于“冻结该 J0 条件下的新增分支”，不能直接当成统计意义下所有 C/P/O 信息已被彻底解释后的条件独立检验。若未来采用 cross-fitted offset，需要单独锁定其数据谱系与新协议，不能在本轮悄悄更换。

## 5. 冻结模型干预及患者配对

本轮冻结 LOCAL 做 Clinical-fit均值、错 Clinical、错 Omics；这三项是诊断而非独立训练消融。

- Relative A、原 Relative Value 消息、本人主效应始终保留；只改变 extra 分支。否则风险变化同时包含基础信息移除，不能隔离新局部路径。
- 错 Clinical：对 extra 的本人和 reference score 同时使用同一个错 Clinical。均值 Clinical 同理，均值仅来自当前 fit。
- 错 Omics：extra 的本人 score 与 Value 同时换成供体 Omics；不是只换 D、不换 V。其效应同时包含候选证据与 Value 内容变化，属于 branch intervention，不能称整个模型的错误配对性能。
- matched/wrong 配对比较使用共同 reference 排除集合。若供体属于 fit，应共同排除 recipient 与 donor，避免干预同时改变 bank 组成；held-out recipient/donor本就不能进入 bank。
- 可用 token 数也应一致。按既定方案使用本人与供体可用槽位交集，并在 matched/wrong 两侧共同应用。**本轮 config/protocol 已预锁硬资格：十个 role/fold 分区均须有全部24槽、支持交集相对本人无变化；否则阻断原型，不静默删患者或修改支持。** 在该前提通过后，交集恒等于本人支持，标准 held-out 风险不会因供体覆盖而发生数值变化。本文件不据一般性的潜在风险指称当前已发生 coverage bias。该受限原型不能直接推广到任意缺失模式；将来若允许交集改变本人输入，必须把该结果另标为“配对诊断前向”，并报告保留的 A 质量，不能冒充标准 matched 风险。
- 错配限在相应 fit/held-out 分区内，不能 self-pair；固定分层应优先控制癌种和已核缺测模式。缺测模式只能称 RNA 来源的 proxy，不能冒称已知真实平台。DSS/事件标签不能用于选择供体。
- 如果冻结干预显示有价值线索，再预注册相应真实重训消融；当前干预不能替代该重训。

## 6. 先于性能解释的数值诊断

以下只增加记录，不增加候选网络或监督目标。按 fit/held-out 分开、等患者聚合，并保留各 seed/fold稳定性。

|层次|记录|用于排除的混淆|
|---|---|---|
|Score/reference|本人score、reference score、D 的均值/SD/RMS/极值；有效 K；当前核版本一致|旧参考、候选数、偏置/范数及非有限值|
|Gate|D/tau分布、tanh均值、平均绝对值、固定阈值的饱和比例、平均 `1-tanh^2`|门控失活或饱和、梯度消失|
|Weights|signed mass `sum A*tanh`、absolute mass `sum abs(A*tanh)`、共同mask保留的 A 质量|“正负存在”等同于信息有效、可用槽数改变|
|Messages|Relative/LOCAL范数、范数比、token→Patch→WSI→patient前后存活、正确/错误条件差|聚合抵消、Value norm捷径、弱患者特异性|
|Gradients|q/k、CP a/b/c、RNA/CNA token encoder、Value、head 的有限性与非零梯度；reference路径梯度单独核验|仅 head 在学习、detach错接、CP条件失活|
|Risk|完整风险、同checkpoint extra置零风险、各条件干预风险差/SD；原baseline对比|head忽略 extra、幅度增加但无排序贡献|
|Ordering|同折可比较对的修复、损伤与 tie credit；Harrell及冻结fit IPCW Uno60权重|净差为零被误写成完全无变化；只列修复不列损伤|

`sum w=0` 不等于消息必为零，因为 Value 不同；`sum w` 非零也不保证消息非零。梯度非零、D较大、局部范数变大均只是可训练性/敏感性证据，不能替代 held-out 净修复收益。

外折错排对只能用于最终评价，不能回流成新教师、额外损失、样本权重或配置选择。先完整患者级同折指标，再做配对汇总；不能把15个 cell 冒称15个独立 seed，也不能将重复 seed 患者对相加作扩大样本量。

## 7. 三篇直接相关的原论文

### 检索与核验方式

遵照用户检索偏好，本次实际尝试了两个 Google Search URL，工具均返回不可访问，**不能声称 Google 检索成功**。随后使用可用联网检索发现原始来源，并读取作者 arXiv HTML 与正式会议全文；未仅依据搜索摘要下结论。已使用 PDF skill 核读会议 PDF。检索不是系统综述，不能以未找到同构方法证明首创。

|原始来源|已核对内容|对 G1 的实际约束|
|---|---|---|
|[Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free，NeurIPS 2025，§2.2/§4](https://papers.nips.cc/paper_files/paper/2025/file/904e89bb4e632e75fb47f093b620b257-Paper-Conference.pdf)，[作者全文](https://arxiv.org/html/2505.06708v1)|系统比较 query/key/value、SDPA output等位置的 gate；其式5为受输入控制的乘法门控。|Value/attention输出上的条件门控已有前作；LOCAL必须用同容量普通 gate 控制。其主要验证来自语言模型，不提供本队列或生存收益保证；G1的逐关系参考残差不是该文的普通输出 gate。|
|[Differential Transformer，ICLR 2025，§2.1式1](https://proceedings.iclr.cc/paper_files/paper/2025/file/00b67df24009747e8bbed4c2c6f9c825-Paper-Conference.pdf)，[作者全文](https://arxiv.org/html/2410.05258v1)|将两个 softmax map相减后乘 Value，允许 signed权重；同时使用额外head normalization。|正负权重消息及差分思想并不新。它减的是两张归一化 map，G1先减当前核本人/参考 logits，再 tanh调制本人A；二者非同一算子，不能直接借用其降噪或梯度稳定保证。|
|[Theory, Analysis, and Best Practices for Sigmoid Self-Attention，ICLR 2025，§2/§5及附录](https://proceedings.iclr.cc/paper_files/paper/2025/file/a43b459e9fab3a703148ba0c83b8a442-Paper-Conference.pdf)，[作者全文](https://arxiv.org/html/2409.04431v1)|研究逐元素 sigmoid替代行softmax，指出初始化 attention norm、bias及候选长度相关稳定性。|不做行概率归一化、保留幅度均已有前作；去归一化不会自动稳定。G1保留A再乘tanh且总绝对质量≤1，不等同于纯sigmoidattention，不能直接搬用其理论结论或训练参数。|

上述三篇足以约束本轮宽泛新颖性表述，不自动增加三篇论文模型的训练矩阵。既有 G0 [前作报告](../phase23_cdei/LITERATURE_AUDIT.md)还覆盖 LSE/energy、跨模态匹配及经验参考相关近邻；本文件不重复建立排行榜。

## 8. 优势、风险与结论可信度

**优势：**两个直接架构限制分别有明确定义；LOCAL与CTRL可以锁定同读出容量；动态fit参考、分支干预及修复/损伤能把“分数有变化”与“预测有用”分开。

**劣势与风险：**端到端会增加优化方差；reference使训练样本间通过特征均值相互作用；tanh可能饱和或退化为患者级幅度gate；保留的O/O×C主效应、fit anchor拟合偏差和Value范数仍可能混淆解释。该TCGA队列反复用于设计，严格当前折评估也不能替代未来独立队列验证。

|判断|可信度/当前状态|
|---|---|
|上述代数、线性核均值等价及参考梯度边界|高；基于直接推导和现有核源码，正式实现仍需数值测试|
|门控、signed消息及非行归一化已有直接前作|高；已核原论文|
|本候选组合没有完全同构前作|未确定；本次非系统排重|
|LOCAL可提供独立DSS增量|未在本审阅检验；只能由锁定实验决定|
|G1失败即可否定所有绝对证据交互|不成立；否定范围限于实际输入、参考、容量、预算和队列|

`novelty_status: UNKNOWN_NOT_ESTABLISHED`

`patient_data_read_for_this_review: false`

`new_survival_fits_in_this_review: 0`

拓展思考：若LOCAL有预测收益但冻结错配不损害，优先检查其是否学成普通幅度/主效应门控；若有条件敏感性却没有净修复，则问题仍可能在风险投影或方差，而非“条件没有进入网络”。这些是解释现有诊断的分支，不是自动追加新模型的授权。

你可能想知道：比旧G0更好与“局部证据机制被证实”是不同结论。前者可能来自共同端到端训练；后者至少还需LOCAL相对REL和同容量CTRL的稳定优势，以及与原主干分离的患者配对依赖。

## 9. 最终只读协议一致性复核

本次追加复核直接读取了当前 config、PROTOCOL、train/runtime、正式 runner 及独立 evaluation 源码，未增加实验或测试，未读取患者级数据。

新增训练源码文件与已锁 G0 身份组合的本地 SHA 为 `8234bca2b7fad388d3862d4d7be940f7e76f291bac516754b1e45fa5639bd18b`，与执行端声明一致。这里复核的是新增源码身份；完整上游 vendor 校验仍须以服务器真实资格回执为准，本机没有全部 pinned MGCT vendor 文件，不能声称在本机完成了该上游全链校验。

|合同|静态复核结果|
|---|---|
|四臂、三seed、五fold|config、train和evaluation均为REL/POOL/LOCAL/CTRL，31/47/83与1–5；共15 cell、60模型|
|每模型固定80 epoch|正式runner显式传80，要求每cell恰好320次更新；回执和evaluator要求各模型连续80条history、80 optimizer steps|
|固定末轮、fit-only训练|未实现inner-val early stopping；训练只经fit_labels取得标签；完整fit风险集只按32患者分块计算|
|完整性先于评测|evaluator先核全15cell的回执/SHA/80轮与共同anchor，再请求DSS；部分模型不能提前进入筛选|
|Harrell/Uno60统计对象|逐fold保留分子分母；每seed先累加五fold后取比，再平均3seed；Uno60以对应fit结局冻结IPCW|
|三项主比较|LOCAL−REL、LOCAL−POOL、LOCAL−CTRL：Harrell与Uno60分别要求3seed平均差>0且至少2seed为正|
|强基线点估计门|LOCAL−J0及LOCAL−Clinical+S32：两个指标平均差均≥0；这不是基于区间的统计非劣效检验|
|同头干预门|LOCAL分别减OFF/NO_C/WRONG_C/WRONG_O：两个指标均值均≥0，且每项至少一个指标严格>0|
|门失败的执行边界|停止当前算子；不自动启动outer、重训消融或超参数搜索|
|修复/损伤|只做描述性同折排序转换，要求repair−damage=净指标差；不得回流生成教师或新门|

上述源码、配置与协议在这些预锁规则上保持一致；这是执行合同复核，不能替代真实模型回放与正式结果验证。

**本轮内层没有确认性置信区间。** inner-val患者可跨fold重复，不能称为完整唯一OOF，也不能以多个seed重复患者扩大样本量。正向点估计、通过内层门、wrong-patient差值或修复对数，都不能单独证明医学机制、因果关系或独立组织学对应。尤其错Omics分支同时改变D及Value，风险差不能拆成其中某一个因素的独立作用。即使后续获准outer并得到条件于固定模型的配对区间，也仍需要独立医学标注/外部验证才能建立更强语义解释。
