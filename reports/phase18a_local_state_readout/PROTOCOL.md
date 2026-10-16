# Phase18A：局部状态读出 × 患者条件交互协议

## 研究范围与来源

独立、有限的探索性 2×2。父提交 `760a6a88feb9b731f8a935c98a8a79c4ca300943`；父训练来源 `2c06cbe6c270466f43512cd8d97c0f95c38f673c849cd3441084c240fed21b86`；本轮锁定训练来源 `3eafc2c455e53e71adc39fa305952ad616d7d87f235d56a894502c4e4a3572a0`。Phase17/17R 原始结果封存，旧 R3 不启动。本轮不扩展扫描方向、gamma 幅度、delta/B 幅度或新的损失。

Mamba 原始逐位置输出与 MambaMIL attention pooling 的实际对照先于训练交付，见 [论文代码映射](READOUT_PAPER_CODE_MAPPING.md)。常规 SSM 局部输出不作为原创算法。此前 `a60aa99e7482d398b46c15738975105226871024` 的四路径注意力是独立 inner-only 探索，不属于本轮 S6，也不并入本轮主比较。

## 唯一操作差异

F0/F1：`z[d]=sum_s h_T[d,s] * sum_t(m_t*C_t[s])/sum_t(m_t)`。

L0/L1：`z[d]=sum_t(m_t*sum_s(C_t[s]*h_t[d,s]))/sum_t(m_t)`。

两条路径保持相同 u、dt、B、C、A、有效 mask、对应模型原写入门及读出参数。局部读出采用 chunk=32 的精确分段 scan/checkpointing；段间不 detach，不使用截断 BPTT。无新增参数、归一化、attention pooling、D skip、残差或正则任务。

|臂|读出|状态条件|训练来源|
|---|---|---|---|
|F0|终态 × 平均 C|无跨模态状态条件|父 R1_S6_noDirect，仅通过资格/重放审计后引用|
|F1|终态 × 平均 C|delta/B 与原写入门|父 R1_A8_noDirect，仅通过资格/重放审计后引用|
|L0|逐位置 C_t h_t 加权池化|无跨模态状态条件|本轮 fresh inner/outer 拟合|
|L1|逐位置 C_t h_t 加权池化|delta/B 与原写入门|本轮 fresh inner/outer 拟合|

无条件模型仍由病理驱动选择性参数；自身 Clinical、RNA/CNA token late 主效应与同源 Clinical+S32 锚点保留。无高维 OmicsDirect 不等于移除组学。F1/L1 的条件因子是整套机制，不能从本轮拆分 delta、B、门的独立因果效应。

## 数据与拟合隔离

原队列 512 名患者、65 个 DSS events；原 5-fold patient CV × seeds 31/47/83。固定 UNI2-h 在 CONCH 足迹上的面积聚合、坐标顺序、每患者至多 512 patch 与多 WSI 等权；仅 geometry-only，未完成原图组织学 QA。所有预处理、基因选择、缺失处理及 anchor 拟合仅用对应训练分区；inner-val 只用于选 epoch；fresh outer 按所选轮数训练，outer-test 不用于任何选择。

共同训练：AdamW，lr=0.001，weight_decay=0.01，dropout=0.25，clip=5.0，最大 160 epochs，EMA alpha=0.2，patience=40，microbatch=128。微批量仅管理前向显存；损失及其梯度使用完整训练患者风险集的 Efron Cox。保持 anchor lambda=10.0；不以训练内风险再拟合自由 stacking 标量，不拼接独立 folds 的 latent 坐标。

## 统计对象与八项家族

主要比较为 L0−F0、L1−F1、L1−L0、(L1−L0)−(F1−F0)，各自 Harrell/Uno60，共八项。每 seed 先汇总五折同折可比较对的分子/分母，再平均三个 seed 指标；不平均风险、不称 15 cells 为 seed mean。

20,000 次固定 outer-fold 内患者 multinomial 配对 bootstrap；所有方法和 seeds 共用患者权重，pair 权重 count_i×count_j；Uno60 的 censoring reverse-KM 来自原 outer-train。DID 每次重算四个完整 C 指标后作差，不拼接单方法区间。报告 95% 与 Bonferroni8 的 99.375% 区间。区间条件于已训练/已选择模型，不含完整重训练与研究者反复选择方差。

PFI 仅冻结 DSS 风险迁移，无 PFI 训练、epoch 选择或配置筛选。ANCHOR_CS、R1_A3_noDirect 和历史 Clinical+S32+CONCH32 旁列；最后一项视觉和训练协议不同，不称同输入消融。

## 诊断定义与工程停止规则

32 名训练患者按原顺序无标签选取，记录风险/状态差、患者等权 WSI 的 z/SSM 风险方差、独立条件路径梯度；只更换条件输入的分子状态，自身 late 证据不变。位置梯度分别对终态平方范数与最终患者风险求导，每 checkpoint 为第一名训练患者的所有 WSI 等权；探针重复患者数不是独立病例总数。位置为几何顺序，无医学时间含义。所选 checkpoint 隔离梯度和同 epoch 真实 Efron 梯度分开。

A：只有 L0−F0 与 L1−F1 在两个指标均正，才登记读出方向一致改善；否则停止自定义 S6 修补。B：读出改善而条件增量门不成立，记录 READOUT_HELPFUL_CONDITION_NOT_ESTABLISHED。C：L1−L0 与 DID 两指标均正，分别至少 2/3 seeds 同向，且 L1 对 ANCHOR_CS 与 R1_A3_noDirect 的两指标均值不下降，才登记 DIRECTIONAL_CANDIDATE。D：校正区间支持仍仅为当前反复研究队列内证据。工程筛选不是统计确认；无显著性不是等效或无信息。

任何状态/梯度增强都不能替代预测门。无论结果如何，完成本轮后停止，不自动追加 Phase18B、第四种读出、额外幅度或大规模 SOTA 矩阵。

## 计数、隐私与交付

正式新神经网络拟合 30 inner + 30 outer。历史 F0/F1 的 30 inner+30 outer、A3 的15 outer、历史 CONCH32 的15 outer 独立计为 90 次引用，不记为新神经网络拟合。标签突变检查的全部实际短拟合（包括审计 helper 报错前完成者）另行计数。冻结重放/探针不更新神经网络参数；但父 Data 会确定性重解同一 Clinical+S32 Ridge anchor：完整合格链的75次历史回放、60次历史探针、60次新checkpoint回放共195次验证重解，均与原checkpoint逐元素/bitwise一致，不产生新模型选择或新的OOF。195不是所有阶段底层solver调用总数，正式/短拟合内部及旧superseded探针的数据构造另外存在，不混称“审计没有任何优化”。

患者级 OOF、检查点、真实标识、原始矩阵和凭据留私有目录；公开仅源码、协议、匿名聚合及修复日志。新增文件隐私检查与全仓库继承旧 ZIP guard 问题分开；不得改历史以使检查表面通过。
