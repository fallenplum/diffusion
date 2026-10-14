# Phase14-R0 — RNA 排序保留与条件交互机制审计

本轮是新探索性实验，不修改 Phase13。DSS：512/65；PFI：510/129，独立次要终点。
沿用固定患者折、31/47/83 seeds、TOP/HALL/MATCH 三组训练分区局部的512基因。

## A. 表征与容量

共享原 RNATokenizer（8×64）架构、初始化、AdamW、80 epoch ceiling、完整 Efron risk set。
两个 producer：Cox-only；Cox + 0.1 训练患者对排序蒸馏。都按 inner-validation Cox loss EMA 选 epoch（最低1），新建 outer 模型按选中 epoch 训练。修复已知最低10 epoch限制，不看本轮 outer-test 后更改。
教师为固定 λ=1 RNA Ridge Cox；每个 producer 的训练分区内部3折产生 scalar OOF 目标；各教师只看其子训练标签，标准化使用该教师子训练风险。不同模型仅拼标量风险，禁止拼 latent coordinates。
这里教师是 **label-OOF**：固定基因选择与RNA特征标准化来自整个producer训练分区，教师子折不重新选择基因/缩放RNA。不能称为全预处理嵌套OOF；不会接触producer的inner-val/outer-test特征或标签，但共享训练分区的无监督预处理可能影响教师目标，应列为限制。
蒸馏为所有无序训练患者对的 sigmoid 排序分布 KL（含删失患者；模仿教师排序，不将这些患者对当作真实生存顺序）。教师不是医学真值。

所有独立读出为同一 λ=1 Ridge Cox，训练局部列标准化：
- RAW512、完整 TOKEN512、监督约束 RANK512：512D容量对照；
- RAW64、TOKEN64、RANK64：各自训练局部 PCA64 后同一64D读出；
- POOL64、RANK_POOL64：同一 producer 的 token 均值64D后同一读出。

PCA64 vs mean64只匹配维度/读出容量，不声称信息内容或 encoder 容量相同。固定均值无可学习池化参数。另保留随机初始化 TOKEN64/POOL64，定位初始化、训练、压缩各环节。排序保留同时报告对训练教师 held-out 风险的线性可解码性与患者对一致性；不以教师分数当真实结局。

## B. 固定表征、严格条件比较

锁定同一 RANK64（不从测试结果挑表征），编码器冻结。所有 inner-readout 仅使用 inner-train producer，outer-readout 使用新 outer producer，坐标不跨模型拼接。
Clinical+S32 原强基线 λ 沿用上一轮 inner-only选择；RNA λ=1固定，禁止新网格。
- CS：原 anchor 逐元素复核。
- COX：Clinical+S32+RANK64 的联合 block Ridge Cox，避免 risk stacking。
- ADDITIVE：相同 block penalty、从 COX最优系数初始化的纯加性模型。
- INTERACT：相同主效应 + rank8 的 CS×RNA bilinear residual；预测上严格限制 |delta risk|≤0.25，不设后续可学习倍率。无 RNA query residual 主效应混入。
- WRONG_COX / WRONG_ADDITIVE / WRONG_INTERACT：同一模型，RNA仅在各 train/val/test 分区内部、癌种×缺失代理组内 derangement，固定 seed、无自身配对。这个控制保持边际分布，破坏患者对应关系。

加性/交互的完整优化预算、选择准则一致；inner 80 epoch，EMA validation Cox loss 选择；outer 同选中 epoch。COX凸解是额外优化充分的简单参考，不能把神经模型早停收益等同架构收益。
冻结有监督 producer 导致读出训练特征可能 in-sample适应；整个两阶段 pipeline 只在 inner/outer TRAIN建立，held-out从未参与 upstream训练，避免inner标签泄露。该限制明确披露。

## 统计、故障与隐私

每seed先拼5-fold OOF，指标只聚合同折比较对的分子/分母；Uno60 IPCW来自对应 outer-train。
共享患者 bootstrap 30000次，先算每 seed/model 对信用再平均，不平均风险；主DSS/HALL五对比×两指标校正10项。PFI及其他比较不作主确认。
独立 sksurv 重算、checkpoint replay、标准化/PCA fit hash、教师与错配分区审计、梯度/实际 epoch/损失/风险幅度、test-label perturbation硬门。
公开源码配置和匿名聚合报告；患者级 OOF/坐标/教师目标/checkpoint仅私有服务器保存。磁盘限额700MB；不复制大原数据，不删除历史正式产物。不保存 smoke预测。

优势：隔离读出维度、排序保留约束、加性与纯交互；劣势：单cohort重复探索、65事件、教师本身有误差、PCA不是最优压缩、冻结producer训练读出的适应偏差；本轮不证明因果机制或通路语义正确性。

方法参考（用于启发，非论文严格复现）：[survival distillation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11387617/)、[IPCW官方文档](https://scikit-survival.readthedocs.io/en/v0.23.0/api/generated/sksurv.metrics.concordance_index_ipcw.html)。
