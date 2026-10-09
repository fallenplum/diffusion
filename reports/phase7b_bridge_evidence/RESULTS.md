# Phase7B：重设计 Bridge 前的两项证据审计

**正式完成：A 复用 Phase7A 的 45 份 outer OOF，新增 60 个 outer fits / 240 个 inner fits；B 使用 512 位患者、五折 train-only nuisance 调整、1999 次患者置换、999 个度数保持随机映射及独立黏液性诊断。实验无 Bridge/通路/Prompt 改动。**

本轮是既往发现和同队列基础上的探索性证据扩展，先锁协议再新增拟合/比较；不是新的外部确认试验。PFS 不参与本轮两项决策，保持 Phase7A 的独立次要终点评估。

## 两项决策

| 问题 | 实测结论 | 决策含义 |
|---|---|---|
| A：在 Clinical+static concepts 后是否有稳定 Omics DSS 增量？ | 五项预先指定比较均未通过锁定判据 | 当前有限表示与共同线性头下，尚不足以推进更大生存 Bridge；不能证明条件信息为零 |
| B：存在患者对应关系及当前医学映射的特异优势吗？ | 正确患者对应关系有证据；15 边 prior 没有超过度数匹配随机映射 | 患者级弱对齐与该手工医学图有效是不同命题；不支持直接沿用该图扩大模型 |

## 1. 对五个设计质疑的核验

- **固定交互与可学习部分：**旧式 rdelta=λ·static cosine·tanh(RNA score×mask) 的特征函数固定，但 Logistic Regression 仍学习其权重、方向和不同交互组合。输入同时依赖患者 RNA 和病理分数，因此有患者差异；它没有在 TITAN 文本编码前学习概念语义适配，也不识别同一 bulk EMT 值背后的细胞来源机制。
- **稀疏映射：**原图确为 15 边、12/32 概念；允许关联掩码未指定正负方向。分数接近零时乘性增量消失，属于公式限制；预设映射未有病例级形态标签验证。
- **RNA/CNA：**旧图两者共享 mask，但基因表达与拷贝数剂量不等价。实际现有状态是 gene-z 值的均值，再 train scaling，不是原始 dosage 的直接平均。
- **λ 和预测约束：**λ 限制输入交互，L2 有助抑制系数；这不构成最终 logit 相对静态 anchor 的显式硬边界。拼接 64 个交互维度也改变全局正则化。
- **任务外推：**TNM/Stage 分类上的不支持不能替代 DSS 条件增量检验；本轮 A/B 分别检验预后与患者级代理语义关联。

## 2. A：固定 Clinical+Static Concepts 的条件预后检验

DSS 原始主终点：512 人/65 事件；三个 seeds 和五患者外折沿用父实验。CS、CRS、C 的旧 OOF 原样复用。
新增四个组合都用相同 Efron ridge Cox，λ=0.01/0.1/1/10，完全相同的保存 inner split、fit-local 预处理与收敛规则；没有 in-sample stacking。整数/基因选择精确一致，浮点 scaler 校验容差 1e-12，实际误差仅 4.5e-16。
“固定”指输入表征、协议和比较基线；联合模型会重新拟合 Clinical/S 权重，并没有冻结 CS 的 risk 作为 offset。因此负结果也可能包含有限样本和联合正则化造成的损失，不是条件互信息检验。

| 模型 | 真正三 seed OOF Harrell | Uno60 | within-fold pair-weighted Harrell |
|---|---:|---:|---:|
| C: Clinical | 0.7767 | 0.7534 | 0.7906 |
| CS: Clinical + static concepts | 0.7968 | 0.7593 | 0.8015 |
| CRS: Clinical + static concepts + RNA Top512 | 0.7979 | 0.8065 | 0.8040 |
| CSM: Clinical + static concepts + RNA states 8D | 0.8024 | 0.7720 | 0.8056 |
| CSCNA: Clinical + static concepts + CNA states 8D | 0.7843 | 0.7635 | 0.7844 |
| CSMC: Clinical + static concepts + RNA/CNA states 16D | 0.7876 | 0.7741 | 0.7907 |
| CRSCNA: Clinical + static concepts + RNA Top512 + CNA states 8D | 0.7943 | 0.8038 | 0.7994 |

每个 seed 先拼五折 OOF 再算 C；这里没有把 15 fold cells 叫 seed mean。Uno60、train-risk 标定和 event/censor 同时点处理与 Phase7A 完全相同；原始 risk 排序的敏感性也公开。

| 主要比较 | ΔHarrell | 95% 条件区间 | 五项族 99% 区间 | ΔUno60 | 三 seed Harrell 同向正 |
|---|---:|---:|---:|---:|---|
| CRS-CS | +0.0011 | [-0.0442,+0.0455] | [-0.0586,+0.0582] | +0.0472 | False |
| CSM-CS | +0.0056 | [-0.0063,+0.0181] | [-0.0103,+0.0218] | +0.0127 | False |
| CSCNA-CS | -0.0125 | [-0.0327,+0.0060] | [-0.0405,+0.0103] | +0.0042 | False |
| CSMC-CS | -0.0092 | [-0.0323,+0.0124] | [-0.0430,+0.0179] | +0.0148 | False |
| CRSCNA-CRS | -0.0036 | [-0.0097,+0.0023] | [-0.0115,+0.0043] | -0.0027 | False |

采用 5000 次 event×fixed-fold 分层、同患者配对 bootstrap，所有模型/seed 使用相同 resamples；区间仅条件于保存的已拟合 OOF，未重训整条 pipeline。
**CRS−CS：**Harrell +0.0011，95% CI [−0.0442,+0.0455]；Uno60 +0.0472，95% CI [−0.0175,+0.1180]。RNA 的 60 月指标点差较大，但区间和主指标都不支持稳定收益。
**CSM−CS：**8D RNA 状态 +0.0056，95% CI [−0.0063,+0.0181]；点值较好、区间较窄，仍未通过，不将它命名为已验证的 molecular mechanism。
**CNA：**CSCNA−CS、CSMC−CS、CRSCNA−CRS 都未通过；这只覆盖现有 gene-set signed-z 均值，不能判定所有 CNA 表示无预后信息。
**最强基线核查：**CRS−C +0.0212 的 95% CI [−0.0289,+0.0756]，CS−C +0.0201 的 95% CI [−0.0135,+0.0551]；旧 CRS−CR 的敏感性优势不等于稳定改善 Clinical 或 CS。

## 3. B：患者级分子—形态代理关联

保留 8 个 RNA 状态、8 个 CNA 状态和冻结 32 个 TITAN cosine scores。在每个 outer-train 拟合 nuisance OLS：Age/Sex/Stage/T/N/M、COAD/READ、GAII-only/HiSeq-only/both RNA 平台；对未见患者产生 residual，除以训练 residual SD，再拼五折。B 无生存结局参与拟合或选择；独立验证器可在不加载任何 DSS/PFS 数组的情况下重现。
患者零模型在 outer-fold×cancer-type×platform 的 24 个区块内置换分子 residual；16 个分子列一起置换，保留其协方差。wrong-patient 99 次区块内 derangement，本数据没有 singleton，512 人均真正错配。
RNA 原图是主对比；CNA 的 8×32 全关联作为独立探索族，不将 RNA prior 的生物解释转给 CNA。15 条原图边采用 max-absolute-correlation 校正；RNA/CNA 全部各 256 对再单独 maxT。

| benchmark | 原图平均 r² | 零模型摘要 | 上尾有限置换 benchmark p |
|---|---:|---|---:|
| 正确患者 vs 区块内患者置换（1999） | 0.02014 | null 均值 0.00197 | 0.0005 |
| 正确患者 vs wrong-patient（99） | 0.02014 | 均值 0.00180，95%分位 0.00482 | 描述性 control |
| 原映射 vs 度数保持随机映射（999 唯一图） | 0.02014 | 均值 0.02753，95%分位 0.03235 | 0.997 |

**患者对应判据通过；映射特异性判据未通过。**多数度数匹配图在相同统计量下比原 15 边更强。它不证明所有外部知识图无效，也不是独立形态真值上的检验。
原图 8 条边有 maxT≤0.05 且 ≥4/5 折同号；其中多条为负相关。不能把它们事后翻成“分子激活就有对应形态”的正向标签。

| 状态 | 病理概念 | 调整后 OOF r | 原图15边 maxT p | 同号外折 |
|---|---|---:|---:|---:|
| proliferation_g2m | C14: mitotic_activity | +0.1281 | 0.0305 | 5/5 |
| invasion_emt | C01: infiltrative_invasion | -0.1390 | 0.0145 | 4/5 |
| invasion_emt | C05: tumor_budding | -0.1574 | 0.0060 | 4/5 |
| invasion_emt | C16: desmoplastic_stroma | -0.0956 | 0.2020 | 3/5 |
| invasion_emt | C17: fibroblastic_stroma | -0.2011 | 0.0005 | 5/5 |
| invasion_emt | C18: collagen_rich_stroma | -0.0390 | 0.9640 | 4/5 |
| immune_ifng | C19: tumor_infiltrating_lymphocytes | -0.0746 | 0.4870 | 3/5 |
| immune_ifng | C20: peritumoral_lymphocytes | -0.0428 | 0.9365 | 4/5 |
| immune_ifng | C21: lymphoid_aggregates | -0.0728 | 0.5205 | 4/5 |
| hypoxia | C12: tumor_necrosis | -0.2071 | 0.0005 | 5/5 |
| hypoxia | C30: perinecrotic_morphology | -0.2018 | 0.0005 | 5/5 |
| tgfb_stroma | C16: desmoplastic_stroma | -0.1183 | 0.0640 | 4/5 |
| tgfb_stroma | C17: fibroblastic_stroma | -0.1582 | 0.0060 | 5/5 |
| tgfb_stroma | C18: collagen_rich_stroma | -0.0947 | 0.2110 | 4/5 |
| angiogenesis | C29: vascular_proliferation | -0.2151 | 0.0005 | 5/5 |

RNA 的 256 对中 81 对有探索性 maxT≤0.05；CNA 为 0 对。大量 RNA 关联分散于图外，提示患者对应可以来自共有表达/形态轴；这些发现未用于挑选或重跑新映射。
负相关不能直接解释为分子过程抑制了形态。可能涉及 conditional adjustment、bulk 细胞组成、不同部位取样或 TITAN 各概念的校准误差；没有患者级对应形态标签可区分这些解释。

## 4. 独立形态记录：C10 黏液形态

- supplementary HISTOLOGICAL_DIAGNOSIS：505 名已知腺癌诊断，66 名含 mucinous，另 7 名未知排除。
- 与主文件 ICD 有 7 处冲突；504 人可比较，其余 ICD 缺失。不更换原诊断标签。
- 冻结 C10 原始 cosine 对独立诊断的零样本 AUC：**0.7204**，2000 次 cancer×diagnosis 分层 bootstrap 95% CI **[0.6424,0.7924]**。
- 诊断/ICD 一致的 497 人（61 黏液性）原始 AUC：**0.7326**。
- 固定 C=1 的 held-out logistic nuisance AUC：0.4640；nuisance+C10：0.5818；两者均低于直接冻结分数的 0.7204。没有据此调参；不声称加 nuisance 一定更好。
**C10 在当前队列有诊断语义支持，但它不是原 15 边覆盖的概念，不能用于验证 invasion/CAF/necrosis 等原图节点。**诊断也不是当前 patch 的盲法人工标注；TITAN 预训练与 TCGA 报告重叠未被排除，所以不是新的外部泛化证据。

## 5. CNA 平均抵消：描述性证据

以下在现有可用 gene-set 基因上比较原始物理 signed dosage 的 |均值| 与均值 |dosage|；不是新 encoder 或新生存输入。它展示原始均值聚合会抵消正负异常，**并未量化实际 gene-z state 的信息损失，也未检验其是否造成 DSS 负结果**。

| 状态 | 原始 mean-absolute dosage 中位数 | 抵消比例中位数 | mean-absolute>0.1 的人数 | 其中抵消≥50%的比例 |
|---|---:|---:|---:|---:|
| proliferation_g2m | 0.222 | 64.9% | 387 | 73.9% |
| invasion_emt | 0.209 | 63.5% | 381 | 69.3% |
| immune_ifng | 0.221 | 57.8% | 386 | 66.8% |
| hypoxia | 0.192 | 65.3% | 377 | 72.1% |
| tgfb_stroma | 0.236 | 65.8% | 395 | 75.2% |
| dna_repair | 0.226 | 56.3% | 391 | 62.1% |
| wnt_beta_catenin | 0.208 | 60.3% | 393 | 63.6% |
| angiogenesis | 0.229 | 62.6% | 401 | 67.6% |

62%–75% 的非平凡原始基因集 dosage 测量存在至少一半幅度抵消，支持将“正负/负荷/基因组结构如何编码”作为单独问题审计；不能从本表断言 RNA/CNA 应生成同一种 Prompt。

## 6. 收敛、修复与独立核验

- 新增 300 次 Cox fit 全部收敛；最大 iterations 390/5000，记录梯度最大 1.86e-06。独立按实际系数计算 Efron 梯度最大 1.15e-06，低于 1e-5。
- 60 份新 OOF 和 coefficients 重构通过，45 份父 OOF 哈希一致，21 个完整 seed OOF。独立指标误差 4.44e-16；预测重构误差 9.99e-16。
- 无 survival 数组的 B 重算、独立 pseudoinverse residual 误差 1.14e-14；512 对 correlations、15 条主边 maxT、999 图唯一性/度数及患者置换样本均验证。
- 首次实现因浮点严格 bit identity 与错误假定 ICD 完全一致而停止，A 没有完成任何 cell。修复仅影响数值核验与形态源 QC，不改变 selection、Bridge、A 模型或 B 关联规则；新目录/manifest 重新锁定，首轮 B 数字没有用于调参。修复记录随报告保留，失败的重复缓存清理。

## 7. 优势、风险与置信度

**优势：**复用已有 CS/CRS；小容量固定候选；相同 inner/final schedule；真实患者 OOF；临床/癌种/平台控制；患者错配和度数保持图控制分开；有独立诊断字段验证一个概念。
**风险：**同队列多轮探索、仅 65 DSS 事件、bootstrap 不覆盖重训方差；A 联合 ridge 可改变 anchor 系数；现有分子压缩与线性头不是所有 Omics 信息的上限。
B 的 cross-fitted residual 区块置换是条件近似参考，依赖 exchangeability；重叠训练折、估计 nuisance 与未测细胞组成/纯度的影响未被完全覆盖。度数保持交换图的有限 Markov 采样也不是所有医学先验的均匀分布。
B 控制 Stage/TNM 可能去除疾病进展介导的关联，所以结论针对这些条件后的剩余关联；不宣称完整分子—形态联系不存在。相关、相似度与疾病真值分别解释。
- **高置信度：**本次数据/代码/OOF/残差与数值核验。
- **中置信度：**当前固定表示和条件下的比较；患者弱对齐存在、原映射缺少特异优势。
- **低置信度：**外部迁移、方向因果、CNA 抵消是否造成 DSS 负结果、复杂新 Bridge 的收益。

## 拓展思考

当前两层证据尚不足以直接扩大生存 Bridge。若继续重设计，先给关键形态概念建立独立标注与校准，再把 RNA expression、CNA 结构证据作为不同来源。可以研究患者弱对齐，但不能将它命名为已成立的这 15 边医学映射机制。

## 你可能想知道

本轮不估计 I(Omics;DSS|Clinical,Pathology) 的真值，也没有测试 token-level TITAN co-adaptation。它回答的是现有冻结表示/指定头能否检测到增量、指定分子状态/概念代理是否有条件关联。GPU 没有重新运行被冻结模型；新增拟合在服务器以 CPU 并行完成。

## 来源与复现

- [Nuisance-aware permutation 的方法边界](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/)：本轮是明确标注的 cross-fitted blocked residual 参考，不宣称实施了完整 Freedman–Lane。
- [EMT/CAF bulk 混杂](https://www.nature.com/articles/s41467-021-22800-1)：不能将 bulk EMT 视为 tumor-cell budding 真值。
- [公开平台/临床来源](https://gdc.cancer.gov/about-data/publications/pancanatlas)。
- 复现入口：`scripts/run_phase7b_bridge_evidence.sh`，依赖已验证 Phase7A/P0/P0R 私有缓存；个体 OOF/residual/权重均留服务器。
