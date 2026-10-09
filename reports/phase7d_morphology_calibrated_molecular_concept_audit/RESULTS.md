# Phase7D：Morphology-Calibrated Molecular Concept Audit

目标是检验 RNA/CNA 是否在冻结 TITAN C10 黏液性形态证据之外提供患者级独立信息。C10 为冻结原始 cosine；分类模型固定 C=1，五折外层患者划分不变。

## 主要结果

|模型/比较|AUC 或 ΔAUC|95% CI|99% family CI|
|---|---:|---|---|
|C10_frozen_cosine|+0.7204|[0.6474,0.7920]|—|
|NS|+0.5819|[0.5018,0.6626]|—|
|NSMC|+0.7014|[0.6328,0.7717]|—|
|NSMCx2|+0.6940|[0.6249,0.7626]|—|
|NSMC-NS|+0.1195|[0.0564,0.1835]|[0.0317,0.2059]|
|NSMC-C10_frozen_cosine|-0.0190|[-0.1032,0.0641]|[-0.1335,0.0886]|
|NSMCx2-NSMC|-0.0074|[-0.0246,0.0089]|[-0.0305,0.0137]|
|NSMCx2-NS|+0.1121|[0.0508,0.1711]|[0.0308,0.1928]|
|NSMC_wrong_patient_mean|+0.5025|[0.4428,0.5748]|—|

## 解释

- `NSMC−NS` 直接回答分子状态是否超过 Clinical+C10 的简单加性模型。
- `NSMC−C10_frozen_cosine` 直接回答完整条件模型是否超过冻结 C10 零样本锚点。
- `NSMCx2` 只增加两个预注册交互维度；若其相对 NSMC 的区间跨零，不支持条件交互机制。
- wrong-patient 只置换外层测试中的 MC16，Clinical 和 C10 保持原患者配对；其高低必须结合临床基线解读。

## 限制

独立标签是患者级 supplementary HISTOLOGICAL_DIAGNOSIS，不是盲法 patch 标注；TITAN 预训练与 TCGA 重叠未完全排除。

## 文件

- `D_metrics.csv`
- `D_fit_audit.csv`
- `D_decision.json`
- `DONE.json`
