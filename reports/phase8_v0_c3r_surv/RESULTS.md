# Phase8-V0：C3R-Surv 三模态统一协议

本轮固定 DSS、512 患者、65 事件、5 个外层患者折和 3 个 seed。所有 PCA、RNA Top512、CNA/RNA state scaler、Clinical offset 与残差映射均按训练折拟合。

## A0–A6 指标

|Arm|within-fold Harrell mean|SD|within-fold Uno60 mean|pooled raw Harrell|pooled calibrated Harrell|
|---|---:|---:|---:|---:|---:|
|A0|0.7938|0.0102|0.7757|0.7764|0.7870|
|A1|0.8043|0.0134|0.7818|0.7885|0.7983|
|A2|0.7891|0.0092|0.7744|0.7722|0.7821|
|A3|0.7982|0.0084|0.7803|0.7814|0.7924|
|A4|0.7938|0.0103|0.7796|0.7780|0.7878|
|A5|0.8023|0.0063|0.7968|0.7890|0.8005|
|A6|0.8068|0.0055|0.7941|0.7912|0.8022|

## 预注册配对比较

|比较|Δ within-fold Harrell|95% CI|family CI|seed 全正|
|---|---:|---|---|---|
|A3-A0|+0.0044|[-0.0067,+0.0163]|[-0.0091,+0.0192]|False|
|A4-A3|-0.0044|[-0.0137,+0.0055]|[-0.0153,+0.0083]|False|
|A5-A4|+0.0085|[-0.0033,+0.0216]|[-0.0058,+0.0243]|False|
|A6-A5|+0.0045|[-0.0047,+0.0144]|[-0.0065,+0.0172]|False|
|A5-A0|+0.0085|[-0.0068,+0.0262]|[-0.0106,+0.0324]|True|
|A1-A0|+0.0105|[-0.0015,+0.0225]|[-0.0047,+0.0270]|False|
|A2-A0|-0.0047|[-0.0115,+0.0009]|[-0.0136,+0.0020]|False|

## 机制判断

研究门槛状态：**V0_NOT_CONFIRMED**。
- `A3-A0` 加性主效应：未通过。
- `A4-A3` 低秩 S×O 交互：未通过。
- `A5-A4` Clinical residualization：未通过。

这组结果只支持当前冻结表征、低容量网络和统一 Cox 协议下的内部机制判断，不是外部确认或最终 SOTA 排名。Phase M2R2、M3R2、M8-P1R、Phase7A–7D 的历史数字未混入本表。

## 审计状态

- 完成 cell：105/105。
- `A0` outer-test Clinical anchor 在同 seed/fold 被全部 residual arms 数值复用。
- 训练/测试 offset、PCA/scaler、残差投影和 epoch 选择均记录在 cell JSON。
- 失败 cell、NaN、无事件 inner-val、外层测试参与选择都会使 validate 失败。

## 文件

- `source_audit.json`、`data_registry.json`、`baseline_registry.csv`
- `architecture_audit.json`、`leakage_audit.json`、`clinical_anchor_identity.json`、`gradient_audit.json`、`independent_validation.json`
- `fold_metrics.csv`、`seed_metrics.csv`、`paired_comparisons.csv`、`ablation_summary.json`、`compute_profile.csv`
