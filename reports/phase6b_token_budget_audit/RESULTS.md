# Phase6B token-level training-budget audit

本轮保持 S0–S4、模型容量、学习率、batch size、词表和输入不变，仅增加训练预算并在 outer-train 内部选择 epoch。TITAN 完全冻结，T/N/M/Stage 是唯一目标。

## Protocol

- 3 seeds × 5 outer folds × 4 endpoints；每个 outer-train 有一个固定 stratified 80/20 inner split。RNA 预处理和所有 trainable parameters 仅拟合 inner-train；inner-val 只用于 epoch selection。
- 至少观察 30 epochs；120 时未饱和则扩展至 240，240 时未饱和继续至 960，扩展仅依据内部曲线。最多 5-epoch trailing mean 的 balanced CE 最小值决定 epoch，epoch 1–4 使用已有历史长度，不排除早期最佳 epoch；CE 与 BA 均 20 epochs 无实质改善才停止。
- epoch 锁定后，从头在完整 outer-train 重训；相同 optimizer/batch schedule，outer-test 每个模型只预测一次。历史 3-epoch OOF 只在全部训练完成后用于预算配对比较。
- 本 cohort 的旧 outer-test 结果已经被研究过程看过，本轮是内部预算审计，不是全新独立确认。区间是逐比较描述性 bootstrap，不提供全家族错误率保证。

## Computational validation

```json
{
  "status": "PASS",
  "oof_files": 300,
  "metric_rows": 60,
  "cells": 240,
  "fold_local_preprocessing_and_inner_split_audits": 240,
  "outer_test_selection_access": false,
  "survival_columns_read": [],
  "validation_saturated_cells": 240,
  "inner_numerical_plateau_cells": 50,
  "published_patient_data": false
}
```

## OOF balanced accuracy

| Endpoint | Method | 3-seed mean OOF BA | Seed SD |
|---|---|---:|---:|
| M | S0_static_concepts | 0.5722 | 0.0406 |
| M | S1_static_token_prompt | 0.6214 | 0.0221 |
| M | S2_rna_token_prompt | 0.5333 | 0.0282 |
| M | S3_static_concepts_plus_rna | 0.6222 | 0.0267 |
| M | S4_wrong_patient_rna_token_prompt | 0.5162 | 0.0287 |
| N | S0_static_concepts | 0.4325 | 0.0240 |
| N | S1_static_token_prompt | 0.4508 | 0.0199 |
| N | S2_rna_token_prompt | 0.3889 | 0.0141 |
| N | S3_static_concepts_plus_rna | 0.4279 | 0.0194 |
| N | S4_wrong_patient_rna_token_prompt | 0.3537 | 0.0036 |
| Stage | S0_static_concepts | 0.3219 | 0.0058 |
| Stage | S1_static_token_prompt | 0.3775 | 0.0173 |
| Stage | S2_rna_token_prompt | 0.3198 | 0.0105 |
| Stage | S3_static_concepts_plus_rna | 0.3333 | 0.0095 |
| Stage | S4_wrong_patient_rna_token_prompt | 0.2757 | 0.0158 |
| T | S0_static_concepts | 0.3481 | 0.0028 |
| T | S1_static_token_prompt | 0.4197 | 0.0175 |
| T | S2_rna_token_prompt | 0.3013 | 0.0273 |
| T | S3_static_concepts_plus_rna | 0.2935 | 0.0237 |
| T | S4_wrong_patient_rna_token_prompt | 0.3027 | 0.0287 |

## Paired comparisons

| Endpoint | Comparison | Mean delta BA | Every seed CI > 0 |
|---|---|---:|---:|
| M | S2-S1 | -0.0881 | False |
| M | S2-S3 | -0.0889 | False |
| M | S2-S4 | 0.0171 | False |
| M | selected_budget-minus-three_epochs/S0_static_concepts | 0.0693 | False |
| M | selected_budget-minus-three_epochs/S1_static_token_prompt | 0.0384 | False |
| M | selected_budget-minus-three_epochs/S2_rna_token_prompt | -0.0173 | False |
| M | selected_budget-minus-three_epochs/S3_static_concepts_plus_rna | -0.0299 | False |
| M | selected_budget-minus-three_epochs/S4_wrong_patient_rna_token_prompt | -0.0022 | False |
| N | S2-S1 | -0.0620 | False |
| N | S2-S3 | -0.0390 | False |
| N | S2-S4 | 0.0351 | False |
| N | selected_budget-minus-three_epochs/S0_static_concepts | 0.0715 | False |
| N | selected_budget-minus-three_epochs/S1_static_token_prompt | 0.0328 | False |
| N | selected_budget-minus-three_epochs/S2_rna_token_prompt | -0.0103 | False |
| N | selected_budget-minus-three_epochs/S3_static_concepts_plus_rna | 0.0314 | False |
| N | selected_budget-minus-three_epochs/S4_wrong_patient_rna_token_prompt | 0.0064 | False |
| Stage | S2-S1 | -0.0577 | False |
| Stage | S2-S3 | -0.0135 | False |
| Stage | S2-S4 | 0.0441 | False |
| Stage | selected_budget-minus-three_epochs/S0_static_concepts | 0.0449 | False |
| Stage | selected_budget-minus-three_epochs/S1_static_token_prompt | 0.0360 | False |
| Stage | selected_budget-minus-three_epochs/S2_rna_token_prompt | 0.0235 | False |
| Stage | selected_budget-minus-three_epochs/S3_static_concepts_plus_rna | 0.0046 | False |
| Stage | selected_budget-minus-three_epochs/S4_wrong_patient_rna_token_prompt | 0.0068 | False |
| T | S2-S1 | -0.1184 | False |
| T | S2-S3 | 0.0078 | False |
| T | S2-S4 | -0.0014 | False |
| T | selected_budget-minus-three_epochs/S0_static_concepts | 0.0928 | True |
| T | selected_budget-minus-three_epochs/S1_static_token_prompt | 0.0971 | False |
| T | selected_budget-minus-three_epochs/S2_rna_token_prompt | 0.0109 | False |
| T | selected_budget-minus-three_epochs/S3_static_concepts_plus_rna | -0.0004 | False |
| T | selected_budget-minus-three_epochs/S4_wrong_patient_rna_token_prompt | 0.0128 | False |

## Budget and convergence

| Endpoint | Method | Selected epoch median (range) | Inner epochs median | Validation saturated /15 | Numerical train plateau /15 |
|---|---|---:|---:|---:|---:|
| M | S0_static_concepts | 54 (1–357) | 71 | 15 | 2 |
| M | S1_static_token_prompt | 19 (1–76) | 46 | 15 | 0 |
| M | S2_rna_token_prompt | 1 (1–28) | 34 | 15 | 0 |
| M | S3_static_concepts_plus_rna | 1 (1–3) | 30 | 15 | 5 |
| N | S0_static_concepts | 87 (7–149) | 101 | 15 | 13 |
| N | S1_static_token_prompt | 21 (4–65) | 49 | 15 | 1 |
| N | S2_rna_token_prompt | 3 (1–13) | 31 | 15 | 0 |
| N | S3_static_concepts_plus_rna | 1 (1–2) | 30 | 15 | 3 |
| Stage | S0_static_concepts | 86 (6–158) | 105 | 15 | 11 |
| Stage | S1_static_token_prompt | 35 (12–55) | 54 | 15 | 2 |
| Stage | S2_rna_token_prompt | 3 (1–22) | 38 | 15 | 0 |
| Stage | S3_static_concepts_plus_rna | 1 (1–1) | 30 | 15 | 3 |
| T | S0_static_concepts | 95 (13–232) | 111 | 15 | 7 |
| T | S1_static_token_prompt | 31 (8–84) | 61 | 15 | 0 |
| T | S2_rna_token_prompt | 5 (1–50) | 39 | 15 | 0 |
| T | S3_static_concepts_plus_rna | 1 (1–2) | 30 | 15 | 3 |

## Interpretation boundaries

训练 loss 的数值平台、内部验证的收益饱和、完整 OOF 的预算改善和 patient-specific mechanism 分别判读。验证饱和不等于数值优化已达全局最优；增加 epoch 后模型仍弱也不能直接否定医学先验。
只有 S2 同时优于 S1、S3、S4 才形成必要的机制证据。gate entropy、患者差异和概念漂移诊断没有医学形态标注，不能单独证明语义正确。

nested selection 的原则参考 [Cawley and Talbot, JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html)。

TITAN 权重、患者 OOF 和私有 checkpoints 保留于服务器；只发布源码、protocol、aggregate metrics、learning curves 和审计结果。
