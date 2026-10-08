# Phase6B token-level TITAN prompt audit — v2 mask repair

本报告对应真正的 token-level prompt：连续 prompt 插入 TITAN 文本序列的 BOS 之后，随后经过冻结的 TITAN text transformer。TITAN 图像编码器、文本编码器和原始静态概念原型均冻结。

v1 在计算层面完成，但其 CLS padding mask 未随插入 token 移动，且模板融合顺序与官方不同，机制比较已作废。v2 只重跑 S1/S2，复用经输入与逐 cell 校验的 S0/S3；完整 S4 OOF 也已在服务器私有保存。

**结果概览：S2 的三 seed 平均 OOF BA 低于 S1 的 endpoint：M, N, Stage, T；S2−S3 均值为正的 endpoint：N。结论限于本训练预算和 cohort，不能据此直接否定领域先验或 patient-specific semantics 的研究 idea。**

## Protocol

- 3 initialization seeds × 5 outer folds；各模型保持相同的固定 batch shuffle，outer-train 使用固定 3 epoch schedule。没有使用 inner-val 标签选择 epoch 或 upstream main effect。
- S0 static TITAN concepts；S1 static learnable token prompt；S2 RNA-conditioned token prompt；S3 static concepts + RNA additive；S4 为 S2 的 wrong-patient RNA permutation。
- RNA 只经过 fold-local Top512/512→16 adapter；S2 的 prompt pool 为 4×4×768；T/N/M/Stage 是唯一读取的临床目标。DSS/OS 未读取。
- 每个 seed 先拼接 5-fold OOF，再计算该 seed 的指标；本报告的 seed mean 是 3 个 seed-level OOF 结果的均值，不是 15 个 cells 的直接均值。
- 配对 point delta 为精确 OOF BA 差值；2000 次 class-stratified paired patient bootstrap 保持稀少类别计数。98.75% 区间是逐比较的描述性区间，不代表全部 12 个 endpoint/comparison 的联合错误率已控制。
- 计算优化只跳过被 mask 的尾部 padding，并保留原始 CLS 位置；S1 batch 共享相同静态 text prototype。已通过 full/compact 前向和梯度一致性门禁。修复前段已完成的正确 full-padding cells 保留，误差见 encoder audit。

## Validation

```json
{
  "status": "PASS",
  "patients": 512,
  "seeds": 3,
  "outer_folds": 5,
  "inner_folds": 0,
  "metrics_rows": 60,
  "pair_rows": 36,
  "oof_files": 300,
  "gpu": "NVIDIA vGPU-32GB",
  "elapsed_seconds": 1472.7041473388672,
  "survival_columns_read": [],
  "text_transformer_trainable_parameters": 0,
  "patient_level_outputs_published": false,
  "reused_baseline_cells": 120,
  "encoder_audit": {
    "status": "PASS",
    "no_prompt_official_max_abs_error": 0.00012497883290052414,
    "prompt_grad_norm": 33.42402648925781,
    "text_transformer_trainable_parameters": 0,
    "shifted_mask_different_positions": 1496,
    "template_aggregation": "normalize_each_template_then_mean_then_normalize",
    "cls_mask": "BOS + nonpad_prompt + shifted_original_concept_mask",
    "compact_prefix_tokens": 30,
    "original_cls_position": 127,
    "compact_full_embedding_max_abs_error": 0.00011518597602844238,
    "compact_full_gradient_relative_error": 0.0023562577553093433,
    "shared_static_embedding_max_abs_error": 0.00013126805424690247,
    "shared_static_gradient_relative_error": 0.002055846620351076
  }
}
```


独立重算全部 300 个 OOF 文件通过：120 个 S0/S3 原文件 hash 一致，120 个 S1/S2 fold-local fit audit 通过，S4 fixed points = 0。

```json
{
  "status": "PASS",
  "oof_files": 300,
  "patient_count": 512,
  "valid_reused_baseline_cells": 120,
  "fold_local_fit_audits": 120,
  "max_recomputed_metric_error": 8.326672684688674e-17,
  "OOF_patient_exactly_once_per_seed_endpoint_control": true,
  "S4_predictions_persisted": true,
  "wrong_patient_fixed_points": 0,
  "survival_columns_read": [],
  "encoder_gate": {
    "status": "PASS",
    "no_prompt_official_max_abs_error": 0.00012497883290052414,
    "prompt_grad_norm": 33.42402648925781,
    "text_transformer_trainable_parameters": 0,
    "shifted_mask_different_positions": 1496,
    "template_aggregation": "normalize_each_template_then_mean_then_normalize",
    "cls_mask": "BOS + nonpad_prompt + shifted_original_concept_mask",
    "compact_prefix_tokens": 30,
    "original_cls_position": 127,
    "compact_full_embedding_max_abs_error": 0.00011518597602844238,
    "compact_full_gradient_relative_error": 0.0023562577553093433,
    "shared_static_embedding_max_abs_error": 0.00013126805424690247,
    "shared_static_gradient_relative_error": 0.002055846620351076
  }
}
```

## Seed-level OOF metrics

| Endpoint | Method | Seed mean BA | Seed SD BA | Seed mean macro-F1 | Seed mean AUROC |
|---|---|---:|---:|---:|---:|
| M | S0_static_concepts | 0.5029 | 0.0047 | 0.4703 | 0.5159 |
| M | S1_static_token_prompt | 0.5830 | 0.0282 | 0.5796 | 0.6206 |
| M | S2_rna_token_prompt | 0.5506 | 0.0097 | 0.5166 | 0.6145 |
| M | S3_static_concepts_plus_rna | 0.6520 | 0.0210 | 0.6132 | 0.7369 |
| M | S4_wrong_patient_rna_token_prompt | 0.5184 | 0.0611 | 0.4916 | 0.5490 |
| N | S0_static_concepts | 0.3610 | 0.0137 | 0.3538 | 0.5455 |
| N | S1_static_token_prompt | 0.4180 | 0.0076 | 0.3915 | 0.5898 |
| N | S2_rna_token_prompt | 0.3992 | 0.0107 | 0.3845 | 0.5846 |
| N | S3_static_concepts_plus_rna | 0.3965 | 0.0055 | 0.3929 | 0.5925 |
| N | S4_wrong_patient_rna_token_prompt | 0.3473 | 0.0256 | 0.3395 | 0.5262 |
| Stage | S0_static_concepts | 0.2771 | 0.0153 | 0.2485 | 0.5545 |
| Stage | S1_static_token_prompt | 0.3415 | 0.0340 | 0.2960 | 0.5906 |
| Stage | S2_rna_token_prompt | 0.2963 | 0.0109 | 0.2249 | 0.5722 |
| Stage | S3_static_concepts_plus_rna | 0.3287 | 0.0118 | 0.3264 | 0.6158 |
| Stage | S4_wrong_patient_rna_token_prompt | 0.2689 | 0.0063 | 0.2122 | 0.5301 |
| T | S0_static_concepts | 0.2553 | 0.0076 | 0.2328 | 0.5541 |
| T | S1_static_token_prompt | 0.3226 | 0.0245 | 0.2718 | 0.6140 |
| T | S2_rna_token_prompt | 0.2904 | 0.0260 | 0.2484 | 0.5633 |
| T | S3_static_concepts_plus_rna | 0.2939 | 0.0114 | 0.2634 | 0.5654 |
| T | S4_wrong_patient_rna_token_prompt | 0.2899 | 0.0031 | 0.2578 | 0.5575 |

## Mechanism paired comparisons

正向 delta 定义为左侧方法 − 右侧方法。支持 token-level patient-specific interpretation 需要同时观察 S2>S1、S2>S3、S2>S4；单独优于 S0 不足以证明该机制。

| Endpoint | Comparison | Mean delta BA | Seed SD | Seed delta range | Every 98.75% CI > 0 |
|---|---|---:|---:|---:|---:|
| M | S2-S1 | -0.0324 | 0.0353 | [-0.0537, 0.0084] | False |
| M | S2-S3 | -0.1014 | 0.0114 | [-0.1102, -0.0886] | False |
| M | S2-S4 | 0.0322 | 0.0668 | [-0.0356, 0.0980] | False |
| N | S2-S1 | -0.0188 | 0.0035 | [-0.0228, -0.0160] | False |
| N | S2-S3 | 0.0027 | 0.0160 | [-0.0157, 0.0128] | False |
| N | S2-S4 | 0.0519 | 0.0235 | [0.0265, 0.0729] | False |
| Stage | S2-S1 | -0.0452 | 0.0449 | [-0.0822, 0.0047] | False |
| Stage | S2-S3 | -0.0324 | 0.0093 | [-0.0418, -0.0232] | False |
| Stage | S2-S4 | 0.0274 | 0.0151 | [0.0178, 0.0449] | False |
| T | S2-S1 | -0.0322 | 0.0487 | [-0.0872, 0.0055] | False |
| T | S2-S3 | -0.0035 | 0.0283 | [-0.0328, 0.0238] | False |
| T | S2-S4 | 0.0005 | 0.0274 | [-0.0308, 0.0202] | False |

## Mechanism verdict

三项平均配对 delta 均为正的 endpoint：无。 三项差值在每个 seed 的 98.75% CI 均高于零的 endpoint：无。 当前协议未形成稳健的 patient-specific token-conditioning 机制证据；这不等于该 idea 已被证明无效。

## Limitations

- 固定 3 epoch schedule 无 inner-val 标签驱动的选择，但不保证收敛；相同 epoch 数不保证不同容量模型同等充分训练，不能把负结果直接解释为 idea 不成立。
- 目标是 T/N/M/Stage proxy，而非 DSS survival endpoint；结果不能直接替代生存分析。
- S4 使用 outer-test 上的循环错配 RNA 作为机制对照；它不是独立训练的第五模型。
- S1/S2 与 S3 的函数形式和参数量不同；即使控制通过，也尚缺独立形态概念标注来验证语义解释。
- 没有因看到 v1/v2 outer-test 结果而调整学习率、epoch、词表或 prompt 容量。后续协议变更需按探索性或新确认实验解释。
- 复用基线原始 GPU 为 RTX 4080 SUPER，修复计算在 NVIDIA vGPU-32GB 上进行；输入和有效 OOF 文件严格不变，跨 GPU 数值差异没有用于模型选择。

## Trainable parameters

| Endpoint / method | Parameters |
|---|---:|
| M/S0_static_concepts | 66 |
| M/S1_static_token_prompt | 12358 |
| M/S2_rna_token_prompt | 20630 |
| M/S3_static_concepts_plus_rna | 8306 |
| N/S0_static_concepts | 99 |
| N/S1_static_token_prompt | 12391 |
| N/S2_rna_token_prompt | 20663 |
| N/S3_static_concepts_plus_rna | 8355 |
| Stage/S0_static_concepts | 132 |
| Stage/S1_static_token_prompt | 12424 |
| Stage/S2_rna_token_prompt | 20696 |
| Stage/S3_static_concepts_plus_rna | 8404 |
| T/S0_static_concepts | 132 |
| T/S1_static_token_prompt | 12424 |
| T/S2_rna_token_prompt | 20696 |
| T/S3_static_concepts_plus_rna | 8404 |

## Training diagnostics

仅保存修复后 S1/S2 的 training CE；S0/S3 为原始有效 OOF 复用，无原轮次逐 epoch loss。

| Endpoint | Method | Epoch 1 mean CE | Epoch 3 mean CE |
|---|---|---:|---:|
| M | S1_static_token_prompt | 0.6975 | 0.6750 |
| M | S2_rna_token_prompt | 0.6985 | 0.5913 |
| N | S1_static_token_prompt | 1.1028 | 1.0740 |
| N | S2_rna_token_prompt | 1.1013 | 1.0083 |
| Stage | S1_static_token_prompt | 1.3934 | 1.3522 |
| Stage | S2_rna_token_prompt | 1.3958 | 1.3101 |
| T | S1_static_token_prompt | 1.3890 | 1.3291 |
| T | S2_rna_token_prompt | 1.3825 | 1.2586 |

S1/S2 的平均 training CE 在 epoch 1→2→3 均仍下降；第 3 epoch 的 S2 training CE 在各 endpoint 低于 S1，但 OOF BA 没有相应优势。缺少收敛门禁，不能把当前结果直接解释为训练预算已经足够，也不能只据此确定过拟合原因。

**Storage:** TITAN 权重保留在服务器；模型权重、RNA、患者 embeddings、OOF 和逐患者标签均未推送。
