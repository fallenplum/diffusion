# Phase6B token-level TITAN prompt audit

本报告对应真正的 token-level prompt：连续 prompt 插入 TITAN 文本序列的 BOS 之后，随后经过冻结的 TITAN text transformer。TITAN 图像编码器、文本编码器和原始静态概念原型均冻结。

## Protocol

- 3 seeds × 5 outer folds；每个 outer-train 使用固定 3 epoch schedule。没有使用 inner-val 标签选择 epoch 或 upstream main effect，避免 inner-val 标签进入配置训练。
- S0 static TITAN concepts；S1 static learnable token prompt；S2 RNA-conditioned token prompt；S3 static concepts + RNA additive；S4 为 S2 的 wrong-patient RNA permutation。
- RNA 只经过 fold-local Top512/512→16 adapter；S2 的 prompt pool 为 4×4×768；T/N/M/Stage 是唯一读取的临床目标。DSS/OS 未读取。
- 每个 seed 先拼接 5-fold OOF，再计算该 seed 的指标；本报告的 seed mean 是 3 个 seed-level OOF 结果的均值，不是 15 个 cells 的直接均值。

## Validation

```json
{
  "status": "PASS",
  "patients": 512,
  "seeds": 3,
  "outer_folds": 5,
  "inner_folds": 3,
  "metrics_rows": 60,
  "pair_rows": 36,
  "oof_files": 240,
  "gpu": "NVIDIA GeForce RTX 4080 SUPER",
  "elapsed_seconds": 12018.880568027496,
  "survival_columns_read": [],
  "text_transformer_trainable_parameters": 0,
  "patient_level_outputs_published": false
}
```

## Seed-level OOF metrics

| Endpoint | Method | Seed mean BA | Seed SD BA | Seed mean macro-F1 | Seed mean AUROC |
|---|---|---:|---:|---:|---:|
| M | S0_static_concepts | 0.5029 | 0.0047 | 0.4703 | NA |
| M | S1_static_token_prompt | 0.5893 | 0.0395 | 0.5687 | NA |
| M | S2_rna_token_prompt | 0.5435 | 0.0329 | 0.5168 | NA |
| M | S3_static_concepts_plus_rna | 0.6520 | 0.0210 | 0.6132 | NA |
| M | S4_wrong_patient_rna_token_prompt | 0.5347 | 0.0224 | 0.5102 | NA |
| N | S0_static_concepts | 0.3610 | 0.0137 | 0.3538 | 0.5455 |
| N | S1_static_token_prompt | 0.4098 | 0.0117 | 0.3676 | 0.5863 |
| N | S2_rna_token_prompt | 0.4217 | 0.0120 | 0.3975 | 0.5995 |
| N | S3_static_concepts_plus_rna | 0.3965 | 0.0055 | 0.3929 | 0.5925 |
| N | S4_wrong_patient_rna_token_prompt | 0.3673 | 0.0019 | 0.3518 | 0.5408 |
| Stage | S0_static_concepts | 0.2771 | 0.0153 | 0.2485 | 0.5545 |
| Stage | S1_static_token_prompt | 0.3621 | 0.0097 | 0.2981 | 0.6022 |
| Stage | S2_rna_token_prompt | 0.3152 | 0.0172 | 0.2661 | 0.5785 |
| Stage | S3_static_concepts_plus_rna | 0.3287 | 0.0118 | 0.3264 | 0.6158 |
| Stage | S4_wrong_patient_rna_token_prompt | 0.2701 | 0.0349 | 0.2271 | 0.5306 |
| T | S0_static_concepts | 0.2553 | 0.0076 | 0.2328 | 0.5541 |
| T | S1_static_token_prompt | 0.3288 | 0.0158 | 0.2812 | 0.6237 |
| T | S2_rna_token_prompt | 0.3526 | 0.0288 | 0.2411 | 0.5823 |
| T | S3_static_concepts_plus_rna | 0.2939 | 0.0114 | 0.2634 | 0.5654 |
| T | S4_wrong_patient_rna_token_prompt | 0.3193 | 0.0597 | 0.2243 | 0.5647 |

## Mechanism paired comparisons

正向 delta 定义为左侧方法 − 右侧方法。支持 token-level patient-specific interpretation 需要同时观察 S2>S1、S2>S3、S2>S4；单独优于 S0 不足以证明该机制。

| Endpoint | Comparison | Mean delta BA | Seed SD | Seed delta range | Every 98.75% CI > 0 |
|---|---|---:|---:|---:|---:|
| M | S2-S1 | -0.0459 | 0.0188 | [-0.0573, -0.0242] | False |
| M | S2-S3 | -0.1075 | 0.0420 | [-0.1558, -0.0792] | False |
| M | S2-S4 | 0.0079 | 0.0409 | [-0.0210, 0.0547] | False |
| N | S2-S1 | 0.0121 | 0.0133 | [-0.0031, 0.0211] | False |
| N | S2-S3 | 0.0251 | 0.0147 | [0.0142, 0.0419] | False |
| N | S2-S4 | 0.0553 | 0.0111 | [0.0461, 0.0677] | False |
| Stage | S2-S1 | -0.0472 | 0.0107 | [-0.0555, -0.0352] | False |
| Stage | S2-S3 | -0.0132 | 0.0272 | [-0.0446, 0.0031] | False |
| Stage | S2-S4 | 0.0454 | 0.0178 | [0.0313, 0.0654] | False |
| T | S2-S1 | 0.0231 | 0.0244 | [0.0005, 0.0490] | False |
| T | S2-S3 | 0.0586 | 0.0395 | [0.0137, 0.0880] | False |
| T | S2-S4 | 0.0335 | 0.0452 | [-0.0155, 0.0736] | False |

## Mechanism verdict

未同时满足 S2>S1、S2>S3、S2>S4 的严格判据；若只有部分 endpoint 为正，应将结论限定为 endpoint-specific evidence，不能宣称机制被整体证实。

## Limitations

- 固定 3 epoch schedule 解决了 inner-val 标签泄漏，但不等同于对所有 endpoint 的最优训练轮数；这是保守的预注册式公平比较。
- 目标是 T/N/M/Stage proxy，而非 DSS survival endpoint；结果不能直接替代生存分析。
- S4 使用 outer-test 上的循环错配 RNA 作为机制对照；它不是独立训练的第五模型。
