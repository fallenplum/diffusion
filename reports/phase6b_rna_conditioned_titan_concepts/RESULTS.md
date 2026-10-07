# Phase6B RNA-conditioned TITAN concept audit

## 结论

**本轮没有证明 RNA-conditioned prompt 改善了 pathology concept interpretation。** RNA-conditioned concepts 在 T、N、M、Stage 的 seed-level OOF balanced accuracy 均低于 static concepts；与 `static concepts + RNA additive` 的 paired 差异接近 0，98.75% patient bootstrap 区间均跨过 0。因此不能把增量解释为 RNA 改变了 concept 含义。

matched 与 wrong-patient RNA 的差异在四个 endpoint 都是正点估计；N endpoint 的 3 个 seed 区间均为正，Stage 只有一个 seed 区间为正，T/M 区间主要跨 0。这说明 RNA 的患者匹配关系可能携带疾病状态信息，但尚不足以支持完整的 patient-specific concept co-adaptation 机制。

## Protocol audit

- 3 seeds × 5 outer folds；每个 outer-train 内 3-fold inner selection。
- RNA 的缺失填补、方差筛选和标准化只在 inner-train 或最终 outer-train 拟合。
- TITAN 图像/文本编码器、Phase6A 图像表示和 32 个形态概念原型全部冻结。
- 只使用 T/N/M/Stage；未读取 DSS、OS 或任何生存列。
- wrong-patient control 在 outer-test 内循环置换 RNA，匹配模型、图像和分类头保持不变。

## Seed-level OOF balanced accuracy

| Endpoint | Method | Seed mean | Seed SD | Seed range |
|---|---|---:|---:|---:|
| M | rna_conditioned | 0.5619 | 0.0265 | [0.5340, 0.5868] |
| M | static_concepts | 0.6402 | 0.0197 | [0.6211, 0.6606] |
| M | static_concepts_plus_rna | 0.5836 | 0.0034 | [0.5805, 0.5872] |
| M | wrong_patient_rna_conditioned | 0.4935 | 0.0028 | [0.4913, 0.4966] |
| N | rna_conditioned | 0.4121 | 0.0149 | [0.4020, 0.4291] |
| N | static_concepts | 0.4398 | 0.0122 | [0.4262, 0.4497] |
| N | static_concepts_plus_rna | 0.4153 | 0.0075 | [0.4091, 0.4237] |
| N | wrong_patient_rna_conditioned | 0.3279 | 0.0024 | [0.3259, 0.3305] |
| Stage | rna_conditioned | 0.3106 | 0.0106 | [0.3009, 0.3219] |
| Stage | static_concepts | 0.3453 | 0.0062 | [0.3401, 0.3522] |
| Stage | static_concepts_plus_rna | 0.3124 | 0.0110 | [0.3000, 0.3206] |
| Stage | wrong_patient_rna_conditioned | 0.2480 | 0.0060 | [0.2433, 0.2547] |
| T | rna_conditioned | 0.2933 | 0.0140 | [0.2842, 0.3095] |
| T | static_concepts | 0.3698 | 0.0077 | [0.3610, 0.3752] |
| T | static_concepts_plus_rna | 0.2982 | 0.0088 | [0.2889, 0.3063] |
| T | wrong_patient_rna_conditioned | 0.2558 | 0.0091 | [0.2475, 0.2656] |

## Paired mechanism comparisons

这些是 3 个 seed-level OOF paired bootstrap 结果的描述性汇总；区间范围保留每个 seed 的 98.75% 区间边界，不把 15 个 cells 当作 seed。

| Endpoint | Comparison | Mean ΔBA | Seed ΔBA range | CI low range | CI high range |
|---|---|---:|---:|---:|---:|
| M | matched - wrong_patient | 0.0688 | [0.0427, 0.0951] | -0.0350 | 0.1804 |
| M | rna_conditioned - static_concepts | -0.0770 | [-0.0859, -0.0724] | -0.1864 | 0.0416 |
| M | rna_conditioned - static_concepts_plus_rna | -0.0211 | [-0.0460, -0.0001] | -0.1201 | 0.0751 |
| N | matched - wrong_patient | 0.0835 | [0.0741, 0.1003] | 0.0035 | 0.1689 |
| N | rna_conditioned - static_concepts | -0.0278 | [-0.0383, -0.0210] | -0.1187 | 0.0550 |
| N | rna_conditioned - static_concepts_plus_rna | -0.0027 | [-0.0107, 0.0061] | -0.0570 | 0.0587 |
| Stage | matched - wrong_patient | 0.0625 | [0.0543, 0.0785] | -0.0150 | 0.1505 |
| Stage | rna_conditioned - static_concepts | -0.0349 | [-0.0439, -0.0301] | -0.1200 | 0.0440 |
| Stage | rna_conditioned - static_concepts_plus_rna | -0.0018 | [-0.0159, 0.0219] | -0.0621 | 0.0702 |
| T | matched - wrong_patient | 0.0368 | [0.0193, 0.0611] | -0.0546 | 0.1366 |
| T | rna_conditioned - static_concepts | -0.0767 | [-0.0905, -0.0513] | -0.1911 | 0.0228 |
| T | rna_conditioned - static_concepts_plus_rna | -0.0042 | [-0.0145, 0.0035] | -0.0572 | 0.0726 |

## Capacity and limitations

参数量：static=132，RNA-conditioned=29332，additive=8404。动态分支仍是约 2.9 万参数的小模型，但本实现是在冻结 TITAN 输出空间中学习 basis/gate；它不是将连续 soft prompt 送入 TITAN text transformer 的 token-level 实现。

优势：嵌套患者级 OOF、inner-only 选择、同患者 paired wrong-context 对照、冻结 foundation representations、无 survival label。

风险：当前 cohort 没有 pathology concept gold annotations；T/N/M/Stage 是疾病状态代理终点；单队列结果不能证明跨队列泛化；output-space adapter 不能替代真正 token-level prompt tuning。

## 门禁

{
  "status": "PASS",
  "patients": 512,
  "seeds": 3,
  "outer_folds": 5,
  "inner_folds": 3,
  "metrics_rows": 48,
  "pair_rows": 36,
  "oof_files": 180,
  "gpu": "NVIDIA GeForce RTX 4080 SUPER",
  "elapsed_seconds": 853.6969349384308,
  "survival_columns_read": [],
  "server_weight_copy_used": false,
  "patient_level_outputs_published": false
}
