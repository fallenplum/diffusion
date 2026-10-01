# Phase M6A 结果

## M5R interpretation 修正

M5R `run_manifest` 的 `combined_backbone_candidate=true` 不能作为下一阶段依据。M5R 中 QK selective gate 未通过，R4 低于 R2 且低于 B0/C0；本阶段不冻结 QK-sparse backbone。

完成单元：**120/120**；种子：20260925、20260926、20260927；患者：512；DSS events：65。

## 主结果

| 方法 | Harrell 均值±标准差 | Uno 均值±标准差 | cells |
|---|---:|---:|---:|
| G0 MCAT6-AdaptiveDense | 0.7791 ± 0.0950 | 0.7650 ± 0.1320 | 15 |
| G1 MCAT6-ProtoDense | 0.7740 ± 0.0904 | 0.7666 ± 0.1285 | 15 |
| G2 Hallmark50RNA-ProtoDense | 0.7701 ± 0.0923 | 0.7668 ± 0.0976 | 15 |
| G3 Hallmark50RC-ProtoDense | 0.7559 ± 0.0850 | 0.7462 ± 0.1021 | 15 |
| G4 Shuffled50RC-ProtoDense | 0.7541 ± 0.0853 | 0.7483 ± 0.1021 | 15 |
| G5 Hallmark50RC-AdaptiveDense | 0.7704 ± 0.0866 | 0.7873 ± 0.0862 | 15 |
| B0 CommonOnly-OriginalFusion | 0.7823 ± 0.0739 | 0.7868 ± 0.0771 | 15 |
| C0 ClinicalOnly | 0.7682 ± 0.0955 | 0.7862 ± 0.0902 | 15 |

## 配对比较

| 比较 | ΔHarrell 均值±标准差 | ΔUno 均值±标准差 | H wins/15 | U wins/15 | both/15 |
|---|---:|---:|---:|---:|---:|
| G1 vs G0 | -0.0051 ± 0.0394 | 0.0015 ± 0.0519 | 7 | 6 | 4 |
| G2 vs G1 | -0.0039 ± 0.0497 | 0.0003 ± 0.0717 | 6 | 5 | 4 |
| G3 vs G2 | -0.0142 ± 0.0830 | -0.0206 ± 0.0932 | 6 | 6 | 5 |
| G3 vs G4 | 0.0018 ± 0.0186 | -0.0021 ± 0.0144 | 8 | 8 | 6 |
| G3 vs G1 | -0.0181 ± 0.0715 | -0.0204 ± 0.1138 | 6 | 6 | 5 |
| G3 vs G5 | -0.0145 ± 0.0342 | -0.0411 ± 0.0664 | 5 | 4 | 4 |
| G3 vs G0 | -0.0232 ± 0.0798 | -0.0188 ± 0.1231 | 5 | 6 | 4 |
| G3 vs B0 | -0.0264 ± 0.0587 | -0.0406 ± 0.0854 | 6 | 5 | 4 |
| G3 vs C0 | -0.0123 ± 0.0733 | -0.0400 ± 0.0859 | 9 | 7 | 7 |

## 审计边界、优势与风险

- 优势：所有主变体使用 dense MCAT-style omic-query → pathology co-attention；G3/G4 参数计数相同；Hallmark 与 shuffled assignment 均 outcome-blind 且 fold-local。
- 风险：G3/G4 语义 gate、G3/G2 CNA coupling gate 和最终候选 gate 均需按表格结果解释，不能用 global OOF diagnostic 替代 fold-level paired gate。
- permutation：G1 max risk difference=4.768e-07，G3 max risk difference=4.768e-07，要求通过=True。
- 审计状态：initialization=True，control initialization=True；协议完整性置信度=高。

## 复现边界

输出目录只保留 prompt 要求的 14 个文件；checkpoint、缓存、worker 日志和 cell 预测不纳入 Git。

## 七个问题的结论

1. **M5A+M5R 合并后 prototype effect 是否稳定？** ΔHarrell=0.0174、ΔUno=0.0154，Harrell wins=16/30、Uno wins=16/30、both=14/30；按 seed 的五折摘要保留在 `cross_phase_prototype_reaggregation.csv`，该诊断不视为独立样本。
2. **50 Hallmark RNA tokens 是否优于 MCAT 6 signatures？** G2 vs G1：ΔHarrell=-0.0039、ΔUno=0.0003。
3. **在 Hallmark pathway 内加入 CNA 是否有稳定贡献？** 不支持；G3 vs G2：-0.0142 ± 0.0830 | -0.0206 ± 0.0932 | 6 | 6 | 5。
4. **Hallmark50-RC 是否优于 size-matched shuffled control？** 不支持；G3 vs G4：0.0018 ± 0.0186 | -0.0021 ± 0.0144 | 8 | 8 | 6。
5. **相同 Hallmark50-RC 下 prototype 是否优于 adaptive pathology？** G3 vs G5：-0.0145 ± 0.0342 | -0.0411 ± 0.0664 | 5 | 4 | 4。
6. **G3 是否稳定超过原始 MCAT、CommonOnly、ClinicalOnly？** G3 vs G0：-0.0232 ± 0.0798 | -0.0188 ± 0.1231 | 5 | 6 | 4；G3 vs B0：-0.0264 ± 0.0587 | -0.0406 ± 0.0854 | 6 | 5 | 4；G3 vs C0：-0.0123 ± 0.0733 | -0.0400 ± 0.0859 | 9 | 7 | 7。
7. **是否进入正式新架构整合？** 暂不进入正式整合；representation hypothesis 尚未同时满足候选 gate。
