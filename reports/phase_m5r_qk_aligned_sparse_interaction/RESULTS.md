# Phase M5R 结果

完成单元：**120/120**；种子：20260922、20260923、20260924；患者：512；DSS events：65。

## 主结果

| 方法 | Harrell 均值±标准差 | Uno 均值±标准差 | cells |
|---|---:|---:|---:|
| R0 MCAT-Original | 0.7402 ± 0.1049 | 0.7566 ± 0.1017 | 15 |
| R1 Adaptive-QKTop | 0.7386 ± 0.1013 | 0.7589 ± 0.1024 | 15 |
| R2 ProtoDense | 0.7796 ± 0.0911 | 0.7975 ± 0.0957 | 15 |
| R3 Proto-CosineTop | 0.7707 ± 0.0886 | 0.7967 ± 0.0905 | 15 |
| R4 Proto-QKTop | 0.7472 ± 0.0966 | 0.7674 ± 0.1036 | 15 |
| R5 Proto-QKBottom | 0.7713 ± 0.0924 | 0.7937 ± 0.1000 | 15 |
| B0 CommonOnly-OriginalFusion | 0.7661 ± 0.0881 | 0.7680 ± 0.0989 | 15 |
| C0 ClinicalOnly | 0.7646 ± 0.0608 | 0.7775 ± 0.0795 | 15 |

## 配对比较

| 比较 | ΔHarrell 均值±标准差 | ΔUno 均值±标准差 | H wins/15 | U wins/15 | both/15 |
|---|---:|---:|---:|---:|---:|
| R1 vs R0 | -0.0016 ± 0.0163 | 0.0023 ± 0.0190 | 6 | 7 | 5 |
| R2 vs R0 | 0.0393 ± 0.0991 | 0.0409 ± 0.1056 | 11 | 11 | 10 |
| R3 vs R2 | -0.0088 ± 0.0158 | -0.0008 ± 0.0172 | 5 | 6 | 5 |
| R4 vs R2 | -0.0324 ± 0.0933 | -0.0301 ± 0.1010 | 3 | 6 | 3 |
| R4 vs R3 | -0.0236 ± 0.0920 | -0.0293 ± 0.0968 | 6 | 5 | 5 |
| R4 vs R5 | -0.0242 ± 0.0921 | -0.0263 ± 0.0991 | 8 | 6 | 6 |
| R4 vs R0 | 0.0069 ± 0.0350 | 0.0108 ± 0.0365 | 9 | 11 | 9 |
| R4 vs B0 | -0.0189 ± 0.0513 | -0.0006 ± 0.0463 | 4 | 4 | 2 |
| R4 vs C0 | -0.0175 ± 0.0792 | -0.0101 ± 0.0843 | 6 | 7 | 4 |

## 七个问题的结论

1. **Q/K-aligned sparse attention 是否优于原始 dense MCAT？** R1 vs R0：ΔHarrell=-0.0016、ΔUno=0.0023；R4 vs R0：ΔHarrell=0.0069、ΔUno=0.0108。R4 是否满足完整 gate：是。
2. **在 prototype tokens 上，Q/K top-k 是否优于 raw cosine top-k？** 不支持：R4 vs R3 的 ΔHarrell=-0.0236、ΔUno=-0.0293。
3. **Q/K top-k 是否优于 Q/K bottom-k？** R4 vs R5 的 ΔHarrell=-0.0242、ΔUno=-0.0263。
4. **是否证明 M5A 的问题来自 selection metric 未对齐？** 不支持。该结论只比较 R4 与 R3；M5A raw cosine top-k 不能解释为已学习的跨模态语义对齐分数。
5. **Proto-QKTop 是否稳定超过原始 MCAT？** 支持；R4 置换审计 max risk difference=0.0000。
6. **Proto-QKTop 是否稳定超过 CommonOnly / ClinicalOnly？** R4 vs B0：ΔHarrell=-0.0189、ΔUno=-0.0006；R4 vs C0：ΔHarrell=-0.0175、ΔUno=-0.0101。
7. **下一阶段应该怎么做？** 固化 prototype + QK sparse interaction。

## 审计边界、优势与风险

- 优势：selection 直接来自 MCAT coattn 的 Q/K 参数空间；R2/R4 permutation-invariant；Q/K correctness audit 与 control seed audit 独立完成。
- 风险：Q/K 对齐只证明 relevance metric 与实际 MHA 一致，不自动证明稀疏化提高生存预测；所有机制结论仍服从预设 paired gate。
- M5A 解释修正：raw cosine top-k 不是已学习的跨模态语义对齐分数，M5A top-vs-bottom 失败不能推出跨模态互补关系不存在。
- 置信度：若 120/120 且审计均通过，协议完整性为高；性能机制结论仅在对应 gate 通过时成立。

## 复现边界

本目录只保留提示词要求的 14 个文件；checkpoint、缓存、worker 日志和 cell 预测不纳入 Git。
