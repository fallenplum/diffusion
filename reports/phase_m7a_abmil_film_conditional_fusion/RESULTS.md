# Phase M7A 结果

## Motivation lock（实验前固定）

M5A/M5R/M6A 的 prototype effect 只作初始化敏感诊断，不能确认 prototype backbone；M6A 的 50-pathway tokenization 没有显示稳定语义优势，CNA coupling gate 未通过；dense token co-attention 接近均匀权重，这是结构诊断而非信息论 collapse proof；M6A MCAT6 约 1.25M 参数、Hallmark50RC 约 2.41M，本阶段保持紧凑交互。

## 主结果

| 方法 | Harrell 均值±标准差 | Uno 均值±标准差 | cells |
|---|---:|---:|---:|
| F0 MCAT-Original | 0.7731 ± 0.0800 | 0.7824 ± 0.0838 | 15 |
| F1 ABMIL-SemanticConcat | 0.7906 ± 0.0768 | 0.7983 ± 0.0801 | 15 |
| F2 ABMIL-MatchedResidualMLP | 0.7871 ± 0.0770 | 0.7966 ± 0.0848 | 15 |
| F3 ABMIL-OmicToPath-FiLM | 0.7921 ± 0.0759 | 0.8016 ± 0.0814 | 15 |
| F4 ABMIL-Bidirectional-FiLM | 0.7896 ± 0.0735 | 0.7968 ± 0.0781 | 15 |
| F5 ABMIL-RawOmic-FiLM | 0.7875 ± 0.0747 | 0.8073 ± 0.0678 | 15 |
| B0 CommonOnly-OriginalFusion | 0.8026 ± 0.0741 | 0.8114 ± 0.0713 | 15 |
| C0 ClinicalOnly | 0.7622 ± 0.0817 | 0.7822 ± 0.0973 | 15 |

## 配对比较

| 比较 | ΔHarrell 均值±标准差 | ΔUno 均值±标准差 | H wins/15 | U wins/15 | both/15 |
|---|---:|---:|---:|---:|---:|
| F1 vs F0 | 0.0175 ± 0.0571 | 0.0159 ± 0.0628 | 7 | 7 | 4 |
| F3 vs F1 | 0.0015 ± 0.0021 | 0.0032 ± 0.0096 | 9 | 9 | 8 |
| F3 vs F2 | 0.0050 ± 0.0118 | 0.0050 ± 0.0315 | 7 | 6 | 5 |
| F4 vs F3 | -0.0025 ± 0.0081 | -0.0048 ± 0.0130 | 4 | 5 | 4 |
| F3 vs F5 | 0.0046 ± 0.0451 | -0.0058 ± 0.0312 | 8 | 6 | 6 |
| F3 vs F0 | 0.0190 ± 0.0573 | 0.0192 ± 0.0666 | 7 | 7 | 4 |
| F4 vs F0 | 0.0165 ± 0.0553 | 0.0144 ± 0.0613 | 8 | 7 | 5 |
| F3 vs B0 | -0.0105 ± 0.0517 | -0.0099 ± 0.0487 | 7 | 6 | 5 |
| F4 vs B0 | -0.0130 ± 0.0486 | -0.0147 ± 0.0420 | 7 | 5 | 4 |
| F3 vs C0 | 0.0299 ± 0.0557 | 0.0194 ± 0.0736 | 11 | 7 | 7 |
| F4 vs C0 | 0.0275 ± 0.0589 | 0.0146 ± 0.0796 | 11 | 7 | 7 |

## 审计边界、优势与风险

- 优势：ABMIL 使用集合加权和并通过结构与置换测试验证；F1-F4 的共享上游和 late fusion 做了 byte-level initialization audit；FiLM 使用 zero-init residual conditioning。
- 风险：prototype 的跨阶段 45 个 paired cells 不是独立样本；FiLM 调制幅度是诊断，不用于调参；确定性算法仍可能出现 PyTorch backend 警告。F5 按固定 compact encoder 公式实现，active 参数量与 F3 的六签名分支有差距；F3/F5 属于结构对照诊断，不能单独证明生物语义的因果贡献。
- 置换审计通过=True；FiLM correctness=PASS；初始化审计=PASS。

## 复现边界

只提交 prompt 要求的 14 个文件；缓存、checkpoint、worker 日志和 cell 预测不进入 Git。

## 八个问题的结论

1. **跨 M5A/M5R/M6A 后，prototype 是否应正式放弃作为主 pathology tokenizer？** 45 个 paired cells：ΔHarrell=0.0099、ΔUno=0.0108、wins=23/45 H、22/45 U、both=18/45；median ΔHarrell=0.0010、median ΔUno=-0.0003；seed-level 两指标均为正=False。结论：prototype effect is initialization-sensitive and not confirmed。
2. **Gated ABMIL 是否稳定优于 MCAT 的 adaptive pathology tokenizer？** 不支持；F1 vs F0：0.0175 ± 0.0571 | 0.0159 ± 0.0628 | 7 | 7 | 4。
3. **Omic→Path FiLM 是否稳定优于 simple concat？** 支持；F3 vs F1：0.0015 ± 0.0021 | 0.0032 ± 0.0096 | 9 | 9 | 8。
4. **FiLM 是否优于参数匹配的 generic residual MLP？** F3 vs F2：0.0050 ± 0.0118 | 0.0050 ± 0.0315 | 7 | 6 | 5；独立 gate=支持。
5. **Bidirectional FiLM 是否优于更简单的单向 FiLM？** 不支持；F4 vs F3：-0.0025 ± 0.0081 | -0.0048 ± 0.0130 | 4 | 5 | 4。
6. **MCAT six-signature semantics 在 FiLM 条件下是否仍有贡献？** 不支持；F3 vs F5：0.0046 ± 0.0451 | -0.0058 ± 0.0312 | 8 | 6 | 6。
7. **最佳 FiLM variant 是否稳定超过 MCAT、CommonOnly、ClinicalOnly？** F3 vs F0/B0/C0：0.0190 ± 0.0573 | 0.0192 ± 0.0666 | 7 | 7 | 4；-0.0105 ± 0.0517 | -0.0099 ± 0.0487 | 7 | 6 | 5；0.0299 ± 0.0557 | 0.0194 ± 0.0736 | 11 | 7 | 7。F4 vs F0/B0/C0：0.0165 ± 0.0553 | 0.0144 ± 0.0613 | 8 | 7 | 5；-0.0130 ± 0.0486 | -0.0147 ± 0.0420 | 7 | 5 | 4；0.0275 ± 0.0589 | 0.0146 ± 0.0796 | 11 | 7 | 7。F3 全部 context gate=False；F4 全部 context gate=False。best 只作描述，不新建 post-hoc gate。
8. **是否可以将 permutation-invariant supervised pathology pooling + semantic omics + feature-wise conditional modulation 固化为下一阶段 backbone？** 不可以；只在对应预注册 gate 通过时保留机制。
