# Phase M5A 结果

完成单元：**105/105**；种子：20260919、20260920、20260921；患者：512；DSS events：65。

## 主结果

| 方法 | Harrell 均值±标准差 | Uno 均值±标准差 | cells |
|---|---:|---:|---:|
| M0 MCAT-Original | 0.7738 ± 0.0747 | 0.7851 ± 0.0845 | 15 |
| M1 ProtoDense | 0.7693 ± 0.0682 | 0.7750 ± 0.0864 | 15 |
| M2 AdaptiveSelective | 0.7749 ± 0.0694 | 0.7857 ± 0.0799 | 15 |
| M3 ProtoSelective | 0.7864 ± 0.0603 | 0.7864 ± 0.0816 | 15 |
| M4 ProtoAntiSelective | 0.7781 ± 0.0632 | 0.7938 ± 0.0765 | 15 |
| B0 CommonOnly-OriginalFusion | 0.7851 ± 0.0767 | 0.8012 ± 0.0761 | 15 |
| C0 ClinicalOnly | 0.7647 ± 0.0598 | 0.7769 ± 0.0834 | 15 |

## 配对比较

| 比较 | ΔHarrell 均值±标准差 | ΔUno 均值±标准差 | H wins/15 | U wins/15 | both/15 |
|---|---:|---:|---:|---:|---:|
| M1 vs M0 | -0.0045 ± 0.0233 | -0.0101 ± 0.0379 | 5 | 5 | 4 |
| M2 vs M0 | 0.0011 ± 0.0237 | 0.0006 ± 0.0316 | 8 | 8 | 7 |
| M3 vs M1 | 0.0171 ± 0.0380 | 0.0114 ± 0.0461 | 11 | 8 | 7 |
| M3 vs M2 | 0.0115 ± 0.0438 | 0.0007 ± 0.0623 | 7 | 8 | 6 |
| M3 vs M0 | 0.0126 ± 0.0438 | 0.0013 ± 0.0628 | 8 | 8 | 4 |
| M3 vs M4 | 0.0083 ± 0.0229 | -0.0074 ± 0.0500 | 8 | 6 | 5 |
| M3 vs B0 | 0.0013 ± 0.0417 | -0.0148 ± 0.0719 | 8 | 5 | 4 |
| M3 vs C0 | 0.0217 ± 0.0477 | 0.0095 ± 0.0847 | 9 | 6 | 6 |

## 七个问题的结论

1. **当前 adaptive pathology tokenizer 的 permutation sensitivity 有多严重？** M0 的每患者风险置换审计为 mean prediction std=0.0012、p95=0.0064、max std=0.0762、max absolute risk difference=0.2185。训练随机顺序与测试 evenly-spaced 顺序不同，属于表示归纳偏差不一致。
2. **Morphology prototypes 是否稳定优于 adaptive pooling？** 不支持：M1 vs M0 的 ΔHarrell=-0.0045、ΔUno=-0.0101，wins=5/15 H、5/15 U。
3. **Selective semantic interaction 是否稳定优于 dense attention？** 不支持：主要比较 M3 vs M1，且同时要求 M3 vs M4 两个均值增益为正。
4. **top-k 是否优于 bottom-k，证明收益来自语义选择而不是单纯 sparsity？** M3 vs M4 的 ΔHarrell=0.0083、ΔUno=-0.0074。
5. **ProtoSelective 是否稳定超过原始 MCAT？** 不支持：M3 vs M0 paired gate 与置换不变性共同判定；M3 max absolute permutation difference=0.0000。
6. **ProtoSelective 是否超过 CommonOnly 与 ClinicalOnly？** M3 vs B0 ΔHarrell=0.0013、ΔUno=-0.0148；M3 vs C0 ΔHarrell=0.0217、ΔUno=0.0095。
7. **下一阶段是否可以把 prototype + semantic selective interaction 固化为新模型 backbone？** 不可以，本阶段预设组合支持门槛未全部通过。

## 审计边界、优势与风险

- 优势：原型拟合严格限制在 outer-train train_pool；M1/M3 使用 permutation-invariant weighted-sum tokenizer；选择性 mask 无新增参数且保持 omic-query→pathology K/V 的 MHA。
- 风险：M0 的 order sensitivity 是诊断结果，不应被解释为数据泄漏；若 gate 未通过，不能将该表示 redesign 固化。
- 置信度：结构审计和 105 单元完整性为 高；机制结论严格服从预设 paired gate，当前为 高。

## 复现边界

本目录只保留提示词要求的 12 个文件；checkpoint、缓存、cell 预测和运行目录不纳入 Git。
