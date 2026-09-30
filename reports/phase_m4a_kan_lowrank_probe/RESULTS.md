# Phase M4A KAN / Low-Rank Fusion Architecture Probe

本报告固定使用 512 patients / 65 DSS events、同一 5-fold、同一 fold-local preprocessing、同一 DSS/Cox、UNI2-h K=1024 与 fresh seeds 20260913/14/15。global OOF 仅作诊断。

## 绝对指标

| Variant | Harrell mean ± std | Uno mean ± std | worst fold H / U | fusion-head params |
|---|---:|---:|---:|---:|
| A0 MCAT-OriginalFusion | 0.7739 ± 0.0796 | 0.7896 ± 0.0819 | 0.6046 / 0.6191 | 56161 |
| A1 MCAT-MatchedMLP | 0.7631 ± 0.0851 | 0.7505 ± 0.0994 | 0.6110 / 0.5856 | 11681 |
| A2 MCAT-CompactKAN | 0.6372 ± 0.1204 | 0.6261 ± 0.1569 | 0.4539 / 0.3256 | 12649 |
| A3 MCAT-LowRankFusion | 0.7352 ± 0.1199 | 0.7710 ± 0.1143 | 0.4604 / 0.5016 | 28777 |
| B0 CommonOnly-OriginalFusion | 0.7870 ± 0.0645 | 0.7809 ± 0.1113 | 0.6862 / 0.4504 | 46753 |
| B1 CommonOnly-MatchedMLP | 0.7311 ± 0.1063 | 0.7509 ± 0.0887 | 0.5106 / 0.5157 | 13729 |
| B2 CommonOnly-CompactKAN | 0.6515 ± 0.1218 | 0.6144 ± 0.1383 | 0.4866 / 0.3393 | 14953 |
| B3 CommonOnly-LowRankFusion | 0.7515 ± 0.0538 | 0.7569 ± 0.0747 | 0.6480 / 0.6599 | 37225 |
| C0 ClinicalOnly | 0.7278 ± 0.1328 | 0.7783 ± 0.1048 | 0.3790 / 0.5712 | 0 |

## Prespecified paired comparisons

| Comparison | ΔH | ΔU | std ΔH | std ΔU | H wins /15 | U wins /15 | both /15 | supported |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A2 vs A1 | -0.1259 | -0.1244 | 0.1058 | 0.1598 | 2 | 3 | 2 | NO |
| A3 vs A1 | -0.0279 | +0.0205 | 0.1169 | 0.1141 | 8 | 10 | 7 | NO |
| B2 vs B1 | -0.0796 | -0.1365 | 0.0776 | 0.1166 | 1 | 1 | 1 | NO |
| B3 vs B1 | +0.0204 | +0.0060 | 0.0719 | 0.1078 | 8 | 6 | 6 | NO |
| A2 vs A0 | -0.1367 | -0.1635 | 0.0898 | 0.1520 | 1 | 0 | 0 | NO |
| A3 vs A0 | -0.0387 | -0.0186 | 0.1321 | 0.1292 | 8 | 7 | 6 | NO |
| B2 vs B0 | -0.1355 | -0.1666 | 0.0982 | 0.1510 | 2 | 1 | 1 | NO |
| B3 vs B0 | -0.0355 | -0.0240 | 0.0532 | 0.0951 | 2 | 5 | 1 | NO |
| B0 vs C0 | +0.0592 | +0.0027 | 0.1056 | 0.0632 | 12 | 8 | 8 | NO |

## 七个问题的结论

1. **KAN 在 MCAT representation 上是否稳定优于 matched MLP？** 否。 A2 vs A1：ΔH -0.1259, ΔU -0.1244, H 2/15, U 3/15, both 2/15。
2. **KAN 在 CommonOnly representation 上是否稳定优于 matched MLP？** 否。 B2 vs B1：ΔH -0.0796, ΔU -0.1365, H 1/15, U 1/15, both 1/15。
3. **Low-Rank Fusion 在两个 backbone 上是否有效？** MCAT：不支持（ΔH -0.0279, ΔU +0.0205, H 8/15, U 10/15, both 7/15）；CommonOnly：不支持（ΔH +0.0204, ΔU +0.0060, H 8/15, U 6/15, both 6/15）。总体判定：`LOWRANK_NOT_SUPPORTED`。
4. **KAN 的收益是否只是参数量增加造成？** 参数匹配审计通过：A2/A1 fusion-head ratio 1.083，B2/B1 ratio 1.089，均在 0.8–1.2 窗口内；因此不能用参数量增加解释 KAN 差异。若对应比较未 supported，则不能宣称 KAN 有效。
5. **哪个机制对 worst-fold / variance 更友好？** 以 paired primary evidence 为准，并结合绝对表中的 fold std 与 worst fold；不能用 diagnostics 替代 Harrell/Uno gate。KAN 判定 `KAN_NOT_SUPPORTED`，Low-Rank 判定 `LOWRANK_NOT_SUPPORTED`。
6. **是否存在一个候选显著优于当前 MCAT / CommonOnly？** 当前 primary mean Harrell 最高的多模态候选为 `B0`；其相对 clinical-only：ΔH +0.0592, ΔU +0.0027, H 12/15, U 8/15, both 8/15。A2/A3 与 A0、B2/B3 与 B0 的直接比较见 paired 表，未达到 gate 的不能称为显著优于。
7. **下一阶段应如何进行？** `do not stack them; return to representation/prototype design`。本阶段不实现最终网络，不加入 prototype、MoE、missing-modality、diffusion 或 imputation uncertainty。

## 审计边界、优势与风险

- 优势：representation、stream projector、Cox、split、preprocessing、sampling、optimizer 和 nested refit 均固定；S1 类比较在同 seed/fold 下配对。
- 风险：KAN 与 Low-Rank 只在已固定 representation 上做 probe；A0/B0 的 original fusion 与 A1/B1 的 matched MLP 不是同一参数化，因此机制判断以 A2/A1、A3/A1、B2/B1、B3/B1 为主。
- 置信度：执行完整性与参数审计为高；跨 backbone 机制结论取决于两个独立 paired gate，未通过 gate 时不外推为通用机制。

## 复现入口

本目录保留规定的 9 个输出文件；运行缓存和 cell 预测不纳入 Git。
