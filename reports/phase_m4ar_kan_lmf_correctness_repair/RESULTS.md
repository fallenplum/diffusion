# Phase M4AR KAN / LMF Correctness Repair and Confirmatory Probe

固定协议：512 patients / 65 DSS events、同一 5-fold、同一 fold-local preprocessing、DSS/Cox、UNI2-h K=1024、fresh seeds 20260916/17/18；global OOF 仅作诊断。

## 绝对指标

| Variant | Harrell mean ± std | Uno mean ± std | worst fold H / U | fusion-head params |
|---|---:|---:|---:|---:|
| A0 MCAT-OriginalFusion | 0.7772 ± 0.0633 | 0.7877 ± 0.0614 | 0.6722 / 0.7123 | 56161 |
| A1 MCAT-MatchedMLP | 0.7524 ± 0.0865 | 0.7327 ± 0.1006 | 0.5566 / 0.5201 | 11681 |
| A2 MCAT-KAN-PM-fixed | 0.6939 ± 0.0988 | 0.6904 ± 0.1348 | 0.5855 / 0.3509 | 12649 |
| A2W MCAT-KAN-WM-fixed | 0.7365 ± 0.0789 | 0.7392 ± 0.1138 | 0.6008 / 0.5427 | 58769 |
| A3 MCAT-ReferenceLMF | 0.7587 ± 0.0928 | 0.7408 ± 0.1212 | 0.6075 / 0.5696 | 28713 |
| B0 CommonOnly-OriginalFusion | 0.7720 ± 0.1006 | 0.7756 ± 0.1065 | 0.5115 / 0.4811 | 46753 |
| B1 CommonOnly-MatchedMLP | 0.7232 ± 0.1096 | 0.7355 ± 0.0930 | 0.5509 / 0.5018 | 13729 |
| B2 CommonOnly-KAN-PM-fixed | 0.6886 ± 0.1362 | 0.7048 ± 0.1061 | 0.3597 / 0.5071 | 14953 |
| B2W CommonOnly-KAN-WM-fixed | 0.7318 ± 0.0894 | 0.7657 ± 0.0857 | 0.5332 / 0.5870 | 72593 |
| B3 CommonOnly-ReferenceLMF | 0.7575 ± 0.0667 | 0.7655 ± 0.0826 | 0.5969 / 0.5657 | 37161 |
| C0 ClinicalOnly | 0.7510 ± 0.0865 | 0.7721 ± 0.0880 | 0.5287 / 0.5889 | 0 |

## Paired comparisons

| Comparison | ΔH | ΔU | std ΔH | std ΔU | H wins /15 | U wins /15 | both /15 | supported |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A2 vs A1 | -0.0585 | -0.0423 | 0.1072 | 0.1463 | 4 | 7 | 4 | NO |
| B2 vs B1 | -0.0345 | -0.0307 | 0.1105 | 0.1056 | 7 | 7 | 5 | NO |
| A3 vs A1 | +0.0063 | +0.0081 | 0.0856 | 0.1241 | 9 | 9 | 8 | YES |
| B3 vs B1 | +0.0344 | +0.0300 | 0.1066 | 0.1304 | 7 | 8 | 7 | NO |
| A2W vs A2 | +0.0425 | +0.0488 | 0.0871 | 0.1282 | 11 | 12 | 10 | YES |
| B2W vs B2 | +0.0432 | +0.0608 | 0.1113 | 0.0696 | 9 | 12 | 9 | YES |
| A2W vs A1 | -0.0159 | +0.0065 | 0.0578 | 0.1004 | 3 | 9 | 3 | NO |
| B2W vs B1 | +0.0086 | +0.0301 | 0.0979 | 0.1101 | 8 | 10 | 8 | NO |
| A2 vs A0 | -0.0833 | -0.0974 | 0.1049 | 0.1301 | 3 | 5 | 3 | NO |
| A3 vs A0 | -0.0185 | -0.0469 | 0.0783 | 0.1108 | 6 | 4 | 4 | NO |
| B2 vs B0 | -0.0834 | -0.0708 | 0.1260 | 0.1091 | 4 | 3 | 2 | NO |
| B3 vs B0 | -0.0145 | -0.0101 | 0.1074 | 0.1242 | 5 | 6 | 5 | NO |
| A0 vs C0 | +0.0262 | +0.0157 | 0.0716 | 0.0803 | 9 | 7 | 6 | NO |

## 六个问题的结论

1. **M4A 的 KAN 负结果是否受到 boundary-basis bug 影响？** 本次修复后的 `kan_basis_audit.json` 通过分区统一性、非负性、有限性和梯度测试；因此 M4A 结果不能继续归因于该边界 bug。
2. **修复后的 KAN-PM 是否优于 matched MLP？** MCAT：否（ΔH -0.0585, ΔU -0.0423, H 4/15, U 7/15, both 4/15）；CommonOnly：否（ΔH -0.0345, ΔU -0.0307, H 7/15, U 7/15, both 5/15）。总判定：`KAN_NOT_SUPPORTED_AFTER_CORRECTNESS_REPAIR`。
3. **KAN-WM 是否提示 hidden=8 under-capacity？** MCAT WM vs PM：ΔH +0.0425, ΔU +0.0488, H 11/15, U 12/15, both 10/15；CommonOnly WM vs PM：ΔH +0.0432, ΔU +0.0608, H 9/15, U 12/15, both 9/15。WM 只用于 capacity sensitivity，不能转化为 parameter-efficient claim。
4. **ReferenceLMF 是否改变 M4A 对 Low-Rank 的结论？** MCAT：支持（ΔH +0.0063, ΔU +0.0081, H 9/15, U 9/15, both 8/15）；CommonOnly：不支持（ΔH +0.0344, ΔU +0.0300, H 7/15, U 8/15, both 7/15）。总判定：`LMF_BACKBONE_SPECIFIC_SUPPORTED`。
5. **当前是否仍应放弃 KAN / LMF？** KAN 判定 `KAN_NOT_SUPPORTED_AFTER_CORRECTNESS_REPAIR`；LMF 判定 `LMF_BACKBONE_SPECIFIC_SUPPORTED`。只有满足预设双指标 gate 的机制才保留。
6. **下一阶段是否正式转向 morphology prototype + semantic token representation redesign？** `build next architecture around ReferenceLMF on MCAT representation only; do not claim CommonOnly generalization`。本阶段不加入 prototype、MoE、diffusion、missing modality 或新的 fusion idea。

## 审计边界、优势与风险

- 优势：修复了右边界 basis、采用 ReferenceLMF 的标准 Xavier 初始化、增加了 dense basis 与 synthetic gradient 审计，并保持 nested protocol 不变。
- 风险：KAN-WM 只是宽度敏感性，不可用于参数效率结论；MCAT attention 仍可能触发 PyTorch 已知确定性警告。
- 置信度：correctness 审计和 165 cells 完整性为高；机制结论以 paired gate 为准，不对未支持机制外推。

## 复现入口

本目录保留 prompt 要求的 12 个文件；运行缓存和 cell 预测不纳入 Git。
