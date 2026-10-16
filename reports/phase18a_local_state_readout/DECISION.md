# Phase18A 决策

**STOP_CUSTOM_S6_READOUT_NOT_SUPPORTED**

| comparison | metric | mean_delta | positive_seeds | direction_positive | Bonferroni8_positive |
|---|---|---|---|---|---|
| L0-F0 | Harrell | -0.0018035 | 1 | False | False |
| L0-F0 | Uno60 | -0.00853764 | 1 | False | False |
| L1-F1 | Harrell | -0.00525367 | 1 | False | False |
| L1-F1 | Uno60 | -0.00950299 | 1 | False | False |
| L1-L0 | Harrell | -0.000156826 | 1 | False | False |
| L1-L0 | Uno60 | 6.49859e-05 | 1 | True | False |
| DID=(L1-L0)-(F1-F0) | Harrell | -0.00345017 | 0 | False | False |
| DID=(L1-L0)-(F1-F0) | Uno60 | -0.000965352 | 1 | False | False |

| reference | metric | L1_minus_reference | no_mean_harm |
|---|---|---|---|
| ANCHOR_CS | Harrell | -0.028856 | False |
| ANCHOR_CS | Uno60 | -0.0100039 | False |
| R1_A3_noDirect | Harrell | -0.0131734 | False |
| R1_A3_noDirect | Uno60 | -0.0211645 | False |

- L0-F0 and L1-F1 are not both positive in both Harrell and Uno60
- L1-L0 and DID fail positive two-metric means and/or at least two positive seeds per metric
- L1 does not preserve both-metric mean performance against all locked strong references

校正区间支持的主要统计量：[]

局部读出是已有SSM输出的聚合改动，不作为原创算法。区间条件于已拟合模型；无显著差异不是无信息或等效。完成本轮后停止，Phase18B不自动启动。完整结果、尺度/优化限制和外部验证边界见RESULTS.md。
