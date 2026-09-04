# Phase M1 Results

本次实验使用固定 `seed=20260904`、patient-level 5-fold outer split，并对 DSS 与 PFS 使用相同 fold assignment。

当前本地完整模态交集为 487 例，DSS 65 events，PFS 126 events。prompt 中预期的 555/512 与 529/135 未在当前病理特征包中复现，因此本结果标记为当前环境 baseline。

## DSS OOF Harrell C

| model | C-index |
|---|---:|
| C0 | 0.786127 |
| R0 | 0.540388 |
| N0 | 0.526036 |
| O0 | 0.543451 |
| P0 | 0.548935 |
| PO0 | 0.556593 |
| PC0 | 0.786918 |
| OC0 | 0.715034 |
| POC0 | 0.716565 |

Clinical 为最强单模态；Pathology+Clinical 未显示实质增量；三模态未超过 strongest unimodal 或 strongest pairwise。

PFS 详细指标见 `pfs_metrics.csv`。所有原始大体积特征、checkpoint 与预测缓存不纳入 git。
