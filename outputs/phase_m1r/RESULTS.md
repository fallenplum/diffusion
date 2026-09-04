# Phase M1R Results

本次正式 DSS cohort 按真实 H5 重新构建。COAD/READ 共 600 张 H5，591 个 unique patient；H5 与 Clinical/RNA/CNA 完整交集为 555。DSS-valid canonical cohort 为 512 例、65 events，与 M0R2 预期一致。

## DSS OOF

完整逐模型结果见 `statistical_tests/fold_metrics_all.csv`。固定 seed=20260904 的 OOF 结果：

| model | Harrell C | Uno C |
|---|---:|---:|
| C0 | 0.7823 | 0.7545 |
| C1 | 0.7581 | 0.7264 |
| R0 | 0.6734 | 0.5904 |
| R1 | 0.6088 | 0.5383 |
| N0 | 0.5396 | 0.4560 |
| N1 | 0.5515 | 0.4346 |
| O0 | 0.5414 | 0.4559 |
| O1 | 0.5794 | 0.4736 |
| P0 | 0.6874 | 0.7110 |
| P1 | 0.6855 | 0.6878 |
| PO0 | 0.5695 | 0.4737 |
| PC0 | 0.7194 | 0.7361 |
| OC0 | 0.5828 | 0.4466 |
| POC0 | 0.5746 | 0.4933 |

Strongest unimodal 为 C0；strongest pairwise 为 PC0。POC0 未超过 strongest unimodal 或 strongest pairwise，当前证据不支持三模态互补性。

PFS sensitivity 未完成正式独立 rerun，见 `NEXT_DECISION.md` 和 experiment manifest 的 failed_runs。
