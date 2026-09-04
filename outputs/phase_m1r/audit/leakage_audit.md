# Leakage Audit

- split unit: patient；每个 patient 仅属于一个 outer fold。
- pathology slide mean 在 split 前构建，但仅为无标签的 patch mean；patient fold assignment 不跨 patient。
- RNA/CNA/clinical transforms 在每个 outer train 上单独拟合。
- DSS labels 仅用于 stratification、Cox loss 和评估。
- POC0 label permutation sanity 与 PFS formal rerun 尚未完成，记录为 failed/incomplete。
