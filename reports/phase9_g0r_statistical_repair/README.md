# Phase9-G0R Statistical Repair

本目录保存 Phase9-G0R 的公开审计产物，不包含患者级 OOF 预测或原始组学数据。

## 修复内容

- R5 的 inner-val epoch 选择和 outer refit 均使用单位邻接矩阵；R6 才使用真实 KEGG 图。
- R7 bootstrap 以五个随机图各自的 paired C-index delta 为基本对象，再对五个 delta 取均值。
- R2 的 inner/outer PCA 都从对应训练折的 Top512 特征重新拟合，不使用 Phase8 缓存。
- 增加 R1−R0 的 Harrell/Uno60 配对比较。
- 图审计报告 215 个映射节点、44 条真实非自环边和 170 个孤立节点；未做事后加边。

## 可复核入口

训练脚本：`scripts/run_phase9_g0r.py`。独立评测脚本：`scripts/validate_phase9_g0r.py`。

患者级运行目录留在服务器并受保护；仓库只提交配置、逐折汇总、配对区间和审计元数据。

