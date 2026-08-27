# Phase P1 Stage 0 数据与泄漏审计

## 结论

- 实际纳入 **487** 位患者、**65** 个 DSS events、**494** 张切片。
- UNI2-h 特征维数为 **[1536]**，patch 数范围 **328–31521**，中位数 **4826**。
- 所有纳入 H5 均包含真实 `coords`，与 feature patch 数逐一对齐；坐标来自原始 TCGA H5，未伪造。
- 所有纳入 NPY/H5 feature 均通过 NaN/Inf 检查，无空 bag。
- 患者级 5 折使用历史代码同一 `StratifiedKFold` 逻辑和 seed `20260723`，可精确复现旧划分生成过程。
- 本阶段输入仅来自 `TCGA/`；根目录 `data/` 未作为特征输入。

## 风险与限制

- DSS 事件仅 65/487，统计功效有限，必须同时报告 Uno C 与配对 bootstrap。
- 8GB GPU 下 1024-token Transformer 需 AMP、梯度累积和逐切片 lazy loading。
- 历史患者标签沿用已审计 manifest；其原始来源是缓存 clinical metadata，而不是本阶段无关的 `data/` 数据集。
