# Phase8-V0R：评测修复与交互优化审计

本阶段追加在 Phase8-V0 之后，旧报告和旧数值未修改。主终点为 TCGA COAD/READ DSS，512 名患者、65 个事件，固定 3 seeds × 5 outer folds。

## 1. 评测修复

Clinical anchor 审计：**PASS**；105 个 arm/seed/fold 的最大逐元素误差 `0`，bitwise equal=True。

|Arm|修复后 within-fold H|修复后 within-fold Uno60|pooled H|pooled Uno60|
|---|---:|---:|---:|---:|
|A0|0.7938|0.7615|0.7764|0.7648|
|A1|0.8043|0.7738|0.7885|0.7792|
|A2|0.7891|0.7609|0.7722|0.7616|
|A3|0.7982|0.7751|0.7814|0.7776|
|A4|0.7938|0.7739|0.7780|0.7754|
|A5|0.8023|0.7895|0.7890|0.7837|
|A6|0.8068|0.7862|0.7912|0.7785|

Uno 修复将每折 IPCW 分子和分母直接合并，同时保留按 outer-train censoring KM 生成的患者权重；旧 Phase8-V0 使用 Harrell pair 数加权折 Uno 的结果继续保留在旧目录。

## 2. A3/A4/A5 训练诊断

|Arm|记录 selected epoch 均值|重放 inner H|outer loss last|最大梯度轨迹|q norm|k norm|interaction head norm|风险修正 max abs|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|A3|28.53|0.8273|5.0302|1.3851|0.0000|0.0000|0.0000|0.2446|
|A4|27.80|0.8241|5.0323|1.3715|0.2144|0.1986|0.0543|0.2227|
|A5|37.40|0.8333|5.0162|1.4179|0.2691|0.2446|0.0763|0.3602|

45 个诊断 cell 均按原记录 epoch 重放，未增加正式预算；未发现 checkpoint，因此内部验证 C-index 来自同 seed/fold 的 exact replay。A4/A5 的 q、k、interaction head 权重均非零。

## 3. 简单 Ridge Cox 对照

|方法|within-fold H|within-fold Uno60|pooled H|pooled Uno60|
|---|---:|---:|---:|---:|
|Clinical+RNA+S32|0.7642|0.7368|0.7632|0.7419|
|Clinical+S32|0.8058|0.7736|0.8009|0.7759|
|JointRidge|0.7455|0.7244|0.7457|0.7220|

### 直接与 A5 比较

|比较|ΔH|H 95% CI|ΔUno60|Uno 95% CI|
|---|---:|---|---:|---|
|Clinical+S32-A5|+0.0035|[-0.0183,+0.0225]|-0.0159|[-0.0530,+0.0191]|
|Clinical+RNA+S32-A5|-0.0381|[-0.0614,-0.0164]|-0.0527|[-0.0962,-0.0118]|
|JointRidge-A5|-0.0568|[-0.0857,-0.0291]|-0.0651|[-0.1115,-0.0179]|

## 4. 决策

- Uno 聚合修复后，旧 Phase8-V0 的 `V0_NOT_CONFIRMED` 结论没有被推翻：A4-A3 的交互增量仍为负，A5-A4 的区间仍跨零。
- `Clinical+S32` 是最强简单对照的点估计之一；它相对 A5 的 Harrell 差异为小幅正值，但区间跨零，不能声称 C3R-Surv 更强。
- `Clinical+RNA+S32` 和 `JointRidge` 均低于 A5，未显示当前全输入线性拼接具有优势。
- **Phase8-V0R 决策：暂不进入新的 V1 模型；先保留 Clinical anchor 主导、交互机制未确认的结论。**

## 限制

65 个 DSS events 使区间较宽；同一 cohort 已被前期研究多次使用，结果属于内部审计，不能作为外部确认。
