# R4 锁定外层评测

状态：COMPLETE_LOCKED_FINALISTS；简单基线已完成 5/5 外折。

每 seed 分别汇总外折内 concordant / eligible，再平均三个 seed；不跨外折比较风险。CP 强锚点的确定性向量跨 seed 引用，不当成三次独立训练。PFI 仅评估冻结的 DSS 风险。

| 模型 | 终点 | Harrell | Uno60 | 最差折 H | 最差折 Uno60 |
|---|---|---:|---:|---:|---:|
| CP_STRONG | DSS | 0.825923 | 0.806707 | 0.759579 | 0.732163 |
| CP_STRONG | PFI | 0.720666 | 0.683313 | 0.650108 | 0.609142 |
| CP_SCALAR | DSS | 0.832667 | 0.812398 | 0.761494 | 0.732425 |
| CP_SCALAR | PFI | 0.728170 | 0.700243 | 0.664147 | 0.644249 |
| NESTED_INTERACTION | DSS | 0.834157 | 0.811839 | 0.760536 | 0.733489 |
| NESTED_INTERACTION | PFI | 0.729516 | 0.703331 | 0.661987 | 0.642783 |

## 同患者配对差值

- DSS：CP_SCALAR−CP_STRONG ΔH=+0.006744，ΔUno60=+0.005691；正向 seed：H 3/3、Uno60 3/3；正向 fold×seed：H 12/15、Uno60 14/15。
- DSS：NESTED_INTERACTION−CP_STRONG ΔH=+0.008233，ΔUno60=+0.005132；正向 seed：H 3/3、Uno60 3/3；正向 fold×seed：H 14/15、Uno60 15/15。
- DSS：NESTED_INTERACTION−CP_SCALAR ΔH=+0.001490，ΔUno60=-0.000559；正向 seed：H 2/3、Uno60 1/3；正向 fold×seed：H 11/15、Uno60 11/15。
- PFI：CP_SCALAR−CP_STRONG ΔH=+0.007504，ΔUno60=+0.016930；正向 seed：H 3/3、Uno60 3/3；正向 fold×seed：H 12/15、Uno60 15/15。
- PFI：NESTED_INTERACTION−CP_STRONG ΔH=+0.008850，ΔUno60=+0.020018；正向 seed：H 3/3、Uno60 3/3；正向 fold×seed：H 15/15、Uno60 15/15。
- PFI：NESTED_INTERACTION−CP_SCALAR ΔH=+0.001346，ΔUno60=+0.003087；正向 seed：H 2/3、Uno60 3/3；正向 fold×seed：H 10/15、Uno60 10/15。

## 解释边界

- 此报告仅包含选择锁允许的简单最终候选；未完成的神经候选不能以此文件表示完成。
- 多次历史开发使用了同一 TCGA 队列；本轮外层拟合隔离不等于新的外部确认队列。
- 不生成置信区间、bootstrap、p 值或显著性结论。正向点估计不自动证明医学机制。
- 风险 SD 比例按折保留，不把不同折任意尺度的风险直接拼接后算总体 SD。
