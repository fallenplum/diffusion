# Phase9-G0R 修复版：KEGG 通路组学公平性审计

本轮保留 G0 原始结果，使用统一 Clinical+S32 锚点、训练折内 Top512、对称 Ridge 网格、inner-val epoch 选择和 degree-preserving 随机图。

|Arm|Harrell|Uno60|
|---|---:|---:|
|O1|0.652160|0.630447|
|O2|0.604956|0.602055|
|O3|0.578295|0.602004|
|R0|0.805850|0.773638|
|R1|0.802243|0.791844|
|R2|0.764212|0.736799|
|R3|0.783816|0.766535|
|R4|0.799420|0.766265|
|R5|0.804125|0.776123|
|R6|0.804360|0.776156|`n|R7(mean5)|0.800847|0.774181|

## 三个预先固定 primary comparisons

|Comparison|Δ Harrell|98.33% CI|Δ Uno60|98.33% CI|
|---|---:|---|---:|---|
|R4-R0|-0.006430|[-0.025967,+0.012373]|-0.007372|[-0.036390,+0.024631]|
|R6-R5|+0.000235|[-0.002768,+0.003147]|+0.000033|[-0.005705,+0.004842]|
|R6-R7|+0.003513|[-0.006784,+0.011939]|+0.001975|[-0.021301,+0.019349]|

## PFS exploratory transfer

PFI/PFS risk is the DSS-trained OOF risk evaluated on PFS labels only; no PFS selection or refit was done. See `pfs_seed_metrics.csv`.

## Decision: NO_GO_STRUCTURE; NO_GO_OMICS_INCREMENT

This is a repair exploration on a repeatedly used TCGA cohort, not external confirmation.

