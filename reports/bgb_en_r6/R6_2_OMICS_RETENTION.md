# R6.2 RNA表征信息保真

状态：**COMPLETE_18_LOGICAL_CELLS_AND_REPLAY**。逻辑18单元，确认新增凸拟合18。

旧RNA512/fullRNA/PCA256按精确SHA及历史实际重放引用，0重拟合。新表征先用当前训练分区拟合观测尺度，未测值mask保留且数值存0。

| 表示 | 完整分区 | Harrell | Uno60 |
|---|---:|---:|---:|
| FIXED_HALLMARK_SUM50 | 3/3 | 0.553482 | 0.583544 |
| FIXED_HALLMARK_MEAN50 | 3/3 | 0.549386 | 0.578622 |
| ST_GENE_UNION_RAW760 | 3/3 | 0.576695 | 0.650317 |
| ST_SIGNED_PROGRAM_MEAN4 | 3/3 | 0.535275 | 0.437094 |
| FROZEN_S7_FLAT256 | 3/3 | 0.555530 | 0.460758 |
| FROZEN_S7_MEAN64 | 3/3 | 0.486345 | 0.438424 |
| LEGACY_FULL_RNA20511_RIDGE | 3/3 | 0.658170 | 0.692661 |
| LEGACY_RNA512_PCA256_UNWHITENED_GENE_L2 | 3/3 | 0.720983 | 0.710068 |
| LEGACY_RNA512_RIDGE | 3/3 | 0.720756 | 0.707561 |

所有新Cox固定lambda1，无超参搜索；sum/mean只在标准化后全部训练/验证值与mask逐元素相同时复用一次拟合。

PCA256旧gene-L2证明只适用于RNA512的正交未whiten坐标。新非线性token的同lambda不代表相同gene-L2几何。

优势是分离支持集、固定聚合与冻结源编码；风险是既有队列反复开发及读出正则几何。结果不能单独证明Clinical+Pathology条件增量或细胞来源。
