# 官方 HEST CRC 基线阶段交付（2026-10-29）

**本轮已完成 8 组统一官方 BF16 图像特征、4 次正式 Ridge 拟合和 4 次实际模型回放。** 这是固定版本 50-HVG 图像到表达基线的复现，不是新的独立外部验证、完整 SOTA 比较或四程序来源机制的性能结论。

| 任务 | 官方折 0 Mean Pearson | 官方折 1 Mean Pearson | 官方两折聚合 | 两折标准差 |
|---|---:|---:|---:|---:|
| COAD | 0.2980 | 0.3056 | 0.3018 | 0.0038 |
| READ | 0.2601 | 0.1850 | 0.2225 | 0.0376 |

数值严格采用官方 split 返回值先保留四位、再两折平均与最终保留四位的代码规则。标准差描述两折差异，不是置信区间；此处固定 seed=1，不将两折或两个任务当作多次独立种子实验。[完整匿名结果](HEST_OFFICIAL_RIDGE_RESULTS_20261029.json)、[逐折表](HEST_OFFICIAL_RIDGE_FOLD_METRICS_20261029.csv)、[200 行逐基因表](HEST_OFFICIAL_RIDGE_GENE_METRICS_20261029.csv)。

所有预处理只拟合当前官方训练分区，目标为官方基因顺序的 `log1p` 表达，不使用源 pilot 的 CP10K 四程序尺度。输入为冻结 UNI2-h 的原始 1536 维 BF16 前向特征，经训练折 StandardScaler、PCA256 后转 float32，使用固定 Ridge。新官方折没有加载历史六供体适配器。[协议](HEST_OFFICIAL_RIDGE_PROTOCOL.md)。

8 组特征共 24,058 个 patch，实际特征提取没有优化器更新。4 个 Ridge 模型均由保存的 joblib 实际重新加载并前向回放；独立元数据算术审查另核折/基因聚合及 artifact SHA，不能把该审查计作额外模型回放。[独立审查](HEST_OFFICIAL_RIDGE_INDEPENDENT_REVIEW_20261029.json)、[执行账本](HEST_OFFICIAL_RIDGE_EXECUTION_LEDGER_20261029.json)。

公开 provider 名称与历史保护清单重叠，原特征脚本、配置及资格收据的冻结字节保留，公开的是明确标注的无损别名投影。使用前须运行 [恢复工具](../../scripts/materialize_hest8_public_projections.py)，它仅请求两个固定版本的公开目录元数据，不下载 RGB、表达或权重，并逐字节恢复原 SHA；投影不能直接替代原可运行配置。[投影清单](HEST_FEATURE_PUBLIC_PROJECTIONS_20261029.json)、[实际恢复审计](HEST_FEATURE_PUBLIC_PROJECTION_RESTORE_AUDIT_20261029.json)。

需要保留的限制是：COAD 预览样本供体未知；READ 两供体属于历史源开发研究，所以该官方 Ridge 复现不能当作源神经网络的新独立外部确认。READ 的存储坐标为提供方中心的行/列整数像素约定，通用 upper-left 标签不成立；Ridge 不使用坐标，空间 GNN 的 μm 几何资格另待完成。[坐标审计](HEST_COORDINATE_SEMANTICS_AUDIT_20261029.md)。

两个新 GEO 来源仍为 PARTIAL，缺失材料包括供体 crosswalk、可靠列—条码映射或原生图像尺度；没有将旧 ST 测试或研究镜像重新称作独立数据。[来源资格的公开投影](INDEPENDENT_CRC_SOURCE_QUALIFICATION_PUBLIC_PROJECTION_20261029.md)。后续 12 个神经拟合单元的实现/执行单独报告，不进入本轮完成计数，也不因其尚未完成而延后本轮已完成基线的交付。
