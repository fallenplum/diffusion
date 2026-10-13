# 数据与特征审计

**本轮固定 512 患者、65 DSS events，同一原始患者五折。** 外折人数 103/103/102/102/102；确切事件数见 fold_metrics.csv。原始 fold CSV 与缓存逐患者一致；患者 hash 和文件 SHA 见 SOURCE_REGISTRY/data_manifest。每个患者所有 slide 固定同一外折，先按 slide 聚合再等权患者聚合，不重复标签。

|输入|实际形状/覆盖|处理及边界|
|---|---|---|
|RNA source|512×20531，309 完整 + 203 每人3024 NA|原表 primary01，重复 primary aliquot 均值，raw cache 完全一致|
|RNA gene|512×20511|删除13无符号行，合并7重复行；float32 log2(x+1) 后转 float64；20 是总减少行数，不是20无符号行|
|CNA source/gene|512×25131 / 512×25101|30重复行合并；原 signed log2 dosage；不重复 log；每人3个原始NA|
|CONCH v1.5|519 H5，512/512患者，原1,278,642 patches|768D，512@20X，H5每文件 SHA、coords整数/唯一与encoder属性实查|
|CONCH训练缓存|512×3×128×768；66,432选定patch|坐标排序后固定均匀选128；519张有效slide均正好128，因此真实有效slide无patch padding|
|UNI2-h|520 H5，patient 512×1536|独立slide均值；256@20X；禁止与CONCH按patch行或相同坐标强配|
|TITAN native|512×768|来源官方 encode_slide 原缓存；本轮train-local PCA32，不是文本对齐向量|
|TITAN S32|512×32|官方视觉投影/L2后与固定32形态文本cosine；并非PCA32；原缓存逐元素一致|
|Clinical|6字段，训练折编码后38–40维|AGE/SEX/Stage/T/N/M；原表与允许补充字段逐元素一致，M有6 UNKNOWN；Age/Sex/Stage敏感性17–18维|

原 RNA 中存在合法小负值，log2 的域均 >−1，未 clipping。当前向量 float32 log2 与历史缓存最大差 1.91e−6，部分小值可达16 ULP；独立 float64 log 后正确舍入的参考与历史缓存最多1 ULP。原矩阵、符号、NA 完全相同，符合数值内核舍入差异，未变更正式特征；详见 raw_source_provenance。CNA collapsed 逐位一致。GAII 平台是缺失图案代理，**独立 patient/aliquot→platform 对照未取得**，不可称已确认平台身份或生物缺失。

所有 imputer/词表/scaler/Top512/PCA/group筛选只fit当前内/外训练集。RNA 要求训练内90%观测率，按训练log表达方差选Top512；每组最多64基因。CNA分8个组的正剂量/负剂量共16维；不是染色体臂负荷，不将signed平均抵消解释为无异常。wrong-patient分别在每个split内按癌种×缺失模式作无固定点置换，无结局输入。

DSS time/status 从原始 PanCancer clinical 表独立重建，一致65 DEAD WITH TUMOR/447 non-event。临床不使用OS/PFS/PFI、未来治疗或DSS。真实测试输入/结局扰动重复完整内外训练后，权重、epoch逐位不变；红队隐名label来源被构造器拒绝。缓存数据/OOF只在服务器私有区。E盘当前环境不可见，但本服务器CONCH已经齐全，本轮无需WSI/权重迁移。
