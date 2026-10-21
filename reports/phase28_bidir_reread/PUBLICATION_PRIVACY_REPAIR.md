# 发布隐私检查的精确字段修复

首次正式发布检查在 `gene_evidence` 键处停止。它在原始数组语义中属于受保护字段，但同名神经模块也出现在已经生成的梯度统计中。

新增许可仅限五条确切的统计路径：

- `rows[].training.data_vs_total_gradient_by_module.gene_evidence`：五个标量训练梯度统计。
- `history.data_versus_total_gradient_by_module.gene_evidence`：五个标量统计及严格80项、epoch按1至80排列的标量梯度范数曲线。
- `arms[].last10_parameter_gradient_norms.gene_evidence`：6个模型的两类梯度范数汇总，以及禁止把范数差当成正则梯度的标记。
- `artifact_audit.rows[].training.last_data_gradient_L2_by_module.gene_evidence`：单个模型最后一步的数据梯度L2标量。
- `artifact_audit.rows[].training.last_total_gradient_L2_by_module.gene_evidence`：单个模型最后一步包含正则项的梯度L2标量。

字段集合、标量类型、有限性、非负性、epoch范围及6模型分母均须符合固定schema。相同键在其他路径、额外字段、原始数值向量、张量、缺项曲线和错误epoch继续拒绝；没有放开一般性的基因数据字段。

本次只改变发布检查器。30个已训练模型、300份风险回放、一次正式评测、预算审计、60份诊断和两份结果报告均保留原字节。追加 `FORMAL_TOOLCHAIN_QUALIFICATION_PUBLICATION.json` 记录当前工具SHA、前序POSTFIT资格及所有已完成科学报告的SHA。
