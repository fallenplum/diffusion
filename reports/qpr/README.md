# QPR-Fusion：范围锁定的关系价值与查询保真审计

本分支按用户提供的压缩包执行，不把其中的创新陈述当成已验证结论。G0完成后，用户已明确授权实际运行门控实验。G3主延迟硬件经用户明确同意改为服务器实际报告的RTX 4080 SUPER；其他门槛不变。不计算置信区间、bootstrap或p值。

## 数据与比较边界

历史固定TCGA COAD/READ 512患者、5外折。G0核验原生CONCH H5及坐标；G1不读取生存结局；G2只解码outer1训练409患者的DSS。两门分别检验当前摘要算法的查询保真，以及完整真实空间关系能否在强Clinical/病理/RNA加性风险之外增加信息。只有G1、G2都通过，才能进入G3；只有G3通过，才能进入G4。

原始真实8邻边在抽样前建立。每张切片最多4096条确定性空间分层边是完整读取教师的参考分布，因此“FULL”指该固定边集的完整读取，并不指未抽样的整张WSI全部边。相同患者多切片等权。

## 可复现入口

所有私有路径均从命令行或服务器私有资产配置提供；不在仓库保存标签数组、逐患者风险、H5或权重。

- G0：`scripts/qpr/audit_native.py`。输出数据、空间、源码和数学资格审计。
- G1准备：`scripts/qpr/prepare_gate1.py`；训练：`scripts/qpr/launch_gate1.py`或受门控的`run_stage.py --stage gate1`；评分：`evaluate_stage.py --stage gate1`。
- G2准备：`scripts/qpr/prepare_gate2.py`；训练：`scripts/qpr/launch_gate2.py`；独立评分：`scripts/qpr/evaluate_gate2.py`。三个入口均在Python函数内部检查G0；未通过时拒绝执行。
- 聚合报告：`scripts/qpr/compile_reports.py --stage gate1`或`gate2`，不进行拟合。
- 发布：`scripts/qpr/package_public.py`采用项目原始隐私扫描器及原始排除名单，显式限定QPR文本文件。

准备和训练前必须先核对相应`G1_PRECOMMIT_LOCK.json`或`G2_PRECOMMIT_LOCK.json`的源码SHA。G0的`PRECOMMIT_LOCK.json`明确标为资格阶段，不冒充完整训练实现锁。保留所有旧锁及软件修复说明；选轮规则、损失、种子和停止门不因结果改变。

G2复用未修改的原R4锚点配方，但训练患者偏置交叉拟合**整个流程，包括监督标量系数**。新拟合156个Cox和12个标量模型。逐患者原始主风险在全部干预中不变；错误RNA只替换Query条件，错误拓扑真实重训，不以仅推理时调包替代。

G2完整Efron风险集采用确定性两遍梯度缓存；微批只用于前向计算。CPU含ties的梯度等价测试及CUDA Query梯度资格均须通过。独立评分需安装`scikit-survival==0.25.0`；本次安装于独立RAM目录，未升级现有训练环境。

## 解释限制

优势是将“关系是否有用”与“压缩是否有效”拆成独立检验，并保留相同强主效应与真实重训负对照。风险是低维PCA、边抽样、有限事件和长期重复开发单队列限制结论。当前数学界是经典有界算子稳定性分析，原创性为`NOT_ESTABLISHED`；任何阶段失败都不能推广为所有生物结构或所有三模态方法无效。没有组织学原图核验，也不声称这些邻边对应真实医学事件。

已完成阶段以各目录`decision.json`为准；运行中和未启动阶段不得以历史结果补齐。G3/G4尚未获得资格时，其指标保持空值。
