# 参数预算口径审计

**旧神经COMPLETE.parameters是registered总量；model.pt.active_parameters是requires_grad资格量，两者都不是实测梯度活跃度。**

已按source/SHA核对186个实际产物，其中神经单元162个；CPU读取，前向/反传/新拟合均0。未改旧收据、配置或nested选择。

|stage|arm|cells|registered总量范围|requires_grad范围|冻结范围|新增优化参数范围|
|---|---|---:|---|---|---|---|
|R1|P0|3|NA：见独立coef/reference口径|NA：见独立coef/reference口径|NA：见独立coef/reference口径|0|
|R1|P1|6|55648|55648|0|55648|
|R1|P2|6|64032|64032|0|64032|
|R1|P3|6|85344|85344|0|85344|
|R1|P4|6|93728|93728|0|93728|
|R1|P5|6|85344|85344|0|85344|
|R1|P6|6|85344|85344|0|85344|
|R1|P7_DIVERSE|6|55648|55648|0|55648|
|R1|P7_GRID|6|55648|55648|0|55648|
|R1|P7_RAND|6|55648|55648|0|55648|
|R2|O0|3|NA：见独立coef/reference口径|NA：见独立coef/reference口径|NA：见独立coef/reference口径|0|
|R2|O1|3|NA：见独立coef/reference口径|NA：见独立coef/reference口径|NA：见独立coef/reference口径|20511|
|R2|O2_PCA128|3|NA：见独立coef/reference口径|NA：见独立coef/reference口径|NA：见独立coef/reference口径|128|
|R2|O2_PCA256|3|NA：见独立coef/reference口径|NA：见独立coef/reference口径|NA：见独立coef/reference口径|256|
|R2|O2_PCA32|3|NA：见独立coef/reference口径|NA：见独立coef/reference口径|NA：见独立coef/reference口径|32|
|R2|O3|6|30052|30052|0|30052|
|R2|O4|6|36388|36388|0|36388|
|R2|O5|6|34308|34308|0|34308|
|R2|O6|6|40644|40644|0|40644|
|R2|O7|6|95212|95212|0|95212|
|R2|O8|6|30152|30152|0|30152|
|R2|O9|6|36388|36388|0|36388|
|R3|F0|3|NA：见独立coef/reference口径|NA：见独立coef/reference口径|NA：见独立coef/reference口径|0|
|R3|F1|3|NA：见独立coef/reference口径|NA：见独立coef/reference口径|NA：见独立coef/reference口径|0|
|R3|F10|6|94510|88174|6336|88174|
|R3|F2|6|88932|82596|6336|82596|
|R3|F3|6|92292|85956|6336|85956|
|R3|F4|6|92292|85956|6336|85956|
|R3|F5|6|93604|87268|6336|87268|
|R3|F5_SHUFFLED_MEMBERSHIP|6|93604|87268|6336|87268|
|R3|F5_WRONG_QUERY|6|93604|87268|6336|87268|
|R3|F6|6|93604|87268|6336|87268|
|R3|F7|6|93604|87268|6336|87268|
|R3|F8|6|123300|116964|6336|116964|
|R3|F9|6|104196|97860|6336|97860|

## 四候选参数tie-break

锁定候选F2, F3, F5, F10的registered−active差值为[6336]；逐partition/seed共36个候选两两差值保持完全一致。因此该常数扣除不改变参数大小关系或精确平局；现有选择规则及结果均未重算或修改。

R3冻结部分为path.mil、path.output、omics.risk_head和omics.pool_query。它们属于registered总量但不进入Adam的requires_grad参数集合。其存在不代表对应路径有非零梯度；本报告没有逐参数反传。

## 凸模型、引用与共享成本

新拟合凸Cox只列coefficient维数；P0/O0/F0/F1仅引用共享上游风险，单列参考维数，新增单元优化参数为0。PCA等训练折预处理数组、共同教师Cox和beta、离线基础特征提取器均不混入新增神经参数预算。逐seed/partition及来源SHA见CSV/JSON。

R4 final_neural现源码使用sum(requires_grad)作为parameters字段；这是active口径，与旧core registered主表不同。本审计只核对该源码表达式，没有读取R4外折分数或改动参数tie-break。
