# R5 读出与条件路径诊断

状态：COMPLETE_REQUIRED_R5_TRAINING_AND_REPLAY。仅使用匿名COMPLETE统计；没有反序列化模型、患者风险或原始数组。

R5_READOUT_DIAGNOSTICS.csv分别记录训练/验证的风险SD比、原始方向量、导数及分位统计；统计量跨模型仅作描述，不是方差贡献分解。rank4没有唯一标量z导数，使用差值摘要Jacobian范数，不能拿它与标量导数比值当同量纲贡献。

R5_ATTENTION_DIAGNOSTICS与GRADIENT_DIAGNOSTICS是固定首8人的有限探针；梯度来自全训练风险集Cox的首微批贡献，不混入参数L2，不是完整患者群的组织学真值。

R5_LEARNING_CURVES保留20/40/80轮；正式成绩只取80轮，不从曲线选epoch。R5_SUBGROUPS保存实际验证人数/事件及分组可比较对；跨组与跨fold风险不直接拼接。

R5_FROZEN_CONTROLS分别保留相对CP_SCALAR和相对同一已训模型的直接字段；R5_DIRECT_RETRAINED_PAIRS只来自独立replay导出的真实重训配对。

## 证据覆盖

- readout_rows：216
- attention_rows：144
- gradient_rows：2142
- subgroup_rows：864
- curve_rows：324
- frozen_rows：1512
- direct_retrained_pair_rows：45
- direct_main_pair_rows：36
- direct_retrained_pair_credit_status：AVAILABLE_FROM_INDEPENDENT_REPLAY

优势：读出、幅度、查询、双向消融和容量分开记录；限制：COMPLETE均值不恢复未保存的逐患者交集，注意力或梯度变化本身不证明医学对应。
