# 限制与结论范围

- 单个既有 TCGA 队列已被历史模型探索；新分支/新seed不构成外部确认。只有65事件；seed初始化不增加独立患者数。原五折固定，只有一个内验证划分。
- 主表是fold-local numerator/denominator micro；raw pooled OOF另表，跨折风险尺度不一致，不能将其混成主指标。
- 2000次bootstrap覆盖患者抽样，不重新拟合，未覆盖训练/内层选择方差。12项主family同时区间的尾端仅约4次重采样，Monte Carlo精度有限。D family为6项同时区间；其它补充区间描述性。
- 真正合法的CP_selected可在小inner-val选择到外层较差的A4/A5；不能只用这个较弱选择器宣称胜过所有C+P。A3在同队列历史已见过，是探索性静态强anchor。
- 历史Phase8私有OOF/训练文件不在当前服务器，因此旧anchor不能逐点复算；仅当前45个锚点逐点验证。旧≈0.80585/0.77364与本轮0.80593/0.77372接近，不能据此称bitwise复现。内层seed/选择、float32特征与C-index tie/IPCW查询口径不同；历史文件缺失，不能逐项分配差值。
- 128 patch/slide只占原全patch的一部分；无空间编码，不能等价完整TransMIL、Sheaf或scFoundation。两类patch尺度独立，未作patch-gene对应。
- RNA Top512在已log/重复符号合并矩阵上筛选，不是历史raw行Top512；真实Hallmark组只是来源明确的输入分组，并不证明医学先验成立。C5只有一次锁定匹配置换；平台实际身份未验证。
- B4包含维持容量的额外分支局部映射，不能从其负结果概括所有共享attention。C2清除全部患者RNA，不是纯bottleneck消融。
- 参数注册相同不等于有效容量完全相同：aux heads在D0不训练，D1/D3新增195有梯度参数、D2新增65；C2清零路径可能有零值gradient，日志参与参数是grad非None的计数，不是有效容量证明。E1新增130活gate参数，E0/E1均有其它未用模块。详见MODEL_MATRIX/梯度/模态不变性测试。
- A5/B1的所谓R budget token来源Clinical，不能将其R probe称RNA能力。Frozen probes只针对明确C/R/P tap；不同维数、缩放与固定lambda的解码能力不是互信息或因果模态贡献。梯度范数受参数尺度影响，输出轴梯度夹角不是原文ICG。
- 多数inner loss在40epoch末回升，不支持统一加预算；少数cell选到40，仍需标训练预算敏感。达到锁定预算不等于全局优化收敛；本轮未按test选择增加epoch或补seed。
- 300次真实神经拟合中，261次最后5epoch的inner EMA斜率为正；33次选到最小10epoch，6次选到最大40且尾段EMA仍下降。这6次标为TRAINING_BUDGET_SENSITIVE，不能将所在模块的阴性结果视为已排除所有优化不足。
- 文献级EXACT全部不成立：CiMP全文、BMLSurv全文/源码缺失；E1 confidence没有TCP校准；本轮只ADAPTED/INSPIRED。B6/C6临床文本Prompt与E2 Sheaf明确DEFERRED。
- 固定表示P/R补充在主矩阵完成后为满足基线覆盖锁定，属描述性补充，不加入主要比较族。PFI未实施，DSS没有更换终点。
- 公开包不含患者级预测/标签，外部读者不能只用公开包独立重算。服务器授权环境内独立审计已实跑；公开重算能力BLOCKED_BY_PRIVATE_INPUTS。
- parent已跟踪病例fold/H5清单；新分支私有归档后移除4个病例文件，2个历史聚合文件仅遮去病例名，legacy finalizer改为调用私有原脚本。历史ref/SHA不改，既有Git对象中的旧公开记录仍存在，本次没有执行历史重写。隐私验收针对新公开树与新增包，不宣称消除过去公开暴露。
- 神经head乘0.1是数值缩放而非预测增量的严格上界；最终学习权重可放大风险修正，风险修正SD/max已记录。
- wrong-patient按癌种/缺失模式配对而未按Stage分块；即使正配优于错配，也不能单独排除RNA主效应或分期相关性的解释。
