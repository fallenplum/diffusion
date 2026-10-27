# 逻辑有效Patch admissions吞吐诊断

**状态：可用完整收据快照；不是所有计划已完成的证明。**

已覆盖 457 个收据/尝试单元；正视觉工作 300，明确零视觉工作 121，NA或未支持 36。

定义：真实mask有效节点按每次视觉前向重复准入的总数，除以对应COMPLETE实测elapsed_seconds。它是冻结调用合同推导的逻辑valid-node吞吐，不是硬件kernel速度、FLOP、去重组织覆盖或实际padding计算量。

## 已核调用合同

- 内层R1：每轮两遍训练风险缓存，共2×80次train；20/40/80各一次train及valid，故163次train、3次valid。
- 内层R3：相同训练/快照调用，再增加6种冻结valid干预，故163次train、9次valid。training_query_mean仅经过omics，不增加视觉工作。
- R2没有path分支，视觉准入数为0；即使曾分配PatchBank也不计成视觉前向。P0/F0/F1/O0等确定性引用为NA，不制造高吞吐排名。
- final_neural源码为160次train加最终预测与6控制共7次test，无中途train/test预测。只有实际COMPLETE且train/test hash与canonical folds一致后才计算吞吐；尚未训练的计划不产生数值。显式缺少outer-private时仍标NOT_SUPPORTED。
- 外部ST source操作单位是spot特征，并且训练timer先于post-fit评估停止；本WSI Patch主表标NOT_APPLICABLE，不混入不同单位的吞吐。

## 数据与计时范围

只读缓存READY、mask.npy、owner.npy及小型私有train/valid索引缓存。首次明确读取原三个以及获授权的R4十二个含测量的既有prepared bundle，经CPU torch.load提取索引；其他数组确实随bundle反序列化，但不分析、不导出。随后报告不再读取bundle。最终outer train/test只访问canonical cohort的folds成员，完全不读取患者名称、终点或风险。没有读取features.npy、原始H5、图像或坐标。

每患者有效节点按owner累加全部原slides，与PatchBank实际输入一致。来源READY、mask/owner、prepared及patient-order均绑定哈希，不输出索引或逐患者节点数。不同预算、选区和UNI2/CONCH分别使用实际训练cache。

单位elapsed依训练driver可能包含bank设置、训练、快照/冻结干预及序列化等开销。因此比值是完整单元口径，不能跨不同计时范围独占解释为算子效率。失败或未完成epoch未知时保留NA，不拿计划80轮补算。

优势：有效mask、真实调用合同和实测单元墙钟可逐项追溯；风险：逻辑节点次数不是profiling trace，空间扫描和padding的真实计算量可能不同，重复epoch也不是新组织。并行单元吞吐不相加充当全GPU的全局速度。
