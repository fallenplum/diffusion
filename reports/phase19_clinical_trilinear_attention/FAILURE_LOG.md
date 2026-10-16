# Phase19 执行与修复记录

## 实际短拟合账本

| id | arm | source | status | generator_steps | discriminator_steps | error_type |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | A3 | ce14bd3053f51bbf277d21392419da70c41208581bb9580d78ca546f953873d5 | RETURNED_SUCCESSFULLY | 2 | 0 |  |
| 2 | A3 | ce14bd3053f51bbf277d21392419da70c41208581bb9580d78ca546f953873d5 | RETURNED_SUCCESSFULLY | 2 | 0 |  |
| 3 | MGCT | ce14bd3053f51bbf277d21392419da70c41208581bb9580d78ca546f953873d5 | RETURNED_SUCCESSFULLY | 2 | 0 |  |
| 4 | CAMR_REPAIRED | ce14bd3053f51bbf277d21392419da70c41208581bb9580d78ca546f953873d5 | RETURNED_SUCCESSFULLY | 2 | 2 |  |
| 5 | MMP | ce14bd3053f51bbf277d21392419da70c41208581bb9580d78ca546f953873d5 | RETURNED_SUCCESSFULLY | 2 | 0 |  |

真实generator/discriminator更新逐step记录；RETURNED_SUCCESSFULLY才表示完整返回，RUNNING或FAILED不能由此推断所有工作已完成。失败但发生更新的记录仍计真实短拟合，不从成功统计中删除。无短拟合checkpoint或患者风险公开/持久化。

## 正式队列

300份完整checkpoint和hash receipt、同源inner epoch选择及五条件回放均已独立核验。当前私有目录未发现FAILED.json调度失败凭证；这描述记录状态，不以它证明不存在任何进程外部故障。

## 已固定的实现资格

simple cache的嵌套NumPy审计先规范为JSON值再比较，并有SHA receipt；非重入checkpoint保留dropout RNG，逐微批native rows/WSI索引闭包显式绑定；外折标签poisoning在新Data构造及短拟合期间持续生效。上述实现已在最终训练source锁之前通过资格测试，正式300模型使用同一锁定source，不覆盖旧实验。

MMP没有KMeans尝试级账本，只能核验已完成缓存producer；失败/中断尝试数量NOT_RECORDED。所有优化、模型引用、凸求解、聚类缓存、冻结干预分别计数。

## MMP只读辅助分析helper修复

| failed_analysis_attempts | scope | cause | repair | private_cache_changed | optimizer_steps | training_source_changed |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | auxiliary label-blind in-memory analysis only; not a model/training failure | centered Gram helper initially mutated a float64 patient-array view; JSON rejected NaN | center a copy; successful outputs recomputed from SHA-verified immutable cache bytes | False | 0 | False |

首次Gram helper对float64患者数组view原位中心化，导致后续概率量出现NaN，严格JSON拒绝输出；改为copy后从SHA核验的磁盘缓存重算全部诊断。磁盘缓存、优化器、训练source与模型均未改变；属于辅助分析失败/修复，不计神经训练失败或新拟合。
