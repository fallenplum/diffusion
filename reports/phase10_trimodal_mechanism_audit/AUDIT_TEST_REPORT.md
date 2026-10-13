# 正确性实跑报告

正式训练v2锁定hash见SOURCE_REGISTRY，正式前18测试PASS；交付加固代码22测试PASS，输出见unit_test_output.txt；失败后不通过门控继续。

|门控|实跑证据|状态|
|---|---|---|
|原始队列/折/组学|raw_source_provenance.json、data_manifest|PASS；RNA舍入WARN已量化|
|原Clinical/DSS/TITAN|raw_clinical_representation_audit.json|逐元素PASS|
|未来输入红队|clinical构造器拒绝 benign alias→DSS source|PASS|
|合成 fold-local|test_test_perturbation_fitted_state_invariant|train状态/张量严格相同|
|真实完整训练扰动|real_test_perturbation_audit.json|B0两次40epoch选择+13epoch refit；全部参数bitwise相同并匹配原checkpoint|
|Efron ties/extreme/gradient|独立标量oracle、有限差分、Ridge残余梯度|PASS；zero-event拒绝|
|C-index/IPCW|每个正式cell与scikit-survival0.23.1核对|420 PASS，IPCW来自各trainfold；event/censor tie支持|
|bootstrap/statistic|重复患者显式展开对照、同患者counts/allseed共享|PASS；不平均模型风险冒充指标均值|
|attention/QKV/顺序|手工矩阵；patch及slide置换；padding值扰动|PASS|
|MRePath gate|source死gate梯度为None；公式版有梯度；完整risk扰动改变输出|PASS，仅激活不是收益证明|
|去掉RNA不变性|A2/A5/B1/C2输入RNA置换/缩放不改输出|PASS；B0正确匹配输出改变|
|wrong patient|split内无固定点，癌种/缺失模式保持|PASS|
|teacherheldout|所有manifest边界；heldout time更改不改自身OOFteacher|PASS，选择labels亦独立|
|Clinical anchor|独立编码/保存系数/独立目标梯度|45次逐元素最大误差0|
|checkpoint replay|300真实神经cell重新forward；D0另15次B0身份|PASS，见checkpoint_replay_audit|

修复记录：v1数值收敛门失败，原结果归档、同目标Newton polish后全部重跑v2；D0继承文件hash错误仅元数据修复，原记录保存，风险完全不改；临床provenance hook和潜在padding防护交付加固，不改变真实默认输入。真实519slide均完整128patch，padding缺陷未影响正式结果；加固代码回放300checkpoint确认输出一致。全部问题见FAILURES.csv；旧supervisor exit2原样保留，新独立验收PASS，不能把原失败日志删去。
