# 复核与运行边界

本轮只修复 M1–M3，不训练 M4/M5，不计算置信区间、bootstrap 或p值。历史患者级数据、原教师、预测和模型检查点保留在私有实验目录，不进入公开仓库。

## 模型命名

|编号|本轮含义|
|---|---|
|R0|原R4的CP_SCALAR强基线，即M0；复用同训练分区教师|
|R1|原M1-Warm在原锁定选择下的预测；不重新训练|
|R2|M1-R分组正则|
|R3|M1-R分组正则与训练内函数保持|
|R4|本轮等容量Mean Pooling控制，不能与历史R4基线混淆|
|R5|8查询MIL与Mean残差|
|R6|R5共同主干与真实分子条件查询|
|R7|不读取患者RNA的学习查询；本人组学value和主效应保留|
|R8|关闭显式乘积；仍有条件读取，不能称为纯加性模型|
|R9|普通权重共享双向Cross-Attention控制，不是论文官方复现|

R0不是无RNA的纯CP模型。原配方是标准化的Clinical＋S32＋固定CONCH GRID32风险，再加同教师规则选出的独立RNA512标量风险。新网络精确Warm映射这一完整函数。训练内函数保持参考是当前fit的完整R4函数，不把其in-sample预测冒称交叉拟合offset。

## 私有运行

公开的master_public是删去机器路径后的完整科学配置投影，绑定实际执行锁SHA；它不能直接冒充原始私有执行锁。复跑需要相应旧教师、已核验输入资产、原分区及本机路径绑定；公开源码本身不包含受限资产。迁移环境时必须建立新的执行绑定并保留所有科学字段与资产哈希，不能声称新机器路径产生的锁与原执行锁字节相同。

已配好原私有执行环境时使用以下命令；变量值由实验者在私有环境提供。

```powershell
python scripts/semantic_m123r_schedule.py --lock $ExecutionLock --expected-lock-sha256 $ExecutionSHA --private $PrivateRoot --stage qualification --max-parallel 6
python scripts/semantic_m123r_report.py --lock $ExecutionLock --expected-lock-sha256 $ExecutionSHA --private $PrivateRoot --output-newdir $QualificationReports --official-scorer-path $ScorerPath
```

第一轮只解码原outer1训练患者的DSS元素。全部540个拟合各完成140次更新；每个arm/seed对所有配置的三个inner合并可比对分子分母，分别选择EARLY_ANY和ENCODER_TRAINED。后者要求至少20次真实编码器参数变化，不能用第10轮输出头激活模型替代。

只有decision.json允许的候选及其必要对照进入正式阶段；调度器校验资格训练完成凭据和决策哈希。全部五个外折的选择先冻结，最终训练全部结束后，formal报告程序才允许读取外折终点。

```powershell
python scripts/semantic_m123r_schedule.py --lock $ExecutionLock --expected-lock-sha256 $ExecutionSHA --private $PrivateRoot --stage formal --decision $Decision --decision-sha256 $DecisionSHA --max-parallel 6
python scripts/semantic_m123r_report_formal.py --lock $ExecutionLock --expected-lock-sha256 $ExecutionSHA --private $PrivateRoot --qualification-decision $Decision --output-newdir $FormalReports --official-scorer-path $ScorerPath
```

## 评测与训练充分性

Harrell与Uno60都采用折内可比对分子分母合并；每个seed先合并对应折，再对seed等权平均。不同模型或外折的原始风险尺度不保证可直接比较，因此不能直接拼风险后建立跨折患者对。Uno60的删失分布只由当前fit训练患者拟合；使用scikit-survival 0.23.1官方API独立重算。

learning_audit只读取现有内层训练预测和曲线，独立重算各快照更新后的Efron损失，并报告编码器阶段及解冻阶段中训练改善、验证双指标恶化的单元数量。它不改变配置选择或门槛。这些候选反复使用相同患者，不是独立样本计数。

```powershell
python scripts/semantic_m123r_learning_audit.py --lock $ExecutionLock --expected-lock-sha256 $ExecutionSHA --private $PrivateRoot --qualification-report $QualificationReports --output-newdir $LearningReports
python scripts/semantic_m123r_verify_reports.py --report $QualificationReports --output $ArithmeticAudit
```

最后一个程序只读公开汇总表，复核候选覆盖、三内折全局选择、折内加权、种子汇总、修复/损伤恒等式和配对方向，不读取患者数组或权重。

补充的drift_audit是查看已选快照后的解释性诊断：用固定预测分解风险尺度变化、非仿射漂移、Spearman排序和DSS可比对修复/损伤；解析斜率不用于重标定预测或挑选系数，原资格门槛不变。这项分析不能被当成新的独立性能证据。

## 优势与限制

优势是精确保留强初始化、将输出头激活和编码器更新分开、实测结构风险依赖，并保留早停优于充分训练的结果。限制是65个DSS事件与长期队列开发；三内折资格结果存在选择乐观；冻结干预不能替代真实重训；随机身份查询和slot没有组织学真值。资格失败只限制本轮实现，不等于否定所有MIL或分子—病理机制。
