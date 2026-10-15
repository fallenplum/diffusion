# Phase15-R0 — RNA 在 Clinical+S32 条件下的互补排序审计

## 问题与锁定范围

RNA有边际预后排序能力，为什么未稳定改善Clinical+S32？本轮检验可迁移的残余排序、错排修复与正确排序破坏的权衡、低容量校准与高维估计方差。不能用非显著结果证明条件独立或唯一归因为65事件。

沿用512人/65 DSS事件；PFI独立次要510人/129事件；固定5个患者外折、原inner split、31/47/83 seeds；TOP/HALL/MATCH三组基因规则不变。DSS/HALL为预先锁定主比较，其他为次要。Clinical仅六个原字段，病理仅冻结TITAN S32。

## 冻结并严格交叉拟合 anchor

每个inner-train或outer-train分别构建整个pipeline。训练分区内3折：

1. 每块 anchor 的临床编码、缺失处理、缩放只用另外2/3患者。
2. λ在该2/3内再做2-fold CV，从[10,1,0.1]选择；不读当前留出块标签。
3. 在留出块预测Clinical+S32风险。训练排序对仅取**同一留出块**内可比较患者对；双方标签都不进入该教师或其模型选择。
4. 对训练分区全量anchor，λ仅通过该训练分区的3-fold CV选择，随后冻结；inner-val只选择RNA修正的正则，不影响upstream anchor。
5. Outer-test及其错误组不参与教师生成、权重生成、超参数选择或新任务定义。

禁止将不同留出块教师产生的风险直接用于跨块排序对；Cox风险存在平移/尺度差异。本轮仅同教师内做pair margins。
现有Phase13强anchor也作为CS_LEGACY原样评价，防止更换anchor训练规则后降低基准。它不参与新教师生成。

## RNA输入与模型对照

保留原始512D，不新建Token或蒸馏任务。每个pipeline沿用对应训练分区的固定基因输入，RNA imputation/scale仅在本pipeline训练分区拟合。标量RNA教师子折另外拟合imputation/scale；其固定基因宇宙来自pipeline训练分区的无监督选择（不是子折重新选基因），此边界须披露。

所有修正使用同一RNA输入及正则候选[0.1,1,10,OFF]，inner-val总Harrell选择，OFF平局优先。各方法独立比较，不从outer结果选赢家。

|模型|训练目标|
|---|---|
|CS / CS_LEGACY|新nested anchor / 原强anchor|
|RNA_ONLY|RNA512 Ridge Cox λ1，边际信息参照|
|JOINT|Clinical+S32+RNA block Ridge Cox，避免risk stacking|
|ALL|冻结anchor margin + RNA修正，所有pair-honest可比较对 logistic loss|
|SOFT|同ALL，以sigmoid(-anchor margin)强调anchor难排患者对|
|HARD|只用anchor margin≤0的对，检验错误聚焦的损伤风险|
|RANDOM|同SOFT，但在每个event owner内部打乱pair权重，保持事件总贡献和权重分布|
|WRONG|同SOFT，但RNA在各train/val/test分区内癌种×缺失代理组derangement|
|NO_OFFSET|同SOFT权重，训练时去掉anchor offset；最终仍将RNA修正加回anchor|
|SCALAR|同ALL，但RNA只有一个交叉拟合线性风险坐标，检查参数方差|

补充JOINT/ALL/SOFT/SCALAR/RANDOM/WRONG的固定λ1版本，区分选择OFF与无信号。标量教师与anchor用相同留出块，训练对双方标签同时排除；仅拼标量，不拼不同模型的latent embeddings。

线性pair objectives为凸函数，L-BFGS及必要Newton polish到梯度阈值；没有神经网络epoch上限疑问。正则只限制系数，不宣称严格限制风险修正幅度。所有方法的源代码、实际梯度、选中λ、fit hash保留。

## held-out机制诊断与统计

配置锁定、所有fit完成后，才用外折结局划分anchor正确/错误患者对，报告RNA-only在各组的排序、模型修复坏对和破坏好对的质量、净增益及事件贡献。外折错误组只用于评价；条件于错误组可能有选择偏差，不等于条件互信息或医学因果机制。

每seed先拼5-fold OOF，以同折比较对分子/分母聚合Harrell/Uno60。IPCW只由outer-train估计。30000共享患者bootstrap，不把pair当独立样本，不平均风险。主6比较×2指标校正12项；其余与PFI为次要。固定模型bootstrap不包含全流程重训练方差。

**优势：** 避免RNA自身排序蒸馏；监督来自anchor诚实错误；原始RNA旁路、凸优化、同输入主效应/权重/身份对照。
**风险：** 每组只约65个总事件，pair-honest分块进一步减少监督对；嵌套anchor估计噪声、低维RNA scalar的校准差异、同cohort重复探索。不同方法可能都不稳定，不能据此否定RNA互补信息。

公开源码与匿名聚合；患者级OOF、教师、pair masks仅服务器私有。先数据/教师泄露硬门，再正式训练；不保存smoke预测。历史正确文件保留，磁盘清理单独记账。

方法依据：[IPCW官方定义](https://scikit-survival.readthedocs.io/en/v0.23.0/api/generated/sksurv.metrics.concordance_index_ipcw.html)、[censored survival ranking](https://arxiv.org/abs/1806.01984)。本轮是自定义机制诊断，不是论文严格复现。
