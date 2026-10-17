# Phase21 — S16 行内修正、消息传播与配对排序审计

## 结论边界

**共同偏移确实被归一化删除，但“概率重分配互相抵消，导致风险信号完全消失”没有被完整证实。** 原模型的相对差异、注意力变化和最终修正均非零。query 平均与风险投影需要分别看，不能把概率行和为零当成值消息为零。被删除的 δ 能量是否包含超越 J0 的 DSS 信息，本轮没有证明。

所有30份原S16做 fit/heldout 逐级诊断。120条原模型风险向量bitwise回放通过；三臂90模型共540向量独立回放通过。P16/S16共60模型是已存在控制，DIRECT仅新增30正式fit，无旧神经网络初始化；实际原D1/D2来源见 MODEL_SOURCE_AUDIT。2次资格短fit共4步，模型和患者预测不持久化。

**本轮效能判定：DIRECT没有稳定超过无修正P16或原S16。** DIRECT−P16的Uno60差为-0.00330821，Bonferroni4区间[-0.00883421, -0.00016107]，完全低于零；该负结果限于当前算子与已反复开发的队列。DIRECT−S16的两项区间跨零。

DIRECT的15个outer checkpoint修正风险SD均值为S16的3.155倍，但中位数分别仅0.00010664和0.00010601，不能称所有模型普遍放大。同checkpoint开启修正的Harrell净差仅0.00007841，95%区间[-0.00077063, 0.00100783]。S16没有改变任何DSS可比较对的credit；DIRECT三个seed修复/损伤分别为0/0、2/2、4/3。增大数值变化没有形成可靠排序收益。独立训练模型的总差异还包含主干优化变化，不能全归给局部修正。

## 完整的未见患者数值链

以下仅15 outer checkpoint，各 checkpoint 内 query→等WSI→等患者，再对15checkpoint取均值；范围及fit分区在 MECHANISM_CHAIN_SUMMARY.csv。公共分量是逐行比例的均值，不是先合并全部能量再求比值。两种统计对象另列，禁止互换。

|arm|direction|raw_row_SD|weighted_centered_RMS|common_fraction_mean_of_ratios|attention_TV|query_average_survival|patient_risk_correction_SD|
|---|---|---|---|---|---|---|---|
|S16|po|0.014072114|0.0085097763|0.91944638|0.00068601265|0.79676688|0.00098004238|
|S16|op|0.02247662|0.017881337|0.84784926|0.0016904957|0.90556702|0.00098004238|
|DIRECT|po|0.019631004|0.010522978|0.91825235|0.00096264076|0.79871535|0.0030915573|
|DIRECT|op|0.027662807|0.020288383|0.84260441|0.0021610303|0.89980322|0.0030915573|

加权中心化方差使用 w=a0*g；常数δ单独代入原算子返回零，减掉加权行均值保持原更新至浮点误差。Taylor项为 rho*w*(δ−μw)。消息三角界效率、value协方差、query/WSI平均、head投影与tanh分别报告；它们单位和分母不同，不能跨指标宣称某层“损失最多”。tanh接近恒等也不能被说成发生输出饱和。

上表risk SD为逐WSI有界修正分支按患者聚合后的量。下表另以实际保存的完整预测计算 matched−ownoff（先把原float32值无损转为float64再相减），包含浮点加法/聚合效应；各值为15个outer checkpoint统计量的平均，绝不把最大值均值叫全体最大值。逐checkpoint/干预明细见RISK_CORRECTION_MAGNITUDES：

|arm|full_prediction_difference_SD|full_prediction_difference_abs_mean|full_prediction_difference_abs_max|difference_SD_over_heldout_anchor_SD|
|---|---|---|---|---|
|DIRECT|0.0030915573|0.0013295857|0.017286481|0.0048490431|
|P16|0|0|0|0|
|S16|0.00098004233|0.00041043066|0.0064666351|0.0015416549|

## 三臂结果与配对区间

DIRECT保留原门控、kernel、encoder、values、head和最终riskcap，只将selected-mass保守更新改为全局归一化的g*(δ−μw) logit更新。注册参数/初始化布局相同；P16不激活修正分支，实际梯度参数更少。DIRECT与保守规则在零附近一阶相同，均对行公共shift不变；不将它包装为已解决公共偏移的新架构。

|arm|Harrell_mean|Harrell_std|Uno60_mean|Uno60_std|
|---|---|---|---|---|
|DIRECT|0.79526386|0.0021084184|0.78485176|0.0047651697|
|J0|0.80357563|0|0.79233946|0|
|P16|0.79941974|0.0017655972|0.78815997|0.0065987013|
|S16|0.79636164|0.0020006823|0.78945966|0.0062147977|

|comparison|metric|delta|CI95_low|CI95_high|Bonferroni_low|Bonferroni_high|positive_seeds|
|---|---|---|---|---|---|---|---|
|DIRECT-P16|Harrell|-0.0041558849|-0.0088729259|-0.00047766788|-0.010494881|0.00039853982|0|
|DIRECT-P16|Uno60|-0.0033082087|-0.0072539687|-0.00070317471|-0.0088342121|-0.00016107029|0|
|DIRECT-S16|Harrell|-0.0010977809|-0.005387015|0.0032451465|-0.0068835651|0.0045986924|1|
|DIRECT-S16|Uno60|-0.0046079065|-0.015037757|0.001703019|-0.019730624|0.0029316571|1|

每seed先汇总5外折同折可比较对分子/分母，再3seed平均；不平均风险、不把15cells称seedmean。20k共享患者bootstrap、4主检验校正；固定模型区间不包含完整重训练或反复研究选择的不确定性。PFI_SUMMARY是冻结DSS风险迁移，不训练、不切换主终点。

## 修复与损伤同时计量

|comparison|metric|C0|C1|repair|damage|net|repair_credit_mass|damage_credit_mass|
|---|---|---|---|---|---|---|---|---|
|DIRECT-J0|Harrell|0.80357563|0.79526386|0.017250843|0.025562613|-0.0083117698|73.333333|108.66667|
|DIRECT-J0|Uno60|0.79233946|0.78485176|0.019237048|0.026724756|-0.0074877081|214.91108|298.56172|
|DIRECT-OWN_OFF|Harrell|0.79518545|0.79526386|0.00047047753|0.00039206461|7.8412922e-05|2|1.6666667|
|DIRECT-OWN_OFF|Uno60|0.78483705|0.78485176|0.00036826464|0.00035356145|1.4703189e-05|4.1141527|3.9498926|
|DIRECT-P16|Harrell|0.79941974|0.79526386|0.0029796911|0.0071355759|-0.0041558849|12.666667|30.333333|
|DIRECT-P16|Uno60|0.78815997|0.78485176|0.0014955306|0.0048037393|-0.0033082087|16.707662|53.666072|
|DIRECT-S16|Harrell|0.79636164|0.79526386|0.0043127107|0.0054104916|-0.0010977809|18.333333|23|
|DIRECT-S16|Uno60|0.78945966|0.78485176|0.0027273105|0.007335217|-0.0046079065|30.468773|81.947054|
|S16-OWN_OFF|Harrell|0.79636164|0.79636164|0|0|0|0|0|
|S16-OWN_OFF|Uno60|0.78945966|0.78945966|0|0|0|0|0|

repair/damage均以全部可比较质量为分母；credit为0/.5/1，tie半分完整计入。mass列为三seed平均的修复/损伤credit质量；Uno60为IPCW质量，不能称独立患者数。所有seed与20k共享患者draw均核验 repair−damage=Cnew−Cbase。9种wrong/tied/correct转移及诊断区间见REPAIR_DAMAGE/PAIR_TRANSITION_INTERVALS。配对不是独立样本，重采样权重为count_i*count_j。

### 每个seed的严格患者对计数

以下Harrell表中 wrong_to_correct_mass 是严格错排→正确的患者对数，correct_to_wrong_mass 是严格正确→错排；tied行另外计量，不与严格错排混称。各seed病例相同，不能把三行相加当作独立患者对。

|comparison|seed|wrong_to_correct_mass|correct_to_wrong_mass|wrong_to_tied_mass|tied_to_correct_mass|correct_to_tied_mass|tied_to_wrong_mass|net_credit_mass|
|---|---|---|---|---|---|---|---|---|
|DIRECT-OWN_OFF|31|0|0|0|0|0|0|0|
|DIRECT-OWN_OFF|47|2|2|0|0|0|0|0|
|DIRECT-OWN_OFF|83|4|3|0|0|0|0|1|
|DIRECT-P16|31|3|22|0|0|0|0|-19|
|DIRECT-P16|47|11|14|0|0|0|0|-3|
|DIRECT-P16|83|24|55|0|0|0|0|-31|
|DIRECT-S16|31|6|25|0|0|0|0|-19|
|DIRECT-S16|47|11|5|0|0|0|0|6|
|DIRECT-S16|83|38|39|0|0|0|0|-1|
|S16-OWN_OFF|31|0|0|0|0|0|0|0|
|S16-OWN_OFF|47|0|0|0|0|0|0|0|
|S16-OWN_OFF|83|0|0|0|0|0|0|0|

OWN_OFF指同一checkpoint的own J0+own pair主干，隔离局部修正；DIRECT−P16/S16是独立训练完整pipeline的比较，还包含主干优化差异；J0是三臂共同冻结主效应锚点。wrongC/wrongO仅干预修正上下文，own values/base/main不改，结果单列而不据外折选择模型。wrongO沿用own与donor的覆盖交集，支持变化须与患者身份效应共同解释。外折错对从未作为新监督或教师生成输入。

## 执行、优势与限制

预算状态：NOT_SUPPORTED_IN_THIS_OPERATOR_CONTROL；DIRECT触顶比例0，选中epoch中位17，实际inner epoch中位57。触顶检查不是全局收敛证明。训练曲线、梯度、逐折、原source和失败账本均保留；无自动预算/selector/encoder/head扩展。

优势：原模型逐元素回放；同参数布局、单算子变化；归一化、消息、风险和患者对闭合；保留所有阴性/不确定结果。

局限：重复开发同一512人/65事件cohort，未做外部确认；DIRECT只比较高阶更新几何；同ck冻结干预是敏感性评价，不能替代重训结果。共同偏移去除本身不能说明有用DSS信息被丢失。注意力和RNA程序均没有独立病理形态真值。

## 可复核文件

PROTOCOL/PROTOCOL_LOCK、QUALIFICATION、SAVED_S16_AUDIT、DIRECT_DIAGNOSTIC_AUDIT、CHECKPOINT_REPLAY、MODEL_SOURCE_AUDIT、PAIR_CREDIT_IDENTITY_AUDIT、BOOTSTRAP_AUDIT、BUDGET_VERIFICATION、MECHANISM_CHAIN_SUMMARY及REPRODUCE。患者级OOF、权重、地图、原始特征、标识和凭据均留在服务器私有目录；公开范围见PUBLICATION_SCOPE_AUDIT。
