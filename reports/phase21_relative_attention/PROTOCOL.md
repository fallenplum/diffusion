# Phase21 — S16 行内修正与风险排序审计

## 问题与边界

检验已有 S16 是否存在“δ 行公共分量占主导 → 概率重分配很小 → 消息与风险衰减 → 无排序修复”的具体链条。相关性不能证明整个因果链。S16 只有一个双向注意力块，报告 Patch→Omics / Omics→Patch 各级，不虚构多层深度。

512 人、65 DSS 事件；5 外折、原 inner split、seeds31/47/83 不变。预处理、J0 权重、TITAN S32、512 Patch、RNA/CNA 24 Token、筛选门、价值投影、风险头和参数初始化规则全部沿用 Phase20。历史文件字节保留。

## 冻结模型诊断

读取全部30份 S16（15 selected inner +15 outer），按其实际 D1/D2 来源、原 receipt 和 checkpoint SHA 验证。全 fit 与全 heldout 样本做不读取结局的 attention 诊断。每方向报告：

1. 原 δ 的有效行 SD、绝对 RMS；w=a0*g，M=sum(w)。
2. μw=sum(wδ)/M、Varw(δ)、中心化 RMS、公共/相对能量占比，保留 M=0、候选<2、退化行计数。
3. 实际 TV、rho*M 上界利用率、Taylor 近似 rho*w*(δ−μw) 的误差。常数 μw-only 应严格为零；去掉 μw 的更新应与原相同，误差以数值精度报告。
4. 每 query 的有向值消息、query 平均后消息、风险头投影和最终有界风险，区分行归一化、跨 query 抵消、预测头投影与输出 cap。

每 query → WSI →患者等权汇总，另保留 pooled 行诊断以揭示查询数差异。所有患者、地图、权重和 raw risk 私有。医学形态真值本轮没有提供，不作组织学解释。

常数 δ 抵消是公式的不变性，不是缺陷本身。条件梯度通常已经忽略公共方向。只有测得行内变异确实很小，并能沿消息和风险传播追踪，才支持它是当前模型的一个限制。不能把行和为零等同于有意义值消息为零。

## 三臂、一个变化

|臂|执行来源|定义|
|---|---|---|
|P16|复用原30模型|无新修正；encoder 与 S16 同布局，修正参数注册但不参与损失|
|S16|复用原30模型|保守修正 a1=a0+rho*w*(expδ/Z−1)，Z=sum(w expδ)/M|
|DIRECT|新增15 inner+15 outer|同 gate/δ，只换为直接条件 logit 更新，见下式|

v=g*(δ−μw)，aD=(1−rho)*a0+rho*normalize(a0*exp(v))，rho=.5。normalize 沿原全部有效 base 候选；零选中质量返回原分布。无 SD 放大、无新标量参数。保留完全相同的最终 cap=.25*SD(J0_fit)。

DIRECT 允许未选边经全局分母发生变化；门的定义、预算和生成网络保持不变。因此这是归一化算子的控制，不是新 selector。两种更新在 δ=0 的一阶响应相同，均不能恢复行公共偏移；正结果也不能单凭此证明学会医学语义。

复用相同协议 controls 是为避免无信息重跑；新 DIRECT 从同 seed 初始化，不从旧模型 warm-start。应分别报告注册/梯度激活参数，不能把 P16 未激活分支说成同等实际拟合自由度。三臂独立 inner EMA 选择 epoch，规则完全相同；不是强行令每臂最终 epoch 一样。新15 inner 全锁后才能开始任何新 outer。

## 训练与有限决策

沿用 max160/min8/patience40、Efron全风险集、AdamW lr.001/wd.01、dropout.1、clip5、microbatch32 与 checkpoint RNG。真实2-epoch资格 fit 比较 outer结局投毒前后 bitwise 相等；不保存 smoke 模型或患者预测。无新凸拟合。若>20% inner 触顶，相关比较标 BUDGET_UNRESOLVED，不自动加预算或换归一化幅度。

主比较 DIRECT−S16、DIRECT−P16，各 Harrell/Uno60，4项校正。20k 相同患者重采样，共享 across model/seed；每 seed 先聚合全部同折分子/分母，再3seed均值。DSS 主终点；PFI 只迁移同一冻结 DSS risk，不做任何 PFI 训练/选择。反复开发 cohort 不称独立确认。

## 配对风险修复与损伤

严格用同一 heldout 外折内可比较对。事件较早者应高 risk；事件/删失同时间按原协议，risk差<=1e−8记半分。每对 credit∈{0,.5,1}；repair=max(credit_new−credit_base,0)，damage=max(credit_base−credit_new,0)。报告3×3转换、修复/损伤的全部可比较质量占比与各自 baseline错误/正确质量条件比例，必须验证 repair−damage=Cnew−Cbase。Uno60只加一次 outer-train 冻结 IPCW。患者 bootstrap 的 pair权重=count_i*count_j，不能把 pair 当独立样本。

两个统计对象分开：同 checkpoint off→on 只体现局部修正；DIRECT 对独立训练的 P16/S16 还含整个主干优化差异。另相同 J0 强锚点作为实用参考。错误患者 Clinical/Omics 干预只动修正支路，保留 own base/values/main。外折错对仅事后评价，不定义新训练监督、teacher或超参。

## 优势与限制

优势：直接追踪可检验的数值链条；新训练只有一个更新算子变化；避免把修复数脱离损伤数解释；原始 checkpoint 来源可审计。

限制：DIRECT 与保守规则小修正下一阶等效，因此不是通用“增强修正”方法；固定模型 bootstrap 不含重训练或研究者反复选择方差；本 cohort 65事件不支持将零结果归因为唯一瓶颈。selector、encoder、head 和幅度正则均不在本轮扩展范围。报告实际负结果或不确定性后停止该有限矩阵。
