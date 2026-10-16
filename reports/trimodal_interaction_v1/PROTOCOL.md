# 三模态条件差分交互 V1：小规模可行性协议

## 范围与研究问题

本轮精读《diffusion：三模态交互的文献与候选机制设计》（2026-10-16）后，检验临床背景与 RNA 共同改变病理关联检索的实现是否可训练，并与普通条件化、完整两两分支、通用差分、FiLM 和幅度控制比较。

**本轮仅为既定 outer folds 1、3、5 内部的 inner-train/inner-val 可行性实验。没有 outer-test 风险，没有完整患者 OOF，没有显著性或 SOTA 声明。** 当前 512 患者、65 DSS 事件的同一队列已经多轮开发，不能当作新的独立确认。

历史 Phase17R 已停止条件 S6 结构支线；这不证明 RNA 条件信息不存在，也不证明所有原始 RNA 主效应旁路必须删除。本轮采用共同正则的原始 RNA 线性主效应，独立检验报告首选候选 A；不同时加入 copula、Mamba、重建或排序蒸馏。

## 数据与参考的隔离

- 使用原始患者折及原 inner 划分，DSS 原始主终点；不使用 PFI 或其他终点进行训练与选择。
- 每个 inner-train 独立拟合 Clinical 编码、缺失处理、标准化、冻结的 Hallmark 富集 512 基因流程、LOCAL-CONCH PCA32、patch PCA64、完整 C/R/RM reference。原生 Clinical 六个字段保留年龄缺失指示。
- 冻结 CONCH v1.5 为唯一交互视觉主干，使用已审计的 geometry-only 候选 ROI；本轮固定每患者 128 patch，不能直接外推到完整切片。多切片患者不会因切片数获得多倍 patch 预算。
- patch PCA64 仅使用 inner-train 每患者确定性抽取的至多 16 个 patch；所有 arm 接收相同输入。静态 S32 和 LOCAL-CONCH32 均对所有 arm 可用。
- C0/R0/RM0 是精确训练分区的完整特征均值，包括 RNA 测量缺失指示。它们是计算参考，不声称真实临床反事实或无疾病状态。
- wrong RNA 是当前 fit 或 validation 分区内、基于癌种/分期分层的确定性错配；不跨分区，不读取 DSS 标签。患者自身的 RNA 主效应及 Clinical/Pathology 输入保持。
- 私有缓存按完整 source、dataset、manifest、预处理和 visual hash 核验；inner 代码拒绝 outer-test 预测索引。真实数据审计另有记录，合成拒绝测试不能冒充该审计。

## 模型与有限差分

共同主效应为 `Clinical + S32 + LOCAL-CONCH-PCA32 + RNA512` 的线性 Cox，固定 L2=1。每个神经模型先进行该凸问题的真实收敛初始化，然后共同优化自身线性主效应及交互；没有教师、stacking 或复用被 inner-val 标签训练的上游。

RNA query 保留值、gene identity 与 measured mask 的独立输入；交互 hidden=64、heads=4，softmax 沿有效 patch 维。Clinical 使用低秩条件化。候选四路径共享权重、mask、样本、归一化和计算规范：

```text
A11 = A(C,R;P)       A10 = A(C,R0;P)
A01 = A(C0,R;P)      A00 = A(C0,R0;P)
DeltaA = (A11-A10) - (A01-A00)
M = DeltaA @ V(P)
```

所有 dropout 固定为零，避免随机差分和 two-pass 梯度不一致。交互输出为无偏置读出后 `0.25*tanh(raw)`；A1 为维持严格两两函数类，分别约束三个分支后仅线性相加，即 `(0.25/3)*sum(tanh(branch))`，禁止先全分支求和再作联合非线性。所有 arm 的每患者风险修正绝对值均不超过 0.25；该风险上界不保证 C-index 不降，也不保证 learned query 语义保真。

完整 C/R/RM reference 下 mixed 消息精确为零；DeltaA 行和及固定 attention 下的 V 常量平移不变按 dtype 容差审计。该差分不等于因果干预、PID 纯三阶协同或分布上已识别的功能 ANOVA。对原始 P 任意平移的整体模型不变性不成立。

## 锁定模型矩阵

| Arm | 实现与解释 |
|---|---|
| A0 | 共同全输入凸 Ridge Cox；每折仅真实拟合一次 |
| A1 | CP、CR、RP 三个两两分支，分支各仅读指定模态，最终线性相加 |
| A2 | Clinical-conditioned RNA-to-patch attention，不作差分 |
| A3 | 官方 Differential Transformer 核心 signed-map/head-RMSNorm 的 cross-attention 适配；不是原语言模型的严格复现 |
| A4 | 四路径 mixed contrast，gate off |
| A5 | 四路径 mixed contrast，post-attention sigmoid gate |
| A6 | 与 A5 相同训练流程，但只在交互分支使用分区内错 RNA；主效应保留自己 RNA |
| A7 | 用 mixed 消息的逐样本范数匹配中心化普通 attention 的方向控制；scale stop-gradient，随后共同 gate |
| F0 | pre-attention FiLM 控制 |
| PR | 不含 Clinical 条件的 RNA-to-patch attention，Clinical 保留在线性主效应中 |

A7 含有 mixed 幅度信息，不能写成完全不含四路径信息的普通 shrinkage；其 stop-gradient 也不同于对完整 forward 表达式求所有导数。这个控制用于初步区分消息方向与幅度，不能凭本轮完成计算量严格匹配的机制确认。

共同 hidden/head/gene/patch/epoch 预算并不意味着相同 FLOPs。登记总参数、实际梯度参数、attention-score 数量、峰值显存及并行环境时耗，不把时耗当隔离速度排名。128 patch/PCA64 是本轮工程范围，不是对完整报告所有输入布局和基线轨道的最终复现。

## 优化、选择和实际拟合计数

- seeds 31、47、83；folds 1、3、5。A0 每 fold 仅 seed31 真实拟合，其余 seed 使用同一确定性凸风险作为评价引用。
- 9 个神经 arm × 3 seeds × 3 folds = **81 次真实神经训练**；A0 为 **3 次真实凸 baseline 拟合**，共 84 个真实模型 cell。
- 每个神经 cell 还实际调用一次 Ridge 初始化，共 **81 次初始化 solver call**；加 A0 得 **84 次 Ridge solver call**。初始化不冒充额外独立神经模型或新 seed 证据。
- A0 共 9 个 seed/fold 评价引用，其中 6 个是复用，不是真实重训。全矩阵有 90 行评价 cell，不能称为 90 次模型训练。
- AdamW lr=0.001，交互 weight decay=0.01，main 手动 L2=1，梯度裁剪5；microbatch16。最少8轮，最多40轮；inner validation 完整 Efron loss 的 EMA(alpha=0.2) 选择 checkpoint，12轮无改善停止，改善容差1e-8。
- **full-risk Efron**：第一遍固定当前参数获得整个训练分区 eta 及完整风险集 dL/deta，第二遍同参数逐 microbatch 重算并累积 VJP，L2仅加一次，最后一次 optimizer step。不是 microbatch 局部 Cox。无 dropout；合成 ties 梯度检查与整图一致。
- **每个神经 arm 分别**核查训练预算上限命中比例，任一相关 arm 超过0.2则预算门失败，不能由其他早停 arm 稀释 A5 的预算不足；总体比例仅描述。这表示本预算下不充分，不能写成机制失败或已充分收敛。命中比例低也仅是停止规则证据，不证明全局优化或生物机制充分。

### 明确记录的预算160敏感性修订

在观察一个 inner cell 触及40轮后，新增独立的 `trimodal_interaction_budget160.json` 与 `extend_trimodal_budget.py`，原配置、源码 identity、checkpoint 与40轮评测保持只读。该修订由 inner 训练预算证据触发，不是预注册的独立确认，也不使用 outer 结果。

先完成原84个真实 cell 并保存完整40轮审计。只有任一神经 arm 的40轮触顶比例>0.2时，启动160轮 sensitivity。原 `actual_epochs==40` 的 cell 使用相同 seed/初始化从头重训至最多160轮；原 `actual_epochs<40` 的 cell 已按相同 patience 停止，在160 schedule下结果相同，作为引用复用，不能重复计真实拟合。A0始终引用原3个凸拟合。

重训前40轮逐字段核验 epoch、train loss、penalty、validation loss/EMA/Harrell、gradient/query/gate norm，最大误差≤1e-7；时间字段因重跑资源状态不同不要求相同。原训练 source 保持，额外 budget source 覆盖原 source、新配置与扩展脚本。所有新 checkpoint 和 receipt 位于私有 `budget160/`，新评测只写公开 `reports/trimodal_interaction_v1/budget160/`；其余 cell 的引用计数与真实新增 Ridge 初始化/神经拟合分别列出。

40轮及160轮分别审计和报告。如果任一相关神经 arm 在160轮仍有>0.2触顶，最终判定为 **BUDGET_UNRESOLVED**；不得据此宣布机制不成立。只有预算充分后才解释预测门的通过/失败。

## 指标、配对诊断和工程门

每个 inner-val 分别计算 Harrell 与 Uno60。独立 reverse KM 仅由该 cell 的 inner-train 拟合，截断60月；独立可比较对实现和 scikit-survival 逐 cell 对照。每 seed 汇总3个 inner-val 的折内对分子/分母，再平均3个 seed 算法结果，禁止跨折比较任意尺度风险或先平均风险。

由于不同 outer fold 的 inner-val 可能包含重复患者，以上汇总称 **overlapping inner-validation diagnostic**，不称 OOF。患者数、患者对数和 seed 数不作为独立重复样本；不计算 bootstrap CI 或 p-value。

主比较预锁 `A5-A1`、`A5-A2`、`A5-A3`、`A5-A7`、`A4-A2`；Harrell/Uno60 均报告。次比较为 `A5-A4`、`A5-A6`、`A2-F0`、`A5-A0`、`A5-PR`。

工程 follow-up 门严格依据已锁配置：每个相关神经 arm 的预算命中比例≤0.2、A5 query/gate 梯度路径可训练；每项主比较的每个 seed 的折内对加权 Harrell 差均>0，且至少2/3 cell 方向为正。Uno60 同时如实报告，不事后改成更有利的门。预算不足判 `BUDGET_UNRESOLVED`；预算充分但预测门未过则停止本实现的扩展，不宣称 idea 无效；通过也只允许另行预注册后续矩阵，不能宣称显著机制成立。

**跨全部 fold 汇总 inner 结果做全局门，会暴露某患者在其他训练分区中的标签。** 因此这不是对未来 outer 结果的独立确认。后续若要严格评估，必须预锁完整算法并在每个 outer-train 内单独选择/决策，或使用真正未参与选择的数据；不得将本轮探索赢家改写为此前预注册成功。

## 审计与交付

`validate_trimodal_interaction.py` 在全部 DONE 后核验训练 source、每个 checkpoint SHA、原 split、风险仅有 inner-val 长度、无 outer/OOF、训练曲线有限及 EMA/停止规则，并重放全部选中 checkpoint、wrong 推理、C/R/RM null 和风险上界。另用独立 Efron 和 C-index 检查保存的成绩。

默认命令审计原40轮；`--budget160` 使用独立 budget source 和 receipt、核验原40文件 SHA及重训前缀，并将全部输出写入 `budget160/` 子目录，不覆盖原阶段结果。

公开仅源码、配置、匿名逐 cell/seed 聚合指标、训练曲线、算子/来源/失败审计与明确决策。患者数据、完整风险、gene selection、permutation、预处理状态和训练权重留在私有目录。没有原始 WSI，不声称已验证细胞或组织学区域机制。

优势是所有比较保留相同主效应和合法输入，能分离四路径方向、gate 与错患者机制；风险是事件少、反复使用同一 cohort、reference 依赖、query 可能压弱 RNA 数值信息，以及128 patch/PCA压缩的表达限制。算子 PASS 不替代真实疾病机制或外部泛化证据。
