# CDEI-G1：端到端局部双证据交互

## 锁定问题

只修正 G0 的两个架构限制：核与证据读出联合优化；额外证据在 Patch–Pathway 局部消息层进入网络。新模型继续使用原始 Clinical、UNI2-h Patch、RNA/CNA 通路 token，不引入新数据或基础模型。不把 tanh、双通道或 CP 核声称为新数学。

匹配分数沿用 Phase19 原始低秩临床条件 CP 公式，hidden32/rank4。A 始终来自原始匹配分数的行 softmax；保留原双向64维 Relative。D 为当前核下本人分数减训练患者参考分数。LOCAL 的额外32维为：

\[
\operatorname{PatchMean}\left[\sum_k A_{ik}\tanh(D_{ik}/\tau)V(O_k)\right],\quad\tau=1.
\]

不在调制后再次 softmax、行归一或消息范数归一。负证据保留符号。tanh 只限制门值，不限制 Value 范数，也不完全消除核缩放、饱和或读出捷径。

## 四组与读出

|组|额外分支|额外维度|
|---|---|---:|
|REL|无，端到端 Relative|0|
|POOL|G0 同公式的 D 的 LME，patch均值/总体SD；开启梯度|2|
|LOCAL|有符号局部证据消息|32|
|CTRL|普通条件消息 Σ A sigmoid(s/τ) V，无参考相减|32|

每组共同读出为 Linear(Relative64,16)，加对应 extra 的无偏 Linear(extra,16)，经过 SiLU、Dropout0.1、无偏 Linear(16,1)、tanh；风险修正乘冻结 fit-anchor SD，患者内等 WSI correction 聚合后仅加一次冻结 J0。没有额外可训练 scalar stacking。

LOCAL/CTRL 的全部参数数与初始化严格相同；REL/POOL 报告实际参数数，不谎称所有组容量匹配。POOL 相对旧 G0 还改变了读出与优化器，跨阶段不能单独作“联合训练”的因果归因。四组内部是同协议比较。

CTRL 是预锁的一种普通 sigmoid 条件门控。LOCAL−CTRL 同时涉及参考相减与有符号 tanh 形式，检验的是整体算子，不单独证明参考相减的因果作用或 tanh 优于其他门；也不能穷尽所有普通门控设计。

## 当前参数下的训练参考

参考固定**当前 fit 的原始组学患者及观测 mask**，不固定编码值或旧分数。每次 forward 先逐患者以当前 encoder 编码，再逐槽位按患者等权平均，用同一个当前核计算 null。完整 autograd 通过 reference encoder 及两条 score 路径，不 detach。不能把原始输入均值先过非线性 encoder 代替逐患者编码均值。

当前 CP score 对编码 Omics 线性，均值 token 等价于同槽位 donor raw score 均值；不等于平均 exp(score)。训练 query leave-self-out；heldout reference 只含 fit。多切片患者不额外增加参考权重。mask/count 明示，空支持拒绝，不删患者。

动态参考能抵消一些共同偏置，但不建立医学“无关联”真值，不自动排除 Omics 主效应或 O×C，也不建立所有尺度的可辨识性。

## 数据与训练

继承已核 Phase19/20/23 缓存与全部患者划分。G0 来源和父提交均固定，历史文件保持字节不变。Clinical one-hot 维度由 fit-fold 决定，不写死。RNA、CNA gain/loss 分开，24 token、每程序16 filters。

3 seeds（31/47/83）×5 folds，四组共60个内层新模型。每组从共同初始化重新训练，固定80 epoch、AdamW lr0.001/weight_decay0.01、clip5，完整 fit 患者 Efron Cox。32患者微批仅改变计算，不改变风险集；checkpoint 重算必须保留 RNG 且绑定本批 patient indices，避免 Python 循环闭包误用最后一批。固定末轮，不读取 inner-val 选 epoch。

J0 主效应全部保留：Clinical、S32、RNA512、CNA gain/loss 程序均值、UNI32与覆盖率。J0 的 fit offset 为 in-sample，并非交叉拟合教师；不声称消除了所有 stacking bias。旧 loader 会临时加载保护结局，新训练接口只提供 fit 标签；真实 non-fit 标签 poisoning 必须证明模型与预测不变。

## 冻结 LOCAL 干预

仅在同一个训练完成、eval/frozen 的 LOCAL 模型上：关闭 extra；Clinical 固定为训练均值；错 Clinical；错 Omics。干预只改变 extra，Relative A/Value 与主效应始终使用本人。

错 Omics 同时更换 extra 中 D 的组学输入及该分支 V；这是**额外分支患者身份干预**，不是整个模型的错配，也不单独隔离 D 与 V 的作用。错 Clinical 只改 D 中两条评分的 Clinical，不改本人 A。

错配沿用当前分区内癌种×全维 RNA 缺测模式、零自身配对，不用 DSS/Stage/Age/Sex。所有 heldout 分支共用本人/供体支持交集及排除并集。资格须核24共同槽位均存在且无支持变化，否则当前原型阻断，不临时改模型或删除患者。基因层覆盖差异仍需披露。

本轮不直接重训 Clinical/Omics 消融。只有原型达到门后才有依据锁定下一轮重训。

## 内层门与统计对象

全部15 cell（60模型）完整回执/SHA/80epoch检查通过后，才独立读取 inner-val 结局。先逐fold计算 Harrell 和修复 Uno60 的分子/分母，每seed累加5fold后取比，再3seed均值。inner-val 可能跨fold重复，不称完整唯一 OOF、不生成确认性CI。

主比较 LOCAL−REL、LOCAL−POOL、LOCAL−CTRL：两指标分别要求3seed均值>0且至少2seed为正。LOCAL−J0 与 LOCAL−J_CS 两指标均值须≥0（点估计筛选，不是统计非劣效）。同头 LOCAL−OFF/NO_C/WRONG_C/WRONG_O 两指标均值须≥0，且每项至少一个指标严格为正。

未通过即停止当前算子，不自动启动 outer、重训消融、增温度/层数/训练预算。若通过，才显式启动同预算 fresh outer，独立患者 OOF、共享患者 bootstrap；PFI 仅作冻结 DSS 风险次要迁移。不得用外折错排患者对定义新监督。

## 机制与解释

核查完整训练轨迹与分支梯度；D、tanh 正负比例及饱和；有符号行质量与局部消息范数；固定头干预造成的风险变化；配对排序修复和损伤。非零变化不等于生存收益，训练损失下降不等于泛化收敛。

优势：同输入、同核公式和明确四组拆分，使“联合训练”和“局部读出”受到更直接检验。风险：LOCAL仍可能读取主效应、Value放大或tanh饱和，POOL容量不同，65事件和反复开发队列限制结论。实现恒等式可高置信核验；效果与整体创新性以真实结果为准。

拓展思考／你可能想知道：若 LOCAL 只超过 REL，但不超过同容量 CTRL，不能将收益归因于参考校准的局部证据。若只有正确配对改变特征、没有配对风险收益，也不能认定临床条件三模态机制成立。
