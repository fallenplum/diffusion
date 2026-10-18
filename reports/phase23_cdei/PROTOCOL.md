# Phase23 CDEI-G0：固定核双证据可行性审计

## 问题与范围

检验：在强 Clinical/Pathology/Omics 主效应及同一临床条件匹配核已经存在时，训练参考校准的原始匹配分数是否提供可泛化、依赖正确患者配对的额外 DSS 信息。

本轮是**固定核信息可行性审计**，不是端到端 CDEI。阴性结果只能约束当前核与读出，不能否定所有可训练 CDEI。原队列已被反复开发使用，不称独立外部确认。数学组件存在强前作，创新状态为 `UNKNOWN_NOT_ESTABLISHED`，见[前作核验](LITERATURE_AUDIT.md)。

## 数学边界

单向行 softmax 对公共加性偏移不变，对分数缩放并不不变。同一矩阵用于双向 attention 时，行偏移可能改变反方向，因此不能声称所有公共信息都被整个网络丢弃。

令 x=s−b，逐行严格满足：LME(x)=mean(x)+LME(x−mean(x))。因此 LME 包含公共水平与离散度，并非纯公共分量；必须保留 MEAN/CENTERED 控制。均值可加分解，但两个 pooled SD 不能相加重建总 SD。Phase21 的公共分数能量并不等于已经证明的条件预后信息。

## 源码与训练锁

继承 Phase19/20 已核验的全部十个 role/fold 预处理缓存及 J0/J_CS，逐元素重建冻结风险；不修改历史文件。原始 RNA 有两种缺测模式，只称平台代理，不冒称经验证的平台标签。

seeds=31/47/83，folds=1..5。先训练15个 inner kernel，全部完成后才独立读取 inner-val 结局。每个 kernel 重新初始化，只用当前 fit；固定80 epoch、AdamW lr=0.001、weight decay=0.01、clip=5，完整 fit 患者 Efron 风险集，32患者微批保留梯度。每患者多 WSI correction 等权聚合后加一次冻结 J0。固定末轮，不用 heldout 选 epoch。

沿用 Phase20 rich tokens 类：RNA8、CNA gain8、CNA loss8，程序内16个 filter 输出；使用原 Phase19 CP score 公式，hidden32/rank4。没有加载旧 inner/outer 神经权重。旧 inner checkpoint 已受同一 val 的 epoch 选择影响，不能冒充纯 fit-only kernel。

kernel、encoder、value 全部冻结后建 reference，再拟合条件 Cox。冻结旧 b 又继续更新 s 会引入偏置/尺度漏洞，本轮不采用该实现。旧 loader 会临时加载保护结局，新训练接口只暴露 fit 标签，不声称磁盘字节从未加载。真实 heldout 标签 poisoning 必须验证 kernel、reference、特征、权重和预测不变。

## 参考和错配

reference 只用当前 fit，患者等权而非按切片数。CP 核对编码后 Omics 线性，平均 donor raw score 可通过平均 donor token 表示，不是平均 exp(score)。保留槽位 mask/count。

正确/错误 Omics 共用排除并集 {i,j}；heldout 仅参考 fit。所有 absolute 对照使用本人/错配 Omics 共同观测槽位交集，防止可用槽位数量成为假机制。Relative 与主效应始终本人。

错配在当前 fit/valid 分区内独立构建，癌种×全维 RNA 缺测模式分层、零自身配对。若单例只可放宽癌种而不能放宽缺测模式；模式单例直接阻断，不删患者。当前所有分区均无单例，无需放宽。

null 使用同一 P/C 及训练 Omics 均值。它不是医学“无关联”真值，可消除 P/C-only 公共偏置，但仍可能保留 O 主效应和 O×C，不能证明纯三元信息。

## 锁定读出

Relative 用原始 s 的双向归一化信息，64维。Absolute 仅对 P→O 方向按用户公式计算，tau=1；WSI 内 patch 均值与总体 SD 得2维，再患者内等 WSI 聚合。无证据患者不得静默删除。

读出均为冻结 J0 加条件 Efron Cox，offset 系数恒为1；fit-only 标准化，固定 λ=10，完整患者风险集、无截距、无 scalar stacking。凸优化须梯度最大值≤1e−7。

|读出|额外输入/干预|系数数|
|---|---|---:|
|REL|本人 Relative|64|
|DUAL|Relative + 校准 E 均值/SD|66|
|ADD|Relative + 分离 C/Pmean/Otokens/mask 的 fit-only PCA2|66|
|NO_C|仅 Absolute 的 Clinical 换 fit 均值|66|
|WRONG_O|仅 Absolute 的 Omics 换患者|66|
|WRONG_C|仅 Absolute 的 Clinical 换患者|66|
|MEAN|Relative + residual 行均值的均值/SD|66|
|CENTERED|Relative + centered-LME 的均值/SD|66|

ADD 是指定两个加性方向，不能穷尽非线性主效应。相同系数数不保证相同信息量或有效复杂度；报告实际秩、方差、参数数及 PCA 保留方差。另用**同一个 DUAL 头**关闭 Absolute 或改变配对/Clinical，区分重新训练与干预。

## 先内层、后条件外层

预锁5个主要比较：DUAL−REL/ADD/NO_C/WRONG_O/WRONG_C。每项在 Harrell 和修复后 Uno60 均须3seed均值>0且至少2seed为正；DUAL−J0 两指标均值须非负。全15个 inner 完成前不读 inner-val 结局。

inner 验证分区可能跨fold重叠，只按同fold分子/分母汇总作诊断，不声称完整唯一 OOF，不输出确认性区间。若门未过，停止固定核分支，不启动 outer。若通过，fresh outer15个 kernel、同预算，最终只评一次。

outer 每seed先汇总同fold Harrell/Uno 分子与分母，再平均3seed；不平均风险产生 ensemble，不把15格称15seed。主要5×2比较用20,000次共享患者 bootstrap、Bonferroni10；repair−damage 必须等于净 C-index 变化。PFI只迁移冻结 DSS 风险，不训练不选择。不用外折错误患者对构造监督。

## 优势、风险和交付

优势：同核、固定参考、分解及同容量控制，使额外分数路径可单独检验。风险：固定核可能没有可解码信息，null 可抵消有用背景，有限 ADD 不穷尽非线性主效应，65事件限制估计精度。

源码、匿名汇总、来源哈希与报告推独立分支；患者级 OOF、reference、mask、indices、weights、凭据全部私有。数学恒等式置信度高；真实条件收益和创新性在结果前不作成立判断。
## 资格核验补记

十个 role/fold 的 Clinical 独热编码维度为 inner：41/38/40/41/39，outer：41/39/41/41/40。修复把第一折维度写死的接口校验后，改为严格沿用各训练折已核维度；未改变输入、通路、损失或模型选择。修复前全部尝试与93次已执行 kernel 更新保留在私有归档，未读取 inner-val 指标，不混入正式矩阵。

全部分区的24个 Omics 槽位在共同支持下可用；当前没有因本人/供体槽位交集减少可用 token。但基因层面的覆盖差异仍存在，不能据此声称RNA平台无影响。

冻结 J0 的 fit 风险来自原 fit_X×w，属于 in-sample offset，不是交叉拟合教师。heldout 预测仍严格未见，但本轮不声称已消除所有训练残差偏差或 stacking bias。标签污染资格证明的是 heldout 结局不影响本轮训练及校准，不能修复原数据不可观测的混杂。