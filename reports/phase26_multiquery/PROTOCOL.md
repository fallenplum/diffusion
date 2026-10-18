# Phase26：Strong-Anchor Multi-Query Interaction 内层原型

## 研究问题与来源边界

本轮从C4的多RNA Query、RNA residual与Patch读取出发，检验分子查询间竞争是否比普通Attention提供额外排序信息。不是修补Phase25的单查询R3。历史C4检索后即mean八slot，本轮保留八slot直至读出，属于新架构，不称C4严格复现。

查询间Softmax与加权均值来自 [Slot Attention原论文](https://arxiv.org/abs/2006.15055) 及 [官方实现](https://github.com/google-research/google-research/tree/master/slot_attention)。本轮无随机无标签物体槽、GRU或官方训练任务，不能称完整官方Slot Attention，也不主张双归一化是新数学。有限表示指八个分子对应的局部slot，不是已经取得组织学真值的ROI。

## 共同强主效应

CP采用Phase25独立拟合的Clinical+S32+GLOBAL_MATCHED CONCH32 λ10 Ridge Cox原系数、原风险，逐元素冻结；不取Phase25 joint R0中的CP子系数。RNA为同一fold-local 512基因的独立RNA-only λ10 Efron Ridge Cox。二者log-risk以固定系数1相加，无meta权重拟合、共同重优化或验证后缩放。

主效应整个pipeline三折交叉拟合，供局部分支训练。复用原三个teacher严格train-only预处理状态，但重新拟合各teacher的独立CP/RNA头；完整fit CP复用，完整RNA新拟合，每分区7次新凸拟合、1次CP复用，共21次新凸拟合。RNA/CP可能重叠计数，故保留原CP-only护栏及RNA-only参考。

CF风险保持真实值，不用含本人标签影响的fullfit统计反向校准。fullfit主风险用于valid；既知的teacher/fullfit分布差异继续逐分支报告，不能宣称已解决。当前RNA-only算子原型不引入CNA新分支。

## 四组与共同输入

|组|结构|目标|
|---|---|---|
|M0|冻结CP＋独立RNA风险|共同主效应|
|QUERY|M0＋RNA Query residual/FF，Patch消息恒0|额外非线性RNA读出控制|
|ATTN|M0＋普通多Query Attention与residual|检索对照|
|COMP|M0＋查询竞争、同residual更新|候选|

三个原inner分区×两个seed，共24槽、18个神经拟合。512基因按固定位置随机平衡分为8×64（seed117301），每基因恰用一次。各slot有64×32基因特异投影，不先算八个标量程序均值。分组不是历史g0字节复刻，也不作为真实通路语义优势。本轮不做分组、宽度或Query数量搜索。

输入为共同冻结CONCH候选（每患者≤512 Patch）、RNA512及Clinical；S32与全局CONCH进入共同主效应。所有臂8×32 slot、同投影、同两轮共享残差FF、同slot-flatten读出、同初始化。QUERY分配相同模块，但Patch/Clinical-score路径不活跃，必须披露有效参数和梯度，不能称完全同有效容量。

## 算子及风险约束

每轮 `score_kn = [q(LN(h_k))+U((V LN(h_k))⊙(T C))]ᵀk(P_n)/sqrt32`。Clinical项随slot变化；仅U零初始化，避免整条乘性路径同时零初始化。共享Clinical偏移会被slot Softmax消除，因此不采用该公式。

ATTN在每slot内对Patch做Softmax。COMP先每Patch对slot做Softmax，再各slot按有效Patch质量归一为加权均值；两者最终每行和均1。占用质量仅作诊断，不作为额外幅度旁路，也不声称双随机或保证每个slot学到不同病理形态。

每轮 `h=h+A V(P); h=h+FF(h)`，QUERY令消息为0；RNA本人residual始终保留。全部8×32展平后经256→16→1读出，没有提前合为患者级消息。末层零初始化，初始风险严格等M0；首步上游零梯度是预期，之后须有有效梯度。

修正为 `0.1*tanh(logit)`，在Cox log-risk单位上硬限幅，无可学习全局增益。0.1沿用C4既有读出系数用于新训练的界限，不是从Phase25缩放表挑得的最优alpha；本轮仍是已反复研究队列上的探索。保留Query residual意味着该修正并非纯交互，必须比较ATTN−QUERY与COMP−ATTN。

## 训练、干预与门槛

固定80轮末轮，AdamW lr0.001/wd0.001、clip5、完整fit患者Efron风险集，microbatch32仅计算拆分；附 `0.5·0.01·mean(tanh(logit)^2)`。不让valid标签进入训练、不选epoch、不改预算或bound；训练loss、末段趋势和饱和比例如实报告，不能预设收敛。

冻结OFF关闭整个修正；VALUE_OFF关闭Patch消息而保留本人RNA residual。其余只替换score-query的RNA或Clinical输入，本人RNA残差、主风险、Patch K/V与head固定。错配按原分区内癌种×raw RNA缺测模式零自配对，无DSS/Stage等匹配；它们不是独立重训消融。

全部槽完成且独立检查点回放通过后才统一评测。沿用原within-fold Harrell、修复Uno60，每seed加总三个分区分子/分母后mean两seed；不称独立OOF、不bootstrap、不把两指标合并排名。

ATTN−QUERY和COMP−ATTN在两指标上均值须正、各2/2seed正；COMP不低于M0和独立CP两指标点估计；COMP本人相对VALUE_OFF、错RNA、错Clinical和Clinical均值控制，每项双指标均值非负且至少一项正。任一失败停止当前竞争候选，不自动改选ATTN进入outer。GO也只准下一步锁定独立完整协议，不自动启动外折。

## 机制、风险与交付

记录slot消息范数、slot间相似度/有效秩、占用与空slot、attention分布和错配敏感性、分组梯度、风险修正相对主风险幅度、饱和率、CF/fullfit分布、配对修复与损伤，以及参数和计算/运行量。

优势是完整保留共同线性RNA信息，分开非线性Query旁路、普通检索与竞争更新；风险是65事件、反复使用同一队列、CF与fullfit风险差异、QUERY有效参数较少、slot仍有压缩，以及Clinical本轮仅有冻结干预而无独立无Clinical重训。任何正向点估计不自动证明语义、因果或新颖性。

历史文件不修改，正式权重/风险/坐标/索引私有保存；只推新源码、锁和匿名报告。短资格权重和数组不落盘。数据与GPU可用性已核实，本轮不下载新数据。
