# Phase19 — Clinical-conditioned Low-rank Patch–Pathway Attention

## 范围与执行边界

这是新 Attention 架构的探索性验证，不延长 SSD-Fusion，不继承任何 S6 神经网络权重。父提交为 `099f3d8cd0815a1f994f22b5ce5801b1f4621105`；历史实验和报告保持不变。完整正式运行必须在算子、数据、源码及标签隔离资格通过后开始。本文件描述预先固定的计划，实际完成数以完成凭证和逐单元表为准。

本队列已被多轮研究，不能把新一次交叉验证称为未触碰队列上的确认。旧四图差分 Attention 已有125次内层拟合而没有外层OOF；不计入本轮新训练或外折证据。

## 唯一架构假设

Clinical是否能调制同一患者的Patch–通路兼容度，并提供超出双模态／两两交互和主效应的DSS排序增量？

\[
\ell_{ik}=q(p_i)^T k(o_k)/\sqrt{32}
+\sum_{r=1}^{4}a_r(p_i)b_r(o_k)v_r([1,c]).
\]

Clinical的结构化变量按当前训练分区拟合编码、插补和标准化。兼容度中的常数1提供全局低秩双线性项，`v`截距初始1，其他Clinical系数初始0；它避免标准化均值约0使固定条件控制退化。A2仅把Clinical非截距输入固定为训练均值，仍有可学习的非零全局rank4项。各机制臂在同seed/fold下共同模块初值逐元素一致。

三元CP项严格等价于Clinical门控的低秩双线性兼容度，亦等价于增广Q/K。输出及所有输入／参数梯度将作等价单测；它不作为两个独立算法参赛，也不声称首次提出三元乘积或FiLM。已有Clinical前置方法不能概括为都只做末端拼接。具体文献边界见文献审计。

同一分数矩阵沿通路轴和Patch轴分别softmax，得到双向消息；在各自有效Token上池化，经共同小读出头得到交互风险。不存在SSM扫描、终态记忆、双向扫描或可搜索gamma。

## 数据和信息路径

- TCGA-COAD/READ，原512患者、65个DSS事件，原5患者外折，seeds31/47/83。
- 冻结UNI2-h，保留既有物理CONCH足迹聚合与geometry选择；每患者总数不超过512个Patch。每WSI出标量风险，再患者内WSI等权平均。
- 冻结训练分区内的Hallmark-enriched512基因选择。新的24个分子Token分别对应8 RNA、8 CNA gain、8 CNA loss；各自membership、测量mask和coverage，不将RNA/CNA混为同一模态证据。所有神经Token编码器在本轮重新训练。
- RNA/CNA值、基因身份及缺失的固定预处理继续训练折内隔离；CNA gain/loss先拆开，再拟合各自标准化。
- 共同线性主效应输入为Clinical、S32、RNA512、CNA gain8/loss8、UNI患者均值PCA32、RNA/CNA coverage各8。PCA及全部缩放只拟合当前训练分区。主效应从同一ridge10凸最优解初始化，与交互联合训练；不把训练内风险再交给自由stacking系数。
- `J_CS`为同源Clinical+S32 Ridge10；`J0`为上述联合线性控制。J0不预称最强基线；历史Clinical+S32+CONCH32仅作为不同视觉／训练协议参照，不能当作同输入因果消融。
- 原图形态核验仍未完成，所有选区解释保持geometry-only。

## 固定机制矩阵

|ID|操作|可识别的比较边界|
|---|---|---|
|A0|普通双向P–O attention，删除整个CP项|没有Clinical直接调制的注意力|
|A1|PO、PC、OC低秩两两score相加再softmax|`PAIRWISE_LOGIT_SUM`；softmax已可能引入三模态非线性，不能称严格全两两函数|
|A2|CP项保留非零全局低秩门，固定Clinical为训练均值|患者特异Clinical条件的消融|
|A3|完整患者Clinical条件CP项|本轮候选|
|A4|A3以错误患者Clinical训练和评估，仅换交互条件|本人Clinical主效应保持不变|
|A5|A3以错误患者Omics训练和评估，仅换交互条件|本人RNA/CNA主效应保持不变|
|A6|独立PO、PC、OC三分支，各自出scalar后线性相加|严格两两风险函数；无联合非线性头或患者特异的混合权重|

Clinical错配仅在各fit/validation/test分区内按COAD/READ循环置换，不能按Stage、TNM、Age或Sex分层。Omics错配保留既有cancer/Stage匹配规则；记录单例回退、Clinical实际改变比例、RNA平台和缺失模式差异。错误患者比较不自动构成生物学语义或因果证据。

MGCT、CAMR、MMP另外作为官方核心的适配方法轨道实际训练；固定源码、原生输入／目标、必要的特征／Cox／Clinical适配及CAMR显式修复均登记。MMP原型只能在相应训练分区拟合。缺少资格不能用自写近似挂论文名，也不能将仅登记来源写成已经训练。

## 训练、选择和预算

每个神经arm均为3seed×5fold，15次inner和15次fresh outer。10个神经arm计划新增300次正式拟合；两个确定性凸控制各5fold×2role，共20个唯一求解，跨seed风险引用而不重复制造优化次数。smoke和资格诊断的实际优化次数另外计数，不持久化其模型或患者风险。

固定d32、rank4、dropout0.1、AdamW lr0.001、weight decay0.01、主效应L2=10、clip5、microbatch32患者、max160、minimum8、patience40、validation loss EMA0.2。无额外rank／lambda搜索。

inner-train拟合全部上游特征和模型，inner-val只选完整pipeline的epoch。每个arm配置固定后，fresh outer从同源初始化在整个outer-train训练准确所选轮数；outer-test不参与任何选择。不能跨独立ABMIL/Transformer坐标拼接latent作meta-training。

微批只管理显存；保留全部患者风险的计算图，通过保留dropout RNG的nonreentrant checkpoint，计算一次完整Efron风险集损失并反传。不得采用不同dropout mask的无控制two-pass，不得微批Cox。CAMR辅助损失与独立判别器步骤、原型拟合等另行统计，不冒充主Cox优化预算完全相同。

逐arm触顶比例超过20%时判`BUDGET_UNRESOLVED`，不能据此否定机制。所选epoch、实际epochs、训练／内层验证损失和C-index及真实梯度均保存；未触顶也不证明全局收敛。正式训练前做外折标签突变检查，完整训练历史／参数／所选epoch不受外折标签影响。

## 评估与预锁定推进门

主比较：A3分别减A0/A1/A2/A4/A5/A6，Harrell和Uno60共12项家族。每seed先汇总5折同折可比较对分子／分母，再平均3seed指标。20,000次固定外折内患者bootstrap，所有模型／seed共享同一患者计数，重算完整指标；报告普通95%与Bonferroni12区间。区间条件于已拟合模型，不包括完整重训练或研究者选择方差，不显著不等于等效。

方法级适配baseline、J0、J_CS及历史CONCH32旁列，次要比较与主家族分开。PFI仅迁移冻结DSS风险，不训练或选配置。

预算和技术门通过后，仅当六个主比较在两指标上均值正、每项至少2/3seed同向，且不损害J0/J_CS两指标均值，才记`DIRECTIONAL_CANDIDATE`。校正区间跨0不得写confirmed；即使有队列内支持，也不是外部确认。方向门失败记`MECHANISM_NOT_SUPPORTED_IN_THIS_ARCHITECTURE`，不能推广为Clinical条件化不可能。

## 诊断、优势与风险

预固定前32名训练患者，不按结局／错误排序挑样本。记录双向attention熵／集中度、真实与错Clinical的map变化、三元／基础logit尺度、交互／主效应风险标准差、独立交互对Clinical/RNA的梯度，以及训练损失中各参数组的真实梯度。全部冻结干预与真实错配重训分别标明。

优势：保留RNA直接排序和强Clinical语义路径，分开RNA/CNA证据；CP、两两logit、严格两两风险及错配控制分别定位机制。

风险：65个事件、重复开发队列、平台／缺失混杂、条件增量估计不稳定；原型/PCA和方法输入适配限制与论文原生比较；注册／实际有梯度参数与算量并不自动相同。逐项披露实际参数和资源，不凭梯度更大或attention更尖锐认定预测有效。

数值与实现资格可达到高置信；唯一生物学机制解释在本队列仍为低置信。完成矩阵后仅针对明确的故障定位，不自动扩大网络、继续SSD修补或为追分切换endpoint。

## 交付及隐私

完整源码／配置／测试、固定源码登记、协议锁、数据审计、逐折／逐seed指标、配对区间、训练与attention诊断、资源和真实拟合次数、RESULTS/decision/失败记录，推送独立分支。患者级OOF、检查点、真实患者/基因标识、PCA／原型状态、原始矩阵和凭据留私有目录；保留正确检查点和历史结果。
