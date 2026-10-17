# Phase21 机制解释与独立复核计划

## 状态与适用范围

本文在读取本轮旧S16新增诊断结果之前固定解释规则，仅补充报告，不修改已锁训练源码、模型、参数、检验家族或运行预算。它不是根据结果新增的成功门槛。

本轮对应两个有限问题：

1. 在已有S16中，条件分支产生的量如何经过行归一化、value汇总、query/WSI平均及风险头，到达患者风险？
2. 保持encoder、selector、Clinical核、value和风险头不变，仅替换概率更新算子，是否产生比原保守算子及无修正模型更好的held-out排序？

这与用户要求追踪机制、控制模型容量和避免盲目增加模块一致。它**不检验保留公共δ分量是否有益**：本轮两个更新均消除该分量。若将本轮解释成检验“去掉归一化是否能保留公共Clinical信号”，则与实际算子不符。

P16、S16分别复用原30份模型；只有DIRECT新增15次inner及15次fresh outer拟合。复用记录保留原checkpoint字节、实际D1/D2来源及独立选出的epoch。三者注册参数布局与初始化规则相同，但P16修正支路不参与损失，不能称有效训练自由度完全相同。已观察过Phase20结果，Phase21仍是同一反复开发队列上的探索。

## 一、确定的数学性质与尚未证实的原因

对一个方向、一条有效query行，令

\[
w_j=a_j^0g_j,\quad M=\sum_jw_j,\quad
\mu_w=\sum_jw_j\delta_j/M.
\]

`M=0`单列并返回原分布，不用epsilon把零质量放大为非零响应。

保守算子为

\[
\Delta a_j^C=\rho w_j\left(\frac{e^{\delta_j}}{Z}-1\right),
\qquad Z=\frac{\sum_jw_je^{\delta_j}}{M}.
\]

对任意行公共数c，把所有有效位置的δ换成δ+c，Z同步乘以exp(c)，所以更新不变。若δ全为c，则更新为零。这是已知不变性；不是观察到患者数据以后才提出的统计发现。

加权能量满足

\[
E_w[\delta^2]=\mu_w^2+\operatorname{Var}_w(\delta).
\]

因此应分别报告公共能量占比与相对能量占比。高公共占比只说明大量数值能量位于算子不可识别的方向，**不说明这部分能量包含DSS剩余信号**。共同J0已含Clinical、RNA512及其他输入；公共分量还可能重复这些主效应。

旧核`h(c)=tanh(a_PO+Uc*c)`在近线性区可能产生近似公共变化，但是否接近该区、实际行内变化是否小，必须依赖新增诊断；不能从全局δ绝对均值或跨患者SD代替逐行证据。即使链条同时出现衰减，也不能将其认定为唯一因果瓶颈。

## 二、DIRECT与保守算子到底比较什么

DIRECT使用

\[
v_j=g_j(\delta_j-\mu_w),\qquad
a^D=(1-\rho)a^0+\rho\operatorname{normalize}(a^0e^v).
\]

对固定a0/g，将δ整体缩放为ε·δ，有

\[
\Delta a^C=\rho\epsilon w(\delta-\mu_w)+O(\epsilon^2),
\qquad
\Delta a^D=\rho\epsilon w(\delta-\mu_w)+O(\epsilon^2).
\]

DIRECT里`sum(a0*g*(δ−μw))=0`，因此全局分母的一阶项为零。二者在小修正下同一阶，区别位于高阶重分配和全局归一化几何。DIRECT还允许g=0的边通过全局分母改变概率；这不是新的selector，也不满足原保守算子的“未选边逐元素不动”性质。

据此预先限制结论：

- DIRECT优于S16：支持本协议下更新几何的有限收益；不证明保留公共δ、医学语义正确或selector被改进。
- DIRECT与S16接近：与同一阶性质相容，但不构成数值等价或所有归一化均无价值的证明。
- DIRECT无收益：不能否定尚未测试的公共幅度信息，也不能据此扩展新尺度/温度/深度搜索。
- 两者相对P16的重训对比包含encoder和共同两两主干的优化差异；同checkpoint的off→on才隔离该checkpoint当前修正支路。

## 三、每一级需要读什么，能说明什么

以下均按两个方向分别报告，不混成所谓多层深度。每个checkpoint同时保留fit/heldout分区；primary汇总采用query→WSI→患者等权，pooled-query统计只作为补充。跨seed/fold重复的患者不是独立病例。

|环节|读取的实际统计|可支持的有限解释|不能推出|
|---|---|---|---|
|δ生成|`raw_delta_row_SD`、`weighted_delta_mean`、`weighted_raw_second_moment`|区分δ数值大小与每条query行内相对变化|全局δ非零就应产生attention更新|
|公共/相对能量|`weighted_centered_variance/RMS`、`common_energy_fraction`、`centered_energy_fraction`|定量描述所选a0*g质量上的公共与相对分量|被去掉的公共分量是有效DSS信息|
|概率更新|实际TV、M、Taylor绝对/相对误差、`TV_over_rho_mass_centered_RMS`|判断实际更新是否与相对δ的一阶响应一致|TV大就更接近真实病理关系|
|value汇总|每query消息范数、三角界效率、value中心RMS、Taylor协方差效率|区分“有概率变化”与“变化对value有作用”|消息很小一定是RNA缺信息|
|query平均|平均前消息范数、平均后向量范数、query-average survival ratio|定量描述不同query的有向抵消|平均有抵消就应删除平均或增加head|
|WSI平均|患者WSI平均后消息范数、WSI-average survival ratio|记录多WSI之间的向量抵消|等WSI协议本身不合理|
|风险投影|head输入重建、投影/Cauchy界效率、投影绝对值|判断消息是否主要落在当前readout不读取的方向|这些方向已经含有可泛化DSS信号|
|有界输出|tanh survival ratio、最终correction risk|区分线性投影小、tanh饱和与固定输出尺度|当前cap是唯一原因，或应该自动提高cap|
|held-out排序|repair、damage、净差、3×3 credit转换与配对区间|区分没有排序变化与正负排序变化相互抵消|C净差为零就代表每个患者对都未改变|

各级使用不同单位，不能把δSD、TV、向量L2和risk SD直接排成一个可相加的“损失比例”。三角/Cauchy上界也可能松，不能仅按某个比例最小就指定唯一瓶颈。

### 3.1 技术身份优先于解释

必须先通过并披露以下实际残差，而不是在结果表中手填“零”：

1. 当前map由原算子重算，与模型map一致。
2. 只用μw构造行公共δ，原算子更新回零。
3. 用δ−μw替代δ，更新不变。
4. `(a1−a0) @ value`经原query平均，重建实际head输入。
5. 原head投影和cap/tanh重建实际correction risk。

接受误差仅使用已锁诊断模块的dtype数值容差；这是技术精度门，不是新的性能阈值。零分母、零能量、候选不足行不伪造为高/低比例，保留有效count及退化计数。

### 3.2 不设“0.95即证实”规则

本轮不根据看到的分布新增common fraction≥0.95之类阈值。完整报告连续量、样本覆盖、checkpoint分布、两方向和fit/heldout差异。

可使用的判断用语：

- 若相对δ在数值精度内为零且身份门通过，可说“这些行的公共偏移被算子消除”。
- 若相对δ非零、TV也非零，但value或query平均后明显衰减，可说“在这些固定表示上观察到后续聚合抵消”；不得只归因行归一化。
- 若消息保留而投影很小，可说“当前head对这些消息变化的响应小”；未证明被忽略方向有held-out监督信息。
- 若局部风险非零但所有可比较对credit不变，可说“修正未跨过本轮排序/tie边界”。
- 若repair和damage均非零且净差约零，应说“修复与损伤抵消”，不能称“没有患者特异性变化”。

这些均为固定模型的描述，不是新显著性检验。不能用百万条query行当独立病例生成极小P值。

## 四、held-out预测判定

保留预锁的DIRECT−S16与DIRECT−P16，各Harrell/Uno60，共4项家族。按每seed同折可比较分子/分母加总得到完整C，再平均3个seed；不平均风险。20k患者级共享bootstrap不把pair或seed当独立人群。

主要比较逐项给点估计、三个seed方向、普通95%及Bonferroni4区间；不因某个指标正而覆盖另一个指标。预算门通过只说明预锁schedule/cap资格，不证明最优收敛；预算不足按原规则标未决，不临时增加训练。

同checkpoint的off→on与独立重训DIRECT→P16/S16分开。净repair−damage应等于相同统计对象的C差；Uno60采用一次对应outer-train IPCW。同fold内可比较对、1e−8 tie类别和半分credit沿原协议。外折错误对只作事后评价，不能进入训练教师、epoch或配置选择。

J0是已经含RNA的固定全输入强anchor。即使DIRECT获本架构内收益，也应另报告是否超过J0；若没有，不能声称最终多模态实用收益成立。相反，未超过J0不能代替4项更新机制比较。PFI仅迁移冻结DSS风险，不是新训练或独立外部确认。

## 五、实际覆盖与交付检查

- 30旧S16的checkpoint/receipt原SHA、D1/D2来源及全部必要risk replay保留；不把旧模型计成Phase21新拟合。
- 冻结诊断必须有明确的fit与heldout两份分区标签、人数及WSI数。heldout风险回放不能代替heldoutδ/message链诊断。若通过仅替换审计view的索引实现heldout遍历，原训练权重、Cref、预处理及原ds.fit/valid不变，并明确这不是重拟合。
- 统计“绝对RMS”时注明对应分区、方向、权重及聚合层。不能把mean(row SD)或全局δSD改名为RMS；也不能从已汇总均值逆推出未保存的逐行分布。
- 旧模型的权重受DSS训练；本轮诊断不读结局不等于表示无监督。
- `TV≤rho*M`是保守算子的**上界**，不是下界。DIRECT的界及未选边行为按其实际算子单列。
- 仅公开匿名矩统计、计数、来源和可核验残差；患者风险、地图、gene/患者身份及模型权重保持私有。

## 优势、风险与置信度

**优势**：以精确不变性和可重建链条检验具体问题；只改一个更新算子；不以新增拟合或尺度放大代替机制证据。

**风险**：固定强anchor已包含组学；独立训练的主干会漂移；DIRECT与Cons一阶相同；65事件、重复开发、bootstrap未含重训/研究选择方差。以上限制不能通过共同能量占比或技术PASS消除。

**置信度**：公共shift不变性、一阶等价和聚合重建的数学判断为高；患者层面的实际衰减位置待真实诊断判定；删除分量是否包含可泛化DSS信息及外部迁移能力，本轮不能单独确证。

**拓展思考／你可能想知道**：若局部risk确有变化但repair/damage都为零，下一步讨论应围绕已观察排序边界，而不是仅提高δ幅度；若DIRECT与Cons都受到同一阶相对分量限制，不能将换一个softmax写成对公共偏移假说的独立证伪。
