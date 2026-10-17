# Phase20：选择性Clinical修正与通路Token容量审计

## 1. 范围、来源和固定矩阵

父提交 `065a6a8dea4f181f5b6e9fb492ed3ebb56f6c68c`；继承Phase19训练来源 `ce14bd3053f51bbf277d21392419da70c41208581bb9580d78ca546f953873d5`。新分支 `research/phase20-selective-capacity-audit`，私有目录 `/root/autodl-tmp/phase20_selective_private`。配置以 [phase20_selective_attention.json](../../configs/phase20_selective_attention.json) 为准，正式训练前锁定源文件、数据及配置哈希。

TCGA-COAD/READ原512患者、65个DSS事件，固定5个患者外折×seeds 31/47/83。九臂各15个inner及15个fresh outer，共 **135 inner+135 outer=270次正式神经拟合**。只读继承两个ridge参考×五折×inner/outer的**20个唯一凸模型，新增凸拟合为0**。跨seed引用不是新增拟合。诊断短拟合、失败后实际optimizer更新和无优化的探针另计，不以文件数代替实际执行。

本轮只分开检验两项：A，Clinical是否能选择性、受约束地修正已有P–O关系；B，在固定融合架构下，通路标量化是否构成有效信息瓶颈。这里预锁的是本轮流程；同一TCGA队列已被反复用于方法开发，**并非未触碰队列上的确认性验证**。不增加RNA-only、排序蒸馏或额外通用排行榜，不改Phase19历史。

## 2. 共同输入与冻结主效应

继承同一Clinical、TITAN S32、RNA512、CNA gain/loss、测量mask、八组membership、UNI2-h物理足迹及患者折。所有预处理只来自相应fit患者；inner数据不包含outer-test。UNI2-h保持冻结，每患者最多512个patch，多WSI患者按WSI标量风险等权汇总。坐标/足迹QA不能替代原图组织学验证。

共同主效应为继承的J0：Clinical、S32、RNA512、CNA gain8/loss8、UNI均值PCA32和RNA/CNA覆盖率8维各一组的全输入Ridge Cox。J0权重冻结为float64 buffer；训练fit offset为`float32(float64(X_fit) @ w_J0_float64)`，validation/test直接使用继承`simple.J0.risk`并逐元素校验，以预计算anchor override进入forward。没有自由stacking系数，也不从outer-test错误定义教师或残差标签。

只在fit患者上计算

\[
s_{fit}=\operatorname{SD}_{ddof=0}(X_{fit}^{float64}w_{J0}^{float64}).
\]

这只是固定预测单位，不是拟合meta模型或风险教师。若为0，两条神经风险上限也为0，不以epsilon人为放大。真实数据中是否非零由审计报告。

## 3. 共同两两主干与风险边界

所有神经编码器、主干、head均fresh初始化，不加载Phase19已训练神经权重。共同主干沿用Phase19 A1架构：P–O点积，加P–Clinical和O–Clinical两两logit项，pair rank4；Clinical及Omics始终为本人。双向softmax、共同value投影、消息汇总和原head架构保持一致。

\[
r_i=r_{J0,i}+s_{fit}\tanh(H_{base}(m_i^0))
 +0.25s_{fit}\tanh(w_{corr}^T\Delta m_i).
\]

`w_corr`对应无bias单层Linear。新修正最后的tanh之后没有可训练乘数或额外预测头，因此绝对风险修正不超过`0.25*s_fit`；主干风险不超过`s_fit`。无新修正的臂直接去掉最后一项。限制最终输出可避免小attention修正被下游无限放大，但**不保证患者排序、C-index或主效应收益必然保持**。

“两两主干”仅描述logit项。两两logit之和经过softmax和共同非线性head，不是严格的`f_PO+f_PC+f_OC`二阶风险函数。新修正不是Clinical首次进入模型，其作用必须在本人Clinical已存在的主干和anchor条件下解释。

## 4. B：同输入、同参数形状的通路容量对照

保留RNA、CNA gain、CNA loss各八个program，共24个Token。每个channel-program的完整成员基因向量接受16条独立signed可学习filter；按每名患者实际测量位置归一化：

\[
z_{ikr}=\frac{\sum_{g\in G_k}W_{krg}x_{ig}M_{ig}}
 {\sqrt{\sum_{g\in G_k}W_{krg}^2M_{ig}}}.
\]

零分母严格返回0。RNA与CNA各自测量mask保留，gain与loss不相抵消；完全未测量Token无效且置0。24组filter分别学习，不把16个坐标当作16种医学语义状态。

P1/S1使用固定rank1映射

\[
Q_1=\sqrt{16}\,\mathbf1\mathbf1^T/16,
\qquad (Q_1z)_r=4\operatorname{mean}(z),
\]

P16及其修正臂使用`I16`。二者都将16坐标与一项coverage拼成17维，再用三份modality-specific `17→16→32` MLP（各自共享该modality的八个program）和channel-program slot。相同filter/head参数形状及fresh随机初始化程序避免将参数数量变化误作维度效应。

`Q1`与`I16`的Frobenius范数相同，只在等方差、各向同性响应下名义匹配期望能量；`Q1`不是单位算子范数的正交投影，真实患者SD未必匹配，必须报告。有效rank受成员数、缺失、filter相关性和训练共同限制，不能默认16条filter就有16个独立信息方向。B不使用PCA。

L0是旧scalar tokenizer的fresh协议桥：原非负softmax基因权重标量与coverage进入`2→16→32`，同样冻结anchor和限制主干输出。**L0名称表示legacy scalar，不表示L₀稀疏正则。** L0与P1还存在signed filter和输入MLP结构差异，不能将L0对比当成纯rank消融；预锁的纯容量主比较是P16−P1。

固定测量mask时，原RNA交互路径最多先观察八个线性标量方向；后面的32维展开不恢复被压缩方向。但这不证明DSS充分统计量必然丢失。主效应中的原始RNA512仍保留，只能说明旁路存在，不能代替Token信息保留证据。固定fusion指使用同一融合架构/协议fresh训练，不是将新坐标直接接到旧融合权重。

## 5. A：质量守恒的受约束Clinical修正

对同一patch–Token pair，以共享32维P/O及Clinical生成width8核：

\[
h(c)=\tanh(U_pp+U_oo+U_cc),\quad
\ell=w_g^Th(c),\quad
\delta=0.5\tanh\{w_d^T[h(c)-h(C_{ref})]\}.
\]

`Cref`仅为fit患者Clinical均值。新支路没有Clinical向风险head的直接拼接。两个attention归一化方向分别校准gate，每行eligible数为n，预算`b=0.25*n`：

实际计算采用`h0_pre=Up*p+Uo*o+Uc*Cref`、`h0=tanh(h0_pre)`、`h=tanh(h0_pre+Uc*(c−Cref))`。Uc无bias，数学上与上式等价，同时避免两条不同浮点求和路径使`c=Cref`产生非零差值；CPU/CUDA的逐元素回退另有真实算子测试。

先按每方向/每行eligible位置作`ℓhat=(ℓ−mean_eligible(ℓ))/max(SD_population,1e−6)`；mean和SD保留梯度，常数行和空行安全处理。两种门使用完全相同的归一化，防止raw score初始幅度很小使投影落在几乎均匀的稠密区；这是看本轮DSS成绩之前锁定的单位约定，不是搜索temperature。

\[
g^{sparse}=\operatorname{clip}(\hat\ell-\tau,0,1),\quad \sum g=b;
\qquad g^{dense}=\sigma(\hat\ell-\tau),\quad\sum g=b.
\]

τ按方向/行分别求解，固定64次solver迭代。**预算是gate质量，不是恰25%的非零边数**；clip门仍有可能所有位置都非零，归一化不保证指定支持，raw/normalized score和实际支持及质量误差必须报告。Dense对照同样校准预算，不能用任意sigmoid均值作为不公平对照。

设本人主干分布为`a0`，在eligible位置：

\[
w=a_0g,\quad M=\sum w,\quad
Z=\frac{\sum w\exp\delta}{M},\quad
a_1=a_0+0.5w(\exp\delta/Z-1).
\]

空eligible或`M=0`直接返回a0，不做无效除法。δ与h的pair定义共享，但两个方向的gate预算、M、Z和归一化分别计算。

可直接核验：

- `gate=0`的位置概率保持a0，而不只是logit不变；δ全0时整行严格回原。
- `sum(a1)=1`且`a1≥a0−0.5w≥0`，总变差`TV(a1,a0)≤0.5M`。
- `C=Cref`使δ严格为0，因此新修正归零；这不删除本人主干中的Clinical。
- Δmessages来自`(a1−a0)`乘**相同的本人value**。只将Δmessages送入最终受限无bias修正头，不增加Clinical直达预测分支。

这些是实现的数学/数值门，不是新理论、医学语义或预后优势。低gate质量不保证选择关系数少，非零gradient或更尖锐attention不证明独立条件预后信息。对归一化attention直接做L₁惩罚恒为1，本协议不采用这种无效稀疏约束。

## 6. 九个固定实验臂

| 臂 | Token/新修正 | 归因用途 |
| --- | --- | --- |
| L0 | 旧scalar，无新修正 | 与历史scalar的fresh协议桥 |
| P1 | signed16经rank1，无修正 | rank1基础 |
| P16 | signed16经I16，无修正 | B容量主对照 |
| S1 | rank1+选择性修正 | 容量×修正交互的低rank参照 |
| S16 | rank16+选择性修正 | 候选 |
| D16 | rank16+同质量预算dense门 | 选择性是否超越普通dense调制 |
| F16 | 仅gate用Cref；δ仍用本人Clinical | gate的Clinical依赖是否必要 |
| WC16 | 新修正核用错误Clinical | 患者Clinical对应关系 |
| WO16 | 新修正核用错误Omics | 分子context对应关系，附支持差异限制 |

WC/WO均真实重训，所有本人主干、value及anchor保持匹配。Donor只在同一fit/held-out分区内置换；沿用已审计癌种/Stage等规则，不以DSS错误择donor。WO的eligible为本人mask与donor测量Token mask的交集；缺donor的关系不修正，gate预算按实际eligible数。这可能同时改变可用支持，须记录覆盖率差异，**不能将WO下降纯粹归因为患者身份或因果语义**。

冻结S16的wrongC/wrongO/constantC/correction_off推理干预另列，不冒充WC/WO重训。`correction_off`只关新修正，保留本人完整A1主干及冻结anchor。

锁前另加两项**只针对冻结P1/P16**的B敏感性诊断，不新增模型或主要比较：`fullOmicsWrong`把interaction的R/G/L/masks换为donor，本人visual/Clinical/冻结main不变，持久化私有`full_omics_wrong_risk`；`uniformPair`将双向a0改为valid候选上的均匀分布，保留本人Token和值，持久化私有`uniform_pair_risk`。公开仅汇总指标，标签分别附`__FULL_OMICS_WRONG`/`__UNIFORM_PAIR`。它们检查rich Token收益是否依赖P–O兼容度，防止把额外非线性Omics主效应称为跨模态机制；**冻结后的分布干预不是同条件重训的公平性能证明**。

连同既有只读`HIST_CS_S32_CONCH32`历史参照（视觉/协议不同，非同输入消融），最终20个model标签×3seeds=60个OOF向量、300行逐外折指标。主要14统计量不变。独立回放包括270checkpoint×5个context模式、270个anchor，以及P1/P16共60checkpoint×2个额外冻结模式，即120次额外、合计1740个prediction回放；这些不是神经网络重训。

## 7. 训练、评测和预锁比较

沿用max160、minimum8、patience40、validation loss EMA0.2、AdamW lr0.001/WD0.01、dropout0.1、gradient clip5。完整患者风险组成一次Efron风险集；microbatch32只管理显存，nonreentrant checkpoint保留dropout RNG，不采用microbatch Cox。所有135个inner选择先统一锁定，再fresh outer按相应臂/seed/fold准确epoch训练。冻结J0不进入optimizer，不增加其L2常数项或自由缩放。

每seed先加总五外折的Harrell/Uno60分子和对应分母，再报告三个seed的C；不把15 cells称为seed mean，不平均风险，也不跨不同fold的隐空间直接拼接训练meta模型。Uno60使用对应outer-train反向KM、tau60，并逐折与独立实现核验。

七个primary比较：

1. P16−P1。
2. S16−P16。
3. S16−D16。
4. S16−F16。
5. S16−WC16。
6. S16−WO16。
7. DID=`(S16−P16)−(S1−P1)`。

每个同时报告Harrell/Uno60，构成**14项固定家族**。20,000次外折内患者bootstrap，各模型/seed共享同一患者计数，比较对象是mean seed C及其线性DID。普通95%与Bonferroni14区间（覆盖率99.642857%）同时报告，置信区间条件于固定已训练模型，不包含完整重训练/研究者选择方差。

Secondary为S16−S1/L0/J0/J_CS及四个冻结修正干预，共16个指标统计量，另列家族校正，不据其选择主模型。PFI仅迁移冻结DSS风险，不做PFI训练、epoch/门槛选择，也不是外部验证。

P1/P16的冻结B敏感性另列探索家族：各自matched−fullOmicsWrong、matched−uniformPair，以及`DID_uniform=(P16−P1)−(P16_uniform−P1_uniform)`，五对×两指标=10项；单独校正，不混入主要14项，也不将冻结分布干预当成真实重训的交互证明。

## 8. 独立判定轴

| 输出 | 预锁方向门 | 不能替代的证据 |
| --- | --- | --- |
| A选择性修正 | S16相对P16/D16/F16/WC16/WO16的两指标均值均正，且每项≥2/3个seed为正 | B或DID正不能代替A |
| B容量 | P16−P1两指标均值正、每项≥2/3个seed为正 | 不自动证明真实基因集语义优于随机集合 |
| DID | 两指标独立报告同方向/seed门 | 不作为A或B的强制通过条件 |
| 实用性 | S16相对J0及J_CS两指标均无均值损害，配对差值/区间单列 | A方向正不等于强简单基线上实际获益 |

上述方向判断要求相应参与臂的预算与算子/风险上限门通过。各臂inner实际触顶比例不超过20%只是预锁预算门，不证明全局收敛；预算不足标对应轴`BUDGET_UNRESOLVED`，完整负/未决结果照实交付。方向门通过只称directional candidate；相应Bonferroni14下限为正才具备本轮校正统计支持，仍不能声称独立外部确认。

A还需预锁的**实际选择行为门**：只读15个selected inner S16 checkpoint，在每个方向的eligible≥2行计算`count(g>1e−6)/nEligible`；先WSI内query平均，再患者内等WSI平均，最后all-fit患者等权平均。15个checkpoint的该均值中位数在两个方向都严格小于0.75，才称“实现部分边选择”。少于2个候选的行比例、所有base-valid行的支持、raw score行内population SD≤1e−6的退化率及各层计数全部同时报告，不能删去常数行再宣称选择。行为门不满足仍照报预测效能，但A不得称选择性机制成立。

S/D的未加权`sum(g)`相同不保证`M=sum(a0*g)`相同；M、`TV/(0.5*M)`与`correctionSD/anchorSD`分别实测，不声称两者实际修正幅度匹配。行为门不证明语义真实性，也不替代性能、患者身份或强基线对照。

## 9. 无标签机制诊断、风险和交付

Map/Token/风险变化摘要覆盖所有fit患者，按chunk聚合，不按DSS标签或模型错误选人，不持久化raw maps。隔离新修正输入梯度取八名固定等距fit患者并去重，私下审计其patient/WSI并集，只公开计数。跨seed/fold重复的探针不当独立病例。相同epoch梯度与selected checkpoint梯度分开，未选中或为0的梯度不伪造为学习成功。

报告gate实际支持/质量、M与TV、正确/错误context的eligible差异、风险cap、filter响应与Token方差/有效rank、输入和参数梯度、实际epoch、计算与GPU采样。16D是可学习坐标，不是已验证医学状态；attention或通路过滤器不替代独立形态真值。容量变化、真实响应尺度、缺失/平台差异和有限事件都可能影响结果。

**优势**：冻结强主效应、最终风险幅度可核验；同形状rank对照；把稀疏选择、Clinical门依赖和患者对应分开；A/B/DID/实用性各自回答问题。**风险**：65事件、反复开发队列、不同支持与响应能量、非凸优化，以及已有gating/FiLM/稀疏attention重合；不能归因唯一生物因果机制。数学和来源定义置信度高，实际效能与外部可迁移性在完成本轮前未知。

公开新增源码、配置、逐折/逐seed及配对表、匿名审计、RESULTS/decision和复现记录；患者OOF、风险、权重、gene列表、donor索引、原始map/token/PCA状态及凭据私有。保留全部历史结果。本轮结束即停止自动rank/depth/gate/scale/endpoint搜索，由实际证据决定后续。

**拓展思考／你可能想知道：** 若B改善而A没有改善，优先说明表示容量有价值；若A方向成立但仍不及J0，说明局部关系机制与实际生存收益尚未闭环。两者都不能用新增模块数量替代验证。

## 10. 执行中D1→D2审计修复记录

本节是追加的故障与来源兼容说明，不覆盖原配置、原始源锁、epoch选择或已完成结果。D1来源为`d1d7f8176f18417b8107767700f68ec0292181c3a0e76fce37894e06cc0c3192`。停止时已完成135个inner与54个outer，共189个checkpoint；`outer/L0_seed31_fold3`在优化完成后的anchor逐元素审计失败，另外7个运行单元由调度器中断。失败/中断实际发生的optimizer更新按增量账本另计，不因未产生完成checkpoint而归零。

该故障来自一个三WSI患者的**重复float32 anchor再取均值**：相同数值的CUDA归约仍可产生1 ULP差异，本次最大差值为`1.4901161193847656e-8`。修复限定在`forward(mode='anchor')`的审计返回路径：先逐WSI验证模型main输出与预计算anchor override位级相同，再从同患者的相同值中取首值。训练以及matched/wrongC/wrongO/constantC/off/fullOmicsWrong/uniformPair的风险计算、WSI平均、模型、损失、优化器、输入、seed及选定epoch不据此改变；“anchor审计取首值”不能推广为修改正常多WSI风险汇总。

D2是否满足上述边界由明确的代码差异、数值兼容测试和修复清单核验，不能只凭“差异很小”认定兼容。189个D1完成checkpoint及其原receipt保持原字节、原source；只接受修复manifest逐项列明身份与SHA的D1产物，不能普遍放行任意旧source，也不能改写checkpoint来源使其看似D2重训。其余81个outer按既有inner锁定epoch和原fresh初始化规则完成，失败/中断的部分更新不续接为另一训练预算。

最终报告同时披露D1/D2完成模型数量、原始协议与修复锁、逐项兼容/回放证据、失败与中断的实际步数。270仍指最终完成的正式模型，额外失败/中断尝试与诊断短拟合分别记账。原14项主要比较、16项次要统计、10项冻结探索及全部科学判断门不因该故障变化；不能借审计修复重新选择架构、预算或终点。
