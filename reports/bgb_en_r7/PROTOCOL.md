# R7：尺度、独立方向、程序读出与状态门控的有限机制审计

**协议草案：由主执行代理定稿并建立独立运行锁后才能正式训练。** 本文与 `configs/bgb_en/r7_inner_v1.json` 描述同一有限实验；不是已经完成的结果报告。科学源码、资产 SHA、分区/配对映射、六个 RMS 和初始化状态必须由不可覆盖的私有运行锁绑定。

## 1. 研究问题与共同范围

R6 的源任务有真实配对敏感的程序预测线索，但目标条件修正对读取与风险的有效影响极小。R7 依次检验尺度接口、串联依赖、程序位置读出、状态门控与患者条件，**不扩展成另一个更复杂的融合网络，也不预设任何修改一定有效。**

基础是已发布 R6 提交 `38fda0f` 与目标合同 `a6545f3e8fdbdb6e992c1310948b31a25845f57905b65eeb0064cb63483673ab`。旧 R6 B7/B3 可只读列作历史背景，不能算作新 R7 重训、确认结果或新独立对照。

- TCGA COAD/READ 原患者级分折不变。仅用 outer1 的 409 名训练患者，在其中复用原三个 inner 分区；103 名 outer1 测试患者继续封存。
- seeds 固定 31/47/83，每臂 3分区×3seed=9次真实拟合；同 seed 的既有共享参数初始化遵循 R6，不按 arm 重新挑 seed。
- 源模型复用完整 R6 full-source S7 MATCHED seed31，包括 projection、四程序 Query、tissue/conditioning 和原基因 stem，全部冻结。原 CP、原 RNA512 教师和 same-teacher scalar beta 权重/预处理精确复用，新 source/teacher/scalar 拟合均为0。
- offset 逐元素固定为原 `CP + beta×RNA512 risk`；共同病理、基因、Hallmark 主效应允许按原同预算训练。所有主效应输入保持 raw 坐标，R7 尺度处理只作用于交互路径。
- UNI2 候选、坐标图、四个 ST 程序、760 gene union/765 signed membership、原始 RNA/mask 与主效应均不换；CNA、文本、新程序/新网络主干不加入本轮。

这是已反复使用的内层开发协议，不是新的独立临床验证。六个 ST 来源供体也被开发复用，source/TCGA 患者互斥仍为 `UNVERIFIED`。

## 2. 固定矩阵与正式预算

| 训练臂 | 交互尺度 | 反向路径 | 风险读出 | 状态门控/轮数 | 条件 |
|---|---|---|---|---|---|
| ADD | 交互关闭 | 无 | 同预算主效应 | 无/0 | 无交互真实重训 |
| RAW | 原始 | R6 串联双差分 | 共享均值 | 无/1 | 正确 |
| NORM | 六个固定 train-RMS | 串联双差分 | 共享均值 | 无/1 | 正确 |
| INDEP | 同上 | 独立反向 | 共享均值 | 无/1 | 正确 |
| SLOT | 同上 | 独立反向 | 四程序独立线性 | 无/1 | 正确 |
| GATE1 | 同上 | 独立反向 | 四程序独立线性 | 有/1 | 正确；**事前固定主候选** |
| GATE2 | 同上 | 独立反向 | 四程序独立线性 | 有/2 | 正确；仅单/双轮对照 |
| SLOT_WRONG_RNA | 同上 | 独立反向 | 四程序独立线性 | 无/1 | 错初始 RNA 上下文 |
| GATE1_WRONG_RNA | 同上 | 独立反向 | 四程序独立线性 | 有/1 | 错初始 RNA 上下文 |
| GATE1_WRONG_PATH | 同上 | 独立反向 | 四程序独立线性 | 有/1 | 仅反向错初始病理上下文 |
| GATE2_WRONG_RNA | 同上 | 独立反向 | 四程序独立线性 | 有/2 | 每轮保留错初始 RNA 条件 |

**全部11×9=99次拟合，每次80个完整风险集更新，共7,920次正式优化更新。** B0/B1 保留18行零拟合参考，引用同一旧风险及真实回放谱系，不重复优化。每个新结构都真实重训；不能将旧 R6 checkpoint 充当 RAW 的新训练。

99次全部执行，不按早期性能删臂、增加轮数/epoch/seed/候选，不从阶段成绩里选一个替代 GATE1。GATE2 不参与单/双轮选优。资格、无更新诊断和独立前向回放另列调用/时间成本，不能算作正式 optimizer fit，也不隐藏实际发生的调用。

## 3. R7.0：只做正标量 train-RMS 接口对照

### 3.1 六个标量与拟合

尺度名固定为 `context / patch / query / evidence / gene_key / reverse_query`。每个接口只有一个全局正标量：

`s = max(sqrt(sum_float64(valid_float32_value²) / valid_scalar_count), 1e−6)`。

平方项在 float64 累加后求均值与平方根；使用真实 mask 剔除 padding/未测项，按有效标量计数，不按765条程序边重复计 gene Key。禁止每患者/每向量 L2、LayerNorm、新可训练归一化、对 Δ 作归一化或验证集选系数。

每个 partition/seed 用同一个**未训练 NORM 模型**，只读取当前 train 患者，分三阶段建立尺度并冻结：

1. 原 source G0、原 adapted Patch、原 base Query 和原 gene Key 分别确定 context、patch、query、gene_key；gene Key 用 union760 中本人实际可测 gene mask，不按 signed765边重复。
2. 用已经缩放的 base Query 与 Patch 重算无条件 P0，确定 evidence RMS。
3. 对 `query_projection(P0/evidence_RMS)` 确定 reverse_query RMS。

所有非 RAW/ADD 配方共享这六个拟合值；不因改方向、slot、gate或错配而重新拟合。六值与 valid scalar counts、原 RMS、epsilon 是否触发、初始化 SHA、train scope/hash 一并留存。有效标量计数为0时必须阻断，不能用epsilon伪造可拟合接口。此过程不读取任何标签或 validation，不做优化。

### 3.2 作用范围和解释

交互 G0/context 除以 context RMS，Patch 除以 patch RMS，Patch Query 除以 query RMS，反向 Evidence/Query 与 gene Key 使用各自固定尺度。交互 gene Value 也除以 **context RMS**，保持 G0 与 GeneRead 的64D状态坐标一致；gene Value 自身 RMS仅作诊断，不成为第七个应用尺度。

主效应仍读取 raw G0、raw Patch 和共同 Hallmark。原 source prior 保持0.1、同时出现在 base/conditioned Patch读取；内容 logits 与固定 prior 的相对大小可能因此改变。

核心比较 `NORM−RAW` 检验的是**联合尺度/参数化和固定 L2 几何共同变化**，不能唯一识别哪个接口造成 R6 的塌缩，更不能将其自动当作生物信息增加。RAW 的新引擎等价性另做资格检查；不以历史分数逼迫新 RAW 必须得到同一优化轨迹。

## 4. R7.1：两个方向可以独立产生修正

记 P0 为本人无条件 Patch读取，G0 为本人原始测量程序状态，所有 Value 均来自本人原始证据。

- 正向仍为 `ΔP = PatchRead(q_base + adapter(RNA_route), ownPatch) − P0`。
- NORM 的旧反向为 `GeneRead(Pcond, ownGene) − GeneRead(P0, ownGene)`，依赖前向变化。
- INDEP 起的新反向为 `ΔG = GeneRead(PATH_route, ownGene) − GeneRead(zeroEvidence, ownGene)`。

两个 GeneRead 使用同一 query_projection、Key、Value、成员与测量 mask。首轮 PATH_route=P0，因此即便前向 ΔP=0，反向仍可能存在独立读取变化。零 Evidence 是同一算子空间内的明确基准，不用不同语义的 G0 直接相减。

核心比较 `INDEP−NORM`。这只检验解开串联依赖是否有帮助；独立反向仍可能含单模态未解释主效应，**不能称为已统计正交的纯二阶交互**。保留方向风险与配对修复/损伤检验。

## 5. R7.2：不提前合并程序的低容量线性读出

每个方向四个64→1、bias=False的线性头。有效程序集合 mask 为 m，读出为：

`r_direction = sum_k(m_k × w_kᵀΔ_k) / sum_k m_k`；全无效时为0。

每个 w_k 初始均等于原共享头 w，保证所有缺失模式下初始函数等于旧有效程序均值头。该方向头的参数惩罚为 `λ/4×sum_k||w_k||²`，λ=1e−4，tied 初值时 L2 数值也等于旧头。其他唯一可训练参数仍用原 λ，只计一次。

每方向从64增加到256个有效可训练参数，双向增加384；若注册状态仍保留冻结旧head，注册参数总量应另报，不与有效参数混用。不声称完全同容量或优化轨迹。SLOT仍是低容量线性相加，不加入新的风险非线性。

核心比较 `SLOT−INDEP`。报告四程序、两方向与总交互风险，记录程序间抵消；程序选区不同不自动证明正确患者条件有效。

## 6. R7.3：真实状态门控和同步单/双轮

以64D的当前状态和对方路由状态拼接：

- `g_p = 2 sigmoid(Linear128→1([currentP, RNA_route]))`。
- `g_g = 2 sigmoid(Linear128→1([currentG, PATH_route]))`。
- `P_next = P0 + g_p × ΔP_current`。
- `G_next = G0 + g_g × ΔG_current`。

两方向各一套 Linear，跨程序共享；权重和 bias 全0初始化，故初始 gate=1。每方向129参数、双向258；范围0–2，不是医学置信度。U为identity，不再加入η、额外投影或新的风险饱和函数。

每轮从同一 currentP/currentG 同步计算两个更新，再同时写入；下一轮 Query 使用更新后的 P/G，读取仍回到本人原始 Patch/gene Value。更新始终锚定 P0/G0，不累加偏移。所有参数跨轮共享，因此 GATE2 相对 GATE1 增加计算、不增加参数。

无效/拒绝消息的 Δ 为0时，状态必须精确退回 P0/G0；读取注意力不加随机 dropout。GeneRead 的自然空成员返回无效/零；非空 gene成员并不因此具有额外医学 null。Patch null Value为0，其权重依赖候选数，不能当作校准的组织来源置信度。

核心比较 `GATE1−SLOT` 与 `GATE2−GATE1`。前者检验真实状态 gate，后者检验共享参数迭代。**GATE1 始终为主候选，不从两个轮数的已见验证成绩选更好者。**

## 7. R7.4：错初始条件的完整逐轮定义

### 7.1 固定映射

沿用 R6 无 DSS/PFI 的 Hungarian 配对算法和优先级：实际 RNA finite pattern、COAD/READ、Clinical Stage、平台 missing-pattern proxy。每个 train/validation 分区各自无自配对双射，base_seed270577、partition stride100、valid offset1；跨 seed、可比 arm共享映射。保存匹配计数、mapping SHA与分区身份资格，不将validation患者当train配对来源。

### 7.2 RNA 控制

SLOT/GATE1/GATE2 的 wrong-RNA 不是整模态替换。每轮只使用：

`RNA_route_l = foreignG0 + (ownG_l − ownG0)`。

因此错误初始分子上下文在两轮中保留，本人的反馈增量仍相同定义；不会第二轮恢复本人初始条件，也不读取另一患者更新后的 G 或对方反馈路径。上下文使用 foreignG0 对应的 source mask；本人原始 gene Value、测量 mask、Patch、raw主效应、RNA512 risk与offset不换。

它只检验**初始分子上下文的患者身份**，不是“整个组学模态都错配”，也不是把所有迭代信息都来自错患者。

### 7.3 病理控制

仅 GATE1_WRONG_PATH 真实重训，反向上下文为：

`PATH_route_l = foreignP0 + (ownP_l − ownP0)`。

foreignP0 按当前参数、当前来源患者的无条件 Patch读取实时核算，不使用陈旧初始化缓存。正向RNA条件仍本人；本人 Patch/gene Value与mask、raw主效应、offset均不替换。只变反向上下文及其对应source mask。该控制为单轮，不额外扩展双轮 wrong-PATH。

训练前必须在同一参数状态的反事实前向中验证 wrongRNA/wrongPATH 时本人的主效应与offset逐元素保持，外来数据只进入指定路由。真实重训臂保持相同主效应输入、公式、容量与初始化，但各自学习后的主效应参数可以不同，不宣称跨不同拟合的主效应风险逐元素相同；冻结关闭与真实重训的结论分开。

## 8. 训练、诊断与逐检查点真实回放

训练80轮，唯一主性能 checkpoint=80。Adam lr1e−3、weight_decay0、batch8、dropout0、parameterL2=1e−4、总风险修正惩罚0.1δ²、交互惩罚0.1cross²、clip5；slot头按前述1/4因子计参数惩罚，所有正则在完整风险集口径加一次。

完整 Efron 风险集使用确定性两遍 output-gradient cache，不在微批独立计算 Cox。严格确定性、CUBLAS `:4096:8`、TF32关闭；缓存重放最大误差2e−5，不能通过放宽阈值处理错误。正式失败或partial保留，不自动新seed重试。

### 8.1 每20/40/80轮

保存实际state和optimizer、训练/验证预测、尺度/状态合同，并对每个checkpoint在 **train与validation** 做独立stateforward真实回放：风险误差≤2e−6且pair-credit变化为0，0新增更新。20/40轮仅作轨迹核验，不选epoch。

### 8.2 每20/40/60/80轮的完整train诊断

不再只取首八训练患者。用当前全部train风险的 Efron Cox 导数 `w_i=∂L_Cox/∂risk_i`，分别记录：

- 同一w下 `sum_i w_i ∂risk_i/∂θ` 与 `sum_i w_i ∂cross_i/∂θ` 的参数组梯度。
- 独立的总/交互 residual penalty 梯度。
- 同一唯一参数分组的 analytic parameter-L2 梯度，含slot头1/4因子。
- 总/主效应/方向风险SD、六接口与Query范数、正确/错误条件Query TV、程序风险抵消、gate/null、测量覆盖和轮间状态轨迹。

这些梯度 scope相同但对象不同，不能相减范数后作因果归因；共享参数不重复计数。诊断使用前向/反向但**不更新参数**，单独记调用与耗时。60轮只做train诊断，不读validation选择或报告60轮验证性能。任何未保存量保持 `NOT_SAVED`，不得回填或补造。

## 9. 预锁配对比较与统计

| 阶段/目的 | 候选−参照 |
|---|---|
| R7.0 尺度接口 | NORM−RAW |
| R7.1 独立方向 | INDEP−NORM |
| R7.2 程序读出 | SLOT−INDEP |
| R7.3 状态gate | GATE1−SLOT |
| R7.3 单/双轮 | GATE2−GATE1 |
| R7.4 主效应条件增量 | GATE1−ADD、GATE1−B1 |
| R7.4 初始分子身份 | GATE1−GATE1_WRONG_RNA |
| R7.4 初始病理身份 | GATE1−GATE1_WRONG_PATH |
| 程序头与双轮的分子身份 | SLOT−SLOT_WRONG_RNA、GATE2−GATE2_WRONG_RNA |

Harrell与Uno60分别报告。每seed先加三个分区内部 concordance 分子/分母，再等权三个seed；不拼跨分区风险。Uno60 censor KM只由当前训练分区拟合，tau60月。真实配对修复/损伤必须来自同分区同患者风险；不能由C-index差值伪造。

PFI只评价冻结DSS模型，不训练PFI模型、不参与选择。不计算bootstrap、CI、p值；不将seed当独立队列。缺失/den0保留null/INCONCLUSIVE，不给假0分。所有99单元及必要回放完整后才能给总体结论。

每比较只有在两项DSS差值均>0且证据完整时，才标“方向性支持”；一正一负不是共同成功。给出每seed/分区方向，不事后增胜折数门槛。R7不从表格中选一个最高值模型，不自动启动outer。患者条件、普通主效应和医学机制分别解释。

## 10. 资格、报告与解释边界

正式运行前必须通过真实数据前向/梯度资格：

- RAW新engine和已锁R6函数的同state前向等价，而不是用历史优化分数作实现验收。
- 六RMS仅由train产生且跨所有归一化配方一致；main不受归一化控制影响。
- INDEP的反向不被前向Δ为0强制关闭；SLOT初始函数/L2匹配旧头的所有mask情况。
- GATE1零初始化gate=1，真实state更新与SLOT相应初始函数匹配；GATE2确实读取上一轮更新，且共享参数。
- 错RNA每轮保留foreign初始条件和own反馈；wrongPATH使用当前foreignP0；本人原Value/mask/main/offset不变。
- 全mask/null为0修正，无NaN；两遍完整风险集梯度一致。来源、教师、prepared和CP scalar引用SHA/hash/分区范围精确回放。

公开交付应包含匿名逐折/seed表、固定配对指标与修复/损伤、六尺度资格、每checkpoint真实回放、20/40/60/80 train梯度/状态诊断、有效参数/计算与成本、缺失块/COAD-READ/平台代理分组、决策与源码/配置SHA。私有风险、IDs、表达、coords、weights和凭据不公开。新增命名空间不得覆盖R6。

报告中事实字段与成功字段分开，例如`changes_source_prior=true`和`direction_supported=false`；避免R6中一个带“changes_prior”名字的比较成功布尔被误解成架构事实。

**优势：** 有限、真实重训、强主效应保留、每阶段一个主要问题，患者身份与读出路径可逐项核验，不需扩大主干。

**风险：** 同一开发人群反复使用，来源少且与TCGA互斥未证；尺度/slot正则几何并非完全无混杂；独立反向可能包含剩余主效应，gate幅度不是医学置信度。即使Δ增大或内层微正，也不能证明细胞来源、因果基因关系或独立临床泛化。65个DSS事件是全队列量级，每个inner-train更少，当前不能把它当作唯一瓶颈。

执行/源码资格与固定预算的置信度在验收后可标高；任何未来性能与医学机制在结果产生前均未知。负结果只约束当前配方，不能否定整个医学研究思想。

## 拓展思考

应分别回答“路径能否改变真实证据读取”“变化是否进入风险”“是否修复多于损伤”“是否依赖本人初始条件”，不要用单个幅度或C-index替代整条链。

## 你可能想知道

本轮11臂是预先锁定的99次真实训练，不是最多99次的性能自适应搜索；外层始终 `NOT_RUN`。报告、绘图或发布守护的可修复问题不授权更改已训科学源码或重跑模型。


执行说明：正式结果固定第 80 次更新；20/40 次检查点也会做真实前向回放，60 次只记录训练诊断。调度初始 4 路，最多 6 路，仅按真实资源资格核验扩并发，不按模型成绩追加预算。R7 继承的 R6 29 份科学文件禁止修改。
