# B0–B7：语义约束、临床条件互补的病理—组学—临床生存预测严格实验方案

> 用途：交给 Codex/实验执行者的正式实验设计与验收清单。日期：2026-10-30。**这是一份实验方案，不是已经执行的实验结果。**
>
> 研究对象：**TCGA COADREAD 512名患者、65例 DSS 事件；原病理＋RNA/CNA＋临床数据，不更换癌种、不更换主要结局。**
>
> 研究的可证伪主张：使用冻结病理视觉语言基础模型 CONCH v1.5/TITAN，通过**具有病理学含义的文本概念**限制病理—分子交互，在临床背景条件下保留额外预后信息，且该优势不能由输入增加、有效参数量、普通 Query、简单加性融合或错误患者配对解释。

## 0. 最重要的科学与执行约束

1. 先审计，后训练；**绝不修改、覆盖或追溯性挑选历史 R0–R8 / Phase6/11/16/19 的结果。** 新分支建议 `research/semantic-complementarity-b0b7`，报告 `reports/semantic_b0b7/`。先保存源码、配置、数据 SHA、可计算成本上限、消融定义与评分器合同（`LOCK.json`）。
2. 不要自行生成“更好”的五折。沿用 canonical 512患者5外折；核对每位患者的 fold、DSS time/event、CONCH/TITAN、RNA/CNA、临床完全一致。原代码 `M2R2` 的跨外折 OOF 拼接 Harrell **不可当本轮主指标**；本轮对每个外折独立算患者对，再按可比较患者对数汇总；统一 60个月 Uno60 与训练折删失分布。若源实物不在当前路径，先调查已有 E盘/服务器资产，不得伪造。
3. 已经反复访问本 TCGA 的五折结果。**任何新五折测试只能称既有队列的进一步内部开发/复核，不得冒称未见独立测试或跨研究外部泛化。** 正式运行之前锁定假设，正式运行之后不根据结果变更模型/特征/参数。新独立确认须另外获得真正未用于研究设计的数据，非本轮硬性前置。
4. **绝不以 WSI Patch 数量充当独立样本数**；患者为拆分与评估单位，多切片同患者永远在同折。数据仍是三种真实模态；文本概念来自冻结病理基础模型，属于外部先验，不宣称新增第四种病人测量模态。
5. B0强锚点用独立预训练特征/已有风险拟合，B1–B7 均获取**同一**可用输入特征（包括 CONCH、TITAN native、TITAN text-aligned及同一临床、RNA/CNA），仅改变交互处理。应同时报告“相同输入的结构比较”和“历史模型真实配方比较”，不得在一张无说明表中混排。
6. 不得把文本相似度/Attention位置解释为已证实的细胞类型、因果路径或局部真实基因表达；Bulk RNA 没有 Spot 位置真值。
7. 禁止把外层测试集用于：语义选词、通路映射、学习率/轮数选择、预训练、缺失规则筛样、显著性挑选；内层验证仅按预锁规则选超参。
8. 不能把 `PASS` 测试、梯度存在、checkpoint 重放、训练 loss 降低等同于方法有预后价值。报告必须区分 `IMPLEMENTED`、`TRAINED`、`EVALUATED`、`MECHANISM_SUPPORTED` 与 `NOT_SUPPORTED`。

## 1. 已有资产与历史负结果（必须从仓库逐项确认）

### 1.1 必须复用的资产

- R4 强 CP+RNA Scalar（原五折报告 DSS Harrell 0.832667 / Uno60 0.812398；历史评估，不是本轮保底成绩）；CP 的输入包括六项临床、冻结 TITAN 相关 S32、CONCH 局部/区域 PCA 表示；RNA512 是训练侧选出的风险监督分支。**不能将该结果与历史 M2R2 MCAT 0.7473 直接作架构胜负**。
- 512名患者、65个DSS事件、5个canonical fold、519张可用 CONCH WSI feature H5、完整病人 T/S32 / RNA / CNA 临床。CONCH 768维，TITAN native与经官方 projection 的 text-aligned 768维；每患者可能有多张切片，原 DX1 缺失不可替换为另一患者。
- Phase6A 原冻结32个病理形态概念及两种模板，零DSS选词；32概念的患者级有效秩约4.49，强相关，不能将32个描述误称32条独立生物证据。TITAN 原生 slide output 不是 text-aligned，按官方 vision_encoder.proj + L2 的已审计流程使用。
- Phase6B 已尝试 output-space RNA-conditioned concepts 和真正 token-level RNA-conditioned prompt，在 TNM/Stage 代理目标没有稳定优于 static/additive；Phase11 语义锚定融合结论 INCONCLUSIVE；Phase19 临床三线性注意力的既定机制结论为不支持。**新模型不得仅是旧 Prompt 加 Gate / Cross-Attention 的换名复刻。**
- 历史 R7/R8 提示：交互数值激活不等于 DSS 泛化；错误配对要实际重训；目标域交互风险增量可能很弱。训练错误与机制负结果都要保留。

### 1.2 固定数据与时序审计

使用训练时合法的临床字段 AGE、SEX、AJCC_STAGE、PATH_T/N/M（以现有代码字段名和时间资格核验为准），不得追加结局之后才能得到的变量。特别报告 COAD/READ 子队列、Stage 分布、DSS censor/时间、65事件在5折/阶段内的分布；病人和切片是否重复；是否有中心、测序批次、病理编码器预训练与 TCGA 的重叠风险。

RNA canonical 20511基因，历史有 **203名患者共享约3015个基因缺失**；CNA 25101个基因近完整。必须保留原始 measured mask，禁止将缺测视为真实0表达。RNA 此 canonical 已经 log2(x+1)，禁止再次对数变换。只在当前训练折拟合填补、尺度及 RNA512 筛选；预定义通路 token 用真实可测基因，记录每患者每通路覆盖率和缺失规则。RNA 病人级“缺失特定基因块”不等于全模态缺失。

CONCH 原始patch几何是其自身1024 level0 pixels网格；UNI2-h 有不同512像素网格。禁止基于 H5 原始行号跨编码器对应。原论文主训练无需新增 UNI2 特征；若需要则单列非主实验。主输入的病理固定采用已有 outcome-blind REGIONS512 或经审计的固定512 patch采样（两个版本不能混用）：**优先沿用 R4 REGIONS512**，每位患者预算512，连通窗口最多64；多切片内先汇聚再患者均值。TITAN 全WSI embedding来自原完整CONCH patch集合，不能假称来自同一512采样子集。

## 2. 建议的模型结构：只保留两种有意义的创新算子

### 2.1 输入及严格对齐

对患者 i：

- 病理 Patch tokens `P_i=[p_i1,...,p_iJ]`：冻结 CONCH 768维；J按共享固定512预算以及有效 mask。TITAN native WSI embedding `s_i^native` 和合法 text-aligned WSI embedding `s_i^text`（768）；静态32个 concept scores `q_i^TITAN` 可作为所有模型共享的输入，但概念身份作用只在 B4/B5/B6/B7 使用。
- 组学 tokens `G_i=[g_i1,...,g_iK]`：预先冻结定义的 Hallmark 通路+有限候选增量程序，RNA/CNA 按训练折归一化与 mask 聚合；重要：**不要求32个病理概念与50通路一一对应**；先创建文献/生物学审计的“已支持/未知”掩码，没证据的不强行连接。主方案 K=50 Hallmark，其他 program 列表只作预锁敏感性；可使用完整 RNA512 形成共享 direct risk，不用全量高维输入做 65事件下的无约束风险拟合。
- 临床 `C_i`：六临床字段，训练侧编码/填补。Stage与T/N/M的高度相关性必须报告，做 Stage-only 与 drop-Stage 临床敏感性，但不根据分数选主方案。
- 病理文本概念：**只用已有 Phase6A 的32概念+两模板**；分别调用兼容的 CONCH text encoder 与 TITAN text encoder，不能把不同编码器的768向量未经对齐互算cosine。文本向量、模型版本与概念文本 hash 锁定。在 CONCH 语义空间计算 patch–phrase cosine，TITAN 在官方 projection 后计算 slide–phrase cosine。导入阶段对几十个真实 slides 逐一核验官方 zero-shot 输出等价。

共享参数规模建议视觉/组学投影 `d=64`，8个有效“概念组”读取头（32原始描述保留，通过事先定义的聚类/组别汇聚，组别只能由冻结文本空间决定，不能用DSS结果），attention head=4，dropout候选统一0.1。必须输出每臂**总参数、预测时激活参数、预训练参数、DSS拟合参数**，以免用未激活参数虚假匹配容量。文本查询若来自冻结 embedding，仍可通过相同的 `W_q` 投影，和无语义查询保持参数预算一致。

### 2.2 核心算子1：语义约束的概念特异局部读取

1. 对语义组 k，使用 CONCH 文本锚点 `t_k^P` 对图像 Patch 打分 `a_ijk=cos(p_ij,t_k^P)`；在同一512 Patch预算、同一mask上，抽取概念特异top-k或soft-attention窗口，**选择规则固定且不使用 DSS**。为避免 top-k 单一尖峰敏感，预先锁一个 softmax 主实现和一个 top-k 敏感性实现；统一有效样本数与温度，不依据outer结果选。
2. 用 TITAN 文字对齐的slide表示和相应文本原型构建全局语义上下文 `s_ik`；局部语义 `v_ik = Pool_j(a_ijk,Wv p_ij)`，融合全局语义时对所有臂共享 `s_i` 输入与等容量分支。
3. 和分子token `g_il` 交互时，使用已审计的稀疏语义关系先验 `A_kl`；对于缺乏证据的关系设置可学习低权重而不是零死门。必要消融用打乱映射、随机/学习Query、无文本直接Cross-Attention；避免只给B5高质量概念而不给B2相同的视觉底座。

### 2.3 核心算子2：临床条件残差互补

`c_i=E_C(C_i)`，利用**训练侧、无DSS**的临床回归器估计在此临床背景下的平均视觉概念与分子程序：

```
mu_P(c_i) = f_P^C(c_i),     mu_G(c_i)=f_G^C(c_i)
v_tilde(i,k) = v(i,k)-mu_P(c_i,k)
g_tilde(i,l) = g(i,l)-mu_G(c_i,l)
interaction_i = sum_(k,l) A_kl <U_k v_tilde(i,k), V_l g_tilde(i,l)>
```

- 训练患者残差必须来自训练侧 **K-fold out-of-fold 背景回归**，以免自身拟合造成残差过分乐观；当前外层验证/测试使用当前外层训练患者拟合的完整背景回归器。所有 B4/B5/B7 接受相同临床变量与合理匹配参数；B4 不允许因为“没有临床模型”而少用临床特征。
- 条件均值预测不能见生存标签，也不能用 outer-test 数据拟合。建议 Ridge/小MLP两个版本，**主版本预锁Ridge**，另一版本固定为训练侧敏感性，不要通过DSS挑选。
- `A_kl` 和 `U,V` 控制有效自由度（rank建议 2/4 两个内层候选，锁选择规则），禁止隐式生成50×32完全自由参数矩阵；记录每层L2和正则。
- 临床条件化可以只作为患者背景分解而不是把临床文本再送入TITAN；此前临床Prompt调整没有明确优势，暂不引入动态token prompt。

### 2.4 无DSS表示学习（必做，避免以65事件训练全部模块）

所有可以预训练的臂都在其**外层训练患者**内部执行无生存监督，使用相同训练患者/epoch数据与优化预算。不得读 DSS/OS/PFI；不能因某种臂的验证生存分数低就更改其表征预训练。

两个预注册任务：

- `L_prog`：**仅从病理+临床**预测当前患者的训练侧已测分子通路；不得将被预测的同患者真实组学从 gene branch 直接送入这一预测头。只在被测基因支持充足的通路/患者上计算 masked Huber / MSE，附报告通路测量覆盖分层。
- `L_pair`：患者级(image, omics)正确配对检索，InfoNCE 双向，但负样本优先从同 Stage/癌种/相似RNA缺失模式选择以减少简单批次识别捷径；禁用同患者不同切片作为负样本。必须报告普通随机负样本与匹配困难负样本分别的结果。对小批量无合适负样本的patient anchor记录覆盖率，不能造假负样本。

主目标 `L_aux = L_prog + lambda_pair L_pair`；推荐 `lambda_pair∈{0,0.1,1}` 作为内层无DSS任务决定的候选，不能使用 outer DSS 选。若比较“语义先验的必要性”，B3/B4 必须共享**相同类型与权重**的无标签监督、相同图像/组学/临床输入、相同训练步数与候选负样本。B6只破坏辅助**正患者配对**，真实图像回归、测量mask、组织临床条件和DSS身份不变；这一对照只检验配对辅助监督，不冒称检验一切生存交互。重要：如果 `L_pair` 和 `L_prog` 与一个臂结构不兼容，提供匹配的小头并如实披露；不允许只在 B5 给最强辅助学习。

## 3. B0–B7 严格定义（主矩阵与关键子控制）

模型的**推理输入预算统一**：`CONCH512 + TITAN_native + TITAN_projected + 32 frozen TITAN concept scores + same RNA/CNA tokens + clinical6 + same fixed r0`。区别仅在跨模态算子。由于“文本原型”和“概念得分”本身携带语义，**B3无语义对照需区分“输入中的概念分数仍可用”与“查询没有医学文本身份”**，这可使B3成为保守强对照；另做概念分数冻结置乱作为敏感性。

- **B0: 固定强基线。** 从当前外层训练折重建 R4 `CP_SCALAR`，接受原已审计的 `Clinical + TITAN S32 + CONCH local PCA32 + RNA512`（不得原样读存好的跨不同协议0.8327风险当新基准）；重新核验risk的完整外折路径。额外维护 `B0_UNIFIED` 同一外层训练行利用统一输入的低容量正则化加性风险，作为输入公平的诊断锚点，不冒充R4方法。
- **B1: 同输入加性融合。** 各模态用共享冻结底座与等预算单模态读出 `r = r0 + uP(P) + uG(G) + uC(C)`；绝不包含 P×G 项或依赖另一模态选择patch的 Attention。允许训练侧无DSS自监督表示学习，但与B2–B7的预训练数据预算一致；记录非线性与线性加性两档，主模型选一档在内层锁定。
- **B2: 常规双模态Cross-Attention。** 输入相同、投影相同，P↔G交互，Clinical作为共享附加风险主效应；无文本语义Query约束，保持同一预训练任务、d/heads/深度与风险头。其参数可能比概念模型大，须给两个容量匹配版本（实际激活参数与FLOPs接近）及其原生强版本，公平问题与真实强基线竞争分开。
- **B3: 无语义概念Query。** 与B4完全相同的8 slots、投影、Attention、预训练、DSS风险头；将真实冻结文本原型替换为**固定随机正交、与文本原型匹配范数和谱分布的伪原型**，并且 B3 与 B4 同样拥有一个可训练的query残差 `ΔQ` 和同样的投影层 `Wq`，例如 `Q_B3=Wq(random_fixed)+ΔQ`、`Q_B4=Wq(text_fixed)+ΔQ`。固定随机基底+可训练残差才能同时比较语义初始化与同等学习能力，避免B3可学习/B4冻结的容量混杂；另附纯随机自由query作为非主诊断。
- **B4: 真实文本概念Query。** B3相同有效可训练结构、投影和残差query参数，唯一区别是Query原型基底使用固定的CONCH/TITAN文本原型；形态—通路关系 A 在B3与B4应保持相同以孤立文本Query价值，另用 B4_shuffled_concept_program_map 专门测试A本身。**不要同时引入临床残差化**；Clinical仍作为相同的主效应输入。对文本语义敏感性设置 `B4_shuffled_concept_program_map`（真实query、打乱对应边并维持相同边数/度分布、相同有效输入）和 `B4_textless_same_geometry`；不同组的词文本若确属同义需要先合并。
- **B5: 完整主模型。** 在B4的文本证据读取之上，加入上述 `v_tilde/g_tilde` 临床背景残差与稀疏低秩交互。风险 `r = r0 + add_P + add_G + add_C + beta^T h`；保持同样已学临床表示、同样输入/预训练/正则。**必须有 B5_CLINICAL_CAPACITY_CONTROL**：用可训练但单模态的临床FiLM/残差加性模块补足B4→B5的有效参数量，排除多参数导致的提升。再做“Stage-only背景”和“去Stage背景”的预锁敏感性。
- **B6: B5错误患者配对**（两层定义，不得混用）：`B6a_WRONG_AUX` 与B5完全同模型、本人真实WSI/RNA/Clinical和DSS，只有训练**辅助对比正配对**按训练折内 matched-stage/missing-pattern/COADREAD优先规则置乱；禁止近亲切片、自身配对、测试输入参与；如果L_prog也用患者表达预测目标，须保持本人真实目标以隔离配对效应。主结论限于“正确辅助配对是否必要”。**附加独立子控制** `B6b_WRONG_INTERACTION_RETRAIN`：训练和评估交互分支都通过预锁同分层错患者基因输入，本人加性组学风险和临床不改；B6b的模型训练要真实重做，而不是仅推理时错换；作为非主诊断，评价时的错配是人为干预，不能解释成真实模型可部署性能。
- **B7: B5容量匹配无交互。** 保留相同的语义读取、无DSS配对预训练、临床残差化和真实概念边；将跨模态乘积 `U(v_tilde)*V(g_tilde)` 换成**等有效参数量的单模态基函数之和**（如 `fP(v_tilde)+fG(g_tilde)` 或独立平方项；选用事前固定的同参数替代）。不能使用常数、unused weights、detach、后验interaction-off冒充真实消融。

**比较关系与结论资格：**

| 科学问题 | 必须比较 | 必需排除的解释 |
|---|---|---|
| B5是否改善真实预后 | B5−B0、B5−B1 | 是否只是用了额外输入或从强风险锚点获益 |
| 普通Cross-Attention是否已足够 | B5−B2 | 是否普通大容量交互本就有效；若容量不同须同时看匹配版 |
| 文本身份有无作用 | **B4−B3**；B4−B4-shuffled | 查询参数量、随机初值、关系数量/输入支持集 |
| 临床条件互补有无作用 | **B5−B4**；B5−B5-clinical-capacity | 额外临床参数与stage重复风险 |
| 真患者配对辅助是否必要 | **B5−B6a**；B5−B6b次级 | batch负例过易、错误配对只出现在测试期 |
| 非加性交互是否必要 | **B5−B7** | 完整主效应/容量/辅助训练差异 |

“B5优于B0”是预测能力证据；只有这些**机制配对比较也得到支持**，才可将方法命名为语义条件互补模型。

## 4. 生存训练：避免 65 DSS 事件下错误否定模型

### 4.1 主训练制度：无DSS预训练 → 冻结表示 → 少量DSS拟合

1. 对每一外层训练折，在其内部以患者为单位拟合共享的预处理和 Phase A 无DSS辅助表示（禁止看到当前外层测试图像、RNA等，除非某个特征来自已冻结公开基础模型，且不作队列内再训练）。5外折须各有自己的source checkpoint；绝不可把512人共同预训练后的下游成绩称为严格归纳式五折。
2. Phase A 训练所有可训练 projection、query / cross-reading模块；统计实验中每个架构的 `L_prog` 和 `L_pair` 曲线、原始输入与错配的检索差异，以及被冻结网络的真实性能。**禁止用 DSS 选无DSS预训练的checkpoint**；主选按内层的预注册 `L_aux`/预测 + matched gap 组合规则，或统一固定步数，所有臂一视同仁。
3. Phase B 冻结 Phase A 表示及基础模型，DSS风险阶段仅拟合**共享低容量读出**：固定强 `r0`，单模态加性与交互都用统一低维8/16维风险输入、一样的中心化/标准化/正则化 Cox。原则上仅少量系数估计；主正则采用统一 Efron Cox+ridge，`lambda` 在当折训练侧内部选。禁止让 B5 端到端更新几万参数却让 B1/B3 只学习线性风险头。
4. Phase C 统一做**部分解冻敏感性**，而非主结果：冻结 CONCH/TITAN，允许各臂相同数量的视觉/组学 projection 或LoRA参数通过DSS更新，采用早停与临床主效应保持正则。若只有某臂额外放开参数则不公平；报告 train-valid gap，不靠增大轮次追外层测试。
5. 所有模型的 `r0` 都来自**同一折同一教师**；训练患者使用合法crossfit风险，验证/测试使用当前fit-train重拟合风险，审计风险scale对齐。不能让 B0 的teacher有更丰富输入或给 B5另用测试均值校准。强锚点可单独作为offset，也可作为附加对照；风险增量加权/clip须对所有适用臂统一。

### 4.2 防止“少训练”“初始化差”误判

- 固定三个初始化 seeds `31,47,83`；seed不是独立队列。每臂按相同seed和相同outer fold比较。相同层使用相同初始化，**随机概念与真概念的Query尺度、可训练性与梯度都必须逐张量审计**。
- 推荐统一的**充分性审查预算**而不是预断定80轮够：无DSS Phase A checkpoints=50/100/200/400 epoch（或按真实优化步数折算，必须事前固定）；DSS readout对凸求解使用收敛准则，不用epoch；部分解冻 Phase C checkpoints=40/80/160/320。具体epoch只是假设性上限，须通过真实profiling确定开销。若400边界仍改善，要对**所有同组臂**预注册再加一档，不能只延长B5；所有最终选择严格在当前fold inner-train/val进行。比较要展示完整曲线而非只报告最终epoch。
- 学习率候选 `1e-4,3e-4`，辅助 `lambda_pair∈{0,0.1,1}`，低秩 `rank∈{2,4}`，主风险 `ridge lambda∈{0.1,1,10}`；避免全因子笛卡尔积爆炸：先无DSS训练侧选择辅助损失与表示预算，再固定其结果，生存超参在合法内层验证选择，**不能根据 outer test 组合选择**。所有臂获得相同搜索资源或明确报告预算不等时只做探索性对照。
- 所有训练记录：每epoch训练loss、内部验证loss、Harrell/Uno60（**仅供合法inner验证**）、有效参数数、各模块梯度范数、权重范数、风险偏移SD、注意力熵、错患者配对差值、样本覆盖、OOM/重试。检查无梯度、mask不生效、norm错配、tanh饱和、表示坍缩、统一的输入张量与随机数源。
- 初始化/数据/负采样/患者排序 hash 锁定；B5 真模型与 B3/B4/B6/B7 交互部件必须进行可复现的关闭/置乱前向与实际重训，**不能拿只做推理干预当重训消融**。

## 5. 三层实验矩阵与计算预算

### P0：零训练的资格与公平性审计（先完成）

**验收必要项：**

- `512` 人的patientID顺序、canonical fold完全对齐；`65` DSS事件、时间单位月与终点hash一致，5折事件数与历史完全匹配。
- CONCH 519张合法 H5 的患者归属、原 patch位置、slide数量、异常缺失；TITAN官方文本投影；32概念模板与词表；RNA/CNA measured mask；Clinical原始变量时间资格；`r0` train-crossfit/test-fit相对标准化。
- 修复统一 survival scorer：Efron ties、Harrell正向风险顺序、同时间事件/删失可比性、Uno60 truncation和 fit-train censor KM，**所有B0–B7调用同一份函数**。使用独立 numpy/参考实现做包含同时间事件与删失的测试。保存外折患者对分母，禁止合并跨折风险算Harrell。
- 只读载入历史 M2R2 fold predictions（若存在），按新scorer重算作为旧baseline协议桥接；此处不得声称重新训练或严格同输入对比。历史 MCAT/MGCT/TransSurv 等适配模型状态需标注：原生/适配/是否含临床/TITAN文本，不要统一称原论文完全复现。
- 对 B2/B3/B4/B5/B6/B7 的真实和合成 batch 运行 `forward→backward`，有效参数不为0，病理/组学/临床梯度有路，禁用 RNA 进入image-only program prediction，错误配对无固定点。冻结文本concept在B4/B5中是否实际改变查询，验证一次真实扰动。
- 输出 `P0_CONTRACT.md/JSON`，所有 hard gates 未PASS不得正式跑。

### P1：**零DSS语义对应先验资格**（强制，不作为改变模型选词的依据）

在内层训练患者用原生真实分子测量做：

1. Clinical-only → 分子通路；Pathology-only → 分子通路；Pathology+Clinical → 分子通路；分别与训练侧患者中位表达对照；度量 donor/patient等权通路 Pearson、Spearman、masked Huber与按stage/测量缺失分层的paired retrieval。
2. 真文本概念 vs 相同归一化的随机/无语义Query；正确患者 vs stage/缺失模式匹配的错误患者；正确概念–程序关系 vs 度/测量支持集匹配乱序。重点检验**临床条件下额外可预测的分子信息**，不是全局组织组成或分期粗信号。
3. 不得用当前验证患者的真实组学参与其病理→程序推理。正负患者只由合法训练侧产生；验证可以有真实RNA用于评价paired retrieval，但不能进入image-only预测。
4. 报告不同 P1 任务对哪种生物程序有支持、在哪些临床层失效；不看DSS、不以P1中显著基因挑新的32概念、不允许根据外层患者结果重定义术语。

**P1不能强制证明B5会提高DSS。** 即使P1成功，仍必须跑P2；如果P1完全无真/错配差别，则可以继续执行预锁的最小P2以判定DSS，但需降低语义机制成立的预期，不能通过新概念词表继续无限搜索。

### P2：内层选择＋正式五外折

- 每个原5外折的out-train内部做3-fold事件分层交叉验证，**每次预训练/填补/文本query的可训练投影/clinical残差只使用对应inner-train患者**，当前inner-val只用于评分与early stopping。外折label保留到本折全部模型与规则锁定之后才打开。训练时间单位为patient epoch/optimizer step，对所有同类模型一致。
- 预先锁定超参选择规则：先不使用DSS的内层 `L_aux`/matched retrieval 主规则选representation checkpoint；再以该内层**Harrell + Uno60联合排序**选择少数Cox head正则/风险幅度，tie按模型更简单；不得单靠某一结果强行忽略另一项。若两个指标冲突，完整报告，不私下更换主目标。
- 每臂3 seeds×5 outer folds = 15份最终神经/有监督结果（B0可为5份确定性引用并被三seed复用，不计为15次优化）；B6a 需真正重训15个seed×fold单元。B6b与B5临床等容量、B4映射乱序等属于**额外真实拟合**，不得并到“8臂总预算”里声称免费。
- 如果执行3内层分区×3seed×5外折×7个非B0主臂，最多315个**主臂内层拟合单元**（不含每臂额外超参/无DSS预训练和诊断控制）＋105个最终主臂拟合单元；B0仅参考。具体硬件耗时必须用 1 fold/1 arm profiling 实测并记录。通过可复用冻结特征、warm start、源模型同seed共享（须报告哪些共享）节省计算，不用复用跨外层测试来源的已训练权重。
- 检查是否至少两个seed与独立fold都达到稳定数值；如预训练明显没到预算端点且loss继续下降，允许**事前写明的同组公平续训**。如发生OOM/NaN，保留失败记录、修复代码、重新锁定版本重跑整个受影响比较组；不得只保留成功fold。

### P3：实验报告与统计验收

主指标：**DSS fold-local Harrell C + Uno60（τ=60个月）**；计算方式各折单独分子分母汇总每seed，再三seed等权平均。明确区分5fold nested CV、15 seed-fold cells、512名患者、65事件；**不能当成15个独立验证样本**。

次指标：按训练侧合法删失模型得到的 IBS/60个月 time-dependent AUC（数据和删失支持允许时）、NLL、calibration slope、risk score SD、train-valid gap、PFI冻结DSS风险只作为预锁次终点、stage-stratified discordance仅在各层事件足够时。不可事后因为DSS效果差切换主结局。

为每个主要比较（尤其 `B4-B3`,`B5-B4`,`B5-B6a`,`B5-B7`,`B5-B2`,`B5-B0`）报告：

- 两指标配对差值；各seed方向、各fold方向、最差fold、最差亚组；按患者为单位且保持fold标签的 **paired bootstrap 2000/5000次** 描述区间（需明确不重新拟合模型、不反映模型选择不确定性），及考虑预设多重比较的调整区间/明确探索性标签。
- 共同比较必须尽量做到正方向和幅度具有实际意义；**不要把0.001的偶然波动写成显著算法贡献**。可预设解释门槛如双指标平均Δ至少+0.005、≥2/3 seeds正向且至少4/5fold正向，仅作工程决策规则，并同时报告原始差值和不确定性；该门槛不代表统计显著性。
- 如果 B5 比 B0 稍优但对 B3/B6/B7 不优，则结论为“强视觉底座+额外容量/一般交互”，不是文本语义条件互补；如果仅P1预测有效但DSS无增量，则说明跨模态对应≠预后互补。
- 如果 B5 在本既有开发队列满足所有门槛，仍需将论文结论限定为**同一研究队列的内部证据**；外部医学队列或独立确认未来再做，不可称首次未见确认。

## 6. 防止漏判的四项专项对照

1. **临床分期主导审计。** 对当前强风险 `r0` 与 B5 加各自训练-only的 `Clinical-only`,`Stage-only`,`Clinical-no-Stage`,`Clinical+CONCH`,`Clinical+TITAN`,`Clinical+CONCH+TITAN`,`Clinical+RNA` 按相同评分器报告；这是解释高0.83的来源，不可把相关的多个临床字段当独立强监督信号。
2. **缺失机制。** 按203人的RNA特定缺失块模式分组；显式含mask与不含mask、保留RNA512覆盖率、相同组学有效支持集的真实/置乱配对。测试人为整模态缺失只能作为额外稳健性实验，不改变主完整三模态任务，也不能把基因块缺失称“整RNA模态缺失”。
3. **编码器/patch选择敏感性。** 主架构固定CONCH+TITAN；`CONCH only`,`TITAN only`,`CONCH+TITAN`和REGIONS128/512/1024作预设单独诊断，不能因某一个版本提高DSS而更换主模型或把新结果当盲确认。概念可见性仅由真实WSI/ROI人工标签验证；若无人工标签只能称语义相似度解释。
4. **原生强方法对比。** MCAT/CAMR/MGCT/MOTCat 等旧代码已经完成适配复现。先重新评分旧保存风险；如风险已不存在、训练对象与本输入差别较大，**只选择 MCAT+1种最相关强方法**在统一数据/目标/训练预算下正式重训（必须清楚适配状态），并保留原生协议报告单独说明；不把所有已完成baseline视为未做。

## 7. 防止错误结论的具体软件验收测试

必需自动化测试用例：

- DSS不进入无DSS pretrain的任何数据结构、loss、模型选词；输入替换为随机DSS不改变所有pretrained hashes。
- B3/B4同层参数个数一致、被使用的参数一致、grad非零，CONCH/TITAN query都在各自匹配的text alignment space；B4更换concept文本会确实改变合法query，但不改变非其负责的输入。
- B6a错配只作用于辅助正对；本人主效应、真实gene值及mask、本人DSS label完全不变；错配无固定点，优先临床分层/缺失模式相近并在合同中登记fallback；所有负对照改变的是实际有效训练目标。
- B7与B5有效训练参数数和主效应、无DSS预训练及临床残差相同或在1%误差内；B7不含跨模态乘积，且对其中模态swap表现出与B5不同、符合消融定义的变化；记录 FLOPs和更新数。
- B0/B1/B2/B5同一外层测试人群、时间、事件、风险排序方向；测试评估不读任何训练内部checkpoint-selection记录；单折不允许拼接跨折risk做Harrell。
- ties：构造2/3例相同事件时间、同时间删失者、不同生存时间的合成单元，比较独立朴素Efron、Breslow及scikit-survival参考（如安装）；单点梯度与数值差分匹配。截断60、IPCW KM和风险分数正负方向同步测试。
- 病理某患者多WSI的所有幻灯片始终同fold；CONCH/TITAN来源排序一致，缺失DX1保留真实缺失不能补假造图像。
- 所有预处理拟合与背景残差模型仅 current train rows：在测试标签/测试基因值随机打乱时，已拟合模型、train统计、train风险与预训练权重无变化（测试输入预测应随合法测试模态改变）。
- 五fold × seed预测重放/哈希、每arm回放指标精确可复现、输出风险finite、没有注入人工测试结果。

## 8. 交付目录、门禁与下一轮决策

```
reports/semantic_b0b7/
  00_PROTOCOL_LOCK.md / LOCK.json
  01_DATA_ENDPOINT_FOLD_AUDIT.md / json
  02_TEXT_ALIGNMENT_AND_CONCEPT_AUDIT.md / json
  03_FAIRNESS_CAPACITY_AND_GRADIENT_AUDIT.md / json
  04_PRETEXT_PAIRED_RETENTION_RESULTS.md / csv
  05_INNER_SELECTION_SUMMARY.md / csv
  06_FOLD_SEED_TEST_METRICS.csv
  07_PAIRED_MECHANISM_COMPARISONS.csv
  08_RISK_REPAIR_DAMAGE_AND_CALIBRATION.md
  09_COMPLETION_AND_REPLAY_AUDIT.json
  10_DECISION.md
  11_LIMITATIONS_AND_NEGATIVE_RESULTS.md
  12_REPRODUCE.md
```

**建议逐门执行，停止不等于失败：**

- G0 数据/公式/对齐/有效参数/概念词表/版本核验全通过后，才能执行内层训练。
- G1 在不使用DSS下明确真实患者配对或概念结构是否提供超越临床与无语义Query的信号；若阴性，继续预锁最小DSS测试，但不得声称“语义对应已被证明”。
- G2 同协议内层：模型确实训练充分（验证不持续被最大预算限制）、负样本可用、正常梯度、风险非坍缩、B5相对B3/B4/B7的改善方向有资格进入主实验；若所有失败原因都是实现/预算问题，先公平修复并重跑必要的同组，不可用Outer test挑。
- G3 五外折结果已经冻结后做预先锁的完整比较，结论严格区分 `MECHANISM_SUPPORTED` / `PREDICTION_GAIN_ONLY` / `SEMANTIC_SIGNAL_ONLY` / `INCONCLUSIVE` / `NO_GAIN_IN_THIS_IMPLEMENTATION`。

## 9. 优缺点评估（事先审计）

**优势：** 三模态、同癌种与同DSS任务不变；共享强基础模型并对控制臂公平开放；把文本语义、临床条件与非加性交互分开检验；无DSS先学习对应、低自由度生存风险头缓解事件少的估计问题；保留历史强基线与复现实物；避免跨fold分数尺度伪差、模型规模混杂和冻结错配伪消融。

**主要风险：** 512人的无监督paired预训练仍有小样本局限；32概念高冗余、文本语义未取得病理专家定位真值；Bulk RNA没有spot级真值，跨模态相关性可能是Stage/肿瘤纯度/批次共同原因；不同arm即使参数量相等也不代表函数先验相同；强临床Stage/TNM加性风险可能吞噬可学习的DSS交互信号；五外折已历史开发暴露，不能成为新独立外部证据；超参、B6子控与预训练计算成本高，必须先真实profile和逐组预算。

## 10. 对 Codex 的最终执行指令

1. **第一轮只执行 P0 + P1 + 计算开销的极小真实profile，不立即启动全部105+拟合。** 给出 audit 和运行命令、所有已锁 SHA、真实内层训练资格、预计成本范围（依据实际profiling、不可猜测）以及明确的下一轮矩阵。
2. P0必须审计之前 Phase6A/6B/11/16/19 与 R0–R8 源码和结果；指出新B5与旧语义/临床交互实现**逐算子的差别**，避免仅改名重复。
3. G0 PASS 后进入 P2，依上述公平预算分阶段执行。训练必须产生真实模型/有效性能/成本/历史失败谱系，不可输出占位 `PASS` 或拟造metrics。
4. 代码/模型/患者预测与原始 H5 严格保存在私有执行目录；Git仅推源码、配置、匿名聚合结果、hash和审计报告；所有私有路径/token/患者标识不推送。
5. 每轮结束给一个**科学诊断**：模型是否在源任务学到匹配关系？是否相对B3/B7真实超越？DSS少事件有没有造成训练—验证分离？哪个尚未检验的因素影响当前结论？不能只输出高分榜。
