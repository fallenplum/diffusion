# P2｜B0–B7 统一协议三模态 DSS 验证：Codex 执行 Prompt

## 任务与不可妥协的研究合同

你是医学多模态生存模型的算法与复现负责人。在 `fallenplum/diffusion` 的 `research/semantic-complementarity-b0b7` (原P0/P1基准 `98f42e298975c3`) 和最新合格 P1-R 分支基础上创建 `research/semantic-b0b7-p2-dss`。先读取并绑定父SHA，读取 P1-R 的 `10_P2_FROZEN_CANDIDATES.json`（若不存在，采用原始B4/B5架构，不得自称修复已完成）；读取原 `B0_B7_语义条件互补三模态_严格实验方案_Codex.md` 与 `reports/semantic_b0b7/` 的P0、P1、历史基线桥接、评分器审计、历史算子审计、当前P1负结果。**本任务必须在本次执行过程中真实完成合格内层实验和 B0–B7 DSS 模型比较，不许只产出计划/空报告。** 若真实环境资产不足，保留明确 blocker 和已经实际完成的单元，不得代填结果。

**研究问题只限：TCGA COADREAD 512患者，65 DSS事件，CONCH/TITAN病理＋RNA/CNA组学＋六项临床，生存预测。** 不换癌种，不改主终点，不因负结果改文本词表/关系映射/实验协议。所有 P2 正式架构、预算、训练路径、结局评分和科学主次比较须在读取任何本轮 outer-test 结果前锁定。旧五折已历史反复开发，所有结果均为**内部、开发暴露的跨验证证据**，不能宣称新外部独立确认。

本轮尊重现有 `EVALUATION_POLICY_ADDENDUM_20261031.md`：**不计算bootstrap、置信区间，不用显著性门槛掩盖点估计或跨fold/seed方向**；仍应完整呈现配对差值、每折每seed和局限。禁止以无CI为理由声称具有统计确认或普适泛化。

## A. 入场核验：必须复用、不能混用的既有资产

1. 现有canonical 512名患者、65个DSS事件，固定5外折103/103/102/102/102，事件13/14/12/13/13；对比P0证明患者身份、原 M2R2固定折、DSS时间/event逐人一致。时间单位月，所有多切片同患者同折。分别检查 CONCH 519切片、TITAN native768、TITAN projection/text aligned768、冻结32文本概念两模板、CONCH原H5及官方匹配projection、RNA20511、CNA25101、临床6列、原 measured mask；RNA已log，禁止重复log；CNA gain/loss先分解。
2. `reports/semantic_b0b7/P0_HISTORICAL_BASELINE_BRIDGE.md` 的**95份旧风险只读重评分**已经完成：旧MCAT统一Harrell `0.780522` / Uno60 `0.776645`；旧EarlyFusion `0.706892` / `0.735238`。这些是不同UNI2输入、不同旧目标训练下的历史参照；不得称之为B0–B7同输入公平比较或原论文原生复现。无需重训全部七个旧SOTA。
3. **B0精确R4强锚点**：重建而不是简单加载历史总分。旧GRID-local CONCH特征（局部均值经训练侧PCA32）＋固定TITAN相关S32概念＋六临床，三个训练侧crossfit teacher块、对应训练侧teacher风险标准化、RNA512原**八程序union**成员范围训练侧方差选基因、`CP+rna_beta*RNA_RidgeCox`，同teacher-heldout pairs上fit标量；其R4历史DSS为Harrell `0.832667`、Uno60 `0.812398`，不作为本轮保证值。`REGIONS512` 和新Hallmark50绝不能偷偷替换旧B0的GRID特征/RNA512仍称精确复现。
4. **B1–B7共同输入预算**：相同 CONCH REGIONS512（每患者最多512真实Patch，多片先片内pool后患者等权）、TITAN native＋合法投影/对齐＋32概念分数、RNA/CNA Hallmark50带测量mask、Clinical6，以及**本折同一强r0训练侧合法crossfit风险/测试侧full-fit风险**。输入列表逐张量对比 SHA。不同架构允许不使用某种信息，但要有匹配容量/明示。`B3` 虽旋转文本Query仍使用真实 `titan_scores`，因此定义为“保守随机Query对照”，另用P1-R合格 `TEXTLESS` 检验完全无显式文本条件。
5. **生存评分**仅调用 `src/semantic_b0b7/survival.py`（如P1R有经验证兼容版须独立映射审计）：完整训练风险集Efron ties（包含事件时点并列删失）、风险高=更早DSS事件；Harrell同时间事件–删失可比较、风险 ties 半分；Uno60当前训练分区reverse-KM与 `t<60`、不可外推缺乏删失支持。每个测试外折单独计算可比较患者对分子分母；每seed按5折合并分子分母后，再等权平均seed，**禁止跨折risk拼接算主Harrell**。无可比对/无IPCW支持返回NA或失败，不伪造。

创建 `configs/semantic_b0b7/p2_lock.json`，列出所有分支SHA、数据SHA、参数预算、候选预训练修复、每个对照的**真实模型行为而不只是名字**，选轮规则、计算硬上限、完整内外折划分、数据访问白名单和失败恢复策略。在外折测试计算前完成锁定；锁后所有更改须明确记录探索性分支，不能替代正式主结果。

## B. 模型 B0–B7 的科学矩阵（保持同一可训练风险头）

| 模型 | 定义及实现约束 | 核心对照 |
|---|---|---|
| `B0_R4_EXACT` | R4原配方严格重建；不改变其原始特征和标量风险教师；确定性5折重拟合并与历史R4风险/分数核查 | 实际性能锚点，输入不匹配的历史配方单列 |
| `B1_SAME_INPUT_ADDITIVE` | 共用CONCH/TITAN/RNA/CNA/Clinical与同一r0；各模态可单独非线性表示，但最终风险为`r0+fP(P)+fG(G)+fC(C)`，不能有P×G、跨模态查询或RNA选择Patch | 同输入加性优势 |
| `B2_CROSS_ATTENTION` | 标准P↔G Q/K/V交互，Clinical作为主效应；包含原生QKVO强版本和至少一个实际活跃参数/FLOPs匹配版本；不能额外隐藏文本Query引导 | 一般交互模型 |
| `B3_RANDOM_QUERY` | B4完全同层同seed同辅助loss，只把CONCH/TITAN固定Query语义方向做保谱随机旋转，保留所有共同text score输入；单独 `B3_TEXTLESS` 不用显式词语原型/概念分数但保持等容量 | 查询语义是否必要 |
| `B4_TRUE_TEXT` | B3同构，固定医学文本Query，关系来源/损失/风险头完全相同，不做临床残差；可带事前冻结P1R的同组修复，但B3必须同步相同修复 | B4−B3只检验查询身份 |
| `B5_CLINICAL_COMPLEMENTARITY` | B4相同概念证据，在**训练侧无DSS OOF**临床背景均值上做病理与分子残差，再建立预锁低秩交互；不能把测试患者真实RNA送进图像程序预测头 | 完整提案 |
| `B6a_WRONG_AUX_RETRAIN` | 真实重训B5表示，唯一改变训练侧辅助InfoNCE**正对分配**；每个人自身图像、真实组学、程序目标、Clinical、DSS标签、主效应不变，匹配同stage/缺失模式/COADREAD，无self；负例支持与B5同seed相同 | 真实患者辅助配对是否必要 |
| `B7_NO_CROSS_PRODUCT` | B5相同真实文本、无DSS Phase A、临床残差和r0；将跨模态乘积完全替成**等有效容量的单模态非线性/平方项**，不可含交叉分支或隐藏FiLM的交互路径 | 交互是否真的必要 |

**必需的子控制，不可标成免费8臂：**

- `B4_SHUFFLED_RELATION`: B4真实查询但置乱concept–Hallmark映射（尽量保持度/测量覆盖），需要真实重新拟合表示而非仅推理置乱。
- `B5_CLINICAL_CAPACITY_CONTROL`: 与B5相同有效训练参数预算、临床输入、任务和训练；用临床单模态加性特征代替条件残差化，排除增加临床参数产生的伪收益。不得用unused参数凑数。
- `B6b_WRONG_INTERACTION_RETRAIN`：仅在机制对照尚无法分辨时启动，训练交互特征时使用训练侧真实错配，在预锁错配评价条件下评估；本人主效应、标签不变。**测试错配属人为干预，不是自然部署性能**。它不同于B6a，绝不能混名。
- `B3_TEXTLESS`：无显式概念语义，禁止TITAN32 concept scores旁路，保持原始视觉/组学/临床基础模型资源及活跃容量。

**强制容量审计：**共享视觉/组学编码与下游风险头的**有效参数**、预训练活跃参数、DSS训练参数、FLOPs/患者、主输入维度、有效非零梯度。参数总量相等但不执行前向的不算匹配。B2原生大模型作为强竞争基线，即使容量不同也可并列报告，但其结论不得冒充同容量机制比较。各模态 raw 的生物学语义及真实缺失mask不发生不可见的变化。

## C. 第一阶段：完整合法的无DSS Phase A（预训练）

1. 每个 outer-train 内进行其自己的pretraining；嵌套内层3折：inner-train可训练，inner-valid只供合法无DSS指标/生存头验证，不允许提前将inner-valid的RNA用于预训练拟合。在 outer-final refit 时，才允许用当前 outer-train全体患者训练无DSS表示；outer-test患者任何模态都不能进入训练统计/预训练。
2. 优先继承P1-R预先锁定的**一个**主修复合同，原始B4同时保留为诊断；不能分别给B5/B4/B3不同的有利超参。P1R最多两个候选只在**当前outer-train内部**选定，不得用测试指标做选择。若P1R没有合格候选，主结果用原始B4及B5，不造假的修复。
3. `L_prog`：病理+临床→本人的 measured RNA50程序，mask Huber，**真实目标RNA不能流入image-only predictor**；`L_pair`：病理与本人组学双向InfoNCE，负例匹配Stage/癌种/缺失模式，共用seed支持集并报告可用anchor/chance。
4. checkpoint预锁候选 `25/50/100/200/400` 患者epoch（若P1R冻结了不同但同预算合格规则，以其锁为准）；学习率/辅助权重有限网格（如lr=1e-4和3e-4、`lambda_pair∈{0,0.03,0.1}`）由**无DSS训练侧**选择，不能用outer-test表现选；损失权重不同不能直接比较合成loss绝对值，选预注册masked Huber与hard-gap规则。避免过度搜索：先对共有结构/代表臂筛固定配置，再**所有匹配对照在同配置和同预算**评估。
5. 400轮若训练下降但验证上升，如旧P1则保留较早选轮，不把延长轮数视为充分性；如果共同未收敛信号且所有同组有相同预算需求，预锁最多一次等预算扩展。任何一臂失败需整个配对比较组均受相同的修复/重新锁版本处理。保留100%训练/验证曲线。
6. 可缓存无结局冻结feature（CONCH/TITAN既有权重）；**绝不可跨outer fold缓存带训练后adapter、PCA、临床归一化、program标准化或背景残差**。B4/B5/B7合法共享同一 Phase-A checkpoint，但只有当参数、初始化、损失、训练分区、梯度路径确实完全一致才能共享；B6a正对不同必须独立真实训练。
7. 一定报告 P1R 静态无DSS程序Pearson与matched retrieval结果，不把这些指标当生存DSS证据，也不能从多patch信息数虚构独立样本数。

## D. 第二阶段：临床背景与低事件风险头（Phase B）

### D1. 训练侧临床背景回归

对每个当前 `fit_train`，在相同**冻结表示坐标**分别预测视觉slot与分子程序的临床期望值；主背景模型 Ridge(alpha=1)，可预先定义 alpha={0.1,1,10}作为内层敏感性，禁止DSS挑最合适的语义概念。**训练患者必须交叉拟合背景预测，验证/外测使用同一fit_train上full背景模型**。临床编码/填补/缩放需要在背景交叉拟合子训练集合重拟合。仅B5/B6a/B7做残差算子；B4/B3仍有同样临床主效应；等容量临床控制补齐新增自由度。切勿将不同 encoder checkpoint 产生的latent OOF混合成同一潜变量空间。

### D2. 固定统一DSS风险读出合同

全部新 B1–B7使用相同形式：

`r_i = r0_i + wP·zP_i + wG·zG_i + wC·zC_i + b·zMechanism_i`。

- `zP/zG/zC` 均从合法冻结表示经过**同一训练分区、无结局的确定性**压缩与标准化，各建议8维；机制特征主rank2（既有实现），所有B1–B7最终头提供**两个真实活跃的机制/非线性容量对应维度**，不能只给B5两个额外系数而让B1/B3没有同参数功能。可以有预锁 rank4 的统一敏感性，但不得由outer结果挑主模型。
- B1 `zMechanism` 为模态内非线性/平方的两维加性控制；B2真正CrossAttention两维；B3/B4语义/伪语义跨模态两维；B5/B6a临床条件残差化交互两维；B7为等维的独立单模态平方和。不能以2个全零维填充假装等容量。
- 每个 `fit_train` 的 `r0`：训练患者必须由这个fit_train内嵌真实crossfit teacher生成，当前val/test由相同fit_train full teacher拟合生成；不得直接把全outer-train in-sample强风险用于下游训练，而拿out-of-sample风险预测验证。风险标准化沿训练teacher原均值/SD，不用验证/测试均值；每个嵌套inner都必须独立重新fit老师。原始R4同teacher风险标量拟合方法和直接Cox residual offset要标名区分，不能将0.8327旧预测混成新风险头结果。
- 主下游风险模型：**完整当前训练风险集 Efron Cox + ridge**，以 `r0` 为冻结offset，ridge lambda从预锁 `{0.1,1,10}` 内在当前outer-train内部选；记录可训练系数数、优化梯度误差、数值差分核验；不在测试分布上做center/clip/score重标定。
- 把`r0 + zero_delta`作为数学单元门禁；损伤/修复按同患者同折排序对统计，记录风险增量 SD/anchor SD和 train/valid的原始/差值表现。
- **不要为B5开放联合DSS端到端更新而让B1/B2仅用冻结表示**。部分解冻 Phase C 单列敏感性，只有在主矩阵完成且事先锁定统一解冻容量/轮数时才能执行，不能代替主结论。

### D3. 验证选择与信息隔离

- 每外折当前train内部3个event-stratified inner folds；每组31/47/83初始化。每个inner fit要重新拟合PCA、程序标准化、Clinical、r0 teachers、无DSS Phase A、背景模型及风险头，绝不能拿当前inner-valid的RNA、临床、DSS参与其fit。
- 先按无DSS指标选pretraining结构/epoch；后在当前inner-val的DSS Harrell主指标、Uno60次指标及模型简单性按锁定**字典序**选择同一风险头lambda。两个指标相反时如实报告，不可临时改成漂亮的加权平均。若inner fold Uno60缺乏删失支持，按锁定降级规则使用Harrell并标NA。
- 各arm相同超参候选、优化耐心/预算与早停合同，记录每seed选出来的配置。严禁用outer测试分数投票挑模型/窗口/词条。outer final refit读取事先锁定的内层候选，当前outer-train完整重训pretrain/teachers/background/readout后**只执行一次**外层预测。
- `B0_R4_EXACT`确定性每fold一次真实重拟合；如果与旧R4差异异常，先重构fold/teacher/S32/GRID/PCA/RNA成员/同teacher正则和风险评分，不能擅自重调到0.8327。保留P0以前历史风险重评分报告作为只读旧阶段对照。

## E. 执行计划、预算与资格门

### E0｜先小范围真实profile和统计信息屏障

只测一个当前outer-train内真实inner-fit：完整CPU预处理、strong teacher 13 fits、预训练训练/保存state回放、背景OOF、Efron风险fit、逐折评估；记录GPU峰值/吞吐与CPU/IO瓶颈、磁盘缓存成本。对所有模型设置一样的GPU候选预算，B2含 native 和匹配版单列。**不能通过“只训练一个模型”来报告完成整套B0–B7。**

### E1｜内层资格验证

从 outer-fold1 current train 内部先完整三内折，代表臂 `B0/B1/B2_matched/B3/B4/B5/B6a/B7` 和必要 `B5_CLINICAL_CAPACITY_CONTROL`，31/47/83；B0按确定性模型不虚增seed。此步骤是**开发资格**而不是新独立测试。至少核验三个seed的真实 checkpoint forward、合法训练/选择、正则风险的数值收敛和病理/组学可用量。若模型存在 NaN、mask错误、query语义根本没用、明显错误固定输入，修复并公平重跑整个受影响比较组。

### E2｜正式预锁5外折

结构与规则冻结后执行五个外折。每折同一三内折规则，真实拟合 B1–B7×3seeds：**105个正式主臂 outer fits**；B0另5个确定性真实fits；内层主臂最多 `5outer×3inner×3seed×7arm=315` 个相应单元（不含每单元预训练/背景/teachers及超参重复），必须以实际 fit ledger 计算工作量。若`B5_CLINICAL_CAPACITY_CONTROL`、`B4_SHUFFLED_RELATION`或`B3_TEXTLESS`进入主机制结论，至少每fold×seed正式重训，对应算力不能偷算进105。B6a必须是不同辅助正配对的真实重训，不能 inference-only swap。

**允许算力优化：**复用冻结CONCH/TITAN编码与纯原始无标签缓存，复用确实共享的同fold同seed训练源，按hash复用B4/B5/B7 Phase A合格checkpoint；同rank内Cox凸目标用warm start只能在当前train内部，所有使用warm start的臂定义和收敛误差一致；多GPU/并发按真实profile控制CPU/显存，避免IO争抢。不能复制 B0 重训计数、跨外折复用带学习的预处理、或把单次model perturbation叫重训。

### E3｜必要敏感性（不覆盖主矩阵）

预锁且算力允许时：`Stage-only / clinical-no-stage`背景，CONCH-only/TITAN-only、RNA缺失块203人分层，REGIONS128/512/1024、统一rank4、Stage-only Clinical对照、PFI冻结DSS风险的预定义次评价、阶段性部分解冻。所有敏感性不得根据正式测试分数选择为论文主版本。无足够事件的子组指标标NA，不用小分母伪高分。

## F. 必须在真数据上通过的软件与科研验收

1. **无测试/无标签泄漏。** 随机打乱outer-test DSS/测试RNA不能改变train预处理、训练输出或checkpoint hash；只允许改变合法测试forward中依赖RNA的预测。无DSS pretraining调用栈不得访问DSS/OS/PFI。训练阶段Clinical不能包含不合法结局后变量。
2. **B3/B4只差Query身份**（保守对照）；B3_TEXTLESS没有 `titan_scores` 与任何显式文本query旁路；B4_MAP_SHUFFLE保持实际有效关系集合支持或诚实披露不可完全匹配；B5与B7正确共享pretrain，B7无交叉项；B6a错配只作用正对且不改变本人RNA程序/临床/DSS；错误配对需**真正梯度更新**。
3. **参数真实活跃。** 已注册/有梯度/影响预测三者分别记录。`r0` 对所有臂共享同一训练teacher。B2强版可大，但必须给capacity-matched版；各臂Cox读出系数真正非零且可检验。
4. **完整生存风险集。** 以朴素独立Efron、官方 `scikit-survival` 真包API（可装环境则做；若仅有官方数学源码/合成测试需明确标partial）、autograd/数值差分、并列event/censor案例核验；Uno60 fit-train reverse KM、60月支持检查，fold-local micro分子分母、相同患者折绝不跨折排序。
5. **诚实预算。** 每臂每fold每seed实际预训练 optimizer steps、risk fits、teacher fits、checkpoint hash、run time、GPU峰值、失败/修复谱系；整组等预算延长或中止。特征权重预训练有TCGA重叠风险则公开记录，绝不把基础模型pretraining当本轮外部独立证据。
6. **患者级核验。** 全部WSI同患者同fold；mask缺测不等于0；不按CONCH/UNI2原行号错误匹配。患者身份不得推送。保存私有患者预测供下游复核，但Git只允许匿名汇总指标/模型代码/manifest。
7. **指标重放与独立核验。** 每外折逐患者风险本地保存，重新加载得到逐折分子/分母相同；独立审查的来源如果只重算已保存分数，不准写成从原始H5到预测全链路独立复现。

## G. 必须报告的对比、决策及论文边界

每个比较在**同一fold与seed**计算 ΔHarrell、ΔUno60；报告5折×3seed明细、3seed平均、每fold方向、最差fold、修复对/损伤对、风险增量SD、相对于 B0 与 B1 的差异，不计置信区间，也不将15个cells视为15个独立队列：

- `B5−B0`: 实际预测竞争力，需另说明输入配方不等价。
- `B5−B1`: 同输入是否增益；`B5−B2_MATCHED`和`B5−B2_NATIVE`: 普通cross-attention竞争力。
- `B4−B3`, `B4−B3_TEXTLESS`, `B4−B4_MAP_SHUFFLE`: 语义Query、显式文本、医学关系分开判断。
- `B5−B4`, `B5−B5_CLINICAL_CAPACITY`: 临床条件残差是否必要，不混入容量增加解释。
- `B5−B6a`: 真患者辅助配对是否必要；`B5−B7`: 非加性交互是否必要。
- `B1−B0`: 使用同输入的加性基线与旧真实配方比较（不同输入，不能得架构因果）。

明确的科学输出：

- `MECHANISM_SUPPORTED_WITHIN_DEVELOPMENT_COHORT`：B5同输入胜过B1/B2/B3/B7、真实文本Query或关系较伪对照有稳定正向、配对/临床对应分支贡献可观察；同时比B0至少不出现实质性恶化，并且Harrell与Uno60没有选择性隐藏；所有训练充分性与对照资格合格。只写内部研究证据。
- `PREDICTION_GAIN_ONLY`：B5有DSS增量，但不能排除随机Query/加性容量/错配效果，**论文不得称语义互补被证明**。
- `SEMANTIC_PRETEXT_ONLY`：无DSS配对任务存在有益信号，但B5没有DSS增量，需如实停止主机制Claim。
- `NO_CONFIRMED_GAIN_IN_THIS_IMPLEMENTATION`：优化、等容量、实际负对照合格，但点估计与方向不支持；不通过无限调参改写此前五折。
- `INCONCLUSIVE_DUE_TO_IMPLEMENTATION_OR_BUDGET`：数值、泄漏、预处理、训练不足或比较设计仍不合格；不得冒充科学失败。

CI不计算，但**差异小于0.005的正向点估计只能描述为弱方向证据**（并非概率推断或统计阈值）；需展示完整 Δ 和方向。5外折已历史反复开发，不能称独立盲测，不把医学解释可视化视为病理专家真值。P2结束后不得自动启动P3架构搜索或换癌种；先交付完整验收与研究裁决。

## H. 最终交付清单（必须落地）

```
configs/semantic_b0b7/p2_lock.json
src/semantic_b0b7/  # 版本锁定的DS S风险、背景、模型接口实现
scripts/semantic_b0b7_p2_*.py
tests/test_semantic_b0b7_p2_*.py
reports/semantic_b0b7_p2/
  00_PARENT_AND_PROTOCOL_LOCK.md / json
  01_SAME_INPUT_CAPACITY_AND_FEATURE_AUDIT.md / json
  02_EFRON_UNO60_SCORER_AUDIT.md / json
  03_NO_DSS_PRETRAIN_AND_CLINICAL_CROSSFIT_AUDIT.md / json
  04_INNER_SELECTION_LEDGER.csv
  05_DSS_B0_B7_FOLD_SEED_METRICS.csv
  06_PAIRED_CONTRASTS.csv
  07_CAPACITY_TEXT_MAP_PAIRING_CONTROLS.csv
  08_RISK_REPAIR_DAMAGE_AND_CALIBRATION.md
  09_TRAINING_CURVES_AND_COMPUTE.csv
  10_INDEPENDENT_REPLAY_AUDIT.json
  11_DECISION.md / json
  12_LIMITATIONS_NEGATIVE_RESULTS.md
  13_REPRODUCE.md
  PUBLICATION_MANIFEST.json
```

**严禁凭空填表**；若某个比较没有成功的真实模型训练和回放，相关数值设 NA/NOT_RUN 并明确原因及影响，不补旧结果。最后报告可复现的运行命令、source/asset/checkpoint哈希、真实 GPU/CPU 时间、完成/未完成fits和最终科学判断；提交GitHub新分支，给出确切commit SHA及主要报告路径。
