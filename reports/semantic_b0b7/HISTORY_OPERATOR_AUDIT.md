# B0–B7：历史算子与资产谱系审计

日期：2026-10-30。状态：**COMPLETE_STATIC_REVIEW_LIVE_QUALIFICATION_PENDING**。

本审计只读取旧源码、配置和匿名聚合报告；0训练、0模型前向/反向，未载入患者数组或私有权重。历史文件未修改。新B5尚无本轮机制或DSS性能结果，旧门禁通过不能替代新服务器克隆后的实物资格。

## 最先需要纠正的合同细节

1. **精确B0的CONCH输入不是REGIONS512。** R4共同CP使用旧无结局GRID局部均值，再训练侧标准化/PCA32；REGIONS512用于空间网络。新交互可以使用REGIONS512，但不能把其均值静默代替历史B0。证据：`reports/bgb_en/DECISION.md:7`，`src/bgb_en/protocol.py:89–102`。
2. **精确B0的RNA512不是任意新版Hallmark512。** 原direct-risk分支按旧八程序union（977外部成员、955 canonical）做训练侧覆盖/方差选择；新版Hallmark50成员不同。两个外部成员来源须独立锁SHA。证据：`reports/bgb_en/R0_ASSET_AUDIT.md:43`。
3. **B3“行正交”和“匹配真实文本谱”一般不能同时满足。** 推荐将正交解释为特征空间随机正交旋转：`T_random=T_text@Q`，可精确保留行范数、Gram与奇异谱并破坏匹配视觉方向；独立随机正交行另作非主诊断。必须在训练前写清定义，不能用外层结果选择。
4. **旧15条候选先验不是新8×50映射。** 旧Phase7仅覆盖12/32概念，连接旧分子state，性质是候选允许关联；不能凭关键词扩为医学真值矩阵。未知关联须保留未知。

## 逐算子比较

|历史实现|实际算子|新B5的可检验区别|既有结果边界|
|---|---|---|---|
|Phase6A|Frozen TITAN slide encoding, official vision projection, 32 morphology phrases and two templates; static cosine descriptors.|Reuses those frozen phrases; adds CONCH-local phrase read and train-side clinical-background residual interaction. No new concept vocabulary.|519 slides/512 patients; official probability agreement error 1.49e-08; effective rank 4.4883. Concept versus native TITAN proxy superiority not established.|
|Phase6B_output_space|Top512 RNA to 16D adapter, basis/gate produces text-vector delta after text encoding.|Does not adapt the text embedding from RNA. Fixed text selects morphology; RNA remains separate molecular evidence.|No stable superiority over static/additive on T/N/M/Stage; no survival endpoint read.|
|Phase6B_token_v2|RNA-conditioned soft prompt inserted after BOS into the frozen TITAN text Transformer; corrected shifted CLS mask and official template averaging.|No dynamic token prompt or TITAN Transformer updates. Query identity and residual complementarity are separate controlled operators.|Three fixed epochs; patient-specific mechanism unconfirmed; v1 invalidated, preserved. Wrong RNA is frozen inference intervention, not real retraining.|
|Phase11|S32 scalars become learned concept tokens plus identity; clinical tokens and RNA8 queries attend raw CONCH patches, bidirectional attention variants, learned anchor and nonlinear small correction.|True frozen text prototypes replace score-token-only identity. Background mean estimators are outcome-free crossfit nuisance regressors, not clinical query injection.|INCONCLUSIVE within repeatedly developed TCGA cohort.|
|Phase16|Official CONCH-Z projection/text space; captions append clinical fields and low/typical/high RNA8 states; equal-budget round-robin concept retrieval followed by prediction.|Fixed phrases and soft local read instead of patient-conditioned natural-language captions and discrete RNA states. Retains continuous measured molecular inputs and separately tests clinical residualization.|NO_CONFIRMED_INCREMENT; official semantic interface passed; raw-WSI histology QA blocked.|
|Phase19|Shared score q(P)k(G)/sqrt(d) + sum_r a_r(P)b_r(G)c_r(C); row/column masked softmax. RNA, gain, loss separate weighted-scalar/coverage tokens.|Clinical context estimates expected unimodal evidence and is subtracted before low-rank products; it does not directly rewrite every P-G score through a trilinear clinical term. Low-rank products themselves are reused mathematics.|MECHANISM_NOT_SUPPORTED_IN_THIS_ARCHITECTURE, not rejection of all clinical conditioning.|
|R0_R4|True-coordinate REGIONS pathology arm, fixed Hallmark50 tokens and separate old-union RNA512 direct risk, strong GRID-local CONCH/S32/clinical Cox anchor, honest same-teacher scalar risk.|New local interaction may use REGIONS512 but exact historical B0 keeps GRID-local mean PCA32. Shared frozen source and new clinical-residual products require new train-local fits.|R4 CP_SCALAR DSS 0.832667/0.812398 are historical fold-local counts; nested interaction versus CP_SCALAR +0.001490/-0.000559.|
|R5|Finite non-saturating/low-capacity readout attribution around fixed CP+RNA offset, 63-model core and 45 selected-control retrains described in local recovery/reproduction metadata.|Not another readout-only repair; introduces pre-DSS semantic correspondence training and explicit crossfit clinical-background subtraction.|Current local R5 folder contains recovery/reproduction metadata but no standalone final result table; do not infer new mechanism statistics from those files.|
|R6|Frozen S7 source with four programs, sparse original-gene value bank, RNA-conditioned original-patch reread and reverse original-gene reread; serial double-difference, no repeated state iteration.|B5 is not original-gene bidirectional rereading; main defined inputs are fixed Hallmark50 tokens and text morphology reads. Do not import S7/ST weights into this new three-modal contract silently.|Local R6 data-qualification reports predate training. Actual R6 saved-state existence is independently referenced by R7 9-cell/36-state interface audit and R8 locked source. Fresh source state bytes remain a live asset check.|
|R7|Train-RMS normalization, independent directions, separate program heads, gated one/two-round states, true wrong-query retrains.|Avoids interpreting numerical activation as benefit; B5 pretrains without DSS and fits a small frozen-feature Cox readout with explicit capacity/pairing controls.|99 fits/297 replays; numerical pathway activated but fixed GATE1 not supported; greater repair than damage not shown.|
|R8|Frozen R6 source evidence, train-local PCA2 per program, low-capacity complete bilinear program states and frozen RNA-conditioned image read variants.|B5 adds text-space queries and OOF clinical nuisance residualization, not merely a new name for bilinear readout.|81 convex fits/replays; nominal deterministic seed duplicates not independent evidence; R8-6 not supported; no new outer training.|

## R4强锚点的实际谱系

`src/bgb_en/baselines.py:7–48` 在当前训练患者内划3个固定随机held-out块，每块重新拟合临床编码、视觉PCA及RNA筛选。LOCAL CP/C+S32 Cox正则10，RNA512 Cox正则1；每位held-out患者仅获得排除本块训练的teacher风险。每个teacher保存**自己的fit-train风险均值/SD**，评估患者沿用这些量，不能以held-out/test均值重新定标。

最终风险是 `CP + beta*RNA`。`beta` 在训练侧crossfit风险上，只使用**同一被排除teacher块内的可比较患者对**，拟合一维pairwise logistic + ridge(1)，并非直接在in-sample融合embedding上再拟Cox。外层评估则使用outer-train重拟合teacher。新B0要复现这个对象；新统一低容量Efron Cox头作为单独对照，不能混名。

R4历史点估计CP_SCALAR=0.832667 Harrell /0.812398 Uno60，来自折内分子/分母汇总；不是本轮保底值，更不是新独立测试。嵌套交互相对标量的DSS差值为+0.001490/−0.000559，两个指标并不共同改善。

## 冻结图文空间的正确接口

- CONCH v1.5原特征768维先做 `normalize(raw @ official proj_contrast)`，再与**CONCH-Z自己**的normalized text比较。`scripts/prepare_phase16_semantic.py:14–16` 与 `src/phase16_semantic/runtime.py` 已留官方等价接口；历史 SPACE_COMPATIBILITY 中projection_replay_error为0。官方模型卡本轮浏览确认预计算v1.5可复用此投影，无须为该操作重新下载SVS。
- TITAN native768本身不是text-aligned；使用其自己的 `vision_encoder.proj`＋L2，再与TITAN text比较。Phase6A历史519切片/512患者、官方zero-shot概率误差1.49e-08；各模板先normalize，再平均后normalize。
- 不允许跨两个768空间直接cosine。保留32概念及两模板原字节；CONCH与TITAN分别编码，分组只用冻结文本几何，不用DSS。
- 当前历史CONCH519片的hop确实全部为1024 level0 pixels，UNI2为512；通用官方接口仍应读取每片真实属性而不是硬猜放大率。几何匹配不等于已验证局部组织学。

## 仍未通过本轮G0的事项

本静态审计没有检查新服务器数据盘克隆完成状态、迁移后模型和数据实物，也未执行新模型tensor-level forward→backward。因此G0保持待核验，不能用这里的源码/历史报告存在把它改成PASS。

- 当前512患者顺序/fold/65事件/月单位与全部519 H5、TITAN原生/投影表示、RNA/CNA measured masks重新绑定。
- 使用新服务器当前官方权重、原32概念/两模板做真实slide级官方等价检查；旧1024-patch投影检查不能自动冒称本轮几十张slide资格。
- 新8概念组/50 Hallmark关系表、B3随机几何定义、B5/B7活跃容量，以及B6a仅改变辅助正配对的路径测试全部实际通过。
- 无DSS预训练数据结构排除DSS/OS/PFI；image-only program头不能经RNA路由读取同患者真实表达。
- 先做方案明确规定的P0/P1和极小真实profile；本审计不授权或启动315内层＋105正式大矩阵。

## 来源与不确定性

现有Phase7候选prior引自CMS、EMT/间质混杂、出芽等外部研究；本轮只验证来源含义，不据TCGA结局重新挑边。尤其Bulk EMT可能反映成纤维细胞，不能从其升高断言tumor budding出现。

- [CONCH-Z官方模型卡](https://huggingface.co/MahmoodLab/CONCH-Z)：预计算特征投影与匹配文本接口，本轮2026-10-30浏览。
- [TITAN官方模型卡](https://huggingface.co/MahmoodLab/TITAN)：CONCH特征、坐标与level0步长接口，本轮浏览。
- [CMS原始研究元数据与摘要](https://pubmed.ncbi.nlm.nih.gov/26457759/)；[EMT与间质解耦原始研究](https://www.nature.com/articles/s41467-021-22800-1)。后者直接页面跳转失败后，经搜索服务读取期刊已索引正文；无新患者数据下载。

**优势：**把固定文本身份、临床背景残差和跨模态乘积分别纳入实际匹配控制，保留强主效应及旧负结果，避免把数值激活当成医学机制。

**风险：**旧队列反复开发、65事件、32描述高度相关、Bulk取材与WSI不一一对应、当前无独立ROI真值；等参数不等于等函数先验。已有低秩乘积/contrastive/reread均不构成新的数学原创主张。静态谱系判断置信度高；新B5语义正确性、预后增量与外部泛化均未测试。

精确逐文件SHA256、算子证据行号与七项合同澄清见 `HISTORY_OPERATOR_AUDIT.json`。
