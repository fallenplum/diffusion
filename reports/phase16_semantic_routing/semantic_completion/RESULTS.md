# Phase16 — 授权后的语义路由完整审计

**研究决策：NO_CONFIRMED_INCREMENT。官方Patch↔Text空间已验证；患者特异性RNA路由的稳定增量尚未确认。**

本报告接续a19eb457冻结的受阻批次，保留旧报告与数值不变。模型授权现已生效，不再是语义分支的阻塞项。G4双视觉分支仍受原始WSI/30组原图QA阻塞。

## 1. 实际完成范围

|模块|arms|外层单元|本次新增训练|
|---|---:|---:|---:|
|G2：8 selectors×128/512及6个基准|22|330|120|
|G3：5类错Prompt×128/512|10|150|150|
|G5：12核心及2随机mask、2omics-only|16|240|240|
|MIL两阶段补充消融|2|30|30|
|合计|50|750|540|

累计690个新outer-fold拟合，60个历史引用；本次新增540个，旧批150个训练不重跑。PFI只迁移同一DSS模型，不另算训练。1500条DSS/PFI逐折记录均交叉核验scikit-survival。Inner拟合、超参trial和两阶段selector训练另外记录，不谎报为新的独立外层cell。G4腾出的预算用于必要的组学/mask及MIL对照；未执行原图不支持的双视觉。

## 2. 数据与官方接口

- 512名患者、65个DSS事件；PFI有效510名/129事件；沿用同一患者折和seed31/47/83。
- **candidate-pool-limited**：519/520张原始切片有CONCH，512/512患者覆盖；使用1,278,642个原始可用patch，不使用128缓存冒充全库。缺失切片不替换/重命名。
- 官方CONCH-Z固定revision60211fced9c25a014494e248eedb6a6be80109f7，完整权重约1.67GB；E盘与服务器各保留一份，不公开权重或派生患者级向量。
- 504条权重精确加载，无missing/unexpected keys。原始CONCH视觉向量经官方768×768 proj_contrast，再L2归一化，才能和CONCH-Z文本余弦；没有用TITAN text直接匹配raw CONCH。
- 1024个投影样本独立重放误差1.86e-07；720组预算/ROI排序抽样检查完全一致；重新编码文本的最大半精度批次舍入差0.00029。
- 官方编码器128-token上下文保留一个CLS位置；所有实际新文本编码都检查≤127，不允许静默截断。本批最长新文本为91 tokens。
- 仍无原图：519张切片仅做计算矩形重叠，平均覆盖约74.2%、约20.1%未匹配；原图QA完成0组，G4保持BLOCKED。不得将这些数字称为组织学对齐正确率。

## 3. 实际Prompt与检索

32个CRC形态概念沿用外部知识词表，不按DSS筛选。实施采用新配置中的简短固定模板，在读取本次语义结果前锁定；与先前受阻批次的预备长模板不同。队列历经多阶段探索，所有结果均属内部探索，不是全新未接触队列的外部确认。

```text
Colorectal adenocarcinoma: {morphology}.
Patient: age {age}, sex {sex}; pathologic T {T}, N {N}, M {M}; AJCC stage {Stage}.
RNA expression programs: {states}. Molecular context, not observed morphology.
```

RNA分子状态为8个固定Hallmark程序：cell cycle、EMT、IFNγ、hypoxia、TGFβ、DNA repair、WNTβ-catenin、angiogenesis。使用本角色训练患者的基因缺失填补、标准化、可测性筛选与33/67分位阈值；量化为low/typical/high，覆盖不足标unmeasured，不把缺失当零表达，也不描述为突变。Clinical只使用原先6个合法字段。

Clinical/RNA上下文与每个形态句子**在编码前拼接**，经过完全冻结的官方文本塔，形成患者×概念query。不是编码后给静态向量加RNA偏移。每概念按Patch余弦降序，32概念round-robin取并集、去重；使用与旧对照相同的每患者配额和多切片合并规则。没有额外用DSS选择词表、ROI、表达阈值或query权重。

同癌种/Stage或缺失模式优先进行分区内derangement；单例时按癌种、再按本分区退化，所有次数见WRONG_PROMPT_FALLBACKS.csv。没有跨inner-train/inner-val/outer-train/outer-test借用患者。并非所有错配都能保持最严格分层，不能隐去退化。

## 4. DSS主要数值

|模型|Harrell / Uno60（3seed OOF均值）|
|---|---:|
|CS_P15|0.815181 / 0.784177|
|C4_LEGACY|0.813691 / 0.802965|
|RAND_512|0.819023 / 0.793462|
|GRID_512|0.818239 / 0.793180|
|MIL_512|0.817063 / 0.787710|
|MIL_TWO_STAGE_512|0.814946 / 0.786893|
|FIXED_TEXT_512|0.818317 / 0.788202|
|CLIN_TEXT_512|0.818788 / 0.788617|
|RNA_TEXT_512|0.818552 / 0.793163|
|CLIN_RNA_TEXT_512|0.819572 / 0.793113|
|JOINT_WRONG_RNA_512|0.818945 / 0.792132|
|SHUFFLED_FIELDS_128|0.820591 / 0.795750|
|G5_RNA_ONLY|0.499294 / 0.580751|
|G5_CNA_ONLY|0.524700 / 0.514815|
|G5_FIXED_TEXT_512_RNA|0.816122 / 0.788096|
|G5_CLIN_RNA_TEXT_512_RNA|0.816043 / 0.789861|
|G5_CLIN_RNA_TEXT_512_RC_UNGATED|0.809457 / 0.794233|
|G5_CLIN_RNA_TEXT_512_RC_GATED|0.810554 / 0.795747|

每seed先按同折可比较对分子/分母拼5-fold OOF，再均值3个seed；没有平均15个cell的C，也没有风险跨fold排序或把seeds合成新患者。历史C4还含RNA交互且预算为每张WSI128，是历史强参考，不能当作面积/输入完全匹配的ROI消融。固定Ridge omics-only在本协议下跨seed完全相同，seed_sd=0不代表3次独立验证。

## 5. 预定义主要比较：Bonferroni16

|比较|指标|Δ|95% CI|校正CI|正向seed|
|---|---|---:|---|---|---:|
|FIXED_TEXT_128-GRID_128|Harrell|-0.000157|[-0.006311,+0.005831]|[-0.010105,+0.009230]|2/3|
|FIXED_TEXT_512-GRID_512|Harrell|+0.000078|[-0.006050,+0.006230]|[-0.009908,+0.010147]|2/3|
|CLIN_TEXT_128-FIXED_TEXT_128|Harrell|+0.001098|[-0.002257,+0.004972]|[-0.004249,+0.007792]|3/3|
|CLIN_TEXT_512-FIXED_TEXT_512|Harrell|+0.000470|[-0.000982,+0.002448]|[-0.002003,+0.004152]|3/3|
|CLIN_RNA_TEXT_128-CLIN_TEXT_128|Harrell|-0.000314|[-0.002080,+0.001237]|[-0.003704,+0.002441]|0/3|
|CLIN_RNA_TEXT_512-CLIN_TEXT_512|Harrell|+0.000784|[-0.001991,+0.003797]|[-0.003888,+0.005783]|3/3|
|CLIN_RNA_TEXT_128-CS_P15|Harrell|+0.003764|[-0.003598,+0.011483]|[-0.008273,+0.016636]|3/3|
|CLIN_RNA_TEXT_512-CS_P15|Harrell|+0.004391|[-0.002339,+0.011661]|[-0.006030,+0.016082]|3/3|
|FIXED_TEXT_128-GRID_128|Uno60|-0.003409|[-0.013370,+0.004289]|[-0.022301,+0.008811]|1/3|
|FIXED_TEXT_512-GRID_512|Uno60|-0.004979|[-0.016818,+0.003640]|[-0.026470,+0.007762]|0/3|
|CLIN_TEXT_128-FIXED_TEXT_128|Uno60|+0.006306|[-0.001181,+0.019352]|[-0.004076,+0.031521]|3/3|
|CLIN_TEXT_512-FIXED_TEXT_512|Uno60|+0.000416|[-0.000580,+0.001955]|[-0.001282,+0.003336]|3/3|
|CLIN_RNA_TEXT_128-CLIN_TEXT_128|Uno60|-0.000554|[-0.002793,+0.000683]|[-0.005318,+0.001505]|1/3|
|CLIN_RNA_TEXT_512-CLIN_TEXT_512|Uno60|+0.004495|[-0.000925,+0.013576]|[-0.003494,+0.021724]|2/3|
|CLIN_RNA_TEXT_128-CS_P15|Uno60|+0.009479|[+0.000234,+0.023266]|[-0.002980,+0.034278]|3/3|
|CLIN_RNA_TEXT_512-CS_P15|Uno60|+0.008935|[+0.000933,+0.020659]|[-0.002137,+0.029472]|3/3|

CLIN_RNA_TEXT512相对强CS点估计约+0.00439/+0.00894，但两项校正区间均包含零；相对GRID512，Uno60点估计也没有提高。不能据较高Harrell单项宣称胜过历史C4，因为其Uno60仍低于C4。

## 6. RNA患者身份与语义负对照

|比较|指标|Δ|校正CI（次要family32）|
|---|---|---:|---|
|CLIN_RNA_TEXT_128-JOINT_WRONG_RNA_128|Harrell|+0.000078|[-0.002460,+0.002718]|
|CLIN_RNA_TEXT_128-SHUFFLED_FIELDS_128|Harrell|-0.001647|[-0.005795,+0.000000]|
|CLIN_RNA_TEXT_512-JOINT_WRONG_RNA_512|Harrell|+0.000627|[+0.000000,+0.003135]|
|CLIN_RNA_TEXT_512-SHUFFLED_FIELDS_512|Harrell|-0.000078|[-0.003659,+0.003806]|
|CLIN_RNA_TEXT_128-JOINT_WRONG_RNA_128|Uno60|+0.000846|[-0.001151,+0.008631]|
|CLIN_RNA_TEXT_128-SHUFFLED_FIELDS_128|Uno60|-0.002094|[-0.011313,+0.000000]|
|CLIN_RNA_TEXT_512-JOINT_WRONG_RNA_512|Uno60|+0.000981|[+0.000000,+0.006493]|
|CLIN_RNA_TEXT_512-SHUFFLED_FIELDS_512|Uno60|-0.000145|[-0.006408,+0.006454]|

真实RNA512对错RNA512的点差只有约+0.000627/+0.000981，校正bootstrap下界为0，未排除零增量。3seed都为正仍不足以通过本轮门槛。错误字段128对照的点估计更高，也削弱“正确患者语义必然推动DSS增益”的解释；这不赋予错误文本任何医学含义。

G1_COVERAGE.csv给出ROI overlap、8×8空间占用/集中度与重复行；它们是无标签检索诊断，没有ROI真值，不能称为检索准确率。wrong-prompt改变文本或ROI，也不等于改变了有效预后证据。

## 7. Query贡献与只在预测头加入组学

G5的RNA预测头保留与query同源的连续8程序分数及缺失比例；加入CNA时RNA部分不变。CNA采用同gene ID对齐的gain/loss分开聚合；same-gene gate为预先固定sigmoid(RNA基因zscore)×CNA gain/loss，不是学习得到的调控关系。True/random mask保持基因数、组间重叠和RNA/CNA同gene身份，按训练RNA均值/SD及可测类别匹配；未匹配染色体空间结构。

G5使用相同ROI/CS主干，加极小分子线性修正；没有新增fusion Transformer。RNA16维、RNA+CNA40维包含显式缺失比例，因此组学输入不同会改变头部参数数目；同类型true/random的参数预算一致。RNA-only与CNA-only为无Clinical/S32/ROI的低容量Ridge对照。

|同组学头下CLIN_RNA query−FIXED query|指标|Δ|校正CI（补充family20）|
|---|---|---:|---|
|G5_CLIN_RNA_TEXT_128_RNA-G5_FIXED_TEXT_128_RNA|Harrell|+0.000392|[-0.005449,+0.005795]|
|G5_CLIN_RNA_TEXT_128_RC_UNGATED-G5_FIXED_TEXT_128_RC_UNGATED|Harrell|+0.001568|[-0.005383,+0.009674]|
|G5_CLIN_RNA_TEXT_128_RC_GATED-G5_FIXED_TEXT_128_RC_GATED|Harrell|+0.002509|[-0.003440,+0.010304]|
|G5_CLIN_RNA_TEXT_512_RNA-G5_FIXED_TEXT_512_RNA|Harrell|-0.000078|[-0.003816,+0.003054]|
|G5_CLIN_RNA_TEXT_512_RC_UNGATED-G5_FIXED_TEXT_512_RC_UNGATED|Harrell|+0.000314|[-0.003427,+0.004566]|
|G5_CLIN_RNA_TEXT_512_RC_GATED-G5_FIXED_TEXT_512_RC_GATED|Harrell|+0.000000|[-0.004556,+0.004865]|
|G5_CLIN_RNA_TEXT_128_RNA-G5_FIXED_TEXT_128_RNA|Uno60|+0.004179|[-0.002079,+0.022772]|
|G5_CLIN_RNA_TEXT_128_RC_UNGATED-G5_FIXED_TEXT_128_RC_UNGATED|Uno60|+0.004667|[-0.004809,+0.023431]|
|G5_CLIN_RNA_TEXT_128_RC_GATED-G5_FIXED_TEXT_128_RC_GATED|Uno60|+0.005900|[-0.001919,+0.024560]|
|G5_CLIN_RNA_TEXT_512_RNA-G5_FIXED_TEXT_512_RNA|Uno60|+0.001765|[-0.002089,+0.014330]|
|G5_CLIN_RNA_TEXT_512_RC_UNGATED-G5_FIXED_TEXT_512_RC_UNGATED|Uno60|+0.000183|[-0.002466,+0.003443]|
|G5_CLIN_RNA_TEXT_512_RC_GATED-G5_FIXED_TEXT_512_RC_GATED|Uno60|-0.000298|[-0.004586,+0.004089]|

这些直接配对用于区分RNA只进入头部与RNA改变ROI选择；没有被追认为primary。G5核心/随机mask的完整配对另见G5/PAIRED_COMPARISONS.csv（family28）。补充direct family20覆盖6个同头比较、2个两阶段MIL基准及2个omics-only对常数风险的比较。各family分别列明，没有从胜出结果挑选主检验。

本轮RNA-only八程序表示的Harrell约0.499、Uno60约0.581；CNA-only约0.525/0.515。不能将它们当成此前Top512或富集512的强RNA预后表征。它们只是固定低维输入，本次未证明其稳定的DSS双指标独立排序能力，也不能据此否定完整RNA或CNA信息。

低容量Cox由独立逐事件Efron梯度核验，最大绝对梯度9.96e-08。它排除了这些固定λ拟合的明显优化未收敛，不能排除内层λ选择方差、表示压缩或小事件效应。OMICS_ONLY_SELECTION.csv保留OFF选择，不把OFF当作生物信息为零的证据。

## 8. MIL两阶段及训练预算

新增两阶段MIL使用相同部署容量：StageA只在fit患者训练，固定80epoch，不用inner-val挑selector epoch；StageB冻结同一attention向量并重新拟合CS/ROI头，只由inner-val选择StageB epoch。没有第二套部署评分网络。30个外层流程额外包含60个StageA角色拟合，预算明确多于原端到端MIL；不是计算量匹配的纯架构因果对照。

|两阶段−端到端MIL|指标|Δ|校正CI（family4）|
|---|---|---:|---|
|MIL_TWO_STAGE_128-MIL_128|Harrell|-0.001333|[-0.004570,+0.001467]|
|MIL_TWO_STAGE_512-MIL_512|Harrell|-0.002117|[-0.006178,+0.000852]|
|MIL_TWO_STAGE_128-MIL_128|Uno60|-0.000707|[-0.003255,+0.002033]|
|MIL_TWO_STAGE_512-MIL_512|Uno60|-0.000817|[-0.005808,+0.008581]|

全部新核心、G3、G5神经流程保留原80epoch/Adam0.001/EMA0.3 validation-Cox选择规则，从同seed初始化fresh outer fit，不根据外折分数扩预算。先前16个仅内层320epoch诊断中14个loss继续下降，但Harrell提高6个、下降6个、相同4个。**训练预算充分性仍未确立。**

|模块|选中80epoch的单元|
|---|---:|
|语义/G3|211/270|
|G5神经头|126/210|
|两阶段StageB|20/30|

训练诊断保留各epoch损失、inner Harrell、真实梯度和mask/冻结检查。RISK_COMPONENTS.csv只作校正幅度/CS重优化的预测分解，不当作重训的因果消融。

## 9. Stage/RNA与缺失敏感性

Prompt消融固定预测头Clinical/S32输入，只改变选区文本：去TNM/AJCC、仅Stage、去RNA。结果见PROMPT_INFERENCE_ABLATION_SEED.csv；它们不新增训练，但存在分布改变，不等于重新训练的公平因果对照。PFI只迁移DSS已训练风险，未用PFI选择参数；不能把PFI当成仅增加事件数的干预。

原始RNA203人×3024列缺失块按缺失模式单独分层，见MISSING_BLOCK_SENSITIVITY.csv。基因去重后的可用集合由训练折决定；没有证实测序平台身份。亚组数值是描述性敏感性，不引入新的主假设或外折反馈。

## 10. 五个研究问题与边界

1. **合法Patch图文空间：已成立。** 官方CONCH-Z投影/文本接口、权重和数值均验证；不是病理标注真值认证。
2. **患者条件路由胜过FIXED/GRID/MIL：尚未确认。** 预定义校正检验没有形成稳定的双指标支持。
3. **真RNA胜过错RNA：仅见极小点差，区间仍包含零。** 不能宣称患者特异性语义机制成立。
4. **UNI2正确匹配后的净增量：未检验。** 缺原图QA，G4 BLOCKED。
5. **同时超过强CS与C4且跨seed稳定：未满足。** 不发布SOTA或医学机制成立结论。

**优势：** 官方空间、全可用H5候选、统一患者预算、真实S32 forward、完整负对照/条件头比较、同gene RNA/CNA与分离gain/loss、独立逐checkpoint/统计重算。
**风险：** 同队列反复探索、65事件、80epoch上限、固定8程序和量化信息瓶颈、随机mask单次实现/未匹配CNA染色体结构、固定gate缺少患者级调控真值、无原图/ROI金标准、固定模型bootstrap不含全流程重训方差。部分极端校正CI的Monte Carlo尾部样本有限。
**置信度：** 已执行范围的代码/指标审计为高；患者特异性语义和医学因果解释仍缺支持。负结果限于当前表示、选区和训练协议，不能简化成idea NO-GO。

## 拓展思考

优先核查用于Prompt的分子表示是否保留已知RNA预后信息，并区分ROI变化与有效排序变化；当前不支持继续盲目扩展融合深度。

## 你可能想知道

E盘与服务器的CONCH-Z权重都保留。患者OOF、临床文本、基因测量、ROI坐标及权重不公开；持久化了小型私有文本重放胶囊，完整大缓存位于/dev/shm可重建。

官方接口依据：[CONCH-Z/LIMIT](https://github.com/mahmoodlab/LIMIT)、[模型页面](https://huggingface.co/MahmoodLab/CONCH-Z)。