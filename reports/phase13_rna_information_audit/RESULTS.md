# Phase13-R0 — RNA预后信息保留与强条件增量诊断



**3330条正式模型/读出记录全部完成并独立复算；90个RNA-only训练管线、45个缺失inner状态精确重建、45个既有outer融合checkpoint复用。**



三组输入固定Phase12训练折基因索引及分组：Top512、Hallmark富集512、匹配非通路512。固定原5外折×3初始化；DSS512/65为主，原CDR验证后的PFI510/129为次要。原月值=原CDR天数/30.417，未改历史时间；两名PFI零时间事件患者按历史有效性规则排除。缓存总计131个事件，正式正时间交集129个事件。



## 当前最具体的解释



|可能原因|证据与限制|
|---|---|
|与强基线重叠|Hallmark RNA残差排序大幅下降、直接RNA加性增量未确认；CS对RNA风险的R²只约−0.16到0.23，不能说大多数分数方差已被解释，更不能证明条件独立。|
|编码/融合未保留|完整融合RNA token仍有0.70313/0.65237排序；相对原RNA的主要区间跨零。mean只有0.65044/0.61079，完整−mean的次要差异提示压缩/读出问题；512vs64容量不同，且次要比较未多重校正。|
|事件稀缺|DSS仅65事件，Uno信用Neff约37.25；PFI有效129事件，Neff约70.75，但RNA条件增量仍不稳定。这个终点对比不能隔离纯功效因素。|



## 主要事实表



|endpoint|model|Harrell|Harrell_seed_SD|Uno60|Uno60_seed_SD|test_Efron_loss|
|---|---|---|---|---|---|---|
|DSS|C|0.79711|0.00000|0.73017|0.00000|3.75558|
|DSS|CS|0.80593|0.00000|0.77372|0.00000|3.46998|
|DSS|HALL_CS_RAW|0.79769|0.00000|0.75513|0.00000|7.43040|
|DSS|HALL_PCA16_tuned|0.62056|0.00000|0.61550|0.00000|3.99871|
|DSS|HALL_RESID_tuned|0.55563|0.00000|0.58979|0.00000|11.10188|
|DSS|HALL_R_fixed1|0.72148|0.00000|0.67678|0.00000|4.23364|
|DSS|HALL_R_tuned|0.72030|0.00000|0.67555|0.00000|9.91090|
|DSS|MATCH_CS_RAW|0.80593|0.00000|0.77372|0.00000|3.46998|
|DSS|MATCH_PCA16_tuned|0.57869|0.00000|0.56805|0.00000|4.08125|
|DSS|MATCH_RESID_tuned|0.50929|0.00000|0.54542|0.00000|11.72469|
|DSS|MATCH_R_fixed1|0.65914|0.00000|0.63433|0.00000|4.28052|
|DSS|MATCH_R_tuned|0.64573|0.00000|0.63185|0.00000|10.72361|
|DSS|S|0.70430|0.00000|0.65291|0.00000|3.93176|
|DSS|TOP_CS_RAW|0.80593|0.00000|0.77372|0.00000|3.46998|
|DSS|TOP_PCA16_tuned|0.53681|0.00000|0.51310|0.00000|4.04709|
|DSS|TOP_RESID_tuned|0.56528|0.00000|0.55461|0.00000|16.37028|
|DSS|TOP_R_fixed1|0.65749|0.00000|0.63016|0.00000|4.22512|
|DSS|TOP_R_tuned|0.64479|0.00000|0.62958|0.00000|7.28782|



RNA-only富集集合的点估计较高，但它相对Top512/匹配集合的Harrell与Uno60配对95%区间仍含零；不能把点估计优势当成已确认的基因集合优越性。MODEL_ABSOLUTE_CI提供所有模型绝对区间，属于描述性结果。



## 1. 是否主要与Clinical/Pathology重叠？



联合分块正则Cox分别拟合Clinical/S32与RNA系数；基线正则inner-only确定，RNA五候选包括精确关闭RNA。不用训练内风险stacking。RNA主效应以原512基因直接进入，因此不经过旧神经encoder或RNA→Patch融合。



**TOP：** Harrell Δ=+0.00000,95%CI[+0.00000,+0.00000],校正/次要CI[+0.00000,+0.00000],正向seed0/3；Uno60 Δ=+0.00000,95%CI[+0.00000,+0.00000],校正/次要CI[+0.00000,+0.00000],正向seed0/3

**HALL：** Harrell Δ=-0.00823,95%CI[-0.04005,+0.02270],校正/次要CI[-0.05601,+0.03862],正向seed0/3；Uno60 Δ=-0.01858,95%CI[-0.08017,+0.03393],校正/次要CI[-0.12359,+0.06266],正向seed0/3

**MATCH：** Harrell Δ=+0.00000,95%CI[+0.00000,+0.00000],校正/次要CI[+0.00000,+0.00000],正向seed0/3；Uno60 Δ=+0.00000,95%CI[+0.00000,+0.00000],校正/次要CI[+0.00000,+0.00000],正向seed0/3



|endpoint|gene|fold|n|events|heldout_reference_RNA_risk_R2|heldout_reference_RNA_risk_correlation|notice|
|---|---|---|---|---|---|---|---|
|DSS|HALL|1|103|13|0.12167|0.41490|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|DSS|HALL|2|103|14|0.22920|0.52101|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|DSS|HALL|3|102|12|0.03146|0.26955|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|DSS|HALL|4|102|13|0.06101|0.42766|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|DSS|HALL|5|102|13|-0.16301|0.12580|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|DSS|MATCH|1|103|13|0.07527|0.39148|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|DSS|MATCH|2|103|14|-0.13905|0.09439|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|DSS|MATCH|3|102|12|0.03914|0.30113|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|DSS|MATCH|4|102|13|-0.13658|0.28595|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|DSS|MATCH|5|102|13|-0.23587|0.14027|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|DSS|TOP|1|103|13|-0.42642|0.15654|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|DSS|TOP|2|103|14|-0.03224|0.26199|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|DSS|TOP|3|102|12|0.13463|0.41321|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|DSS|TOP|4|102|13|0.02034|0.33514|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|DSS|TOP|5|102|13|0.02857|0.29009|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|PFI|HALL|1|101|28|0.11125|0.40350|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|PFI|HALL|2|103|19|0.29700|0.55145|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|PFI|HALL|3|102|32|0.17968|0.46476|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|PFI|HALL|4|102|29|0.14388|0.42358|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|PFI|HALL|5|102|21|0.12830|0.37435|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|PFI|MATCH|1|101|28|-0.04048|0.29565|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|PFI|MATCH|2|103|19|0.19765|0.46311|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|PFI|MATCH|3|102|32|0.20545|0.46869|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|PFI|MATCH|4|102|29|-0.07780|0.28708|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|PFI|MATCH|5|102|21|-0.11692|0.16706|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|PFI|TOP|1|101|28|0.14682|0.45367|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|PFI|TOP|2|103|19|0.19191|0.47044|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|PFI|TOP|3|102|32|0.00144|0.32284|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|PFI|TOP|4|102|29|0.05192|0.34986|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|
|PFI|TOP|5|102|21|-0.13429|0.04835|unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence|



预测性重叠是连续诊断：训练内无监督CS→RNA线性map，用heldout RNA-only Cox参考分数检验可预测性；参考分数不是患者真值。相关性高不能证明条件独立；条件增量未检出也可能来自模型限制或区间过宽。Clinical-only/S32-only条件增量及RNA残差模型见ALL_PAIRED。



## 2. 编码/融合是否保留RNA排序？



|model|Harrell|Uno60|test_Efron_loss|
|---|---|---|---|
|F_HALL_after_RNApatch_full|0.68329|0.62103|5.15540|
|F_HALL_after_RNApatch_mean|0.68478|0.64320|4.43052|
|F_HALL_after_processed_full|0.70313|0.65237|4.76499|
|F_HALL_after_processed_mean|0.65044|0.61079|4.27075|
|F_HALL_before_processed_full|0.71364|0.64961|4.17544|
|HALL_R_fixed1|0.72148|0.67678|4.23364|
|RO_HALL_HEAD|0.69874|0.65732|4.00543|
|RO_HALL_after_processed_full|0.71238|0.66102|4.44678|
|RO_HALL_after_processed_mean|0.65561|0.62612|3.99142|

本轮同时读出完整8×64 token与64D均值，不再只测mean。匹配主要比较是同512维、同train-local标准化、同Ridge λ=1的原始RNA与融合后完整token。



**TOP：** Harrell Δ=-0.03121,95%CI[-0.06634,+0.00419],校正/次要CI[-0.08310,+0.02245],正向seed0/3；Uno60 Δ=-0.00633,95%CI[-0.05494,+0.04922],校正/次要CI[-0.07803,+0.08181],正向seed1/3

**HALL：** Harrell Δ=-0.01835,95%CI[-0.05467,+0.01788],校正/次要CI[-0.07151,+0.03504],正向seed0/3；Uno60 Δ=-0.02441,95%CI[-0.08885,+0.03248],校正/次要CI[-0.13422,+0.06089],正向seed1/3

**MATCH：** Harrell Δ=-0.02274,95%CI[-0.07054,+0.02503],校正/次要CI[-0.09264,+0.04895],正向seed0/3；Uno60 Δ=-0.04443,95%CI[-0.11041,+0.01578],校正/次要CI[-0.14959,+0.04691],正向seed0/3



**TOP均值−完整：** Harrell Δ=-0.02878,95%CI[-0.06507,+0.00729],校正/次要CI[-0.06507,+0.00729],正向seed1/3；Uno60 Δ=-0.03688,95%CI[-0.08131,+0.00727],校正/次要CI[-0.08131,+0.00727],正向seed1/3

**HALL均值−完整：** Harrell Δ=-0.05269,95%CI[-0.09171,-0.01283],校正/次要CI[-0.09171,-0.01283],正向seed0/3；Uno60 Δ=-0.04158,95%CI[-0.08145,-0.00203],校正/次要CI[-0.08145,-0.00203],正向seed0/3

**MATCH均值−完整：** Harrell Δ=-0.03889,95%CI[-0.07887,+0.00074],校正/次要CI[-0.07887,+0.00074],正向seed0/3；Uno60 Δ=-0.00518,95%CI[-0.06923,+0.06041],校正/次要CI[-0.06923,+0.06041],正向seed1/3



**TOPRNA-only学习−原始固定读出：** Harrell Δ=-0.03199,95%CI[-0.06306,-0.00118],校正/次要CI[-0.06306,-0.00118],正向seed0/3；Uno60 Δ=-0.01534,95%CI[-0.05702,+0.02998],校正/次要CI[-0.05702,+0.02998],正向seed1/3

**HALLRNA-only学习−原始固定读出：** Harrell Δ=-0.00910,95%CI[-0.04797,+0.03055],校正/次要CI[-0.04797,+0.03055],正向seed1/3；Uno60 Δ=-0.01576,95%CI[-0.08303,+0.03824],校正/次要CI[-0.08303,+0.03824],正向seed1/3

**MATCHRNA-only学习−原始固定读出：** Harrell Δ=-0.02611,95%CI[-0.07149,+0.02075],校正/次要CI[-0.07149,+0.02075],正向seed0/3；Uno60 Δ=-0.02207,95%CI[-0.07921,+0.03085],校正/次要CI[-0.07921,+0.03085],正向seed0/3



PAIR_RETENTION_TRANSITIONS逐事件对报告原正确排序被丢失/原错误排序被挽救，以及Clinical+S32错误对上的RNA线索。REFERENCE_RISK_RECOVERY以train-only RNA Cox风险为参考，检查各级表征的heldout可解码性。此处只识别当前线性读出的信息保留，不能声称信息论上的互信息消失。



**均值pooling对照属于预锁次要检验，未做多重比较校正；完整512维与均值64维读出的有效容量/正则也不同。因此现有结果支持进一步检查聚合/读出瓶颈，尚非纯粹信息损失的因果证明。**



RNA-only与融合使用相同RNA tokenizer/分组与匹配初始化，但RNA-only用flatten小head，融合用RNA→Patch/mean/联合head及锚点。差别同时涉及训练目标与producer读出路径，不能单独归因为多任务目标抑制。旧融合latent对PFI属于DSS-producer transfer，PFI新RNA-only则按PFI训练，已单列。



## 3. 65个事件是否足以归因？



|endpoint|metric|events_observed|events_with_comparable_credit|effective_event_credit_Neff|largest_event_mass_fraction|comparable_mass|notice|
|---|---|---|---|---|---|---|---|
|DSS|Harrell|65|65|54.38797|0.02329|4251.00000|credit concentration,not independent-pair sample size|
|DSS|Uno60|65|62|37.25452|0.07254|11171.72870|credit concentration,not independent-pair sample size|
|PFI|Harrell|129|129|109.59841|0.01236|8173.00000|credit concentration,not independent-pair sample size|
|PFI|Uno60|129|127|70.74613|0.06665|18514.96539|credit concentration,not independent-pair sample size|



DSS/PFI的观测事件数、可比较事件数、IPCW信用集中程度不同；effective_event_credit_Neff是权重集中诊断，不是有效独立患者或正式功效计算。逐事件删除仅作固定模型影响分析，不用于改变主队列/主要比较。Bootstrap不包含重训、历史探索与内层模型选择方差。



**PFI TOP条件增量：** Harrell Δ=-0.01187,95%CI[-0.03593,+0.00838],校正/次要CI[-0.03593,+0.00838],正向seed0/3；Uno60 Δ=-0.00593,95%CI[-0.09734,+0.08485],校正/次要CI[-0.09734,+0.08485],正向seed0/3

**PFI HALL条件增量：** Harrell Δ=+0.00697,95%CI[-0.01588,+0.02872],校正/次要CI[-0.01588,+0.02872],正向seed3/3；Uno60 Δ=+0.00745,95%CI[-0.05972,+0.06666],校正/次要CI[-0.05972,+0.06666],正向seed3/3

**PFI MATCH条件增量：** Harrell Δ=-0.02435,95%CI[-0.05333,+0.00208],校正/次要CI[-0.05333,+0.00208],正向seed0/3；Uno60 Δ=-0.01242,95%CI[-0.05882,+0.04129],校正/次要CI[-0.05882,+0.04129],正向seed0/3



PFI事件数更多，但生物终点、有效患者、删失与事件时间同时变化。PFI改善最多是与监督稀缺解释一致的敏感性线索，不能证明DSS失败仅因65个事件；若PFI也不改善，也不能否定所有组学互补信息。



## 训练与选择规则诊断



90个RNA-onlyinner均完成80epochs，选中epoch范围10–25；预算边界敏感单元0。75个单元在未限制epoch范围时最佳EMA早于10。保留继承的min10规则，未依据新外折成绩改训练预算或选择规则；这个限制可能影响新RNA-only目标，不能把低分只归于encoder结构。完整损失/梯度和epoch表交付。



## 研究判断与置信度



|gene|conditional_increment|matched_full_token_readout_loss|overlap_causal_claim|event_scarcity_causal_claim|
|---|---|---|---|---|
|TOP|INCONCLUSIVE_OR_NOT_DETECTED|INCONCLUSIVE|NOT_IDENTIFIED|NOT_IDENTIFIED|
|HALL|INCONCLUSIVE_OR_NOT_DETECTED|INCONCLUSIVE|NOT_IDENTIFIED|NOT_IDENTIFIED|
|MATCH|INCONCLUSIVE_OR_NOT_DETECTED|INCONCLUSIVE|NOT_IDENTIFIED|NOT_IDENTIFIED|



## 匹配正则附录：结果查看后的探索性控制



|endpoint|gene|Harrell|Uno60|unique_outer_fits|scope|
|---|---|---|---|---|---|
|DSS|TOP|0.78240|0.80183|5|POSTHOC_EXPLORATORY_MATCHED_CAPACITY_CONTROL|
|DSS|HALL|0.80028|0.77431|5|POSTHOC_EXPLORATORY_MATCHED_CAPACITY_CONTROL|
|DSS|MATCH|0.75653|0.72231|5|POSTHOC_EXPLORATORY_MATCHED_CAPACITY_CONTROL|
|PFI|TOP|0.66793|0.64829|5|POSTHOC_EXPLORATORY_MATCHED_CAPACITY_CONTROL|
|PFI|HALL|0.70329|0.68475|5|POSTHOC_EXPLORATORY_MATCHED_CAPACITY_CONTROL|
|PFI|MATCH|0.67919|0.67389|5|POSTHOC_EXPLORATORY_MATCHED_CAPACITY_CONTROL|



原RAW条件模型选择RNA正则，token条件probe固定λ=1。新增30个凸Cox拟合，对三输入和两终点全部使用既有probe的λ=1匹配控制，基线λ仍来自原inner-only选择。APPENDIX_PROTOCOL_LOCK明确记为posthoc，36项区间作家族校正但仍不等于新的独立确认；不改变3330条主矩阵及12项主要结果。



Hallmark固定RAW在DSS条件下为0.80028/0.77431，较CS为−0.00565/+0.00059；普通与附录校正区间均含零。这个控制没有揭示被正则选择隐藏的稳定DSS正增量。完整token条件读出与固定RAW的差值同样不能通过附录校正，见APPENDIX_PAIRED全部负结果。



Top/MATCH的CS+RAW模型在五个outer fit都选择了OFF，因此预测与CS精确相同、增量区间为[0,0]。这只描述预锁选择程序的输出，不能证明真实RNA条件效应为零。凸Ridge各seed记录复制同一个唯一拟合，不是三个独立初始化或额外患者证据。



Harrell-based正则选择与Efron目标存在差别：Hallmark RNA-only tuned的heldout Efron loss明显高于固定λ=1，条件模型也可能产生更极端风险而没有排序增量。该loss是同协议的heldout partial-risk-set诊断，并非独立绝对风险校准验证。早于min10的最佳验证时点与正则选择不稳定，都属于当前训练/选择配置限制；未事后优化并宣称原预注册。



本轮结果可以区分哪些配置保留排序、哪些加性增量可检出；三种解释可能并存，尚不能由单一相关性、CI含零或PFI对比排他地确定原因。执行/输入与数学核验置信度高；医学机制及外部泛化置信度有限。独立程序复算不等于独立外部审稿人验收。



## 优势与风险



优势：三组固定输入、强条件基线、低容量联合Cox、完整token与mean、匹配RNA初始化、固定probe及两种终点共享患者折。风险：65个DSS事件、同队列多轮探索、监督producer训练内适配可能过拟合、producer目标/head不同、固定λprobe不能覆盖所有可解码函数。RNA3024缺失区块仅做代理，不伪称已验证平台身份；子组少事件诊断见SUBGROUP_METRICS。



## 拓展思考 / 你可能想知道



若直接RNA加性模型已有稳定条件增量而旧token读出不足，优先保留低容量RNA风险/表示再比较新的交互；若连加性条件增量也不稳定，应先报告区间与重叠诊断，再决定外部验证或更丰富终点。不得仅因使用医学通路就认定知识语义创新成立。



## 审计和来源



指标3330记录与sksurv交叉验证；checkpoint3330回放；100独立特征矩阵、60PCA/残差transform、1800显式重复患者bootstrap标量检查通过。六主要比较×两指标作12项校正；所有PFI与其它诊断为次要未校正，不能事后升级。原Phase12产物保持只读。
