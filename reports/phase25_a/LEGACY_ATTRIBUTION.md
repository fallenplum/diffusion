# Phase25-A：M7R FiLM 与 M8P1R SPO 历史归因审计

日期：2026-10-18。只读检查历史源码、预测、既有诊断与公开汇总；不训练，不修改历史结果，不重新 bootstrap。匿名数值补充由 [LEGACY_ATTRIBUTION.json](LEGACY_ATTRIBUTION.json) 记录。

## 1. 已确认的证据与缺口

|问题|已有证据|本轮可支持的结论|仍缺少的证据|
|---|---|---|---|
|M7R FiLM vs concat/MLP|175个历史正式单元；F1/F2/F3/F6为独立训练并fresh outer-refit；共同组件初始化核验；FiLM调制幅度CSV|可以比较这些完整训练流程的实际预测；FiLM有非零调制不等于有净收益|训练权重未持久化，不能做当前checkpoint关闭FiLM、幅度扫描或风险回放|
|M7R FiLM vs generic MLP|F2交互18,592参数，F3为18,688参数，差异约0.51%；共有encoder/fusion/head初始相同|这是近似参数匹配的真实重训对照|不分离在线FiLM作用和共同主干联合优化；不是完全同参数、同计算|
|SPO两个残差头|75个残差正式单元；保存了anchor、P/O residual及risk诊断；另有25个冻结S0|可审计输出重建与P/O输出相关性；SP/SO/SPO是独立重训|无保存模型权重或逐模块真实训练梯度，不能从零初始化单测推断训练后各分支梯度|
|SPO是否共享同一个编码器|正式 `MultimodalBackbone` 内P/O各有独立encoder；SPO风险为anchor+dp+do|共享一个容器和Cox目标，不共享P/O参数；无P–O跨注意力或临床条件交互|不能把SPO方向性增量称为已证明跨模态交互|

服务器现场检查：5个 `diffusion_runtime_m7r_<seed>` 与 `m8p1r_workers` 均未发现 `.pt/.pth/.ckpt`；前者保留预测旁车，后者保留100份预测及诊断。源码也没有保存训练 `state_dict` 的调用。该结论限于已核历史运行目录，不能证明磁盘任何位置都绝无备份。

本地 `phase-m7r` 是陈旧草稿，FiLMModule缺forward，不能当作正式执行源码。已核服务器正式源码与 `tmp/phase10_trimodal_git/src/phase_m7r` 字节哈希一致，正式FiLMModule有forward；M8P1R正式源码也以服务器及正式归档为准，不用临时旧runner替代修复版。

## 2. FiLM：幅度、关闭与重训分别回答什么

正式公式是 `p_mod=p+tanh(gamma(o))*LayerNorm(p)+beta(o)`。最后生成层零初始化只保证初始恒等；gamma有界激活不限制beta、后续head或最终风险幅度。历史CSV可验证gamma/beta及表征修正大小，不能从表征范数反推预后贡献。

F3−F1或F3−F2的整体性能差同时包含FiLM、encoder、late fusion及risk head的完整优化轨迹，且各模型独立inner选epoch后fresh outer-refit。即使共同初始化相同，也不能把完整差值全部归因于推理时FiLM修正。缺权重使当前冻结关闭和幅度缩放审计阻断；不能新训练一份模型冒充恢复历史checkpoint。

## 3. SPO：加性分支并不是显式交互

风险为 `r=anchor.detach()+hP(encP(P))+hO(encO(RNA,CNA))`；ClinicalEncoder虽被计算，但不进入两残差输出，临床信息由冻结anchor提供。P/O训练参数彼此独立；共同Cox目标会通过总风险耦合优化，但这与跨模态特征交互不同。

两个末层均零初始化，因此初始化时encoder和隐藏层可能没有来自残差输出的梯度；以后是否形成有效更新必须依实际训练记录验证。当前没有权重/梯度日志，不能声称已核验训练后梯度路径；可以依据源码明确哪些路径存在。

保存的 `delta_pathology/delta_omics` 允许在不读结局时计算逐cell输出相关性及anchor相关性。高相关不证明冗余，低相关不证明互补，负相关也可能是共同优化中的补偿。输出相关性需与实际条件排序增量分开。SPO−SP/SO是独立重训差异，不是SPO同模型关一个头；保存的加性分量可代数构造去分支风险，但不能代替特征级权重回放。本轮不自动增加此性能矩阵。

## 4. 历史统计问题：旧CI不能作为当前确认

### M7R

已逐行核对正式 `aggregate._fast_bootstrap`：candidate与baseline分别调用 `one`，每次在内部单独生成 `rng.multinomial` 权重。虽然历史报告称paired bootstrap，**实际两者没有共享同一患者重采样权重**。另外，它对跨fold拼接OOF风险求C，Uno使用全队列censoring估计；主表则是fold C等权均值。旧CI不对应当前within-fold统计对象，也不能作为FiLM机制确认。

因此历史报告“Gate C PASS/FiLM-specific inductive bias supported”只保留为旧报告的描述，不继承为当前证实结论。这里不覆盖旧文件、也不重新bootstrap。

### M8P1R SPO

正式aggregate确用同一个idx同时重采candidate和baseline，是真正paired；但仍对跨fold拼接OOF求C、Uno用全cohort censoring且未采用当前Uno60协议。主表/seed表为fold C等权均值，CI与主表也不是同一统计对象。其旧 `DIRECTIONAL` 不等于当前协议确认，亦不证明两个残差头存在医学交互。

当前追加评测只允许：严格核对原患者顺序和折号后，以现有预测重算within-fold Harrell、当前修复Uno60与配对修复/损伤；无bootstrap，无新的学习或选择。历史训练损失也与后期Efron协议不同，重新评测不会将旧训练变成Efron训练。

## 5. 患者与结局谱系

现场从M7R历史commit `6821df983928cb9cc09a56a30c18cc65eca29159`、M8P1R commit `59d558c45b6e9cc474838d5a7cfd547ea3a7cc69` 的Git blob读取原固定患者清单；两者原blob SHA256均为 `273ec884ea2bc7afbe817305c5c423e56c012790e610965e12f01ea1052e5f24`。

按原loader的patient排序后，512患者顺序和5-fold分配均与当前canonical dataset逐元素一致。当前DSS endpoint文件哈希匹配Phase13 SOURCE_REGISTRY，patient_hash匹配，time/event/valid与canonical dataset逐元素相同，65事件且全valid。进一步完成75份M7R预测和100份SPO系列预测的shape、旁车或worker指标核验：旧Harrell逐份精确复现，最大误差均为0，不是仅凭长度拼接。

## 6. 现存预测的统一评测：只追加点估计

实际执行 [audit_phase25_a_legacy.py](../../scripts/audit_phase25_a_legacy.py)，未构建模型、未训练、未读原始RNA/WSI矩阵、未bootstrap、未改变旧预测。当前统计使用within-fold Harrell及修复Uno60（事件时间严格小于60月、IPCW由对应outer-train估计、当前风险tie规则）。每seed先加总五fold concordance分子/分母再取比，最后平均5seed；不是跨fold风险ensemble，也不是25个独立seed。

|M7R完整训练流程|Harrell|Uno60|
|---|---:|---:|
|F1 concat|0.780334|0.789839|
|F2 generic residual MLP|0.782686|0.791465|
|F3 FiLM|0.787391|0.793864|

|配对比较|指标|修复质量|损伤质量|净变化|正seed|
|---|---|---:|---:|---:|---:|
|F3−F1|Harrell|0.012703|0.005646|+0.007057|3/5|
|F3−F1|Uno60|0.008838|0.004813|+0.004025|3/5|
|F3−F2|Harrell|0.018113|0.013409|+0.004705|4/5|
|F3−F2|Uno60|0.017724|0.015325|+0.002399|5/5|

F3相对近容量MLP的方向性线索仍存在，但当前没有有效确认区间、没有同checkpoint FiLM关闭归因，也没有消除全流程联合优化差异。因此不能沿用旧“FiLM归纳偏置已被确认”的结论。

|M8P1R完整训练流程|Harrell|Uno60|
|---|---:|---:|
|S0 frozen Clinical|0.773724|0.780935|
|SP|0.776664|0.784614|
|SO|0.774077|0.781085|
|SPO|0.777982|0.784556|

|配对比较|指标|修复质量|损伤质量|净变化|正seed|
|---|---|---:|---:|---:|---:|
|SP−S0|Harrell|0.008798|0.005857|+0.002940|4/5|
|SP−S0|Uno60|0.007883|0.004204|+0.003679|3/5|
|SO−S0|Harrell|0.000470|0.000118|+0.000353|4/5|
|SO−S0|Uno60|0.000317|0.000167|+0.000150|2/5|
|SPO−S0|Harrell|0.011127|0.006869|+0.004258|5/5|
|SPO−S0|Uno60|0.010799|0.007177|+0.003621|4/5|
|SPO−SP|Harrell|0.005269|0.003952|+0.001317|4/5|
|SPO−SP|Uno60|0.006204|0.006261|−0.000058|4/5|
|SPO−SO|Harrell|0.010915|0.007010|+0.003905|5/5|
|SPO−SO|Uno60|0.010673|0.007202|+0.003471|4/5|

SPO相对Clinical的方向性增量仍存在；相对SP则没有双指标增益，不能说同时加入O已经稳定超越只加P。修复/损伤是对全部同折可比较对的归一化信用质量，Uno60采用原IPCW，不是“有多少独立患者被修复”；每项均核验 `repair−damage=ΔC`。不依据这些外折错误对定义新教师、改模型或做选择。

这些是对旧训练模型的统一**评测**，不是使用当前Efron训练协议重新验证的模型比较；不改变历史DSS训练或外折选择过程，也不生成新确认性门。

## 7. 已有调制与残差输出诊断

以下均先求cell内患者摘要，再等权汇总25个seed×fold cell；5seed复用患者，不视为独立队列。

- M7R F3平均绝对gamma为0.007913、绝对beta为0.005619；表征修正L2均值0.133894，修正/原path范数均值0.013623（约1.36%）。这是表征尺度，不能据此推断风险修正也只有1.36%、或调制可以忽略。缺权重阻断head投影与冻结FiLM-off验证。
- SPO两头输出逐cell Pearson的均值0.002044，范围−0.176060至+0.186206，说明这组输出没有普遍强线性同向相关；不证明独立、正交或条件互补。P输出SD的等cell均值0.024769，O为0.002936，总残差为0.025083；O分支幅度较小，不足以单独断言其监督无效或不存在信号。
- 诊断中记录的风险重建误差与保存分量的双精度代数重建误差分别保留于匿名JSON。它们只核输出分解，不是缺失模型权重的完整前向/梯度重放。

## 8. 本阶段可据此锁定的边界

**优势：**现有预测与诊断仍能回答完整流程表现、调制量及加性输出关系；无需重新训练即可补统一评测。**风险：**缺权重阻断历史反事实回放；旧CI存在对象与配对问题；同队列长期开发及65事件限制外推。

归因结论：M7R不能据小正点估计认定FiLM已被机制确认；SPO不能据加性双头方向性增量称为显式交互成功。缺权重是证据缺口，不是证明模块无效。当前审计不授权扩大模型、补训历史模型或重复已有bootstrap。

拓展思考／你可能想知道：完整模型胜过简单模型、同模型关闭分支变差、与正确患者配对有关，是三个不同层次的证据；后续原型应保存足够权重与回执，让这三者都能独立核验。
