# R8／SPGEL：交付范围、任务与证据边界审计

审计日期：2026-10-29。依据为两份用户实验prompt、当前冻结配方、源码、接口与公开S0文档。**当前锁定的是两个有限开发实验，不是两份prompt全部研究要求已经完成。** 本审计未读取真实R8成绩、风险向量、生存标签、模型权重或逐spot预测；实时训练、回放及导出是否完成，由对应运行负责人依据真实收据裁定，不从已有代码或合成测试推断。

对照合同：原任务单`R8_冻结空间分子证据迁移_Codex_并行Prompt.md`、`S0_S4_空间病理分子对应主线_Codex_实验Prompt.md`在本地保留，不发布；公开对应材料为[S0任务合同](S0_TASK_CONTRACT.json)、[源pilot配方](../../configs/spgel/pilot_v1.json)、[S4配方](../../configs/spgel/s4_export_v1.json)。

本文的`PARTIAL`表示仅部分研究要求已有实现、资格或交付证据；`NOT_RUN`表示本次配方没有执行该任务，不能填推测成绩；`IMPLEMENTED / DELIVERY_NOT_VERIFIED_HERE`表示入口存在，但本文没有读取私有完成收据来认证实际训练完成。三者均不能被解释为方法有效或无效。

## 1. 最重要的研究范围差距

| prompt要求 | 本轮实际锁定范围 | 当前交付边界 |
|---|---|---|
| HEST官方COAD/READ、50固定HVG、官方split及强baseline | 已审计官方代码和数据合同；现有源pilot只预测四个固定程序 | **NOT_RUN**：四程序pilot不是官方50HVG表，不能写HEST官方成绩或SOTA |
| 新CRC来源的独立供体／研究验证 | 当前仍用同一历史研究的6供体、12切片、13,011 spots | **NOT_RUN**：新独立来源尚未下载并完成逐样本配准和去重 |
| 从未使用的空间测试集 | 旧四份ST-test已有Phase22使用记录 | 不能重新叫全新、未见过的独立确认；本轮不打开它们也不会抹去历史使用 |
| 独立组织／细胞来源验证 | 三类粗组织标注及实测spot表达程序 | **PARTIAL**：不是细胞比例、细胞内在活性、病灶分割或病理因果对应真值 |
| 全套机制负对照及逐loss消融 | 六个固定源模型，关闭若干辅助loss；R8有有限错配重训 | **PARTIAL**：源线未包含全部错供体／错程序／错空间图重训，不能把包级差值归因到每个模块 |
| 新SPGEL源表征进入R8下游 | 当前R8固定使用旧`R6_S7_locked` | **NOT_RUN**：新源导出不等于已经做了新源版本的R8迁移 |

旧源数据的来源与当前原始H5AD复核是有证据的；它们支持开发数据资格，不支持独立外部泛化结论。原始完整文库为36,601基因；`13011×760`缓存仅含固定程序gene union，其中每spot实测成员746，不得称为完整RNA矩阵。[数据登记与历史测试记录](S0_DATA_REGISTRY.md)。

## 2. SPGEL：TASK-A／B／C必须分开交付

| 任务 | 当前输入与实现 | 需要交付的证据 | 状态与限制 |
|---|---|---|---|
| TASK-A：图像单支表达预测 | `ImageBatch`仅含图像特征、相对邻域几何、有效mask；`image_only_predict()`不接受真实RNA、组织真标签或深度 | 每留出供体四程序Pearson／Spearman、MAE／MSE／Huber；native signed log1pCP10K与当前source-train Z误差分表；逐section与组织分组支持数 | **IMPLEMENTED / DELIVERY_NOT_VERIFIED_HERE**。只有四程序；官方50HVG头未启用 |
| TASK-B：配对跨模态检索 | 显式`paired_mode=True`；真实ST允许进入分子分支。每留出slide按已注册行序等距取最多128个query，candidate为该slide全部spots，不按测试组织／深度筛candidate | 图像→基因、基因→图像的Recall@1/5、median rank、pair AUC；每slide候选和query数；common四分数与native编码器分别列 | **IMPLEMENTED / DELIVERY_NOT_VERIFIED_HERE**。全部六臂有common表；native仅A3/A4/A5适用。不是纯图像预测，也不是对任意候选库的完整query普查 |
| TASK-C：程序到区域定位 | 由图像单支预测每spot程序值；高活性阈值固定为当前训练供体程序90%分位数 | 留出实测ST程序作评价；Pearson／Spearman、AUPRC、阳性数、阈值与缺失支持；对应空间图数据 | **IMPLEMENTED / DELIVERY_NOT_VERIFIED_HERE**。图数据／接口存在不等于已交付可审核的heldout图或原图组织学核验 |

指标实现位于[metrics.py](../../src/spgel/metrics.py)，实际采集、分组和检索协议位于[training.py](../../src/spgel/training.py)。主TASK-A先要求同一供体四程序相关性全部定义，再计算其宏平均；之后对六供体等权、按seed分别报告。TASK-C主AUPRC同样要求四程序全部定义。常量相关性、无阳性或无阴性AUPRC为`null`，保留定义数，不替换成0。

TASK-B先在同一供体内对其两张slide等权，再对供体等权；native与common表的信息预算不同。10/20 epoch的TASK-B明确`NOT_EVALUATED_FOR_TASKB`，仅40 epoch／A0 solver-final实际进行完整候选库检索。训练hard-negative可以利用组织与训练深度分层；这些元数据不进入图像预测，也不作为heldout检索candidate过滤器。

A5的双向重读在配对API中对每个候选重新计算，而TASK-A/C部署不输入真实RNA。若A5在图像任务改善，只能先解释为配对训练改变了图像表征，不能声称部署时已由该测试spot RNA重新选择病理区域。

## 3. 源模型、辅助目标与负对照差距

当前六臂为：A0图像组织分类器＋训练组织程序均值表，A1容量匹配图像MLP，A2空间GraphSAGE，A3图像—基因对比对齐，A4程序slot＋预测组织主效应／残差包，A5在A4之上增加双向原始证据重读。[模型定义](../../src/spgel/models.py)。

| prompt中的机制要求 | 当前配方对应 | 可作出的归因 |
|---|---|---|
| 图像主任务监督 | 所有神经臂使用训练Z尺度Huber δ=1 | 可比较同协议预测；不能因有梯度便宣布真实医学对应 |
| 图像—基因对齐 | A3/A4/A5固定ITC权重0.1；基因值与真实mask保留 | 需要实际TASK-B及配对负对照证据；存在对比loss本身不是原创性证明 |
| 组织主效应／程序slot | A4相对A3增加一个预定义组合 | A4−A3是包级比较，不是逐一孤立组织CE、slot与背景分解的因果消融 |
| 双向原始证据重读 | A5相对A4增加两个方向的条件读取，保留四程序位置 | 必须看实际配对任务及相同结构错误对应重训；未做的方向／错配拆分仍为**NOT_RUN** |
| masked gene recovery | 权重0 | **NOT_RUN**；不能写其改善了信息保真或将其作为已完成机制验证 |
| matching classifier | 权重0 | **NOT_RUN**；检索评价不能替代matching classifier真实重训 |
| spatial smoothing | 权重0 | **NOT_RUN**；空间GNN存在不等于完成平滑及随机图消融 |
| 原prompt全部错误基因／错误组织匹配／随机空间图等源重训 | 未进入72-cell有限pilot | **NOT_RUN**；训练中抽hard negatives不是这些同结构错误配对模型的替代品 |
| Hist2ST、HisToGene、TRIPLEX、OmiCLIP、FOCST、HINGE、spEMO、ST-ConMa等 | 官方来源、输入、权重与许可已建资格表 | **NOT_REPRODUCED**；代码审计不等于实际复现或强baseline胜出 |

原prompt所要求的全部研究空间不必一次混入pilot，但缩小范围必须公开。冻结配方中关闭的目标及未做对照，不可在结果讨论中作为已获支持的机制。[文献资格表](S0_LITERATURE_BASELINES.md)。

## 4. 源checkpoint、实际回放和报告门槛

| 项目 | 冻结预算／实现 | 完整交付判据 |
|---|---|---|
| 六供体LODO × 两seed × 六臂 | 72 fits：12凸＋60神经；神经40 epoch、batch512、不丢尾批 | 必须有完整72份唯一`COMPLETE.json`，失败或部分单元不能自动重试并抹掉旧证据 |
| 神经更新 | LODO合计52,000；逐留出供体每神经fit为880／720／960／880／960／800 | 实际cell计数同时符合其训练行数、预锁逐供体预算及总预算 |
| 保存状态 | A0每fit一个真实solver-final；神经每fit保存10/20/40 | 12＋60×3＝192真实状态；不能给A0制造虚构epoch checkpoint |
| 源实际回放 | `spgel_source_replay.py`逐一加载实际保存状态、做前向并重算指标 | 192状态全通过；预测误差≤2e−6、指标误差≤1e−8、每unit新增fit和update均为0；只核文件SHA或原receipt不能替代前向回放 |
| 独立报告器 | `spgel_source_report.py`只读JSON metadata及绑定SHA | 缺cell、缺state、SHA不符、伪前向、重拟合或超误差均拒绝完整汇总；10/20只诊断，不挑最优epoch |
| 公开源报告 | 逐供体、逐seed、逐程序、任务A/B/C、分组及预算 | 训练学习曲线不能冒充独立评价；公开表不能包含spot预测、身份或坐标 |
| S4全源fit | 固定A1/A5、seed31，另2神经fit、2,080 updates、6保存状态回放 | 与LODO结果分开；全部合计74 fits／54,080 updates／198状态。全源训练成绩不是独立source test |

实现入口：[资格与锁](../../scripts/spgel_source_qualify.py)、[训练](../../scripts/spgel_source_fit.py)、[实际回放](../../scripts/spgel_source_replay.py)、[报告](../../scripts/spgel_source_report.py)。本审计不读取运行中的私有收据，所以不为上述真实完成数量盖章；合成测试通过只能证明报告或接口约束的特定行为。

## 5. S4接口：实现存在与迁移完成不是一回事

| prompt接口 | 当前实现对应 | 仍须验收的边界 |
|---|---|---|
| `encode_image_spots` | 同slide真实μm邻域，输入逐行L2 UNI2，返回`[B,N,4,128]`程序embedding及Z/native预测和mask | 当前患者／切片token必须有实际几何资格，不能因维度相同就假设TCGA与source物理支持一致 |
| `image_only_program_map` | 图像单支程序map与有效mask，无RNA输入 | 图输出及独立原图形态QA需实际交付；程序map不是病灶真值 |
| `encode_bulk_genes` | 仅A5分子编码器适用；必须传当前R8训练分区基因Z、原始mask、gene-order／membership SHA和独立bulk scale合同 | 不能复用source spot scaler；Z形式相同不证明spot/bulk生物学对齐；coverage不是医学置信度 |
| `route_bulk_to_wsi` | A5条件query读取已编码128D patch values，返回程序embedding与attention | 调用者必须提供合格的单WSI patch-value表示；高层`encode_image_spots`的四程序输出不能不加适配直接当作单个128D patch value。实际TCGA适配、multi-slide聚合与新R8评价仍待独立资格 |
| encoder权重／scaler／gene集／模型维度／heldout结果 | `spgel_source_export.py`输出`source_encoder_manifest.json`及冻结状态，绑定代码、membership、顺序、source检查点、回放和LODO结果 | 导出前必须有72／192及S4 2／6真实完成回放。manifest不是独立泛化证明 |

接口位于[interface.py](../../src/spgel/interface.py)。A1没有学得的分子编码器或四个原生程序slot；其重复展开的baseline embedding被显式标记，不能包装成与A5相同的四语义token。向下游切换新source必须另开`R8_SOURCE_SPGEL_V1`，不得在当前旧源R8结果中静默换权重。

## 6. R8：当前有限配方与整份prompt的对应

本节只核源码和合同，不读取任何真实R8评价数值，也不向SPGEL反馈模型胜负。

| 要求 | 当前对应 | 范围与未交付部分 |
|---|---|---|
| 固定强主效应 | 复用CP＋训练交叉拟合RNA512 scalar及same-teacher β；逐元素核对offset | 无新教师／β重拟合；读取旧512行全局容器与后续只消费outer1训练409人必须分开披露 |
| 冻结证据 | 旧`R6_S7_locked`及随机未ST监督状态；六个partition×source-kind feature family | 源曾接受source训练，不是新SPGEL源；source实际输入为观察值统计的gene Z(log1pCP10K)，target为当前inner-train中位数填补矩统计Z并恢复原始mask |
| 有限候选 | 9臂×3分区×3名义seed＝81凸拟合；另18份零fit基线引用 | 数值训练问题为27，三个确定性seed重复不能作为独立随机稳健性证据 |
| 低容量风险保护 | 完整风险集Efron＋ridge λ=1；谱缩放＋系数球，γ=0.1 | 只保护训练SD比，不保证每患者修正、患者对排序或验证集SD |
| 主要交互 | R8-6：冻结image64及gene64逐程序训练PCA2，再做每程序完整2×2双线性 | rank≤2是2×2输入的自然上限；R8-6不是RNA条件Patch重读，也不是额外学习出的低秩结构 |
| 真／错对应与容量控制 | R8-8a随机source、R8-8b错程序、R8-9错患者交叉项、R8-10等32系数加性二次项，均真实重训 | R8-9保留本人加性G、mask、图像与offset，只交换交叉项query；它检验R8-6，不证明R8-5的Patch检索配对有效 |
| RNA条件图像读取 | R8-5使用冻结条件图像证据；wrong-query为冻结诊断 | **PARTIAL**：没有R8-5同结构错query真实重训；冻结替换不能替代分支训练必要性检验 |
| R8-7训练侧正交化、反向`E^(G|H)`等扩展 | 未进入当前固定9臂 | **NOT_RUN**；不能根据看到当前结果再追加并混入同一独立评价表 |
| 真实保存风险回放及修复／损伤 | 81实际模型回放、18引用、90个直接配对、27确定性检查；逐partition绑定候选组件和repair−damage，再做分区内num/den聚合 | 只对回放完整集合报告；修复／损伤是归一化concordance credit，不是被修复患者人数 |
| PFI次要终点 | 冻结DSS风险评价，不参与模型／source选择 | 不能把PFI改善或某个seed正分当DSS主要机制成立 |

R8报告器要求全部证据验证后才写公开产物，`R8_RESULTS.json`作为最后完整标志；预计9个匿名诊断分片加8个结果／表格文件，共17项。当前审计不通过目录中存在一个文件来认证完整运行。[R8入口](https://github.com/fallenplum/diffusion/blob/4d9147cd29f2346cf17b90d4ffcb69f1bc54db8c/scripts/bgb_en_r8_run.py)、[报告实现](https://github.com/fallenplum/diffusion/blob/4d9147cd29f2346cf17b90d4ffcb69f1bc54db8c/scripts/bgb_en_r8_report.py)。

prompt另外要求`R8_SOURCE_INTERFACE_AUDIT.md`和`R8_REPRODUCE.md`专门文档；仅有核心代码、本文边界说明或报告模板，不能代替这些最终交付。结果、训练ledger、逐fold配对及决策需在实际81拟合回放完成后输出；本审计未读取其真实完成件，状态为**DELIVERY_NOT_VERIFIED_HERE**。

## 7. S0两份公开文档的隐私与两线隔离审查

只读检查了两份S0文档。模式扫描未发现私有绝对目录、服务器／凭据、目标患者标识或精确spot barcode；人工复核确认保留的是公开研究链接、匿名支持数、程序名、公共代码／数据checksum及私有收据basename，没有原始表达、患者标签、风险预测、逐spot坐标或权重。`barcode`等泛称、公开GEO accession与数据集样本链接不是项目患者身份表；不能仅凭关键词命中判定泄漏。

| 文档 | 本轮SHA256 | 上述四类扫描命中 |
|---|---|---:|
| S0_DATA_REGISTRY.md | `79b509cce2aa26f930fa4ee8ee0d28932208867ae716e02446a70aaa4e671986` | 0 |
| S0_LITERATURE_BASELINES.md | `d3a4026fda738784928716e6c5fa1f58b5f3a2b9e1b1d61509df41b0e0148ea0` | 0 |

这不是整个仓库或未来生成报告的无限保证；每次发布仍须扫描实际待提交文件。两线隔离已由固定配置、各自LOCK和不同数据API表达：source测试不输入TCGA终点或R8风险；R8不反馈DSS/PFI结果修改source。本文只核接口和规则，没有据R8胜负决定source基因集、loss、epoch、候选或预算。

## 8. 本轮发现并修复的报告budget schema漂移

只读审计发现：实际`pilot_v1.json`已加入`neural_optimizer_updates=52000`和逐供体更新数，而旧报告器仍对较早7键budget字典作全等比较，导致真实配置被错误拒绝。先前匿名fixture只包含旧字段，原13项测试没有覆盖这次配置扩展；不能用其通过记录声称实际配置已经兼容。

经主执行器明确授权，仅修改报告器和其测试，未改模型、训练器、源数据、loss或配置：保留全字段精确比较，加入52,000及逐供体880／720／960／880／960／800约束；每神经cell同时检查行数推导与预锁更新数，汇总强制52,000。补充直接读取当前实际配置的测试，以及总更新、逐供体更新、额外未注册budget键三类篡改拒绝。**17项报告测试通过，23.24秒**；这是合成／配置资格证据，不是17个真实源实验。

修复后SHA：报告器`9b153223d7fc42675c69f88169b8970f4c6f2a2dc56594aaa316a4d6942b16e3`；报告测试`422702f854b5526316ebd3a2a29f1a3e99eb5c6c0e64f3917a09edabac689f7f`。主执行器已通知源负责人暂停LOCK、刷新科学文件manifest再继续；不得在已经绑定的科学LOCK之后无提示替换报告文件。

## 9. 尚未完成的外部数据与论文级证据

当前E盘仍不可用，新增大文件按用户要求等待写入E盘，不擅自改到服务器或其他盘。GSE267401的human／mm10元数据冲突尚需实际基因ID／参考核验；GSE283052的H&E配准、供体和重复研究身份尚未资格通过。官方HEST当前manifest、每slide donor、50HVG列表与split也必须在实际下载版本上绑定，不能直接沿用口头“各两位供体”。

下一步的数据工作是解除文件位置阻断、执行只读身份／配准／测量审计，再锁官方或新独立来源的评估合同；不是按现有开发正分挑一套更有利的测试集。优势是当前pilot范围、原始证据与数值预算可审计；主要风险是源供体少且复用、任务／信息预算不同，以及源—目标单位和取材支持未自然等价。对本文源码与合同对应关系置信度高；对未读取的实时训练完成状态和未资格的新来源可用性，本文不作成功判断。
