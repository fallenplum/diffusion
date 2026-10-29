# SPGEL S0：源数据、身份与任务资格登记

日期：2026-10-29。状态：**既有源开发数据具有可追溯资格，且本轮主执行器已完成当前服务器原始H5AD与缓存的重新核验；官方基准及新增独立来源尚未完成本轮数据资格。** 本文不是训练结果、独立外部验证或重新下载成功声明。

## 1. 本次访问边界与启动判断

本子任务只读本项目的匿名源域资格、协议及历史空间评估报告，并检索公开官方数据/代码元数据；未读取本项目DSS/PFI标签、患者风险预测、权重或原始表达数组，无模型前向、反向或拟合。没有打开历史四份ST-test H5AD，也没有下载新的图像、表达矩阵或大模型。

| 工作 | 当前资格 | 启动边界 |
|---|---|---|
| 既有6供体源pilot | 历史来源、身份、完整count与mask合同有实际核验收据；本轮当前服务器重核PASS，见3.1 | 仍须绑定本轮split、配置、预算与任务合同；数据复核不等于训练或独立确认完成 |
| HEST官方COAD/READ | 官方任务、输入和现行代码已核验；当前数据manifest/splits/HVG列表仍须本轮绑定 | 先核官方文件和供体分组；旧Phase22自定义split不能冒充官方split |
| 新独立研究 | GEO列出候选配准材料，但尚无本轮逐样本配准及测量资格 | 未通过时为NOT_QUALIFIED，不编造heldout成绩 |
| 新增大文件下载 | 本次子进程访问E盘返回“驱动器不存在” | 依用户要求等待E盘恢复；不擅自把新原始数据下载到服务器或其他盘 |

两条研究线分离：SPGEL不以R8的DSS成绩选择基因集、源架构、loss或checkpoint；R8只接收冻结且带SHA的源接口。现成R6 S7能否供R8使用由R8的独立接口审计决定，不替代新源模型的source-heldout证据。

## 2. 已有开发源：6供体、12切片，不是13,011名患者

来源链为HEST中的同一Zenodo CRC研究，原研究是Valdeolivas等的CRC空间分子亚型研究；概念记录7551712与版本记录7760264属于同一个研究，不能算两套独立数据。原论文与作者仓库可交叉追溯。[正式论文](https://www.nature.com/articles/s41698-023-00488-4)、[Zenodo版本](https://zenodo.org/records/7760264)、[作者代码](https://github.com/alberto-valdeolivas/ST_CRC_CMS)。

| 对象 | 已有实际资格数值 | 解释 |
|---|---:|---|
| 开发供体/切片 | 6 / 12 | 每位两张连续切片；必须同供体分组 |
| 全部已标注spot | 17,392 | 不等于主任务支持集 |
| 固定主任务spot | 13,011 | tumor 4,040；stroma 5,237；tumor–stroma mixed 3,734 |
| 排除spot | 4,381 | 原因见Stage A审计；不能当阴性或临时重新纳入 |
| 身份唯一键 | section＋精确barcode | 全局不同barcode字符串仅3,634；不能用全局barcode字典合并切片 |
| 冻结局部视觉特征 | 1,536D UNI2-h | 逐行float32 L2，零范数拒绝；不是重提特征完成声明 |
| 原始参考基因 | 36,601 | library size按完整count词表求和，不能只用四程序成员求分母 |
| 四程序注册gene union / signed edges | 760 / 765 | 并非每个成员均测得，也并非互斥集合 |
| 每spot当前可测程序成员 | 746 | 自然零计数仍是测得；缺失成员由真实mask区分 |
| 当前邻域 | 300μm，最多16（含自身） | 最少4、中位16；跨切片边0；真实断边不能补成稠密网格 |

原count已验证有限、非负、整数及逐行身份对应；这不证明上游从未过滤。缓存保留完整文库深度、测量mask、坐标和程序目标。source目标为 `mean(log1p(CP10K(up)))−mean(log1p(CP10K(down)))`，空down项为0；非空方向缺测不可冒充零。四目标的训练尺度只能由当前source-train拟合。

| 程序 | 注册up/down | 实测up/down | 支持边界 |
|---|---:|---:|---|
| iCMS2 | 308 / 279 | 300 / 276 | 外部上皮状态分数，不是当前队列已验证的分型 |
| iCMS3 | 74 / 54 | 71 / 54 | 保留原方向，不能按本轮相关性替换成员 |
| FAP-associated top25 | 25 / 0 | 25 / 0 | 按既有外部表固定构造，不是实测CAF比例 |
| MARCO-associated top25 | 25 / 0 | 25 / 0 | 不因含SPP1就静默改名为SPP1程序 |

源组织真标签是病理医师标注的tumor/stroma/mixed三个粗组织类，不是淋巴细胞类、单细胞来源或组织比例。TASK-A/C测试输入不得含这些真标签，只能使用图像预测的组织概率。来源定义和排除规则详见 [Stage A协议](../sami_surv_stage_a/PROTOCOL.md) 与 [Stage B协议](../sami_surv_stage_b/PROTOCOL.md)。

## 3. 可复核的历史来源绑定

以下是本轮实际读取的既有公开资格文件所记录的SHA；本子任务没有重新读取大型原始包或派生数组。当前服务器存在性、原始字节和复载检查必须在新的S0运行合同中另外记录。

| 对象 | SHA256 |
|---|---|
| Stage A合格manifest | `6f40062efacecc8a43a17ae6fa34ffa22f3b7b98ab2a5db874e1ab5a691da27f` |
| Stage A派生数据 | `f067dfff45ff7c75ec4b61c93567b8218e11fedc9d68c2338e393935151e177e` |
| Stage B合格manifest | `9cc2f840b8010eb1881831e971682780461607437bb220b1a6d7a0a39efafeb0` |
| Stage B派生数据 | `b5a2a44862123f629314cb28ad66abae437d7d08e923fb6128f82f7e9059c025` |
| 四程序定义 | `f8031c03cbb9ec2409d72c799129a6de74f080860afbc81cb30fb187d2560143` |
| R6原基因缓存manifest | `ef611b45d8d821bd7489131f3bab60f3b314f28f5228b60f2c170044c58848d3` |
| source gene union顺序 | `0e56b530399d863077cff1caba105b97ce49b5fa7dfbc728a1603bd29f62be85` |
| 原病理标注archive | `2468ad1208468277eaa7693cbaf92735e76d52207e87d534e0ac8fe77fcacb47` |

直接证据入口：[R6源数据资格](../bgb_en_r6/R6_0_SOURCE_DATASET_QUALIFICATION.json)、[R6完整count核验](../bgb_en_r6/R6_0_DATA_QUALIFICATION.md)、[Stage B数据审计](../sami_surv_stage_b/DATA_AUDIT.json)。既有原基因缓存约235.56 MB，开发12份H5AD历史传输总量642,926,336 bytes；这不是本轮新下载量。

### 3.1 2026-10-29当前服务器重新核验

本轮主执行器已实际读取12份开发集原始H5AD，按完整36,601基因文库重新计算library size与四程序目标，并与当前sourcecache的16项约定字段核对；结果为PASS，16字段精确一致，程序分数最大绝对误差为`6.661338e-16`。当前sourcecache manifest的实际SHA256仍为`ef611b45d8d821bd7489131f3bab60f3b314f28f5228b60f2c170044c58848d3`。

必须区分两个物理对象：原始H5AD提供完整36,601基因文库，当前union缓存的`gene_counts`形状为`13011×760`；后者是固定程序成员缓存，**不是完整RNA表达矩阵**。完整文库分母由原H5AD重放核实，不能从760列union重新求和来替代。

私有核验收据名为`S0_FRESH_SOURCE_REQUALIFICATION.private.json`，SHA256为`c67cb60255b25692e3da81510d53cf640df9f754518b11a40acf935060a1520c`。该收据由主执行器生成并回报；本资格子任务没有另行读取原始数组或复制私有收据到公开仓库，公开文档不列实际私有目录。本次重核未消费DSS、PFI或历史ST-test，新增拟合数为0；它证明当前源文件与约定缓存一致，不证明源模型已通过新任务或源—TCGA语义迁移有效。

## 4. 历史测试与独立性不能重置

六位开发供体已经在Phase22、SAMI和R6等阶段使用。重新LODO是新的固定候选开发比较，不是全新独立确认。

四张历史ST-test也不能改称“从未见过”：Stage A协议明确记录其Phase22历史评估；`reports/phase22_spatial_bridge/B_TEST_SUMMARY.json`状态为COMPLETE，评估范围覆盖三位Xenium供体；另一个Visium切片因固定词表部分缺测不具备该投影任务资格。本次S0只读其既有匿名汇总，未打开原四份H5AD。R6/R7自身“本轮未读测试”与“历史上从未使用”是两个不同命题。[历史来源与身份裁定](../phase22_spatial_bridge/DATA_SOURCE_REVIEW.md)、[Stage A使用历史](../sami_surv_stage_a/PROTOCOL.md)。

同一原研究在HEST、GEO、Zenodo镜像出现不构成新增独立来源。R6已有研究来源链支持外部组织采集，但其source与TCGA患者身份命名空间不同，缺共同主体crosswalk；不重叠状态仍为UNVERIFIED，不能以零字符串匹配证明零共同患者。[R6来源审计](../bgb_en_r6/R6_SOURCE_COHORT_PROVENANCE.md)。

## 5. HEST官方基准：先锁版本，再锁实际split

本轮在线核验的官方文档列COAD四张Xenium切片、READ四张Visium切片；均为图像局部112μm输入预测50个官方固定HVG，不能写成八名独立患者。[官方任务文档](https://hest.readthedocs.io/en/latest/tutorials/4-Running-HEST-Benchmark.html)。

已有Phase22固定v1.3.0元数据审计支持：COAD中三张来自已纠正供体标识的高分辨率CRC研究，分别对应三个供体；另一个preview来源供体未明。READ四切片分属两供体。该数值来自已锁历史元数据审计，不是本子任务重新下载当前CSV后的新身份认证；执行基准仍须绑定当前实际CSV、单样本JSON、官方split与原研究。[既有逐源裁定](../phase22_spatial_bridge/DATA_SOURCE_REVIEW.md)、[维护者纠错记录](https://github.com/mahmoodlab/HEST/issues/126)。

官方split用于“官方设定”表；若其按历史版本出现同供体跨fold，则保留其原始复现并明确披露，再另列严格donor-disjoint版本，不能静默改官方表或把spot随机拆分冒充供体外。官方50 HVG必须用发布列表，不在本轮test上重选。现行代码的目标加载与四程序CP10K不是同一处理：`normalize=True`的默认非平滑路径实际仅float64后log1p；详见文献/基线审计。

## 6. 新独立研究候选与当前阻断

| 候选 | 官方已证实 | 当前不能假定 | 下载与后续资格 |
|---|---|---|---|
| GSE267401 | 4份CRC FFPE/CytAssist登记；存在表达矩阵、barcode、图像、坐标和scalefactors附件；样本字段明确patient id，原发/转移成对 | 不能把4样本当4供体；GEO同时写Homo sapiens与mm10，物种/参考基因组描述冲突尚未实际features核验 | 官方RAW.tar显示358.6 MB；待E盘可用后只取必要附件，核gene IDs/配准/患者配对和MPP |
| GSE283052 | 12份CRC Visium组织登记，法国Lille OrgaRES来源；RAW.tar列CSV/JPG/JSON/MTX/PNG | 有PNG不等于完整H&E已与spot逐行配准；12组织也不能仅凭标题自动认证12独立供体 | 官方RAW.tar显示668.1 MB；待核每样本组织图、坐标、scalefactors、donor/study及与已有研究重复 |

依据：[GSE267401](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE267401)、[其官方样本元数据中的物种/assembly冲突](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8265211)、[GSE283052](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE283052)。GEO系列页本次直接打开触发反爬检查，使用其官方索引返回及可读样本元数据核验；未把第三方数据库的Visium版本或细胞类别当作裁定依据。下载显示量不是解压/缓存总量，未测网络速度时不提供ETA。

新来源必须按study/donor/slide/spot四级去重；对同一人的原发/转移、串行切片或跨平台切片整体分组。只有完成图像—坐标—barcode—表达一致性、物种与测量mask核验后，才能锁独立研究holdout；不得先看预测好坏再决定来源是否“独立”。

## 7. 空间支持、许可和证据上限

既有source UNI2-h来自224×224像素、名义0.5μm/px的约112μm方形图像上下文；Visium捕获区域为名义55μm圆，中心间距约100μm不是捕获直径。Xenium发布文件还存在绘图55μm字段与100μm方形分子聚合定义的历史冲突，应保留裁定，不对未知文件通用覆盖。TCGA WSI patch网格没有因同为UNI2-h而自动取得相同物理支持；`source_to_tcga_geometry_manifest.json`须由独立无结局接口审计生成。[原几何资格](../phase22_spatial_bridge/DATA_SOURCE_REVIEW.md)、[官方定义](https://hest.readthedocs.io/en/latest/tutorials/4-Running-HEST-Benchmark.html)。

HEST库/数据/benchmark声明CC BY-NC-SA 4.0；UNI2-h权重独立受其gated和非商业条款约束，数据授权不能替代模型授权。既有Zenodo数据记录为CC BY 4.0，但不因此覆盖所有镜像/权重/图片的许可。原始像素、表达、坐标、身份表和权重不进公开Git。UNI2-h对这些组织是否有逐样本预训练重叠仍UNKNOWN。[HEST官方许可说明](https://github.com/mahmoodlab/HEST)、[UNI2-h官方card](https://huggingface.co/MahmoodLab/UNI2-h)。

优势是现有完整count、真实mask、独立粗组织标注和切片内空间图具有可追溯来源，可支持有限的source开发pilot。风险是供体少且已被复用、混合细胞spot、跨平台及取材尺度差异、模型预训练重叠未知。对历史资格字段和本轮读取的官方代码内容置信度高；对尚未下载/配准的新来源可用性置信度低／未核验，不宣称源任务或迁移已经成功。
