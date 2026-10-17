# Phase22 数据来源与有限外部空间队列审计

日期：2026-10-18。范围：官方数据源、最新元数据、供体隔离、空间足迹、冻结特征资格及下载规模。本文不报告新模型成绩，也不将数据可下载等同于实验资格全部通过。

## 1. 已核验的来源与固定版本

| 来源 | 固定版本 | 当前核验范围 |
|---|---|---|
| MahmoodLab/hest | `7e8d5a0b0aace41d8c8ec0f6ecea80e4ad2a61ec` | 15,608 个文件的官方索引、HEST v1.3.0 供体表；数据仓库 gated |
| MahmoodLab/hest-bench | `b039b5965ed7092f280d4f336da7972ef9a10e1a` | 296 个文件的官方索引；COAD/READ 的表达与 split 已由主进程下载并校验；该仓库 ungated |
| Zenodo 原始 CRC Visium | 官方记录 `7760264` | 小型官方 API 元数据、14 个原始 Space Ranger/H&E 压缩包清单、校验值及许可；未在本子任务下载压缩包 |

可追溯文件保存在 `E:/diffusion/dataset/spatial_bridge/metadata/`。HEST v1.3.0 CSV 的 SHA256 为 `5703d7169c7a31aa4cc74a900df9179326e0cc51bda617a1cf7ac179b6ed5d63`；hest 索引为 `8266563eddc74a4f3d4304555f7c0fdb1fcba19dfd3439e3ed20e0f335610dc2`；hest-bench 索引为 `fb6338d13eaae31b7d82fd4abbf4ad247ceff940d77cd8662db9c3a3c57747f1`。

官方入口：[HEST 仓库](https://github.com/mahmoodlab/HEST)、[HEST 数据集](https://huggingface.co/datasets/MahmoodLab/hest)、[HEST-Bench 数据集](https://huggingface.co/datasets/MahmoodLab/hest-bench)、[Zenodo 原始数据记录](https://zenodo.org/records/7760264)。原始病例、切片和条形码映射只进入私有 manifest，不写入本报告。

## 2. 不能沿用旧 benchmark 的供体计数

最新 CSV 包含 1,276 份空间样本，其中人类 Bowel 84 份；64 份的 OncoTree 编码为 COAD、READ 或 COADREAD。这些数字是切片数，不能当作独立患者数。

公开 benchmark 的 COAD 四切片，历史文档和旧 issue 曾被描述成两供体；最新 v1.3.0 把其中来自同一高分辨率 CRC 研究的三份 Xenium 切片分别标为三个不同供体。其余一份来自另一 10x preview 采集页，供体字段缺失。READ 四切片仍明确分属两个供体。因此，当前应记录 **五个可核来源内供体 + 一份供体尚不确定的 preview 切片**，不能继续把八切片称为八供体，也不能未经纠正地沿用“四供体”描述。

该纠正具有独立的原始研究支持：Nature Genetics 2025 的样本表区分了各 CRC 供体，并说明 Xenium、Visium 与 Visium HD 是同一 FFPE block 的串行切片。其跨 assay 的同供体样本必须整体锁入一个分区。[原始研究及样本表](https://pmc.ncbi.nlm.nih.gov/articles/PMC12165841/)

HEST 作者还确认过两项问题：旧版本中该研究的供体标注有误，v1.2.1 已修正；另有两份 Visium HD 记录是完全重复的原始样本。去重必须沿原始来源及供体链进行，不能只检查 HEST ID 是否不同。[维护者对错误与重复的确认](https://github.com/mahmoodlab/HEST/issues/126)

历史供体描述仅作为差异审计证据保留，不再控制本轮 split。[旧 benchmark 供体讨论](https://github.com/mahmoodlab/HEST/issues/87)

### 固定 CSV 与单切片 JSON 的字段冲突处理

本轮实际下载的三份单切片 JSON 仍保留旧的相同 `patient` 值，但其 `subseries` 已指向具体的不同供体或同供体不同 assay 样本；固定 v1.3.0 CSV 也据此分组。原研究 Table 1 列出三个不同 CRC 供体，Xenium 方法明确说明对这三个供体的组织进行检测，表中还明确标出其中一个供体同时接受 Visium v2 测量；其中晚期病例的分期还与相应 JSON 的疾病描述一致。因此，供体分组采用**固定 CSV 的供体表，并由 JSON 内的具体样本描述及原研究交叉核验**，不是因为字段冲突而忽略供体身份。同供体的 Xenium 和 Visium 样本继续共用供体分组，不能人为算成两个独立患者。[原研究 Table 1 与 Xenium 方法](https://pmc.ncbi.nlm.nih.gov/articles/PMC12165841/)

原始 JSON 的字节保持不变。资格审计另外记录原始值、采用值、两种来源的 SHA256 和裁定依据；只允许这三项经过核验的完整样本描述，其他身份冲突继续拒绝。Issue #87 记录的是 2025 年 1 月对旧分组的确认，**它不是纠错证据**；纠正依据是后续固定 CSV、原始研究和样本描述。该身份裁定不自动解决测量足迹、单位或其他元数据冲突，也不改变已锁定的分区。

## 3. 有限候选集与可执行分区

### 3.1 可直接开始资格核验的两研究子集

| 采集研究 | 全部相关样本 | 本轮最小候选 | 供体与分区安排 |
|---|---:|---:|---|
| CRC 共识分子亚型空间研究，Zenodo 原始来源 | 14 Visium 切片；7 供体，其中 6 个 Cancer、1 个 Healthy | 12 个 Cancer Visium 切片 | 4 个供体/8 片 train，2 个供体/4 片 validation；每供体两张串行切片不拆分 |
| 高分辨率 CRC 研究，Nature Genetics 2025 | 多 assay，存在同供体串行切片 | 3 Xenium + 1 Visium | 3 个供体/4 片 test；其中同供体的 Xenium 与 Visium 一并留在 test |

训练/验证供体仅按预定字符串盐与来源内供体编码的 SHA256 排序分配，不查看表达数值、病理结果或生存标签。主进程已形成私有正式 split lock：`E:/diffusion/dataset/spatial_bridge/metadata/CRC16_ST_DONOR_SPLIT_LOCK_PRIVATE.json`；盐为 `Phase22-CRC-donor-lock-v1:`，以该 lock 的确切供体分配为准。文件下载、空间足迹及模型资格仍需逐项通过，split 锁定不等于训练已完成。

审计阶段另保留私有候选文件 `tmp/phase22/study_donor_split_candidate_private.json` 供追溯；候选盐与正式 lock 不同，**该早期候选文件不控制训练**。正式 lock 包含 16 份切片、9 个来源内可核供体、48 个 metadata/ST/patch 文件的固定路径、字节数、上游 SHA256、像素尺度、分区与不确定性状态。两个文件均不得进入公开交付。

**证据上限：这是一项两研究、供体隔离的可执行外部 ST 评估；不满足理想的三独立研究确认。** 外部 test 同时跨研究和技术平台，必须分别报告 Xenium 组织与仅一份 Visium 组织，不能把平台差异解释成纯研究迁移效果。test 的三个供体也不足以给出精确、高功效的跨人群推断。

Zenodo 原始研究明确使用七位个体的两张串行切片；有独立病理医师的 spot compartment 注释，但本轮不把这些组织类别替代分子实测或 DSS 结局。[原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10781769/)、[作者代码与数据说明](https://github.com/alberto-valdeolivas/ST_CRC_CMS)

### 3.2 第三研究候选暂不混入正式分区

HEST 中 COLON MAP 包含 41 份 Visium 切片，但其 patient 字段全部空白/空格。`subseries` 提供若干重复采样线索，**不能仅以其字符串前缀声称已证明供体身份**。须补取 HTAN 的 participant–biospecimen–file 映射，与 HEST 小型 metadata JSON 逐项对照，再排除同个体的多 ROI、连续切片和不同发表库镜像。

这 41 片还混有癌前病变，不能全部标为 CRC。原始 atlas 的多区域设计也明确存在同个体多个空间区域。[原始 Cell 研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC10756562/)、[作者分析仓库](https://github.com/Ken-Lau-Lab/spatial_CRC_atlas)

不因第三研究的映射尚未完成而停止两研究 A 阶段，但不得据此解锁三研究确认或独立 CRC 生存验证。

## 4. 下载规模及 E 盘布局

所有下载以 `E:/diffusion/dataset/spatial_bridge/` 为本机数据根；metadata、raw、patches、features、processed 分开。服务器只迁移后续运算需要的文件，常用冻结权重与特征可保留。

| 有限下载对象 | 原始字节数 | 说明 |
|---|---:|---|
| 公共 benchmark 的 8 份表达文件 | 338,294,749 | 不含 split CSV，不能自动视为供体正确的现成划分 |
| 公共 benchmark 的 8 份 RGB patch H5 | 3,725,456,784 | 是图像 patch，不是 UNI2 特征 |
| 上述表达 + patch | **4,063,751,533** | 3.78466 GiB；不需要整库下载 |
| 两研究 16 切片正式候选的 metadata + ST + patch | **5,982,730,274** | 约 5.57 GiB；与已下载公共 bench 文件的重复内容须按 SHA 去重复用 |
| Zenodo 原始 14 个 Space Ranger/H&E 压缩包 | **1,449,570,508** | 仅原始包；不下载大体积 deconvolution、中间对象等无关包 |
| Zenodo 独立病理 spot annotation 压缩包 | 702,660 | 可追溯独立形态标签候选，不是当前 A 阶段必需输入 |

这些是固定版本索引/API 给出的文件传输字节数，不包含解压、模型权重、缓存和新特征。未知网络速度时不编造完成时间。下载后记录实测速率、校验耗时和剩余量，再更新 ETA。

## 5. 尚未发现官方可直接复用的 HEST UNI2 特征

对上述固定版本的两个官方仓库完整文件索引检索，均无 HEST UNI2 embedding/features 发布目录。HEST 的官方 benchmark 在本地从 RGB H5 调用 patch encoder，生成 embeddings 缓存。[官方提取代码](https://github.com/mahmoodlab/HEST/blob/main/src/hest/bench/benchmark.py)

因此，第一轮可用的路线是：**已提供的 HEST 112 µm RGB patch → 官方冻结 UNI2-h → 新建 HEST 特征缓存**。现有 TCGA UNI2-h 缓存不能替代这些外部组织的空间局部特征。

UNI2-h 官方 card 指定输出 1536 维、224 像素输入及相应 ImageNet normalization；提供的现成大型特征集是 TCGA/CPTAC/PANDA，并未声明其中包含 HEST。模型文件需要独立授权，HEST 数据授权不能替代它。[官方 UNI2-h model card](https://huggingface.co/MahmoodLab/UNI2-h)

特征资格必须核验：官方模型、权重 SHA、实际输出维数、224 像素输入、完整预处理及物理视野。相同 1536 维数不证明与别的 encoder 或旧的面积平均缓存兼容；不把 GigaPath、UNI v1、CONCH 或其他模型冒称为 UNI2-h。

## 6. 图像与表达足迹实际不同

HEST 官方定义的图像 patch 为 112 × 112 µm、224 × 224 像素，即重采样到 0.5 µm/px；这不意味着表达也在整个 112 µm 方形区域被测序。[官方 benchmark 文档](https://hest.readthedocs.io/en/latest/tutorials/4-Running-HEST-Benchmark.html)

对已经下载的八个真实 H5AD 本轮额外检查：

- 四份 Xenium 组织的 `obsm/spatial` 为全分辨率 H&E 像素坐标；乘以 v1.3.0 的 MPP 后，各方向中心网格间距均为 100 µm（浮点舍入范围内）。官方 Xenium pooling 默认也是 100 µm 正方形。
- 四份 READ Visium 组织的最近邻中心距离中位约为 99.9998–99.9999 µm；其测量捕获范围是名义直径 55 µm 的圆形，不能把中心间距当捕获直径。
- 因而首版应明确记录 **图像 112 µm 方形、Xenium 表达 100 µm 方形、Visium 表达 55 µm 圆形**；本次实际发布文件的捕获形状仍需与各 metadata JSON/原始处理历史核对。
- 对 Visium 旋转网格不能以单坐标轴相邻 unique 值之差估计中心间距。本轮使用二维 KD-tree 最近邻距离；空间邻域必须在真实 µm 坐标上建立。

[官方 reader 与 pooling 实现](https://github.com/mahmoodlab/HEST/blob/main/src/hest/readers.py)

### Xenium 的 55 µm 字段与 100 µm 聚合网格冲突

本轮三份正式 Xenium 原始 JSON 的 `spot_diameter` 均为 55 µm；H5AD 绘图用 `uns/spatial/ST/scalefactors/spot_diameter_fullres` 乘以 MPP 后也为 55 µm。这些原始字段予以保留，不能静默改成 100。仅凭中心间距为 100 µm，也不足以证明计数捕获范围是 100 µm 方形。

独立核验的官方源码固定在 commit `3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3`：`XeniumReader.read` 将默认 `spot_size_um=100` 传入分子聚合函数；`get_indices_chunk` 对每个 transcript 的物理位置按该尺度取 `floor`，得到二维方形 bin，再按 bin 与基因进行计数。返回坐标是各 bin 的中心，表达矩阵是相应 bin 内 transcript 的汇总。[固定 reader 与聚合定义](https://github.com/mahmoodlab/HEST/blob/3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3/src/hest/readers.py#L915-L1226)

绘图元数据有独立来源：同一 reader 调用 `register_downscale_img` 时未传入 `spot_size`，该函数默认用 55 µm 生成可视化标记直径并写入 H5AD scalefactor。因此，绘图用的 55 µm 不能作为已聚合 Xenium transcript 的圆形捕获范围。[固定绘图注册函数](https://github.com/mahmoodlab/HEST/blob/3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3/src/hest/utils.py#L1133-L1157)

实际三份 H5AD 的二维最近邻距离均为 100 µm（数值舍入范围内）；`array_col/array_row` 与物理坐标的逐轴拟合斜率也均为 100 µm/格，最大坐标残差低于 `1.3e-10` µm。这与上述方形 bin 定义相符。基于源码定义与实际网格共同证据，本轮仅对固定来源的这三份 Xenium 文件记录**名义 100 µm 方形聚合单元**，保留字段冲突和裁定记录；不对未知 Xenium 文件开放通用覆盖。此结论仍不等于已经对原始 transcript 表逐分子验证总量守恒，也不等于图像的 112 µm 视野。

## 7. count、探针及聚合风险

对已下载四份 Xenium H5AD，本轮查得每份均有 541 个存储列。三份来自高分辨率 CRC 研究的组织中，每份有 119 个名称明确的负对照、空白或未分配 codeword，留下 422 个候选生物基因；preview 组织有 116 个此类列，留下 425 个。**541 不能直接写成 541 个生物基因。** 仍须用正式基因表/原始 panel 对候选生物基因再确认，并按 assay 建立 measured mask。

进一步只比较已发布 gene 词表、未根据 test 表达值选择：三份正式 Xenium 的 422 列候选集合完全相同；四份 READ Visium 的 36,601 列集合完全相同；两者共享 **419 个符号**。剩余三个 Xenium 项是特定 clonotype 的定制探针，不能作为普通 HUGO 基因直接塞入通用桥接字典。正式 G 还需与 TCGA 原始 assay 的可测基因和正式源 panel 对照；1024 是上限，不要求把 419 补到 1024。所有空间组织保留 measured mask，未测值不得作生物学零。

Xenium 存储矩阵为无负值的整数；本轮最大单元 count 为 2,694–3,673，未见 `uint16` 饱和值。这只排除明显的整数饱和迹象，不能证明没有上游过滤、质量阈值、重采样或丢计数。READ 的 CSR float32 非零值为整数、无 raw/layers 的独立计数副本，同样不能仅凭整数性声称“未经任何预处理的完整原始计数”。`var` 中的 log1p QC 字段也不表示 X 本身已经 log1p。

正式资格需读取原始说明、检查 X 的非负/整数性与单位、按条形码对齐图像，保存包括过滤在内的历史。Pseudo-bulk 从符合资格的 count 聚合后再变换；不能平均 log 表达，不把未测 gene 当零。

### 固定词表下的测试组织缺测

在正式锁定的 378 维分子词表中，外部测试的一份 Visium 组织仅包含其中 370 个基因，缺少 8 个。其缺失来自发布文件的未覆盖或上游过滤，具体原因尚未确定，不能把这 8 项当作测得的零，也不能看过测试数据后删除它们或重新选择字典基因。训练和验证组织仍必须完整覆盖固定词表。

该 Visium 组织保留在阶段 A 外部评估中，只在已观测基因集合上条件化组成并计算相应空间指标；私有缓存始终保存完整 378 维词表及显式 `measured_mask`，存储填充值不具有表达测量含义。它不具备阶段 B 当前完整组成投影的测量资格，须以 `full_dictionary_measurement_projection_eligible=false` 单独记录并排除于 B 投影评价。阶段 B 的外部测试因此只覆盖三位供体的 Xenium 组织；阶段 A 仍保留跨 assay 测试，但必须同时披露测量覆盖和条件化评价的限制。公开报告仅给出缺测数量，不发布基因列表或患者级测量记录。

### Visium HD 的特定已知风险

HEST 的开放 issue 报告 Visium HD pooling 使用重复花式索引 `+=`，可能漏累加，尚未在本轮确认已发布文件是否重新生成。该报告指向 **HD bin pooling**，不能泛化成 Xenium/Visium 文件必定错误，也不能将讨论中的数据损失比例冒作本次核验结果。[问题记录](https://github.com/mahmoodlab/HEST/issues/143)

源码中 Xenium 的 Dask histogram/`np.add.at` 是另一条累加路径，没有这一重复索引的同一写法。但本轮尚未把每份发布矩阵与原始 transcript 表做逐分子守恒对照，也未证明不存在其他代码问题。有限正式候选不包含 HD；以后若纳入 HD，须从原始 bins 修复聚合并验证 count 总量守恒后再用。

## 8. 许可和预训练重叠

| 对象 | 官方许可/条款 | 本轮处理 |
|---|---|---|
| HEST 数据、HEST-Benchmark 与官方库 | CC BY-NC-SA 4.0；以具体源及声明为准 | 保留来源、固定版本和上游许可，不公开重分发患者数据或地图 |
| Zenodo 原始 CRC 数据记录 | API 声明 CC BY 4.0 | 原始文件留 E 盘，可追溯到作者公开记录；代码仓库 GPL-3.0 与数据许可分开登记 |
| 10x 原始公开数据 | 相应页声明 CC BY 4.0 | 下载前核对具体 dataset 页，不把镜像仓库许可自动替代原源 |
| COLON MAP/HTAN 在 HEST 表的许可字段 | CC BY-NC-ND 4.0 | participant mapping、源文件和使用条款未完成前，不并入正式候选 |
| UNI2-h 权重及相关条款 | CC BY-NC-ND 4.0，非商业学术用途，独立 gated 授权，不允许公开复制权重 | 冻结使用；权重及衍生特征不推送本项目公开 GitHub |

**UNI2-h 对候选 ST 组织的逐样本预训练重叠状态是 UNKNOWN。** 官方 card 描述机构来源和预训练规模，但没有可公开逐片比对的训练清单。不同来源、不同发表年份或“外部数据”字样都不足以证明完全未进入视觉基础模型预训练。该未知必须区别于本轮 ST adapter 的明确 train/validation/test 供体隔离；前者未确认不能掩盖，后者必须严格执行。

## 9. 推进门槛与本审计结论

**可以推进：** 将有限两研究子集的 metadata、count、图像与权重资格完成后，训练冻结 UNI2 上的 A0/A1/A2 小适配器；按锁定供体评估，报告两研究与跨 assay 的限制。第三 HTAN 研究映射可并行补齐。

**当前不能据此宣称：** 三研究确认、无基础模型预训练重叠、完整 raw molecule 守恒已证明、外部独立 CRC 生存验证已完成、空间预测等于第四个实测模态、或局部结构改善已带来 DSS 增量。

优势：官方小子集足以提供真实局部测量配对；已有串行切片支持取材压力测试；数据量约 6 GB，不需要下载整个 HEST 或全量 WSI。

限制：供体仍少；训练与测试平台分布不同；图像足迹与测量范围不等；原始转录本守恒和 TCGA bulk 单位映射未完成；UNI2 逐片预训练重叠不可验证。因此，这次数据审计不会单独解锁 Phase22-C 的 120 次生存拟合。
