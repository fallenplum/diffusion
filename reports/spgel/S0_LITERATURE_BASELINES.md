# SPGEL S0：官方基准、强对照与前作边界

核验日期：2026-10-29。**这是可复现性审计，不是复现实验成绩。下表所有尚未真实执行的方法均为NOT_REPRODUCED，不填Pearson、C-index或排名。** 既有R0–R7/SAMI结果保持原身份，不冒充新主线独立实验。

## 1. 检索与核验方式

已实际尝试Google搜索URL，当前Web工具对这两个Google结果页返回不可访问；随后使用会话Web搜索发现候选，逐条打开官方论文、出版页或官方GitHub。双引擎论文搜索脚本也已实际同时调用OpenAlex与AnySearch：本子任务通用histology/ST/contrastive查询返回OpenAlex 15条、AnySearch 0条，交叉验证0；父进程的HEST查询同样无双源命中。**不能把尝试两个引擎写成两个引擎均支持。** 元数据仅作候选发现，方法事实以下列primary来源为准。

官方GitHub文件通过绑定账号的只读插件读取；记录的是本次读取文件的Git blob SHA，不把它冒称仓库commit SHA。真正运行前仍须锁repo commit、文件/权重SHA、配置与依赖环境。Google不可访问、Nature/OUP部分跳转受限和缺文件均如实保留，不按二手聚合页面补造结果或许可。

## 2. HEST官方Ridge必须先准确复现

HEST官方README当前说明50 HVG、112×112μm、0.5μm/px、Pearson与PCA256 Ridge。官方原始论文表、当前README榜及后续代码版本不能混作同一协议；本轮选定哪个版本必须先锁定，再执行。[官方仓库](https://github.com/mahmoodlab/HEST)、[任务说明](https://hest.readthedocs.io/en/latest/tutorials/4-Running-HEST-Benchmark.html)。

| 现行实现项 | 本次实际代码核验 |
|---|---|
| split | 读取发布的`train_i.csv`/`test_i.csv`与各sample_id/patches_path/expr_path，不做spot随机拆分 |
| HVG | 默认`var_50genes.json`；不是本轮按test重新选择 |
| X处理 | `StandardScaler → PCA(n_components=256, random_state=seed)`只fit train；同pipe transform test |
| y处理 | `load_adata`按barcode与官方gene表取子集；`normalize=True, smooth=False`实际为float64后`log1p`，没有`normalize_total`调用 |
| 头 | `Ridge(solver='lsqr', fit_intercept=False, max_iter=1000, random_state=seed)` |
| alpha | trainer实际赋值`100/(X_dim*n_genes)`；256×50时为0.0078125。benchmark中的alpha参数没有传入该调用，trainer也重赋值，不能只改YAML便声称改变了实际正则 |
| 指标 | 每gene跨当前test spots的Pearson，再平均50gene，最后汇总官方fold；独立donor表另列，不把slide数当donor数 |
| NaN | 当前trainer不把NaN自动修成0；复现时保留并单列常数目标/常数预测规则，不能静默丢失败gene后宣称官方值 |

最容易出错的是把S1四程序的`log1p(CP10K)`当作官方50gene处理。这是两种任务合同：四程序按完整count文库计算；官方gene主表按锁定官方实现，不静默统一尺度。若另做CP10K敏感性，另表标注自定义设置。

本次文件身份：

| 官方文件 | Git blob SHA（文件对象） |
|---|---|
| [benchmark.py](https://github.com/mahmoodlab/HEST/blob/main/src/hest/bench/benchmark.py) | `f7179acb4462dfc84b949eb16f8728f72387af6a` |
| [trainer.py](https://github.com/mahmoodlab/HEST/blob/main/src/hest/bench/trainer.py) | `397649041ed227c82c61ea4ef67fb033d6550465` |
| [st_dataset.py](https://github.com/mahmoodlab/HEST/blob/main/src/hest/bench/st_dataset.py) | `57655967cc9d540c315806cec31948aa2c62d23f` |
| [bench_config.yaml](https://github.com/mahmoodlab/HEST/blob/main/bench_config/bench_config.yaml) | `06fb3234d227fc03efa1a6d7d78f846e608522a5` |

HEST当前默认COAD是Xenium、READ是Visium，官方各四切片；真实donor需实际manifest核定。历史v1.3.0身份纠正与未知preview供体见 [S0数据登记](S0_DATA_REGISTRY.md)。此处尚无本轮HEST模型运行成绩。

## 3. 必含方法的资格矩阵

“代码可读”不等于“依赖安装/权重授权/输入兼容/数值复现已通过”。基础视觉编码器或数据不同的方法不能直接与UNI2固定特征小头混排后称同预算SOTA。

| 方法 | 已核primary来源与方法角色 | 实际输入与本项目兼容性 | 权重/许可/复现状态 |
|---|---|---|---|
| HEST Ridge＋PCA256 | [官方代码](https://github.com/mahmoodlab/HEST)；固定encoder线性probe | 冻结embedding→官方50gene；可用UNI2-h，但须官方图像/特征单位与split资格 | HEST CC BY-NC-SA 4.0；本轮NOT_REPRODUCED，作为官方首要入口 |
| UNI2 linear / ridge / MLP | [UNI2-h card](https://huggingface.co/MahmoodLab/UNI2-h)；主效应强对照 | 同一合格1536D source特征与预处理；MLP新增参数必须另计 | 已有源特征不证明新头已拟合；NOT_TRAINED，权重不公开 |
| tissue-only / additive / mean-neighbor / GraphSAGE-GAT | 本项目明确工程对照，不冒充某篇论文官方实现 | 组织真标签仅训练与分层，测试用图像预测组织；邻域只同slide真实μm坐标 | 新pilot待固定合同真实训练；已有SAMI比较不重命名为新实验 |
| Hist2ST，Briefings in Bioinformatics 2022 | [论文DOI](https://doi.org/10.1093/bib/bbac297)、[官方代码](https://github.com/biomed-AI/Hist2ST)；CNN＋Transformer＋图与ZINB | 官方forward需要RGB patch、坐标、adjacency；不是直接把UNI2向量接入就算官方复现 | README链接HER2+/cSCC训练权重到Synapse；根目录`LICENSE`请求404，仅表明该路径缺失，许可仍待核验；NOT_REPRODUCED |
| HisToGene，原2021预印本 | [作者论文](https://doi.org/10.1101/2021.11.28.470212)、[官方代码](https://github.com/maxpmx/HisToGene)；图像ViT空间依赖 | 官方forward输入flatten RGB与坐标；如改成UNI2需单独标改编版 | 已读官方MIT LICENSE；未验训练checkpoint或运行，NOT_REPRODUCED |
| BLEEP，NeurIPS 2023 | [官方论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/df656d6ed77b565e8dcdfbf568aead0a-Abstract-Conference.html)、[官方代码](https://github.com/bowang-lab/BLEEP)；双模态对比与reference表达检索 | 训练配对图像/表达，测试query图像依赖训练reference；不可将test RNA放入reference | 官方Apache-2.0 LICENSE已读取；尚未运行或验证对应权重，NOT_REPRODUCED |
| TRIPLEX，CVPR 2024 | [CVF论文](https://openaccess.thecvf.com/content/CVPR2024/html/Chung_Accurate_Spatial_Gene_Expression_Prediction_by_Integrating_Multi-Resolution_Features_CVPR_2024_paper.html)、[官方代码](https://github.com/NEXGEM/TRIPLEX)；target/neighbor/global多分辨率 | 需要多尺度局部与邻域图像及slide上下文；现有单一112μm embedding不足以自动重现所有输入 | 当前README有2025实现修订，不能混原论文与修订版；根`LICENSE`404，许可待核；NOT_REPRODUCED |
| OmiCLIP / Loki，Nature Methods 2025 | [论文DOI](https://doi.org/10.1038/s41592-025-02707-1)、[官方代码](https://github.com/GuangyuWangLab2021/Loki)；图像—gene-sentence预训练/检索/表达预测 | 使用自己的图像与转录组双编码空间，不能把UNI2 embedding冒充其image embedding；CRC/HEST预训练样本重叠须审计 | README明确[HF权重](https://huggingface.co/WangGuangyuLab/Loki)，下载/授权未验；README为BSD-3-Clause并附非商业学术限定，不能按无限制BSD理解；NOT_REPRODUCED |
| FOCST，PLOS Computational Biology 2026 | [正式论文](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1014762)、[官方代码](https://github.com/mattthuang/FOCST)；UNI＋空间GAT＋对比/表达重建 | 论文为可训练UNI视觉路径，并有reference式预测；冻结UNI2改写不能冒充官方端到端设定 | 本次官方README只有一句方法说明；训练入口/权重/数据合同及代码许可尚需逐文件确认；NOT_REPRODUCED |
| HINGE，CVPR 2026 | [作者arXiv及会议接收注记](https://arxiv.org/abs/2603.19766)、[官方代码](https://github.com/donghaifang/HINGE)；冻结CellFM、图像条件masked生成 | 官方数据接口要求UNI与CONCH两路spot特征及CellFM gene词表；单UNI2 cache不自动同构 | 代码有训练/数据检查入口，README仍说checkpoint/处理特征/评测脚本待链接；CellFM另需依赖/许可；根LICENSE404，待核；NOT_REPRODUCED |
| spEMO，Nature Biomedical Engineering 2026 | [正式书目信息](https://pubmed.ncbi.nlm.nih.gov/41644824/)、[官方代码](https://github.com/HelloWorldLTY/spEMO)；多模态FM表征/检索与多任务 | 不是仅四程序回归小头；按其任务分别复现，不能拿多模态测试输入成绩充IMAGE_ONLY预测 | 有demo与分任务代码，基础FM另有gated要求，README不承诺全部结果checkpoint；根LICENSE404，待核；NOT_REPRODUCED |
| ST-ConMa，Briefings in Bioinformatics 2026 | [论文DOI](https://doi.org/10.1093/bib/bbag535)、[官方代码](https://github.com/DMCB-GIST/ST-ConMa)；contrastive＋matching预训练 | PathoDuet image与C2S/Pythia410M gene-sentence，HEST v1.1.0来源；与UNI2-only输入不同，需预训练重叠审计 | README有训练/评估代码及Google Drive checkpoint数据链接，但注明需访问申请，未实际获得；根LICENSE404，待核；NOT_REPRODUCED |

本次读取的README文件blob SHA分别为：Hist2ST `d721798c13413bc839b0f09cb793b6d2ba8201e5`；HisToGene `3a341e073bf7c7b1691899e57cfa1793472b5a71`；BLEEP `06604017db1268a62790882ee57a53678a35e016`；TRIPLEX `05e70bd1b6a43159ab60da72e659c002d34a7cdf`；Loki `2d19487dcaf1a7df1b19ae6e9d5f1400b07af805`；FOCST `a99eb2b71cb67b14c1979dc2093cd846ebf7c452`；HINGE `6c281fcfc8713efb43bea52e8b40f1011161cdf5`；spEMO `bc4b3e0c45fa07e68e3c0a318077efd650e23cd2`；ST-ConMa `80c1711b08ccbde3e3c3f4ce9e90b388e208fa22`。只记录实际读取，不说明代码已执行。

## 4. 新颖性和信息预算的否证边界

OmiCLIP已经有image–transcriptome预训练和双向检索；BLEEP/FOCST已有图像—基因对比及reference预测；TRIPLEX已有多尺度；spEMO已有多模态表征与alignment任务；ST-ConMa还直接覆盖contrastive＋matching。因此“图像—基因对比＋多尺度＋图结构＋masked loss”的组合本身**不足以建立SPGEL原创性**。这些是现有方法覆盖范围，不由本报告虚构性能优劣。

SPGEL应实际检验的差异是：保留可验证程序位置，拆开图像可预测的组织主效应与局部程序残差，控制同组织/同深度难负例和空间自相关，并用严格image-only任务、配对检索及独立来源分别验证。只有超过同输入、容量和训练预算匹配的强对照及错误对应，才有机制证据；这个设计也不能自动成为医学因果或细胞来源识别。

额外发现但不擅自扩入pilot：2026已有HiST（ICML，分层稀疏格点）、HESpotEx及其他新方法。与同名HiST（Advanced Science）必须按题名/作者/代码区别；没有核完输入/许可/权重和数据重叠前仍NOT_REPRODUCED，不为追逐当期榜无限增加训练。[ICML HiST作者代码](https://github.com/wwyi1828/HiST)、[HESpotEx正式论文](https://www.nature.com/articles/s43588-026-00992-0)。

## 5. 进入真实复现前的逐项清单

1. 冻结官方repo commit及对应文件、checkpoint SHA；README blob身份不能代替完整源码版本。
2. 记录数据study/donor/slide/spot，公开split/HVG来源、训练/测试平台与物理支持；查重及来源重叠未知须保留。
3. 运行官方入口，先验证测试图像分支不接收真实RNA/tissue/depth；reference型模型只允许训练reference。
4. 保留官方RGB／FM／多尺度设定；任何UNI2替换另命名“同特征改编”，不得并排宣称严格官方复现。
5. 完成实际训练、预测文件、指标重算、参数/凸解/更新/GPU时间计数后，才将状态从NOT_REPRODUCED改为COMPLETE。
6. HEST官方Pearson主表、四程序定位、paired检索及R8生存迁移分表，不混不同测试信息预算，也不以R8选择源模型。

优势是官方代码和输入差异均有具体证据，可避免弱baseline或跨协议混排。风险是部分权重尚未可得、许可未齐、官方版本存在演化、研究供体很少且可能与FM预训练重叠。对已读代码的事实置信度高，对尚未执行方法的性能无新增结论；本报告不宣称SPGEL优于这些方法或达到SOTA。
