# R6 外部 ST 来源与 TCGA 患者独立性只读审计

日期：2026-10-28。范围：R6 开发源的 6 名供体、12 个切片，以及目标队列的 512 个患者登记标识符。此文件是私有审计说明，不是新的实验结果或患者排重证书。

## 结论

**外部研究与组织采集来源链：SUPPORTED，置信度高。跨库受试者不重叠：UNVERIFIED。**

12 个开发切片可追溯到 HEST 中同一 Zenodo 研究。正式原论文记载组织来自 HTCR Foundation，具有知情同意及慕尼黑大学医学院、巴伐利亚州医学协会伦理审批。这支持该数据具有明确的外部组织采集流程；原文没有提供与本项目 TCGA 队列的共同受试者登记或非重叠证明。[原论文 Methods：Collection of CRC samples](https://www.nature.com/articles/s41698-023-00488-4)

原始来源的 6 个供体标识符与 TCGA 的 512 个标识符，直接匹配和去符号、转大写后的匹配均为 0。**这些标识符属于不同命名空间；0 次字符串匹配不能证明 0 名共同患者。**未找到共同 subject registry、匿名标识符交叉表或可比对的统一受试者标识。因此不能使用“已证明与 TCGA 患者无重叠”或“已完成受试者排重”等表述。

## 来源链与范围

| 项目 | 本轮实际核验 |
|---|---|
| 开发源 | 12 切片、6 供体；Stage A/B 开发切片集合一致 |
| 原始研究登记 | `Zenodo_7760264`，全部开发切片相同 |
| HEST 仓库 | `MahmoodLab/hest`，revision `7e8d5a0b0aace41d8c8ec0f6ecea80e4ad2a61ec` |
| HEST 原始字段 | 人类肠道、Visium、fresh-frozen；来源链接为该 Zenodo 记录和原预印本 |
| 公开正式论文 | Valdeolivas 等，npj Precision Oncology 8, 10 (2024)，DOI `10.1038/s41698-023-00488-4` |
| 论文主研究范围 | 7 名 CRC 患者、每位两份连续切片；R6 使用其来源记录中已锁的 6 名供体开发子集，不把整篇论文的其他参考数据视为本次开发源 |
| 既有开发缓存 | 主分析 13,011 spots；原始开发 H5AD 行数 17,392 为既有资格报告的数值，本轮未重读表达矩阵验证 |

原论文的数据可用性指向概念 DOI `10.5281/zenodo.7551712`；本地来源记录指向版本 DOI `10.5281/zenodo.7760264`。本轮只读取官方 Zenodo 小型 JSON 元数据，确认两者同属一个概念记录：前者解析到后者，后者的 `conceptdoi` 为前者。二者不是两个独立队列。[版本记录及研究说明](https://zenodo.org/records/7760264)、[概念 DOI](https://doi.org/10.5281/zenodo.7551712)、[作者代码仓库](https://github.com/alberto-valdeolivas/ST_CRC_CMS)

## 原始供体字段与标识符核对

1. 本机开发源 12 份原始 HEST JSON 均验证了既有收据的文件 SHA。每份 `patient` 与锁中 `patient_source_field_verified` 完全一致；锁中供体键也与“研究键 + 原始 patient 字段”的组合完全一致，12/12 通过。原始 `Sample ID` 依既有精确命名/单个文件别名规则，与官方 Zenodo JSON 所登记的 14 份原研究 Space Ranger ZIP 比对，开发集 12/12 均匹配；仅读取文件登记，没有下载 ZIP。
2. `patient_source_field_verified` 的实际类型是字符串，不是布尔标志。不能把它不等于布尔 `True` 解释为供体字段未核验。
3. HEST 汇总 CSV 中相同 12 个开发行的 patient 字段与原始 JSON、锁逐项一致，12/12 通过。原始 JSON 及汇总表均未发现 TCGA/GDC/crosswalk/统一 subject registry 字段。
4. 服务器仅打开 12 个开发 H5AD 的元数据结构。`uns` 顶层仅有 `spatial`；其已遍历小型元数据节点只有 24 个标量形状数据集，没有可读取的字符串型供体、study 或 accession 字段。没有读取图像、表达矩阵、坐标数组、layers 或其他测量数组。
5. 目标登记仅从既有队列容器读取 `patient_ids` 成员：512 个唯一值，均符合 TCGA 患者条码格式；按既有协议以换行连接、UTF-8 编码计算的顺序 SHA 精确匹配既有固定值。
6. 原始 6 个来源 patient 字符串与目标登记直接匹配 0，规范化匹配 0；来源元数据字符串中 TCGA 条码模式计数 0。此结果只排除了观察到的显式同名条码，不排除同一个人被不同系统重新编号。

## 可复核哈希

| 对象与算法 | SHA256 |
|---|---|
| Stage A 资格 manifest，本轮服务器字节 | `6f40062efacecc8a43a17ae6fa34ffa22f3b7b98ab2a5db874e1ab5a691da27f` |
| Stage B prepared manifest，本轮服务器字节 | `9cc2f840b8010eb1881831e971682780461607437bb220b1a6d7a0a39efafeb0` |
| 原始 ST 划分锁，本轮本机字节 | `ca560935440e117c5d70d55ce14ea1c5f524bbad2f2ab261d0bf7ef7a63a08fa` |
| HEST 汇总 CSV，本轮字节，553,819 bytes | `5703d7169c7a31aa4cc74a900df9179326e0cc51bda617a1cf7ac179b6ed5d63` |
| 12 份开发原始元数据 SHA 排序后 JSON 紧凑序列的 SHA | `b3018cda4d538647d1f90d5cd2042aeb16570072daa4a4aff3f797537f87b25e` |
| 6 个锁内复合供体键排序、去重后 JSON 紧凑序列的 SHA | `9747cebb475a8c0934debae3d67c173cc666eaf36b1c5d7da6cd9f52d68ab742` |
| 512 患者顺序，换行连接 UTF-8 | `76a144d544b7461d669487625855dc7de65f115a6a2f65d7a20002b2a8e5008f` |
| HEST provenance 下载计划 | `609b7532dec44dce5310d6cfb219e8c8ddebb16285069145362de20348bf2af2` |
| HEST provenance 下载收据 | `d2e6937c9c3c9e00399fb3eccab62330e303820d83b83ffba90727515f706ec3` |
| 官方 Zenodo JSON 响应，两个入口返回相同字节 | `5b1a994631bd30fecdba731c684093b4178a19f8047c784697429ba1ed275c21` |
| 本机 `R6_0_DATA_QUALIFICATION.json` | `baa94a649c2ebe6785d2eff23578986b7172cd047fa4302aad4788d771ae4aef` |
| 本机 `R6_0_SOURCE_DATASET_QUALIFICATION.json` | `86280c3a2551ae15fa66adba8f81aa03ee7f30214d1fbe1108ae2462fd658a6a` |
| 本机 `R6_0_SOURCE_CACHE_REPLAY.json` | `064f2482e1e7ad9dc7516f28e047ce5000726f77764ba778da5e8bbb91e66056` |
| 本机 `R6_0_SOURCE_CACHE_NUMERICAL_REPLAY.json` | `075a29a7ffc013f4a628a2da54fa045ea261ba7926aaf45f1056a496aeb964a6` |

Stage A/B 大型数据包 SHA 仅引用既有 manifest/资格证书，本轮未重新读取数据包；本轮没有以其历史“未读 TCGA/DSS”标志推导受试者互斥。若将这些 SHA 写入后续公共报告，应保留这个区别。

## 本次访问边界

- 读取既有划分锁元数据以筛选 12 个开发切片；其原文件包含 16 个切片登记。未打开 4 份原 ST test H5AD 或 test 表达文件，也未用 test 数据进行分析。
- 实际打开开发 H5AD 元数据 12 份；读取目标 `patient_ids` 成员一次后又为确认原协议哈希及原始 patient 字符串比对读取一次。没有调用会加载 RNA/CNA/临床/结局的通用原始数据 loader。
- DSS/PFI 数值读取 0；表达矩阵读取 0；模型/权重文件读取 0；训练、优化器步骤及 GPU forward 均为 0。
- 没有修改原始数据、已有源码、配置、锁、训练目录或运行进程。仅新增本说明文件；未部署程序，未下载数据文件。

## 优势、风险与后续所需最小证据

**优势：**来源链接、版本关系、供体分组字段和目标登记哈希都有实际字节/元数据核验，能防止把不同来源或技术重复切片误当独立供体。

**风险：**命名空间不同且缺跨库主体交叉表；HTCR 的采集与伦理说明不等于证明任何一个受试者从未参与 TCGA。6 名开发供体已在历史阶段重复使用，当前供体留出仍不能称为全新、从未接触的外部确认；此项与 TCGA 是否重叠是两个不同问题。

若后续结论必须要求“与 TCGA 512 名患者严格不重叠”，最小新增证据应是原研究作者或数据管理者出具可审计的非 TCGA 招募/贡献确认，或在授权范围内提供共同标识规则下的匿名主体交叉核对。现有公开元数据不足以完成该证明；不需要为此新增模型、读取结局或扩大训练。

建议对外措辞：**“采用有明确 HTCR/Zenodo 来源的外部 ST 开发数据；未观察到显式 TCGA 标识符匹配，但缺少共同受试者登记，患者级不重叠尚未核验。”**
