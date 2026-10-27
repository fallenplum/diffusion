# BG-BEN 文献、公开实现与新颖性审计

核查日期：2026-10-27。状态：**六个指定名称均找到对应原论文；三项找到并核对公开作者源码，另三项未核实可用的作者代码。未复现任何论文的原始成绩。**

## 检索证据与限制

实际调用已安装的“双引擎论文搜索”脚本，对 SurvPath、QCAgent、SliceWorld、ME-Mamba、MCAMamba、TriBind-Mamba 分别并行请求 OpenAlex 与 AnySearch。检索收据为 `LITERATURE_SEARCH.json`，含查询、返回数量、`sources`、`cross_validated` 及已验证官方源码的提交和 SHA256。六次 AnySearch 均无可用结果，**没有一篇达到双引擎交叉验证**；不能把零结果推断为论文或代码不存在。默认沙箱第一次请求受套接字权限阻断，获准的网络执行后 OpenAlex 正常返回。

Google Search 入口实际尝试：`https://www.google.com/search?q=QCAgent+SliceWorld+TriBind-Mamba`；当前浏览工具返回不可访问。随后通过可用网页搜索发现候选，并打开作者预印本、出版机构、会议页面与作者仓库核对。搜索元数据仅用于发现：例如 ME-Mamba 作者在 OpenAlex 中写作 Xia Liu，原论文与期刊摘要页为 **Xiaoyu Liu**；SliceWorld 检索还返回无关古地理文献，未作为医学证据引用。

以下是截至本次访问的核验，不是穷尽性专利或全球文献检索。`未核实官方代码` 表示在所读原文、可用性说明与名称+GitHub 检索中未取得可信实现链接，不等于断言代码永远不公开。

## 六个指定工作的实际内容

| 工作与发表信息 | 原文和实现确认的机制 | 代码资格及与本项目关系 |
|---|---|---|
| **SurvPath**；Jaume、Vaidya、Chen、Williamson、Liang、Mahmood；CVPR 2024，11579–11590，DOI `10.1109/CVPR52733.2024.01100` | 通路内基因经各自的可学习网络形成 token；保留 pathway→pathway、pathway→patch 与 patch→pathway 三类注意力，省略 patch→patch。 | **作者代码已核验**。`model_SurvPath.py` 和 `models/layers/cross_attention.py` 已读取；并非只有单向通路查询。原头为离散生存 logits，与本轮 Efron Cox/强临床锚点存在协议差异；改写版本须称 SurvPath-style，不能称严格官方成绩复现。 |
| **QCAgent**；Wang 等；MICCAI 2026，LNCS 16878，460–470，DOI `10.1007/978-3-032-38059-3_44`；预印本 `2603.01647` | 诊断检查清单、报告批评、文本检索 patch、局部视觉核验、迭代修订；任务为 WSI 病理报告生成。 | **作者代码已核验**。`pipeline.py` 调用 CONCH 检索、SVS 裁剪、Patho-R1 及报告 QC；依赖第三方模型和原始图像。可以启发“重新查证原始证据”，**不构成 RNA/CNA 双向生存网络或细胞来源监督**。 |
| **SliceWorld**；Yuanhe Tian、Yan Song；arXiv `2605.24371v1`，2026-05-23 | 按真实 CT z 轴编码切片序列，区分 anatomy/lesion/uncertainty 状态，经 world-token 接口作未来切片预测和配对报告生成。 | **作者代码已核验**。`SliceWorldCore` 实际包含 sequence encoder、factor heads、world projector 和 future-prediction heads。其有序 CT 轴及报告监督不能直接移植为无序基因扫描；latent lesion intervention 也不等于物理移除病灶。未核实正式会议/期刊录用。 |
| **ME-Mamba**；Chengsheng Zhang、Linhao Qu、Xiaoyu Liu、Zhijian Song；*Computerized Medical Imaging and Graphics* 129，102733，2026，DOI `10.1016/j.compmedimag.2026.102733`；预印本 `2509.16900` | 病理、基因组两个单模态专家及协同专家；注意力引导扫描；OT 局部对应及 MMD 全局分布约束。 | **未核实可运行作者代码**。预印本写明将公开；正式摘要可核验。预印本 HTML 的 journal 字段与正式 CMIG 记录不一致，引用采用正式 DOI。其单模态保留与 attention/SSM 分工已有先例。 |
| **癌症生存 MCAMamba**；Haiyan Cui、Xiaofei Liu、Weiping Ding、Weixing Wang；*Science Progress* 109(2)，2026-05-28，DOI `10.1177/00368504261456869` | 平行模态编解码、Mamba/SR-Mamba 和双向 cross-attention；对四个 TCGA 癌种进行生存实验。 | **未核实可运行作者代码**。原文可用性说明未提供已核实的源码仓库。`github.com/Dmygithub/MCAMamba` 属于另一篇 **2025 遥感分类**同名工作（DOI `10.1109/TGRS.2025.3618301`），不得错用为癌症官方实现。 |
| **TriBind-Mamba**；Yijiang Ding、Yuanwei Jing、Wanhan Zhang；*Frontiers in Oncology* 16，1825286，2026-08-04，DOI `10.3389/fonc.2026.1825286` | Clinical narratives、冻结文本编码、稀疏 gene–pathway MLP、双模态/三模态 binding 及 SSM；区分共有与模态特异表征。 | **未核实可运行作者代码**。原文/补充材料可用性陈述没有给出本次核实的仓库。临床文本、通路 token、三模态 Mamba 组合已发表；不能把这三个要素简单并列当 BG-BEN 首创。其COADREAD人群与本项目512人协议不同，旧绝对成绩不能直接排名。 |

原始证据链接：

- SurvPath：[原论文](https://arxiv.org/html/2304.06819)、[作者仓库](https://github.com/mahmoodlab/SurvPath)、[注意力实现](https://github.com/mahmoodlab/SurvPath/blob/3f73ddd6705ec67d643020c5bb04fb13f9f382cc/models/layers/cross_attention.py)。
- QCAgent：[出版机构](https://link.springer.com/chapter/10.1007/978-3-032-38059-3_44)、[完整预印本](https://arxiv.org/html/2603.01647)、[作者实现](https://github.com/cs-w0r1d/qcagent/blob/af57e36f82083d76089f1ec5ab5ed69edba0f79d/pipeline.py)。
- SliceWorld：[原论文](https://arxiv.org/html/2605.24371)、[作者实现](https://github.com/synlp/SliceWorld/blob/bdbef659f7c05b3f6dba77c075a637e1b54efaf9/sliceworld/models/sliceworld.py)。
- ME-Mamba：[正式出版页](https://www.sciencedirect.com/science/article/pii/S0895611126000364)、[作者预印本](https://arxiv.org/html/2509.16900)。
- MCAMamba：[癌症生存原文](https://journals.sagepub.com/doi/full/10.1177/00368504261456869)、[同名遥感工作的出版记录](https://ieeexplore.ieee.org/document/11194194/)。
- TriBind-Mamba：[原文](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2026.1825286/full)。

## 病理 Mamba 与空间分子对应的既有工作

**MambaMIL**（Shu Yang、Yihui Wang、Hao Chen，MICCAI 2024）已将 Mamba 用于病理 MIL，并提出 sequence reordering。其[原论文](https://papers.miccai.org/miccai-2024/paper/3255_paper.pdf)和[作者代码](https://github.com/isyangshu/MambaMIL)可访问。该事实支持把无坐标序列版本作为必要对照；“使用 Mamba 处理长 patch bag”不能作为新贡献，重排序也不能自动等同真实二维组织结构。

**HEST-1k**（Jaume 等，NeurIPS 2024 Datasets and Benchmarks）已经提供形态—局部表达配对及从图像预测表达的基准；[正式论文页面](https://papers.nips.cc/paper_files/paper/2024/hash/60a899cc31f763be0bde781a75e04458-Abstract-Datasets_and_Benchmarks_Track.html)及[作者库](https://github.com/mahmoodlab/HEST)可核验。因此“增加外部 ST 监督”本身已有基础，新增机制须解释它如何改善当前独立患者下的证据读取，而不是只提高训练表达相关。

**STFlow**（Huang 等，预印本 `2506.05361`）[原文](https://arxiv.org/html/2506.05361)已组合 whole-slide 局部空间 attention 与表达联合分布建模。**CGL-Linear**（Nguyen 等，预印本 `2609.32857`）[原文](https://arxiv.org/html/2609.32857)将 slide 均值与 slide 内变异区分，说明空间恢复实验须报告区域内变化，不能只凭供体/切片均值判断成功。这里只用于排重和任务设计；本轮没有复现这两个模型，也未把预印本当独立医学验证。

## BG-BEN 能检验的候选贡献

**当前没有已被本轮实验确证的新颖机制。** 组件级排重结论置信度高（原文及上述关键源码可见）；“尚未发现完全相同组合”的置信度中，不能写成“首次”。

| 候选主张 | 必须满足的实验支持 | 优势与风险 |
|---|---|---|
| 固定强 CP 和 RNA 主效应后，跨模态重新读取原始局部/基因证据仍提供增量 | 同分区 F1/F2、单向、双向、普通消息控制；保留相同 value 的 query-only 错配与 attention-assignment 对照；两指标各自报告 | 优势：把主效应与交互问题直接对应。风险：读取权重也可编码另一模态主效应，胜弱加性模型不够。 |
| 外部局部分子弱监督提高正确患者—程序—区域读取，而非仅编码 donor/组织类 | F11 对 F5，真配对对 wrong-donor/spatial-shuffle；困难负样本；ST 与 TCGA 编码空间一致 | 优势：降低65事件学习局部对应的负担。风险：Visium spot 是混合测量，且尺度/平台迁移不能变成细胞来源真值。 |
| 分子多模式保真与有真实拓扑的病理 SSM 有可归因的分工 | O0 direct 风险保留；O3/O4/O5/O6/O9；P3 对 P5/P6；容量、训练预算和RNA风险保真诊断 | 优势：可直接找出信息损失环节。风险：高维先验投影仍有损；随机顺序多向 SSM 只能称顺序鲁棒控制，不能宣称基因生物序列。 |
| 可拒绝的双向交互形成有用、幅度受控的增量 | null使用率、双方向有效梯度/风险幅度、修复/损伤、真实重训关闭分支 | 优势：允许不发生交互。风险：null是模型拒绝程度，未校准成医学置信度；更多门控不自动构成新理论。 |

## 本轮实现与引用边界

`src/bgb_en/omics.py` 是本项目受控实现，**不是 SurvPath、ME-Mamba、MCAMamba 或 TriBind-Mamba 官方复现**。其稀疏聚合保留每通路多个可学习基因投影；mask 与 coverage 显式保留；RNA 与预先分离的 CNA gain/loss 采用不同权重。Set Attention 是通路集合上的标准关系算子，不是基因因果网络。

`sequence_backend='mamba'` 需要官方 `mamba_ssm.Mamba`；依赖不可用会明确报错。`sequence_backend='ssm'` 明确实现实数对角输入依赖递推，名称为 `DiagonalSelectiveSSM`，不得在结果表写成官方 Mamba。枚举、反向与固定随机多顺序仅控制任意通路顺序；原始顺序、实际序列及扰动须随训练收据记录。

本地既有 Hallmark 2024.1 GMT 为50集合。当前文件原始 SHA256 `0d4d9ae2d000835a71fd5e7f9633701f5fa96e488c1115582e024a10a7adceee`；只将50处 CRLF 改为 LF 后为历史 `ee2463540042078bfa3f67828e1e223bb354446d9fbb4d22845866835ba5c772`，已逐字节验证差异来源为换行。仍应锁住本轮实际读取字节与表达基因顺序，不能只根据版本名假定一致。Hallmark 来自外部固定注释，参考[MSigDB 官方说明](https://www.gsea-msigdb.org/gsea/msigdb/collection_details.jsp)，不等于本队列中生物语义优势已成立。

最终论文是否具有可支持的算法贡献，取决于新协议下完成的真实消融；本审计不以文献名称、模块数量或已有高分替代机制证据。
