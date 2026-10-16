# Phase19：临床条件低秩注意力的文献与基线审计

核验日期：2026-10-17。**本文件是设计前的来源审计，不是本轮训练完成记录，不预填成绩。** 已读取 Phase10 来源/基线表、Phase17 实际适配代码与审计、Phase18A registry，以及此前独立四图差分 inner-only 报告。当前可用工具提供网页检索及原站访问，未能指定或验证其底层是 Google，因此不声称完成了专有 Google 搜索。具体论文/官方代码均联网读取；网页抓取失败的 raw GitHub 文件经 HTTPS/API 实际获取，未以标题推断实现。

## 1. 先界定公式与既有工作的重合

用户给定 patch p、pathway token o、患者 Clinical c 的兼容度：

\[
s_{ik}=\frac{(W_p p_i)^T(W_o o_k)}{\sqrt d}
 +\sum_{r=1}^{R}(a_r^Tp_i)(b_r^To_k)(c_r^Tc).
\]

令 A、B、D 分别按列收集 a_r、b_r、c_r。由代数直接得到：

\[
s_{ik}=p_i^T\left[W_p^TW_o/\sqrt d+A\operatorname{diag}(D^Tc)B^T\right]o_k.
\]

这是 **Clinical 门控的低秩双线性兼容度**：固定患者时，Clinical 改变 P–O 匹配矩阵；对应三阶参数张量采用 CP 型秩 R 分解。它也等价于增广 query/key 的点积：

\[
\bar q_i=[W_p p_i/d^{1/4};(A^Tp_i)\odot(D^Tc)],\quad
\bar k_k=[W_o o_k/d^{1/4};B^To_k],\qquad s_{ik}=\bar q_i^T\bar k_k.
\]

这一等价只要求这些投影与公式一致；若另加 bias、非线性、不同归一化或缩放，必须重新核对。不能把两种等价写法分别计为两种独立新机制。参数本质与已有 [LMF，ACL 2018](https://aclanthology.org/P18-1209/)、[双线性注意力 BAN](https://arxiv.org/abs/1805.07932) 和 [FiLM 条件调制](https://arxiv.org/abs/1709.07871) 有明确重合。LMF 作者 [固定代码](https://github.com/Justin1904/Low-rank-Multimodal-Fusion/blob/ea4755124d19bddf7941695215e98f16fbb1fff9/model.py) 已对三个模态的低秩因子逐元素相乘。

**允许提出的问题**：将这种已知因子化放到临床背景下的 patch–pathway attention logits，能否在保留强主效应时提供可重复的 DSS 条件增量。**不能据公式本身声称**：低秩乘积首次、Clinical conditioning 首次、可识别的纯三阶信息协同、因果关系或医学语义已成立。Softmax 非线性及后续读出仍会引入复杂依赖；非零三阶 logit 不等于统计增量。

## 2. 最近的直接前作：Clinical 前置并非空白

[CIGTSurv 原论文 v1](https://arxiv.org/html/2608.03247v1) 将 Clinical 文本经 BiomedBERT 编码，用 Clinical Q 分别检索病理与基因 tokens，再加入 MMD 对齐；不是仅在末端拼 Clinical。因此“现有三模态方法都没有前置 Clinical”不成立。与本候选的可检验区别是 C→P/C→O 两组配对检索，与 C 直接调制 P–O 兼容度之间的区别。

作者 [仓库固定提交 b0b8d5c00d55bcf33c5aa913c1f4caf19f0c3fb2](https://github.com/Daijing-ai/CIGT-Surv/tree/b0b8d5c00d55bcf33c5aa913c1f4caf19f0c3fb2) 的递归 Git tree 本次实际仅发现 README.md，正文为 Under Construction。**论文存在不等于官方模型已可运行**；未取得实现/代码许可，不得将自行搭建的 Clinical-anchor cross-attention 挂为官方严格复现。若纳入，只能标 paper-inspired/adapted，并另行锁定。

[ProtoPathway 原论文](https://arxiv.org/html/2605.21454v1) 与 [官方 MIT 代码](https://github.com/AmayaGS/ProtoPathway/tree/1e678a59b40d1b4f418f1f22b70b60dc7b69a388) 已覆盖病理 prototypes、分子 pathway 与跨模态 attention；实际 `CrossAttentionFusion` 以 prototype 作 Q、pathway 作 K/V，再学习 prototype 汇聚门。它不是本候选的 Clinical 三元 logits，但“pathway–形态原型交互”不能当作无人研究。

[DIAGNijmegen 官方框架固定版本](https://github.com/DIAGNijmegen/multimodal-survival-fusion/tree/19e584d4573e92f5d214d54211648af764af6d73) 的 `lowrank_bilinear.py` 实际是 P–R、P–C、R–C 三个低秩两两块后拼接，**不是**三模态相乘的三元 CP。这适合作为全两两容量对照的来源；目前只核验代码，不把仓库 README 的其他癌种分数放入本队列表。

## 3. 三个必须真实运行的方法：来源、原生定义与当前缺口

固定 SHA、核心文件 blob/SHA256、访问状态及许可证的实际快照见 [BASELINE_SOURCE_REGISTRY.csv](BASELINE_SOURCE_REGISTRY.csv)。代码公开不自动意味着可任意再分发；本轮优先采用私有固定检出与薄适配器，不将缺独立许可的源码整包公开。

### MGCT：可复用已核验的批量适配，但不是现成 Phase19 成绩

[原论文](https://arxiv.org/html/2311.11659v1) 为 Mutual-Guided Cross-Modality Transformer，输入病理 patch bag 与分组组学，双向 cross-attention、attention pooling、两阶段融合；原始工作使用 OS。作者代码默认离散 hazards，训练入口提供 CE/NLL/Cox 选项；存在 Cox 选项不证明它等于本项目完整风险集 Efron。

固定官方提交 `daa60d7ae602e329299b86563cf29f571259e34d`。本仓库 [Phase17 适配器](../../scripts/phase17_paper_models.py) 与 [核验脚本](../../scripts/audit_phase17_papers.py) 已有真实执行：15 inner+15 outer，另1个预锁定诊断。适配包括六组改八组、UNI2-h1536输入、RNA/CNA/missing、scalar Cox、Clinical/S32共同主效应；B=1原层输出/梯度逐元素核验通过。这些历史训练不能登记成当前协议的新训练。

当前复用缺口：重新锁定本轮 main effects、输入、预处理与训练预算；不要静默继承旧高维 Direct。原公开 stage2 每次使用同一 stage1，前面 stage2 层未进入最终预测，此行为在旧适配中被显式保留。原样与后续修复必须区分，不将有效参数不足隐藏为公平优势。

历史实测可作为资源起点：Phase17 MGCT 约6.386M注册参数、4.802M有梯度参数，512 patch outer 峰值6045.91 MiB，**含共同Data缓存**。它不等于本轮隔离速度/额外激活预算，须实际单cell profile后再确定并行。

### CAMR：原生三流是 RNA、病理、CNA，已经有低秩三模态乘积

[Bioinformatics 原文](https://academic.oup.com/bioinformatics/article/39/1/btad025/6986972) 的 CAMR 是 cross-aligned multimodal representation learning。原生病理为 CellProfiler 特征，经选择得到与 RNA/CNA 同维的80维患者向量；不是 patch attention。保留共享/私有表示、跨模态门与低秩融合，并联合 Cox、重建、差异及对抗目标。原生三流中 C 表示 CNA，**不是 Clinical**。

固定 `8fbae0a0bc4e9da31547b5eab5d773e6b2049e93` 的 `networks.py`/`fusion.py` 实际已有 rank16 LMF 乘积。先前 Phase17/18A 只做清单与源代码核查，没有正式训练 CAMR；不能拿注册行补齐实际 baseline 数量。

本次代码核验发现必须公开处理的差异：

- `utils.CoxLoss2` 使用三角风险集与 event-count+1 分母，没有 Efron ties。统一 Efron 是 loss 适配。
- `train_test.py` 创建 discriminator，但 optimizer 只接收预测模型参数；原公开执行链没有单独更新 discriminator。不能把该代码执行称为完整 min–max 对抗训练。
- `networks.py` 的第三个 gate 用 `x_path_common` 作 bilinear query；h2/h3 调用共用 h1相关层，部分另行注册层未使用。这是固定代码事实，未据此推断作者未公开环境或论文成绩。
- 导入 `fusion.py` 会执行 `options.parse_args()`、设置GPU并写选项；加载时需隔离这些CLI副作用，不能让上游解析器覆盖本轮目录或配置。

UNI2患者均值/PCA替代CellProfiler、训练折内RNA/CNA降维、Clinical共同主效应都应标 **CAMR-UNI2+Cox+Clinical adapted**。若修复索引、判别器或对抗目标，另加明确 repair 标签与测试，不静默修改后仍称原样；也不能故意只提供存在问题的弱实现来衬托新模型。

### MMP：真实 prototype/distribution 表征，不能用任意均值池化替代

[ICML 2024 原文](https://proceedings.mlr.press/v235/song24b.html) 的 Multimodal Prototyping 将病理表示压为形态分布/prototype tokens，与分子 pathway tokens 融合。官方固定 `1fd75e37592f7ab47b787f737733ba3c5a7e714c` 的 `main_survival.py` 默认 NLL，另提供 Cox 等选项；内部 `model_multimodal.py` 支持概率/均值/协方差结构的 prototype 输入。

实际官方 `main_prototype.py` 从 `dataset_splits['train']` 拟合 K-means；不能因为无监督而在全512患者上拟合原型。PANTHER/PrototypeTokenizer 的精确表示、初始化来源、维度及 role/fold 的训练患者摘要都需保留。原生无 Clinical；CNA与临床并入共同主效应、替换 backbone、风险头/损失均属适配。

本仓库当前尚未有 MMP 的正式适配训练成绩。接入条件是：真实固定官方模块可导入、训练折 prototype 生成、PANTHER/tokenizer 数值和维度检查、完整 Efron 路径与 fresh outer 协议通过；缺任意一项先 fail closed。不能将普通 attention/均值/PCA 网络重命名 MMP。许可证本次明确核验为 **CC BY-NC-SA 4.0**，不是笼统的“creative commons”。

## 4. 本轮优先锁定的三个实验控制

1. **信息与容量控制**：强 Clinical+S32+冻结视觉+RNA/CNA 自身主效应，以及相同输入/风险头/训练预算的普通 P–O attention。所有臂共享保真的主效应路径；删除三元项、删除普通attention、删除Clinical条件分别只改对应模块。完全移除Clinical主效应是另一个问题，不能与仅移除Clinical条件混为一项归因。
2. **既有条件化替代与代数等价控制**：同rank/近似参数预算的 Clinical-Q/FiLM；若按第1节实现后严格等价，则用输出/输入梯度/参数映射 parity 证明等价，不制造一个假独立 baseline。更一般的非线性FiLM可能包含本候选的函数类。要先固定 softmax 轴：若沿patch归一化，仅加不随patch变化的Clinical key偏置会整行抵消，不能用它当唯一 Clinical-conditioned 对照。
3. **患者身份控制**：固定训练分区内的错误Clinical或错误Omics donor映射，只替换 interaction context，患者自身 main effects 始终正确。冻结推理错配与从头错配训练分开，记录 donor退化/自配对、source及patient partition；不能破坏RNA主效应后把损失全归因于交互。

若声称超越全两两关系，还须保留独立 f_CP+f_CO+f_PO 后线性求和的容量对照；不能让该对照在非线性最终头重新读取全部三模态。若声称生物知识结构优于任意分组，必须另有匹配随机基因集合；不在本轮未经锁定就扩成新的搜索矩阵。

## 5. 与历史结论和下一步的边界

旧四参考图 attention 在 `a60aa99e7482d398b46c15738975105226871024` 已完成 inner-only 预算与A3F修复并停止；它不是未试过的方法，也不是本轮 CP logit。Phase18A 仅更改自定义 S6 读出，同样不能作为当前 attention kernel 的成败证据。历史高分、旧论文适配及不同协议表格只能旁列，不直接复用为本轮同协议 leaderboard。

优势：CP写法使Clinical对P–O兼容度的作用直接可检验，参数/零点/错配可审计，冻结特征降低上游训练差异。风险：已知LMF/FiLM/BAN重合、65 DSS事件、八程序token可能压缩RNA排序信号、logit尺度和softmax轴导致的替代解释、反复研究队列与无原图形态学QA。attention变动、非均匀选区或非零梯度不是医学语义真值或独立预后增量。

置信度：上述代数等价与所读固定代码事实高；“该候选能稳定改善DSS”当前无新实验，不作概率承诺；“首创性已成立”当前不支持。先完成正式协议和基线资格，再训练；不因查新自动叠加MMD、MoE、重建、蒸馏或新的扫描模块。

## 6. 当前实际实现与资格（训练前锁定）

已实现 [薄适配器](../../src/phase19_attention/baselines.py)、[固定源清单](../../configs/phase19_baseline_sources.json) 和 [独立资格测试](../../tests/test_phase19_baselines.py)。清单锁定三库10个实际用到的Python文件，同时校验SHA256和Git blob。`prepare_baseline_sources()` 已把旧私有检出中10份匹配文件复制至本轮私有 `external`；本次没有下载模型权重、患者数据，也没有将第三方文件公开。`verify_baseline_sources()` 只读文件和hash，不执行官方模块、不初始化GPU；缺源或内容变更直接报错。依赖补齐为einops0.8.2（65.6kB安装包，无额外依赖安装）。

加载方式为对固定文件中原有 `ClassDef`/`FunctionDef` AST 进行原体执行，注入所需Torch/NumPy/einops符号，隔离CAMR的CLI解析和未使用的可选依赖；不是重写一个近似网络再冠官方名称。下列适配不属于原生论文复现：

| 方法 | 本轮精确定义与差异 |
| --- | --- |
| MGCT-Cox+Clinical adapted | 保留官方256维核心、1层stage1/2层stage2、SNN与gated pool；1536维UNI2输入、原六组改为八组，组内五通道分别是RNA/CNA gain/CNA loss/RNA mask/CNA mask；仅移植batch与padding轴。原stage2前一层不连最终风险的结构保留并测试披露。 |
| CAMR-REPAIRED-Cox+Clinical adapted | 输入为训练折内RNA PCA32、UNI2患者均值PCA32、CNA gain/loss合并后PCA32；保留官方内部宽度、共享/私有分支、LMF rank16、有界风险以及重建/差异函数。唯一网络AST修复是第三gate query从pathology改为CNA；仍保留原共用h1/gate层和未用注册层。另设真正D优化：RNA为D正类，P/CNA为负类并detach；G阶段冻结D参数而向P/CNA共同编码传导“判为RNA”的fooling梯度。G辅助目标权重固定为对抗0.6、三重建之和0.8、三差异均值0.05、原网络L1为2e-6。其aux按实际microbatch均值加权，不宣称等于全队列DiffLoss。 |
| MMP-PANTHER-Cox+Clinical adapted | fit患者的patch固定抽最多8192，sklearn Lloyd KMeans16、n_init1、seed191019、max_iter100；这替代原FAISS原型初始化，须记作预处理实现适配。随后使用官方PANTHER EM（3次、tau10、prior covariance scale0.1）、官方tokenizer的prob/mean/cov，得到每WSI16×3073，再用官方256维coattn及八组五通道组学。分布特征保持私有；原型训练只用fit患者，无监督也不读取held-out患者来拟合centroid。held-out WSI仅独立EM更新其自身分布，固定prior。 |

三者共同Clinical+S32+RNA/CNA/UNI2线性主效应、Efron完整风险集及fresh outer训练由统一外层提供；原生核心不再额外继承旧Direct或Clinical头。使用相同原始信息不意味着三种方法的粒度相同：CAMR为患者向量，MGCT为patch，MMP为分布prototype。应把这些差异写进本轮结果，不称三者为同参数预算的严格同构消融。

**实际资格：服务器7项测试全部通过（2026-10-17，最近一次4.63秒，非skip）。** 包括10文件真源防漂移；MGCT B=1原层输出、输入和全部参数梯度parity及批量mask对照；MGCT完整核心梯度和原stage2未用层；CAMR原版/修复版第三query与其他分支不变；实际D优化一步、D/G梯度隔离与重建回传；官方PANTHER/tokenizer逐元素相等和padding不变；MMP无survival字段的数据准备、fit-only centroid、缓存重载、WSI索引与官方coattn前向parity。首次运行因未装einops阻断，补齐后通过；锁前发现并修复私有prototype provenance的tuple/NumPy到JSON回读规范化，补充回归后再次7/7。均为源码/合成数据资格检查，**不算正式DSS训练、OOF、模型优势或机制证据**。正式训练数与GPU预算另由执行记录给出。
