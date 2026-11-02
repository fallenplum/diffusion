# 直接近邻核验：NOVELTY_NOT_ESTABLISHED

访问日期2026-11-02。已直接核验近邻的原始论文页面/正文方法；这不是原作者代码复现或穷尽式排重。候选可检验区别是：无病理坐标的全局分子/临床query，读取真实邻接边分布；摘要带质量并针对冻结查询答案训练；在强加性主效应之外先检验未压缩关系，再冻结同一个风险头检验压缩。

Querent的正文§3.2公式(1)-(2)使用区域min/max元数据、query-patch选择区域；定理3.1明确要求空间距离衰减，不能直接套给无坐标的bulk RNA查询。[正文](https://arxiv.org/html/2501.18984)。它已研究query-aware近似，不能声称首次查询压缩。

Nearly Optimal Attention Coresets的Theorem1/4已对所有有界query给KV子集近似保证。[正文](https://arxiv.org/html/2605.05602)。QPR的简单失真上界不是更优的coreset理论，也未证明M32最优；子集coreset与学习的软重心摘要并非相同对象。

ProtoSurv已经研究组织异质图与空间交互：[NeurIPS2024原文入口](https://proceedings.neurips.cc/paper_files/paper/2024/hash/760341adc5632de3f1cf2e8d22215a93-Abstract-Conference.html)。TMI2024的GraphMixer、八邻接与基因签名attention也已存在，见§II-B、邻接消融TableIII及官方vkola-lab/tmi2024代码入口：[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11374469/)。真实坐标加组学查询本身不能作为原创。

MMP §3.1-3.2使用GMM形态比例/均值/协方差与通路原型；不是只保留均值：[正文](https://arxiv.org/html/2407.00224)。PIBD公式(8)-(9)使用生存监督原型瓶颈：[正文](https://arxiv.org/html/2401.01646)。SlotSPE的结构槽位、重建与交互也属于直接近邻：[正文](https://arxiv.org/html/2512.01116v1)。本轮不把语义槽位、瓶颈或低容量读出单独称为新算法。

DiffPool公式(3)-(6)已有软分配和图粗化：[正文](https://arxiv.org/html/1806.08804)。Graph Coarsening with Message-Passing Guarantees §2-4把传播算子与粗图绑定：[正文](https://arxiv.org/html/2405.18127)。谱/传播保真与本轮指定attention答案保真不是同一个目标，但“针对下游算子压缩”的原则已有前作。

MCAT的组学coattention、MOTCat的OT全局分配、SurvPath的通路/病理注意力块和CMTA的翻译与PPEG均是已有机制，不能把普通attention、组学query、模态对齐或GNN重新命名为创新。详见literature_novelty_matrix.csv的核验范围；未访问到的具体源码细节不作已经复现的主张。

G0只判定存在可测的联合任务差别，允许检验；原创性仍为NOT_ESTABLISHED。优势是将关系医学价值与压缩保真分开；风险是可能只是现有KV近似与图建模在新任务的组合。若后续没有真实优势，不能以这些差异补写成功论文结论。
