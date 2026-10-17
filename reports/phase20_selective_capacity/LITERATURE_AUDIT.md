# Phase20：聚焦文献来源与归因边界

核验日期2026-10-17。具体论文、官方代码及GitHub提交均实际联网核对；仅采用主来源。当前检索工具不能核验其底层是否Google，因此不声称使用了专有Google后端。本轮不是论文排行榜，不新增外部模型训练矩阵，也不改Phase19。

## 1. 四项直接相关来源

| 来源与固定官方代码 | 已核验实现 | 本轮复用与边界 |
| --- | --- | --- |
| [Gated Attention，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html)；[官方代码f4c2a5f](https://github.com/qiuzh20/gated_attention/blob/f4c2a5f6ffd6ec709e0c60072c95ed4f5ce5b5d2/modeling_qwen3.py) | Query生成headwise/elementwise sigmoid门，乘在SDPA输出之后，再做output projection。 | 借鉴“条件门可轻量控制更新”及普通门对照。它不直接改P–O关系概率；不能将普通输出门或FiLM称为首次Clinical交互。Sigmoid数学上不产生精确零支持。 |
| [Adaptively Sparse Transformers，EMNLP 2019](https://aclanthology.org/D19-1223/)；[entmax c2bec6d](https://github.com/deep-spin/entmax/blob/c2bec6d5e7d649cba7766c2172d89123ec2a6d70/entmax/activations.py) | α-entmax替代softmax，产生精确零权重，作者代码支持1.5-entmax/sparsemax及其他求解。 | 稀疏attention与输入相关支持早已存在。本轮采用capped-simplex gate及质量重分配，不将其冒称entmax原实现，也不因attention变稀疏就证明病理语义。 |
| [Learning Sparse Neural Networks through L₀ Regularization，ICLR 2018](https://openreview.net/pdf?id=H1Y8hhg0b)；[官方39a5fe6](https://github.com/AMLab-Amsterdam/L0_regularization/tree/39a5fe68062c9b8540dba732339c1f5def451f1b) | Hard-concrete门及期望L₀惩罚；官方`qz_loga`为共享可训练参数。代码注明未维护。 | 说明精确零/稀疏预算有成熟来源，但全局参数门不等于患者context条件门。本轮不实现hard-concrete，不复制其旧训练代码；实验臂L0仅表示legacy scalar，不能误解为该正则。 |
| [SurvPath，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Jaume_Modeling_Dense_Multimodal_Interactions_Between_Biological_Pathways_and_Histology_for_CVPR_2024_paper.html)；[官方3f73ddd](https://github.com/mahmoodlab/SurvPath/blob/3f73ddd6705ec67d643020c5bb04fb13f9f382cc/models/model_SurvPath.py) | 每条通路完整基因向量进入独立SNN，维度为`n_genes→256→256`，再与patch Token融合；不是先取一个均值再升维。 | 借鉴完整通路向量先形成多维Token。Phase20不搬331通路、256维和整套fusion，只做24个Token内signed16与rank1的容量审计，不宣称原生SurvPath复现。 |

上述公开仓库身份用于可追溯来源，不等于本轮已训练这些官方方法。Phase20没有下载其巨量权重，也不将未核验许可证的第三方代码整包公开；本轮所实现的自有算子按其实际数学及测试说明。

## 2. A不是把全局稀疏attention重新命名

本轮有本人P–O/PC/OC基础分布和冻结J0；Clinical只通过新的关系修正核改变有限概率质量，再形成有最终上限的风险修正。真实要检验的是：在完整两两主干已存在时，**哪些关系接受Clinical修正**是否需要患者对应和Clinical门依赖。

需要区分：稀疏Δlogit后整行softmax仍会改变未选概率；只限制输入λ不能限制最终预测头放大；对归一化非负attention的L₁恒为1。本轮因此检验gate0位置概率不变、质量守恒、TV上限和最终风险cap，但不将这些基本数学性质当作原创理论。

Capped-simplex的`sum(g)=0.25*n`是质量预算，不能写成恰25%边开启；普通dense sigmoid门也按同预算校准，避免把不同总修正量误归因为稀疏选择。F16只去掉gate的Clinical依赖，δ仍保留本人Clinical；它不等于整个Clinical条件分支删除。WC/WO只错配新修正，不能破坏本人主效应后称关系机制成立。WO支持交集还会受缺失影响，必须披露。

优势：修正位置、幅度和身份对照可归因。风险：普通gating/FiLM已有大量重合；支持与尺度、Clinical非线性旁路或正则差异仍可能解释收益。没有组织学真值时，稀疏关系只能称模型关系，不能称真实医学发现。

## 3. B不是用维数标签宣称信息恢复

原scalar Token在固定mask条件下每program只观察一个表达方向，升到32维不恢复被压缩方向；但单个方向可能恰好已包含DSS相关充分统计量，所以结构瓶颈不能代替held-out效能检验。

Phase20固定八组membership、三种RNA/CNA通道、缺失输入和融合架构，使用同16条signed filter，在中间以rank1映射或I16区分可用方向数。所谓名义能量匹配只在各向同性响应假设下成立；真实patient SD、rank、相关性需实测。完全相同的输出Token尺寸不等于信息容量相同；更多有效方向也不必然提升条件DSS信息。

优势：P16−P1是清楚的容量对照，DID又检查容量是否改变新修正收益。风险：16条filter可能相关、成员不足或缺失，使实际rank小于16；65个事件可能不足以估计增量。若容量改善，只能支持该表示对照，不证明通路真实生物语义优于匹配随机集合，更不能把filter坐标解释为新医学概念。

## 4. 新颖性及结论口径

可提出的研究问题：在低事件生存任务中，保留冻结强主效应后，Clinical关系修正是否需要选择性；同时，通路标量压缩是否限制该修正。不能据此声称首次FiLM、首次稀疏attention、首次通路Token或首次Clinical前置。

A、B、DID和相对J0/J_CS的实用性独立报告；方向通过不等于Bonferroni14统计支持，更不等于独立外部确认。上述来源事实和数学边界置信度高；实际DSS效能未知，须等待预锁270次训练和独立评估。
