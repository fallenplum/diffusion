# 三模态条件交互：文献与机制核验矩阵

核验日期：2026-10-16。本文登记原方法及本项目适配边界，**不登记尚未产生的本轮性能**。原报告的阅读等级不被自动继承为本次已复现证据。

## 1. 证据等级与版本

- **方法与核心代码核验**：实际读取论文方法及相关官方核心文件；不表示已跑完整原生 benchmark。
- **公开执行链核验**：额外追踪训练入口、loss 和 optimizer；仍不表示复现作者成绩。
- **方法核验**：方法已读，代码仅定位或锁定 revision，未逐行审查。
- **出处筛选**：正式页/摘要核验；未审计的 Q/K/V、gate 和训练细节留空，不据题目补全。

精确 repository revision 与核心 Git blob 见 [SOURCE_REVISIONS.json](../reports/trimodal_interaction_v1/SOURCE_REVISIONS.json)。下列固定链接对应本次实际读取版本；代码的当前默认分支可能继续变化。

## 2. 与首选候选直接相关的核心方法

| 方法 / 出处 / 核验范围 | 原始输入及主要损失 | Q / K / V 与 softmax 轴 | 残差、归一化及 gate 位置 | 本项目适配与不可宣称内容 |
|---|---|---|---|---|
| **Differential Transformer**；ICLR 2025；方法与核心代码核验。[论文 v2](https://arxiv.org/html/2410.05258v2)、[固定官方代码](https://github.com/microsoft/unilm/blob/31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd/Diff-Transformer/multihead_diffattn.py) | 原始文本自回归语言建模；token 预测目标 | 同一序列投影为两套 Q/K、共享 V；各图沿 key/token 维 softmax；输出为两图带可学习 lambda 作差后乘 V | 官方各差分头 RMSNorm **有可学习affine**，之后乘 `1-lambda_init`、再 output projection；模型宏结构有 residual | 原A3是遗漏affine的缺陷cross-attention适配，不完整遵循官方norm。A3F仅修复affine后独立重训，最终A5−A3F替代A5−A3。两者均移除文本因果mask/伪基因顺序、普通4头对应2diff头；仍不是原生语言模型、未严格匹配算量 |
| **Gated Attention for Large Language Models**；NeurIPS 2025；方法与核心代码核验。[正式页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html)、[固定官方代码](https://github.com/qiuzh20/gated_attention/blob/f4c2a5f6ffd6ec709e0c60072c95ed4f5ce5b5d2/modeling_qwen3.py) | 文本语言模型；语言建模目标 | Q/K/V 来源于隐状态；沿 key/token 轴 softmax；gate 从 query 输入的投影获得，提供 headwise/elementwise 版本 | sigmoid 乘 **SDPA 输出，早于 output projection**；不是重归一 attention weights；之后才进入模型残差 | A5 将 C/R 条件用于消息 gate 是任务适配。必须有 gate-off 对照；gate 值不等于医学可信度，不能据语言任务收益承诺 DSS 改善 |
| **SlotSPE**；ICLR 2026；方法与核心代码核验。[正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2ef7205f8eab7d97fa07c3e5b4af84a8-Abstract-Conference.html)、[方法](https://arxiv.org/html/2512.01116v1)、[官方固定 revision](https://github.com/zylvemvet/SlotSPE/tree/02051a2083add7b427727e0200e4263516903561) | WSI patch 与 RNA/通路；生存目标、slot decoder 生存辅助目标、模态内重建及 WSI→omics 重建 | 动态 slots 作 Q，输入作 K/V；先沿 **slot 轴**竞争 softmax，再沿输入轴 L1 归一；跨 slots 有双向注意力 | GRU/MLP 迭代更新；可微 top-k 选择与 soft gate；不等同冻结的固定通路均值 | 代码已公开，不能沿用“不可取得”状态。原生末端可选 Cox 标量头，但 slot decoder 仍有 NLL；统一 Efron/Clinical 后需标 adapted。患者动态 slots/跨模态重建不是本项目原创空白 |
| **MDCS-MoAME**；作者仓库标 CVPR 2026；公开执行链核验。[官方固定 revision](https://github.com/BioMedIA-repo/MDCS-MoAME/tree/4ae3c4d7027c07848cd631f89a17f7a094b8e898) | 五方向 region/patch 特征、坐标与六组 omics；离散 hazards/NLL 加跨模态特征匹配与 intra 匹配项 | attention 专家以一流作 Q、另一流作 K/V；沿 key 轴 softmax；另有 Mamba 专家。最终三流是 patch/region/genomics | 模态交互后平均残差式融合；router 主 forward 用 argmax Python 选择，详见第 3 节；不是 Clinical/P/R 三流 | 当前首轮**没有训练 MDCS baseline**。原图/区域信息布局、离散头、router 修复及 Clinical 适配需分别标记，不能复制相同 tensor 伪造方向/region；不能用失效 router 故意弱化基线 |
| **I2MoE**；ICML 2025；方法核验、代码目录定位。[方法](https://arxiv.org/html/2505.19190v1)、[官方固定 revision](https://github.com/Raina-Xin/I2MoE/tree/75b578e1f7ec20ebe990f6e6e5680b1a8860046f) | 多种医疗/通用模态；分类/回归任务损失，加完整及随机替换模态 forward 的交互监督 | backbone-agnostic；没有一套可代表所有专家的统一 Q/K/V。soft weights 沿交互专家归一，不应和 patch attention 轴混淆 | 专家预测经 reweighting 加权；并非固定四参考图差分。具体训练代码本轮未逐行核验 | 仅作 novelty 边界；本轮没有加入专家扰动辅助 loss。unique/redundant/synergistic 专家名称不能直接充当可识别生物机制 |
| **CM²**；ICML 2025；方法核验、代码目录定位。[正式页](https://proceedings.mlr.press/v267/wu25d.html)、[方法](https://arxiv.org/html/2511.03196v1)、[官方固定 revision](https://github.com/HKU-MedAI/CMCM/tree/7e80bacf9e883d24a94a156506f0e72881080331) | MIMIC EHR/CXR/notes；任务目标联合 GMM 边缘及 copula 变分目标 | GMM 边缘及 copula 依赖是主要机制；没有本候选 RNA→patch Q/K/V；混合分量 softmax 不是 patch attention | 拼接表示后 fusion/classifier；本轮未逐行核验具体代码 gate/residual，因此不补写 | 候选 B 是独立备选，本轮未训练。Sklar 定理不保证选定边缘/copula 正确拟合，也不自动提供条件预后增量 |

## 3. MDCS 公开实现的已核验问题

核心 [MDCS_MoAME.py](https://github.com/BioMedIA-repo/MDCS-MoAME/blob/4ae3c4d7027c07848cd631f89a17f7a094b8e898/models/MDCS_MoAME/MDCS_MoAME.py) Git blob 为 `56174af1e0f54c09f034bb362ad0e4680ad0be20`，与用户报告登记相同。

1. `Gating_network.forward(x1,x2,temp=1.0,hard=False)` 与 `MoAME.forward` 的位置调用不对应：`routing_network(x1,x2,hard)` 将 hard 传给 temp。
2. 模型四次 MoAME 调用均 `hard=True`，随后 `argmax(...).item()` 只产生 Python 专家索引，不把任务梯度传给 router。
3. soft 分支直接求三个专家之和，没有乘 router weights；其 `hard=False` 还被传作温度 0，但 logits 未被用于输出。
4. 已追踪 [main.py](https://github.com/BioMedIA-repo/MDCS-MoAME/blob/4ae3c4d7027c07848cd631f89a17f7a094b8e898/main.py) → [engine.py](https://github.com/BioMedIA-repo/MDCS-MoAME/blob/4ae3c4d7027c07848cd631f89a17f7a094b8e898/models/MDCS_MoAME/engine.py)：仅一次综合 loss.backward/optimizer.step，没有独立 routing loss、路由返回值或另一路优化。
5. [隔离探针](../reports/trimodal_interaction_v1/MDCS_ROUTER_ISOLATED_PROBE.json) 保留原 AST 的 router 控制流，仅用 CPU 线性专家替代 Mamba 依赖；8 个 router 参数张量梯度均为 None，选中专家有非零梯度。它只证明该控制流，不是完整模型原生训练。

本次 [CVF 正式页面](https://openaccess.thecvf.com/content/CVPR2026/html/Qu_MDCS-MoAME_Multi-directional_Composite_Scanning_with_Mixture_of_Attention_and_Mamba_CVPR_2026_paper.html) 返回 403，论文全文未成功取得。以上是公开版本代码结论，**不是对作者未公开实验环境、论文成绩或方法总体有效性的判定**。未来原样、router 修复、统一 Cox 版本分别登记。

## 4. SlotSPE 复现时必须登记的随机性

[slot_attention.py](https://github.com/zylvemvet/SlotSPE/blob/02051a2083add7b427727e0200e4263516903561/models/slot_attention.py) 每次 forward 用正态噪声初始化 slots，top-k 函数每次加 Gumbel 噪声，均未依据 training 状态切换。因此 `eval()` 本身不保证相同患者重复推理一致。原样运行需明确 RNG 状态及推理抽样规则；将 slots 改为均值或关闭 Gumbel 是另一种适配，不能静默归为官方原样。本轮未训练 SlotSPE。

## 5. 其余文献：仅登记此次真正达到的核验范围

| 方法 | 此次核验来源与范围 | 原生对象/提示 | 本轮使用状态 |
|---|---|---|---|
| CoMM / ICLR 2025 | [正式页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/108030643e640ac050e0ed5e6aace48f-Abstract-Conference.html)，出处/摘要 | 对增强后的多模态联合表示做一致性；不是只对齐共同信息 | 不新增 contrastive loss；未审计本轮官方代码/QKV |
| Gated DeltaNet / ICLR 2025 | [camera-ready arXiv 记录](https://arxiv.org/abs/2412.06464)，出处/摘要 | 文本序列的遗忘与 delta 定向写入 | 候选 C 背景；本轮不添加状态模型；未锁定代码 revision |
| Mamba-3 / ICLR 2026 | [arXiv 记录](https://arxiv.org/abs/2603.15569)，出处/摘要 | 以解码效率为目标的复值状态及 MIMO | 不将语言解码收益迁移为生存效果保证；未读本轮官方代码 |
| MODA / ICML 2025 | [正式页](https://proceedings.mlr.press/v267/zhang25cg.html)，出处/摘要 | 视觉语言的模态内精炼/模态间交互、correct-after-align | 未审计实现，不纳入首轮训练，不补写 QKV/gate 细节 |
| HAF / ICML 2026 | [正式页](https://proceedings.mlr.press/v306/chen26ch.html)，出处/摘要 | RGB-D 的非对称精炼、频域相关与门控驱动 | 只保留辅助噪声回流问题意识，不在无自然格点的基因上复制 FFT |

## 6. 本轮实验的适配声明

首轮是探索性 **inner-only** 审计：既有外折编号 1/3/5 的 inner-train/inner-val，seed 31/47/83；这些 inner-val 分区并不构成独立的完整外折 OOF。patch=128，从现有 512 几何匹配候选中均匀取样；单冻结主干，hidden=64，普通 head=4，RNA 保留 512 个基因及其 identity/missing。原 512-patch 历史结果只作参考。

共同主效应采用 C+S32+LOCAL32+RNA512，L2=1 的凸解初始化后共同训练；无 teacher/stacking。A0–A7、pre-attention FiLM F0 和 PR-attention+C-late 对照共享输入、分区及训练预算。40 epoch 上限、至少 8 epoch、patience=12、dropout=0；实际完成、选中 epoch 及预算充分性由运行产物登记。

原A3有`elementwise_affine=False`实现缺陷。原40/160的A3结果保留原样，但不作为最终合法官方norm对照。独立脚本`repair_trimodal_diff_norm.py`只将headwise RMSNorm改为affine=True，以原始source+脚本SHA+配置SHA形成A3F独立实现来源，按160上限真实重训9单元；最终A5−A3F替代原A5−A3，不在缺陷/修复版本中择优。A3F仍使用cross-attention/统一Cox，并且**算量尚未匹配**。

原40预算之外，观察内层触顶后追加160敏感性；仅重训触顶单元，完整early-stop单元引用原checkpoint并区分真实拟合计数。A3F另行追加，不覆盖前两阶段。预算门允许每臂至多20%单元触顶，它是工程判据，**不等于所有单元优化收敛或达到全局最优**。

A7 对普通 `A11-uniform` 消息按 detached norm 缩放到同参 mixed 消息范数，再应用与 A5 相同 gate；匹配本身不读取生存标签。该控制只覆盖这个明确的收缩替代，不能泛称排除了所有正则化解释。

优势：算子零点、mask、梯度与患者配对可逐项检验。风险：内层患者重复、65 个 DSS 事件、128 patch 覆盖限制及未完全匹配计算预算。文献核验不能替代这些实验风险检查，也不构成新机制成立的证据。
