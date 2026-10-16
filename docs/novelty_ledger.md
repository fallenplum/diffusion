# 三模态条件交互：创新边界登记

日期：2026-10-16。状态：待证伪的实现假设；本文件不声称超过基线、首创或可接受性结论。文献与版本依据见 [机制矩阵](literature_mechanism_matrix.md)。

## 1. 核心问题

在完整 RNA 主效应和强 Clinical/Pathology 证据保留时，临床背景与患者 RNA 共同引起的 RNA-to-patch 匹配变化，能否提供普通加性、两两交互、条件注意力和一般差分不能解释的预测收益？

该问题区分三个层次：数学上依赖多模态、可训练且响应患者上下文、在未用于拟合的患者上提供条件 DSS 信息。前一层通过不自动证明后一层。

## 2. 引用、适配与拟议贡献分开

| 项目 | 分类 | 已有依据 | 本轮可以检验/不能宣称 |
|---|---|---|---|
| RNA Q 对 P K/V 的 cross-attention | 已有组件/医学适配 | 广泛使用的跨模态注意力；本轮数值 RNA 与冻结 patch 特征 | 可比较患者对应与查询变化；不能称注意力本身新颖 |
| Clinical 低秩调制 Q/K 或 pre-attention FiLM | 工程适配 | 条件化、FiLM 及安全方向信息流已有研究 | F0/A2 检验是否已足够；不能将 low-rank 或 Clinical-conditioning 单独当作创新 |
| 两张 softmax 图作差 | 引用组件 | [Differential Transformer](https://arxiv.org/html/2410.05258v2) | 原A3遗漏RMSNorm affine，旧值标缺陷；A3F独立来源只修复affine，替代最终主比较。差分不自动是医学噪声消除 |
| SDPA 后 sigmoid gate | 引用组件 | [Gated Attention](https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html) | A5−A4 分离 gate 贡献；gate 不等于语义可信度 |
| 四条共享参数参考路径及固定 mixed contrast | 拟议归纳偏置 | mixed finite difference / inclusion-exclusion 是已有数学运算；这里对其赋予 C/R 参考语义 | 检验特定条件化检索实现；不能称新的数学运算、因果反事实、PID 纯三阶协同或已识别功能 ANOVA |
| raw RNA 直接读出、强 CP 主效应保留、分块正则 | 信息保留与实验控制 | 本项目此前 RNA 表示/强基线审计 | 所有臂共享，改善不得只归因于交互核；不保证高维旁路永远有益 |
| bounded scalar interaction risk 与 bias-free readout | 稳定性工程 | 有界残差/小容量读出的一般思路 | 核验零消息时交互风险为零及实际输出边界；不能把输入小扰动当最终风险自动小扰动 |
| wrong RNA 仅替换交互通道 | 机制控制 | 患者配对扰动与本项目信息保留需求 | 排除部分患者身份无关路径；不能替代强基线/非线性容量控制 |
| 范数匹配、零均值消息控制 | 归因控制 | 避免差分仅收缩的替代解释 | A7 只检验已实现的具体收缩控制；不能推出所有正则化替代均被排除 |

## 3. 与最相关前作的区别

| 前作 | 已覆盖的空间 | 候选 A 的区别 | 仍缺的证据 |
|---|---|---|---|
| [SlotSPE](https://github.com/zylvemvet/SlotSPE/tree/02051a2083add7b427727e0200e4263516903561) | 患者动态 slots、竞争归属、选择激活、跨模态重建 | 首轮不生成 slots/重建目标；保留 gene identity，检验有指定参考语义的共享四图检索 | gene-level 不自动优于 slots；本轮未训练官方 SlotSPE，不作 SOTA 比较 |
| [I2MoE](https://arxiv.org/html/2505.19190v1) | 专家分工、模态扰动监督、患者级重加权 | 不设置多专家/PID 式监督；四参考图在一次预测算子内部共享参数与 mask | 不能由结构差异证明互补更强；未得到可识别 PID 量 |
| [CM²](https://proceedings.mlr.press/v267/wu25d.html) | 分开拟合边缘与依赖、变分联合目标、缺模态生成 | A 不拟合 copula/联合密度，只检验条件检索残差；B 是独立备选 | 预测增量不证明概率依赖已识别，未检验 copula 优劣 |
| [MDCS-MoAME](https://github.com/BioMedIA-repo/MDCS-MoAME/tree/4ae3c4d7027c07848cd631f89a17f7a094b8e898) | region/patch/omics 层次、多方向扫描、attention/Mamba 专家 | A 不新增方向扫描或专家，Clinical 明确调制 gene→patch 检索；其 3 模态与 MDCS 三流定义不同 | 未运行输入语义正确的 MDCS；公开 router 问题不能作为故意弱化基线的理由 |
| [Differential Transformer](https://github.com/microsoft/unilm/blob/31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd/Diff-Transformer/multihead_diffattn.py) | 两自由 softmax 图带可学习 lambda 作差、headwise norm | A 四图共用投影、固定系数、train-derived C0/R0；两参考零点可直接验证 | 必须超过普通差分与收缩控制；同 hidden 不等于算量匹配 |
| FiLM / 普通常规条件注意力 | 先调制输入/查询、再检索 | A 在 attention-map 空间提取 C/R 联合变化，而非只调制全局融合向量 | 若 F0/A2 已解释收益，不能归功于四图核 |

## 4. 预先锁定的可证伪性质

`Aab=A(Ca,Rb;P)`，`DeltaA=A11-A10-A01+A00`。四条路径共用参数、有效 patch mask、采样和上游随机实现。首轮 attention dropout 及上游 dropout 均为 0。

- C=C0 或 R=R0 时，DeltaA 及其无偏置交互读出为零。
- 每行 DeltaA 和为零；固定 Q/K/四图后给所有 V 同加常量，消息不变。不能泛化成任意原始 P 平移的全模型不变性。
- padding 不改变有效结果；共同患者置换保持输出置换；错误来源/分区应 fail closed。
- 梯度检查区分真实 task gradient、隔离路径梯度及参数本来应为零的初始化时刻。单独非零梯度不能证明 held-out 预测收益。
- 不把微小 mixed 消息单位化，以免把数值噪声放大成机制证据。

## 5. 当前首轮的比较范围

当前计划为 **3 个既有外折的内层分区 × 3 seeds** 的探索性工程门，不产生新的完整 outer-test OOF。所有上游拟合、参考、gene selection、PCA 与凸初始化限制在 inner-train；inner-val 只用于已锁定的模型选择/工程评测。

| 对照 | 机制问题 | 解释边界 |
|---|---|---|
| A0 强 CP+RNA 加性主效应 | 是否无需新交互 | 复杂模型若不稳定超越它，不扩大结构 |
| A1 全两两分支与 PR+C late | 二阶容量是否已足够 | 每个分支只读其两种模态，最终线性求和；不称 PID 分解 |
| A2 常规 Clinical-conditioned 注意力；F0 pre-attention FiLM | 条件化本身是否足够 | 超过 wrong 不等于超过这些对照 |
| 原A3 / 修复A3F adapted Diff | 通用差分是否足够 | A3缺affine，不作最终合法norm对照；A3F修复后另训9单元，仍非原生语言模型、实际计算预算未匹配 |
| A4 mixed 无 gate；A5 mixed+gate | 差分和 gate 的独立贡献 | 交互项被训练不等于有效 |
| A6 wrong RNA 交互通道 | 患者配对是否影响机制 | 仍保留自身 RNA 主效应，不能将 main-effect 身份破坏当交互损失 |
| A7 同门控、mixed 范数匹配的普通去均匀消息 | 是否只是消息收缩 | 缩放 detached、无标签；不代表一般正则化理论等价 |

原锁定主要机制方向是A5相对A1/A2/A3/A7，以及A4−A2。发现A3缺affine后，最终以A5−A3F替代有缺陷的A5−A3；旧比较完整保留，不挑选较有利版本。A3F实现source由原source、修复脚本SHA、配置SHA共同登记，其他已锁定训练源码不变。该修复不改变“多次使用同队列、内层已参与选择”的探索性边界。

40触顶后的160追加为预算敏感性；原checkpoint与前40轮确定性历史核验后保留，真实重训与引用分开计数。逐臂≤20%触顶门是工程门，允许少数单元仍触顶，不能写成所有单元收敛。gate和配对另行解释，Harrell/Uno60全部披露；若方向相反，不能挑有利指标宣称整体收益。内层描述没有CI，不与历史完整OOF榜单直接排名。

## 6. 明确限制与判定用语

优势：固定参考让零点与不变性可逐项验证；共同主效应旁路减少信息保留差异；简单替代解释被列为显式控制。

风险：softmax 本身的非线性可生成 mixed 项；128 patch 未覆盖所有组织；同队列长期探索、少事件与重叠内层分区限制推断；A3 算量未匹配，原生方法大矩阵尚未运行；没有原始 WSI 形态 QA。应报告这些限制，不能因本轮方向有利就改写为正式外部确认。

**当前允许的主张**是“实现并检验临床/RNA 条件参考差分检索的可训练性和内部条件增量”。只有未见患者上的稳定增量及全部相关控制支持后，才可进一步讨论该具体归纳偏置；仍需独立数据确认。若失败，仅否定本实现及当前预算，不否定三模态互补或 RNA 的独立预后信号。
