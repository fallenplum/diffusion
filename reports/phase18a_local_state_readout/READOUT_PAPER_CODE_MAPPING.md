# Phase18A：论文、官方实现与本项目读出对照

核验日期：2026-10-16。本文先于本阶段训练交付，只登记已实际读取的原始文献、固定代码和可由代码核验的结构事实；不预填 Phase18A 成绩。父实现固定为 `760a6a88feb9b731f8a935c98a8a79c4ca300943`，源文件摘要见同目录 `OFFICIAL_SOURCE_METADATA.json`。

## 1. Mamba 的逐位置输出

实际读取 [Mamba 原论文 v2，§2、§3.2](https://arxiv.org/html/2312.00752v2)。式 (2a–b) 给出递推状态与同位置输出；选择性机制使输入相关的参数随位置改变：

\[
h_t=\bar A_t h_{t-1}+\bar B_t u_t,\qquad y_t=C_t h_t.
\]

这是一组逐位置输出，而不是仅取末状态再乘所有位置的平均输出投影。论文式 (4) 给出 ZOH 离散化；当前项目采用 exact-ZOH，不能据此声称复现了官方 Mamba 全部模块或 kernel。本文不引入原架构的卷积、D skip、外部门控、归一化或残差网络。

## 2. MambaMIL 固定作者代码

实际联网读取 [作者 `models/MambaMIL.py`，revision `9aea39efc13fdcd3703b5b54c28a64887b4acbd7`](https://github.com/isyangshu/MambaMIL/blob/9aea39efc13fdcd3703b5b54c28a64887b4acbd7/models/MambaMIL.py)，并核验 [原论文](https://arxiv.org/abs/2403.06800)。

- 87–105 行：输入线性层与逐位置 Mamba/SRMamba/BiMamba 层，序列表示始终保留位置轴；代码变量 `h` 是逐位置输出特征，不等于本项目 `[D,S]` 的内部递推状态。
- 106–111 行：对位置特征计算 attention，沿 patch 轴 softmax，再 `bmm` 聚合并分类。
- 116–120 行：生存分支输出离散 hazards 及其累计生存概率。改为本项目连续风险 Efron Cox 是明确的适配。

因此，官方 MambaMIL 的 attention pooling、父项目的终态读出和本轮固定加权逐位置池化是三种不同读出。本轮不增加 attention pooling 的可学习参数，也不把局部池化称为官方 MambaMIL 复现。

## 3. 父项目实际计算

已逐行读取 `src/phase17_ssd/ssm.py`、`src/phase17_ssd/model.py`、`src/phase17r_ssd/model.py` 及父训练入口。下式是本地源码的结构归纳。

|位置|父代码实际行为|Phase18A 边界|
|---|---|---|
|`ssm.zoh`|用 `expm1` 与小参数级数稳定计算 exact-ZOH；A 为负对角速率|保持离散化公式不变|
|`ssm.reference_scan`|逐位置递推；`states=True` 返回所有 h_t|仅作小规模输出与梯度 oracle|
|`ssm.cuda_final_scan`|通过后缀累计速率的闭式表达求 h_T；普通 autograd 连接所有递推贡献|父路径继续返回终态，不改历史实现|
|`ConditionalS6.forward`|`dt=m*softplus(dt(sx)+dd)`；`B=base_B+bb`；`C=C(sx)`|u、dt、B、C、A、原写入门及有效 patch mask 一致|
|父 `RepairS6.forward`|`Cpool=sum_t(m_t C_t)/sum_t(m_t)`；`y=sum_s(h_T*Cpool)`|F0/F1 保留这一公式|
|父 noDirect 主效应|`risk=anchor+late_risk+gamma*readout(y)`；自身临床与分子 token late 分支保留；高维 RNA512/CNA Direct 删除|L0/L1 仅替换 y 的构造，不删除自身组学，也不加自由 stacking 标量|
|父 `train.train_cell`|每 epoch 先收集全部训练患者风险，完整风险集 Efron；内层 EMA validation loss 选 epoch；fresh outer 训练所选轮数|微批量不改变风险集，保持父训练日程|

## 4. 本轮唯一变化

令 m_t 包含原模型的写入门与有效 patch mask。每个通道 d：

\[
z_{final,d}=\sum_s h_{T,d,s}\frac{\sum_t m_t C_{t,s}}{\sum_t m_t},
\]

\[
z_{local,d}=\frac{\sum_t m_t\sum_s C_{t,s}h_{t,d,s}}{\sum_t m_t}.
\]

`m_t` 同时保持原来的 dt 写入作用与读出权重含义；不因读出变化重新定义 gate。学习门为零的有效位置与 padding 不应注入状态，非零 h0 应保持；全零读出权重按父数值约定得到有限零读出。多 WSI 仍先各自递推、再患者级等权聚合，不跨切片延续状态。

N=1 且有效时两式相同。N>1 一般不相同；即使 C_t 为常数，也不能将平均 h_t 与 h_T 混同。新分段实现必须和显式递推的输出、输入梯度及参数梯度相符，段间状态不能 detach。

优势：2×2 设计保持参数与递推一致，可直接检验固定模型族内读出选择的影响。风险：局部读出同时改变风险尺度与优化条件；性能改善不能单独识别“信息保留”这一因果解释，更不能证明患者条件机制成立。

## 5. 2×2 归因与需要保留的参照

F0/L0 的状态转移不读取跨模态条件；病理仍驱动选择性参数，自身 RNA/CNA late 主效应仍存在。F1/L1 是整套 delta/B 条件化与原写入门，不能拆作三个独立因果因素。主要统计对象为 L0−F0、L1−F1、L1−L0 及其差中差，各自 Harrell/Uno60 共八项家族；关模块、错条件重放是解释性冻结干预，不记作重训。

ANCHOR_CS、历史 Clinical+S32+CONCH32 和 R1_A3_noDirect 旁列。历史 CONCH32 与本轮 UNI2-h 的视觉表示及训练流程不同，不能当同输入严格消融。

## 6. 查新：登记重叠，不启动 Phase18B

|方法与本次实际阅读范围|与后续候选的重叠|本轮处理|
|---|---|---|
|[Differential Transformer，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html)；正式页及继承的固定官方核心代码记录|两 attention 图带可学习系数作差；有符号消息不是新数学运算|不加入本轮 S6；固定 revision 与 affine RMSNorm 修复事实保留|
|[DEX，NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/7df4ce44e1af49db6f37b75708c8a237-Abstract-Conference.html)；[原论文 v2 §3](https://arxiv.org/html/2505.16333v2)|保留 pretrained attention，输出端 O′=O−λf_D(O)，辅以头选择与退火；提示差分的收益不能直接等同“消噪”|只核验方法，不编造已取得或已训练的官方 DEX 代码；与四参考图算子区分|
|[I2MoE，ICML 2025](https://proceedings.mlr.press/v267/xin25c.html)；正式页/方法登记与继承 revision|不同交互专家及弱监督扰动重加权；局部交互解释已有前作|不新增 MoE/扰动监督；专家名称不等于可识别医学机制|
|[SlotSPE，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2ef7205f8eab7d97fa07c3e5b4af84a8-Abstract-Conference.html)；正式页与继承官方代码审查|患者动态 slots、竞争归属及跨模态结构建模已有前作；其 latent event 不等于观察到的 DSS event|官方代码已可取得；不沿用“尚无代码”旧状态，不在本轮增加 slots/重建|

固定官方 revisions 与许可证状态另见 `BASELINE_REGISTRY.csv`、`OFFICIAL_SOURCE_METADATA.json`。没有逐行重新审查的代码明确写继承核查，不冒充本轮完整复现。

## 7. 对工作单时间边界的更新

用户工作单基于 Phase17/17R 快照，其“尚未检验四条件差分”仅能描述这两阶段。随后 `research/trimodal-conditional-interaction-v1 @ a60aa99` 已实际完成独立 **inner-only** 探索、训练预算追加及 A3F norm 修复；公开报告记录 125 个实际模型单元并给出 `STOP_CURRENT_INTERACTION_PILOT`。这些不是完整外层 OOF，不是 Conditional S6，也不并入本轮 2×2 或改变本轮八项主比较。

本轮不自动扩展 Phase18B。局部读出是常规 SSM 输出与聚合选择，不能单独作为新算法贡献；即便结果改善，也须将读出收益、条件增量、强基线比较和独立队列证据分别陈述。
