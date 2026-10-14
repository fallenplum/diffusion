# Phase 12｜C4 机制确认：患者 RNA、随机基因分组与 Patch 粒度（Codex 执行 Prompt）

你是本项目的**实验执行者兼第一轮审计员，不是结论裁判**。直接在 `fallenplum/diffusion` 最新 Phase11 分支 `codex/phase11-semantic-anchored-trimodal` 基础上新建 `codex/phase12-c4-mechanism-confirmation`；所有训练在原服务器进行。**先读取 Phase11 源码、协议锁定、BUG_LOG、私有 OOF/checkpoint 和完整报告**，不要只读 RESULTS.md。不得覆盖任何 Phase10/11 产物。新目录 `reports/phase12_c4_confirmation/` 与 `/root/phase12_c4_private/`。

## 0. 已知证据与目标

- 固定队列：TCGA COAD/READ，512 患者、65 DSS 事件、原患者五外折；主指标 Harrell 与 Uno60（60 月）。不更换终点、不碰扩散/缺失模态填补。
- Phase11 强基线 **A1 = Clinical+TITAN-S32 Ridge**：Harrell `0.805928` / Uno60 `0.773719`；**S32 是病理文本概念相似度，不是 TITAN-PCA32**。
- Phase11 **C4 = 随机匹配基因分组 + S32/Clinical↔CONCH Patch + RNA→Patch**：`0.813691/0.802965`，仅探索性点估计；真实 Hallmark 分组 C3=`0.801694/0.789755`。原 Hallmark 实际仅覆盖 Top512 中约 **19—22 个基因**，其余为 fallback。
- Phase11 512 Patch 下同预算 RNA 增量 `ΔH≈+0.01168，ΔUno≈+0.01351`，三个初始化方向均正；尚未独立确认。Phase11 的部分模型使用 `risk=jointly_trainable_C+S32_anchor+0.1*deep_head`，**不得把它包装成无残差的全新预测架构**。
- 目标不是追逐新最高分，而是回答：**C4 是否依赖正确患者的 RNA？随机分组优势是否稳定？Patch 数量是否改变 RNA 的条件收益？改进 Hallmark 覆盖后真实通路是否优于匹配随机控制？**

## 1. Hard Gate：冻结数据、协议与复现

1. 先将 Phase11 的 C4（128 Patch，原随机分组 g0）、A1 和原无 RNA 对照在锁定环境中 **checkpoint 回放**；记录原 Git SHA、数据/患者/WSI/fold/特征 SHA256、环境版本、模型/参数/实际有效参数、原风险哈希。若不能与 Phase11 私有 OOF 在数值容差内一致，**停止**，不得使用“Phase11 已复现”措辞；先审计输入及计算图差异。
2. 预先锁定 5 个外折、三个初始化 `[31,47,83]`、inner split、Cox-Efron full-risk-set、训练预算/优化器/epoch 选择、特征选择及 Patch 采样。若要扩展训练预算，所有同组臂同步改版且必须在看外折结果**之前**决定。
3. 仅允许六个合法临床字段；Top512、RNA 缺失填补、标准化、TITAN-PCA、基因筛选、临床编码均 train-fold fit；inner 模型严格 inner-train fit，禁止使用外折事件选择模型或分组。保留 RNA 3024 缺失基因块及测序平台代理的分层诊断。
4. Patch 使用 Phase11 已验证的**嵌套** 128⊂512⊂1024 采样；保留 519 张 slide 的患者/文件/hash/空间坐标一致性。CONCH 与 UNI2 不得按 Patch 行序强行配对。

## 2. 主要实验矩阵（优先一次做完）

所有深度臂均沿用 Phase11 C4 的 S32、Clinical、CONCH、RNA、主干和训练损失；**只能修改本表明确的因素**。测试集患者身份及标签始终不变。

| 组 | 必做实验（每个新训练臂 3 seeds × 5 outer folds） | 回答的问题 |
|---|---|---|
| **A：RNA 内容有效性** | C4-g0@128 原样；C4-NullRNA@128（学得的常量 RNA token，禁用真实 RNA）；C4-WrongRNA@128（train/inner-val/outer-train/test 各自 split 内 derangement）；C4 的**测试时 RNA 置换/置零**（只作固定模型敏感性，非新训练） | 正确患者 RNA 是否产生不可由容量、噪声、初始化解释的收益？ |
| **B：Patch × RNA 因子** | 在 512 Patch 和 1024 Patch 分别跑 **RealRNA、NullRNA、WrongRNA**；128 Patch 复用 A 组；均维持同样的网络、Query 数量和采样嵌套 | RNA 的增量是否在特定 Patch 预算下更可靠？ |
| **C：随机分组稳定性** | 在 128 Patch 用 **g0—g7 共 8 个事前锁定独立随机基因分组**，每个分组跑 3 seeds×5 folds；g0 与 A 组精确复用；严禁从外折挑最佳 g | C4 是某次随机分组幸运结果，还是分组归纳偏置的稳健改善？ |
| **D：生物分组保真** | 128 和 512 Patch：`Top512-原Hallmark+fallback`、`Pathway-enriched Top512`、`Pathway-enriched 512 matched-random grouping`；所有表达基因必须合法 train-fit，披露真实通路覆盖率、缺失率/平台分布与基因交集 | Phase11 真 Hallmark 仅覆盖约4%，是不是无效的先验检验？ |
| **E：预测头依赖性** | 在 128 Patch 运行 C4-g0 **无 C+S32 线性风险加法通路**、单一联合融合 Head（保持 C/S32/P/R 输入）；同配置 NullRNA 对照；容量/优化差异单列 | C4 的结果是否高度依赖强线性风险锚定，而非跨模态机制？ |

**严格定义：**
- `WrongRNA` 不改变 C、S32、Patch、DSS 标签；分别在 train / inner-val / outer-train / outer-test 内错配，且保持癌种和 RNA 缺失模式代理的可行分层，不允许固定点；分层过小必须记录并使用预先设定的回退，不得越折。真实/错配/常量模型须尽量匹配有效容量、训练预算与 dropout 随机数路径；做不到就如实披露。
- `NullRNA` 必须经输入扰动证明**真实 RNA 对最终预测无影响**，但 RNA token 位置/融合网络尽量与 RealRNA 一致；单纯删去一个网络分支不等于严格容量匹配。
- `g0—g7` 的**随机分组种子与神经初始化 seed 独立**；每组保留相同组数、组大小/重叠、训练折均值/方差/缺失率分箱匹配与全局基因 permutation 哈希。**8 个随机分组不能作为 8 组独立患者样本做显著性检验**。
- `Pathway-enriched` 仅根据事先固定的 Hallmark/通路注释和训练折可观测性/方差筛选，不得根据外折 DSS 选择基因；如果可用通路基因不足，明确报告、改走预先记录的可行分支，**绝不补造覆盖率**。为了归因，尽量保持相同 512 基因预算，新增基因与原 Top512 差异须同时设匹配随机集合对照。
- `No-anchor` 是机制诊断，不得因为分数低就私自恢复锚点或更改训练配置；与原 C4 比较时明示容量、风险尺度及优化差异。

## 3. 提前锁定比较与统计

**主要比较（探索性内部确认，需同时看两项指标）：**
1. `C4-g0@128 − NullRNA@128`；2. `C4-g0@128 − WrongRNA@128`。
3. `RealRNA@512 − NullRNA@512`；4. `RealRNA@512 − WrongRNA@512`。
5. `[(Real−Null)@512] − [(Real−Null)@128]`（**配对的差中差**，保持同患者/初始化）。
6. `C4-g0@128 − A1`（系统级参照，**已反复使用此测试队列，仅探索性**）。

次要：1024 对照、`g0—g7` 稳健性、真实 vs 匹配随机 Hallmark、No-anchor、原 Phase11 C3/其他主干公平比较、RNA-only线性 Cox/冻结 Token Cox probe、COAD/READ与 RNA 缺失模式子组。不能把点估计最好的一组事后升为预注册主要比较。

报告**每折事件数、每 seed 的 fold-local Harrell/Uno 分子分母聚合、3 seed 均值/SD、患者级配对 Bootstrap、原始 95% CI 与对 6 个主比较×2指标的多重比较校正区间**。Bootstrap 在原患者层采样，三个 seed 共用采样索引；不得把 15 folds 或 8 种分组当独立患者。明确 Bootstrap 不包含重训/模型选择方差，本队列反复探索导致所有推断均为 exploratory。若主要条件不成立，结论写 `INCONCLUSIVE/NOT_SUPPORTED_IN_THIS_CONFIGURATION`，不制造“突破”。

## 4. 审计与 Bug 处置（不可省略）

- **独立评估器**不能 import 训练指标逻辑：读取私有原始 OOF+标签+fold，独立重算 Harrell/Uno（与 `scikit-survival` 数值交叉核验）、配对 Bootstrap 与差中差；从原始 fold CSV 再重建所有汇总表。验证预处理、patient/slide 分折、错配映射、Query/K/V 方向、RNA/S32 输入响应、模块梯度和有效参数、A1 锚点重拟合、最终 checkpoint 逐例回放。
- 测试扰动：随机改变**外折测试 DSS** 不得改变训练权重、特征选择、epoch、初始化；合法外折 RNA/Clinical/S32/Patch 扰动只应改变测试预测。故意注入**泄漏、错折、错患者、失活 RNA/S32、错 Cox 风险方向、wrongRNA 跨 split、padding 参与 softmax、损坏 OOF/checkpoint**，验证器必须实际拒绝。
- P0：一切可能改变结论的训练/数值/数据 Bug → **暂停、保存旧源码/旧结果并标 INVALID、定位受影响 cells、修复后完整重训**；P1：统计或消融不等价 → 定量判断受影响范围，必要时重训；严禁看完外折结果再换筛选门槛却称原预注册。
- **Codex 自己的 PASS 不构成最终验收**；要交付训练脚本、真实模型 forward、逐折配置、checkpoint/OOF 索引和哈希、独立评估脚本、完整负结果及 Bug 修复证据。隐私患者级材料留服务器，公开只上传代码与匿名聚合报告；授权下可由外部审计员在服务器重跑。

## 5. 交付与研究结论

`reports/phase12_c4_confirmation/` 必须包含 `PROTOCOL_LOCK.json`、`MODEL_MATRIX.csv`、`RESULTS.md`、`PRIMARY_PAIRED.csv`、`FACTORIAL_PATCH_RNA.csv`、`GROUPING_8_REPLICATES.csv`、`PATHWAY_COVERAGE.csv`、`RNA_SIGNAL_PROBES.csv`、`TRAINING_DIAGNOSTICS.csv`、`BUG_LOG.csv`、`SOURCE_REGISTRY.json`、`INDEPENDENT_VALIDATION.json`、`FAULT_INJECTION.json`、`REPRODUCE.md`、`DECISION.md`；保留不可公开的 OOF/checkpoint 于私有目录并提供可复算命令。

**最后必须直接回答：**
1. C4 真 RNA 是否同时优于 NullRNA 与 WrongRNA？正向是否跨 seed、患者抽样稳定？
2. g0 是随机分组特例吗？g0—g7 方差有多大？不能以最好的 g 作为新 SOTA。
3. 512 Patch 是否提高 **RNA 真实增量**，而不只是全模型总 C-index？差中差支持吗？
4. Hallmark 加强真实覆盖后是否优于匹配随机先验？是否是生物语义收益？
5. 去掉线性风险锚点后还成立吗？
6. 关键统计是否经**独立重算**？有哪些未解决 Bug 和证据边界？

**执行规则：**先 Hard Gate，再锁定矩阵；运行完整实验，不要中途向用户反复索要逐步确认。算力不足时先完成 A+B 的 128/512 对照，再做 C+D+E；完整记录哪些预锁实验未完成，禁止拿残缺矩阵称全部完成。最终提交 GitHub 分支、commit SHA、运行说明和公开审计报告。不要只生成实验计划，必须实现、运行、验收并推送实际证据。
