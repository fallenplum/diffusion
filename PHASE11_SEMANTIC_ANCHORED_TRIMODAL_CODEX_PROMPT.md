# PHASE 11 — 强 TITAN-S32 语义锚定的三模态深度融合、RNA 表征保真与 Patch 预算机制验证

> **执行方：Codex（负责代码、运行、独立验证、提交 Git、交付审计包）。** 这是可直接执行的实验指令，不是仅供讨论的研究提纲。
>
> **研究目标：** 在 TCGA-COAD/READ 既有 512 例、65 个 DSS 事件的患者级固定协议中，系统检验 **Clinical (C) + TITAN 病理语义 S32 / Pathology Patch (P) + RNA (R)** 的深度融合，确定组学是否能在 **保留最强 Clinical+S32 表征**的条件下，通过基因—形态交互提供可泛化的生存排序信息。
>
> **当前明确不做：** 扩散模型、缺失模态补全、生成模型、跨模态重建、仅为组合而组合的 MoE/GNN、纯患者级风险残差作为主模型。CNA 仅作为小规模探索分支，不得改变 RNA 主问题。
>
> **基本原则：** 尽量一次完成多个机制与相互区分的消融，而不是跑一个模型就继续向用户请示；**先核实代码和数据 → 预锁方案 → 多机制实验 → 独立复核 → 按预设标准裁决**。不能为了取得正结果看测试后修改机制、比较组、病理编码器或指标。不允许只提交一份自称 PASS 的报告。
>
> **重要诚实声明：** 当前 512/65 队列经历了历史多轮研究，本阶段即使看到改善，也属于在同一队列上持续研究后的**探索性内部证据**；重复初始化 seed 不是独立临床队列。未经新队列确认，不得用“稳定 SOTA / 临床可用”措辞。

---

## 0. 启动条件与 Git 基线（必须实际执行）

1. 仓库：`fallenplum/diffusion`。以最新审计完成的 `codex/phase10-trimodal-mechanism-audit` 为历史基线，当前已知提交 `9040f5cb32d311832b3107bbba1569a8f6f98761`；**必须 `git fetch` 并核查实际分支 HEAD、数据、SHA，不能仅信这条历史信息**。
2. 新建 `codex/phase11-semantic-anchored-trimodal`；不得覆盖 Phase10 原始 checkpoint、OOF、`reports/` 或私有数据。新建私有实验目录 `/root/phase11_semantic_anchor_private/`，运行目录不可与 Phase10 共用可写输出。
3. 精读并记录来源版本：
   - `reports/phase10_trimodal_mechanism_audit/{RESULTS.md,DECISION.md,KNOWN_LIMITATIONS.md,DATA_AUDIT.md,INDEPENDENT_AUDIT.md,MODEL_MATRIX.csv,PROTOCOL_LOCK.json}`；
   - `src/phase10_trimodal/{models.py,features.py,train.py,math.py,reference.py,data.py}` 与对应 `training_source_v2/`；
   - `reports/phase6a_titan_semantic_preflight/RESULTS.md`；`reports/phase6_conch_v15_download_audit/RESULTS.md`；
   - `reports/phase8_v0r/simple_ridge_controls/RESULTS.md`、M3/M5/M6/M7/M9、PO/CPO 已完成的机制记录；
   - SurvPGC 官方 `Houjiaxin123/SurvPGC`、CiMP 官方 `DeepMed-Lab-ECNU/CiMP` 相应网络文件；保留 paper-versus-code 差异表。
4. 新实验报告路径：`reports/phase11_semantic_anchored_trimodal/`。创建 `SOURCE_REGISTRY.json`，包括当前 Git SHA、每份历史参考文件的 ref/path/sha、运行的真实依赖版本、CUDA/GPU/CPU 信息、数据及缓存 hash、特征编码器版本。
5. **务必先判定当前主任务和 Phase10 的不同：** Phase10 B/C/D 深度神经网络基本使用 `C+CONCH Patch+RNA`，而其强 A3 为 `C+TITAN-S32 Ridge`。本轮目标是把 `S32` 实际输入到神经计算图，并验证该输入如何参与机制，不能仅在结果表放上 A3 就声称“语义锚定”。

## 1. 本阶段待验证假设（不得预设成立）

- **H1，强语义锚定：** 把 TITAN-S32 显式输入深度 C+P 网络，能改善、或至少更准确诊断 RNA 融合时对强病理语义的利用；它与仅 CONCH Patch 的输入不等价。
- **H2，组学—形态交互：** RNA 与 S32 病理概念、局部 CONCH Patch 的交互，可能比患者级 concat 更有效；效果必须超出匹配容量/无 RNA/错患者 RNA 对照。
- **H3，组学保真：** 对 RNA 的某些分组压缩或 GroupMSA 处理会丢失 DSS 相关信息；保留 Top512 低维主效应可解释性与 Token 表示能力，可能改变交互结果。
- **H4，病理信息预算：** Phase10 每切片仅 128 Patch（约 5.2% 原始 Patch），可能限制有效 RNA—形态交互；扩大到 512/1024 必须在相同输入/架构/训练规则下作对比。
- **H5，临床—病理路径：** Phase10 `B0-B2` 对临床—病理路径的改善具有描述性支持，但这一作用能否在 S32 条件下保持、以及组学是否额外增加价值，尚未确认。

所有方向均以**可证伪的机制对照**为中心。若 H1–H4 不获支持，保留所有负结果，不能事后宣称只是 seed 或训练不足，除非有预先规定的优化诊断和重跑证据。

## 2. 数据与信息边界（Hard Gate D：不过不得正式训练）

### 2.1 队列和终点

- 固定 512 患者、65 DSS 事件，使用已有 **patient-level 固定五外折**，禁止按 Slide 随机分折。DSS 时间/删失方向、同日事件、多个 WSI 病例的映射必须复核；有一例缺失原 DX1，但 DX2 存在的情况，不能偷偷补切片或重复同患者标签。
- 保留原合法 Clinical 字段 `AGE, SEX, AJCC_PATHOLOGIC_TUMOR_STAGE, PATH_T_STAGE, PATH_N_STAGE, PATH_M_STAGE`，另做仅 AGE/SEX/Stage 的敏感性分析；不得使用 DSS、OS/PFI、未来治疗、随访结果、转归文字等泄漏字段。
- **S32 为固定 32 个形态/临床病理概念的 TITAN 病理相似度分数，而非 TITAN-PCA32**。核实来自固定的病理/文本模型与 outcome-blind 描述词；PCA32 另作为不同基线。S32 的概念形成不得利用 DSS 或本轮外折测试表现调词。
- 读取实际私有 `patient_representations.npz`、Phase10 私有队列与原始切片匹配清单，确保 512 例顺序和 hash 一致；不要把用户 ID、原始矩阵、患者级 OOF、checkpoint、密钥提交 Git。

### 2.2 RNA / CNA

- RNA 原登记 `(512,20531)`，经去无符号/重复合并为约 `(512,20511)`；CNA 原登记 `(512,25131)`。**运行时重审并以真实文件为准**。不可用测试折计算均值/中位数/方差/基因排名/通路筛选/PCA/监督基因集。审计 RNA 309 完整患者 + 203 患者缺 3024 个基因的模式；不把此模式无证据地写成已确认 GAII 平台身份。
- RNA 缺失中位数填充、标准化和 Top512 选择全部 fit 在 outer-train；选择 epoch 的 inner 训练亦必须 inner-train-local 重拟合。将 gene symbol 索引、排序 hash、缺失 mask、各组 token 真正包含的基因号和输入形状写入 manifest。
- CNA 单独 gain/loss 特征，**不许与 RNA 简单混成 1 个伪转录组**。如做探索，必须另列样本输入与容量，不放入主要机制因果结论。

### 2.3 病理数据与 Patch 网格

- 服务器应有 CONCH v1.5 的 519 张匹配 WSI（512/512 患者）H5：`features(N,768)`，20X/512px，约 1,278,642 原 Patch；但**只接受实际实查清单、SHA 和特征范围，不信既往报告就默认齐备**。核实 519 个 H5，坐标一致/整数/唯一、非空、有限值、病理 encoder 属性。
- 原 UNI2-h 为 20X/256px，CONCH v1.5 为 20X/512px。**绝对禁止按 Patch 行号配对，禁止在未经验证坐标映射时声称二者具有局部空间一一对应**。
- S32 / TITAN slide 表征与 CONCH patch 的关系是**患者/slide 语义条件输入**，不是已经由标注证明的 Patch ↔ 某个文本概念对应。机制解释只能称“学习到的条件相关性”，不能直接称病理专家定位证据。
- 对每位患者记录真实使用的 slide、原始 Patch 总数、每档实际采样数及复用情况，严禁患者因 Patch budget 改变而从队列消失。128/512/1024 在病例和 Slide 集合上完全一致。

**Hard Gate D 的程序化测试：** 故意将 test DSS/Stage 输入、S32 输入、RNA 输入和 test Patch 修改，检查训练折预处理、特征筛选、训练初始化、选 epoch 不被 test 数据影响；合法测试输入改变可以影响 test risk，但不能影响 fitted parameters。所有阶段写明针对何种扰动的 expected behavior，不能把“修改合法 test Clinical 不应改变预测”写成不现实的不变性要求。

## 3. 机制核心：所有深度三模态模型必须显式使用 S32

统一符号：合法临床 `C∈R^{d_c}`；S32 概念 `S=(s_1,…,s_32)`；原 CONCH patch token `P∈R^{N×768}`；RNA 表达 `R∈R^{512}`（Top512，训练内选择）；所有模型维度/参数量与计算预算固定后公开。

### 3.1 语义 Token 化，不是假装直接具备 Patch—文本对齐

设计一个可审计的 `ConceptTokenizer`：

```
E_s[k] = LN( MLP_scalar(s_k) + trainable_concept_id[k] )
S_tok = stack(E_s[1],...,E_s[32])      # [B,32,d]
```

- `trainable_concept_id[k]` 表示固定词项身份的可训练嵌入，并**不自动意味着学到医学语义**。如采用官方 TITAN 文本嵌入必须提供真实编码器、版本、维度、投影/L2 与对应词项查证；绝不混用旧 CONCH text v1 与 CONCH v1.5 embedding。
- S32 scaler 仅 fit train fold。临床将每个字段编码成可解释 token（与 Phase10 Clinical Ridge 输入同源），另保留标准化数值 baseline。
- `S32_used_by_model` 必须由单元测试证明：替换 S32 后，预测与至少一个可训练模块的梯度发生响应；若把 S32 清零而 B/C 模型完全不变，硬失败。
- 不要让 `s_k` 在 Transformer 前被 sigmoid 到常数、被错误 reshape 成 batch/seq、或通过 `.detach()` 阻断计算。初始条件中数值输入可冻结，但经其驱动的可训练 Q/K/V 权重必须收到有效梯度。

### 3.2 关键机制 1：病理语义引导局部 Patch

设 `Q_S` 来自 S32 token、`K_P,V_P` 来自 Patch 表征：

```
H_{S<-P} = CrossAttention(Q=S_tok, K=P_tok, V=P_tok, key_mask=patch_mask)
```

该块对局部 Patch 的语义条件聚合允许 32 概念的不同检索结果。必须与 `ConceptMeanPool`、`UnguidedPatchAttention` 和匹配容量的无 cross-attention 表征比较。禁止因为使用了 `CrossAttention` 名称就默认跨模态交互有效。

### 3.3 关键机制 2：RNA 与 S32 / Patch 的交互

至少实现三种**机制不同、可定位 Query 方向**的版本：

- `RNA→Patch`：RNA Token 作为 Query，检索 Patch Key/Value；同时保留独立的临床—S32/Patch 路径，参考 SurvPGC 的双路径思想。
- `RNA→S32`：RNA Token 与 TITAN 概念 Token 交互，测试“组学与形态语义”是否比“组学与原 Patch”更容易对齐；注意不能把这种统计交互解释成真实基因—概念因果关联。
- `C→RNA→Patch` 三层引导：先用合法临床 Token 选择/重标定 RNA Token，再让 RNA 条件化检索 Patch，同时保留 S32；参考 CiMP 的条件化 Token/Bottleneck 思想，但明确标注 **ADAPTED/INSPIRED 而不是原文复现**。

另外可比较 `Patch→RNA` 反向和双向互交；每个变体只改变关注的机制，维度、Patch、临床输入、RNA 特征集合、优化器、训练预算不变。**核心网络最终必须融合 C+S32+P+R，而不是像 Phase10 那样仅以 C+P_patch+R 与静态 A3 比较。**

## 4. 预先锁定实验矩阵（主要 26 个 arms；敏感性可拓展，不许按 test 筛选）

### A 组：强基线和输入保真（8 arms）

| 编号 | 输入/架构 | 必须回答的问题 |
|---|---|---|
| A0 | Clinical-only，训练内正则 Cox | 重新确认临床强度 |
| A1 | Clinical + S32，训练内正则 Cox | **当前真正的主要系统级锚点**，目标约 Phase10 A3，但必须全量重算 |
| A2 | Clinical + TITAN-native PCA32，训练内正则 Cox | PCA32 与 S32 是不同表示 |
| A3 | RNA-only Top512 Cox | 基因组预测信息是否还在 |
| A4 | Clinical + S32 + RNA Top512 Cox | 简单高维加入 RNA 的稳健程度 |
| A5 | Clinical + S32 + CONCH Patch，统一浅 Token backbone，禁止 RNA | **匹配深度输入的 CP anchor** |
| A6 | A5 + `S32→Patch` 概念查询 | S32 有无在表征学习阶段发挥作用 |
| A7 | A6 + RNA（匹配容量、无跨模态 attention 的简单 concat） | 可归因的三模态融合对照 |

说明：A5/A6/A7 使用共同的 S32 token、Patch 输入、Clinical 编码和相同训练/评价；如果某种结构为了避免空分支而改了参数，必须报告可训练参数与实际活跃参数，并提供同容量对照。A1 是历史选中过的强 anchor，必须明确其在此队列上属于探索性强参照，而不是独立未看过的测试赢家。

### B 组：跨模态对齐核心（8 arms）

| 编号 | 在同一个 C+S32+P 骨干上的改变 | 主要比较 |
|---|---|---|
| B0 | 双路径 `C/S32↔P` 与 `RNA→P`，两路径独立 QKV | B0−A7、B0−A6 |
| B1 | `RNA→S32` 概念交互取代 RNA→P；保留 C、S32、P、RNA | B1−B0 |
| B2 | 三元 `C→RNA→P`，并保留 S32 语义条件路径 | B2−B0 |
| B3 | `P→RNA` 与 B0 原 Query 方向互换 | B3−B0（描述性） |
| B4 | 双向 `RNA↔P`，容量匹配 | B4−B0（描述性） |
| B5 | B0 但取消 S32→Patch 的内容依赖交互，保留原始 S32 输入晚融合 | B0−B5：语义查询机制 |
| B6 | B0 但移除有效 RNA 路径；结构/输出槽/预算尽量匹配 | B0−B6：RNA 的可用贡献 |
| B7 | B0 的 **wrong-patient RNA**（训练、内验证、测试各 split 内独立错配） | B0−B7：患者对应性 |

**不能使用误导性名称：** B6 不可以替代使用 Clinical 特征却仍称为 RNA 控制；如有容量补足 surrogate，列为“clinical-surrogate, no RNA”，同时做完全无 RNA 的直接对照。Wrong-patient 允许固定癌种和 RNA 缺失模式分层，Stage-matched 为预先定义的单独敏感性，不能用 DSS 标签分层。

### C 组：组学 Token 表征保真（6 arms）

共享 B0 的三模态语义锚定与匹配 Patch backbone，只替换 RNA 编码方式或监督约束。**RNA 输入 genes 必须相同**，只有单独的 TopK 敏感性可改变 gene set。

| 编号 | RNA 构造 | 要判别的机制 |
|---|---|---|
| C0 | 8×64 顺序分块 Token（Phase10 可核查版本） | B0 的 RNA 编码参考 |
| C1 | 8×64 分组后独立轻量投影，无 inter-token 内容依赖注意力 | GroupMSA/注意力是否有益 |
| C2 | Top512 gene-wise soft selection/attention 汇聚为等预算 Token | 基因信息是否能在学习中动态保留 |
| C3 | 锁定真实 Hallmark 通路分组，组内 gene aggregation + Token | 生物组结构是否提供贡献 |
| C4 | C3 相同组数/大小/重叠结构，训练内固定 outcome-blind 随机分组 | 排除“只是分组降维”的伪机制 |
| C5 | C1 + RNA-only 辅助 Cox 监督（固定超参数、同额外辅助头的 inactive control） | 组学表征监督保真，不能把性能变好直接解释成 BMLSurv 复现 |

- **重要：** Phase10 C0/C4 中的 GroupMSA 实现与这里的 C0 编号无对应关系；所有表格写完整文字模型名，避免 ID 跨期混淆。
- C0（本阶段）与 B0 可直接复用**同一结果**，前提是它们完全同配置/初始种子/输入/源码哈希，必须严格核对 OOF 逐元素后记录 `REUSED_IDENTICAL`；否则都实际训练。
- 对所有 C 组编码器，在相同 Top512 RNA 输入和患者级测试折上执行 **RNA-only frozen representation Cox probe**（训练折拟合），同时报告投影前 Top512 的原始线性 Cox。Probe 的λ需要在训练折内部选定，不许 test 选择。不同 probe 维度应做维度匹配敏感性，不能把低 C-index 宣称成信息论意义的信息损失。

### D 组：语义强锚点和 RNA 的负对照（4 arms）

| 编号 | 设置 | 检查点 |
|---|---|---|
| D0 | B0，但将 S32 替换为匹配统计分布的无患者对应 concept scores，训练/测试分别 split-local 置换 | 验证患者级 S32 条件信号，而非仅 Token 容量 |
| D1 | B0，但 S32 只进入最终输出 head，不作为 Attention Query | 检验语义引导 vs 普通表征拼接 |
| D2 | B0，Stage/T/N/M 降级为 Age/Sex/Stage 合法临床子集 | 检验临床强先验冗余对 RNA 互补性的影响 |
| D3 | A6 + patient-matched RNA-only auxiliary representation（**不允许进入预测**），同预算负对照 | 检查新增参数/监督是否被误认为 RNA 融合收益 |

D3 的 no-prediction 监督头必须没有到主预测的错误路径；如无法实现完全匹配活跃容量，明确只可作诊断，不做纯机制因果比较。注意错配 `S32` 也应在训练/测试分开做，绝不能引入 test 临床标签作为置换指导，除非 Stage 是合法输入并且事前单列敏感性。

### 重要数量与预算

- 上述 **8+8+6+4=26 个主矩阵 arms**。核心正式执行：**26 arms × 3 seeds × 5 outer folds = 390 条记录**（若 B0/C0 真正相同，可复用相应 15 条记录，报告 actual fits ≠ records）。正式 seeds 建议 `31,47,83` 以便与 Phase10 配对；历史 seeds/固定患者测试集意味着 **探索性纵向比较**，不构成新的独立确认。
- **不能在外层测试看到正向结果后，只增加赢家 seed 或 Patch budget。** 如果算力受限，必须按事先定义的优先级完整执行：`A0–A7、B0–B2、B5–B7、C0–C2、D0–D1` 优先；剩余 arm 标为 `RESOURCE_DEFERRED`，不能在主报告中写“26 arms COMPLETE”。完整矩阵与部分完成都须准确计数。
- 为了公平，大多数深度模型应统一 hidden dim、预训练 patch features、inner validation、优化器、weight decay、epoch selector、有效训练预算。训练时间、峰值显存和活跃参数数必须按 cell 记录。不同模型导致的结构性额外参数不可隐瞒。

## 5. Patch 预算敏感性（独立预锁；不得挑外折赢家）

阶段 A–D 的 Patch 基准为 **128 / slide**。另预注册以下独立预算矩阵：

| ID | 模型 | Patch 数 | 目的 |
|---|---|---:|---|
| P0 | A6：C+S32→Patch，无 RNA | 128 | 对照主结果，可复用 |
| P1 | A6 | 512 | 病理强锚点随信息量的变化 |
| P2 | A6 | 1024 | 同上，高预算敏感性 |
| P3 | B0：语义锚定 + RNA→Patch | 128 | 对照主结果，可复用 |
| P4 | B0 | 512 | RNA 交互是否依赖更多局部区域 |
| P5 | B0 | 1024 | 同上，高预算敏感性 |

- Patch 选择采用**同一患者、同一 slide、基于坐标的稳定排序/覆盖策略**，构造一个 outcome-blind **嵌套的采样顺序**，使 128 的集合真包含于 512，512 真包含于 1024；不得三种不同独立随机抽样后直接把差异解释为“Patch 数量”。在至少 3 张不同 Patch 分布的 WSI 上检查嵌套真包含；完整队列统计有效覆盖数及 patch count 分布。若原始 Patch 数少于预算，使用实际数并 mask，不补重复 Patch 冒充真实信息。
- `P0/P3` 精确复用仅在完全相同输入 Patch subset、seed、模型和风险数组时允许；若原 128 Patch 规则不同，则须重跑。
- 1024 Patch **不应使用全局 O(N²) Patch self-attention 当作唯一实现**。优先用 concept/gene Query → Patch Key/Value 的线性 Query×N 交互或局部池化，不改变关键机制；固定 Q 数，必要时用 gradient checkpoint、分患者 streaming 前向 / 全风险集损失。严禁静默截断 1024 回 128。
- 正式 P1/P2/P4/P5 每个做 3×5（60 个新增 cells）。先做无标签、无外折测试指标的 GPU 显存/吞吐 preflight；如高预算物理不可运行，写 `RESOURCE_BLOCKED`，保留 512 档，不可伪装 COMPLETE。
- **Patch 预算的改变会改变信息来源、训练开销和优化难度**。必须比较训练曲线、选择 epoch、性能与真实计算量，不能只看单个 Harrell 数值。

## 6. 训练协议：小事件数下正确性优先

1. 固定患者五外折；每个 outer-train 中生成**严格嵌套的 inner train/val**。所有超参数和 epoch 仅根据 inner validation，外层测试只被一次正式预测使用。多个候选机制共用相同预定义预算，不得看测试后挑训练长度。
2. 所有新模型采用 **full training-partition Cox-Efron partial likelihood** 或经过数学证明和显式单元测试等价的全风险集流式计算。禁止直接对 mini-batch 计算 Cox 再平均冒充 full-risk-set。Patch 编码可分块或重算激活，但每个事件的风险集必须覆盖正确的全训练样本。
3. 仍以 `Harrell C` 与 `Uno60 C` 为联合主要指标；务必准确处理事件方向、相同时间与预测风险 tie、删失分布，报告 fold-local numerator/denominator micro。风险不同折的尺度不一定一致，直接 pooled OOF 只能列作敏感性，不得在不同方法间偷换口径。
4. 训练预算不能沿用 Phase10 40 epoch 就称充分收敛：Phase10 记录 6 个最大边界 epoch 仍在下降、33 个选最小允许 epoch，另有大量末段验证损失回升。预先固定 **inner max=80 epoch，minimum=10，EMA/early-stop 由 inner-val 选 epoch，统一上限和 patience**；如 GPU 显示所有模型在 80 epoch 仍有未饱和情况，须依据**仅 inner 诊断且预先定义的资源门槛**整体统一延长（例如所有核心 arm 同条件最大 120），不能只加预算给某个外层结果好/差的模型。
5. 选择 epoch 后，必须**重新初始化**模型，在完整 outer-train 运行被选定的 epoch，预测 outer-test。不能把 inner 模型训练完后继续在 outer-train 上补几轮当作同样的 refit；也不能用 outer-test 的 Cox/Uno 选择 epoch。
6. 维持较小 hidden dim（64 或 128）并报告神经编码器可训练参数量，且明确所有基础病理特征由冻结预训练模型计算，不要写成端到端训练 CONCH/TITAN foundation weights。可通过不同有效维度做事先锁定的容量敏感性，但不要因某模型弱就私下扩容。
7. 对所有固定线性对照，使用同样的训练内 lambda 选择；优化器应通过梯度/数值残差门槛。若数值优化未收敛，不得自动判失败机制。
8. 如果使用生存蒸馏/辅任务，严格无未来信息，aux head 和权重实际参与主干训练；如 only architecture budgets control 要标注活跃参数差异。

## 7. 机制级对照与输入真正生效验证（Hard Gate M）

以下为强制单元/性质测试，不得通过结果高低替代：

1. **S32 真的生效：** 在固定权重下打乱/清零 S32，B0 的风险或重要中间层响应；A5、B5 对应分支的响应必须符合各自架构，而不是所有模型均一致。对概念 Token / Scalar Projection 的梯度检查至少 2 个随机初始化 + 2 个非平凡 batch。
2. **RNA 真正生效：** B0 改 RNA 后风险改变；B6（无 RNA）对任何 RNA 扰动严格不变；RNA-only/模型主干的梯度与目标一致。对于 wrong-patient 分支，应审计 sample mapping 只在本 split 内，并且不出现固定点。
3. **S32 患者身份：** D0 错配时仅改变 S32 分数对应关系，不改变 Clinical、RNA、Patch、患者标签；合法 Stage 配对条件若做须单独分层注明。
4. **CrossAttention 的 Q/K/V 方向：** 构造小矩阵手算 oracle，比较 learned attention 的 softmax 轴、scale=1/√d、Key Padding Mask、输出 token 顺序；切换 B3 时确实交换预期方向，不是简单交换变量名字。
5. **患者与 Slide 聚合不变性：** 同一 slide 的 Patch 顺序随机置换不应影响无坐标敏感的结果；如果引入显式空间编码，需要与坐标同步置换。多个 Slide 的顺序变化不改变患者预测。测试 Patch mask 的 padded token 内容改变不会影响输出。
6. **梯度路径：** 交互模块、RNA 编码器、概念查询、Patch 投影、final head 均必须在预期实验接收到非零且有限的梯度。某些阴性臂应无梯度的路径要在报告说明。记录前/后训练 probe、梯度分布、最终修改风险和 attention entropy；梯度范数不能被直接解释为模态因果贡献。
7. **无偷换输入：** A1 C+S32 不能默认为 PCA32；CONCH Patch 模型不能拿 UNI2 native 1536 直接送 768 projection；RNA no-input 对照不得注入 clinical-surrogate 然后声称“去掉 RNA 的纯消融”。
8. **复现检验：** 使用一组固定随机模型、冻结输入对比 PyTorch forward、独立小矩阵手算与 checkpoint replay；如果 S32/RNA 某支没有连到最终 loss，直接 STOP，不允许带 bug 跑完整表。

## 8. 统计比较与门槛事先锁定（Hard Gate S）

### 8.1 主要比较家族：预定义，不随外折结果更改

固定至少以下 **8 个主要机制比较**，每个同时对 Harrell 和 Uno60 作配对：

1. `B0 − A7`：RNA→Patch 深度交互是否优于匹配输入的简单三模态 concat。
2. `B0 − A6`：在**相同 S32+Clinical+Patch 骨干**上 RNA 是否真正提供增益。
3. `B0 − B6`：RNA 模态消融是否改善/退化；容量不完全相同需列限制。
4. `B0 − B7`：真实患者 RNA 是否优于错配组学。
5. `B0 − B5`：S32 的内容依赖 Patch 查询是否优于后端直接输入。
6. `B1 − B0`：RNA→S32 相比 RNA→Patch 的不同语义对象。
7. `B2 − B0`：临床指导 RNA/病理的三元交互增益。
8. `C1 − C0`：RNA tokenwise 编码是否比 Phase10-style inter-token attention 更稳健。

另外把 `B0 − A1`、`B1 − A1`、`B2 − A1`、`A6 − A1` 及 `P4 − P3`、`P5 − P3` 列为**预先指定的强系统 anchor / Patch budget 证据**，但 A1 是同 cohort 历史已见过的赢家，应归类为探索性系统比较，不能假装新独立测试；不要仅比较较弱的 inner CP_selector。

### 8.2 统计执行

- 对三个 seed 的**相同患者、相同外折**，按 fold-local 可比较患者对分子分母聚合，得到每 seed 一个 Harrell/Uno；然后报告三个 seed 的均值、方差和逐 seed 方向。Ridge 结果即使重复写三个 seed，也不是三个独立拟合样本。
- Bootstrap 使用**患者级共享重复抽样索引**，同一抽样作用于所有模型/seed；同一患者多次出现时必须正确处理重复与自比较，不可把 15 seed-fold 当作独立患者级样本。
- 正式主要 family 至少 **20,000 次**患者级 paired bootstrap；建议 30,000–50,000 次以改善 Bonferroni 极端尾部量化稳定性。比较 8 对 × 2 指标共 16 个方向时，采用 Bonferroni/Holm 预定方法并报告 simultaneous CI；未校正 95% CI 仅描述性。记录实际 resamples、随机种子、极端分位数 Monte-Carlo 精度和置信区间是否包含 0。
- Bootstrap 仅衡量已训练预测的患者抽样不确定性，**不包含重新训练/inner 选模型/架构选择方差**，不得将其夸大为独立泛化证明。由于重复研究过同一外层测试患者，统计显著性只能定位为**探索性**。
- 所有 fold / seed 展示 Harrell、Uno60、测试事件数、比较对数量、模型运行时间、最差折表现；同一主要结论必须两指标方向一致，并有机制对照支持。负结果原样保存。
- 对 RNA 的性能损失与 Patch budget 敏感性，增加不根据外折结果选择的 subgroup（COAD vs READ、Stage、RNA 缺失模式）描述性分析。子组事件数极低时明确不可判定、不要画虚假的显著性星号。

## 9. 独立正确性审计（严禁 Codex 自证即终审）

### 9.1 真正隔离训练与评价

独立程序 `scripts/validate_phase11_independent.py` **不得 import 训练数学/指标模块**，应从**原始私有 OOF risk、time、event、fold manifest** 直接重新计算 Harrell/Uno（独立实现 + `scikit-survival` 交叉验证），并独立抽样重算 bootstrap。每个正式 cell 必须比较两实现指标误差，不能只核对历史 `RESULTS.md`。

必须审查：

- 患者 hash、每外折 test ID 列表、event/time 与原始标签的独立重建；
- RNA/S32/Clinical 的 outer/inner 预处理状态仅使用合法训练索引；
- 所有神经 checkpoint → 对相同私有输入重新 forward，逐患者预测与归档 OOF 数组比较（设置预定义浮点容差）；
- `A1` Clinical+S32 Ridge 重新拟合后与 checkpoint 系数/风险复现（不得只检查旧分数接近）；
- 同 seed 同 fold 的复用 arm 的 risk **逐元素完全一致**（否则不得申报重复利用）；
- 病理 patch mask、slide 映射及 128→512→1024 集合包含；
- 所有有意 zero/wrong/constant 模态对照的输入和输出证明；
- 多模态 attention 方向与 Softmax 归一化轴、梯度、auxiliary head 是否真的生效；
- 独立 `.py` 验证器本身在带 tie、删失、0 events、输入错误、重复抽样、极端风险的人工合成例子上通过。

### 9.2 红队故障注入

至少故意注入以下错误，验证审计器能够拒绝：

1. 把 `DSS_time` 别名藏进 Clinical 字段、S32 生成 Prompt 或 RNA 基因选择；
2. 将同一患者的两个 Slide 分别放入 train/test；
3. 使用全 512 例拟合 PCA/标准化/TopK，却伪装成 fold-local；
4. 将 CONCH/UNI2 不同网格错误按行配对；
5. 把 S32 token 或 RNA token 在最终计算图中意外断开；
6. 使用 mini-batch Cox 替代全风险集 Cox；
7. 将被遮盖 Patch 改成极大值仍能改变输出；
8. 将错患者 RNA 错配到外层 train/test 之间；
9. 把 `test risk` 方向反转或反转 event/censor 标记；
10. 元数据写错（arm/seed/fold/source SHA）、存储的 OOF 与 checkpoint 不对应。

每个故障注入保存：错误注入文件、应该触发的规则、实际失败日志、修复后 PASS。**如果验证器检测不到故障，验收不通过**。合成注入不得污染真实实验或提交真实患者数据。

### 9.3 Bug 修复规则

- **P0 级（可能改变主结论）：** 泄漏、错误分折、RNA/S32 特征混用、模型某分支失活、错误风险方向、损失函数风险集错误、checkpoint/OOF 错配、outer-test 选择。发现立即停止，冻结旧结果并标记 `INVALID`，修复后按影响范围**完整重训**并重新评估；绝对不能只修改汇总 CSV。
- **P1 级（可能改变少量 cell 或推断区间）：** patch mask、数值精度、bootstrap 计算、模型容量不等价、训练未收敛。保留旧日志及 diff，定量定位所有受影响 arm/fold/seed，必要时整组重跑，不得无证据称不影响。
- **P2 级（展示或元数据）：** 允许只修文档，但必须保留原始文件 SHA、说明数据和 risk 未变，并独立抽样证明。
- 不得把验收 `FAIL` 静默改成 `PASS`、清理原失败日志、重命名 arm 来掩盖不等价、因为测试结果不理想就调整训练规则。修正发生在看过外折表现之后，必须标记 `POST_HOC_REPAIR`，所有后续受影响比较的确认性状态降级，不能重新声称未经测试选择。

## 10. 多机制实验的合理停机与资源纪律

- 先运行**不涉及正式外层 DSS 指标**的 CPU/GPU 小样本合成 smoke、满配置 shape/显存/吞吐测试，测定 batch size、梯度 checkpoint 和 Patch budget 能否运行。
- 先完成 Gate D/M/S 和全部 A 组基线；然后按预注册优先级一次性运行主要 B0–B7、C0–C5、D0–D3；Patch 扩展使用相同预锁规则，不因看到外层正/负结果而临时取消某组。
- 一个 512 病人的 Cox 风险集不等于把所有 WSI Patch 一次装入 GPU。如果显存不够，可分患者编码并保留计算图、使用梯度 checkpoint 或经数学检验的流式全风险集反向，不能改成错误的 mini-batch Cox 损失。
- 统计进程和大量模型训练分离，以免 CPU 并行过多导致 GPU 饥饿；记录样本加载耗时、GPU busy、峰值 VRAM、真实 wall time。不要用单次 `nvidia-smi` 快照解释全过程。
- 至少每完成一个预定义 family，输出阶段性报告，**但不要基于外层测试指标调整下一阶段固定模型列表**。
- 确有算力硬阻塞时交付 `RESOURCE_BLOCKED.md`：GPU/显存日志、最小可复现实验、失败原因、替代批量/梯度检查方案、未完成准确 cell 数；**不虚构结果，不以旧模型旧数据替代**。

## 11. 必须交付的结果与项目布局

`reports/phase11_semantic_anchored_trimodal/` 至少包含：

```
README.md
RESULTS.md                    # 所有正式/探索臂指标，且不跨协议混排
DECISION.md                   # 预设主要比较最终判定
PROTOCOL_LOCK.json             # 数据/模型/对照/统计/训练预算/哈希锁定
SOURCE_REGISTRY.json
DATA_AUDIT.md
PATIENT_SLIDE_ALIGNMENT_AUDIT.json
S32_SEMANTIC_SOURCE_AUDIT.md
RNA_PREPROCESS_AND_TOKEN_AUDIT.md
MODEL_MATRIX.csv               # 每个 arm 完整输入、参数、实际活跃参数、token 方向
FOLD_METRICS.csv               # 全部 seed×fold，原始 numerator/denominator
SEED_OOF_METRICS.csv
PAIRED_COMPARISONS.csv         # 逐 seed 变化 + CI + 多重校正
MODULE_ABLATIONS.csv
RNA_SIGNAL_PROBES.csv
PATCH_BUDGET_COMPARISONS.csv
NEGATIVE_CONTROLS.csv
TRAINING_DIAGNOSTICS.csv
ARCHITECTURE_GRADIENT_AUDIT.md
INDEPENDENT_AUDIT.md
INDEPENDENT_VALIDATION.json
FAULT_INJECTION_REPORT.md
BUG_LOG.csv                     # severity、影响范围、修复 SHA、重训证据
KNOWN_LIMITATIONS.md
RUN_MANIFEST.json
PUBLIC_ARTIFACT_MANIFEST.json
PUBLIC_AUDIT_PACKAGE_SHA256.txt
```

发布：可复现源代码、配置、正式消融、聚合统计与独立验证器到 Git 新分支；**原始矩阵、任何可识别患者信息、具体患者级 OOF、checkpoint、保密 token、原 WSI H5** 保留私有服务器。提供 `PHASE11_PUBLIC_AUDIT.zip` 的可访问路径和 SHA256，私有包 `PHASE11_PRIVATE_REPRO.zip` 仅本机/服务器保存，需用户授权才能读取。所有归档与真实文件 SHA 一致，不能发布不存在的链接。

为了使其他审计者能够真的验证，而不是只能看 `PASS`，提供严格的 `RUN_INSTRUCTIONS.md`：依赖锁定、数据目录约定、最小单元测试命令、全量运行命令、独立验收命令、怎样对单个 arm/seed/fold 复跑、如何重建表格与核对哈希。

## 12. 最终回答的研究问题（逐项，不准只给一个冠军模型）

最终 `RESULTS.md` 必须逐条回答：

1. S32 是否真正被输入到每个深度 C+P+RNA 模型？相对于 Phase10，是哪个原先没有参与计算的病理信息被显式利用？
2. `Clinical+S32` Ridge 是否在完全统一的新协议中仍然强？深度 `Clinical+S32+Patch` 有没有匹配其性能？
3. 在匹配强 C+P+S32 输入、匹配病理 Patch 预算时，真正的 RNA 路径是否稳定有增益？真实 RNA 是否优于错患者 RNA？
4. RNA→Patch、RNA→S32、C→RNA→Patch 三种交互哪种在**同协议、同 RNA 基因集、同训练预算**下表现更好？其机制消融是否支持解释？
5. 是否存在 RNA-only 能预测，但组学 Token 经过编码后 DSS 信息弱化的证据？C0–C5 哪个编码较稳健，机制是什么？
6. 真实通路分组是否超过匹配随机分组，还是只是正则化/降维收益？
7. 128、512、1024 Patch 是否改变了 C+P 和 C+P+RNA 的相对关系？变化是否可以和训练欠收敛、输入覆盖率或计算预算区分？
8. 阶段性最优 Harrell 和 Uno 是否来自同一模型？是否同时超过强 `Clinical+S32` anchor？差异置信区间和跨 seed 一致性如何？
9. 有无 P0/P1 Bug，是否影响关键结果？哪些结果经过了真正完整重训，哪些只有文档修正？
10. 下一阶段究竟应当沿哪种**已被支持的机制**创新？哪些方向应停止？哪些在 65 事件下仅能保持 `INCONCLUSIVE`？

### 最终裁决规则

- `SUPPORTED_INTERNAL`：主要比较方向一致、双指标通过已预锁多重校正与三 seed 方向门槛、真实机制/输入路径负对照通过、无影响关键结论的未修复 Bug；仍是内部探索证据，不代表外部验证。
- `DIRECTIONAL_ONLY`：点估计或普通 CI 有一定支持，但 simultaneous 或重复性不足。
- `INCONCLUSIVE`：统计功效/结果冲突/训练预算限制不能回答问题；不是自动等于模型无效。
- `NOT_SUPPORTED_IN_CONFIGURATION`：本设置下明确不支持，但不能把单一阴性实验推广到所有深度融合。
- `INVALID`：确认性程序缺陷导致数据/训练/指标不可信；不得参与模型排名。
- `BLOCKED`：原数据/资源不可用或预定义必要实验未跑完，不能声明完成。

**交付成功的标准是“一个经过代码与统计正确性审核、能解释实验机制边界的完整实验包”，不是“某个模型 C-index 越高越好”。请从代码和数据核查开始，完成这套实验；除必要授权或基础设施阻塞外，不要把工作拆成一串仅询问下一步的对话。**
