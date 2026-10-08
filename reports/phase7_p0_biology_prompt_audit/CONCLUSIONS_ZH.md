# Phase7-P0：Biology-Grounded Prompt Feasibility Audit

**正式完成：512 患者，3 个 inner-split seeds × 5 固定 outer folds，600 份 held-out 预测，120 个真正 seed-level OOF 指标。只使用 T/N/M/Stage，不读取 DSS。**

## 1. 本轮检验什么

本轮把两个问题分开：固定生物学基因集是否带来优于同容量随机基因集的疾病状态表示；保留静态 TITAN 与分子加性主效应后，候选分子—形态关系是否再提供可泛化交互。

**这是固定有界证据交互的可行性审计，不是 RNA-conditioned TITAN token prompt，也没有训练 TITAN/text encoder。没有验证完整可学习 molecular prompt 架构。**

RNA 状态为固定 8 组 Hallmark 的训练集基因 z-score 均值；CNA 为相同基因集的 signed log2 dosage 均值，两者保留为 16 个独立维度。CNA 值不代表 pathway activation，也没有声称已计算 chromosome-arm burden。

RNA Top512 按训练集 log2 表达插补后的方差选择，再标准化。随机组通过训练集表达均值/SD 的 5×5 分箱内全局基因置换生成，保留基因集大小、重叠结构与测量覆盖；不使用标签。

## 2. 数据与拟合审计

- RNA 原始维度 20531；CNA 原始维度 25131；各 512 位患者均有 primary tumor 测量。
- RNA 非有限比例 5.8398%，全缺失 gene rows 0；CNA 非有限比例 0.0119%，全缺失 gene rows 0。
- Batch-normalized RNA 含 167186 个小负值；有限值均 > −1，因此原样保留并 log2(x+1)，没有截断到零。
- RNA/CNA 相同选中 aliquot 列表：512/512；重复基因符号在状态评分中先按 gene symbol 合并。
- 所有插补、Top512、基因 z-score、状态 scaling 和随机集合匹配只用 inner-train；锁定 C 后在完整 outer-train 用相同算法重拟合。C 仅比较 0.1/1/10；不存在 epoch 选择或 in-sample stacking。
- 全部分类器收敛，最大 531/5000 iterations；验证/测试没有参与 upstream 表示拟合。
- 独立重算 120 个预处理分区；OOF 指标最大误差 8.33e-17；B2/B4 counterfactual 误差 0。

## 3. Molecular Representation Audit

| Endpoint | R0 Raw RNA 512D | R1 真基因集 RNA 8D | R2 匹配随机 8D | R3 RNA+CNA 16D |
|---|---:|---:|---:|---:|
| T | 0.3120 ± 0.0081 | 0.3013 ± 0.0061 | 0.3046 ± 0.0125 | 0.2800 ± 0.0051 |
| N | 0.4264 ± 0.0013 | 0.3836 ± 0.0060 | 0.4082 ± 0.0020 | 0.4027 ± 0.0145 |
| M | 0.5850 ± 0.0008 | 0.6143 ± 0.0084 | 0.5658 ± 0.0060 | 0.6132 ± 0.0100 |
| Stage | 0.3077 ± 0.0094 | 0.3423 ± 0.0033 | 0.2413 ± 0.0028 | 0.3593 ± 0.0050 |

以上为三个 seed 各自拼接五折 OOF 后的 BA 均值 ± seed SD；seed 只改变 inner split，三者来自同一队列。

RNA 表示的语义证据看 R1−R2；信息保留看 R1−R0；CNA 是否额外有效看 R3−R1，独立判断，不把 CNA 增益作为 RNA 生物语义成立的前提。内部 CV 不能证明跨队列迁移。

## 4. Concept Bridge Audit

| Method | 定义 |
|---|---|
| B0 | 静态 TITAN 32 concepts |
| B1raw | 静态 concepts + Raw RNA Top512 |
| B1mol | 静态 concepts + RNA/CNA 16 molecular states |
| B2 | 完整 B1mol 主效应 + 有界 RNA/CNA concept-specific 交互，λ=0.1 固定 |
| B3 | B2 相同容量，用行/列度数均相同的打乱 prior mapping |
| B4 | 原封不动复用 B2 权重，仅错配交互 conditioning；病理和分子加性主效应均匹配 |

交互为 λ·static cosine·tanh(molecular state × allowed mask)。RNA/CNA 交互分开，不把 dosage 与 expression 平均成“激活”。每条证据修正绝对值 ≤ 0.1×|静态 cosine|；这约束输入证据，不保证最终 head 的 logit 修正同样有界。交互使用静态分数同一 scaler，不单独标准化，避免取消 λ。

固定 prior 只含 15 个候选关联，20 个概念无交互边；unknown 不强行映射。它们有外部来源和置信度，但没有本队列形态真值验证。

| Endpoint | B0 | B1raw | B1mol | B2 | B3 | B4 |
|---|---:|---:|---:|---:|---:|---:|
| T | 0.4330 ± 0.0042 | 0.3175 ± 0.0101 | 0.4122 ± 0.0172 | 0.4090 ± 0.0185 | 0.3865 ± 0.0212 | 0.3897 ± 0.0120 |
| N | 0.4440 ± 0.0076 | 0.4333 ± 0.0019 | 0.4668 ± 0.0062 | 0.4773 ± 0.0053 | 0.4740 ± 0.0083 | 0.4416 ± 0.0144 |
| M | 0.6703 ± 0.0056 | 0.6157 ± 0.0024 | 0.6458 ± 0.0147 | 0.6560 ± 0.0080 | 0.6527 ± 0.0035 | 0.6484 ± 0.0074 |
| Stage | 0.4047 ± 0.0032 | 0.3285 ± 0.0028 | 0.3839 ± 0.0051 | 0.3705 ± 0.0065 | 0.3738 ± 0.0117 | 0.3566 ± 0.0106 |

## 5. 同 seed 配对差值与不确定性

Class-stratified patient bootstrap 2000 次，对三个 seed 使用同一批 resampled patients，再平均 seed 差值。98.75% 区间是固定已训练模型条件下的描述性区间；未重训练整条 pipeline，不是 familywise 显著性保证。

| Endpoint | Comparison | 平均 ΔBA | 98.75% CI | 三个 seed 点差均正 |
|---|---|---:|---:|---|
| T | R1-R0 | -0.0108 | [-0.0939, +0.0784] | False |
| T | R1-R2 | -0.0033 | [-0.0972, +0.0905] | False |
| T | R3-R1 | -0.0213 | [-0.0915, +0.0400] | False |
| T | B2-B0 | -0.0240 | [-0.0992, +0.0559] | False |
| T | B2-B1raw | +0.0915 | [-0.0139, +0.1905] | True |
| T | B2-B1mol | -0.0032 | [-0.0653, +0.0629] | False |
| T | B2-B3 | +0.0225 | [-0.0200, +0.0729] | True |
| T | B2-B4 | +0.0193 | [-0.0572, +0.0942] | True |
| N | R1-R0 | -0.0428 | [-0.1138, +0.0260] | False |
| N | R1-R2 | -0.0245 | [-0.1032, +0.0508] | False |
| N | R3-R1 | +0.0191 | [-0.0395, +0.0838] | False |
| N | B2-B0 | +0.0334 | [-0.0187, +0.0900] | True |
| N | B2-B1raw | +0.0440 | [-0.0318, +0.1138] | True |
| N | B2-B1mol | +0.0106 | [-0.0210, +0.0429] | True |
| N | B2-B3 | +0.0033 | [-0.0205, +0.0269] | False |
| N | B2-B4 | +0.0357 | [-0.0214, +0.0941] | True |
| M | R1-R0 | +0.0293 | [-0.0632, +0.1168] | True |
| M | R1-R2 | +0.0485 | [-0.0567, +0.1492] | True |
| M | R3-R1 | -0.0011 | [-0.0606, +0.0593] | False |
| M | B2-B0 | -0.0142 | [-0.0856, +0.0518] | False |
| M | B2-B1raw | +0.0403 | [-0.0522, +0.1282] | True |
| M | B2-B1mol | +0.0102 | [-0.0239, +0.0476] | False |
| M | B2-B3 | +0.0033 | [-0.0303, +0.0361] | False |
| M | B2-B4 | +0.0076 | [-0.0651, +0.0830] | True |
| Stage | R1-R0 | +0.0346 | [-0.0289, +0.0992] | True |
| Stage | R1-R2 | +0.1009 | [+0.0224, +0.1712] | True |
| Stage | R3-R1 | +0.0170 | [-0.0223, +0.0603] | True |
| Stage | B2-B0 | -0.0341 | [-0.0838, +0.0148] | False |
| Stage | B2-B1raw | +0.0421 | [-0.0177, +0.1062] | True |
| Stage | B2-B1mol | -0.0134 | [-0.0499, +0.0211] | False |
| Stage | B2-B3 | -0.0032 | [-0.0337, +0.0270] | False |
| Stage | B2-B4 | +0.0139 | [-0.0331, +0.0621] | True |

## 6. 机制判断

- 真实 RNA 状态同时通过 Raw RNA 与匹配随机组比较的 endpoint：**无**。
- CNA 额外增益通过保守区间判断的 endpoint：**无**。
- Bridge 五项必要对照同时通过的 endpoint：**无**。

**区间跨零表示证据不足，不等于证明没有疾病信号。B2 高于 B0 不能排除 additive 信息或额外参数容量；B2 高于 B4 也必须结合 B1mol/B3，才能进一步讨论 pairing 与 mapping。任何这些辅助任务结果均不能外推 DSS C-index。**

若五项对照未同时支持，当前结果不足以扩大 molecular prompt 模型，也不能因此否定所有医学先验或完整可学习桥。若 true RNA 不优于匹配随机组，不能把低维压缩收益归因于 Hallmark 语义。

## 7. 优势与劣势（风险）

| 优势 | 劣势 / 风险 |
|---|---|
| 明确保留静态预训练病理证据和分子主效应 | 小型固定桥只覆盖部分候选联系，不代表完整可学习 prompt |
| 真/随机基因集、degree-matched mapping、conditioning-only mismatch 相互区分 | 仅一个随机 gene permutation 和一个随机 mapping realization，不能概括所有随机先验 |
| 同算法 inner/final fit，所有分类器收敛，120 份真正 OOF | 单队列、一次 inner split、三个相关 seed；训练方差未被 bootstrap 完整覆盖 |
| 关系来源和 uncertainty 可审计 | bulk EMT 混有 CAF、gene-dose 均值可抵消扩增/缺失；病例级 morphology 未标注 |

## 8. 置信度及后续边界

- **高：**本轮计算、患者对齐、OOF 指标、源文件哈希与错配控制的实现完整性。
- **中：**当前队列、指定评分方法与 prior 下的性能比较。
- **低：**外部泛化、分子—形态因果联系、DSS 改善。

Clinical Text 为可选补充，本轮暂缓：尚未建立不包含 T/N/M/Stage、且通过字段审计的文本输入。

## 拓展思考

将“压缩有效”“真实基因集语义有效”“跨模态配对有效”作为不同结论；只给得到必要对照支持的层级命名。

## 你可能想知道

这轮只有线性分类器需要训练，在服务器并行用 CPU 更合适；没有为提高显存占用而训练额外网络。TITAN 权重保留在服务器，所有患者预测和大文件不进入 GitHub。

## 文献依据

- [MSigDB Hallmark collections](https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp)：状态定义来源；评分方法为本轮明确规定的 gene-z 均值，并非 GSEA/GSVA。
- [CRC consensus molecular subtypes](https://www.nature.com/articles/nm.3967)：免疫、mesenchymal/TGF-β、stromal context 的动机，不能直接作为局部形态标签。
- [Microdissected budding RNA profiles](https://www.nature.com/articles/bjc2016382)：budding/EMT 候选联系。
- [EMT and stromal expression confounding](https://www.nature.com/articles/s41467-021-22800-1)：bulk EMT 的解释限制。
- [cBioPortal file formats](https://github.com/cBioPortal/cbioportal/blob/master/docs/File-Formats.md)：log2 CNA 编码解释。

- [cBioPortal data engineer response](https://groups.google.com/g/cbioportal/c/prm-A08br9Y)：PanCancer batch-normalized RSEM 中的负值也存在于其发布来源。

协议和 prior 修正在查看有效 OOF 之前锁定。旧 v3 因 Top512 和错配控制错误作废，其不完整结果没有用于本轮调参或机制判断。

## 本轮主要发现

1. **Stage 的基因集语义有局部提示：**R1−R2=+0.1009 BA，98.75% 条件区间 [0.0224, 0.1712]；R1−R0=+0.0346，但区间跨零。该证据只针对一套固定 gene-set 评分和一个匹配随机 realization，不能称为跨队列可迁移性已成立。
2. **CNA 增益尚未得到支持：**R3−R1 在 T 为负，在 N/Stage 为正，在 M 接近零；四个终点的条件区间均跨零。不能据此断言 CNA 无信息，尤其当前是 gene-set signed dosage 均值，未测试 arm burden。
3. **概念桥机制没有通过必要对照：**N 的 B2 点估计最好（0.4773），但 B3 也为 0.4740，B2−B3=+0.0033 且区间跨零；真实 mapping 没有稳定超过匹配打乱 mapping。T/M/Stage 的静态 B0 点估计仍高于 B2。
4. **本轮不能归因于线性优化预算不足：**2160 次 classifier fits 全部收敛，最大 531/5000 iterations。这个结论仅针对本轮凸线性拟合，不评价尚未测试的可学习 prompt。
5. **下一步边界：**保留 Stage 的局部分子状态线索，当前不扩大 molecular prompt/桥模型；优先明确分子测量覆盖和外部形态证据。BA 改善不能自动转化为 DSS C-index 改善。

固定 C 网格及每种方法的边界选择次数见 `budget_audit.json`。若大量选择 C=0.1，则更强正则化没有在本轮被覆盖；这是搜索范围的限制，不能在看过 outer OOF 后回调网格并将结果算作同一轮确认性实验。

详细有效基因数与每个终点的配对解释见 `effective_gene_set_coverage.csv` 和 `MECHANISM_SUMMARY_ZH.md`。
