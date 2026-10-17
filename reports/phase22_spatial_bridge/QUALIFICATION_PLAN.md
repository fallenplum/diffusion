# Phase22：分阶段资格审查与执行边界

日期：2026-10-18（Asia/Shanghai）。依据：用户提供的技术路线和锁定草案，以及本轮明确“继续实验，数据下载放 E 盘”的请求。

## 授权与阶段闭锁

草案中的 `DESIGN_ONLY_NOT_EXECUTED`、`execute_automatically: false` 描述生成草案当时的状态；当前用户已授权执行。该授权允许准备数据、实现、数值验证以及合格后的阶段 A/B 实验，**没有取消草案的数据资格要求或 A→B→C 顺序**。不能因获准执行而把 `actual_ST_donor_manifest_verified` 等未核验项设为 true。

本文件不是训练完成证据。截至本资格审查时，本子任务只实现并验证计数目标函数，没有启动任何正式外部模型或 TCGA 生存训练。主执行器的后续数据核验及实际运行以带哈希的运行凭据为准。

## A：9 次外部空间拟合何时可以启动

最多 A0/A1/A2 三种模型，每种种子 31/47/83。9 次拟合是合格后的上限，不是必须立即凑齐的运行数量。以下各项必须由实际数据产物支持。

| 资格项 | 必需证据 | 不合格时的处理 |
|---|---|---|
| 真实局部配对 | 原始非负 raw counts、对应 H&E/patch、真实坐标与映射；文件 SHA | 不用 normalized/z-score 冒充计数，不启动 A |
| 独立分组 | study、private donor、section、原始 accession 与镜像去重关系 | 不随机拆 spots；未知供体先补来源 |
| ST 开发/验证/测试锁定 | 先按研究/供体锁分区；同供体多切片同组；测试不选基因/字典/超参 | 供体较少时按预定留研究/供体评估并降低证据等级，不能称独立外部确认 |
| 图像尺度 | encoder、normalization、mpp、112µm 输入视野、表达捕获足迹分别核验 | 不凭 1536 维相同复用缓存；不把 Visium 捕获圆解释成整个图像方框 |
| 许可与预训练重叠 | 数据/代码/权重分别核实；已知、未知重叠分别记录 | unknown 不等于无重叠；不能将预训练见过的材料称该基础模型独立测试 |
| 桥接基因与字典 | ST-train 及目标已测交集，不看 DSS；最多 1024 基因，K=32；供体等权字典 | 不用目标结局选基因或 K |
| 实现资格 | count/mask/spot-axis/组织隔离、空间等变与跨 WSI 无边、字典 oracle | 不以单个能下降的训练 loss 代替资格 |

**本次文件审查不能把上述数据资格判为 PASS。** 目前附件只提供候选数据来源及设计，没有实测供体清单、分区哈希或可复用特征尺度凭据。下载元数据并完成资格后才可以登记正式 A0/A1/A2；单元测试、profile 和 smoke 均不得计为 9 次正式拟合。下载的大型资产位于 E 盘对应数据目录；公开仓库只保存来源、协议、匿名汇总和代码。

训练与选择：AdamW，lr=3e-4、wd=1e-2、最多 100 epochs、至少 10、patience=15；只有 ST-validation 能选择 epoch。A0 组成小模型；A1 相同 UNI2 上的局部 Transformer 加组成目标；A2 同 A1 加相对目标。冻结视觉特征；三臂使用相同基因、字典、训练/验证分区与采样规则。不以 DSS 或 ST-test 调整任何上游模块。

主指标先按 heldout 组织计算同基因跨 spot Spearman、去组织基因均值后的误差、局部高表达区定位，再按供体聚合；bootstrap 单位是 donor。固定字典 oracle 是表达重建参照，不是空间排序指标的数学上界，也不向图像预测模型提供真实测试表达系数。若 A2 不稳定优于 A1 就不保留相对目标；A1 不优于 A0 就不强留 Transformer。供体数/普通区间/跨研究分层必须同时披露。

## B：先证明局部结构，不能仅证明均值匹配

固定阶段 A 选定的外部先验与字典。只在 ST-validation 锁定 lambda/rho；ST-test 只评估。比较 no-bulk、gene-wise 全局缩放、rho=1、含未观测组织项的弱约束和同 study/assay 内错 bulk。pseudo-bulk 从 raw counts 聚合后变换；conditioning 和评估优先采用独立 count thinning，避免复制同一测量噪声。

同视野、图像子区域/更大 bulk 视野，以及有实测连续切片时的跨切片分别计量。几何/图像可估测量权重与真实 RNA 质量 oracle 权重分开。25/50/75/100% 仅是合成压力设置，不是患者真实取材重叠估计。

推进 C 必须看到部署可得权重下的局部排序/误差改善，且不是只有 bulk 均值更准；正确 bulk 不应与错 bulk 同样有效。仅有 oracle 或同计数噪声复制造成的收益不通过。混合方向结果只登记为候选，不能作等效检验或生物真值恢复。

## C：TCGA 生存入口当前仍闭锁

只有 A/B 局部机制门和目标 assay/图像兼容门均有证据才解锁 C1–C4。四臂 × 5 folds × 3 seeds × inner/fresh outer = 120 次正式新拟合；wrong-bulk 和边际保持分配置换的额外 60 次受结果门限制，不自动追加。共同主效应 J0、原 RNA/CNA、Clinical 和视觉参照保留；PFI 仅作 DSS 冻结风险转移。

| 现有目标证据索引 | 已支持的事实 | 仍不能推出 |
|---|---|---|
| `reports/phase7_p0r_robustness/RESULTS.md`，“203×3024 缺失区块” | 官方表与本项目原值/NA 相同；203 人同缺 3024 基因；GDC 元数据中 203 仅 GAII、295 仅 HiSeq、14 双平台记录 | 缺失不是表达零；14 双平台的最终 aliquot 未锁；该处理表不是 ST raw UMI 或已证明的同量纲组成 |
| `reports/phase11_semantic_anchored_trimodal/raw_source_provenance.json` | RNA source SHA、log/float rounding 历史、原矩阵与 cache 精度审计；CNA 为独立 dosage 输入 | log-cache 还原不能自动恢复完整测量历史或跨技术转换 T；CNA 不能作 RNA composition |
| `reports/phase11_semantic_anchored_trimodal/RNA_PREPROCESS_AND_TOKEN_AUDIT.md` | Top512、标准化、缺失及 fold-local 路径历史 | 早期“平台未知”不能覆盖后来 P0R 的元数据证据，也不能反推同块取材 |
| `reports/phase16_semantic_routing/DATA_INTERFACE_AUDIT.md`、`COORDINATE_AUDIT.json` | 有坐标与计算匹配历史；519 张候选切片；当时原图视觉 QA 为 0 | 坐标可计算映射不等于原图足迹核验或 112µm 输入一致 |
| `reports/phase20_selective_capacity/DATA_AUDIT.json` | 当前历史融合表示记为 UNI2-h 的面积加权物理 CONCH 足迹 | 面积平均不同足迹的 embedding 不是重新编码一个 112µm RGB patch 的严格等价物 |

因此优先保持三项分离：平台与缺失来源已部分核实；患者 RNA aliquot–WSI 的同块/邻块/未知关系尚需记录；目标测量转换 T 与严格图像尺度还未获资格。若暂时只能完成外部 ST/pseudo-bulk，则报告该子任务的结果并保留 C 为未解锁，不能拿量纲不明的归一化强行启动生存矩阵。

## STRank 的已核验源码与本项目适配

本轮实时读取固定作者源码：[STRank loss_functions.py, commit e1917b4](https://github.com/naivete5656/STRank/blob/e1917b4b311a72d9150c03bc4c546d1a6550888e/strank/loss_functions.py)。`STRankLossList` 按 `sample_id` 分组，`LogSoftmax(dim=-2)` 沿 spot 轴；元素损失使用原始 count 作为权重，`calc_raw_lpi_norm` 加每 spot mean-count 的 log-offset。对应[作者论文方法](https://arxiv.org/html/2512.06612v1)用于同组织内相对表达关系，不能宣称已解决任意 assay 偏移。

本项目 `strank_objective.py` 没有复制整套官方训练器，适配差异明确为：

1. 预测是分子字典生成的正组成 `mu`；relative logits 是 `log(mu)+log(exposure)`。默认 exposure 为共同已测基因 raw-count 总数，也可显式提供有测量依据的 library exposure。固定 G 时总数与官方平均数仅差可抵消的 log(G)；本实现没有给零 exposure 人为加 epsilon，正 counts 要求正 exposure。
2. relative 对每个组织、每个基因在有效 spots 之间归一，不跨组织、不沿 gene 轴。所有 measured-zero 均保留为候选；unmeasured 与 padding 从分母移除，不插零伪装测得。
3. composition 对每 spot 已测基因条件化：`-sum(y log(mu / sum_observed mu))`。两项目标各自按组织 raw-count 总质量归一，然后同 donor 内组织均值，最后 donor 等权。不同于官方拼接后逐元素 batch mean；两者绝不标成同尺度官方严格复现。
4. 单个零 count 项按数学极限不贡献 numerator，但测量零候选仍影响 denominator。零 exposure/零 count spots 不伪造曝光。全零 count 组织不人为贡献零来稀释 donor loss；全零 batch 直接拒绝。singleton 组织的相对目标为零，应由数据资格保证训练区域有实际位置对比信息。
5. 只接受非负整数计数；负数、连续标准化值、有效位置 NaN、组织跨 donor ID 冲突都会拒绝。模块没有 DSS、PFI 或临床标签入口。

数值测试包括独立二 spot 算式的轴检验、固定 G 下官方列表公式的尺度对照、跨组织隔离、供体等权与切片重复、计数整体尺度、exposure 效应、测量零/未测区别、NaN padding 的输出与梯度、正确/错误空间配对负对照、联合重排与有限非零梯度。**本机运行 16 项通过；这只是目标函数资格，不是 A/B/C 科学结果。**

## 优势、局限与置信度

优势：以真实局部计数提供独立于 DSS 的监督；计数曝光、组织分组、缺失与供体权重均有可测合同；A/B/C 分离使局部重建收益和生存增量不混淆。

局限：ST 供体与研究数尚需实际查明；预训练重叠和 patch 尺度可能限制外部确认；PanCancer RNA、ST UMI 和取材关系的差异未因形式上的组成归一消失。数学/单元测试正确不能保证局部生物状态可识别，更不能保证 65 DSS events 下存在增量。

置信度：目标函数的轴/分组/零值数值行为为高（已测试）；真实 ST 可学性与目标量纲兼容目前未定；对 DSS 提升无现有新证据，不给成功百分比。
