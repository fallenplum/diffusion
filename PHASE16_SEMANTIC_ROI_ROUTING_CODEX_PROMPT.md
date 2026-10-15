# Phase 16｜患者条件化病理语义路由：多组实验与严格验收（交给 Codex）

你是负责实现、复现和审计的资深计算病理研究员。直接在 `fallenplum/diffusion` 最新 Phase15 分支 `codex/phase15-rna-conditional-complementarity`（起点提交 `04150dc976057f825702d5fe3502696060790dd7`）新建独立 Phase16 分支，**先审计资产，再运行实验，最后提交全部源码、配置、统计与修复记录**。不得覆盖 Phase10–15 结果。无需逐步询问我；缺少模型、WSI 或授权时，明确记录阻塞项并继续可以合法执行的子实验，禁止伪造完成状态。

## 目标与边界

本阶段停止继续开发 RNA 错误对残差修正；研究 **患者临床/分子表型指导的 WSI Patch 语义检索是否比随机、均匀、形态多样性、传统 MIL 和固定语义查询更有效，并能给强 Clinical+TITAN-S32 带来新增 DSS 排序信息**。DSS 为主终点（512 人、65 事件），PFI（510 人、129 有效事件）仅作次要敏感性分析。固定历史 5 outer folds × seeds `31/47/83`、患者身份与原始终点，不创造新患者、事件或终点；本队列跨 Phase 多次探索，所有正结果必须标为内部探索性，不能当成外部确认。

## G0：必须先完成的数据、模型接口及空间硬审计（不训练生存模型）

1. 自动定位（不要猜路径）原始 WSI/SVS、切片—患者关联、全部 CONCH v1.5 512px@20× Patch 特征与 level-0 坐标、UNI2-h 256px@20× Patch 特征与坐标、TITAN-S32、TITAN-native WSI 特征、官方 CONCH/TITAN 文本编码器及权重。输出患者/切片/patch 覆盖率、dim、尺度、倍率、坐标系、缺失清单、SHA256、授权与存储/GPU 预算。历史 CONCH 128-patch 缓存只覆盖原始 patch 的一小部分，**不得将该缓存称为全 WSI 候选库**；若无法恢复全量，所有实验必须标注 `candidate-pool-limited`，所有 selector 在同一候选池内比较。
2. **证实**图文空间兼容：查官方 encode_text/encode_image、投影层、归一化、版本及输出维度；只有同一经过验证的图文对齐空间才计算 Patch↔Prompt 余弦。严禁直接把 TITAN text 与未经验证投影的 CONCH patch 做 cosine（同维度≠同空间）；若 TITAN 只验证为 slide 级对齐，保留 TITAN slide baseline，Patch 路由改为经验证的 CONCH 自身图文编码器；都不成立则标 `SEMANTIC_ROUTE_BLOCKED`，不伪造路由实验。
3. **坐标对齐**：CONCH 512px 与 UNI2-h 256px 不是逐行对齐。使用 WSI level-0 坐标、MPP/倍率及 ROI 面积重叠，建立“一块 CONCH ROI ↔ 多个 UNI2 子块”的空间映射；随机人工/可视化检查不少于 30 组 ROI，导出面积重叠与未匹配率。无原图或可靠坐标时停止双视觉分支，绝不以相同行号伪对应。
4. 分子来源：RNA、CNA 按同一 gene ID 映射，明确扩增/缺失与表达量的区别；不把 RNA 高表达写作“突变”。缺失的 ~203 人/3024 基因块单独统计；缺失处理、通路富集与文本分级阈值全部由训练折估计；不可宣称已证实测序平台身份。
5. G0 提交 `DATA_INTERFACE_AUDIT.md`、`SPACE_COMPATIBILITY.json`、`COORD_ALIGNMENT.csv`、`G0_DECISION.json`；若部分被阻塞，继续其他通过的子分支，同时如实缩小结论。

## G1：统一候选池的检索质量和可靠性（低成本，可与 G0 同步）

在不查看外折 DSS/PFI 标签的情况下，固定 prompt 模板（列出实际文字）和检索逻辑；比较以下 8 个 ROI selector，均考虑多 WSI 患者、组织掩码和空间覆盖：

- `RAND`：患者内部可重复随机采样；`GRID`：分层空间均匀采样；`DIVERSE`：无标签 k-center/聚类多样性；`MIL`：训练折监督的简单 attention-MIL ROI 评分（selector 及其超参均训练折内拟合）。
- `FIXED_TEXT`：固定结直肠癌组织形态学概念查询；`CLIN_TEXT`：年龄/性别/TNM/AJCC 结构化文本；`RNA_TEXT`：训练折标准化后的 Hallmark 通路状态文本；`CLIN_RNA_TEXT`：临床+RNA 联合文本。

先比较每个 selector 在预算 `M={128,512}` 下的 ROI 覆盖、slide 多样性、空间冗余、文本敏感性及检索重复率；额外用 `M=64/256/1024` 作无监督预算曲线（若计算允许，不强制训练全部生存模型）。同一预算、同一原始候选池、同一候选切片和统一 ROI 合并规则。使用未经事件标签选出的文本正负语义 sanity-check；没有病理 ROI 金标准时不得伪称具有“检索准确率”。

## G2：核心生存实验（预注册，建议完整运行）

**主要 16 arms = 8 selectors × 2 Patch 预算 (128/512)**。每 arm 使用冻结的同版本 CONCH Patch 特征和**完全匹配**的轻量 ROI→patient 聚合器及预测头；监督 MIL 的额外选区训练要单独计入参数/预算，并比较同容量端到端 MIL。评价最终 `Clinical + TITAN-S32 + ROI`，**S32 必须在模型 forward 内真实使用**；保留强 Clinical+S32 基线。

另外固定 6 个 anchor/control：`Clinical-only`、历史 `Clinical+S32`、Phase15 新嵌套 `Clinical+S32`、`Clinical+S32+CONCH-global`、`Clinical+S32+TITAN-native`（可用时）、历史 Phase12 `C4(128 patch)`；仅在输入/折/目标/评价完全一致时允许原样复用，否则重新拟合并标清不可直接比较。各模态预训练 backbone 冻结，禁止因额外数据下载改变患者队列。核心 G2 **最多 22 arms × 3 seeds × 5 folds = 330 个 formal fold fits**（完全等价历史锚点只作引用，不谎报新拟合）。

## G3：机制负对照（10 arms，必须优先于盲目扩架构）

针对 `M={128,512}`，分别运行：

1. `CLIN_TEXT` 的同分期/同癌种**错患者临床 Prompt**（2 arms）；
2. `RNA_TEXT` 的同癌种、同缺失模式**错患者 RNA Prompt**（2 arms）；
3. `CLIN_RNA_TEXT` 仅错 RNA、临床正确（2 arms）；
4. `CLIN_RNA_TEXT` 仅错临床、RNA 正确（2 arms）；
5. `CLIN_RNA_TEXT` 打乱文本字段语义但保持长度/格式相近（2 arms）。

错配只在各自 inner-train/inner-val/outer-train/outer-test 分区内执行，不能跨 fold 或读取测试标签；匹配分层无法执行时须显式退化/报错并计数。所有负对照与正确 Prompt **匹配 ROI 预算、编码器、参数量和训练日程**。另外做 `no-stage-text` 与 `no-RNA-text` 两类无需新参数的 Prompt 消融（如需新拟合，独立计数）。G3 **最多 150 个 formal fits**。

## G4：双视觉 ROI 对齐与表征实验（G0 坐标通过才执行）

对 `GRID`、`FIXED_TEXT`、`CLIN_RNA_TEXT` 各预算 128/512 共 **6 arms**，在选定 CONCH ROI 的实际空间位置聚合对应 UNI2-h 子块并接入**同一患者聚合头**：比较 G2 已有 CONCH-only、`UNI2-only`、`CONCH+UNI2`（双视觉默认 6 arms，UNI2-only 如算力允许再加 6 arms）。必须报告匹配 ROI 比例、UNI2 子块数分布、参数量、推理成本；禁止用增加 Patch 总覆盖面积掩盖 backbone 增益。坐标不成立则这一节记 BLOCKED，不补造随机映射。

## G5：组学与 CNA 增量实验（G0/G2 合格时执行）

使用 `FIXED_TEXT`、`CLIN_RNA_TEXT` × 128/512 Patch × 3 个组学输入，共 **12 arms**：`RNA-Hallmark`、`RNA+CNA ungated`、`RNA+CNA same-gene gated`。通路映射用训练折内可用的 MSigDB Hallmark/Reactome，设置 pathway mask 与同参数量随机掩码、`RNA-only / CNA-only` 低容量对照；若需额外正式训练，报告新增 arm 及预算。不可先在完整 512 人做表达筛选；不将缺失当零表达；真实通路 vs 随机通路要匹配基因数及模型容量。重点比较**患者条件化查询贡献**与**组学仅在最终预测头参与的贡献**，避免将 RNA 输入增加与检索机制混为一谈。

**总工作量上限**：G2 22 + G3 10 + G4 6 + G5 12 = **50 arms × 3 seeds × 5 folds ≈ 750 次 formal fold evaluations/fits**，含可复用基线；不是必须无条件全部执行。先进行每模块 1 seed×2 folds 的资源/正确性 pilot，**pilot 只能验证速度、OOM、数值及梯度，不能用测试分数筛选方案**。通过门槛的模块再执行完整 3×5；GPU 显存不够使用小 batch/流式 patch 与梯度累积，不私自把 512 Patch 改成 128。

## 统计、审计与失败处理

- 主终点 DSS，主指标 **Harrell C-index 与 Uno60 C-index 必须共同改善**；报告 5-fold 同折可比较对分子/分母后按 seed 聚合，禁止直接平均 fold C-index、跨 fold 比较风险、合并 seeds 为新患者。PFI 次要；报告事件/删失/风险集、train-only IPCW、按患者共享权重 paired bootstrap（≥20,000 次）及预先冻结的多重校正。
- G2 **8 个主要预定义 contrast**：FIXED_TEXT 对 GRID（128/512）、CLIN_TEXT 对 FIXED_TEXT（128/512）、CLIN_RNA_TEXT 对 CLIN_TEXT（128/512）、CLIN_RNA_TEXT 对最强预定义 Clinical+S32 锚点（128/512）；两指标共 16 个检验，采用 Bonferroni 或 Holm（实验开始前写死）。历史 C4 和 TITAN-native 等另做次要表；不得看外层后把胜出方案追认为 primary。
- 对 MIL 选区、ROI 聚合、患者多张 WSI 融合、RNA/Clinical Prompt、CNA gate 开展模块消融；记录每模块实际梯度、冻结状态、mask 正确性、wrong-prompt 性能和同容量控制。特别检查**临床 Stage 文本化导致的捷径**：保持预测头 Clinical 输入不变，仅改变 Prompt 查询内容，分开报告 Stage-only 与分子条件增量。
- 独立评价程序不引用训练指标核心函数；交叉核验 `scikit-survival`，独立验证 Efron Cox / discrete NLL（如有）、Harrell/Uno、同时间 ties、patient IDs、外折分割、preprocessing fit IDs、训练验证选择、坐标单位、ROI 排序、缓存 SHA、OOF 与 checkpoint 逐单元回放。
- **故障注入必须真的失败**：跨患者 ROI、错误 level/MPP、错 CONCH↔UNI2 ROI 匹配、wrong RNA/Prompt、外折标签影响训练、删失时间被篡改、风险方向反转、复用不匹配 checkpoint。发现关键 Bug：保留旧记录、标 `INVALID`、修复后重训受影响的所有 cells，保留 `BUG_LOG.csv` 与源哈希；不接受仅靠自报 `PASS`。
- 输出按 arm/seed/fold 的主表、paired Δ/CI/校正 CI、参数量/FLOPs/显存/耗时、ROI 可视化示例（不泄露患者信息）、坐标与语义匹配误差、训练诊断、失败及跳过清单、统计不确定性、私有/公开 artifact 清单、复现一键命令。

## 必须交付

新分支推送 `configs/phase16_*.json`、`src/phase16_*/`、`scripts/run_phase16.py`、`scripts/validate_phase16_independent.py`、`reports/phase16_semantic_routing/{RESULTS.md,MODEL_SUMMARY.csv,FOLD_METRICS.csv,SEED_OOF_METRICS.csv,PAIRED_COMPARISONS.csv,DATA_INTERFACE_AUDIT.md,SPACE_COMPATIBILITY.json,COORD_ALIGNMENT.csv,BUG_LOG.csv,DECISION.md,REPRODUCE.md,ARTIFACT_MANIFEST.json}`。报告中写明**哪些 arm 真正完成、哪些仅复用、哪些被 G0 阻断**；提供 commit SHA、实际命令与独立验收记录。

**最终研究决策必须回答**：(1) 官方验证的文本/视觉空间能否做 Patch 路由？(2) 在预算匹配下患者条件化路由是否胜过 FIXED/GRID/MIL？(3) 真实 RNA Prompt 是否胜过错患者 RNA Prompt？(4) UNI2 在坐标正确匹配时是否有净增量？(5) 这些改善是否同时超过强 Clinical+S32 和 Phase12 C4，且经校正/跨 seed 稳定？如无，明确 `NO_CONFIRMED_INCREMENT` 或 `BLOCKED`，不包装为 SOTA、病理机制已证实或可直接投顶会。
