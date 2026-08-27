# Phase P1：UNI2-h whole-slide context + DSS validation

## 结论先行

在固定的患者级五折外层交叉验证下，**本实验没有支持“带坐标的 spatial masked-context SSL 能稳定优于 mean/MIL/context-scratch”这一核心假设**。主 proposed pipeline P1/P2/P3 的 Harrell C 均低于 B0；P2/P3 的 Uno C 点估计高于 B0，但 paired bootstrap 区间均跨 0。坐标去除后的冻结 probe（P0_nocoords）点估计较高，但它是坐标 ablation，不应被改写成原始 spatial 机制的确认。

## 数据、审计与隔离

- 仅使用 `TCGA/` 下的 UNI2-h bags、原始 H5 坐标和 DSS 标签；根目录 `data/` 未作为输入。
- 最终数据：487 位患者、494 张选定 slide、约 3,077,884 个 patch、1536 维特征、65 个 DSS event。
- H5 中的真实 `coords` 与 features 一一对齐；未用数组顺序伪造坐标，空 bag/NaN/Inf 检查通过。
- 五折逻辑为 `StratifiedKFold(n_splits=5, shuffle=True, random_state=20260723)`；患者与 slide 均无跨折重叠。
- OOF 完整性：20 个方法×seed 组合全部通过 487 行、487 个唯一患者、标签/折号回填一致性检查。
- SSL 隔离：30 个 fold×配置组合的 outer-test overlap 全为 0；SSL 只读取 outer-train 的 train/val 患者。

详细证据见 `audit/data_audit.md`、`audit/coords_audit.csv`、`audit/oof_integrity.csv`、`audit/ssl_leakage_check.csv`。

## 主 seed（20260723）OOF 结果

| 方法 | Harrell C | Uno C | ΔHarrell vs B0 | ΔUno vs B0 | 五折中优于 B0 | paired bootstrap ΔHarrell 95% CI | paired bootstrap ΔUno 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.6600 | 0.6392 | 0.0000 | 0.0000 | 0/5 | [0.0000,0.0000] | [0.0000,0.0000] |
| B1 | 0.6633 | 0.6558 | 0.0033 | 0.0166 | 3/5 | [-0.0540,0.0663] | [-0.0510,0.0884] |
| B2 | 0.6225 | 0.6941 | -0.0374 | 0.0549 | 3/5 | [-0.1112,0.0348] | [-0.0318,0.1467] |
| P0 | 0.6050 | 0.6284 | -0.0550 | -0.0108 | 2/5 | [-0.1202,0.0190] | [-0.0871,0.0623] |
| P1 | 0.5995 | 0.6504 | -0.0605 | 0.0112 | 2/5 | [-0.1413,0.0251] | [-0.0716,0.0887] |
| P2 | 0.6118 | 0.6758 | -0.0482 | 0.0366 | 3/5 | [-0.1270,0.0360] | [-0.0442,0.1088] |
| P3 | 0.5985 | 0.6706 | -0.0615 | 0.0314 | 3/5 | [-0.1370,0.0162] | [-0.0465,0.1027] |

B0 从 TCGA bags 重新计算得到 Harrell C=0.6600，历史值 0.660046 的差为 -0.000049，因此比较基准可复核。

## 额外稳定性

| 方法 | seed | Harrell C | Uno C |
|---|---:|---:|---:|
| B1 | 20260723 | 0.6633 | 0.6558 |
| B1 | 20260724 | 0.5975 | 0.6577 |
| B1 | 20260725 | 0.6224 | 0.6503 |
| B2 | 20260723 | 0.6225 | 0.6941 |
| B2 | 20260724 | 0.5750 | 0.5956 |
| B2 | 20260725 | 0.6115 | 0.6468 |
| P0_nocoords | 20260723 | 0.6733 | 0.6761 |
| P0_nocoords | 20260724 | 0.6936 | 0.6673 |
| P0_nocoords | 20260725 | 0.6695 | 0.6705 |

P0_nocoords 三个 seed 的 Harrell C 为 0.6788±0.0129，Uno C 为 0.6713±0.0045；B1/B2 的 seed 间波动更大，不能只报告最优 seed。

## 对研究问题的回答

### Q1：mean pooling 是否限制表征？

B1 end-to-end 仅为 Harrell 0.6633、Uno 0.6558；B2 为 Harrell 0.6225、Uno 0.6941。因此 mean pooling 可能不是唯一瓶颈；引入可学习聚合并未产生稳定的双指标改善。

### Q2：Context Transformer 本身是否有效？

B2 相比 B1 的 Harrell 下降 -0.0408，Uno 上升 0.0384，方向冲突且五折异质，结论为**未得到稳定支持**。

### Q3：masked context pretraining 是否带来真实收益？

冻结 probe 的 Harrell：B1_probe=0.6932、B2_probe=0.6546、P1_probe=0.6515、P2_probe=0.6645；带坐标 SSL 的 P0/P1/P2 end-to-end 则分别为 0.6050/0.5995/0.6118。这说明 representation probe 与 end-to-end survival 结论不一致，SSL 表征的可预测结构不等于 DSS 相关结构。

### Q4：spatial block mask 是否优于 random mask？

P0 spatial-0.40 为 Harrell 0.6050、Uno 0.6284；P0 random-0.40 为 0.5888、Uno 0.5775。spatial 点估计更高，但仍低于 B0，不能据此宣称机制确认。mask 0.25 的 Harrell 0.6027 也未改善。

### Q5：coords 是否有用？

P0 with-coords=0.6050/0.6284，P0_nocoords=0.6733/0.6761（Harrell/Uno）。在本数据和实现下，去除 coords 反而更好；这是一项 ablation 结果，提示坐标归一化/采样或坐标信号本身需要独立诊断，不能把它解释为“空间信息已被证实”。

### Q6：Cox+ranking 是否更适合 DSS？

P1/P2/P3 的 Harrell 为 0.5995/0.6118/0.5985，Uno 为 0.6504/0.6758/0.6706。P2/P3 的 Uno 点估计高于 P1，但 paired intervals 跨 0；ranking 是可能的辅助目标，不是已确认的改进。

### Q7：重建质量与 DSS 性能是否一致？

训练曲线显示 masked cosine/loss 改善，但 P1/P2/P3 没有同步超过 B0。由此只能说 SSL 学到了可重建的 feature-space 结构，**不能推出这些结构与 DSS 风险相关**；这正是本实验中 motivation 链条未闭合的关键证据。

## Sanity check 与限制

- P2 标签置乱的五折 OOF 为 Harrell 0.5140、Uno 0.4427，paired delta 均为负，未见置乱后异常高性能。
- 所有主要指标均为 patient-level OOF；1000 次 paired bootstrap 的区间在 `statistical_tests/*bootstrap_1000.csv`。
- AUC@12/24/36 已按 outer-train censoring KM 计算；IBS 记为 NA，因为当前 risk-only checkpoint 未保存 fold-wise Breslow baseline survival，强行重建会引入不可审计的校准假设。
- Survival 训练采用每次一个 event 加最多 7 个 outer-train at-risk comparator 的随机风险集近似，而非完整 Cox 风险集；这是计算约束，结果应按近似目标解释。
- 没有 RNA、clinical 或 multimodal 输入；坐标图只表示 contextual importance scatter，不是病理因果热图。

## 文件索引

- `results_master.csv`：每个方法×seed 的 OOF、Uno、AUC、bootstrap CI 和资源列。
- `statistical_tests/fold_metrics_all.csv`、`statistical_tests/*bootstrap_1000.csv`：逐折与不确定性。
- `embeddings/`、`statistical_tests/embedding_diagnostics.csv`：患者表征与 collapse 检查。
- `figures/`：t-SNE 与 P2 contextual-importance 坐标散点。
- `experiment_manifest.json`：数据范围、配置、软件和完整性证据。
