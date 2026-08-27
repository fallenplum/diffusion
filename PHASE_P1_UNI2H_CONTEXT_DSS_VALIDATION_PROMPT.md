# Phase P1：UNI2-h Whole-Slide Context Pretraining + DSS Weak Supervision 实验验证 Prompt
> **目标**：验证“先利用 UNI2-h patch features 学习 WSI/患者级病理上下文，再用 DSS 生存监督进行风险校准”是否能比现有 mean pooling / 直接弱监督 MIL 学到更好的患者级病理表征。  
> **本阶段只研究 pathology branch，不做 RNA/clinical 多模态融合。**

---

## 0. 你的角色与硬性任务

你是一名负责 **计算病理 + 生存分析 + 表征学习** 的科研工程师。你必须在当前项目根目录中：

1. **先审计现有数据、代码、划分和既有结果**；
2. 在不破坏现有输出的前提下实现并实际运行本阶段全部核心实验；
3. 严格患者级、严格 outer-fold 隔离，禁止任何形式的数据泄漏；
4. 验证以下核心假设：

> **Hypothesis H1**  
> Frozen UNI2-h patch embeddings 已包含较强的 patch-level pathology semantics，但简单 mean pooling 忽略 patch 间空间/上下文关系。  
> 在仅有少量 DSS events 的情况下，先用不依赖 DSS 标签的 masked spatial-context pretraining 学习 WSI 内 patch-to-patch / local-to-global 关系，再用患者级 DSS 生存监督进行轻量 fine-tuning，应当得到比 mean pooling、ABMIL 和从零训练 context encoder 更有预后信息的 patient-level pathology representation。

5. 不能只写代码或只生成配置文件；**必须真正执行实验并生成完整结果**。
6. **所有核心实验没有执行完成前，不允许生成“已完成”式最终总结。**
7. 如果运行中断，必须可恢复；禁止因单次错误直接结束整个任务。

---

# 1. 已知项目背景与既有事实

先检查当前根目录和已有 `outputs/`，不要假设路径一定与下面完全一致；以实际文件为准。

当前已有审计/基线结果曾包含：

```text
outputs/audit/uni2_bag_manifest.csv
outputs/audit/uni2_patient_manifest.csv

outputs/baselines/uni2_mean_cox/
outputs/baselines/morphology_prototypes/
outputs/baselines/multimodal_comparison/
```

已知历史结果（必须先复核，不能直接当作新实验结果）：

- 可用 UNI2-h pathology 患者：约 **487**
- DSS events：约 **65**
- UNI2-h patch feature dim：**1536**
- 每位患者 patch 数量变化很大，单 slide 可从数百到约 3 万+
- 少数患者存在多张 diagnostic slides
- 历史 seed：`20260723`

既有 pathology baseline：

```text
UNI2-h patch mean
-> patient-level mean
-> fold-wise scaler/PCA
-> ridge Cox
OOF Harrell C-index ≈ 0.660046
```

历史 morphology baseline 还包括：

```text
UNI2 mean + PCA32        OOF C ≈ 0.668939
prototype only           OOF C ≈ 0.569883
UNI2 mean + prototype    OOF C ≈ 0.638407
```

已有 multimodal baseline 中 clinical-only 很强，但**本阶段禁止将 clinical 或 RNA 输入 pathology encoder**，避免无法判断 pathology representation 本身是否改善。

---

# 2. 必须先澄清的概念

本阶段必须遵循以下定义。

## 2.1 输入

每位患者 \(i\) 有一个或多个 slide。

每张 slide 包含：

\[
H_i = \{h_{i1}, \ldots, h_{iN_i}\}, \quad h_{ij}\in\mathbb{R}^{1536}
\]

其中每个 `h_ij` 是 **冻结的 UNI2-h patch embedding**。

如数据中存在坐标：

\[
c_{ij}=(x_{ij}, y_{ij})
\]

则必须用于 spatial/context 建模。

## 2.2 患者级监督

不是 C-index。

真正监督标签为：

\[
y_i=(T_i,\delta_i)
\]

其中：

- `T_i = dss_months`
- `δ_i = dss_status`
- `δ_i=1` 表示观察到 disease-specific death
- `δ_i=0` 表示 censored

## 2.3 C-index

C-index 是**评价指标，不是原始监督标签**。

训练可以使用：

- Cox partial likelihood
- pairwise survival ranking surrogate
- IPCW-weighted ranking

但不能把离散 Harrell C-index 直接当作普通可微 loss。

---

# 3. Stage 0：数据与泄漏审计 —— 必须最先完成

建立新目录：

```text
outputs/pathology_context_phase1/
```

禁止覆盖任何旧 `outputs/baselines/*`。

首先生成：

```text
outputs/pathology_context_phase1/audit/
```

至少输出：

```text
data_audit.json
data_audit.md
patient_manifest_used.csv
slide_manifest_used.csv
coords_audit.csv
split_audit.csv
```

## 3.1 检查原始 UNI2-h bags

优先寻找类似：

```text
data/uni2_h_bags/<PATIENT>/<SLIDE>.npy
```

以及原始 `.h5` 或任何带 `coords` 的文件。

必须检查：

- patient_id
- slide_id
- patch count
- feature dim
- dtype
- 是否存在 NaN/Inf
- 是否存在 coords
- coords shape 是否与 features 一一对应
- 多 slide patient
- 是否有空 bag
- DSS time/status 完整性
- 患者数、event 数

## 3.2 坐标是本阶段核心信息，不允许伪造

如果当前导出的 `.npy` 只有 `features` 而没有 `coords`：

1. 搜索原始 `.h5`；
2. 搜索数据导出脚本；
3. 检查 HuggingFace UNI2-h feature 原格式；
4. 如果本地原 `.h5` 中有 `coords`，重新导出 feature+coords 的轻量索引；
5. **绝对禁止用数组顺序假装空间坐标**；
6. **绝对禁止生成虚假的规则网格 coords**。

如果本地确实不存在 coords：

- `spatial block masking` 主实验标记为不可执行；
- 保留 feature-context random masking 作为退化方案；
- 在 `data_audit.md` 中明确写出；
- 不得隐瞒。

## 3.3 审计旧 baseline 的 5-fold 划分

检查旧 baseline 代码，尽可能复现原始 5-fold patient-level split。

要求：

- patient 不能跨 fold
- 一个 patient 的所有 slides 必须处于同一 fold
- 如原 split 文件不存在，找到旧代码中的 split 生成逻辑
- 优先使用原 seed `20260723`
- 如果可以精确复现，保存为：

```text
splits/outer_fold_1.csv
...
splits/outer_fold_5.csv
```

每个文件至少：

```text
patient_id
split      # train / test
dss_status
dss_months
```

如果无法证明与旧实验完全一致：

- 创建新的固定 5-fold split；
- 尽量保证各 fold event 数接近；
- 把这一事实写进 `split_audit.md`；
- 新阶段所有方法统一使用这套 split；
- 不得将新结果与旧 0.660 做不严格的显著性比较，必须同时重新跑 B0 baseline。

---

# 4. 核心科研原则：严格 fold-wise pretraining

本阶段最容易发生的严重错误是：

> 在全部 487 个患者上先做自监督预训练，再 5-fold 做 survival test。

这是**主结果禁止的**，因为 test 患者的 pathology features 已经参与 encoder pretraining，会造成 transductive leakage。

主结果必须对每个 outer fold 独立执行：

```text
Outer train patients
    ↓
SSL pretraining（只允许 outer train pathology）
    ↓
survival fine-tuning（只允许 outer train DSS）
    ↓
Outer test inference
```

Outer test：

- 不能参与 normalization fitting
- 不能参与 PCA fitting
- 不能参与 prototype clustering
- 不能参与 SSL pretraining
- 不能参与 early stopping
- 不能参与 hyperparameter selection

如为了探索额外运行 “all-patient SSL pretrain”：

- 只能放在 `exploratory_transductive/`
- 不能作为论文主结果
- 必须明显标注 `TRANSDUCTIVE / NOT PRIMARY`

---

# 5. 本阶段模型总体框架

本阶段只输入 pathology。

目标结构：

```text
Frozen UNI2-h patch embeddings [N × 1536]
                 +
             2D coords
                 ↓
       learnable projection
           1536 -> 256
                 ↓
       context encoder
                 ↓
      patient representation
             z_path
                 ↓
          linear Cox head
                 ↓
          patient risk score
```

其中 UNI2-h 本身永远冻结。

---

# 6. 为避免 3 万 patch 的 O(N²) 爆炸，采用可执行的轻量 Context Encoder

不能直接对所有 30k patch 做 full global self-attention。

## 6.1 每个训练 step 的 patch sampling

默认：

```text
max_tokens_per_slide = 1024
```

要求：

- 若 slide patches <= 1024：全部使用
- 若 > 1024：
  - 训练时动态采样
  - 不允许永久只固定同一批 patch
  - 每 epoch/step 改变 sampling，使长期覆盖更多 patch
- 测试时使用 deterministic multi-sampling：
  - 例如 4 个固定 seed 的 1024-patch subsets
  - 对 patient representation / risk 做平均
- 同时记录 sampling coverage

如果显存允许，可额外测试 2048，但 1024 为默认主配置。

## 6.2 投影

```text
LayerNorm
Linear(1536, 256)
GELU
Dropout(0.1)
```

得到：

```text
[N, 256]
```

## 6.3 坐标位置编码

若 coords 可用：

1. 对每张 slide 的 x/y 做相对归一化；
2. 输入一个小 MLP：

```text
(x_norm, y_norm)
-> Linear(2, 64)
-> GELU
-> Linear(64, 256)
```

3. 与 patch token 相加。

必须做 `no_coords` ablation。

## 6.4 Context Transformer

保持低容量：

```text
d_model = 256
n_heads = 8
n_layers = 2   # 主配置
ffn_dim = 512
dropout = 0.1
```

不要一开始使用 8~12 层大 Transformer。

患者/slide summary 使用：

- learnable `[CLS]` token，或
- attention pooling

主配置固定一种；另一种可作为小规模 ablation。

---

# 7. Stage 1：Masked Spatial-Context Pretraining

这是本阶段核心创新验证。

## 7.1 目标

不使用 DSS。

要求模型学习：

> 给定一部分 UNI2-h patch embeddings 及空间上下文，推断被 mask 的 patch embedding。

说明：

- UNI2-h 已学习 patch-level morphology semantics；
- 本阶段不重新预测 RGB；
- 直接在 UNI2-h feature space 中做 masked feature modeling。

## 7.2 Mask 策略

若 coords 可用，必须比较：

### M1：Random token masking

随机 mask patch tokens。

### M2：Spatial block masking —— 主方法

随机选择一个或多个空间 anchor，按坐标邻域形成连续 block，mask 连续区域。

主 mask ratio：

```text
0.40
```

额外 ablation：

```text
0.25
```

原则：

- block mask 应明显比完全随机 token mask 更强调 long-range context
- 实现时记录每个 batch 实际 mask ratio
- 不允许 mask 后没有足够 visible tokens

若 coords 不可用：

- M2 不执行
- 明确记录原因
- M1 继续执行

## 7.3 Mask token

被 mask 的 patch 不允许把原 embedding 输入 encoder。

使用可学习：

```text
[MASK] ∈ R^256
```

同时保留对应位置编码。

## 7.4 Reconstruction head

```text
Linear(256, 512)
GELU
Linear(512, 1536)
```

预测原始 frozen UNI2-h feature。

目标 feature 在 loss 前做 L2 normalization。

## 7.5 SSL loss

主 loss：

\[
L_{mask}
=
\frac{1}{|M|}
\sum_{j\in M}
\left[
1-\cos(\hat h_j, h_j)
\right]
\]

同时记录但不用于主优化：

```text
masked cosine similarity
masked MSE
```

可选辅助 variance regularization 只能在发现 representation collapse 时加入，不能默认堆复杂 loss。

## 7.6 SSL early stopping

Outer-train 内再划分：

```text
inner_train
inner_val
```

该 inner split 仍必须 patient-level。

早停依据只能使用：

```text
inner_val masked cosine loss
```

不能看 outer-test。

建议：

```text
max_epochs = 50
patience = 8
```

如果训练很快可增加，但必须记录实际 epochs。

---

# 8. Stage 2：DSS Survival Fine-tuning

SSL 完成后，用相同 context encoder 初始化 survival model。

Patient / slide representation：

```text
z_path ∈ R^256
```

风险：

\[
r_i = w^T z_i + b
\]

## 8.1 多 slide 患者

先分别获得：

```text
z_slide_1
z_slide_2
...
```

患者级主配置采用：

```text
patient_z = mean(slide_z)
```

原因：只有极少患者有多 slide，不应为此引入复杂大模型。

可选轻量 slide attention 只做 supplementary ablation。

---

# 9. Survival loss 设计

必须比较以下损失，而不是直接“优化原始 C-index”。

## S1：Cox only

\[
L=L_{Cox}
\]

这是主 survival baseline。

## S2：Cox + pairwise ranking

对于可比较 pair：

\[
\delta_i=1,\quad T_i<T_j
\]

期望：

\[
r_i > r_j
\]

使用平滑 logistic ranking：

\[
L_{rank}
=
\frac{1}{|\mathcal P|}
\sum_{(i,j)\in\mathcal P}
\log\left(1+\exp(-(r_i-r_j)/\tau)\right)
\]

默认：

```text
tau = 1.0
```

总 loss：

\[
L=L_{Cox}+\lambda_{rank}L_{rank}
\]

默认先：

```text
lambda_rank = 0.2
```

只能在 outer-train 内调整。

## S3：Cox + IPCW ranking

由于约 487 人只有约 65 个 DSS events，censoring 很重。

实现 IPCW-compatible pair weighting：

- censoring distribution 只能在 outer-train 拟合
- 禁止使用 outer-test 信息拟合 \(G(t)\)

输出：

```text
L = L_Cox + lambda_ipcw * L_IPCW_rank
```

如果 IPCW 实现/数值稳定性出现问题，不得偷偷替换；必须记录并保留 S1/S2。

---

# 10. 必须运行的模型矩阵

为了验证 motivation，至少运行以下方法。

## B0：历史 mean baseline 复现

```text
UNI2-h
-> patch mean
-> multi-slide mean
-> fold-wise scaler/PCA
-> ridge Cox
```

目标：

- 新 split 下得到严格 comparable baseline
- 若成功复现旧 split，结果应接近历史 `0.660046`

## B1：ABMIL / Gated Attention MIL

```text
Frozen UNI2-h
-> projection
-> gated attention pooling
-> z_path
-> Cox
```

用途：

> 判断单纯 weak-supervised patch weighting 是否足够。

## B2：Context Transformer from scratch

与 Proposed 使用完全相同 context architecture，但：

```text
不做 SSL pretraining
```

直接：

```text
random init
-> DSS survival training
```

用途：

> 判断收益来自 architecture 还是 self-supervised context pretraining。

## P0：SSL Context Encoder + Frozen Linear Cox Probe

```text
masked context SSL
-> freeze encoder
-> extract z_path
-> L2/linear Cox probe
```

用途：

> 最直接判断 SSL 学到的 representation 本身是否包含更多预后信息。

## P1：SSL Context Encoder + Cox Fine-tuning

```text
SSL
-> survival fine-tune
-> Cox
```

这是核心 proposed。

## P2：SSL Context Encoder + Cox + Ranking Fine-tuning

```text
SSL
-> Cox + pairwise ranking
```

## P3：SSL Context Encoder + Cox + IPCW Ranking

如实现稳定则运行。

---

# 11. 关键消融 —— 不要无限堆实验，只做能回答核心问题的

必须做：

| Ablation | 要回答的问题 |
|---|---|
| no SSL vs SSL | 收益是否来自 self-supervised pretraining |
| random mask vs spatial block mask | 空间连续上下文是否真的有价值 |
| no coords vs coords | 2D pathology spatial information 是否必要 |
| mask 0.25 vs 0.40 | mask 难度是否影响表示 |
| Cox vs Cox+rank | 显式排序约束是否有价值 |

不要在本阶段加入：

- RNA
- clinical
- multimodal Transformer
- diffusion
- large foundation model finetuning
- 复杂 prototype branch
- 不必要的 10+ loss

本阶段必须保持因果链清楚。

---

# 12. 表征好坏如何评价 —— 必须同时做 representation probe 与 end-to-end survival

这是本阶段最重要的评价规范。

## 12.1 Representation Probe

对每个方法得到的 patient embedding：

```text
z_path
```

统一使用**同一个低容量 Cox probe**：

```text
StandardScaler（fold-wise）
L2 / ridge Cox
```

禁止每种 representation 使用不同强度的复杂 prediction head。

至少比较：

```text
mean representation
ABMIL representation
context-from-scratch representation
SSL-frozen representation
SSL+survival-FT representation
```

这样才能讨论：

> patient-level pathology representation 本身是否更好。

## 12.2 End-to-End Survival

同时报告每种 pipeline 自己的最终 risk performance。

---

# 13. 生存评价指标

由于 censoring 很重，禁止只报 Harrell C-index。

## 13.1 Primary

### 1. OOF Harrell C-index

所有 outer-fold test prediction 合并后：

```text
OOF Harrell C
```

### 2. OOF Uno's C-index

作为对 censoring 更稳健的共同主要指标。

IPCW censoring estimator 必须严格训练折内处理；计算 OOF 汇总时实现方式需在报告中说明。

## 13.2 Secondary

若统计条件允许，报告：

```text
time-dependent AUC @ 12 months
time-dependent AUC @ 24 months
time-dependent AUC @ 36 months
Integrated Brier Score (IBS)
```

如果某个时间点：

- at-risk 人数过少
- event 不足
- estimator 不可识别

则输出：

```text
NA
```

并解释，不允许强算一个不可靠数字。

## 13.3 KM 风险分层

每 fold：

- 只用 training patients 确定风险 cutpoint（默认 train median）
- 再应用到 test
- 禁止使用 test median 切组

最后可汇总 OOF high/low risk：

- Kaplan-Meier
- log-rank

该结果为辅助，不作为唯一判断。

---

# 14. 不确定性与统计比较

所有主方法必须产生：

```text
patient-level OOF risk
```

然后做 paired patient-level comparison。

至少：

```text
1000 次 bootstrap
```

输出：

```text
OOF metric
95% CI
delta vs B0
paired bootstrap CI of delta
```

同时报告：

```text
5 fold test metrics
mean ± std
每个 fold 是否相对 B0 改善
```

不要仅因为 CI 跨 0 就说“模型完全无效”，也不要仅因为点估计更高就说“显著优于”。

使用以下分级：

### Strong support

- Proposed 对 B0 的 Harrell C 与 Uno C 均为正向改善；
- fold 层面改善具有较强一致性；
- paired CI 对至少一个主要指标给出明确统计支持；
- 且其他 calibration/discrimination 指标未出现明显恶化。

### Partial / promising support

- OOF 主要指标改善；
- fold 方向总体一致或多数为正；
- 但受 65 events 限制，CI 仍跨 0。

### Not supported

- OOF Harrell/Uno 无稳定改善；
- 或改进只来自个别 fold；
- 或 SSL reconstruction 明显改善但 survival performance 不改善。

---

# 15. 必须记录的自监督指标

每 fold 至少输出：

```text
ssl_train_loss.csv
ssl_val_loss.csv
masked_cosine_similarity.csv
masked_mse.csv
```

并比较：

```text
random masking
spatial block masking
```

但必须强调：

> SSL reconstruction 更好 ≠ survival representation 一定更好。

最后要检查两者是否相关。

---

# 16. 表征分析 —— 只做有科学意义的

输出 patient embeddings：

```text
embeddings/
  B0_mean/
  B1_abmil/
  B2_context_scratch/
  P0_ssl_frozen/
  P1_ssl_cox/
  P2_ssl_rank/
```

每个至少保存：

```text
patient_id
fold
dss_months
dss_status
embedding
oof_risk
```

另外做：

1. embedding norm 分布；
2. collapse 检查：
   - feature std
   - pairwise cosine distribution
3. UMAP/t-SNE 仅作可视化，不作为主要性能证据；
4. 可以按：
   - DSS event
   - risk quartile
   - cancer site / tissue source site
   上色检查明显 batch/site effect。

---

# 17. Attention / spatial 可解释性

如果 coords 可用：

对至少：

```text
5 个高风险患者
5 个低风险患者
```

输出 patch attention 或 contextual importance：

```text
patient_id
slide_id
x
y
attention_score
```

生成 coordinate heatmap。

注意：

- 没有原始 WSI 图像时，不要伪造病理热图底图；
- 仅生成坐标空间 scatter heatmap；
- 如果原始 thumbnail 可访问，再叠加；
- 明确区分“attention importance”与“因果病灶”。

---

# 18. 超参数与防止小样本过拟合

本数据约：

```text
487 patients
65 DSS events
```

所以必须控制容量。

默认：

```text
projection_dim = 256
transformer_layers = 2
heads = 8
ffn_dim = 512
dropout = 0.1
```

Survival fine-tuning：

```text
batching:
  尽量采用 risk-set compatible full-batch / sufficiently large patient batch
```

如使用 mini-batch Cox：

- 必须解释近似方式；
- 优先 patient batch > patch batch；
- 不允许把同一患者不同 sampled subsets 当成独立生存样本。

建议：

```text
AdamW
weight_decay = 1e-4
```

学习率候选只允许在 inner train/val 中选择，例如：

```text
1e-4
3e-4
```

不要做巨大 grid search。

---

# 19. 训练稳定性

固定随机种子：

```text
20260723
20260724
20260725
```

主结果至少：

- B0 可单 seed
- B1/B2/P1/P2 等神经模型最好跑 3 seeds

如果算力不足：

1. 先完成所有方法 seed `20260723`
2. 对最强 proposed + B1/B2 再补 3 seeds
3. 不能悄悄减少实验；在 run manifest 明确说明

---

# 20. GPU / 工程要求

自动检测：

```text
CUDA
GPU memory
PyTorch version
```

使用：

- AMP mixed precision
- gradient accumulation
- deterministic evaluation
- checkpoint resume
- cached data index

不要一次把全部几百万 patch feature 载入 RAM/GPU。

使用 lazy loading / mmap / per-slide loading。

---

# 21. 断点恢复与防止 Codex 提前终止

创建：

```text
outputs/pathology_context_phase1/run_state.json
```

至少字段：

```json
{
  "stage": "",
  "fold": 0,
  "model": "",
  "seed": 0,
  "status": "",
  "completed_jobs": [],
  "pending_jobs": []
}
```

每完成一个：

```text
model × fold × seed
```

立即更新。

每个 job：

```text
checkpoints/
logs/
metrics/
```

独立保存。

如果程序崩溃：

- 从已完成 checkpoint / manifest 恢复
- 不重新覆盖成功结果

**只要 pending_jobs 非空，不允许输出“实验全部完成”。**

---

# 22. 最终目录要求

最终至少：

```text
outputs/pathology_context_phase1/
├── audit/
├── splits/
├── configs/
├── checkpoints/
├── logs/
├── ssl_metrics/
├── embeddings/
├── oof_predictions/
├── fold_metrics/
├── statistical_tests/
├── figures/
├── run_state.json
├── experiment_manifest.json
├── results_master.csv
├── RESULTS.md
└── NEXT_DECISION.md
```

---

# 23. results_master.csv 必须包含

每一行一个方法 × seed，至少：

```text
model
ssl_pretrained
mask_type
use_coords
survival_loss
seed
n_patients
n_events
oof_harrell_c
oof_uno_c
auc_12m
auc_24m
auc_36m
ibs
delta_harrell_vs_B0
delta_uno_vs_B0
bootstrap_ci_low_harrell
bootstrap_ci_high_harrell
bootstrap_ci_low_uno
bootstrap_ci_high_uno
n_folds_better_than_B0
parameter_count
training_time
peak_gpu_memory
```

---

# 24. OOF prediction 文件要求

每个方法保存：

```text
patient_id
outer_fold
dss_months
dss_status
oof_risk
```

必须保证：

```text
每个 patient 恰好一条 OOF prediction
```

多 seed 时可以：

```text
patient_id
seed
...
```

或分文件保存。

---

# 25. 必须做的 sanity checks

## 25.1 Baseline replication

如果旧 split 可复现：

```text
B0 OOF Harrell C
```

应与旧值约 `0.660046` 接近。

如果明显偏离：

- 先停止 proposed 结果解读；
- 排查 split / feature / scaler / PCA / Cox 实现；
- 记录原因。

## 25.2 Label permutation test

对最强 proposed 做至少一次：

```text
permute DSS labels within outer-train
```

期望 OOF C 回到接近无信息水平。

如果仍异常高：

> 高度怀疑 leakage。

## 25.3 Patient leakage

检查：

```text
train_patient_ids ∩ test_patient_ids == empty
```

以及所有 slide。

## 25.4 SSL leakage

记录每个 fold SSL 用过的所有 patient id，确认：

```text
ssl_train_patients ∩ outer_test_patients == empty
```

---

# 26. 本阶段禁止事项

禁止：

1. 把 C-index 当 supervision label；
2. 把 censored patient 简单标成“低风险类别”；
3. 把 `dss_status=0` 当作“最终不会死亡”；
4. 在全体患者上 SSL 后再声称是严格 CV；
5. 使用 test fold 调 mask ratio / layer 数 / LR；
6. 用 clinical/TNM/RNA 帮 pathology model；
7. 把 slide 当独立患者导致同一 patient 跨 fold；
8. 用 feature 数组顺序伪造 coords；
9. 只运行 1~2 个容易的 baseline 就提前总结；
10. 只生成代码而不执行实验；
11. 看到运行时间长就擅自删除 P1/P2/B1/B2；
12. 只报告最优 seed；
13. 只报告 fold mean 而不报告 pooled OOF；
14. 只报告 Harrell C 而忽略 censoring-sensitive metrics；
15. 未完成全部 jobs 时写“任务完成”。

---

# 27. RESULTS.md 必须回答的科研问题

不要只是堆数字。

按以下问题回答：

## Q1

Mean pooling 是否确实限制了 UNI2-h pathology representation？

比较：

```text
B0 vs B1 vs B2
```

## Q2

Context Transformer 本身是否有效？

比较：

```text
B1 vs B2
```

## Q3

Masked context pretraining 是否带来真实收益？

比较：

```text
B2 vs P0 vs P1
```

尤其区分：

```text
representation probe gain
end-to-end survival gain
```

## Q4

Spatial block masking 是否比 random masking 更符合 WSI 上下文建模？

比较：

```text
random mask vs spatial block mask
```

## Q5

coords 是否真正有用？

比较：

```text
with coords vs no coords
```

## Q6

显式 survival ranking 是否比 Cox-only 更适合当前重删失数据？

比较：

```text
P1 vs P2 vs P3
```

## Q7

SSL reconstruction 与 survival performance 是否一致？

如果：

```text
mask loss ↓
但 C-index 不升
```

必须明确说明：

> 模型学到了结构可预测性，但这些结构不一定与 DSS 相关。

---

# 28. NEXT_DECISION.md：必须给出明确决策，不允许模糊

最后根据结果选择以下之一。

## Decision A：核心假设成立

如果：

- SSL proposed 在 Harrell C / Uno C 上稳定优于 B0/B1/B2；
- fold 方向总体一致；
- representation probe 也有改善；
- 无 leakage 证据；

则建议下一阶段：

```text
Pathology encoder 固定/半冻结
+
RNA encoder
+
Clinical encoder
+
incremental multimodal fusion
```

并首先验证：

```text
Clinical
vs
Clinical + Pathology
```

再验证：

```text
Clinical + Pathology
vs
Clinical + Pathology + RNA
```

## Decision B：只有弱支持

如果点估计改善但统计不稳定：

- 保留 pathology encoder
- 优先检查 event scarcity / DSS endpoint
- 尝试更稳健 survival objective
- 不立刻堆复杂 multimodal architecture

## Decision C：假设不成立

如果：

```text
P1/P2 <= B1/B2
```

且多个指标一致：

不要继续强行优化 masked context。

分析失败原因：

- UNI2-h 已经足够强，额外 SSL 无增益？
- 1024 sampling 丢失 global morphology？
- DSS event 太少？
- 预训练目标与 prognosis 无关？
- coords / patch context 不可靠？
- pathology 对 DSS 本来就是弱模态？

然后提出下一条最小验证路线。

---

# 29. 最终完成条件

只有以下全部满足才算完成：

- [ ] 数据审计完成
- [ ] coords 审计完成
- [ ] 5-fold 严格 patient split 固化
- [ ] B0 跑完
- [ ] B1 跑完
- [ ] B2 跑完
- [ ] P0 跑完
- [ ] P1 跑完
- [ ] P2 跑完
- [ ] P3 若可稳定实现则完成，否则给出明确失败记录
- [ ] random/block mask ablation 完成
- [ ] coords ablation 完成
- [ ] mask ratio ablation 完成
- [ ] OOF predictions 完整
- [ ] Harrell C 完整
- [ ] Uno C 完整
- [ ] time-dependent AUC/IBS 在可计算条件下完成
- [ ] paired bootstrap 完成
- [ ] leakage checks 完成
- [ ] label permutation sanity check 完成
- [ ] embeddings 导出完成
- [ ] `results_master.csv` 完成
- [ ] `RESULTS.md` 完成
- [ ] `NEXT_DECISION.md` 完成
- [ ] `run_state.json` 中 `pending_jobs=[]`

如果任何一项未完成：

```text
status = PARTIAL
```

并继续执行剩余任务，不得声称全部完成。

---

# 30. 最终研究判断标准

本阶段不是为了得到“一个更复杂的模型”，而是验证下面这条因果链：

```text
UNI2-h
已经提供 patch-level pathology semantics
        ↓
Mean pooling
忽略 WSI patch context
        ↓
Masked spatial-context pretraining
学习 patch-to-patch / local-to-global morphology
        ↓
Patient-level pathology representation
        ↓
DSS weak supervision
进行 outcome-guided survival calibration
        ↓
更可靠的 patient risk
```

最终必须回答：

> **在严格 inductive patient-level cross-validation 下，whole-slide contextual self-supervision 是否真的比直接 mean/MIL aggregation 学到了更有 DSS 预后信息的患者级病理表示？**

如果答案是“否”，要接受结果并解释原因；禁止为了支持假设而选择性汇报。

---

# 31. 开始执行

现在开始：

1. 扫描项目根目录；
2. 读取已有 audit/baseline 代码和结果；
3. 输出 Stage 0 审计；
4. 固化 split；
5. 先复现 B0；
6. 再按 B1 → B2 → P0 → P1 → P2 → P3 顺序执行；
7. 每完成一个 job 更新 `run_state.json`；
8. 不要在实验真正完成前结束会话。
