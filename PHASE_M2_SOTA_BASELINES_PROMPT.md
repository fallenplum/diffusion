# Phase M2 — Published Multimodal Survival SOTA Unified Benchmark

## Goal
在 `fallenplum/diffusion` 上完成公开多模态癌症生存预测 SOTA baseline 的统一输入适配与公平复现。

本阶段不设计新模型、不做缺失模态、不做 diffusion。只做：
1. 跑通代表性 published baselines；
2. 得到可信的性能、稳定性、计算量对比，为下一阶段 failure mechanism audit 提供依据。

## 1. 固定数据与协议
沿用 `codex/phase-m1-multimodal-baseline`：

```text
DSS endpoint
512 patients
65 DSS events
fixed 5-fold patient-level split
seed = 20260904
```

复用：

```text
outputs/phase_m1r/fixed_patient_folds.csv
```

禁止重新划分 split。

统一输入：

```text
Pathology: TCGA COAD/READ UNI2-h H5 patch embeddings
Genomics: RNA + CNA
Clinical: AGE, SEX, AJCC stage, T/N/M stage
```

所有 preprocessing / feature selection / normalization / fitting 必须 outer-train only。

## 2. Baselines
必须实现：

```text
B0  EarlyFusion / Concat + Cox
B1  TransSurv-style multimodal Transformer
B2  MGCT
B3  CAMR
B4  MMP
B5  MOTCat / MCAT-style co-attention
```

要求：
- 保留每篇方法的核心机制；
- 输入统一替换为 UNI2-h + RNA/CNA + Clinical；
- 不照搬原论文 feature extractor 差异；
- 不得把复杂 baseline 退化成 concat；
- 可做 architecture-faithful adaptation；
- 无法可靠复现则标记 `FAILED_TO_REPRODUCE`。

## 3. 统一输入接口
建立公共 loader：

```text
patient_id
outer_fold
dss_time
dss_event
pathology patch tokens
RNA features
CNA features
clinical features
```

Pathology：

```text
真实 H5 patch features
K = 1024
train: random sampling
test: deterministic sampling
```

RNA/CNA：outer-train 内完成 preprocessing。

Clinical：fold-wise imputation + encoding。

## 4. 公平性
所有 baseline 必须使用：

```text
same 512 patients
same 5 folds
same DSS labels
same pathology H5
same RNA/CNA
same clinical variables
```

主指标：

```text
OOF Harrell C-index
OOF Uno C-index
```

同时记录：

```text
5-fold mean ± std
worst fold
parameter count
peak GPU memory
training time
```

神经 baseline 固定 3 seeds：

```text
20260904
20260905
20260906
```

## 5. Survival objective
优先保持论文原始 survival objective。

若需统一 Cox，必须使用 Phase1.5R2 已验证正确的实现。

正式训练前通过：

```text
autograd check
finite-difference check
one-step descent
risk-direction sanity
```

严禁重新引入旧 Cox gradient bug。

## 6. 禁止
```text
重新划分 outer split
seed hunting
根据结果删 baseline
修改 endpoint
test fold 参与预处理/调参/early stopping
把失败复现写成成功
设计我们的新模型
加入 missing-modality diffusion
```

## 7. 输出
新分支：

```text
codex/phase-m2-sota-baselines
```

代码建议：

```text
src/phase_m2/
  data.py
  common.py
  early_fusion.py
  transsurv.py
  mgct.py
  camr.py
  mmp.py
  motcat.py
  runner.py
```

结果：

```text
reports/phase_m2_sota_baselines/
  RESULTS.md
  main_metrics.csv
  fold_metrics.csv
  seed_metrics.csv
  compute_profile.csv
  reproduction_status.csv
  protocol.json
  run_manifest.json
```

不要上传 patient IDs、raw data、checkpoints、H5/NPY/NPZ、patient-level predictions。

## 8. RESULTS.md 主表
必须包含：

```text
Method
Mechanism
Harrell C
Uno C
seed mean ± std
fold mean ± std
worst fold
params
GPU memory
training time
reproduction status
```

机制标签：

```text
Concat      -> no explicit interaction
TransSurv   -> transformer cross-modal interaction
MGCT        -> mutual-guided cross-modal interaction
CAMR        -> shared/specific + alignment
MMP         -> multimodal prototypes
MOTCat/MCAT -> co-attention / transport-guided interaction
```

## 9. 完成条件
只有满足以下条件才写：

```text
Phase M2 COMPLETE
```

要求：
- 至少 5 个 representative baselines 跑通；
- MGCT、CAMR、MMP 必须包含；
- 全部采用统一 512-patient / 5-fold protocol；
- 成功 neural baseline 完成 3 seeds；
- Harrell / Uno / fold / seed / compute 信息齐全；
- 无数据泄漏；
- 无 Cox 实现错误。

否则输出：

```text
Phase M2 BLOCKED
```

并列出失败模型和原因。

## 10. 最后停止
完成后只输出：

1. baseline 排名；
2. 每个模型的核心机制；
3. 哪类机制有效；
4. 哪类机制失败；
5. 值得下一阶段分析的 failure signatures。

不要设计新模型。

push 到：

```text
codex/phase-m2-sota-baselines
```

然后停止。
