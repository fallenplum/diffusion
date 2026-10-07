# Phase5 — Endpoint & Survival-Supervision Budget Audit

## 0. 目标

本阶段不设计新融合模块，只回答两个问题：

\[
\boxed{\text{当前性能瓶颈主要来自 65 个 DSS events，还是来自 patient representation 本身？}}
\]

以及：

\[
\boxed{\text{如果换成事件更多的生存终点，现有表示与简单融合是否会自然变得更稳定？}}
\]

本阶段只做诊断，不根据结果挑最好看的 endpoint 投稿。

## 1. 分支

从：

`codex/phase4ar-dss-free-grounding-repair`

新建：

`codex/phase5-endpoint-supervision-budget-audit`

输出：

`reports/phase5_endpoint_supervision_budget_audit/`

## 2. Endpoint audit

在同一 512-patient complete-case cohort 上审计：

- DSS
- OS
- PFI
- PFS：仅当仓库存在明确、可追溯、预先定义的 PFS 字段时使用；禁止临时自行拼接定义

对每个 endpoint 报告：

- valid N
- event N
- censoring rate
- median follow-up
- per-fold event counts
- 与 DSS event 的重叠比例
- missingness
- time range / obvious invalid values

如果 PFI/PFS/OS 不能在同一 512 人上合法构造，则如实记录，不补值。

输出：

`endpoint_audit.csv`

## 3. Endpoint 选择原则

DSS 仍是原始研究 endpoint。OS/PFI/PFS 只是监督预算审计。

不得因为某 endpoint C-index 更高就事后把它改成主终点。

只有满足以下条件，后续才允许正式考虑更换：

1. 医学问题与 endpoint 匹配；
2. endpoint 定义与来源清晰；
3. event 数量明显高于 DSS；
4. 缺失率可接受；
5. 提升不是仅来自 Clinical shortcut；
6. 在相同模型下，非临床模态增量和稳定性也同步改善。

## 4. 锁定模型：禁止新 architecture

对每个合法 endpoint，只跑以下模型。

### B0 Clinical-only

当前 strongest legal C-full。

### B1 Pathology-only

当前锁定 Gated ABMIL + UNI2-h。

### B2 RNA-only

当前锁定 RNA-Linear。

### B3 Additive C+P+R

使用 outer-train cross-fitted：

\[
r=a_Cr_C+a_Pr_P+a_Rr_R
\]

只训练三个 scalar weights。

### B4 Frozen DSS-free P+R representation probe

复用 Phase4A-R 的 M2 Naive P+RNA representation，不使用 M3/Hallmark mechanism。

必须：

- encoder 完全 frozen；
- 每个 outer fold 只使用该 fold outer-train 上预训练得到的 checkpoint；
- outer-test 从未参与 representation pretraining；
- 只训练一个极小 Cox head：

\[
r_Z=w^\top z_{PR}
\]

最终和 Clinical 组合：

\[
r=a_Cr_C+a_Zr_Z
\]

这里只训练 Cox head 与 \(a_C,a_Z\)，不得 fine-tune representation encoder。

目的：

\[
\boxed{\text{检验 DSS-free disease-progression representation 是否能减少 survival label 需求}}
\]

## 5. Cross-validation

每个 endpoint 独立使用 deterministic patient-level 5-fold CV。

要求：

- 每 fold 至少有可计算 Harrell / Uno 的事件；
- 若现有固定 folds 对某 endpoint 极度失衡，则创建 endpoint-specific deterministic event-stratified folds；
- 新 folds 必须在任何模型训练前固定并保存；
- 同一 endpoint 的所有模型使用完全相同 folds。

输出：

`endpoint_split_manifest.json`

## 6. Metrics

每个 endpoint 统一报告：

- Harrell C-index
- canonical Uno C-index
- 3 seeds
- fold metrics
- seed-level OOF metrics
- 2000 次 patient-level paired bootstrap

主要比较：

\[
B3-B0
\]

和：

\[
B4-B0
\]

即：

- 简单三模态 additive 是否超过 Clinical；
- DSS-free frozen representation 是否超过 Clinical。

## 7. Event-budget learning curve

对每个事件数明显多于 DSS 的 endpoint（优先 PFI，其次 OS）：

在每个 outer fold 的 outer-train 内，只改变 survival-supervised training budget。

固定三个预算：

- 约 25% 可用 events
- 约 50% 可用 events
- 100% events

要求：

- 只对子训练集做 deterministic patient subsampling；
- outer-test 完全不变；
- 每个预算保持尽量相似的 censoring / event 比例；
- representation encoder 始终 frozen；
- 只比较 B0 / B3 / B4；
- 每个预算 3 个固定 subsample seeds。

输出：

`event_budget_curve.csv`

目标是观察：

\[
\text{event count}\uparrow
\Rightarrow
\text{variance 是否下降？multimodal gain 是否出现？}
\]

## 8. 决策

### A. LABEL_SCARCITY_PRIMARY

满足：

- OS/PFI event 数明显高于 DSS；
- 随 event budget 增加，B3 或 B4 的 Harrell/Uno 稳定改善；
- multimodal gain 相对 Clinical 从接近 0 变为稳定正值；
- bootstrap CI 明显收窄。

解释：

\[
\boxed{\text{当前主要瓶颈是 survival supervision 太少}}
\]

下一阶段进入：

`DSS-free multimodal representation pretraining + low-capacity survival transfer`

### B. REPRESENTATION_PRIMARY

满足：

- OS/PFI events 明显增加；
- 但 B3/B4 仍不能稳定超过 Clinical；
- event-budget 增加也不产生清晰 multimodal gain。

解释：

\[
\boxed{\text{瓶颈不只是 65 个 DSS events，而是当前 patient representation / conditional signal 本身}}
\]

### C. PRETRAINING_TRANSFER_SUPPORTED

如果 B4 在至少一个医学合理 endpoint 上：

- mean Harrell > Clinical
- mean Uno > Clinical
- ≥2/3 seeds 双指标同向
- 至少一个主指标 bootstrap CI lower > 0

则说明：

\[
\boxed{\text{“先学 representation，再用少量 survival label 学 prognostic direction”获得直接证据}}
\]

下一阶段才构建真正 tri-modal DSS-free pretraining。

### D. ENDPOINT_SWITCH_JUSTIFIED

只在以下条件同时满足时允许讨论换主 endpoint：

- endpoint 医学含义适合研究问题；
- event N 显著更高；
- 数据完整；
- 非临床增量比 DSS 更稳定；
- 不只是 Clinical-only 自己变强。

若 PFI 满足而 OS 不满足，优先考虑 PFI。

### E. DSS_REMAINS_PRIMARY

若 OS/PFI 虽 event 更多，但：

- Clinical dominance 更强；
- multimodal gain 没改善；
- 或数据定义不稳定；

则继续 DSS，不因性能改变 endpoint。

## 9. 禁止事项

- 不新增 Transformer / Q-Former / cross-attention；
- 不使用 Hallmark M3 作为主 proposed；
- 不对 endpoint 做 performance-based cherry-picking；
- 不用 outer-test 调 endpoint、fold、event budget；
- 不把 OS/PFI 高 C-index 自动解释成“模型更好”；
- 不做 end-to-end 大规模 fine-tuning。

## 10. 输出

至少包含：

- `RESULTS.md`
- `DECISION.md`
- `endpoint_audit.csv`
- `endpoint_split_manifest.json`
- `seed_oof_metrics.csv`
- `fold_metrics.csv`
- `paired_bootstrap.csv`
- `event_budget_curve.csv`
- `frozen_probe_metrics.csv`
- `endpoint_provenance.json`
- `validation.json`
- `failure_ledger.csv`

## 11. RESULTS.md 必须回答

1. DSS / OS / PFI / PFS 各有多少 events？
2. 哪个 endpoint 的监督预算明显高于 DSS？
3. Clinical-only 在不同 endpoint 下如何变化？
4. Pathology / RNA 的单模态能力是否随 endpoint 改变？
5. Additive C+P+R 在事件更多的 endpoint 上是否开始稳定超过 Clinical？
6. Frozen DSS-free P+R representation 是否能被一个极小 Cox head有效利用？
7. event budget 从 25%→50%→100% 时，性能方差和 multimodal gain 如何变化？
8. 当前瓶颈更符合：
   - `LABEL_SCARCITY_PRIMARY`
   - `REPRESENTATION_PRIMARY`
   - `PRETRAINING_TRANSFER_SUPPORTED`
   - `ENDPOINT_SWITCH_JUSTIFIED`
   - `DSS_REMAINS_PRIMARY`
9. 下一阶段是否应保留 DSS 做 event-efficient representation pretraining、换 PFI/OS，或停止复杂 multimodal architecture。

## 12. 本阶段真正要回答的问题

\[
\boxed{\text{65 个 DSS events 是否只是“最后一层 prognostic mapping”的监督不足，}}
\]

还是：

\[
\boxed{\text{即使 survival supervision 增加，当前三模态 patient representation 仍然没有可泛化增量。}}
\]

只有把这两个问题分开，后续才知道应该优化 endpoint、representation learning，还是停止继续堆模型。
