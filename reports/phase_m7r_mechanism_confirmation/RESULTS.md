# Phase M7R Results

## 1. Executive conclusion

正式完成 7×5×5=175 个 cells。主要观察：F3 vs F1（P2）为 ΔH=0.0064, ΔU=0.0041, seed-positive H/U=3/3, pooled 25-cell direction=True；F3 vs B0（P6）为 ΔH=-0.0118, ΔU=-0.0050, seed-positive H/U=1/2, pooled 25-cell direction=False。Decision D。下述结论区分点估计、seed-level 证据和 patient-level bootstrap。

## 2. Protocol integrity

N=512，DSS events=65，fixed outer folds=5，fresh seeds=[20261002, 20261003, 20261004, 20261005, 20261006]，K=1024，formal cells=175。Cox=PASS；initialization=PASS；FiLM=PASS；permutation=True。

## 3. Main metrics

| method | Harrell mean±SD | Uno mean±SD | worst Harrell | worst Uno |
|---|---:|---:|---:|---:|
| F0 MCAT-Original | 0.7920 ± 0.0637 | 0.8135 ± 0.0664 | 0.6722 | 0.6684 |
| A0 AdaptivePath-SemanticConcat | 0.7830 ± 0.0597 | 0.8048 ± 0.0690 | 0.6709 | 0.6652 |
| F1 ABMIL-SemanticConcat | 0.7804 ± 0.0796 | 0.8027 ± 0.0913 | 0.6324 | 0.6070 |
| F2 ABMIL-MatchedResidualMLP | 0.7827 ± 0.0830 | 0.7997 ± 0.0931 | 0.5998 | 0.5854 |
| F3 ABMIL-OmicToPath-FiLM | 0.7868 ± 0.0805 | 0.8068 ± 0.0912 | 0.6324 | 0.6073 |
| F6 ABMIL-ShuffledSixSignature-FiLM | 0.7885 ± 0.0791 | 0.8023 ± 0.0894 | 0.6065 | 0.5880 |
| B0 CommonOnly-OriginalFusion | 0.7986 ± 0.0656 | 0.8118 ± 0.0609 | 0.6556 | 0.7038 |

## 4. Seed-level replication

每个 comparison 按 5 个 seed 的五折均值解释；25 个 cells 不当作 25 个独立实验。
- P1: 20261002:ΔH=0.0305,ΔU=0.0052; 20261003:ΔH=-0.0394,ΔU=-0.0230; 20261004:ΔH=0.0280,ΔU=0.0045; 20261005:ΔH=-0.0155,ΔU=-0.0316; 20261006:ΔH=-0.0167,ΔU=0.0344
- P2: 20261002:ΔH=-0.0012,ΔU=-0.0093; 20261003:ΔH=0.0283,ΔU=0.0302; 20261004:ΔH=0.0031,ΔU=0.0006; 20261005:ΔH=0.0023,ΔU=0.0001; 20261006:ΔH=-0.0004,ΔU=-0.0012
- P3: 20261002:ΔH=-0.0044,ΔU=-0.0071; 20261003:ΔH=0.0116,ΔU=0.0178; 20261004:ΔH=0.0006,ΔU=0.0017; 20261005:ΔH=0.0121,ΔU=0.0106; 20261006:ΔH=0.0005,ΔU=0.0122
- P4: 20261002:ΔH=-0.0047,ΔU=-0.0062; 20261003:ΔH=-0.0006,ΔU=0.0111; 20261004:ΔH=0.0008,ΔU=0.0147; 20261005:ΔH=0.0024,ΔU=0.0026; 20261006:ΔH=-0.0060,ΔU=0.0002
- P5: 20261002:ΔH=0.0142,ΔU=0.0114; 20261003:ΔH=-0.0362,ΔU=-0.0049; 20261004:ΔH=0.0153,ΔU=0.0094; 20261005:ΔH=0.0179,ΔU=-0.0271; 20261006:ΔH=-0.0368,ΔU=-0.0225
- P6: 20261002:ΔH=-0.0207,ΔU=-0.0180; 20261003:ΔH=-0.0281,ΔU=0.0033; 20261004:ΔH=0.0310,ΔU=0.0313; 20261005:ΔH=-0.0010,ΔU=-0.0178; 20261006:ΔH=-0.0400,ΔU=-0.0236

## 5. Paired comparisons

P1 F1 vs A0: ΔH=-0.0026, ΔU=-0.0021, seed-positive H/U=2/3, pooled 25-cell direction=False
P2 F3 vs F1: ΔH=0.0064, ΔU=0.0041, seed-positive H/U=3/3, pooled 25-cell direction=True
P3 F3 vs F2: ΔH=0.0041, ΔU=0.0070, seed-positive H/U=4/4, pooled 25-cell direction=True
P4 F3 vs F6: ΔH=-0.0016, ΔU=0.0045, seed-positive H/U=2/4, pooled 25-cell direction=False
P5 F3 vs F0: ΔH=-0.0051, ΔU=-0.0067, seed-positive H/U=3/2, pooled 25-cell direction=False
P6 F3 vs B0: ΔH=-0.0118, ΔU=-0.0050, seed-positive H/U=1/2, pooled 25-cell direction=False

## 6. OOF paired bootstrap uncertainty

每个 seed/comparison 使用 2000 次 patient-level paired resampling，同一索引同时作用于 candidate 和 baseline；bootstrap 仅量化不确定性。
- P1 seed 20261002: ΔH=0.1082 [0.0116,0.1974], ΔU=0.0561 [-0.0509,0.1573]
- P1 seed 20261003: ΔH=-0.0608 [-0.1505,0.0303], ΔU=-0.0361 [-0.1429,0.0644]
- P1 seed 20261004: ΔH=0.0350 [-0.0543,0.1222], ΔU=-0.0176 [-0.1014,0.0722]
- P1 seed 20261005: ΔH=-0.0090 [-0.0946,0.0773], ΔU=-0.0469 [-0.1646,0.0695]
- P1 seed 20261006: ΔH=0.0174 [-0.0737,0.1131], ΔU=0.1131 [0.0085,0.2227]
- P2 seed 20261002: ΔH=0.0046 [-0.0816,0.0890], ΔU=0.0156 [-0.0789,0.1149]
- P2 seed 20261003: ΔH=0.0407 [-0.0553,0.1317], ΔU=0.0576 [-0.0290,0.1441]
- P2 seed 20261004: ΔH=0.0126 [-0.0739,0.0978], ΔU=0.0072 [-0.0827,0.0962]
- P2 seed 20261005: ΔH=0.0008 [-0.0891,0.0920], ΔU=0.0146 [-0.1098,0.1376]
- P2 seed 20261006: ΔH=-0.0114 [-0.1041,0.0718], ΔU=-0.0047 [-0.0767,0.0641]
- P3 seed 20261002: ΔH=-0.0060 [-0.0886,0.0790], ΔU=0.0008 [-0.0845,0.0942]
- P3 seed 20261003: ΔH=0.0485 [-0.0420,0.1406], ΔU=0.0491 [-0.0396,0.1342]
- P3 seed 20261004: ΔH=0.0051 [-0.0791,0.0955], ΔU=0.0153 [-0.0683,0.1030]
- P3 seed 20261005: ΔH=0.0129 [-0.0793,0.1038], ΔU=-0.0209 [-0.1344,0.0870]
- P3 seed 20261006: ΔH=0.0032 [-0.0918,0.0911], ΔU=0.0114 [-0.0656,0.0896]
- P4 seed 20261002: ΔH=0.0032 [-0.0839,0.0883], ΔU=0.0089 [-0.0789,0.0994]
- P4 seed 20261003: ΔH=0.0139 [-0.0773,0.1038], ΔU=0.0187 [-0.0636,0.0986]
- P4 seed 20261004: ΔH=0.0077 [-0.0802,0.0945], ΔU=0.0268 [-0.0579,0.1154]
- P4 seed 20261005: ΔH=0.0034 [-0.0884,0.0906], ΔU=0.0067 [-0.1199,0.1286]
- P4 seed 20261006: ΔH=-0.0095 [-0.0994,0.0726], ΔU=0.0005 [-0.0749,0.0823]
- P5 seed 20261002: ΔH=0.0337 [-0.0583,0.1269], ΔU=-0.0023 [-0.0879,0.0786]
- P5 seed 20261003: ΔH=-0.0283 [-0.1218,0.0648], ΔU=0.0499 [-0.0679,0.1798]
- P5 seed 20261004: ΔH=0.0315 [-0.0620,0.1233], ΔU=-0.0240 [-0.1075,0.0607]
- P5 seed 20261005: ΔH=0.0240 [-0.0726,0.1183], ΔU=-0.0447 [-0.1612,0.0699]
- P5 seed 20261006: ΔH=-0.0094 [-0.1085,0.0862], ΔU=-0.0009 [-0.0794,0.0793]
- P6 seed 20261002: ΔH=0.0076 [-0.0820,0.0986], ΔU=0.0340 [-0.0734,0.1469]
- P6 seed 20261003: ΔH=-0.0121 [-0.1051,0.0803], ΔU=0.0422 [-0.0544,0.1492]
- P6 seed 20261004: ΔH=0.0489 [-0.0411,0.1371], ΔU=0.0240 [-0.0743,0.1239]
- P6 seed 20261005: ΔH=0.0018 [-0.0893,0.0980], ΔU=-0.0479 [-0.1583,0.0639]
- P6 seed 20261006: ΔH=-0.0173 [-0.1111,0.0748], ΔU=-0.0005 [-0.0789,0.0734]

## 7. ABMIL matched-control result
Observation: P1 ΔH=-0.0026, ΔU=-0.0021, seed-positive H/U=2/3, pooled 25-cell direction=False。Evidence: Gate A=FAIL。Interpretation: ABMIL pooling is not confirmed against A0.

## 8. FiLM vs concat
Observation: P2 ΔH=0.0064, ΔU=0.0041, seed-positive H/U=3/3, pooled 25-cell direction=True。Evidence: Gate B=FAIL；至少一个 seed-level patient bootstrap CI 非负条件=不满足。Interpretation: FiLM advantage is not confirmed; a small-event variance explanation remains plausible.

## 9. FiLM vs matched residual MLP
Observation: P3 ΔH=0.0041, ΔU=0.0070, seed-positive H/U=4/4, pooled 25-cell direction=True。Evidence: Gate C=PASS。Interpretation: FiLM-specific inductive bias is supported.

## 10. Real six-signature vs shuffled six-signature
Observation: P4 ΔH=-0.0016, ΔU=0.0045, seed-positive H/U=2/4, pooled 25-cell direction=False。Evidence: Gate D=FAIL；F3/F6 active parameter counts were audited equal. Interpretation: biological grouping semantics are not confirmed; this is not causal biology proof.

## 11. F3 vs MCAT
Observation: P5 ΔH=-0.0051, ΔU=-0.0067, seed-positive H/U=3/2, pooled 25-cell direction=False。Evidence: Gate E context F3-vs-F0=FAIL。Interpretation: F3 is not stably better than MCAT.

## 12. F3 vs CommonOnly
Observation: P6 ΔH=-0.0118, ΔU=-0.0050, seed-positive H/U=1/2, pooled 25-cell direction=False。Evidence: architecture-level gate=FAIL。Interpretation: do not freeze F3 as the final backbone.

## 13. Variance diagnostics

variance_decomposition.csv reports per-seed means, between-seed variance, within-seed fold variance, and worst fold for all seven variants. The 65-event endpoint is a direct limitation on precision; this diagnostic is descriptive, not a tuning gate.

## 14. Mechanism interpretation

The only valid mechanism claims are those whose preregistered gates pass. A positive point estimate without seed replication or a bootstrap interval excluding zero is not treated as confirmation. F3/F6 differences isolate grouping under the specified fold-local mapping but do not establish biological causality.

## 15. Failure analysis

Potential small-event variance was tested through five fresh seeds, paired folds, paired OOF bootstrap and variance decomposition. Failed formal cells: none if all 175 completed; observed formal coverage=175/175. Any gate failure is retained as a negative result and is not repaired by post-hoc tuning.

## 16. Final decision

**Decision D**. Gate A=False; Gate B=False; Gate C=True; Gate D=False; F3 vs F0=False; F3 vs B0=False. This decision follows the prompt’s fixed hierarchy.
