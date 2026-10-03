# Phase M9 Conditional Information Audit

Decision: **C. CONDITIONAL_SIGNAL_NOT_SUPPORTED**

## Direct answers to the 12 required questions
1. Clinical envelope Harrell means: {'C-full': 0.7662655627657975, 'C-min': 0.5505812344274129, 'C-stage': 0.6890849430350577}.
2. Stage/TNM contribution is measured by C-min→C-stage in clinical_envelope.csv; C-stage→C-full is reported separately.
3. Pathology marginal signal is reported in marginal_signal.csv; P-MEAN-LIN/P-ABMIL were evaluated over fixed folds and seeds.
4. Omics marginal signal is reported for PCA20/PCA40/Hallmark/ENET.
5. P+O marginal signal is reported for additive and concatenated low-capacity models.
6. Given frozen C-full, RP-L0 OOF mean delta H=-0.005862, U=-0.010355; positive seeds H=2/5, U=2/5.
7. Given frozen C-full, RO-L0 OOF mean delta H=-0.001266, U=-0.012896; positive seeds H=1/5, U=2/5.
8. Given frozen C-full, RPO-L0 OOF mean delta H=+0.014543, U=-0.018736; positive seeds H=3/5, U=1/5. No cross-attention or new fusion architecture was introduced.
9. L0→L1→L2 capacity points are in capacity_frontier.csv; no post-test best point was selected. RP-L1 uses the fold-local Gated ABMIL embedding followed by a linear residual head.
10. Clinical completeness deltas are in clinical_completeness_curve.csv and are interpreted as predictive redundancy only.
11. Conditional-neighborhood permutation mean drops: {'RO-L0': {'ordinary_harrell_drop': 0.0031395568745326407, 'conditional_neighborhood_harrell_drop': -0.013776008483205976}, 'RP-L0': {'ordinary_harrell_drop': 0.06016130643047, 'conditional_neighborhood_harrell_drop': 0.04742055559178828}, 'RPO-L0': {'ordinary_harrell_drop': 0.08678582662601983, 'conditional_neighborhood_harrell_drop': 0.051498247346711594}}; neighborhoods use encoded Clinical distances with same-stage preference and k=10 donors within each outer-test fold.
12. The primary evidence favors an information-limited interpretation for this cohort/endpoint; 65 events still leave substantial uncertainty. The capacity frontier, clinical completeness, redundancy, and historical B0 context are reported in the corresponding tables. Historical B0 is context only and is not a decision gate.

## Protocol boundaries

No DSS labels were used for feature selection; DSS was used only as the declared endpoint. No patient IDs, OOF arrays, H5/NPY/NPZ, checkpoints or raw omics were written to the repository. No survival extension was run.
