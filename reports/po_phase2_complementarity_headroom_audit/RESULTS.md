# Phase2 RNA–Pathology complementarity headroom audit

- Fixed cohort: 512 patients, 65 DSS events, fixed five outer folds, three seeds.
- All Phase1 prediction files were reused from the locked repair cache; scalar cross-fit risks were reconstructed only because the required private cache was not persisted.
- The pairwise oracle is a non-realizable diagnostic upper bound, not a patient-level model.

## Key results

- Mean Pearson correlation: 0.1822; mean Spearman correlation: 0.1931.
- Mean RNA rescue fraction among Pathology-wrong pairs: 0.5900 (IPCW 0.5315); reverse Pathology rescue among RNA-wrong pairs: 0.6144 (IPCW 0.6488).
- Mean pairwise oracle C-index: Harrell 0.8676 vs Pathology 0.6772; Uno 0.8596 vs Pathology 0.6997.
- Mean pairwise oracle headroom: Harrell 0.1904; Uno 0.1598 (both exceed the pre-registered 0.02 engineering threshold).
- D3 seed OOF deltas vs Pathology: Harrell [0.0024, 0.0146, -0.0025]; Uno [0.0008, 0.0009, -0.0149].
- Conditional permutation mean drop: Harrell 0.0111; Uno 0.0019; gamma_R-positive outer-fold fraction mean 0.867.
- Decision: **HIGH_HEADROOM_LOW_RECOVERABILITY**.

## Direct answers required by the protocol

1. **Risk dependence:** Pearson mean 0.1822; Spearman mean 0.1931.
2. **High-risk overlap:** 10%: Jaccard 0.0911, overlap coefficient 0.1667, overlap count 8.7; 20%: Jaccard 0.1732, overlap coefficient 0.2945, overlap count 30.3; 30%: Jaccard 0.2571, overlap coefficient 0.4091, overlap count 63.0.
3. **RNA rescue:** among Pathology-wrong comparable pairs, RNA rescues 0.5900 unweighted and 0.5315 IPCW-weighted.
4. **Reverse rescue:** among RNA-wrong comparable pairs, Pathology rescues 0.6144 unweighted and 0.6488 IPCW-weighted.
5. **Pairwise oracle:** Harrell 0.8676; Uno 0.8596; these are non-realizable pairwise diagnostics.
6. **Oracle headroom vs Pathology-only:** Harrell 0.1904; Uno 0.1598.
7. **Threshold:** the headroom is at least 0.02 on both metrics, so it is algorithmically large enough to justify an audit of representation-level recovery.
8. **Scalar recovery:** D3 mean delta is Harrell 0.0048 and Uno -0.0044; the three seed deltas are not uniformly positive. A representative paired-bootstrap interval is Harrell [-0.0339, 0.0377] and Uno [-0.0121, 0.0158] for seed 20260904; all seed rows are in `bootstrap_summary.csv`.
9. **Conditional permutation:** mean drop Harrell 0.0111, Uno 0.0019; paired-bootstrap means across seeds are [0.01, 0.0207, 0.002] (Harrell) and [0.0049, 0.0169, -0.0174] (Uno), with non-degenerate intervals in `bootstrap_summary.csv`.
10. **Decision category:** **HIGH_HEADROOM_LOW_RECOVERABILITY**.
11. **Next-stage decision:** yes, representation-level RNA–Pathology fusion remains justified by the large non-realizable headroom, but no claim is made that a complex architecture will recover it.
12. **Specific next problem:** recover patient-level RNA-only ranking information that scalar risk correction cannot express, using strictly outer-train-fitted, fold-aligned representation interactions while preserving the fixed 512/65 cohort, splits, and endpoint.

## Research decision

The data support continuing representation-level RNA–Pathology fusion, with the specific problem of recovering pairwise RNA-only ranking information that scalar correction cannot access.
