# Phase M2R2 Omics Fidelity

## Decision

**Phase M2R2 COMPLETE**

- Cohort: 512 patients, 65 DSS events; committed 5-fold split reused.
- All B0-B6 were rerun after the RNA preprocessing correction; no M2R rows are reused.
- Completion requires 95/95 cell evidence, five folds per method, 3 seeds for B1-B6, and 310 semantic overlap rows.

## Formal ranking

| Rank | Method | Harrell C seed mean +/- std | Uno C seed mean +/- std | Harrell C fold mean +/- std | Worst Harrell fold | GPU GB | Train seconds | Fidelity |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | MCAT | 0.7473 +/- 0.0352 | 0.7939 +/- 0.0458 | 0.7776 +/- 0.0692 | 0.6747 | 5.30 | 734.0 | PAPER_FAITHFUL_REIMPLEMENTATION |
| 2 | CAMR | 0.7159 +/- 0.0145 | 0.7453 +/- 0.0084 | 0.7528 +/- 0.0849 | 0.6429 | 5.28 | 528.7 | PAPER_FAITHFUL_REIMPLEMENTATION |
| 3 | EarlyFusion | 0.6986 +/- 0.0000 | 0.6857 +/- 0.0000 | 0.7056 +/- 0.1044 | 0.5885 | 0.02 | 28.5 | PAPER_FAITHFUL_REIMPLEMENTATION |
| 4 | MOTCat | 0.6580 +/- 0.0519 | 0.7097 +/- 0.0245 | 0.6928 +/- 0.1071 | 0.4962 | 5.28 | 497.1 | PAPER_FAITHFUL_REIMPLEMENTATION |
| 5 | TransSurv | 0.6550 +/- 0.0680 | 0.6792 +/- 0.0336 | 0.6780 +/- 0.0812 | 0.5912 | 5.47 | 478.1 | PAPER_FAITHFUL_REIMPLEMENTATION |
| 6 | MMPTrans-Hallmark | 0.6515 +/- 0.0323 | 0.6781 +/- 0.0136 | 0.7020 +/- 0.0791 | 0.5899 | 3.00 | 715.5 | PAPER_FAITHFUL_REIMPLEMENTATION |
| 7 | MGCT | 0.6376 +/- 0.0333 | 0.6743 +/- 0.0431 | 0.6779 +/- 0.0780 | 0.5816 | 5.59 | 594.9 | PAPER_FAITHFUL_REIMPLEMENTATION |

## Change from M2R

| Method | M2R Harrell C | M2R2 Harrell C | Delta | M2R provenance |
|---|---:|---:|---:|---|
| MCAT | 0.7433 | 0.7473 | 0.0040 | reused only for comparison |
| CAMR | 0.7109 | 0.7159 | 0.0050 | reused only for comparison |
| EarlyFusion | 0.7113 | 0.6986 | -0.0127 | reused only for comparison |
| MOTCat | 0.6528 | 0.6580 | 0.0052 | reused only for comparison |
| TransSurv | 0.6638 | 0.6550 | -0.0088 | reused only for comparison |
| MMPTrans-Hallmark | 0.6399 | 0.6515 | 0.0116 | reused only for comparison |
| MGCT | 0.6614 | 0.6376 | -0.0238 | reused only for comparison |

## Fidelity status

- B0 EarlyFusion: PAPER_FAITHFUL_REIMPLEMENTATION; source=fixed protocol baseline; provenance=rerun_M2R2.
- B1 TransSurv: PAPER_FAITHFUL_REIMPLEMENTATION; source=TransSurv paper-faithful reimplementation adapted to UNI2-h; provenance=rerun_M2R2.
- B2 MGCT: PAPER_FAITHFUL_REIMPLEMENTATION; source=https://github.com/lmxmercy/MGCT; provenance=rerun_M2R2.
- B3 CAMR: PAPER_FAITHFUL_REIMPLEMENTATION; source=https://github.com/wxq-ustc/CAMR; provenance=rerun_M2R2.
- B4 MMPTrans-Hallmark: PAPER_FAITHFUL_REIMPLEMENTATION; source=https://github.com/mahmoodlab/MMP; provenance=rerun_M2R2.
- B5 MCAT: PAPER_FAITHFUL_REIMPLEMENTATION; source=https://github.com/mahmoodlab/MCAT; provenance=rerun_M2R2.
- B6 MOTCat: PAPER_FAITHFUL_REIMPLEMENTATION; source=https://github.com/Innse/MOTCat; provenance=rerun_M2R2.
