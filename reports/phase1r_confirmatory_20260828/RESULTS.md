# Phase1R confirmatory split-replication

## Scope

This is a same-cohort split-replication, not an independent external validation.
The purpose was to test the pre-registered Phase P0_nocoords signal on a new patient-level outer split; no post-result tuning or seed hunting was performed.

## Data and split audit

- Cohort: TCGA only; 487 patients, 65 DSS events, 494 slides, feature dimension 1536.
- All selected bags and coordinate arrays were resolved to existing local artifacts; coordinates were taken from the audited real H5-derived arrays.
- Outer split: StratifiedKFold(n_splits=5, shuffle=True, random_state=20260828), patient-level, locked hash `68238e4be6e690cb96fe5eac4752202f716977a7ca503e611beb2bff1119d945`.
- All PCA/scaler fitting, SSL pretraining, and survival training were restricted to outer-train patients for each fold.

## Four-method primary results

| Method | OOF Harrell C | OOF Uno C | Fold Harrell mean ± SD | Fold Uno mean ± SD |
|---|---:|---:|---:|---:|
| B0_mean | 0.609752 | 0.592353 | 0.625999 ± 0.054942 | 0.576421 ± 0.128137 |
| B1_ABMIL | 0.648733 | 0.605460 | 0.676934 ± 0.072651 | 0.636316 ± 0.117025 |
| P0_nocoords | 0.607974 | 0.622525 | 0.616822 ± 0.110376 | 0.607855 ± 0.115712 |
| P0_coords | 0.638210 | 0.561995 | 0.644086 ± 0.095774 | 0.561263 ± 0.134621 |

## Per-fold Harrell C

| Method | Fold 1 Harrell | Fold 2 Harrell | Fold 3 Harrell | Fold 4 Harrell | Fold 5 Harrell |
|---|---:|---:|---:|---:|---:|
| B0_mean | 0.690955 | 0.619303 | 0.579730 | 0.672909 | 0.567100 |
| B1_ABMIL | 0.748744 | 0.557641 | 0.717568 | 0.685393 | 0.675325 |
| P0_nocoords | 0.734925 | 0.530831 | 0.694595 | 0.474407 | 0.649351 |
| P0_coords | 0.798995 | 0.554960 | 0.627027 | 0.579276 | 0.660173 |

Per-fold Uno C is available in `fold_metrics.csv`.

## Paired bootstrap (1000 replicates)

| Comparison | ΔHarrell | 95% CI | ΔUno | 95% CI | Valid Harrell / Uno |
|---|---:|---|---:|---|---:|
| B1_ABMIL vs B0_mean | 0.038980 | [-0.026609, 0.106523] | 0.013107 | [-0.054546, 0.085935] | 1000 / 1000 |
| P0_nocoords vs B0_mean | -0.001779 | [-0.081451, 0.080986] | 0.030172 | [-0.109233, 0.145062] | 1000 / 1000 |
| P0_coords vs B0_mean | 0.028457 | [-0.055083, 0.107740] | -0.030358 | [-0.137759, 0.083951] | 1000 / 1000 |

## Pre-registered gate

- P0_nocoords gate: **FAIL**; B1 gate: **PASS**.
- Decision: **FALLBACK_B1**; selected pathology branch: **B1_ABMIL**.
- P0_nocoords exceeds P0_coords on both OOF metrics: **False**.
- P0_coords was diagnostic only and did not participate in fallback selection.

## Limitations

This is not external validation; cohort-specific bias and limited DSS events remain. The neural survival implementation uses the locked nested risk-set approximation documented in Phase P1. Bootstrap intervals are uncertainty evidence, not a tuning criterion.

