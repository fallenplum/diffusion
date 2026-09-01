# Phase1.5R2 Functional Cox Repair Results

## Decision

**MIL_SIGNAL_RESTORED**

Repaired GatedABMIL exceeds Full Mean on pooled Harrell; the old simple-pooling conclusion is not retained.

This is the functional repair stage only. Phase2, tri-modal, pan-cancer and morphology-level risk attribution were not entered.

## 1. Functional gates

```json
{
  "cox_autograd_match": true,
  "cox_finite_difference_match": true,
  "cox_one_step_descent": true,
  "rank_autograd_match": true,
  "rank_finite_difference_match": true,
  "combined_gradient_match": true,
  "two_pass_equivalence": true,
  "status": true
}
```

The pre-repair coefficient had the opposite direction (legacy cosine is recorded in `unit_tests/cox_gradient_test.json`); corrected analytic, autograd, finite-difference, ranking, combined λ={0.05,0.2} and two-pass tests all passed before TCGA training.

## 2. Fresh Phase1R B1 anchor

| Method | Historical Harrell | Fresh Harrell | Abs Δ | Gate |
|---|---:|---:|---:|---|
| B1_Phase1R_Fresh | 0.648733 | 0.648733 | 0.000000 | True |

The fresh anchor directly calls `src/phase1_neural.py`'s original sampled risk-set implementation with seed 20260723, frozen Phase1R folds, raw 1536D, learned 256D projection and K=1024. Historical OOF was used only as a metric reference; the manifest records no historical OOF training read.

## 3. Repaired neural benchmark

| Method | Harrell | Uno | Δ vs Full Mean | Δ vs B1 |
|---|---:|---:|---:|---:|
| B5_PANTHER_style_GMM | 0.5664 | 0.5583 | -0.0689 | -0.0926 |
| B0_full | 0.6353 | 0.6206 | 0.0000 | -0.0237 |
| B0_1024 | 0.6364 | 0.6256 | 0.0011 | -0.0226 |
| B3_TransformerMIL_reimplementation | 0.6449 | 0.6258 | 0.0096 | -0.0141 |
| B2_PlainABMIL | 0.6584 | 0.6305 | 0.0231 | -0.0006 |
| B1_GatedABMIL | 0.6590 | 0.6526 | 0.0237 | 0.0000 |
| B6_SpatialNeighborhoodMIL | 0.6607 | 0.6477 | 0.0254 | 0.0017 |
| B7a_GatedABMIL_Cox | 0.6667 | 0.6468 | 0.0314 | 0.0077 |
| B7b_GatedABMIL_CoxRank | 0.6801 | 0.6502 | 0.0448 | 0.0211 |

The Full Mean, Mean@1024 and PANTHER-style GMM rows are explicitly reused from Phase1.5R because they are not affected by the neural Cox analytic-gradient bug. The old Phase1.5R bugged neural rows remain in `tables/phase1_5r_bugged_neural_metrics.csv` for provenance.

Risk-direction warnings: **8** fold/method records. They are retained as model-level diagnostics; a global inversion would be a hard integrity stop, and no warning is silently hidden or sign-flipped.

## 4. Token sensitivity

| Method | K | Harrell | Uno |
|---|---:|---:|---:|
| B1_GatedABMIL | 128 | 0.6665 | 0.6596 |
| B1_GatedABMIL | 256 | 0.6811 | 0.6624 |
| B1_GatedABMIL | 512 | 0.6701 | 0.6339 |
| B1_GatedABMIL | 1024 | 0.6590 | 0.6526 |
| Mean | 128 | 0.6369 | 0.6278 |
| Mean | 256 | 0.6274 | 0.6212 |
| Mean | 512 | 0.6290 | 0.6136 |
| Mean | 1024 | 0.6364 | 0.6256 |
| TransformerMIL_reimplementation | 128 | 0.6372 | 0.6196 |
| TransformerMIL_reimplementation | 256 | 0.6545 | 0.6339 |
| TransformerMIL_reimplementation | 512 | 0.6575 | 0.6392 |
| TransformerMIL_reimplementation | 1024 | 0.6449 | 0.6258 |

## 5. Projection sensitivity

| Projection | Dim | Harrell | Uno |
|---|---:|---:|---:|
| random64 | 64 | 0.4767 | 0.5061 |
| learned64 | 64 | 0.6759 | 0.6535 |
| learned256 | 256 | 0.6590 | 0.6526 |
| learned512 | 512 | 0.6730 | 0.6928 |

## 6. Statistical comparison and required scope answers

- Paired bootstrap uses 5000 replicates and outer-training censoring KM values held fixed per patient/fold; see `bootstrap/paired_bootstrap.csv`.
- PlainABMIL, TransformerMIL reimplementation, SpatialNeighborhoodMIL and Cox+Rank are all rerun with the corrected gradient.
- MambaMIL and CLAM retain `FAILED_TO_REPRODUCE`; dependency expansion was outside this functional repair.
- Any Full Mean advantage, fold/seed variability and Harrell/Uno conflict are reported rather than converted into an architectural claim.
- No morphology-level credit assignment, new pathology architecture, pan-cancer benchmark or tri-modal experiment is authorized by this stage.

## 7. Provenance and data boundary

- `Phase1R historical`, `Phase1.5`, `Phase1.5R bugged` and `Phase1.5R2 repaired` are kept distinct.
- Raw TCGA/H5/NPY/NPZ files, checkpoints, patient-level OOF and attention remain local and ignored by Git.
- The tracked report contains only compact aggregate tables, functional evidence, audit/provenance metadata and figures.
