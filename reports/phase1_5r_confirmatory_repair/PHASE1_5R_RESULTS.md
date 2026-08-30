# Phase 1.5R confirmatory benchmark repair results

## Decision boundary

**Decision A — SIMPLE_POOLING_SUFFICIENT**

Anchor passed; Full Mean is not weaker than the repaired advanced methods, Mean@1024 is not weaker than B1, and the strongest-vs-Full-Mean paired bootstrap has no two-metric stable gain.

This report follows the attached Phase1.5R protocol. It is a pathology-only, same-cohort confirmation on frozen UNI2-h features and the frozen Phase1R patient split; it does not start Phase2, tri-modal modeling, or morphology-level credit assignment.

## 1. Phase1R anchor gate

| Method | Historical Harrell | New Harrell | Δ | Historical Uno | New Uno | Gate |
|---|---:|---:|---:|---:|---:|---|
| B0_mean | 0.609752 | 0.609752 | 0.000000 | 0.592353 | 0.592353 | True |
| B1_ABMIL | 0.648733 | 0.648733 | 0.000000 | 0.605460 | 0.605460 | True |

Both anchors are re-read from the original Phase1R implementation/runtime OOF and agree with the historical report within ±0.02. The gate was completed before repaired baseline fitting.

## 2. Repaired input protocol

- Cohort: 487 patients, 65 DSS events, 494 slides, raw feature dimension 1536.
- Split: frozen Phase1R patient-level five-fold manifest `confirmatory_patient_manifest.csv`; no re-splitting.
- Main neural protocol: learned projection from 1536D, K=1024 stochastic train sampling, five deterministic evaluation repeats, slide-first then mean across slides.
- Mean baselines: Full Mean and Mean@1024; PCA/standardization/Cox fitting are train-only.
- Survival loss: exact full patient-level risk sets; no patch-level survival samples.
- Uno: censoring KM is fitted on outer-training patients and evaluated on the held-out fold; test-fold KM is not used.

## 3. Main repaired benchmark

| Method | Seeds | Harrell | Uno | ΔHarrell vs B1 |
|---|---:|---:|---:|---:|
| B2_PlainABMIL | 3 | 0.4075 | 0.4136 | -0.0045 |
| B1_GatedABMIL | 3 | 0.4121 | 0.4531 | 0.0000 |
| B7a_GatedABMIL_Cox | 3 | 0.4121 | 0.4531 | 0.0000 |
| B3_TransformerMIL_reimplementation | 3 | 0.4138 | 0.3894 | 0.0017 |
| B6_SpatialNeighborhoodMIL | 3 | 0.4333 | 0.4196 | 0.0212 |
| B7b_GatedABMIL_CoxRank | 3 | 0.4754 | 0.4558 | 0.0633 |
| B5_PANTHER_style_GMM | 3 | 0.5664 | 0.5583 | 0.1543 |
| B0_full | 3 | 0.6353 | 0.6206 | 0.2232 |
| B0_1024 | 3 | 0.6364 | 0.6256 | 0.2243 |

TransformerMIL is explicitly a generic reimplementation and is not labeled official TransMIL. SpatialNeighborhoodMIL is the paper-faithful coordinate-only equivalent and is not labeled EAGLE. B5_PANTHER_style_GMM uses train-only PCA64/GMM and is not labeled official PANTHER.

## 4. Token budget sensitivity

| Method | K | Harrell | Uno | Seeds |
|---|---:|---:|---:|---:|
| B1_GatedABMIL | 128 | 0.4350 | 0.4666 | 3 |
| B1_GatedABMIL | 256 | 0.4231 | 0.3967 | 3 |
| B1_GatedABMIL | 512 | 0.4598 | 0.4296 | 3 |
| B1_GatedABMIL | 1024 | 0.4121 | 0.4531 | 3 |
| Mean | 128 | 0.6369 | 0.6278 | 3 |
| Mean | 256 | 0.6274 | 0.6212 | 3 |
| Mean | 512 | 0.6290 | 0.6136 | 3 |
| Mean | 1024 | 0.6364 | 0.6256 | 3 |
| TransformerMIL_reimplementation | 128 | 0.4173 | 0.3913 | 3 |
| TransformerMIL_reimplementation | 256 | 0.4192 | 0.4032 | 3 |
| TransformerMIL_reimplementation | 512 | 0.4116 | 0.3918 | 3 |
| TransformerMIL_reimplementation | 1024 | 0.4138 | 0.3894 | 3 |

## 5. Projection sensitivity

| Projection | Feature dim | K | Harrell | Uno | Seeds |
|---|---:|---:|---:|---:|---:|
| random64 | 64 | 1024 | 0.4452 | 0.5120 | 3 |
| learned64 | 64 | 1024 | 0.4239 | 0.4133 | 3 |
| learned256 | 256 | 1024 | 0.4121 | 0.4531 | 3 |
| learned512 | 512 | 1024 | 0.4710 | 0.4673 | 3 |
| random64_old_Phase1.5 | 64 | 128 | 0.5498 | 0.5939 | 3 |

## 6. Required answers

1. **Anchor:** both Phase1R anchors pass.
2. **Phase1.5 performance loss:** the repaired tables quantify token and projection effects; no architecture conclusion is drawn from the old unequal budget.
3. **128→1024:** see token table and `sensitivity/token_budget_sensitivity.csv`.
4. **random64→learned256:** see projection table; the historical K=128 random64 row is kept separately.
5. **Repaired ABMIL vs Mean:** compare B1 with B0_full and B0_1024 using 5000 paired bootstrap replicates.
6. **Gated vs plain:** compare B1 and B2 in the main table and paired bootstrap.
7. **TransMIL:** only `TransformerMIL_reimplementation` is reported because official TransMIL was not vendored.
8. **MambaMIL:** dependency attempts and outcome are in `failures/mamba_dependency_attempts.json`.
9. **PANTHER:** the result is explicitly PANTHER-style and uses train-only PCA/GMM.
10. **Spatial:** the method is explicitly SpatialNeighborhoodMIL; no official EAGLE claim is made.
11. **Cox+Rank:** B7b uses only event_i with T_i<T_j comparable pairs.
12. **Bottleneck:** Anchor passed; Full Mean is not weaker than the repaired advanced methods, Mean@1024 is not weaker than B1, and the strongest-vs-Full-Mean paired bootstrap has no two-metric stable gain.
13. **New pathology model:** no new architecture is authorized by this report.
14. **Pan-cancer benchmark:** remains a possible follow-up only after independent protocol review.
15. **Tri-modal:** remains a possible follow-up; this stage does not execute it.

## 7. Integrity and limitations

- Failed/unavailable implementations remain in `PHASE1_5R_FAILURE_LOG.md`; they are not replaced by paper numbers.
- Only outer-training data are used for scaling, PCA, prototype fitting, hyperparameter selection and censoring KM.
- The cohort has 65 events; conflicting Harrell/Uno directions are reported as mixed evidence.
- Official MambaMIL/CLAM assets may remain unavailable; a queued dependency install is not treated as a successful reproduction.
- The long run resumed after a compute-only batch adjustment (patient_batch=32, slide_batch=32; earlier interrupted work used 8). Raw 1536D features, K values, split, seeds, folds and epoch grid were unchanged, but bitwise identity across batch settings is not claimed.
- Training/inference timing and model memory are recorded in `tables/compute_profile.csv` when measurable.

## 8. Data boundary

Raw TCGA files, H5/NPY/NPZ features, checkpoints, patient-level OOF/attention and caches remain local/ignored. GitHub contains only code, protocol, compact aggregate tables, audit summaries and figures.
