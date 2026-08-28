# Phase 1.5 benchmark results

## Decision boundary

**Decision A — SIMPLE_MIL_SUFFICIENT**

No stable large improvement over ABMIL is established by the available pooled/fold evidence; strongest=B0_mean, Δ vs ABMIL=0.1126, bootstrap CI crosses 0=False, Cox+rank CI crosses 0=True.

This is a pathology-only audit on frozen UNI2-h features and a frozen patient-level Phase1R split. It is not evidence for a new architecture and it does not start Phase 2.

## Answers to the ten required questions

1. **Cohort:** 487 patients and 65 DSS events.
2. **WSI per patient:** mean 1.0144, median 1.0; 6 patients have multiple slides.
3. **Patches per WSI:** see `data_audit/feature_inventory.csv`; min 328, median 4826, max 31521.
4. **Mean strength:** see Table 1 below; B0 is evaluated with OOF Harrell and Uno.
5. **ABMIL gain:** the paired bootstrap and fold-win columns are the required stability evidence; a single lucky seed is not used.
6. **CLAM/TransMIL/MambaMIL:** CLAM is explicitly failed because no validated survival adaptation is available; TransMIL is a recorded survival adaptation; MambaMIL is explicitly failed because its official dependency stack is unavailable.
7. **Prototype:** B6 uses train-only GMM fitting and a pre-registered K search; no full-data clustering is used.
8. **Spatial coordinates:** B8 is a paper-faithful spatial equivalent, not an official EAGLE reproduction; no semantic tissue labels are assigned to coordinates.
9. **Cox+ranking:** B9a is an alias of ABMIL Cox-only; B9b uses only comparable survival pairs (event i with T_i<T_j) and inner validation for λ.
10. **Bottleneck:** the evidence is summarized by the decision above; uncertainty from only 65 events is retained.

## Table 1 — Main benchmark

| Method | Seeds | OOF Harrell | OOF Uno | Fold Harrell mean ± std | Fold wins vs ABMIL |
|---|---:|---:|---:|---:|---:|
| B2_GatedABMIL | 3 | 0.5056 | 0.5574 | 0.5532 ± 0.1000 | 8 |
| B6_PANTHER_GMM | 3 | 0.5193 | 0.5617 | 0.5228 ± 0.1039 | 5 |
| B4_TransMIL | 3 | 0.5262 | 0.5666 | 0.5304 ± 0.0997 | 5 |
| B1_ABMIL | 3 | 0.5498 | 0.5939 | 0.5674 ± 0.1153 | 0 |
| B9a_ABMIL_Cox | 3 | 0.5498 | 0.5939 | 0.5674 ± 0.1153 | 0 |
| B8_EAGLE_spatial | 3 | 0.5620 | 0.6007 | 0.5661 ± 0.1033 | 9 |
| B9b_ABMIL_CoxRank | 3 | 0.5761 | 0.5856 | 0.5787 ± 0.1370 | 7 |
| B0b_mean_std | 3 | 0.6402 | 0.6329 | 0.6421 ± 0.0893 | 10 |
| B0_mean | 3 | 0.6624 | 0.6471 | 0.6516 ± 0.0775 | 10 |

## Integrity boundaries

- No clinical/RNA/TNM/stage input was used.
- Censored cases remain survival observations, never a binary negative class.
- Every neural update uses a full patient-level Cox risk set.
- Test folds are not used for tuning, early stopping, PCA, normalization, GMM, or prototypes.
- Patient-level OOF files, attention files, checkpoints, raw features, and coordinates remain local/ignored; they are not copied to GitHub.
- Failed methods retained in the ledger: `B3_CLAM_SB_survival`, `B5_MambaMIL`, `B7_PAMoE`.

## Limitations

The local feature-only benchmark has 65 observed events, so confidence intervals crossing zero are evidence-insufficient. The token cap is a deterministic order-preserving cap shared across neural methods and is not a claim that all original long-sequence implementations were fully reproduced. Peak memory and inference time were re-measured from saved checkpoints; per-job training wall time was not persisted by the initial run and is therefore left blank in `compute_profile.csv`.
