# Phase 1.5 reproducibility

See `PHASE1_5_REPRODUCIBILITY.json`, `configs/hyperparameter_registry.yaml`, and `splits/split_manifest.json`. All runtime dependencies, seeds, token cap, projection seed, folds, and full-risk-set Cox rules are persisted. Paired bootstrap uses fixed full-cohort KM censoring weights for the resampled comparisons; this is recorded explicitly.
