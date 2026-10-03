# Phase K1R — Text-to-UNI2 Grounding

## Protocol gate

- Fixed TCGA COAD/READ cohort: 512 patients, 65 DSS events, five committed outer folds.
- Fixed neural seeds: 20260904, 20260905, 20260906; four two-concept hold-out rotations covering K1–K8.
- Stage 1 used no DSS time/event values for fitting, selection, or the grounding decision.
- Raw WSI/SVS and the CONCH image encoder were not used. The exact existing CONCH text embedding artifact was reused and hashed in `run_manifest.json`.
- Path input was the existing UNI2-h deterministic 1024-patch feature cache.

## Required grounding findings

1. **Grounding: FAIL.** The frozen rule required a positive real-text minus shuffled-text alignment in at least 75% of unseen cells and a positive mean. Observed: 120 cells, positive fraction **0.416667**, mean delta **-0.000092936**.
2. Unseen concept alignment: real text mean **0.997102**, shuffled text mean **0.997195**, mean difference **-0.000092936**. The near-zero difference means the hold-out adapters did not use the intended text semantics.
3. Attention entropy: real **0.780179** and shuffled **0.780179**, with 1.0 as the uniform reference. Top-10% patch overlap: real **0.463235**, shuffled **0.463235**.
4. P/O alignment: unseen real text to real gene teacher **0.997102**; real text to shuffled gene teacher **0.997351**. Seen concept alignment was **0.997370**, and mean patient score variance was **0.611083**. The high alignment that remains after text and gene perturbation is evidence that the pathology score is tracking the omics teacher without concept-specific text grounding.

## Stop decision

Decision: **D. TEXT_TO_UNI2_GROUNDING_FAIL**. Stage 2 survival variants were not run. The four survival CSVs are explicit `NOT_RUN_GROUNDING_FAIL` placeholders; no survival metric was fabricated.

## Files

All required audit tables, the frozen decision rule, source hashes, and the run manifest are in this directory. `grounding_fold_audit.csv` records fold-local gene mapping and standardization provenance.
