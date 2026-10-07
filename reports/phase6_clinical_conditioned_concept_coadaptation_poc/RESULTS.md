# Phase6 clinical-conditioned concept co-adaptation POC

## Scope

This is an outcome-blind representation feasibility experiment on the locally available 487-patient UNI2-h cohort. It uses patient-held-out evaluation and does not reproduce CONCH, raw-SVS inference, CTF, or CTF_2 survival evaluation.

## Results

Held-out static patient-level alignment gap: **0.2107 ± 0.0205**; dynamic Clinical-conditioned gap: **0.2087 ± 0.0185**.
Held-out static top-1 retrieval P→O/O→P: **0.062/0.062**; dynamic: **0.062/0.058**.
Held-out static concept alignment gap: **0.1964 ± 0.0461**; dynamic: **0.2868 ± 0.0204**.
Held-out static concept retrieval P→O/O→P: **0.041/0.027**; dynamic: **0.079/0.072**.
Dynamic concept token RMS movement from fixed token: **0.0892 ± 0.0024**; static control is **0.0000**.

## Interpretation

The corrected loss trains both modality alignment and concept-score alignment. Patient-specific token movement is therefore connected to the optimization objective. A larger held-out concept alignment gap or retrieval score would support useful complementary alignment; token movement alone is insufficient. These are representation signals, not DSS evidence. The next formal gate requires the 512-patient server cohort, fixed outer folds, and a label-blind CONCH/SVS preflight or an explicitly declared UNI2 surrogate analysis.

## Validation

Validation status: **PASS**.
