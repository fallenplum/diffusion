# Phase K0 CRC Knowledge Concept Fusion Audit

## Protocol

- Fixed DSS cohort: N=512, events=65, five outer folds, seeds 20261012-20261016.
- Clinical risk is the exact frozen M8-P0 C0-Full outer-test anchor; K0 adds only a residual.
- Concept bank, gene mapping, and text metadata were frozen before training; no DSS labels selected concepts.
- Formal cells: 125/125 (five variants x five seeds x five folds).

## Required summary

1. Branch: `codex/phase-k0-crc-knowledge-concept-fusion`
2. Commit SHA: `b3b8342dffb11658e4a9f6c5d03b0ca2bd733d30`
3. 125/125 cells complete: **YES**
4. Concept bank SHA256: `830f7d3574b95aada65fcd23cec7bb3b0aeae34641539726654aac937b6c8a40`

| Variant | Harrell C | Uno C |
|---|---:|---:|
| S0 | 0.771574 | 0.785990 |
| K-O | 0.771939 | 0.786197 |
| K-P | 0.771931 | 0.786117 |
| K-ADD | 0.772116 | 0.786125 |
| K-MSG | 0.771520 | 0.785986 |
| K-SHUFFLE | 0.771469 | 0.785919 |
| SO | 0.771979 | 0.786089 |
| SP | 0.774909 | 0.787610 |
| SPO | 0.776268 | 0.787799 |
| B0 | 0.778339 | 0.783535 |
K-O vs SO: ΔH=-0.000040, ΔU=+0.000108; positive seeds H=2/5, U=3/5
K-P vs SP: ΔH=-0.002978, ΔU=-0.001493; positive seeds H=1/5, U=2/5
K-ADD vs SPO: ΔH=-0.004152, ΔU=-0.001673; positive seeds H=0/5, U=1/5
K-MSG vs K-ADD: ΔH=-0.000595, ΔU=-0.000139; positive seeds H=1/5, U=0/5
K-MSG vs K-SHUFFLE: ΔH=+0.000051, ΔU=+0.000068; positive seeds H=1/5, U=1/5
K-MSG vs S0: ΔH=-0.000054, ΔU=-0.000004; positive seeds H=3/5, U=2/5
K-MSG vs B0: ΔH=-0.006819, ΔU=+0.002452; positive seeds H=2/5, U=2/5

## Bootstrap

All seven required comparisons used 2,000 patient-level paired bootstrap replicates per seed. Exact 95% intervals are in `paired_bootstrap.csv`; intervals crossing zero are retained as uncertainty.

## Variance

Between-seed and within-seed fold variances are in `variance_decomposition.csv`.

## Concept diagnostics

Top K-MSG held-out absolute risk-contribution concepts: K3 metabolic_phenotype, K5 proliferation_cell_cycle, K1 immune_inflamed_interferon

## Gates

- A_omics_recovery: **FAIL** (ΔH=-0.000040, ΔU=+0.000108; positive seeds H/U=2/5, 3/5)
- B_concept_semantics: **FAIL** (ΔH=+0.000051, ΔU=+0.000068; positive seeds H/U=1/5, 1/5)
- C_message_complementarity: **FAIL** (ΔH=-0.000595, ΔU=-0.000139; positive seeds H/U=1/5, 0/5)
- D_strong_anchor_increment: **FAIL** (ΔH=-0.000054, ΔU=-0.000004; positive seeds H/U=3/5, 2/5)

## Final decision

**Decision D**

KNOWLEDGE_PRIOR_NOT_SUPPORTED

## Audit status

- Strong anchor identity: PASS
- Nested Clinical cross-fit: PASS
- Cox risk-set: PASS
- Zero initialization and parameter stop rule: PASS
- K-MSG/K-SHUFFLE initialization identity: PASS

## Output directory

/root/diffusion_m3_verified_9f859ef/reports/phase_k0_crc_knowledge_concept_fusion
