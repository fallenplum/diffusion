# Phase M8-P1R Repaired Strong Anchor Residual Confirmation

## Protocol

- Fixed DSS cohort: N=512, events=65, five outer folds, seeds 20261012-20261016.
- S0 is the frozen M8-P0 C0-Full outer-test prediction; SP/SO/SPO use additive zero-initialized residual heads.
- 75/75 residual cells were trained; S0 was reused without retraining.

## Required summary

1. Branch: `codex/phase-m8p1r-strong-anchor-residual-repair`
2. Source execution SHA: `59d558c45b6e9cc474838d5a7cfd547ea3a7cc69`
3. 75/75 residual cells complete: **YES** (100 total rows including 25 frozen S0 cells)
4. StrongAnchor == C0 audit: **PASS**, max abs diff=0

## Repair of the two reported protocol defects

The previous M8-P1 result is invalid for confirmatory interpretation. Its residual test path used a refit `test_offset` rather than the exact frozen C0-Full outer-test anchor, and its cross-fit epoch selection used held-out labels. This repaired run removes both defects: every residual test prediction uses the exact frozen C0-Full outer-test offset, while epoch selection and preprocessing fit are nested entirely inside each cross-fit training partition.

The repaired result supersedes the earlier M8-P1 decision.
S0 H/U: 0.771574 / 0.785990
SP H/U: 0.774909 / 0.787610
SO H/U: 0.771979 / 0.786089
SPO H/U: 0.776268 / 0.787799
SP delta vs S0: H=+0.003335, U=+0.001620; positive seeds H=4/5, U=3/5; **STRONG_ANCHOR_INCREMENT_NOT_CONFIRMED**
SO delta vs S0: H=+0.000404, U=+0.000098; positive seeds H=4/5, U=2/5; **STRONG_ANCHOR_INCREMENT_NOT_CONFIRMED**
SPO delta vs S0: H=+0.004693, U=+0.001808; positive seeds H=5/5, U=4/5; **STRONG_ANCHOR_INCREMENT_DIRECTIONAL**

## Bootstrap

Each SP/SO/SPO vs S0 comparison used 2,000 patient-level paired bootstrap replicates per seed. Exact CIs are in `paired_bootstrap.csv`; the decision uses the prompt's conservative CI gate.

SPO vs B0: H=-0.002072, U=+0.004264 (context only).

## Variance

Between-seed and within-seed fold variances for S0/SP/SO/SPO are in `variance_decomposition.csv`.

## Final decision

**Decision B: STRONG_ANCHOR_INCREMENT_DIRECTIONAL**.

If A/B, the next permitted step is low-capacity shrinkage residual modeling. If C, stop the Clinical-residual line and retain C0/B0.

## Audit status

- strong_anchor_identity_audit.json: PASS
- clinical_crossfit_audit.csv: PASS
- cox_riskset_audit.json: PASS
- zero_init_audit.json: PASS

## Output directory

reports/phase_m8p1r_strong_anchor_residual_repair
