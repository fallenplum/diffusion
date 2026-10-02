# Phase M8-P1 Strong Anchor Residual Confirmation

## Protocol

- Fixed DSS cohort: N=512, events=65, five outer folds, seeds 20261012-20261016.
- S0 is the frozen M8-P0 C0-Full outer-test prediction; SP/SO/SPO use additive zero-initialized residual heads.
- 75/75 residual cells were trained; S0 was reused without retraining.

## Required summary

1. Branch: `codex/phase-m8p1-strong-anchor-residual-confirmation`
2. Commit SHA: `93ec9391f2787c92527a8ce92d6fef604b55219d`
3. 75/75 residual cells complete: **YES** (80 total rows including 25 frozen S0 cells)
4. StrongAnchor == C0 audit: **PASS**, max abs diff=0
S0 H/U: 0.771574 / 0.785990
SP H/U: 0.751957 / 0.766710
SO H/U: 0.743356 / 0.755836
SPO H/U: 0.752670 / 0.769913
SP delta vs S0: H=-0.019618, U=-0.019281; positive seeds H=3/5, U=3/5; **STRONG_ANCHOR_INCREMENT_NOT_CONFIRMED**
SO delta vs S0: H=-0.028219, U=-0.030154; positive seeds H=2/5, U=2/5; **STRONG_ANCHOR_INCREMENT_NOT_CONFIRMED**
SPO delta vs S0: H=-0.018904, U=-0.016077; positive seeds H=3/5, U=3/5; **STRONG_ANCHOR_INCREMENT_NOT_CONFIRMED**

## Bootstrap

Each SP/SO/SPO vs S0 comparison used 2,000 patient-level paired bootstrap replicates per seed. Exact CIs are in `paired_bootstrap.csv`; the decision uses the prompt's conservative CI gate.

SPO vs B0: H=-0.025669, U=-0.013622 (context only).

## Variance

Between-seed and within-seed fold variances for S0/SP/SO/SPO are in `variance_decomposition.csv`.

## Final decision

**Decision C: STRONG_ANCHOR_INCREMENT_NOT_CONFIRMED**.

If A/B, the next permitted step is low-capacity shrinkage residual modeling. If C, stop the Clinical-residual line and retain C0/B0.

## Audit status

- strong_anchor_identity_audit.json: PASS
- clinical_crossfit_audit.csv: PASS
- cox_riskset_audit.json: PASS
- zero_init_audit.json: PASS

## Output directory

reports/phase_m8p1_strong_anchor_residual_confirmation
