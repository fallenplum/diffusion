# Phase1 orthogonal fusion PoC repair

- RNA + Pathology only; no Clinical, CNA or LLM inputs.
- Interaction selection uses a complete inner Stage-A/Stage-B pipeline: RNA and ABMIL are optimized on inner-train only, then candidates are scored once on inner-val.
- The repaired interaction grid has 4 candidates: d=8, Hadamard/Bilinear, rank=4 and lambda_perp in [0.01, 0.1].
- Additive and interaction training use 4-fold outer-train cross-fitted scalar main risks. Pathology embeddings come from one final outer-train ABMIL fit, so independent latent coordinate systems are never concatenated.
- Outer-test labels are withheld until the configuration is locked.

## Seed OOF summary

|Method|Harrell OOF mean|Uno OOF mean|
|---|---:|---:|
|RNA-Linear|0.6565|0.6004|
|Pathology-ABMIL|0.6772|0.6997|
|Additive-Main|0.6623|0.6502|
|Interaction-noOrth|0.6429|0.5888|
|Interaction-Orth|0.6188|0.6109|
|Interaction-Orth-noGate|0.6277|0.6111|
|Interaction-Orth-Gated|0.6157|0.6144|
|MGCT-PO|0.6894|0.6644|

## Required questions

1. RNA linear main effect is evaluated in `main_effect_metrics.csv`; its OOF mean is 0.6565 Harrell.
2. Pathology ABMIL is evaluated under the same fixed outer split.
3. Additive-Main is the protected two-main-effect comparator.
4. Interaction gains are assessed against Additive-Main using paired OOF and bootstrap files.
5. Orthogonality is measured by empirical Cov(h,r_R) and Cov(h,r_P); it is not a strict ANOVA guarantee.
6. Gate behavior is recorded per cell in `interaction_diagnostics.csv`.
7. Seed-wise opening stability is recorded by `gate_alpha` and `seed_oof_metrics.csv`.
8. Expansion to Clinical + RNA + Pathology is not authorized by this PoC unless the decision is supported; current decision is **MAIN_EFFECT_DEGRADATION**.

See `DECISION.md` for the fixed decision rule.
