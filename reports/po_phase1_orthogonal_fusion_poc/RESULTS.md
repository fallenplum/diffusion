# Phase1 orthogonal fusion PoC

- RNA + Pathology only; no Clinical, CNA, LLM, knowledge graph, Q-Former or cross-attention Transformer.
- RNA and Pathology main effects were trained first. Interaction branches used frozen outer-train main-effect outputs; all grid selection was inner-only.
- Formal run: 105 outer-fold cells, 3 seeds, 5 folds, 0 failures.

## Genuine seed OOF summary

|Method|Harrell OOF mean +/- SD|Uno OOF mean +/- SD|
|---|---:|---:|
|RNA-Linear|0.6565 +/- 0.0135|0.6004 +/- 0.0177|
|Pathology-ABMIL|0.6794 +/- 0.0161|0.6972 +/- 0.0118|
|Additive-Main|0.6627 +/- 0.0199|0.6078 +/- 0.0256|
|Interaction-noOrth|0.6466 +/- 0.0016|0.6212 +/- 0.0215|
|Interaction-Orth|0.6525 +/- 0.0056|0.6287 +/- 0.0354|
|Interaction-Orth-noGate|0.5879 +/- 0.0487|0.5896 +/- 0.0236|
|Interaction-Orth-Gated|0.6536 +/- 0.0067|0.6294 +/- 0.0348|
|MGCT-PO|0.6894 +/- 0.0170|0.6644 +/- 0.0232|

## Mechanism diagnostics

- Full gated model alpha: mean 0.1096, range [0.0368, 0.1703]. It opened without saturating at 0 or 1.
- Mean absolute covariance sum, Orth: 0.2555; noOrth: 1750.3741. The empirical covariance penalty reduced redundancy in this diagnostic.
- Full gated model selected interaction dimensions/functions from the pre-registered inner-only grid; see `interaction_diagnostics.csv`.

## Paired bootstrap (Interaction-Orth-Gated minus comparator)

|Comparator|Harrell delta mean [95% CI]|Uno delta mean [95% CI]|
|---|---:|---:|
|RNA-Linear|-0.0030 [-0.0355, 0.0287]|0.0268 [-0.0115, 0.0684]|
|Pathology-ABMIL|-0.0232 [-0.0922, 0.0462]|-0.0633 [-0.1472, 0.0170]|
|Additive-Main|-0.0091 [-0.0404, 0.0203]|0.0194 [-0.0161, 0.0563]|
|MGCT-PO|-0.0336 [-0.1008, 0.0408]|-0.0326 [-0.1243, 0.0520]|

## Required questions

1. **RNA linear main effect:** retained as the fixed low-capacity main effect; its seed-OOF mean is Harrell 0.6565, Uno 0.6004.
2. **Pathology ABMIL main effect:** retained under the same outer split; seed-OOF mean is Harrell 0.6794, Uno 0.6972.
3. **Additive-Main:** it improves over RNA-Linear in both mean metrics but does not improve over Pathology-ABMIL in Harrell or Uno.
4. **Interaction gain:** Interaction-Orth-Gated is Harrell 0.6536, Uno 0.6294; it is below Additive-Main in Harrell (0.6627) and therefore does not establish non-degrading complementary gain.
5. **Orthogonality:** empirical covariance is substantially lower for Orth than noOrth, but this is a diagnostic penalty result, not a strict RKHS-ANOVA guarantee.
6. **Gate:** the learned gate opens at a moderate level and avoids saturation, but avoiding gate collapse did not prevent Harrell degradation.
7. **Seed stability:** the full gated interaction is positive versus Additive-Main on Uno in all three seed OOF comparisons, but negative on Harrell in two of three seeds; the paired bootstrap CIs include zero.
**8. Expansion decision:** the pre-registered decision is `MAIN_EFFECT_DEGRADATION`; do not expand this mechanism to Clinical + RNA + Pathology yet.

See `DECISION.md`, `validation.json`, `interaction_diagnostics.csv` and `paired_bootstrap.json` for the auditable details.
