# Phase M8-P0 Results

## 1. Executive conclusion

The locked M8-P0 audit completed 150/150 cells. Canonical incomplete DSS-labelled patients add **35 patients and 4 observed events**. The corrected OOF audit does not confirm a conditional residual signal. Canonical incomplete cases add four observed DSS events, so the missing-aware question remains a cohort-coverage problem rather than evidence for a new fusion architecture.

## 2. Protocol integrity

- DSS complete-case: 512 patients / 65 events; fixed five outer folds.
- Seeds: 20261012鈥?0261016; variants: C0-Full, C0-NoStage, B0, RP, RO, RPO.
- Clinical anchors were cross-fitted per outer fold and shared across residual variants.

## 3. Full COAD/READ modality-availability census

Canonical manifest: 626 patients. Current runtime lacks staged H5 files for 79 patients; these are runtime staging gaps, not canonical source missingness. Aggregate pattern counts: DSS C+G: N=31, events=3; DSS C+P: N=4, events=1; DSS C+P+G: N=512, events=65; PFS C+G: N=33, events=7; PFS C+P: N=4, events=2; PFS C+P+G: N=529, events=135.

## 4. DSS complete-case vs incomplete-case event counts

Outcome-labelled DSS: 547 / 69; incomplete DSS-labelled: 35 / 4.

## 5. PFS availability/event census

Outcome-labelled PFS: 566 / 144; incomplete: 37 / 9.

## 6. Clinical dominance audit

C0-Full mean 螖 over C0-NoStage is H=0.211964, U=0.281608; seed directions are 5/5 for H and 5/5 for U. The pattern is consistent with stage/TNM being a major component of Clinical prognostic signal, but the bootstrap intervals are not uniformly separated from zero.

## 7. Clinical cross-fit integrity

All outer-train patients were covered once by held-out Clinical cross-fit predictions, no self-training rows were observed, and outer-test patients were excluded from anchor fitting.

## 8. Pathology incremental signal

RP vs C0-Full: 螖H=-0.019618, 螖U=-0.019281; positive seeds H=3/5, U=3/5; bootstrap 95% CI by seed: 20261012: H[-0.051467,0.054403], U[-0.126091,0.091565]; 20261013: H[-0.118907,-0.017719], U[-0.134552,-0.013423]; 20261014: H[-0.124187,-0.026233], U[-0.144912,-0.013514]; 20261015: H[-0.015270,0.070289], U[-0.006477,0.105758]; 20261016: H[-0.041823,0.129543], U[-0.132171,0.046027]; **INCREMENTAL_SIGNAL_NOT_CONFIRMED**.

## 9. Omics incremental signal

RO vs C0-Full: 螖H=-0.028072, 螖U=-0.030080; positive seeds H=2/5, U=2/5; bootstrap 95% CI by seed: 20261012: H[-0.051294,0.051984], U[-0.132635,0.076225]; 20261013: H[-0.127632,-0.019160], U[-0.135017,-0.013867]; 20261014: H[-0.133185,-0.028993], U[-0.168384,-0.020616]; 20261015: H[-0.020163,0.067530], U[-0.007249,0.102998]; 20261016: H[-0.053044,0.115146], U[-0.150619,0.024082]; **INCREMENTAL_SIGNAL_NOT_CONFIRMED**.

## 10. Additive multimodal incremental signal

RPO vs C0-Full: 螖H=-0.015518, 螖U=-0.017066; positive seeds H=3/5, U=3/5; bootstrap 95% CI by seed: 20261012: H[-0.049835,0.057326], U[-0.127901,0.088973]; 20261013: H[-0.104646,0.000824], U[-0.110157,0.004003]; 20261014: H[-0.123234,-0.025694], U[-0.154953,-0.013040]; 20261015: H[-0.021922,0.072036], U[-0.009289,0.106831]; 20261016: H[-0.042534,0.132730], U[-0.131421,0.042757]; **INCREMENTAL_SIGNAL_NOT_CONFIRMED**.

## 11. Residual formulation vs CommonOnly

RPO mean H/U=0.756057/0.768924; B0 mean H/U=0.778339/0.783535. 螖H=-0.022283, 螖U=-0.014611; gate: **RESIDUAL_FORMULATION_NOT_YET_SUPPORTED**.

## 12. Seed-level robustness

The required `seed_metrics.csv` reports one row per variant and seed, including five-fold mean, standard deviation, worst fold, patient count and event count.

## 13. Paired bootstrap uncertainty

Each comparison uses 2,000 patient-level paired OOF bootstrap replicates with fold-order alignment; intervals are in `paired_bootstrap.csv`.

## 14. Residual-risk diagnostics

`residual_diagnostics.csv` contains patient-level residual parts plus summary rows reporting mean absolute residuals, residual standard deviations, and correlations with the frozen Clinical offset.

## 15. Variance decomposition

RPO has lower between-seed H/U variance and lower within-seed fold variance than B0; exact values are in `variance_decomposition.csv`.

## 16. What missing-aware learning could recover

Canonical incomplete DSS-labelled cases add 35 patients and 4 observed events (6.15% of the complete-case event count). Missing-modality work should target cohort coverage and robustness; it does not justify a complex residual architecture under this audit.

## 17. Advantages

The audit separates canonical source missingness from runtime staging, uses cross-fitted Clinical offsets, freezes the residual anchor, and restricts residual heads to additive low-capacity corrections.

## 18. Risks / limitations

DSS has only 65 events. No incremental residual signal was confirmed under the locked seed and bootstrap rules. DSS has only 65 complete-case events, so the negative mechanism result should be validated on an independent cohort before changing the research direction. Runtime staging gaps remain an operational data-transfer issue.

## 19. Final decision

**Decision C**: DO_NOT_BUILD_COMPLEX_MULTIMODAL_RESIDUAL_MODEL_YET
