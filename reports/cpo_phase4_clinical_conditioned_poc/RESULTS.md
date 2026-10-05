# CPO Phase4 Clinical-Conditioned POC

- Cohort: 512 patients / 65 DSS events.
- Formal cells: 3 seeds x 5 outer folds.
- Clinical fields retained: AGE, SEX, AJCC pathologic stage, PATH T/N/M.
- Decision: **ADDITIVE_NONCLINICAL_GAIN_SUPPORTED**.

## Canonical metrics

|Method|Harrell seed OOF mean +/- SD|Uno seed OOF mean +/- SD|
|---|---:|---:|
|Clinical-only|0.7569 +/- 0.0332|0.7543 +/- 0.0500|
|B1 C+P additive|0.7520 +/- 0.0310|0.8002 +/- 0.0222|
|B2 C+R additive|0.7317 +/- 0.0179|0.7028 +/- 0.0437|
|A1 Additive-C+P+R|0.7692 +/- 0.0144|0.7596 +/- 0.0443|
|B4 equal standardized C+P+R|0.7692 +/- 0.0144|0.7590 +/- 0.0444|
|A2 CC-RiskScale|0.7644 +/- 0.0175|0.7576 +/- 0.0395|
|A3 CC-LowRank-full|0.7686 +/- 0.0114|0.7638 +/- 0.0408|
|A4 CC-LowRank-CP-only|0.7659 +/- 0.0138|0.7581 +/- 0.0443|
|A5 CC-LowRank-CR-only|0.7646 +/- 0.0353|0.7600 +/- 0.0736|
|Selected-conditioned|0.7584 +/- 0.0172|0.7584 +/- 0.0314|
|Pathology-only|0.6772 +/- 0.0092|0.7198 +/- 0.0031|
|RNA-only|0.6565 +/- 0.0135|0.5801 +/- 0.0129|

## Direct answers

1. Canonical Harrell/Uno unified: **yes**; Uno uses fold-local IPCW micro-aggregation.
2. Clinical-only: H=0.7569, U=0.7543.
3. Pathology-only: H=0.6772, U=0.7198; RNA-only: H=0.6565, U=0.5801.
4. Additive C+P+R: H=0.7692, U=0.7596; additive gain over Clinical: H=0.0122, U=0.0054.
5. CC-RiskScale: H=0.7644, U=0.7576; support gate=False.
6. CC-LowRank: H=0.7686, U=0.7638; support gate=False.
7. CP-only and CR-only results are in ablation.csv; their stability is compared by seed-level means and SD.
8. Conditioning diagnostics include alpha/lambda, interaction variance/correlation, gradients, parameter counts, GPU memory, and gate standard deviations.
9. Best historical strict CPO comparator: M7-CommonOnly; reaggregation and exclusions are in historical_reaggregation.csv and historical_provenance.json.
10. Formal audit status: PASS; inner-val labels entered upstream main-effect training: **no**.

## Interpretation

The result is interpreted only under the fixed 512/65 DSS cohort, fixed outer folds, legal full Clinical baseline, strict two-stage training, and the canonical metrics above.

## Next step

Conditioning is not supported by the paired gates; retain the additive conclusion and do not expand to P-R cross-attention or large fusion.
