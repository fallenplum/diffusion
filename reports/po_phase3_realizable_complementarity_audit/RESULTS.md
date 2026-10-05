# Phase3 realizable complementarity audit

- Fixed cohort: 512 patients, 65 DSS events, fixed five outer folds, three seeds.
- All proposed risks are single patient-level scores; the pairwise diagnostic is not used to construct risk.

## Direct answers

1. Patient-level reliability learnability: mean pairwise P-vs-R AUC 0.5838; balanced accuracy 0.5413.
2. Dynamic gate vs Pathology-only: mean ΔHarrell 0.0057; mean ΔUno -0.0508; seed-wise Harrell [0.002, 0.0138, 0.0013], Uno [-0.0098, -0.0094, -0.1333].
3. Dynamic gate vs static two-weight fusion: mean Harrell 0.0154; mean Uno -0.0124.
4. Gate adaptivity: mean test gate std 0.3525; mean test gate weight 0.5514; distributions are in `gate_diagnostics.csv`.
5. Feature selection counts: {20260904: {'F0-RiskOnly': 1, 'F1-RiskPlusPathReliability': 1, 'F2-RiskPlusPathPlusRNAUncertainty': 3}, 20260905: {'F0-RiskOnly': 2, 'F1-RiskPlusPathReliability': 1, 'F2-RiskPlusPathPlusRNAUncertainty': 2}, 20260906: {'F0-RiskOnly': 2, 'F1-RiskPlusPathReliability': 2, 'F2-RiskPlusPathPlusRNAUncertainty': 1}}; model counts: {20260904: {'G0-Logistic': 4, 'G1-TinyMLP': 1}, 20260905: {'G0-Logistic': 2, 'G1-TinyMLP': 3}, 20260906: {'G0-Logistic': 4, 'G1-TinyMLP': 1}}.
6. Phase2 oracle headroom recovered by the dynamic gate: Harrell 0.0057; Uno -0.0508; Phase2 headroom was approximately 0.1904/0.1598.
7. Paired bootstrap Proposed-Pathology lower bounds: Harrell -0.0677; Uno -0.0817.
8. Historical MGCT-PO comparison is in `paired_comparisons.csv`; no new MGCT training was performed.
9. Decision: **ORACLE_NOT_REALIZABLE**.

## Interpretation

the high Phase2 oracle headroom is not recovered by a learnable patient-level reliability gate; stop using the oracle as a sufficient motivation for more complex fusion.

## Boundaries

Attention entropy, attention concentration, and RNA bootstrap spread are empirical reliability proxies. They do not establish biological credibility, full epistemic uncertainty, or causal complementarity.

## Required next step

If the decision is REALIZABLE_COMPLEMENTARITY_SUPPORTED, the next experiment may test a strictly low-capacity representation interaction. Otherwise, stop using dynamic reliability or pairwise-oracle headroom as justification for Q-Former, cross-attention, or other large fusion architectures.
