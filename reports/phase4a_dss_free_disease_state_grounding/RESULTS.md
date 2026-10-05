# Phase4A Results

- Formal run: **complete**; 3 seeds × 5 outer folds × (M0, M1, M2, M3, M5, M6 + 3 M4 shuffle controls) = **135 cells**.
- DSS supervision used for training/selection: **false**.
- Decision: **STRUCTURE_HELPFUL_SEMANTICS_UNSUPPORTED**.

## Direct answers

1. DSS supervision: **not used**; DSS time/event/censoring were excluded from training tensors, loss, early stopping, selection, and metrics.
2. Official source: MSigDB 2024.1.Hs Hallmark H, 50 sets; matched RNA gene universe: **4191** genes. Fold-local matched/unmatched counts are in `hallmark_mapping_audit.json`.
3. True versus shuffled controls: **matched** pathway sizes and gene degrees; controls are 9401, 9402, 9403, with separate OOF scores and mean ± SD.
4. RNA-only G_TNM: **0.4150**.
5. Pathology-only G_TNM: **0.4586**.
6. Naive P+RNA G_TNM: **0.4328**.
7. DiseaseState-True G_TNM: **0.4329**.
8. True minus shuffled mean ΔG_TNM: **0.0079**; seed deltas: [0.0111, -0.0051, 0.0177].
9. Paired bootstrap True minus shuffled: **not uniformly supported**; 95% CI lower bounds are [-0.017, -0.0198, 0.0078] (2000 patient resamples per seed).
10. True > shuffled direction: T 2/3 seeds, N 3/3, M 1/3.
11. M3 versus M2 mean ΔG_TNM: **0.0001**; this is a marginal directional difference, not stable superiority.
12. M3 versus M6 mean ΔG_TNM: **0.0007**; no robust cross-seed claim.
13. Masked pathway reconstruction: auxiliary reconstruction metrics are in `masked_reconstruction.csv`; they were not used for DSS evaluation.
14. Disease tokens collapse: **no** under the preregistered variance flag.
15. Pathway-conditioned pathology attention: stability statistics are in `pathology_attention_stability.csv`; stability is not biological validation.
16. Current conclusion: **STRUCTURE_HELPFUL_SEMANTICS_UNSUPPORTED**.
17. Phase4B survival transfer allowed: **no**.
18. If later allowed, freeze the disease-state encoder and train only a small adapter/survival head.

## Audit files

All required audit files are present in this directory. Patient IDs, raw predictions, embeddings, and attention arrays remain outside Git in the runtime cache.
