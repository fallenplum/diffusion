# Phase M2 SOTA Baselines

## Decision

**Phase M2 COMPLETE**

- Cohort: 512 patients, 65 DSS events.
- The committed Phase M1 fixed patient split was reused; no split was regenerated.
- Reports contain aggregate fold/seed/compute summaries only; patient IDs, H5/NPY/NPZ, checkpoints and patient-level predictions are excluded.

## Baseline ranking

| Rank | Method | Harrell C seed mean ± std | Uno C seed mean ± std | Fold mean ± std | Worst fold | Params | GPU GB | Time s | Reproduction status |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | CAMR | 0.7180 ± 0.0454 | 0.7322 ± 0.0275 | 0.7713 ± 0.0866 | 0.6288 | 401887 | 5.20 | 421.6 | ADAPTED_ARCHITECTURE_FAITHFUL |
| 2 | MGCT | 0.6896 ± 0.0072 | 0.7438 ± 0.0091 | 0.7271 ± 0.0894 | 0.6314 | 392863 | 5.30 | 503.9 | ADAPTED_ARCHITECTURE_FAITHFUL |
| 3 | TransSurv-style Transformer | 0.6716 ± 0.0422 | 0.7003 ± 0.0240 | 0.6976 ± 0.0926 | 0.5638 | 393535 | 5.39 | 418.9 | ADAPTED_ARCHITECTURE_FAITHFUL |
| 4 | MOTCat / MCAT-style co-attention | 0.6555 ± 0.0117 | 0.7167 ± 0.0194 | 0.7055 ± 0.0896 | 0.5979 | 299743 | 5.25 | 755.1 | ADAPTED_ARCHITECTURE_FAITHFUL |
| 5 | EarlyFusion / Concat + Cox | 0.5908 ± 0.0000 | 0.4923 ± 0.0000 | 0.5979 ± 0.1343 | 0.4357 | 2367 | 0.02 | 25.1 | ADAPTED_ARCHITECTURE_FAITHFUL |
| 6 | MMP | 0.5750 ± 0.0118 | 0.5335 ± 0.0510 | 0.5613 ± 0.0794 | 0.4580 | 394015 | 5.25 | 352.7 | ADAPTED_ARCHITECTURE_FAITHFUL |

## Core mechanisms

- B0: early concatenation with a Cox head.
- B1: Transformer self-attention across pathology, omics and clinical tokens.
- B2: bidirectional mutual-guided cross-attention between pathology and omics.
- B3: shared/specific representations with alignment and orthogonality penalties.
- B4: learned pathology and omics prototype tokens with prototype-level cross-attention.
- B5: co-attention with entropic transport coupling between pathology and omics tokens.

## Failure signatures for the next phase

- Compare whether explicit cross-modal interactions improve both Harrell and Uno C consistently across folds and seeds.
- Inspect methods whose worst fold collapses while seed means remain competitive; this indicates fold sensitivity rather than a stable mechanism gain.
- Treat any `FAILED_TO_REPRODUCE` row as a boundary condition and do not describe it as a successful official reproduction.
