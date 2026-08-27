# diffusion

Pathology-only experiments for UNI2-h whole-slide context pretraining and DSS survival validation.

## Repository scope

This repository contains source code, experiment protocol, lightweight summaries, and audit reports. Patient-level data and large derived artifacts are intentionally excluded:

- `TCGA/` and `data/` raw/derived datasets;
- H5/NumPy bags, checkpoints, caches, and large generated outputs;
- local literature PDFs and application documents.

The data paths used by the experiments are documented in the manifest, but the data themselves must be supplied locally under the terms applicable to the dataset.

## Phase P1 status

The completed patient-level five-fold evaluation did not support the primary claim that coordinate-aware spatial masked-context SSL reliably improves DSS risk prediction. The exact results, uncertainty intervals, leakage checks, and bounded next decision are in:

- `outputs/pathology_context_phase1/RESULTS.md`
- `outputs/pathology_context_phase1/NEXT_DECISION.md`
- `outputs/pathology_context_phase1/results_master.csv`
- `outputs/pathology_context_phase1/experiment_manifest.json`

The lightweight reports are committed; fold checkpoints and raw embeddings remain local and ignored.

## Local setup

For source checks:

```bash
python -m pip install -r requirements-ci.txt
python -m compileall -q src
```

The neural experiments additionally require a CUDA-compatible PyTorch environment. Their commands, seeds, data scope, and split rules are recorded in `PHASE_P1_UNI2H_CONTEXT_DSS_VALIDATION_PROMPT.md` and the source scripts under `src/`.

## Reproducibility and safety

All reported survival metrics are patient-level out-of-fold metrics. The outer split is fixed with `StratifiedKFold(n_splits=5, shuffle=True, random_state=20260723)`, and the repository workflow checks compilation, statistical utility imports, and the absence of protected data/model binaries.
