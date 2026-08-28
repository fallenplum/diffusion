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

## Phase 1.5 benchmark audit

The pathology-only UNI2-h/DSS benchmark is resumable with:

```bash
python src/phase1_5_benchmark.py --root . --resume
python scripts/validate_phase1_5.py --root .
```

It reuses the frozen Phase1R patient split, runs three seeds for every locally reproducible method, writes the full runtime tree under `outputs/phase1_5_benchmark_audit/`, and keeps raw data, checkpoints, patient-level OOF/attention files out of Git. Lightweight results, failure ledger, bootstrap comparisons, figures and the next decision are in `reports/phase1_5_benchmark_audit/`. MambaMIL, PAMoE and the unvalidated CLAM survival adaptation are retained as explicit `FAILED_TO_REPRODUCE` entries rather than being replaced by paper numbers.

## Local setup

For source checks:

```bash
python -m pip install -r requirements-ci.txt
python -m compileall -q src
```

The neural experiments additionally require a CUDA-compatible PyTorch environment. Their commands, seeds, data scope, and split rules are recorded in `PHASE_P1_UNI2H_CONTEXT_DSS_VALIDATION_PROMPT.md` and the source scripts under `src/`.

## Reproducibility and safety

All reported survival metrics are patient-level out-of-fold metrics. The outer split is fixed with `StratifiedKFold(n_splits=5, shuffle=True, random_state=20260723)`, and the repository workflow checks compilation, statistical utility imports, and the absence of protected data/model binaries.

## Development setup

Install formal runtime dependencies with `python -m pip install -r requirements.txt`. The CI-only dependencies (pytest, ruff, PyYAML and schema validation) are listed in `requirements-ci.txt`. The observed local GPU environment and the CPU/GPU boundary are documented in `docs/environment.md`.

## Testing and CI

Run the same lightweight checks locally:

```bash
ruff check src tests scripts
pytest
python -m compileall -q src scripts tests
python scripts/check_protected_data.py
python scripts/run_research_experiment.py --request automation/requests/example_phase1_smoke.yaml --dry-run
```

`.github/workflows/ci.yml` runs these checks on a GitHub-hosted CPU runner and installs only the official CPU PyTorch wheel for synthetic model tests. It never downloads patient data or large model artifacts.

## Experiment configs and automated workflow

The historical Phase P1 parameters are declared under `configs/phase1/`. A request is validated against `automation/schemas/experiment_request.schema.json` and executed through `scripts/run_research_experiment.py`. `mode: smoke` is synthetic and cannot be treated as a scientific result; `mode: full` requires an explicit repository-relative command and a self-hosted GPU/data environment.

The self-hosted workflow is `.github/workflows/research-experiment.yml`. It accepts one request per same-repository PR (and manual dispatch), applies fork protection, serializes GPU runs, fails fast in `scripts/gpu_preflight.py`, and uploads state/logs/lightweight metrics with `if: always()`. It does not auto-commit results, preventing an Action loop. See `docs/auto_research_loop.md` and `docs/self_hosted_runner.md` before registering a runner.

## Data safety

Set `DIFFUSION_DATA_ROOT` only on the local/GPU runner. Do not place a real data path, patient identifier, API key, PAT, HuggingFace token, SSH key or runner registration token in a request, workflow, README or commit. `TCGA/`, `data/`, raw H5/NPY/NPZ files, checkpoints and large outputs remain local and are checked by both `.gitignore` and the CI guard.
