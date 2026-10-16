# Reproduction

Requires private parent Phase17 assets and exact parent commit16d335f. No patient arrays, credentials, weights or checkpoints are supplied publicly. Python3.12/torch2.5.1+cu124 and scikit-survival from the preserved project environment.

```bash
cd /root/phase16_routing_repo
export PYTHONPATH=.:/root/phase10_trimodal_eval_deps
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/audit_phase17r_operator.py
python scripts/schedule_phase17r.py
python scripts/complete_phase17r.py
```

The initial audit must only be run before a fresh experiment; it records a protocol lock. The scheduler is resumable with checkpoint hashes and a fixed training-source identity. Once architecture decisions are locked, it does not rerun selection using outer results. R2 candidates are selected separately in each seed/fold. R3 executes only if the locked inner gate passes.

Completion independently reconstructs Clinical/S32 anchors, validates all fold metrics against scikit-survival, bootstraps shared patient counts, replays every outer checkpoint under four interventions and actually repeats training after mutating outer labels. Short audit fits persist no checkpoint. PNGs must be actually inspected before marking visual QA or publishing.

Reported OOF contains only test-patient risk from each patient's corresponding fold. Raw risks are never compared across differently fitted folds. Bootstrap uncertainty is conditional on saved fits and does not estimate external validation or all training uncertainty. This is exploratory research on a repeatedly developed cohort.
