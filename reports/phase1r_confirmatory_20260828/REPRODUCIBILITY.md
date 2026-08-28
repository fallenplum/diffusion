# Reproducibility

Use the validated local CUDA environment (PyTorch 2.4.1+cu121) and keep protected data outside Git.

```powershell
$env:DIFFUSION_DATA_ROOT = '<local data root containing TCGA>'
python src/phase1r_confirmatory.py --config configs/phase1r/confirmatory.yaml --run-dir outputs/phase1r_confirmatory_20260828 --preflight-only
python -u src/phase1r_confirmatory.py --config configs/phase1r/confirmatory.yaml --run-dir outputs/phase1r_confirmatory_20260828 --resume
```

The split is generated once and then verified by SHA256 on every resume. Intermediate checkpoints, patient-level predictions, and absolute local paths remain below the ignored runtime directory.

