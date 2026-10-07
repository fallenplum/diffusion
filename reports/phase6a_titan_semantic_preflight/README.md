# Phase6A TITAN semantic preflight

Frozen CRC morphology vocabulary, official TITAN slide/text representations,
label-independent geometry checks, and nested patient-level T/N/M/Stage probes.
No molecular pathway inputs or survival model is used.

## Run

Use an isolated environment with the official TITAN dependencies. The server uses
Python 3.12, CUDA torch 2.5.1, torchvision 0.20.1, transformers 4.46.0, timm 1.0.3,
einops 0.6.1, einops-exts 0.0.4, and huggingface-hub 0.36.0. NumPy, h5py, pandas,
scipy, scikit-learn, and threadpoolctl are also required.

```powershell
# Set PHASE6A_HF_TOKEN in the process environment using your authorized account.
python scripts/download_phase6a_titan.py --destination E:\diffusion_phase6a_titan\model
python scripts/phase6a_titan_semantic_preflight.py --stage all
python scripts/summarize_phase6a_titan.py
```

For the authorized server execution, `scripts/phase6a_titan_remote.py` transfers
at most two bounded windows of original H5 files into a temporary input queue
(the final run uses four transfer channels and 16 files per window).
The server consumes and deletes each H5, reads its existing clinical table, and
uses the original official attention implementation on a 32 GB GPU. The model
weight is transferred temporarily under explicit authorization, then deleted
after the formal run. E drive retains the original H5, weights and returned
representation/OOF caches. The initial local inference encountered host RAM
allocation failures and did not produce the Phase6A cohort results.

The existing verified CONCH cohort inventory is the input manifest. Default local
H5 delivery root is `E:\diffusion_conch_v15_features`. Weight checksums, source
checksums, exact loading keys, dimensions, finite checks, and agreement with the
official zero-shot scoring path must pass before interpretation. A code/config
change after the first successful slide output requires a new cache directory.

Native slide features are projected using `vision_encoder.proj` for text scores.
The generic baseline uses native patient features. No fitted semantic projector
or feature-row correspondence with UNI2 is introduced.

Large source files stay on E drive; derived representation/OOF caches are
mirrored there. Server clinical labels and small run caches remain outside Git.
Only fixed configs, scripts and aggregate diagnostics/results are publishable.
`DONE.json` is required to claim completion; a running log is not a result.
