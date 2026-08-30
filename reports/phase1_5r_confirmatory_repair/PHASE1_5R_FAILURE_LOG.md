# Phase 1.5R failure log

## CLAM_survival

- status: `FAILED_TO_REPRODUCE`
- attempts: 2
- reason: Official CLAM repository and a separate survival-adaptation search were checked; the available official training path is classification-focused and no validated survival checkpoint/adapter is present. Generic gated attention plus clustering is not reported as CLAM.
- references: https://github.com/mahmoodlab/CLAM; survival-adaptation search recorded in this run

## B4_MambaMIL

- status: `FAILED_TO_REPRODUCE`
- attempts: 4
- reason: Official dependency preflight/install routes are recorded in failures/mamba_dependency_attempts.json; no result is promoted unless mamba_ssm and causal-conv1d import successfully.
- references: https://github.com/isyangshu/MambaMIL

## TransMIL

- status: `RENAMED_REIMPLEMENTATION`
- attempts: 1
- reason: The local method is named TransformerMIL_reimplementation because the official TransMIL structure/checkpoint was not vendored; no generic Transformer result is labeled official TransMIL.
- references: https://github.com/szc19990412/TransMIL

## EAGLE

- status: `RENAMED_REIMPLEMENTATION`
- attempts: 1
- reason: SpatialNeighborhoodMIL is a paper-faithful coordinate-only neighbourhood equivalent; official EAGLE/MASE implementation was not available in the current assets.
- references: paper-faithful equivalent; official implementation unavailable

## PANTHER

- status: `RENAMED_REIMPLEMENTATION`
- attempts: 1
- reason: PANTHER_style_GMM uses train-only PCA64 and GMM prototypes from raw 1536D features; it is not called official PANTHER.
- references: https://github.com/mahmoodlab/PANTHER
