# Phase 1.5 failure log

## B3_CLAM_SB_survival

- status: `FAILED_TO_REPRODUCE`
- attempts: 1
- reason: The official CLAM repository is classification-focused and no validated survival adaptation/checkpoint is available in the current assets. An initial bag-level Cox adaptation was run as a diagnostic but is excluded from the primary benchmark because it is not a defensible CLAM survival reproduction.
- official reference: https://github.com/mahmoodlab/CLAM

## B5_MambaMIL

- status: `FAILED_TO_REPRODUCE`
- attempts: 1
- reason: Official isyangshu/MambaMIL code requires the mamba package/causal-conv1d stack; mamba_ssm is absent in the locked environment and the project does not contain the official source or checkpoint.
- official reference: https://github.com/isyangshu/MambaMIL

## B7_PAMoE

- status: `FAILED_TO_REPRODUCE`
- attempts: 1
- reason: The official PAMoE repository is not vendored and its released demo requires the original training integration/extra assets; current scope contains frozen UNI2-h features only. No approximation is reported as PAMoE.
- official reference: https://github.com/wjx-error/PAMoE
