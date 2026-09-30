# MMPTrans fidelity record

## Paper module/equation

Train-only Ch=16 GMM morphology prototypes, transcriptomic pathway prototypes, prototype-specific feed-forward encoding, MMPTrans fusion and Cox.

## Official code module

fit_morphology_prototypes, fit_transcriptomic_prototypes and MMPModel in src/phase_m2/mmp.py

## Our adapted code module

fit_morphology_prototypes, fit_transcriptomic_prototypes and MMPModel in src/phase_m2/mmp.py

## Preserved?

yes, status=OFFICIAL_ADAPTED

## Adaptation reason

Outer-train PCA/GMM banks are fitted because the fixed cohort has no external hallmark matrix; six RNA/CNA signature groups provide the transcriptomic mapping.
