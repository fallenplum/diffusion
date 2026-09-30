# CAMR fidelity record

## Paper module/equation

Invariant/specific encoders, adversarial cross-modal alignment, reconstruction, orthogonality, CMFM and survival loss.

## Official code module

CAMRModel, LowRankMultimodalFusion and gradient_reverse in src/phase_m2/camr.py

## Our adapted code module

CAMRModel, LowRankMultimodalFusion and gradient_reverse in src/phase_m2/camr.py

## Preserved?

yes, status=OFFICIAL_ADAPTED

## Adaptation reason

Official gene/path/CNA inputs map to fold-local RNA/UNI2-h/CNA; clinical is fused before the Cox head.
