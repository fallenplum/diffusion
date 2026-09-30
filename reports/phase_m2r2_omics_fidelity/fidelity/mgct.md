# MGCT fidelity record

## Paper components

MGCA Q/K/V cross-attention, gated pooling, feed-forward refinement and two-stage mutual-guided fusion.

## Official repository file/module

mahmoodlab/MGCT reference tree: model/*.py and utils/coattn_train_utils.py (audited cached tree; not imported at runtime).

## Local adapted file/module

src/phase_m2/mgct.py:MGCTModel, MutualGuidedFusionLayer.

## Preserved components

mutual-guided cross-attention, gated pooling and fusion stages

## Changed components

UNI2-h and fold-local RNA/CNA/clinical tensors; official training/data loader is not used

## Truthful status

PAPER_FAITHFUL_REIMPLEMENTATION
