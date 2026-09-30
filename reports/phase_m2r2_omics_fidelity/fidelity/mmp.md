# MMPTrans-Hallmark fidelity record

## Paper components

Outer-train Ch=16 morphology prototypes, 50 Hallmark pathway groups, pathway-specific encoders, MMPTrans fusion and Cox.

## Official repository file/module

mahmoodlab/MMP: src/mil_models/model_multimodal.py, src/mil_models/components.py, src/wsi_datasets/wsi_survival.py, src/utils/proto_utils.py and src/data_csvs/rna/metadata/hallmarks_signatures.csv (metadata vendored locally).

## Local adapted file/module

src/phase_m2/mmp.py:MMPModel, fit_morphology_prototypes; src/phase_m2/data.py:_fit_group_matrix.

## Preserved components

morphology-only GMM bank, 50 pathway group semantics, pathway-specific encoders and bidirectional fusion

## Changed components

UNI2-h pathology tensor, fixed RNA-only Hallmark branch because mutation/CNA pathway inputs are unavailable, and benchmark Cox dimensions

## Truthful status

PAPER_FAITHFUL_REIMPLEMENTATION
