# MCAT fidelity record

## Paper components

Six genomic signature networks, omic-query histology co-attention, path/omic transformers, gated attention heads and fusion.

## Official repository file/module

mahmoodlab/MCAT: models/model_coattn.py, models/model_utils.py:SNN_Block and datasets/dataset_survival.py signature intersection; metadata file datasets_csv_sig/signatures.csv.

## Local adapted file/module

src/phase_m2/mcat.py:SixSignatureEncoder, MCATModel; src/phase_m2/data.py:_fit_group_matrix.

## Preserved components

six token structure, SNN-style per-signature encoders, omic-query co-attention and transformer/pooling/fusion body

## Changed components

fold-local RNA+CNA symbol intersections, UNI2-h pathology, fixed clinical/Cox dimensions; mutation is unavailable and not fabricated

## Truthful status

PAPER_FAITHFUL_REIMPLEMENTATION
