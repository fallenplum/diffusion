# Pathway representation repair

- Pathway-RNA was fully rerun with 3 seeds x 5 outer folds.
- Pathway-RNA+CNA and Signature raw cells were preserved; only archived-epoch replay was used to recover private OOF predictions.
- `seed_oof_*` is computed after concatenating five outer-test predictions per seed. It is not a mean over 15 cells.

## Genuine per-seed OOF metrics

|Method|OOF Harrell mean +/- std|OOF Uno mean +/- std|
|---|---:|---:|
|Pathway-RNA|0.6180 +/- 0.0056|0.5642 +/- 0.0371|
|Pathway-RNA+CNA|0.6054 +/- 0.0175|0.5204 +/- 0.0508|
|Signature-RNA+CNA|0.6013 +/- 0.0192|0.4953 +/- 0.0377|

## Same-seed paired comparison

- Delta definition: Pathway-RNA+CNA minus Pathway-RNA.
- Bootstrap paired-patient Harrell delta: -0.0142 (95% CI -0.0584 to 0.0307).
- Bootstrap paired-patient Uno delta: -0.0442 (95% CI -0.1012 to 0.0092).
- The comparison is conditional on the two fitted architectures and does not establish a universal biological claim about CNA.

## Linear-layer mechanism controls

The controls hold the outer split and preprocessing fixed; alpha and epoch are selected only on inner data.

|Control|OOF Harrell mean +/- std|OOF Uno mean +/- std|Median parameters|
|---|---:|---:|---:|
|RNA-LN-Linear|0.6409 +/- 0.0010|0.5733 +/- 0.0021|512|
|RNA-Linear|0.6564 +/- 0.0003|0.5886 +/- 0.0030|512|
|RNA-MLP|0.5849 +/- 0.0246|0.5543 +/- 0.0071|50369|
|RNA-MLP-noLN|0.6001 +/- 0.0201|0.5847 +/- 0.0124|49345|
|RNA-PCA32-Linear|0.6116 +/- 0.0000|0.5617 +/- 0.0000|32|
|RNA-PathwayMean-Linear|0.5708 +/- 0.0151|0.5776 +/- 0.0250|50|

- These controls explain the direct RNA linear-vs-MLP result only; they do not claim that every hidden pathway head should be linear.
