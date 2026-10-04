# Omics pathway representation probe

## Corrected genuine per-seed five-fold OOF results

|Representation|Per-seed OOF Harrell mean +/- std|Per-seed OOF Uno mean +/- std|Parameters|
|---|---:|---:|---:|
|Pathway-RNA|0.6180 +/- 0.0056|0.5642 +/- 0.0371|509,990|
|Pathway-RNA+CNA|0.6054 +/- 0.0175|0.5204 +/- 0.0508|542,119|
|Signature-RNA+CNA|0.6013 +/- 0.0192|0.4953 +/- 0.0377|477,666|

- Each row first concatenates the five disjoint outer-test predictions within each seed. The reported mean/std is over three per-seed OOF metrics.
- Pathway-RNA is a full 3-seed x 5-fold rerun. Existing Pathway-RNA+CNA and Signature raw cells were preserved; their private predictions were replayed at archived epochs and matched every archived metric and parameter count.

## Scope

- These are conditional model comparisons under the fixed protocol, not universal biological claims about CNA or pathway priors.
- Detailed paired comparison, mechanism controls and validation are in `reports/po_phase0_pathway_repair_20261004/`.
