# Reproducibility

Run from the M8-P1R source branch with the fixed M1R patient split, M2R2 fold-local RNA/CNA preprocessing, frozen M8-P1R C0-Full outer-test anchor, fresh M9 seeds 20261020–20261024, 2000 patient bootstrap replicates and 200 conditional permutation repeats. RP-L1 is the fold-local Gated ABMIL embedding plus a linear residual head; conditional neighborhoods are Clinical-distance nearest donors with same-stage preference and no DSS inputs. Uno censoring weights remain outer-fold local in OOF and bootstrap summaries. All repository outputs are lightweight summaries only.

Historical B0 context uses runtime/cells/B0_20260904_<fold>.npy and is reported in paired_bootstrap.csv; it does not enter the M9 decision gate.
