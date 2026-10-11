# G0R old failure map

1. G0 used Clinical-only OOF anchors for graph arms. G0R constructs the same outer Clinical+S32 Ridge anchor for R0/R5/R6/R7 and verifies test vectors elementwise.
2. G0 selected Top512 after standardization. G0R selects by raw training-fold log-expression variance, then standardizes with training statistics.
3. G0 used asymmetric ridge lambda. G0R uses the same inner grid for every Ridge arm.
4. G0 graph audit had 215 nodes and 44 KGML edges without complete coverage reporting. G0R records node/edge/isolated/pathway coverage and preserves sparse structure.
5. G0 permuted adjacency without preserving per-node degree. G0R uses double-edge swaps and checks exact degrees/edge count.
6. G0 stopped on training loss and used a placeholder validation branch. G0R selects epoch on inner-val Harrell and refits outer-train at the selected epoch.
7. G0 averaged fold C and silently zeroed unsupported Uno weights. G0R aggregates numerators/denominators and reports unsupported event counts.
8. G0 published test indices. G0R keeps OOF/index files private and publishes only aggregate artifacts.
