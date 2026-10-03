# Phase K2RR — Exact Topology Audit Repair

- Decision: **B. TOPOLOGY_DIRECTIONAL**
- Real topology mean: **0.173088**
- 23-permutation mean ± std: **0.173066 ± 0.003640**
- Real percentile: **50.00%**; exact p-value: **0.541667**
- Positive seed/fold fraction: **0.600000**
- Repair A: complete outer-test-fold predictions/targets were collected before Pearson; patient-order invariance passed.
- Repair B: RNA and CNA remained separate in both the dual-teacher training loss and evaluation metric.
- DSS, survival, and raw WSI/SVS were not used.

Metric tests: test_full_fold_metric_order_invariant=PASS, test_no_batchwise_pearson=PASS, test_rna_cna_not_preaveraged=PASS, test_same_model_capacity_all_permutations=PASS, test_same_initialization_all_permutations=PASS, test_dss_not_used=PASS
