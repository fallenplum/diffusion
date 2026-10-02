# Phase M8-P0R — Exact Anchor Comparator Repair

## Protocol and provenance

512 patients / 65 DSS events; frozen 5 outer folds; seeds 20261012–20261016.
No training, refit, repartitioning, parameter changes or prediction modification.
Exact anchor agreement across RP/RO/RPO: **PASS**, max absolute difference=0.
A0 equals the saved clinical_offset; source fold scores were independently reproduced from frozen predictions.

## Anchor and variance

| Model | Harrell | Uno | Between seed H/U variance | Within seed H/U variance | Worst fold H/U |
|---|---:|---:|---|---|---|
| A0 | 0.743356 | 0.755839 | 0.000936173/0.00284506 | 0.0106725/0.0140269 | 0.517460/0.479411 |
| C0-Full | 0.771574 | 0.785990 | 0.000948504/0.000866436 | 0.00430972/0.00992074 | 0.666454/0.497749 |
| B0 | 0.778339 | 0.783535 | 0.000302895/0.00016846 | 0.00651507/0.00970396 | 0.623022/0.581452 |
| RP | 0.751957 | 0.766710 | 0.000829501/0.002538 | 0.00856369/0.0112541 | 0.539683/0.489052 |
| RO | 0.743503 | 0.755910 | 0.000936025/0.00284374 | 0.0106791/0.0140381 | 0.517460/0.479411 |
| RPO | 0.756057 | 0.768924 | 0.000570832/0.00219463 | 0.00774512/0.0108675 | 0.585459/0.489052 |

## Matched comparisons

Fold metrics are primary; positive seed counts refer to the five-fold mean differences.

| Comparison | Mean delta Harrell | Mean delta Uno | Positive seeds H/U | Decision |
|---|---:|---:|---|---|
| RP_vs_A0 | +0.008601 | +0.010871 | 5/5, 4/5 | RESIDUAL_SIGNAL_DIRECTIONAL |
| RO_vs_A0 | +0.000147 | +0.000072 | 4/5, 4/5 | RESIDUAL_SIGNAL_DIRECTIONAL |
| RPO_vs_A0 | +0.012701 | +0.013086 | 5/5, 4/5 | RESIDUAL_SIGNAL_DIRECTIONAL |
| A0_vs_C0-Full | -0.028219 | -0.030152 | 2/5, 2/5 | context |
| RPO_vs_B0 | -0.022283 | -0.014611 | 1/5, 3/5 | context |

## Paired patient bootstrap

2,000 paired patient resamples per seed. Global OOF is diagnostic, not model selection.
Global Uno uses the unchanged complete-OOF censoring KM; fold Uno uses each outer-train KM.
The decision uses the most conservative lower bound over the five per-seed intervals.

| Comparison | Seed | Bootstrap delta H | H 95% CI | Bootstrap delta U | U 95% CI |
|---|---:|---:|---|---:|---|
| RP_vs_A0 | 20261012 | -0.000667 | [-0.014628, +0.013153] | +0.007488 | [-0.006666, +0.026396] |
| RO_vs_A0 | 20261012 | +0.000048 | [-0.000408, +0.000518] | +0.000025 | [-0.000297, +0.000366] |
| RPO_vs_A0 | 20261012 | +0.001886 | [-0.010617, +0.014098] | +0.008937 | [-0.004751, +0.027063] |
| A0_vs_C0-Full | 20261012 | +0.001281 | [-0.052261, +0.050313] | -0.027621 | [-0.134605, +0.071809] |
| RPO_vs_B0 | 20261012 | +0.003176 | [-0.046379, +0.052625] | -0.034241 | [-0.117112, +0.035750] |
| RP_vs_A0 | 20261013 | +0.005220 | [+0.001380, +0.009463] | +0.002041 | [-0.007386, +0.007175] |
| RO_vs_A0 | 20261013 | +0.000003 | [-0.000354, +0.000340] | -0.000012 | [-0.000230, +0.000157] |
| RPO_vs_A0 | 20261013 | +0.020473 | [+0.009821, +0.032108] | +0.020032 | [+0.005949, +0.038393] |
| A0_vs_C0-Full | 20261013 | -0.070669 | [-0.128529, -0.018962] | -0.071564 | [-0.133625, -0.014039] |
| RPO_vs_B0 | 20261013 | -0.038951 | [-0.108680, +0.028971] | -0.061026 | [-0.148973, +0.019326] |
| RP_vs_A0 | 20261014 | +0.006478 | [-0.000627, +0.014059] | +0.016065 | [-0.006423, +0.043678] |
| RO_vs_A0 | 20261014 | +0.000190 | [-0.000236, +0.000724] | +0.000115 | [-0.000079, +0.000408] |
| RPO_vs_A0 | 20261014 | +0.006122 | [-0.000915, +0.013551] | +0.014296 | [-0.006005, +0.040076] |
| A0_vs_C0-Full | 20261014 | -0.079557 | [-0.133457, -0.029089] | -0.088488 | [-0.169662, -0.022893] |
| RPO_vs_B0 | 20261014 | -0.005091 | [-0.081351, +0.065530] | +0.015605 | [-0.071281, +0.097395] |
| RP_vs_A0 | 20261015 | +0.003361 | [-0.000509, +0.007429] | +0.002646 | [-0.007150, +0.013664] |
| RO_vs_A0 | 20261015 | +0.000047 | [-0.000216, +0.000367] | +0.000017 | [-0.000084, +0.000138] |
| RPO_vs_A0 | 20261015 | +0.001087 | [-0.007359, +0.009175] | +0.000655 | [-0.010207, +0.012215] |
| A0_vs_C0-Full | 20261015 | +0.025053 | [-0.019264, +0.067791] | +0.047398 | [-0.005862, +0.104618] |
| RPO_vs_B0 | 20261015 | +0.061662 | [+0.001504, +0.127599] | +0.068785 | [+0.015346, +0.131977] |
| RP_vs_A0 | 20261016 | +0.013650 | [-0.001030, +0.029855] | +0.017288 | [+0.000779, +0.040738] |
| RO_vs_A0 | 20261016 | +0.000004 | [-0.000219, +0.000251] | +0.000030 | [-0.000091, +0.000285] |
| RPO_vs_A0 | 20261016 | +0.014521 | [+0.000172, +0.030315] | +0.017353 | [+0.001111, +0.040759] |
| A0_vs_C0-Full | 20261016 | +0.028961 | [-0.052119, +0.114628] | -0.061513 | [-0.147475, +0.024117] |
| RPO_vs_B0 | 20261016 | -0.041416 | [-0.093068, +0.009223] | -0.029632 | [-0.099470, +0.043668] |

## Interpretation

A0 vs C0-Full mean differences: Harrell -0.028219, Uno -0.030152. The previous residual-vs-C0 comparisons mixed changes in Clinical anchor quality with modality corrections. Only residual-vs-A0 tests estimate the change relative to the exact frozen Clinical risk.
An apparent change from the original Decision C must be justified by the matched seed directions and intervals, not by selecting the best residual variant.

## Advantages and limitations

Advantage: identical Clinical offsets and frozen predictions isolate the residual correction without new training.
Limitations: only 65 events; five-seed intervals assess the fixed cohort, not external generalization. This repair does not certify unrelated M8-P0 model or preprocessing choices. The inherited M8-P0 missingness census remains 35 additional DSS-labelled patients / 4 observed events.
Confidence: high for frozen-anchor identity and numerical reproduction; medium for generalizable incremental information.

## Final decision

**B. RESIDUAL_SIGNAL_DIRECTIONAL**

A subsequent shrinkage residual model is justified for a separate phase; do not add complex cross-attention. No next-phase training was performed.
