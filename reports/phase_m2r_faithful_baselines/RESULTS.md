# Phase M2R Faithful Baselines

## Decision

**Phase M2R COMPLETE**

- Cohort: 512 patients, 65 DSS events.
- The committed Phase M1R fixed patient split was reused.
- The ranking table contains only models with complete required seed/fold evidence.

## Formal ranking

| Rank | Method | Harrell C seed mean +/- std | Uno C seed mean +/- std | Harrell C fold mean +/- std | Worst Harrell fold | Params | GPU GB | Time s | Fidelity |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | MCAT | 0.7433 +/- 0.0221 | 0.8016 +/- 0.0104 | 0.7906 +/- 0.0651 | 0.6862 | 737657 | 5.28 | 770.9 | OFFICIAL_ADAPTED |
| 2 | EarlyFusion | 0.7113 +/- 0.0000 | 0.6744 +/- 0.0000 | 0.7200 +/- 0.1172 | 0.5928 | 2347 | 0.02 | 26.7 | PAPER_FAITHFUL_REIMPLEMENTATION |
| 3 | CAMR | 0.7109 +/- 0.0494 | 0.7276 +/- 0.0092 | 0.7542 +/- 0.0799 | 0.6547 | 1051210 | 5.28 | 662.4 | OFFICIAL_ADAPTED |
| 4 | TransSurv | 0.6638 +/- 0.0394 | 0.6968 +/- 0.0517 | 0.7019 +/- 0.0999 | 0.5653 | 652983 | 5.47 | 695.1 | PAPER_FAITHFUL_REIMPLEMENTATION |
| 5 | MGCT | 0.6614 +/- 0.0114 | 0.7065 +/- 0.0064 | 0.6828 +/- 0.1049 | 0.4770 | 839997 | 5.59 | 682.6 | OFFICIAL_ADAPTED |
| 6 | MOTCat | 0.6528 +/- 0.0453 | 0.7075 +/- 0.0208 | 0.6927 +/- 0.0980 | 0.5154 | 353849 | 5.28 | 722.4 | OFFICIAL_ADAPTED |
| 7 | MMPTrans | 0.6399 +/- 0.0210 | 0.6679 +/- 0.0637 | 0.7097 +/- 0.0847 | 0.6113 | 1807097 | 2.91 | 764.7 | OFFICIAL_ADAPTED |

## Fidelity status

- B0 EarlyFusion: status=PAPER_FAITHFUL_REIMPLEMENTATION; source=fixed protocol baseline.
- B1 TransSurv: status=PAPER_FAITHFUL_REIMPLEMENTATION; source=TransSurv paper-faithful reimplementation adapted to UNI2-h.
- B2 MGCT: status=OFFICIAL_ADAPTED; source=https://github.com/lmxmercy/MGCT.
- B3 CAMR: status=OFFICIAL_ADAPTED; source=https://github.com/wxq-ustc/CAMR.
- B4 MMPTrans: status=OFFICIAL_ADAPTED; source=https://github.com/mahmoodlab/MMP.
- B5 MCAT: status=OFFICIAL_ADAPTED; source=https://github.com/mahmoodlab/MCAT.
- B6 MOTCat: status=OFFICIAL_ADAPTED; source=https://github.com/Innse/MOTCat.

## Failure signatures

- No formal baseline failed its required cell evidence.
