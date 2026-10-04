# Omics DSS Signal Probe

## Protocol

- Cohort: 512 patients, 65 DSS events, existing patient-level five outer folds.
- RNA and CNA use the audited M2R2 fold-local preprocessing. Clinical fields are excluded from every forward path and selection step.
- Linear Cox controls select alpha from `{1e-3, 1e-2, 1e-1}` using only the inner training split. The main table keeps the Cox risk direction learned by the training objective.
- The permutation null shuffles training event labels within each outer fold and evaluates on the untouched test labels: 10 repetitions × 5 folds.

## Main results

| Method | Harrell C, fold mean ± SD | Uno C, fold mean ± SD |
|---|---:|---:|
| RNA-only linear Cox | 0.6547 ± 0.1014 | 0.5916 ± 0.1672 |
| CNA-only linear Cox | 0.4332 ± 0.0859 | 0.4253 ± 0.1651 |
| RNA+CNA linear Cox | 0.6498 ± 0.1257 | 0.6101 ± 0.1407 |
| Direct MLP RNA+CNA | 0.5670 ± 0.0915 | 0.5320 ± 0.0834 |
| Existing Phase 0 Omics-only encoder, 3-seed reference | 0.5183 ± 0.0115 | 0.4822 ± 0.0263 |
| Permutation null, 50 fold replicates | 0.5106 ± 0.1056 | 0.5254 ± 0.1451 |

## Risk-direction audit

Selecting alpha and the sign from the inner validation split gives:

| Method | Harrell C | Uno C | Selected signs by outer fold |
|---|---:|---:|---|
| RNA-only | 0.5903 | 0.6042 | `[+, +, +, +, -]` |
| CNA-only | 0.4902 | 0.5649 | `[-, +, -, -, +]` |
| RNA+CNA | 0.5739 | 0.5924 | `[+, +, -, +, -]` |

The CNA signs change in three of five folds. A fixed sign inversion gives CNA H=0.5668 and U=0.5747, but this is an orientation audit rather than a predeclared confirmatory model; the fold-level direction instability prevents a stable CNA claim.

## Conclusion

1. **RNA has independently transferable DSS signal.** The RNA-only linear control is well above the permutation null on both metrics, although variance is large because only 65 DSS events are available.
2. **CNA does not show stable independent DSS signal under this cohort and protocol.** Its learned-direction result is below 0.5, and the inner-selected direction changes across folds. The inverted audit suggests a weak or unstable reverse association, not a reliable CNA predictor.
3. **The 0.52 Omics-only result is mainly an encoder limitation, not evidence that omics data contain no DSS signal.** A simple RNA+CNA linear Cox reaches H=0.6498/U=0.6101, while the existing chunk-token Omics-only encoder reaches H=0.5183/U=0.4822. A direct MLP recovers part of the gap (H=0.5670/U=0.5320) but remains below the linear control.
4. **The current bottleneck is strongest for RNA information extraction and fusion.** CNA weakness contributes to instability, but cannot explain the full drop from the linear control to the current encoder.

## Limits

This is a signal diagnostic, not a new confirmatory benchmark. The five outer folds contain only 65 DSS events, so fold variance and direction changes must be retained in interpretation. No patient-level predictions are included in the repository.
