# Omics pathway representation probe

## Protocol

- 512 patients, 65 DSS events, fixed five outer folds, three fixed seeds (`20260904`, `20260905`, `20260906`).
- RNA/CNA preprocessing is the audited M2R2 fold-local procedure. No Clinical field enters any forward path or selection step.
- `Pathway-RNA` uses the 50 fold-local Hallmark groups as sparse gene-to-pathway tokens, followed by one small transformer and gated pooling.
- `Pathway-RNA+CNA` adds four standardized CNA chunk tokens to the RNA pathway representation.
- `Signature-RNA+CNA` uses the six audited MCAT RNA+CNA signature groups as six omics tokens.

## Results

| Representation | Harrell C seed mean ± SD | Uno C seed mean ± SD | Parameters |
|---|---:|---:|---:|
| Existing Omics-only encoder | 0.5183 ± 0.0115 | 0.4822 ± 0.0263 | 86,401 |
| RNA-only linear Cox control | 0.6547 ± 0.1014 | 0.5916 ± 0.1672 | 512 |
| RNA+CNA linear Cox control | 0.6498 ± 0.1257 | 0.6101 ± 0.1407 | 768 |
| Pathway-RNA | 0.6282 ± 0.0882 | 0.5672 ± 0.1244 | 509,990 |
| Pathway-RNA+CNA | 0.5915 ± 0.1088 | 0.5476 ± 0.0970 | 542,119 |
| Signature-RNA+CNA | 0.5938 ± 0.0822 | 0.5109 ± 0.1563 | 477,666 |

## Interpretation

1. **Pathway prior improves the current encoder.** Pathway-RNA raises Harrell C by 0.1099 and Uno C by 0.0850 relative to the existing Omics-only encoder.
2. **The pathway representation does not yet recover the full linear RNA signal.** It remains 0.0265 Harrell and 0.0244 Uno below the RNA-only linear control.
3. **Adding CNA as generic chunk tokens reduces performance.** Pathway-RNA+CNA is lower than Pathway-RNA by 0.0367 Harrell and 0.0196 Uno, consistent with the earlier CNA direction-instability audit.
4. **The six-signature mixed representation is not a successful replacement in this protocol.** Its Uno score is close to the existing encoder and its fold variance is high.

## Decision

The first useful direction is **RNA-first pathway routing**: retain gene-to-pathway sparse connections and improve its small-sample regularization before adding a CNA branch. CNA should not be concatenated as generic chunks. A future CNA branch needs modality-specific pathway mapping, stronger shrinkage and an inner-fold gate or additive residual whose removal cannot damage the RNA branch.

This is a representation probe, not a claim that the cited papers are reproduced. The design is motivated by P-NET's biologically constrained sparse hierarchy, PASNet's pathway-associated sparse network and Pathformer's gene/pathway embedding with pathway crosstalk attention.
