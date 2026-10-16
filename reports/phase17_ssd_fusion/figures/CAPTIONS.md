# Figure notes

All figures derive from committed aggregate CSVs. Training curves use the fixed seed31/fold1 illustration. Gate and context diagnostics use the first32 training rows, independent of outcomes. State input gradients use one training WSI per checkpoint and target final-state squared norm, not survival risk. Seeds/folds reuse patients and are not independent cohorts. Reordering uses no outcome labels. Primary uncertainty is patient bootstrap; it does not capture external population shift or all training uncertainty. No figure is histological ground truth.
