# Evidence boundaries

- Each fitted model and its epoch/scale selection use only its corresponding outer-train partition. Clinical/RNA/CNA preprocessing is separately inner-train or outer-train fitted. Wrong-patient donors remain within each partition.
- The decision to expand the research branch into R2/R3 summarizes multiple inner partitions. Those partitions overlap and include patients who are outer-test in other folds. This is an exploratory research execution gate, **not** an unbiased nested-CV estimate of an entire adaptive research procedure. No outer prediction scores are used for that gate. Fixed-arm outer results and paired intervals are reported with this limitation; no external/confirmatory claim follows.
- Within each fixed arm, R2 scale selection remains local to the respective model/seed/outer-train; scale is never selected using other folds' validation scores or outer scores.
- Conditional gradients, parameter/state changes, branch risk contribution and predictive benefit are different checks. Larger gradients or larger state changes do not prove improved prognosis or biologically meaningful localization.
- Isolated SSM gradients are derivatives of SSM risk without outcome labels. Efron training gradients use training outcomes. The former do not show that outcome supervision learned useful complementary RNA information.
- Wrong-context, gamma-one and SSM-off frozen interventions are not retrained models. Stage gate comparisons against wrong context test inference sensitivity, not the performance of a separately retrained wrong-context model.
- Confidence intervals resample shared patients inside each original outer fold and condition on fitted checkpoints. They exclude full retraining variability, study selection and external population shift. Seeds are algorithmic repeats of the same cohort.
- No original WSI histological labels or visual localization are claimed. Coordinate ordering has no biological time meaning.
