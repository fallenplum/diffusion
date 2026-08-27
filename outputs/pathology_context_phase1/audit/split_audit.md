# Split audit

5-fold patient-level split exactly follows the historical baseline code: `StratifiedKFold(n_splits=5, shuffle=True, random_state=20260723)`. All slides inherit their patient's fold; train/test patient and slide overlap are zero.
