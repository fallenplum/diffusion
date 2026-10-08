# Completed but invalidated: v1 token-prompt audit

This run finished 240 training cells with exit code 0. Its original `PASS`
and `DONE.json` describe computational completion only. The token-prompt
mechanism comparisons are invalidated by the post-run encoder audit:

1. The CLS padding mask retained the original token positions after four
   prompt tokens were inserted. This masked shifted concept/EOS tokens.
2. S1/S2 averaged raw template embeddings before normalization, whereas the
   official TITAN zero-shot ensemble normalizes each template first.

S0 and S3 do not call the custom text projector. Their OOF cells are retained
and reused after input, schedule, patient, label, class and epoch validation.
S1/S2 are rerun with the corrected text path. S4 predictions will also be
persisted privately for independent OOF auditing. No model selection or
hyperparameter changes are made from the v1 endpoint outcomes.

Original CSVs and computational completion metadata are retained here for
provenance, not as valid confirmatory mechanism evidence.
