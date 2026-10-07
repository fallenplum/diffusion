# Phase6B-R token-level TITAN prompt audit

This run implements continuous prompts before the frozen TITAN text
transformer. The prompt is inserted after the original BOS token and before
the locked CRC morphology concept tokens. S1 uses a patient-independent prompt
pool; S2 uses an RNA-conditioned mixture of the same pool; S4 evaluates the
matched S2 model with a deterministic wrong-patient RNA context.

The fixed three-epoch schedule is chosen before outer evaluation. No inner-val
labels are used for model or epoch selection, so upstream main effects remain
independent of held-out labels. Only T/N/M/Stage are read.
