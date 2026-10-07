# Phase6B RNA-conditioned TITAN concepts

This directory is reserved for the formal 3-seed × 5-outer-fold audit. The
server run uses the frozen Phase6A TITAN output cache and the server-side RNA
matrix. It reads only the precomputed T/N/M/Stage target table and never loads
survival columns.

The dynamic arm is deliberately named **output-space prompt adaptation**. It
keeps TITAN image/text encoders frozen and learns a 16-dimensional RNA adapter,
a 16×768 basis, and a 32-concept gate. This is a low-capacity mechanism probe;
it does not claim to be token-level TITAN prompt tuning.

The primary interpretation test is the pair of held-out comparisons:

1. `rna_conditioned` versus `static_concepts_plus_rna`;
2. matched RNA conditioning versus the same model under wrong-patient RNA.

If the first comparison is null, RNA is only an additive signal. If the second
comparison is null, the patient-specific interpretation mechanism is not
supported by this audit.
