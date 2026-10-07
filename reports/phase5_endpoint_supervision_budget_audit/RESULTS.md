# Phase5 Endpoint & Survival-Supervision Budget Audit

- Fixed cohort: 512 patients; formal endpoints: DSS and OS.
- PFI has no explicit fields; PFS has valid_n=510 and was audited only.
- Three seed workers were run independently and merged after artifact validation.

|Endpoint|Model|Harrell OOF mean|Uno fold mean|
|---|---|---:|---:|
|DSS|B0-Clinical-only|0.7569|0.7530|
|DSS|B1-Pathology-only|0.6772|0.7149|
|DSS|B2-RNA-only|0.6565|0.6012|
|DSS|B3-Additive-C+P+R|0.7574|0.7611|
|DSS|B4-Frozen-M2-P+R|0.7494|0.7704|
|OS|B0-Clinical-only|0.7372|0.7315|
|OS|B1-Pathology-only|0.6291|0.6588|
|OS|B2-RNA-only|0.5832|0.5437|
|OS|B3-Additive-C+P+R|0.7316|0.7345|
|OS|B4-Frozen-M2-P+R|0.7361|0.7417|

- Decision: **DSS_REMAINS_PRIMARY**.
- Fold rows: 150/150; seed OOF rows: 30/30; event-budget rows: 405/405.
- Event-budget rows use mapping-only survival refits with frozen upstream representations; this limitation is retained in each row.
- Public reports contain aggregate metrics only; patient-indexed arrays and checkpoints remain private.

Validation status: **PASS**.
