# Phase M8A Results

1. Tied DSS event times: **yes**; tied groups=5, events inside ties=15. Historical implementation was **non-equivalent** to grouped Breslow.
2. Grouped Breslow audit: **PASS**.
3. M7R FiLM-vs-matched-residual under corrected objective: **NOT CONFIRMED** (A1 vs A2 mean ΔH=-0.000509, mean ΔU=-0.002114).
4. FiLM transfer to CommonOnly: **NOT SUPPORTED**.
5. CommonOnly-FiLM stably beats CommonOnly: **no**.
6. CommonOnly-FiLM beats matched residual: **no**.
7. CommonOnly-FiLM beats MCAT on primary Harrell C: **yes** (mean ΔH=0.013375).
8. Freeze CommonOnly + omics-conditioned pathology FiLM + clinical late fusion: **no**.
9. If Gate A fails, stop FiLM and return to CommonOnly failure analysis: **triggered**.

## Gate records

- Gate A: **FAIL**
- Gate B: **FAIL**
- Gate C: **FAIL**
