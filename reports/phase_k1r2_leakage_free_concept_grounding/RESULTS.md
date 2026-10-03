# Phase K1R2 — Leakage-Free Teacher–Student Concept Grounding

- Decision: **C. TEXT_SEMANTICS_NOT_SUPPORTED**
- Leakage tests: **PASS**
- Student forward: `student_forward(pathology, text)`; no clinical, omics, or teacher input.
- Unseen cells: 120; positive real-minus-shuffled fraction: **0.533333**; mean delta: **0.004154901**.
- Real text better than concept-ID-only fraction: **0.483333**.
- Mean real minus shuffled-gene-teacher alignment: **-0.029417609**.
- Text query rank: **8**; text-query collapse: **False**.
- Real attention entropy: **0.476339**; real top-10% overlap: **0.649556**; real attention distance: **0.370726**.
- DSS, Clinical, and survival were not used; raw WSI/SVS was not used.

Only leakage-free grounding outputs are saved. Survival is not run in K1R2.
