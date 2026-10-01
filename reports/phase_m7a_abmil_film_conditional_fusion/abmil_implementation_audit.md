# ABMIL implementation audit

The formal worker audit uses `src/phase_m7a/models.py:GatedABMIL`: `LayerNorm(1536) -> Linear(1536,96) -> GELU`, followed by `w^T(tanh(Vh) * sigmoid(Uh))` with hidden attention dimension 48, softmax over the unordered patch bag, weighted sum, and `LayerNorm(96)`. No positional or patch-order feature is present. This is a direct reimplementation of the exact equation prescribed by the M7A prompt; no prototype, adaptive pooling, or Phase1.5R2 implementation is imported. Omics reuses the existing src.phase_m2.mcat.SixSignatureEncoder without modification.
