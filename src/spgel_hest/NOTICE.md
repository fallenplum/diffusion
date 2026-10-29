# HEST numerical recipe reproduction notice

This directory adapts the fixed-feature expression-prediction recipe published
by the HEST contributors (Mahmood Lab). It does not present Ridge, PCA, Pearson
correlation or the upstream aggregation as a new algorithm.

Upstream: https://github.com/mahmoodlab/HEST

Pinned commit: `3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3`.

Audited source files and their Git blob identities:

- `src/hest/bench/benchmark.py`: `f7179acb4462dfc84b949eb16f8728f72387af6a`.
- `src/hest/bench/trainer.py`: `397649041ed227c82c61ea4ef67fb033d6550465`.
- `src/hest/bench/st_dataset.py`: `57655967cc9d540c315806cec31948aa2c62d23f`.

The upstream repository supplies the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International license. Its complete
`LICENSE.md` is preserved without alteration as `HEST_LICENSE.md` in this
directory (Git blob `183ee3ce13dcb521aa941f04b2d66d5a70d05c33`, SHA256
`32b8056672bc415bbd3829a9a737d776795f6b73af61ce0934107ed81ee8a7a0`).
These recipe adaptations retain that attribution and license.

Our changes provide explicit data/certificate/byte provenance, train-only
preprocessing guards, private estimator persistence, actual prediction replay,
and separate full-precision diagnostics. Undefined values are serialized as
JSON null with support counts instead of being imputed or omitted from the
official average. The official split-round4 then fold-aggregate-round4 boundary
is retained. Original upstream source snapshots used for numerical certification
remain in the local audit cache; this package is not an upstream distribution.

The image encoder is independently attributed to UNI2-h and TRIDENT; its model
and data terms are separate. This directory publishes no model weights,
expression matrices, RGB patches or spot-level identifiers/predictions.
