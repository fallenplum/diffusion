# CompactSplineKAN implementation audit

## Fixed implementation

- Layer formula: `base_weight * SiLU(x) + spline_coeff * B_m(x) + bias`.
- Basis: Cox–de Boor B-spline recursion, fixed clamped range `[-3, 3]`.
- `grid_size=5`, `spline_order=3`, no dynamic grid update.
- Requested hidden width was 48. Its fusion-head parameter count was 58,769 for MCAT and 72,593 for CommonOnly, versus matched MLP 11,681 and 13,729; ratios 5.031x and 5.288x, outside the prespecified 1.2x ceiling.
- Frozen hidden width before formal training: `8`. Final fusion-head counts are 12,649 (MCAT) and 14,953 (CommonOnly), ratios 1.083x and 1.089x versus matched MLP, within the prespecified 0.8x–1.2x window.

## Numerical checks

- Local forward/backward finite test: PASS.
- Fixed boundary and out-of-range clamping test: PASS.
- Same-seed repeated initialization: PASS.
- Formal-run profile sample A2: fusion head `12649` parameters; B2: `14953` parameters.
- Formal-run matched MLP profile sample A1: `11681`; B1: `13729`.

No optimizer, endpoint, split, preprocessing, or hidden width was changed after observing test-fold results.
