# Phase M3 MCAT Mechanism Audit

## Decision

**Phase M3 COMPLETE**

- Fixed 512 patients, 65 DSS events, original 5 folds and seeds 20260904/20260905/20260906.
- All 12 variants have 3 seeds x 5 folds = 180 completed cells.
- Preprocessing, Cox loss, clinical inputs, pathology sampling, optimizer, epoch limits and split were locked to M2R2.

## Main results

| Variant | Label | Harrell C seed mean +/- std | Uno C seed mean +/- std | Harrell C fold mean +/- std | Worst Harrell fold |
|---|---|---:|---:|---:|---:|
| A3 | MCAT-ReverseCoAttention | 0.7725 +/- 0.0125 | 0.7849 +/- 0.0269 | 0.7823 +/- 0.0550 | 0.6824 |
| A5 | MCAT-NoPathTransformer | 0.7636 +/- 0.0206 | 0.7748 +/- 0.0484 | 0.7809 +/- 0.0642 | 0.6977 |
| A4 | MCAT-BidirectionalCoAttention | 0.7635 +/- 0.0171 | 0.8063 +/- 0.0274 | 0.7909 +/- 0.0761 | 0.6582 |
| B3 | CAMR-NoCMFM-CommonConcat | 0.7626 +/- 0.0378 | 0.7973 +/- 0.0539 | 0.7962 +/- 0.0735 | 0.6760 |
| A2 | MCAT-NoCoAttention | 0.7529 +/- 0.0167 | 0.7917 +/- 0.0365 | 0.7819 +/- 0.0698 | 0.6556 |
| A0 | MCAT-Full | 0.7512 +/- 0.0287 | 0.7894 +/- 0.0421 | 0.7780 +/- 0.0755 | 0.6417 |
| A6 | MCAT-NoOmicTransformer | 0.7483 +/- 0.0180 | 0.8001 +/- 0.0367 | 0.7826 +/- 0.0729 | 0.6569 |
| A7 | MCAT-MeanPooling | 0.7476 +/- 0.0355 | 0.7979 +/- 0.0684 | 0.7812 +/- 0.0812 | 0.6468 |
| A1 | MCAT-NoSemanticGroups-EqualWidth | 0.7432 +/- 0.0158 | 0.8077 +/- 0.0361 | 0.7782 +/- 0.0591 | 0.6875 |
| B2 | CAMR-NoModalitySpecificBranch | 0.6871 +/- 0.0408 | 0.7131 +/- 0.0366 | 0.7211 +/- 0.1019 | 0.4770 |
| B0 | CAMR-Full | 0.6598 +/- 0.0644 | 0.7220 +/- 0.0393 | 0.7232 +/- 0.1031 | 0.4860 |
| B1 | CAMR-NoAdversarialAlignment | 0.6516 +/- 0.0483 | 0.7218 +/- 0.0387 | 0.7224 +/- 0.1036 | 0.4860 |

## Ablation questions

### 1. Is semantic six-signature grouping necessary?

- **A0 MCAT-Full**: Harrell C 0.7512 +/- 0.0287; Uno C 0.7894 +/- 0.0421.
- **A1 MCAT-NoSemanticGroups-EqualWidth**: Harrell C 0.7432 +/- 0.0158; Uno C 0.8077 +/- 0.0361.
A1 minus A0: Harrell C -0.0080; Uno C 0.0183. Thus semantic grouping contributes to the Harrell result in this protocol, while the Uno change has the opposite sign; the conclusion is metric-specific.

### 2. Is omic-to-path co-attention better than reverse or bidirectional attention?

- **A0 MCAT-Full**: Harrell C 0.7512 +/- 0.0287; Uno C 0.7894 +/- 0.0421.
- **A2 MCAT-NoCoAttention**: Harrell C 0.7529 +/- 0.0167; Uno C 0.7917 +/- 0.0365.
- **A3 MCAT-ReverseCoAttention**: Harrell C 0.7725 +/- 0.0125; Uno C 0.7849 +/- 0.0269.
- **A4 MCAT-BidirectionalCoAttention**: Harrell C 0.7635 +/- 0.0171; Uno C 0.8063 +/- 0.0274.
No for Harrell C under this protocol: no co-attention changes Harrell C by +0.0016, reverse changes it by +0.0213, and bidirectional changes it by +0.0123. Reverse is the strongest of these three on Harrell C; bidirectional has the highest Uno C.

### 3. Which Transformer stream matters more?

- **A5 MCAT-NoPathTransformer**: Harrell C 0.7636 +/- 0.0206; Uno C 0.7748 +/- 0.0484.
- **A6 MCAT-NoOmicTransformer**: Harrell C 0.7483 +/- 0.0180; Uno C 0.8001 +/- 0.0367.
Removing the path Transformer changes Harrell C by +0.0123; removing the omic Transformer changes it by -0.0029. The path Transformer is the larger influence in magnitude, but its removal improves Harrell C here; the omic Transformer removal is near-neutral with a slight decrease.

### 4. Is gated pooling necessary?

- **A7 MCAT-MeanPooling**: Harrell C 0.7476 +/- 0.0355; Uno C 0.7979 +/- 0.0684.
A7 minus A0: Harrell C -0.0036; Uno C +0.0085. Gated pooling gives a small Harrell advantage but is not uniformly better on Uno C.

### 5. What explains CAMR stability?

- **B0 CAMR-Full**: Harrell C 0.6598 +/- 0.0644; Uno C 0.7220 +/- 0.0393.
- **B1 CAMR-NoAdversarialAlignment**: Harrell C 0.6516 +/- 0.0483; Uno C 0.7218 +/- 0.0387.
- **B2 CAMR-NoModalitySpecificBranch**: Harrell C 0.6871 +/- 0.0408; Uno C 0.7131 +/- 0.0366.
- **B3 CAMR-NoCMFM-CommonConcat**: Harrell C 0.7626 +/- 0.0378; Uno C 0.7973 +/- 0.0539.
The largest reduction in Harrell seed standard deviation among B1-B3 is B3 (-0.0266 versus B0), and B3 also has the highest mean. This is an observed stability/result association, not a causal claim.

### 6. What is MCAT’s clearest structural limitation?

A1 is the only tested MCAT removal with a negative Harrell change relative to A0 (-0.0080); the other tested removals are neutral or improve Harrell C. This exposes dependence on semantic grouping for the full model’s Harrell result, while also showing that the default attention direction and path Transformer are not necessarily optimal under this protocol.

### 7. Which mechanism is the Phase M4 priority?

The most actionable mechanism to investigate is co-attention direction: A3 and A4 improve Harrell C over A0, with A3 highest. This is a Phase M4 investigation priority only; Phase M3 does not implement or propose a Phase M4 architecture.

## Reproducibility status

- Every row is `MECHANISM_ABLATION` of the M2R2 hand reimplementation; no official repository module was silently substituted.
- No seed hunting, resplitting, preprocessing change, Cox change or patient-level report output was used.
- GMM or other model warnings, if any, remain in worker logs and are not treated as completion evidence.
