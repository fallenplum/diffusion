# Phase M3R Confirmatory Metric & Mechanism Audit

## Decision

**Phase M3R COMPLETE**

- Primary metric: test-fold Harrell C aggregated over 15 seed×fold cells.
- Secondary metric: test-fold Uno C aggregated over 15 seed×fold cells.
- Global OOF C-index is diagnostic only.

## Confirmatory paired results

| Comparison | Δ Harrell mean ± std | Δ Uno mean ± std | Harrell wins | Uno wins | Both wins | Decision |
|---|---:|---:|---:|---:|---:|---|
| C1 Reverse vs C0 | -0.0041 ± 0.0253 | -0.0215 ± 0.0415 | 8/15 | 6/15 | 5/15 | mixed / not confirmed |
| C2 Bidirectional vs C0 | 0.0010 ± 0.0325 | -0.0030 ± 0.0340 | 7/15 | 8/15 | 6/15 | mixed / not confirmed |
| D1 Simple concat vs D0 | 0.0720 ± 0.0848 | 0.0571 ± 0.1011 | 12/15 | 9/15 | 8/15 | confirmed |

## Confirmatory absolute test-fold metrics

| Variant | Harrell mean ± std (15) | Uno mean ± std (15) |
|---|---:|---:|
| C0 MCAT-Full | 0.7806 ± 0.0680 | 0.7905 ± 0.0732 |
| C1 MCAT-ReverseCoAttention | 0.7765 ± 0.0603 | 0.7690 ± 0.0818 |
| C2 MCAT-BidirectionalCoAttention | 0.7816 ± 0.0814 | 0.7875 ± 0.0875 |
| D0 CAMR-Full | 0.7187 ± 0.0864 | 0.7431 ± 0.0959 |
| D1 CAMR-NoCMFM-CommonConcat | 0.7907 ± 0.0556 | 0.8002 ± 0.0605 |

## Re-aggregated M3 evidence

| Variant | Fold Harrell mean ± std (15) | Fold Uno mean ± std (15) |
|---|---:|---:|
| A0 Full MCAT | 0.7780 ± 0.0755 | 0.7871 ± 0.0924 |
| A3 Reverse | 0.7823 ± 0.0550 | 0.7843 ± 0.0697 |
| A4 Bidirectional | 0.7909 ± 0.0761 | 0.7967 ± 0.0917 |
| B0 Full CAMR | 0.7232 ± 0.1031 | 0.7302 ± 0.1020 |
| B3 CAMR-NoCMFM-CommonConcat | 0.7962 ± 0.0735 | 0.7886 ± 0.0773 |

### M3 re-aggregated paired audit

| Comparison | Δ Harrell | Δ Uno | Harrell wins | Uno wins | Both wins |
|---|---:|---:|---:|---:|---:|
| A3_vs_A0 | 0.0043 | -0.0028 | 7/15 | 5/15 | 3/15 |
| A4_vs_A0 | 0.0129 | 0.0096 | 11/15 | 9/15 | 9/15 |
| B3_vs_B0 | 0.0731 | 0.0585 | 13/15 | 12/15 | 12/15 |

## Required answers

### 1. A3 reverse vs Full MCAT

Decision: **A3_REVERSE_NOT_CONFIRMED**. C1 has 8/15 Harrell wins and 6/15 Uno wins, with mean deltas -0.0041 and -0.0215.

### 2. A4 bidirectional vs Full MCAT

Decision: **mixed / not confirmed**.

### 3. CAMR simple concat gain

Decision: **CMFM_OVERCOMPLEXITY_CONFIRMED**.

### 4. Phase M4 mechanism

Decision: **NO_CONFIRMED_MECHANISM; DO NOT PROCEED TO M4**.

No M4 implementation or design is included.
