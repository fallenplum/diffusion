# Phase6A — TITAN CRC morphology semantic preflight

Run status: **COMPLETE**.

## 1. Frozen official slide/text representation

- Successful CONCH slides: **519/519**; patients: **512/512**.
- Original UNI2 inventory has 520 slides. One unavailable CONCH DX1 is retained as missing; the same patient's original matched DX2 is available. No patch or patient substitution.
- Native and projected semantic dimensions: 768 / 768; finite embeddings: True.
- Native slide norm range: [10.906086921691895, 17.20969009399414]; peak CUDA allocation: 6.285 GiB.
- Maximum official zero_shot probability agreement error: 1.49e-08.
- Native output of encode_slide_from_patch_features is NOT text-aligned by itself. Apply the official vision_encoder.proj and L2 normalization, then compute raw cosine scores. Official zero_shot softmax is used solely as an equivalence check.
- All patches and original coordinates are used; coordinate scale is read from each H5. Native patient representation: normalized arithmetic mean of slide embeddings. Semantic patient representation: normalized equal-weight mean of normalized projected slides.
- Exact weights and official source revision are pinned. Runtime uses CUDA torch 2.5.1+cu124 instead of documented torch 2.0.1; Transformers/timm/einops match the official listed versions. Tokenizer and recursive-import I/O are routed to identical local pinned source files. Allocation mode is documented in allocation_equivalence.json.

## 2. Outcome-blind concept variation

- Fixed 32 CRC morphologic/clinicopathologic concepts and two text templates; no molecular pathway vocabulary, DSS concept selection, prompt tuning, or cohort-fitted semantic projector.
- Geometry gate: **PASS_NONCOLLAPSED**.
- Covariance-spectrum entropy effective rank: **4.4883**; first component fraction: 0.4631.
- Correlation-spectrum effective rank: 4.3994; concept standard deviation range: [0.07756616920232773, 0.148436039686203].
- Median absolute interconcept correlation: 0.5526; constant concepts: 0.
- Preregistered hard stop: raw covariance effective rank <=1.5 or no patient variation. Text geometry is also reported; correlated descriptors alone do not validate individual morphologic annotations.

## 3. Patient-level TNM/Stage information audit

| Endpoint | Valid patients | Excluded | Class counts |
|---|---:|---:|---|
| T | 511 | 1 | {'T2': 94, 'T4': 56, 'T3': 343, 'T1': 18} |
| N | 511 | 1 | {'N0': 293, 'N1': 127, 'N2': 91} |
| M | 452 | 60 | {'M0': 383, 'M1': 69} |
| Stage | 501 | 11 | {'I': 94, 'III': 151, 'II': 184, 'IV': 72} |

- Only patient ID and T/N/M/Stage columns are loaded into experiment computations; historical 512-patient cohort membership is preserved.
- Three seeds x five patient folds; identical folds across representations; three-fold inner selection over four fixed regularization strengths. Scaling and PCA are fitted only on each training partition.
- Seed metrics are computed after concatenating the five test folds. Patient OOF probabilities and fold assignments are persisted outside Git and mirrored to the local E-drive cache.

### Three-seed OOF balanced accuracy (mean +/- seed SD)

| Endpoint | TITAN native | CRC concepts | TITAN PCA32 | Prior only |
|---|---:|---:|---:|---:|
| T | 0.3927 +/- 0.0158 | 0.4391 +/- 0.0210 | 0.4103 +/- 0.0241 | 0.2500 +/- 0.0000 |
| N | 0.4314 +/- 0.0095 | 0.4599 +/- 0.0144 | 0.4689 +/- 0.0139 | 0.3333 +/- 0.0000 |
| M | 0.6626 +/- 0.0096 | 0.6403 +/- 0.0245 | 0.6436 +/- 0.0112 | 0.5000 +/- 0.0000 |
| Stage | 0.3535 +/- 0.0124 | 0.3849 +/- 0.0099 | 0.3695 +/- 0.0050 | 0.2500 +/- 0.0000 |

### Three-seed OOF macro one-vs-rest AUROC (mean)

| Endpoint | TITAN native | CRC concepts | TITAN PCA32 |
|---|---:|---:|---:|
| T | 0.6618 | 0.6654 | 0.6559 |
| N | 0.6330 | 0.6287 | 0.6482 |
| M | 0.7433 | 0.6934 | 0.7261 |
| Stage | 0.6224 | 0.6232 | 0.6408 |

### Paired concept-minus-comparator OOF differences

| Endpoint | Comparator | Seed deltas | Paired mean | 95% interval |
|---|---|---|---:|---|
| T | TITAN_embedding | +0.0523, +0.0424, +0.0444 | +0.0464 | [-0.0028, +0.0935] |
| T | TITAN_embedding_PCA32 | +0.0423, +0.0378, +0.0063 | +0.0288 | [-0.0150, +0.0733] |
| T | prior_only | +0.2120, +0.1707, +0.1844 | +0.1891 | [+0.1271, +0.2480] |
| N | TITAN_embedding | +0.0432, +0.0198, +0.0227 | +0.0286 | [-0.0107, +0.0662] |
| N | TITAN_embedding_PCA32 | +0.0132, -0.0202, -0.0198 | -0.0089 | [-0.0406, +0.0204] |
| N | prior_only | +0.1379, +0.1105, +0.1314 | +0.1266 | [+0.0828, +0.1712] |
| M | TITAN_embedding | -0.0087, -0.0382, -0.0199 | -0.0223 | [-0.0665, +0.0233] |
| M | TITAN_embedding_PCA32 | +0.0146, -0.0176, -0.0070 | -0.0033 | [-0.0505, +0.0423] |
| M | prior_only | +0.1616, +0.1136, +0.1458 | +0.1403 | [+0.0867, +0.1957] |
| Stage | TITAN_embedding | +0.0314, +0.0094, +0.0534 | +0.0314 | [-0.0042, +0.0667] |
| Stage | TITAN_embedding_PCA32 | +0.0139, +0.0027, +0.0295 | +0.0154 | [-0.0162, +0.0463] |
| Stage | prior_only | +0.1373, +0.1239, +0.1434 | +0.1349 | [+0.0952, +0.1726] |

Clinical information gate: **PASS_CLINICAL_INFORMATION**; passing targets: ['T', 'N', 'M', 'Stage'].
Gate requires concept-minus-prior balanced accuracy >0 for all three seeds and positive lower bound of a familywise 98.75% paired patient-bootstrap interval. Paired bootstrap resamples the same patients across all seeds (2,000 draws); it does not refit models or provide external-cohort uncertainty.
Repeated seeds reuse the same cohort and are not three independent validation datasets.

Endpoints with a positive 95% paired lower bound versus native TITAN: **[]**.
Concept balanced accuracy averages are higher for T/N/Stage and lower for M. All four native-comparator intervals include zero in this run; superiority over generic TITAN is not established. The PCA32 control also prevents attributing all benefits to medical semantics, and AUROC does not show a consistent concept advantage.

## 4. Interpretation and limits

- Advantage: a pretrained and verified pathology-language space is available without raw SVS or fitting a small-cohort UNI2-to-text projector; native TITAN signal is preserved as a comparator.
- Risk: frozen cosine descriptors compress native representation; any predictive advantage can reflect regularization/sample efficiency rather than additional information. The training-only PCA32 control helps distinguish semantic prior from dimensional compression.
- Noncollapse does not prove concept correctness. TNM/Stage prediction does not prove DSS benefit, causality, or that every named feature is visible in every diagnostic slide. Independent pathology annotation and cohort validation remain necessary.
- TNM/Stage are cross-sectional disease-state labels here; this audit does not establish longitudinal progression prediction.
- No prompt/config revision from these targets; no survival model, Omics branch, or CTF tuning is executed in this phase.
- Execution disclosure: an administrative preview displayed one raw clinical row during setup. The experimental readers load only the five permitted patient/TNM/Stage columns; no survival value enters concept definition, model fitting, or evaluation.
- Confidence: high for verified technical/geometry evidence; clinical estimates are limited to repeated OOF evaluation within this cohort.

## Sources

- [TITAN official model and usage](https://huggingface.co/MahmoodLab/TITAN)
- [Official zero-shot example](https://github.com/mahmoodlab/TITAN/blob/9e34c66ff66445c6c590da0dbf7acc103d39a40b/notebooks/zeroshot_demo.ipynb)
- [ICCR colorectal cancer pathology dataset](https://www.iccr-cancer.org/datasets/published-datasets/digestive-tract/colorectal/)
