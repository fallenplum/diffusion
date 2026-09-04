# Data Source Audit

- Pathology: `TCGA/TCGA-COAD/*.h5`, `TCGA/TCGA-READ/*.h5`，逐个读取真实 `features`。
- RNA/CNA/Clinical: prompt 指定的 `data/data_8` 文件。
- Clinical supplement 仅用于 `HISTOLOGICAL_DIAGNOSIS` 和 `PRIMARY_SITE_PATIENT`。
- 旧 `outputs/features/uni2_h_patient_mean.npz` 未用于 pathology cohort 构建。
