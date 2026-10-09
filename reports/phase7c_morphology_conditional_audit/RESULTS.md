# Phase7C：独立形态标签、分子对应关系与 DSS 条件增量审计

本轮只使用实际存在的患者级病理字段；TITAN concept score 始终是预测变量，不是标签。RNA/CNA 状态和所有 scaler 均在外层训练患者上拟合。

## 1. 标签可用性

|项目|来源|有效人数|阳性人数|未知人数|
|---|---|---:|---:|---:|
|mucinous|supplementary.HISTOLOGICAL_DIAGNOSIS|505|66|7|
|lvi|supplementary.LYMPHOVASCULAR_INVASION_INDICATOR|478|193|34|
|vascular|supplementary.VASCULAR_INVASION_INDICATOR|462|114|50|
|perineural|supplementary.PERINEURAL_INVASION|217|56|295|
|grade|primary_sample.GRADE|0|—|512|
|tumor_budding|no corresponding field in locked clinical tables|0|—|512|
|stromal_indicator|no corresponding field in locked clinical tables|0|—|512|

GRADE、tumor budding 和 stromal indicator 没有可用字段，因此未进入模型，也没有用 TITAN 分数补标签。

## 2. 病理语义可靠性（冻结 TITAN）

|目标|concept|n|阳性|AUC|95% bootstrap CI|
|---|---|---:|---:|---:|---|
|mucinous|C10|505|66|0.7204|[0.6427,0.7912]|
|lvi|C02|478|193|0.6339|[0.5832,0.6841]|
|lvi|C03|478|193|0.5639|[0.5123,0.6143]|
|lvi|C02_C03_mean|478|193|0.6173|[0.5664,0.6673]|
|vascular|C03|462|114|0.5686|[0.5075,0.6274]|
|perineural|C04|217|56|0.5539|[0.4647,0.6423]|

## 3. 分子状态对真实形态标签的患者级 OOF 检验

|目标|模型|n|阳性|AUC|95% CI|BA|
|---|---|---:|---:|---:|---|---:|
|mucinous|N|505|66|0.4641|[0.3897,0.5338]|0.4670|
|mucinous|NS|505|66|0.5819|[0.4987,0.6667]|0.5761|
|mucinous|NR|505|66|0.6131|[0.5398,0.6804]|0.5655|
|mucinous|NCNA|505|66|0.5990|[0.5269,0.6721]|0.5614|
|mucinous|NMC|505|66|0.6836|[0.6082,0.7489]|0.6399|
|mucinous|NSMC|505|66|0.7014|[0.6251,0.7690]|0.6827|
|lvi|N|478|193|0.7946|[0.7536,0.8345]|0.7226|
|lvi|NS|478|193|0.8031|[0.7628,0.8406]|0.7139|
|lvi|NR|478|193|0.8080|[0.7669,0.8459]|0.7330|
|lvi|NCNA|478|193|0.7913|[0.7496,0.8317]|0.7131|
|lvi|NMC|478|193|0.7991|[0.7571,0.8400]|0.7295|
|lvi|NSMC|478|193|0.8090|[0.7699,0.8463]|0.7459|
|vascular|N|462|114|0.6006|[0.5370,0.6596]|0.5933|
|vascular|NS|462|114|0.5983|[0.5338,0.6606]|0.5789|
|vascular|NR|462|114|0.6020|[0.5382,0.6598]|0.5802|
|vascular|NCNA|462|114|0.5873|[0.5239,0.6504]|0.5833|
|vascular|NMC|462|114|0.5798|[0.5164,0.6394]|0.5863|
|vascular|NSMC|462|114|0.5795|[0.5166,0.6403]|0.5441|
|perineural|N|217|56|0.5830|[0.4928,0.6719]|0.5672|
|perineural|NS|217|56|0.5804|[0.4841,0.6734]|0.5699|
|perineural|NR|217|56|0.6003|[0.5083,0.6907]|0.5854|
|perineural|NCNA|217|56|0.5690|[0.4744,0.6577]|0.5734|
|perineural|NMC|217|56|0.5835|[0.4899,0.6788]|0.5582|
|perineural|NSMC|217|56|0.5816|[0.4892,0.6756]|0.5520|

### 相对临床基线 N 的分子增量与错误患者对照

|目标|比较|AUC/均值|95% CI 或 wrong-patient 分位区间|
|---|---|---:|---|
|mucinous|NR-N_delta (delta vs N)|+0.1489|[+0.0861,+0.2158]|
|mucinous|NCNA-N_delta (delta vs N)|+0.1349|[+0.0830,+0.1846]|
|mucinous|NMC-N_delta (delta vs N)|+0.2194|[+0.1490,+0.2892]|
|mucinous|NSMC-N_delta (delta vs N)|+0.2372|[+0.1625,+0.3163]|
|mucinous|NMC_wrong_patient_mean (wrong-patient)|+0.4600|[+0.4113,+0.5170]|
|mucinous|NSMC_wrong_patient_mean (wrong-patient)|+0.5021|[+0.4611,+0.5512]|
|lvi|NR-N_delta (delta vs N)|+0.0134|[-0.0038,+0.0307]|
|lvi|NCNA-N_delta (delta vs N)|-0.0032|[-0.0138,+0.0071]|
|lvi|NMC-N_delta (delta vs N)|+0.0046|[-0.0150,+0.0240]|
|lvi|NSMC-N_delta (delta vs N)|+0.0145|[-0.0074,+0.0358]|
|lvi|NMC_wrong_patient_mean (wrong-patient)|+0.7750|[+0.7582,+0.7964]|
|lvi|NSMC_wrong_patient_mean (wrong-patient)|+0.7834|[+0.7672,+0.8050]|
|vascular|NR-N_delta (delta vs N)|+0.0014|[-0.0223,+0.0254]|
|vascular|NCNA-N_delta (delta vs N)|-0.0133|[-0.0291,+0.0032]|
|vascular|NMC-N_delta (delta vs N)|-0.0208|[-0.0490,+0.0064]|
|vascular|NSMC-N_delta (delta vs N)|-0.0210|[-0.0507,+0.0087]|
|vascular|NMC_wrong_patient_mean (wrong-patient)|+0.5790|[+0.5503,+0.6092]|
|vascular|NSMC_wrong_patient_mean (wrong-patient)|+0.5792|[+0.5523,+0.6076]|
|perineural|NR-N_delta (delta vs N)|+0.0172|[-0.0301,+0.0664]|
|perineural|NCNA-N_delta (delta vs N)|-0.0140|[-0.0341,+0.0048]|
|perineural|NMC-N_delta (delta vs N)|+0.0005|[-0.0538,+0.0578]|
|perineural|NSMC-N_delta (delta vs N)|-0.0014|[-0.0553,+0.0535]|
|perineural|NMC_wrong_patient_mean (wrong-patient)|+0.5614|[+0.5158,+0.6093]|
|perineural|NSMC_wrong_patient_mean (wrong-patient)|+0.5562|[+0.5100,+0.6011]|

正确配对只有在分子特征仍有增量、而 wrong-patient 对照回落时，才支持患者级对应关系；若临床基线已很强，wrong-patient 可能仍保持较高 AUC，这不能被误读为分子匹配有效。99 次 wrong-patient 的逐次结果留在服务器私有输出，仓库只保留聚合值。

## 4. Clinical + 原始病理证据下的 DSS

|比较|三 seed Harrell 增量|99% CI|三 seed Uno 增量|稳定正向|
|---|---:|---|---:|---:|
|CUM-CU|-0.0049|[-0.0132,+0.0027]|-0.0009|False|
|CTM-CT|+0.0023|[-0.0184,+0.0233]|+0.0054|False|

## 决策边界

- 独立形态字段只支持对应概念的语义审计；不能推断其他概念或 patch 级真值。
- 分子对形态标签的 OOF 关联不等于因果，也不能把 TNM 代理结果替代真实形态标注。
- DSS 结果只判断当前冻结表示和低容量 Cox 头的条件增量；若区间跨零，不把点估计写成已验证机制。

## 可复核文件

- `L1_titan_independent_morphology.csv`
- `L2_oof_metrics.csv`（含 wrong-patient 聚合统计；逐次私有曲线留服务器）
- `L3_seed_oof_metrics.csv`、`L3_paired_increments.csv`
- `DONE.json`
