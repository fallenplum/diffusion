# Phase18A — Local State Readout × Patient Conditioning

## 判定与适用范围

**STOP_CUSTOM_S6_READOUT_NOT_SUPPORTED**。本轮严格限定为读出 × 条件机制的探索性 2×2；常规逐位置 SSM 读出不是原创算法。

训练来源 `3eafc2c455e53e71adc39fa305952ad616d7d87f235d56a894502c4e4a3572a0`，父提交 `760a6a88feb9b731f8a935c98a8a79c4ca300943`。正式新增 **30 inner + 30 outer = 60 次神经网络拟合**；标签突变诊断另有 **4 次真实短拟合、8 次神经网络 optimizer steps**。历史60个主要 inner/outer、15个A3 outer及15个CONCH32 outer，共90次引用，历史神经网络重训0次。

锚点审计另计：父Data在完整合格的75次历史回放、60次历史冻结probe与60次新checkpoint回放中，确定性重解同一Clinical+S32 Ridge anchor共195次，并强制与原checkpoint逐元素/bitwise一致。它们不更新历史神经网络、不生成新OOF或模型选择；不能写成“审计完全没有优化”。195仅覆盖该完整合格验证链，不是包含正式/短拟合内部及superseded流程的全部底层solver调用总数。

## DSS 主结果

| arm | Harrell | Uno60 |
|---|---|---|
| F0 | 0.787266 | 0.781786 |
| F1 | 0.790559 | 0.782817 |
| L0 | 0.785462 | 0.773249 |
| L1 | 0.785305 | 0.773314 |
| ANCHOR_CS | 0.814161 | 0.783318 |
| R1_A3_noDirect | 0.798479 | 0.794478 |
| HIST_CS_S32_CONCH32 | 0.825923 | 0.806707 |

原512人、65个DSS events；每seed先汇总五折同折可比较对分子/分母，再平均3seed C值。不是跨折平均风险，也不是将15个cell称为seed mean。历史CONCH32为不同视觉/训练协议的强参照，不能据其数值作同输入因果消融。

## 四个预锁定主要比较

| comparison | metric | delta | CI95_low | CI95_high | Bonferroni_low | Bonferroni_high |
|---|---|---|---|---|---|---|
| L0-F0 | Harrell | -0.0018035 | -0.0276616 | 0.0223157 | -0.0398643 | 0.0314567 |
| L0-F0 | Uno60 | -0.00853764 | -0.0562809 | 0.0340401 | -0.0793243 | 0.0532722 |
| L1-F1 | Harrell | -0.00525367 | -0.031474 | 0.0185093 | -0.0442396 | 0.0279251 |
| L1-F1 | Uno60 | -0.00950299 | -0.0575091 | 0.0341971 | -0.0809985 | 0.0541786 |
| L1-L0 | Harrell | -0.000156826 | -0.00173786 | 0.00153737 | -0.00252454 | 0.00239385 |
| L1-L0 | Uno60 | 6.49859e-05 | -0.000901453 | 0.00117602 | -0.0013631 | 0.00180532 |
| DID=(L1-L0)-(F1-F0) | Harrell | -0.00345017 | -0.00741546 | 0.000419737 | -0.00915006 | 0.00214032 |
| DID=(L1-L0)-(F1-F0) | Uno60 | -0.000965352 | -0.00894026 | 0.0101138 | -0.0128924 | 0.0197985 |

20,000次固定折内患者配对bootstrap，所有模型和seed共享患者计数，pair权重为count_i×count_j，Uno60使用对应outer-train的反向KM。DID每次重算四个完整C值后求差。主要家族8项，校正区间覆盖率99.375%；普通95%区间同时披露。这些区间条件于已训练模型，不包含完整重训练或反复研究选择方差。未检出差异不等于统计等效、无信息或否定所有读出机制。

实际有效draws=20000/20000，因零分母排除=0；抽样及分母审计见BOOTSTRAP_AUDIT.json。

### 逐 seed 方向与工程门

| comparison | seed | Harrell | Uno60 |
|---|---|---|---|
| L0-F0 | 31 | -0.0101153 | 0.0122918 |
| L0-F0 | 47 | 0.00588097 | -0.0108209 |
| L0-F0 | 83 | -0.00117619 | -0.0270838 |
| L1-F1 | 31 | -0.0143496 | 0.0111597 |
| L1-F1 | 47 | 0.00235239 | -0.0102515 |
| L1-F1 | 83 | -0.00376382 | -0.0294172 |
| L1-L0 | 31 | -0.000470478 | -2.74664e-05 |
| L1-L0 | 47 | 0.000235239 | 0.000242697 |
| L1-L0 | 83 | -0.000235239 | -2.02725e-05 |
| DID=(L1-L0)-(F1-F0) | 31 | -0.0042343 | -0.00113217 |
| DID=(L1-L0)-(F1-F0) | 47 | -0.00352858 | 0.000569464 |
| DID=(L1-L0)-(F1-F0) | 83 | -0.00258763 | -0.00233335 |

| comparison | metric | mean_delta | positive_seeds | direction_positive | Bonferroni8_positive |
|---|---|---|---|---|---|
| L0-F0 | Harrell | -0.0018035 | 1 | False | False |
| L0-F0 | Uno60 | -0.00853764 | 1 | False | False |
| L1-F1 | Harrell | -0.00525367 | 1 | False | False |
| L1-F1 | Uno60 | -0.00950299 | 1 | False | False |
| L1-L0 | Harrell | -0.000156826 | 1 | False | False |
| L1-L0 | Uno60 | 6.49859e-05 | 1 | True | False |
| DID=(L1-L0)-(F1-F0) | Harrell | -0.00345017 | 0 | False | False |
| DID=(L1-L0)-(F1-F0) | Uno60 | -0.000965352 | 1 | False | False |

L1 相对预锁定强参照的均值门（Clinical+S32锚点为输入子集，A3为同输入神经模型；历史CONCH32单列，不当同输入参照）：

| reference | metric | L1_minus_reference | no_mean_harm |
|---|---|---|---|
| ANCHOR_CS | Harrell | -0.028856 | False |
| ANCHOR_CS | Uno60 | -0.0100039 | False |
| R1_A3_noDirect | Harrell | -0.0131734 | False |
| R1_A3_noDirect | Uno60 | -0.0211645 | False |

- L0-F0 and L1-F1 are not both positive in both Harrell and Uno60
- L1-L0 and DID fail positive two-metric means and/or at least two positive seeds per metric
- L1 does not preserve both-metric mean performance against all locked strong references

### 预列强参照与正确/错误条件比较

| comparison | metric | delta | CI95_low | CI95_high |
|---|---|---|---|---|
| L0-ANCHOR_CS | Harrell | -0.0286991 | -0.0753116 | 0.0191609 |
| L0-ANCHOR_CS | Uno60 | -0.0100689 | -0.081229 | 0.0663632 |
| L1-ANCHOR_CS | Harrell | -0.028856 | -0.075694 | 0.0194776 |
| L1-ANCHOR_CS | Uno60 | -0.0100039 | -0.0810878 | 0.0666442 |
| L1-HIST_CS_S32_CONCH32 | Harrell | -0.0406179 | -0.07923 | -0.00276268 |
| L1-HIST_CS_S32_CONCH32 | Uno60 | -0.033393 | -0.0920039 | 0.0294632 |
| L0-R1_A3_noDirect | Harrell | -0.0130165 | -0.0405889 | 0.0124004 |
| L0-R1_A3_noDirect | Uno60 | -0.0212295 | -0.0703918 | 0.0224181 |
| L1-R1_A3_noDirect | Harrell | -0.0131734 | -0.0405203 | 0.0121867 |
| L1-R1_A3_noDirect | Uno60 | -0.0211645 | -0.0706356 | 0.0223687 |
| F1-F1__WRONG | Harrell | 0 | 0 | 0 |
| F1-F1__WRONG | Uno60 | 0 | 0 | 0 |
| L1-L1__WRONG | Harrell | -7.84129e-05 | -0.000457483 | 0 |
| L1-L1__WRONG | Uno60 | -3.33397e-05 | -0.000198939 | 0 |

该表是次要比较，完整次要家族及其校正区间见PAIRED_COMPARISONS.csv，不挤占或混淆主要八项家族。WRONG只换状态条件中的RNA/CNA，自身late证据不变；冻结推理不是重训消融。

## 状态与患者风险的读出诊断

所选checkpoint第一批32名训练患者、无标签/错误样本筛选。以下分别按inner/outer列出15个checkpoint的中位数；标量诊断已按checkpoint去重，未把两个位置目标重复当成两个模型。

### inner：所选 checkpoint

| arm | z_matched_wrong_relative_L2 | h_relative_L2 | matched_wrong_risk_MAE | wrong_context_relative_risk_drift | patient_z_variance | patient_z_mean_channel_variance | patient_SSM_risk_SD | patient_anchor_SD | patient_SSM_over_anchor_SD | gamma | isolated_SSM_g_gradient_L2 | isolated_SSM_RNA_gradient_L2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 0 | 0 | 0 | 0 | 39.335 | 27.5812 | 0.907368 | 0.427039 | 1.99424 | 0.112643 | 0 | 0 |
| F1 | 0.00065561 | 0.000596669 | 0.000554662 | 0.000543852 | 38.6411 | 27.3762 | 0.912343 | 0.427039 | 2.00517 | 0.112634 | 0.0082724 | 0.000102997 |
| L0 | 0 | 0 | 0 | 0 | 42.0693 | 19.5279 | 0.663017 | 0.427039 | 1.4572 | 0.11267 | 0 | 0 |
| L1 | 0.000502281 | 0.000505397 | 0.000420038 | 0.00049337 | 42.1801 | 19.4202 | 0.656335 | 0.427039 | 1.44251 | 0.11267 | 0.00130853 | 7.28511e-05 |

### outer：所选 checkpoint

| arm | z_matched_wrong_relative_L2 | h_relative_L2 | matched_wrong_risk_MAE | wrong_context_relative_risk_drift | patient_z_variance | patient_z_mean_channel_variance | patient_SSM_risk_SD | patient_anchor_SD | patient_SSM_over_anchor_SD | gamma | isolated_SSM_g_gradient_L2 | isolated_SSM_RNA_gradient_L2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 0 | 0 | 0 | 0 | 74.8343 | 21.5231 | 0.823498 | 0.456586 | 1.83271 | 0.111848 | 0 | 0 |
| F1 | 0.000469671 | 0.000412187 | 0.000546056 | 0.000505653 | 79.1305 | 21.4792 | 0.822559 | 0.456586 | 1.83062 | 0.111848 | 0.00679096 | 0.000155519 |
| L0 | 0 | 0 | 0 | 0 | 80.1727 | 16.1189 | 0.704409 | 0.456586 | 1.59217 | 0.112693 | 0 | 0 |
| L1 | 0.000478554 | 0.000555119 | 0.00042782 | 0.000514494 | 79.345 | 16.1923 | 0.709909 | 0.456586 | 1.61078 | 0.112694 | 0.00234376 | 0.000115313 |

`patient_z_*`先在同一患者内等权聚合各WSI，再统计患者间方差；旧`ssm_over_anchor_SD`等字段沿用WSI粒度，只作历史兼容，不冒充患者级值。z尺度/方差变化也会影响优化，不能将读出改善自动归因于纯信息损失。gamma与readout可缩放耦合，不凭gamma大小推断分支是否被忽略。

`isolated_SSM_*`以SSM风险为目标，隔离late与anchor且不使用生存标签；是真实路径敏感性探针，不是Efron训练损失的梯度。正确/错配状态差或梯度更大不构成推进门。

### 两种明确区分的位置梯度目标

| role | arm | position_probe_target | checkpoints | WSI_probe_instances | tail10_mass_median | first10_mass_median | position_entropy_median |
|---|---|---|---|---|---|---|---|
| inner | F0 | final_patient_risk | 15 | 15 | 0.648559 | 0.0387356 | 0.792467 |
| inner | F0 | terminal_state_squared_norm | 15 | 15 | 0.938958 | 3.73008e-06 | 0.611346 |
| inner | F1 | final_patient_risk | 15 | 15 | 0.638174 | 0.0326869 | 0.801241 |
| inner | F1 | terminal_state_squared_norm | 15 | 15 | 0.932051 | 5.64638e-06 | 0.621572 |
| inner | L0 | final_patient_risk | 15 | 15 | 0.0926187 | 0.0840669 | 0.980882 |
| inner | L0 | terminal_state_squared_norm | 15 | 15 | 0.931058 | 5.73504e-06 | 0.615903 |
| inner | L1 | final_patient_risk | 15 | 15 | 0.0928995 | 0.0841157 | 0.980525 |
| inner | L1 | terminal_state_squared_norm | 15 | 15 | 0.925592 | 8.90591e-06 | 0.624113 |
| outer | F0 | final_patient_risk | 15 | 27 | 0.592003 | 0.0411451 | 0.814925 |
| outer | F0 | terminal_state_squared_norm | 15 | 27 | 0.763916 | 0.000312407 | 0.690916 |
| outer | F1 | final_patient_risk | 15 | 27 | 0.565548 | 0.0406878 | 0.823408 |
| outer | F1 | terminal_state_squared_norm | 15 | 27 | 0.744216 | 0.000394017 | 0.704984 |
| outer | L0 | final_patient_risk | 15 | 27 | 0.0873373 | 0.0951046 | 0.980433 |
| outer | L0 | terminal_state_squared_norm | 15 | 27 | 0.792459 | 0.000253281 | 0.666116 |
| outer | L1 | final_patient_risk | 15 | 27 | 0.0862521 | 0.0946412 | 0.980747 |
| outer | L1 | terminal_state_squared_norm | 15 | 27 | 0.780544 | 0.00027958 | 0.670017 |

每checkpoint一个按原顺序选取的训练患者，纳入其全部WSI，最终风险为WSI等权患者风险；表中WSI_probe_instances是重复探针实例总数，不能称独立患者数量。terminal_state_squared_norm针对h_T的平方范数，final_patient_risk针对实际患者风险。位置仅为坐标几何顺序；末10%梯度质量不是末10%组织的因果生存贡献。各checkpoint的探针分布保存在READOUT_MECHANISM.csv，不能只报告一个代表病例。

真实探针去重覆盖（由每个checkpoint的fit[0]/fit[:32]及原owner映射独立重建，不公开患者/WSI标识）：

| arm | role | checkpoint_count | gradient_distinct_patient_count | gradient_distinct_WSI_count | gradient_WSIs_per_checkpoint_min | gradient_WSIs_per_checkpoint_max | variance_distinct_patient_count | variance_distinct_WSI_count | variance_WSIs_per_checkpoint_min | variance_WSIs_per_checkpoint_max |
|---|---|---|---|---|---|---|---|---|---|---|
| F0 | inner | 15 | 5 | 5 | 1 | 1 | 141 | 142 | 32 | 33 |
| F0 | outer | 15 | 2 | 3 | 1 | 2 | 46 | 48 | 32 | 34 |
| F1 | inner | 15 | 5 | 5 | 1 | 1 | 141 | 142 | 32 | 33 |
| F1 | outer | 15 | 2 | 3 | 1 | 2 | 46 | 48 | 32 | 34 |
| L0 | inner | 15 | 5 | 5 | 1 | 1 | 141 | 142 | 32 | 33 |
| L0 | outer | 15 | 2 | 3 | 1 | 2 | 46 | 48 | 32 | 34 |
| L1 | inner | 15 | 5 | 5 | 1 | 1 | 141 | 142 | 32 | 33 |
| L1 | outer | 15 | 2 | 3 | 1 | 2 | 46 | 48 | 32 | 34 |

15个checkpoint不等于15名独立梯度探针患者：seeds/folds可能重复同一病例。上表distinct数是在各arm×role内取并集，不应跨组再相加当独立人数；32人方差探针同样存在重复。位置梯度表描述这些固定训练样本上的模型探针，不是队列总体梯度分布或生存风险依赖比例。

## 优化预算与真实梯度

| arm | cells | selected_epoch_median | selected_epoch_min | selected_epoch_max | selected_epoch1_cells | epochs_ran_median | hit160 |
|---|---|---|---|---|---|---|---|
| F0 | 15 | 17 | 11 | 21 | 0 | 57 | 0 |
| F1 | 15 | 17 | 11 | 21 | 0 | 57 | 0 |
| L0 | 15 | 17 | 8 | 20 | 0 | 57 | 0 |
| L1 | 15 | 17 | 8 | 20 | 0 | 57 | 0 |

继承160上限、min_epoch1、EMA alpha0.2、patience40；未按外折成绩扩大预算。所选epoch可以为1，这可能使观察受初始尺度/正则及短训练影响，不能称纯信息恢复因果证据。hit160单元仍有预算截断风险；零触顶也不证明全局优化收敛。全部逐epoch训练/内层验证损失、Harrell和梯度保存在TRAINING_CURVES.csv，选中行另存SELECTED_EPOCH_DIAGNOSTICS.csv。fresh outer严格训练内层所选轮数。

本轮实际汇总：触顶=0/60，选epoch1=0/60。实际是否发生截断或极早选择以此计数为准，不能将协议允许的情况写成本轮已发生。

所选内层epoch的真实Efron梯度中位数（选中训练长度可能不同）：

| arm | g_attention_grad | delta_grad | B_condition_grad | write_grad | readout_grad | gamma_grad | gamma |
|---|---|---|---|---|---|---|---|
| F0 | 0 | 0 | 0 | 0 | 2.93853 | 5.26833 | 0.112643 |
| F1 | 0.00920921 | 0.00465325 | 0.0813482 | 0.020438 | 2.86316 | 5.18797 | 0.112634 |
| L0 | 0 | 0 | 0 | 0 | 2.64 | 4.73122 | 0.11267 |
| L1 | 0.00469667 | 0.000495746 | 0.0533827 | 0.00927061 | 2.64884 | 4.88033 | 0.11267 |

同epoch、同seed/fold的真实Efron梯度比较：

| comparison | epoch | group | paired_cells | left_median | right_median | nonzero_reference_cells | median_paired_ratio |
|---|---|---|---|---|---|---|---|
| L0-F0 | 1 | g_attention_grad | 15 | 0 | 0 | 0 | NA |
| L0-F0 | 5 | g_attention_grad | 15 | 0 | 0 | 0 | NA |
| L0-F0 | 10 | g_attention_grad | 15 | 0 | 0 | 0 | NA |
| L0-F0 | 20 | g_attention_grad | 15 | 0 | 0 | 0 | NA |
| L0-F0 | 40 | g_attention_grad | 15 | 0 | 0 | 0 | NA |
| L1-F1 | 1 | g_attention_grad | 15 | 0 | 0 | 0 | NA |
| L1-F1 | 5 | g_attention_grad | 15 | 6.29721e-05 | 6.4668e-05 | 15 | 0.914531 |
| L1-F1 | 10 | g_attention_grad | 15 | 0.000764708 | 0.000812219 | 15 | 0.849489 |
| L1-F1 | 20 | g_attention_grad | 15 | 0.0115738 | 0.0133469 | 15 | 1.09643 |
| L1-F1 | 40 | g_attention_grad | 15 | 0.242929 | 0.199651 | 15 | 0.823433 |
| L1-L0 | 1 | g_attention_grad | 15 | 0 | 0 | 0 | NA |
| L1-L0 | 5 | g_attention_grad | 15 | 6.29721e-05 | 0 | 0 | NA |
| L1-L0 | 10 | g_attention_grad | 15 | 0.000764708 | 0 | 0 | NA |
| L1-L0 | 20 | g_attention_grad | 15 | 0.0115738 | 0 | 0 | NA |
| L1-L0 | 40 | g_attention_grad | 15 | 0.242929 | 0 | 0 | NA |

完整多梯度组见MATCHED_EPOCH_GRADIENTS.csv。参照梯度为0时比值NA；不能把零初始化或无条件臂的预期零路径当错误。所选checkpoint隔离梯度、所选epoch训练梯度、同epoch比较是不同对象，不能用不同训练长度证明唯一的梯度竞争原因。

## 算力与并行

| fit_classification | role | arm | cells | selected_epoch_median | cell_seconds_median | cell_seconds_sum | peak_GPU_MiB_max | parameters | last_step_gradient_parameters_median |
|---|---|---|---|---|---|---|---|---|---|
| historical_reference | inner | F0 | 15 | 17 | 81.4169 | 1206.38 | 5687.37 | 170266 | 152089 |
| historical_reference | inner | F1 | 15 | 17 | 78.7811 | 1203.66 | 5746.22 | 170266 | 164122 |
| historical_reference | outer | F0 | 15 | 17 | 19.1087 | 287.505 | 6014.45 | 170266 | 152473 |
| historical_reference | outer | F1 | 15 | 17 | 18.3598 | 267.598 | 6070.84 | 170266 | 164506 |
| historical_secondary_reference | outer | R1_A3_noDirect | 15 | 14 | 18.6743 | 269.18 | 6030.61 | 170266 | 166809 |
| new_fit | inner | L0 | 15 | 17 | 140.101 | 1960.27 | 2970.6 | 170266 | 152089 |
| new_fit | inner | L1 | 15 | 17 | 139.711 | 1970.54 | 3103.55 | 170266 | 164122 |
| new_fit | outer | L0 | 15 | 17 | 47.4662 | 676.535 | 3336.96 | 170266 | 152473 |
| new_fit | outer | L1 | 15 | 17 | 48.4972 | 686.337 | 3466.88 | 170266 | 164506 |

peak_GPU_MB沿用旧字段名，实际单位MiB（bytes/2^20）。父历史训练时间只作历史描述，不计本轮训练成本。并行下cell_seconds_sum不是墙钟时间，也不是隔离速度排名。局部/终态读出参数相同，但计算路径不同，不宣称FLOPs完全一致。

| role | samples | sample_span_seconds | workers_limit_max | running_max | GPU_percent_interval_weighted | GPU_percent_median | GPU_percent_max | memory_MiB_interval_weighted | memory_MiB_max |
|---|---|---|---|---|---|---|---|---|---|
| inner | 79 | 787.951 | 6 | 6 | 95.8009 | 100 | 100 | 17815.5 | 21120 |
| outer | 30 | 293.128 | 6 | 6 | 96.1367 | 100 | 100 | 19134 | 23058 |

GPU_PROFILE以调度器10秒级采样相邻时间区间作左值保持加权，只覆盖正式inner/outer队列的采样跨度；整卡显存/利用率可能包含并发进程，不等于单模型显存或完整端到端时长。未为占显存制造无效计算。

## PFI：冻结 DSS 风险迁移

可用510人、129个事件；未以PFI训练或选择本轮配置。

| arm | Harrell | Uno60 |
|---|---|---|
| F0 | 0.687263 | 0.665504 |
| F1 | 0.690566 | 0.668375 |
| L0 | 0.68241 | 0.672448 |
| L1 | 0.682124 | 0.67303 |
| ANCHOR_CS | 0.711978 | 0.670275 |
| R1_A3_noDirect | 0.69126 | 0.673692 |
| HIST_CS_S32_CONCH32 | 0.720666 | 0.683313 |

PFI不是独立外部队列；其终点与DSS不同，不能用有利PFI结果改写预锁定DSS决策。

## 实现、资格及审计证据

| gate | status |
|---|---|
| INDEPENDENT_VALIDATION.json | PASS |
| CHECKPOINT_REPLAY.json | PASS |
| HISTORICAL_REUSE_AUDIT.json | PASS |
| OPERATOR_PARITY.json | PASS |
| DATA_LEAKAGE_AUDIT.json | PASS |
| HISTORICAL_READOUT_PROBE_AUDIT.json | PASS |
| BOOTSTRAP_AUDIT.json | PASS |
| FIT_AND_REFERENCE_COUNTS.json | PASS |
| EVALUATION_MATH_AUDIT.json | PASS |
| CONTRAST_POINT_AUDIT.json | PASS |
| PARENT_PUBLISHED_METRIC_PARITY.json | PASS |
| PROBE_COVERAGE_AUDIT.json | PASS |

27项算子/模型测试加5项独立评估/公开隐私回归测试，共32项通过；CPU/CUDA逐位置输出、输入/参数梯度oracle通过，N=1、padding、tiny dt、非零h0、gate=0等已覆盖。全部60新checkpoint与75历史checkpoint四条件重放最大误差=0。独立anchor重建、stationarity及跨arm/seed逐元素一致性通过；指标独立核对sksurv，历史逐seed和三seed均值与已发表父结果核对。主比较点估计及DID独立审计PASS，共享bootstrap的每折患者计数SHA已保存；公开位置探针已剔除单患者raw target_value。每项检查只证明各自边界，不构成生物机制确认。

外折标签突变检查本次合格对的参数、inner风险、所选epoch和完整训练历史一致；审计过程中先前两次已完成的短拟合因bool比较helper报错而追加重做，累计四次如实记录。未保存smoke检查点或患者风险。首个历史probe批次完成4条记录后中止，以补充患者/WSI定义，旧不完整批次被标记superseded；后续完整60个冻结probe补齐。它们不更新神经网络参数，不算神经网络训练失败或新拟合；Data中的anchor验证重解已单列。详见FAILURE_LOG.md。

## 优势、限制与结论解释

优势：只改读出公式的2×2、无新增参数；历史来源与软件资格核验、完整checkpoint重放、患者级完整风险集Efron及独立指标/配对DID；患者风险与状态目标、患者/WSI粒度及新拟合/引用分别披露。

风险：65个DSS事件及反复开发的单队列；inner epoch选择噪声、潜在预算截断；读出尺度与优化条件同时变化；空间顺序无天然医学时间；patch只做geometry-only对齐，无原图形态核验。外部资源目前仍是资格清单，不能提升本轮证据等级。

**最终执行判断：STOP_CUSTOM_S6_READOUT_NOT_SUPPORTED。** 置信度：算子及本轮数值/计数正确性高；将观测归结为唯一生物学原因的置信度低。本轮未证明RNA无条件剩余信号，也未证明所有Mamba或读出方法无效。完成Phase18A后停止；不自动追加Phase18B、归一化、调幅或第四种读出。

### 读出瓶颈问题的直接回答

1. **读出作用**：无条件 L0−F0 为Harrell -0.001803 / Uno60 -0.008538，有条件 L1−F1 为Harrell -0.005254 / Uno60 -0.009503。按预锁定规则，读出方向一致门=未通过；这只界定本次公式与训练族的工程推进，不能推广为所有局部读出无效/有效。

2. **条件机制作用**：局部读出下 L1−L0 为Harrell -0.000157 / Uno60 +0.000065，DID 为Harrell -0.003450 / Uno60 -0.000965。方向/seed门=未通过，强参照均值门=未通过；应同时看八项区间，不能以单个有利差值代替机制证据。

3. **条件是否到达实际读出**：inner所选checkpoint的正确/错分子条件 z 相对差中位数，F1=0.00065561、L1=0.000502281；终态 h_T 相对差则为0.000596669与0.000505397。这是不同重训模型的冻结敏感性，不是把同一checkpoint替换读出后的纯因果信息损失实验。

4. **优化/尺度替代解释**：F1/L1所选inner epoch中位数分别17/17，选epoch1的cell分别0/0；outer训练探针患者级SSM/anchor风险SD比中位数为1.83062/1.61078。训练长度与尺度变化必须同读出结果一起解释；本轮没有据外折结果追加归一化或调幅。

5. **空间梯度扩散不等于预测改善**：冻结outer模型的实际患者风险目标，末10%位置梯度质量中位数从F1的0.565548变为L1的0.0862521。这些探针仅覆盖2名不同训练患者、3张不同WSI，不能泛化为队列比例或医学组织重要性。此实现确实改变了探针上的梯度位置分配，但未满足方向一致的预测收益门；因此不能据此把终态遗忘认定为已解决的唯一瓶颈。

**拓展思考／你可能想知道：** 若读出改善而条件增量未建立，允许保留局部读出作为普通对照；这与提出新的患者特异性交互机制是两个独立的证据问题。
