# Phase20 — Rich Omics Tokens × Selective Clinical Correction

## 判定

**NOT_SUPPORTED_IN_THIS_ARCHITECTURE**。表示、选择性机制、二者交互及实用预测增量分别判定，不能用未胜Clinical+S32替代机制检验。

当前审计source `e7b6383cfc23e4fbb8f94419a88bff5db1113babcb967672f87e0f13c74ba548`；135 inner + 135 fresh outer = 270个正式神经模型；20个继承凸参考仅校验、没有重新求解。

审计路径修复前后实际拟合source分别保留，原checkpoint及receipt未改写：

| actual_fit_source | completed_models |
| --- | --- |
| d1d7f8176f18417b8107767700f68ec0292181c3a0e76fce37894e06cc0c3192 | 189 |
| e7b6383cfc23e4fbb8f94419a88bff5db1113babcb967672f87e0f13c74ba548 | 81 |

D1在多WSI患者的anchor-only审计中出现浮点聚合identity失败；D2仅验证各WSI offset等于本人缓存后取首值，不改任何正式预测、训练梯度或epoch选择路径。135 inner + 54 outer = 189份D1模型经原SHA及全部正式模式逐元素兼容核验保留，剩余81份outer在D2继续训练；不是270份全新D2拟合。失败与中断的实际步骤另计，原记录保留，见审计修复文件及EXECUTION_COUNTS。

独立评估另修复一处历史LOCAL引用资格错误：原Phase16R的OOF来自独立重建特征后的float64风险重算，Phase19既有门为绝对误差≤1e−9；Phase20误用bitwise相等而中止。15cell原SHA、患者折、shape及float64均通过，最大差仅2.220446049250313e-16，全pair严格顺序和1e−8 tie类别也相同。仅恢复原历史引用规则并新增回归；不cast、不改OOF原值，当前270预测/anchor门及主检验代码不变，保留已通过的1740回放，只恢复统计。原失败日志、新旧代码与实际7+5测试见HISTORICAL_REFERENCE_REPAIR_AUDIT.json。

## DSS完整外折指标

| arm | Harrell_mean | Harrell_std | Harrell_min | Harrell_max | Uno60_mean | Uno60_std | Uno60_min | Uno60_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D16 | 0.7993413 | 0.002822865 | 0.7965185 | 0.8021642 | 0.7873721 | 0.007551646 | 0.780536 | 0.7954782 |
| F16 | 0.7958127 | 0.003512864 | 0.7918137 | 0.7984004 | 0.7837288 | 0.009569045 | 0.7732671 | 0.7920389 |
| HIST_CS_S32_CONCH32 | 0.8259233 | 0 | 0.8259233 | 0.8259233 | 0.8067066 | 0 | 0.8067066 | 0.8067066 |
| J0 | 0.8035756 | 0 | 0.8035756 | 0.8035756 | 0.7923395 | 0 | 0.7923395 | 0.7923395 |
| J_CS | 0.8141614 | 0 | 0.8141614 | 0.8141614 | 0.7833175 | 0 | 0.7833175 | 0.7833175 |
| L0 | 0.7980083 | 0.004035791 | 0.7936956 | 0.8016937 | 0.7818011 | 0.004336341 | 0.778729 | 0.7867614 |
| P1 | 0.7947934 | 0.002580488 | 0.7918137 | 0.7962832 | 0.7884361 | 0.004772413 | 0.785387 | 0.7939359 |
| P16 | 0.7994197 | 0.001765597 | 0.7984004 | 0.8014585 | 0.78816 | 0.006598701 | 0.7829551 | 0.7955816 |
| P16__FULL_OMICS_WRONG | 0.8002039 | 0.001086521 | 0.7995766 | 0.8014585 | 0.7856563 | 0.009412575 | 0.7773929 | 0.7959022 |
| P16__UNIFORM_PAIR | 0.7976162 | 0.001160405 | 0.7962832 | 0.7984004 | 0.792995 | 0.003060727 | 0.790809 | 0.796493 |
| P1__FULL_OMICS_WRONG | 0.8047518 | 0.01185799 | 0.7944013 | 0.81769 | 0.7904266 | 0.01317911 | 0.7775731 | 0.8039089 |
| P1__UNIFORM_PAIR | 0.7995766 | 0.001411433 | 0.7981651 | 0.800988 | 0.7936318 | 0.004006077 | 0.7909434 | 0.798236 |
| S1 | 0.7954207 | 0.003202474 | 0.7918137 | 0.7979299 | 0.7816272 | 0.00403349 | 0.7776851 | 0.7857463 |
| S16 | 0.7963616 | 0.002000682 | 0.7944013 | 0.7984004 | 0.7894597 | 0.006214798 | 0.7824736 | 0.7943739 |
| S16__CONSTANT_C | 0.7963616 | 0.002000682 | 0.7944013 | 0.7984004 | 0.7894597 | 0.006214798 | 0.7824736 | 0.7943739 |
| S16__CORRECTION_OFF | 0.7963616 | 0.002000682 | 0.7944013 | 0.7984004 | 0.7894597 | 0.006214798 | 0.7824736 | 0.7943739 |
| S16__WRONG_C | 0.7963616 | 0.002000682 | 0.7944013 | 0.7984004 | 0.7894597 | 0.006214798 | 0.7824736 | 0.7943739 |
| S16__WRONG_O | 0.7963616 | 0.002000682 | 0.7944013 | 0.7984004 | 0.7894597 | 0.006214798 | 0.7824736 | 0.7943739 |
| WC16 | 0.7976947 | 0.001469066 | 0.796048 | 0.7988709 | 0.7870071 | 0.005745529 | 0.7814154 | 0.792895 |
| WO16 | 0.7984788 | 0.001160405 | 0.7976947 | 0.7998118 | 0.7864015 | 0.002602328 | 0.7843876 | 0.7893398 |

每seed先拼完整五折并加总同折可比较对分子/分母，再平均3个seed的C。3seed不是独立患者队列，不平均患者风险。历史LOCAL输入/协议不同，只是强参照。

## 七项预锁定配对比较

| comparison | metric | delta | CI95_low | CI95_high | Bonferroni_low | Bonferroni_high | positive_seeds | positive_fold_cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P16-P1 | Harrell | 0.004626362 | -0.002844754 | 0.01244604 | -0.006540457 | 0.01654515 | 3 | 7 |
| P16-P1 | Uno60 | -0.0002761164 | -0.01511164 | 0.01164205 | -0.02769963 | 0.01885658 | 1 | 7 |
| S16-P16 | Harrell | -0.003058104 | -0.008089627 | 0.001111239 | -0.01101025 | 0.003091331 | 0 | 2 |
| S16-P16 | Uno60 | 0.001299698 | -0.005626867 | 0.01177823 | -0.009589614 | 0.02094355 | 1 | 3 |
| S16-D16 | Harrell | -0.002979691 | -0.007936562 | 0.001345067 | -0.01059873 | 0.003373153 | 0 | 2 |
| S16-D16 | Uno60 | 0.002087574 | -0.004806007 | 0.0115672 | -0.008488093 | 0.02048864 | 2 | 3 |
| S16-F16 | Harrell | 0.0005488905 | -0.002831541 | 0.003976716 | -0.00501359 | 0.006785449 | 1 | 3 |
| S16-F16 | Uno60 | 0.005730829 | -0.003299748 | 0.02507707 | -0.00655298 | 0.04598095 | 2 | 5 |
| S16-WC16 | Harrell | -0.00133302 | -0.005871563 | 0.003179521 | -0.00855888 | 0.005560065 | 1 | 1 |
| S16-WC16 | Uno60 | 0.002452554 | -0.006624552 | 0.01455645 | -0.01214521 | 0.02545543 | 2 | 3 |
| S16-WO16 | Harrell | -0.002117149 | -0.005896382 | 0.0009366505 | -0.008323557 | 0.002359439 | 1 | 1 |
| S16-WO16 | Uno60 | 0.003058172 | -0.005343153 | 0.01959088 | -0.009088072 | 0.03581859 | 2 | 2 |
| DID=(S16-P16)-(S1-P1) | Harrell | -0.003685407 | -0.009588877 | 0.00169034 | -0.01311801 | 0.004426308 | 0 | 4 |
| DID=(S16-P16)-(S1-P1) | Uno60 | 0.008108627 | -0.004163183 | 0.03086383 | -0.00764354 | 0.05213821 | 2 | 6 |

20,000次外折内患者bootstrap，共享所有模型/seed的患者计数；pair权重count_i×count_j，Uno60采用对应outer-train反向KM。DID每draw由四个完整C相减。14项家族校正区间覆盖率99.642857%。区间条件于已拟合模型，不包括完整重训和反复研究者选择。

### representation

**NOT_SUPPORTED_IN_THIS_ARCHITECTURE**；预算门=True；方向/seed门=False；全部校正区间为正=False。

### selective_clinical_correction

**NOT_SUPPORTED_IN_THIS_ARCHITECTURE**；预算门=True；方向/seed门=False；全部校正区间为正=False。

### representation_by_correction_interaction

**NOT_SUPPORTED_IN_THIS_ARCHITECTURE**；预算门=True；方向/seed门=False；全部校正区间为正=False。

### 选择性行为门

| direction | median_selected_inner_checkpoint_patient_mean_support | strict_maximum | behavior_gate_pass |
| --- | --- | --- | --- |
| po | 0.4558264 | 0.75 | True |
| op | 0.426206 | 0.75 | True |

仅取15个selected inner S16的all-fit label-blind探针；先WSI内query均值、患者内等WSI、患者等权，再取checkpoint中位数。两方向support(g>1e−6, eligible≥2)中位数均须严格<0.75；退化row与全部row支持另披露，不根据outer成绩修改。该门只证明操作上选择了部分边，不证明临床有效。

## 实用预测收益及冻结干预

| reference | metric | delta_S16_minus_reference | no_mean_harm |
| --- | --- | --- | --- |
| J0 | Harrell | -0.007213989 | False |
| J0 | Uno60 | -0.002879802 | False |
| J_CS | Harrell | -0.01779973 | False |
| J_CS | Uno60 | 0.006142118 | True |

| comparison | metric | family | delta | CI95_low | CI95_high | Bonferroni_low | Bonferroni_high | family_tests | positive_seeds | seeds | positive_fold_cells | fold_cells | draws_requested | draws_valid | statistical_object |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S16-S1 | Harrell | secondary | 0.0009409551 | -0.007002378 | 0.008849643 | -0.0117228 | 0.01282038 | 16 | 1 | 3 | 4 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| S16-S1 | Uno60 | secondary | 0.007832511 | -0.008007967 | 0.02810916 | -0.01744008 | 0.04698475 | 16 | 3 | 3 | 9 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| S16-L0 | Harrell | secondary | -0.001646671 | -0.01220427 | 0.007632592 | -0.0187586 | 0.01196212 | 16 | 1 | 3 | 5 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| S16-L0 | Uno60 | secondary | 0.007658517 | -0.006680105 | 0.02705127 | -0.01416268 | 0.0443663 | 16 | 2 | 3 | 7 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| S16-J0 | Harrell | secondary | -0.007213989 | -0.01888802 | 0.004268679 | -0.02510456 | 0.01089984 | 16 | 0 | 3 | 3 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| S16-J0 | Uno60 | secondary | -0.002879802 | -0.02342106 | 0.0179247 | -0.04065952 | 0.03443693 | 16 | 1 | 3 | 3 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| S16-J_CS | Harrell | secondary | -0.01779973 | -0.06219676 | 0.0264012 | -0.08349324 | 0.05114285 | 16 | 0 | 3 | 6 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| S16-J_CS | Uno60 | secondary | 0.006142118 | -0.06005225 | 0.0766741 | -0.09529124 | 0.1182137 | 16 | 2 | 3 | 9 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| S16-S16__WRONG_C | Harrell | secondary | 0 | 0 | 0 | 0 | 0 | 16 | 0 | 3 | 0 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| S16-S16__WRONG_C | Uno60 | secondary | 0 | 0 | 0 | 0 | 0 | 16 | 0 | 3 | 0 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| S16-S16__WRONG_O | Harrell | secondary | 0 | 0 | 0 | 0 | 0 | 16 | 0 | 3 | 0 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| S16-S16__WRONG_O | Uno60 | secondary | 0 | 0 | 0 | 0 | 0 | 16 | 0 | 3 | 0 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| S16-S16__CONSTANT_C | Harrell | secondary | 0 | 0 | 0 | 0 | 0 | 16 | 0 | 3 | 0 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| S16-S16__CONSTANT_C | Uno60 | secondary | 0 | 0 | 0 | 0 | 0 | 16 | 0 | 3 | 0 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| S16-S16__CORRECTION_OFF | Harrell | secondary | 0 | 0 | 0 | 0 | 0 | 16 | 0 | 3 | 0 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| S16-S16__CORRECTION_OFF | Uno60 | secondary | 0 | 0 | 0 | 0 | 0 | 16 | 0 | 3 | 0 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| P1-P1__FULL_OMICS_WRONG | Harrell | exploratory_representation | -0.009958441 | -0.02135198 | -0.0003631437 | -0.02759242 | 0.00349488 | 10 | 1 | 3 | 5 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| P1-P1__FULL_OMICS_WRONG | Uno60 | exploratory_representation | -0.001990522 | -0.01359828 | 0.009329403 | -0.01979433 | 0.01597706 | 10 | 1 | 3 | 6 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| P1-P1__UNIFORM_PAIR | Harrell | exploratory_representation | -0.004783188 | -0.01344545 | 0.003778744 | -0.01791175 | 0.008192434 | 10 | 0 | 3 | 4 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| P1-P1__UNIFORM_PAIR | Uno60 | exploratory_representation | -0.005195669 | -0.01971565 | 0.007607742 | -0.02812365 | 0.0142951 | 10 | 1 | 3 | 6 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| P16-P16__FULL_OMICS_WRONG | Harrell | exploratory_representation | -0.0007841292 | -0.00557789 | 0.004357799 | -0.007684021 | 0.00679778 | 10 | 0 | 3 | 5 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| P16-P16__FULL_OMICS_WRONG | Uno60 | exploratory_representation | 0.00250369 | -0.002661007 | 0.01073454 | -0.004388331 | 0.01733953 | 10 | 2 | 3 | 7 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| P16-P16__UNIFORM_PAIR | Harrell | exploratory_representation | 0.001803497 | -0.009454999 | 0.01343134 | -0.01496464 | 0.01986085 | 10 | 2 | 3 | 5 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| P16-P16__UNIFORM_PAIR | Uno60 | exploratory_representation | -0.004835026 | -0.02550924 | 0.01219103 | -0.04112193 | 0.02165211 | 10 | 1 | 3 | 6 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| DID_uniform | Harrell | exploratory_representation | 0.006586685 | -0.002167635 | 0.01592117 | -0.005949292 | 0.02062873 | 10 | 3 | 3 | 7 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |
| DID_uniform | Uno60 | exploratory_representation | 0.0003606434 | -0.01837685 | 0.01501386 | -0.03139407 | 0.02135026 | 10 | 1 | 3 | 6 | 15 | 20000 | 20000 | mean complete seed C after within-fold num/den sum; DID is four complete Cs in every draw |

S16的wrongC/wrongO只替换新增修正的context；本人main、两两主干及value保持匹配。WC16/WO16为真正重训对照，冻结S16干预不是重训。constantC使delta为零，与correction_off应逐元素一致；这两个相同干预不构成两份独立机制证据。off保留J0+pairwise，不是main-only；独立anchor模式另逐元素核验。

P1/P16另有冻结fullOmicsWrong（替换整个P–O分支的Omics，main/Clinical/visual仍匹配）及uniformPair（双向map变为valid均匀，values不变）。它们用于区分注意力交互与非线性组学主效应，不是重训消融。DID_uniform由四个完整C计算，作为单列探索；不追加到14项主要家族，也不将探索结果写成确认机制。

## 表示和机制边界

P1/P16使用相同16个learned signed gene-linear filters及encoder，P1仅经预固定rank1投影和幅度匹配广播；P16保留全16坐标。它们是监督训练的丰富表示，不是PCA；L0才是原scalar机制的新鲜参考。共同main冻结，Clinical修正最终风险有训练折anchor尺度下的显式上限；小logits本身不能保证小预测影响。

程序基因覆盖、有效表达秩、missingness/平台关联及真实query覆盖必须按审计披露。attention、gate或TITAN分数不是真实形态标签；错患者是机制干预控制，不是疾病因果实验。当前65 DSS事件、反复开发队列下，CI跨零不证明等效、无RNA信息或所有交互无效。

共同J0已经包含Clinical、RNA512及其他全部输入主效应；因此B检验的是冻结全输入强anchor之后额外Token容量的条件增量，没有重新检验独立RNA预后能力。Clinical没有新增修正head的直接通路且Cref严格归零，但仍可能通过slot及可区分value的重加权间接表达Clinical信息；即使配对控制通过，也不足以单独证明真实分子—形态学机制。

P16−P1同时改变RNA、CNA扩增与缺失三个通道，不能单独归因于RNA。rank1与完整摘要的功能、梯度及有效正则不同；各臂独立联合训练的两两主干也不同，S16−P16不是同一冻结checkpoint上的局部修正效应。F16仅gate固定Clinical参考，delta仍由本人Clinical条件化。J0在fit患者上的offset来自同fit拟合，不是交叉拟合teacher；本轮冻结offset且没有可学stacking权重，但不能将其称为cross-fitted anchor。

选择器的25%预算限制sum(g)，不保证仅25%的边非零。若实测支持接近全体边，不能称已定位少数关键Patch–Pathway关系；rank1的能量匹配也只对各向同性等方差filter输出具有名义解释，须同时报告实际响应SD/rank。

### 实测表示诊断（checkpoint中位数）

| arm | role | RNA_summary16_before_effective_rank | RNA_summary16_after_effective_rank | RNA_tokens_effective_rank | RNA_measured_response_map_rank_after_mean | summary16_empirical_after_before_energy_ratio | RNA_program_union_genes | RNA_genes_outside_program_union | RNA_anchor_beta_program_union_energy_fraction | RNA_anchor_beta_nominal_filter_span_energy_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | inner | 23.14191 | 3.362329 | 5.805671 | 1 | 2.490083 | 512 | 0 | 1 | 0.09688743 |
| P1 | outer | 22.44869 | 3.758482 | 5.833501 | 1 | 2.967317 | 512 | 0 | 1 | 0.06633518 |
| P16 | inner | 18.95465 | 18.95465 | 10.14083 | 16 | 1 | 512 | 0 | 1 | 0.3327701 |
| P16 | outer | 19.53558 | 19.53558 | 12.76128 | 16 | 1 | 512 | 0 | 1 | 0.3355688 |
| S1 | inner | 24.86818 | 3.363216 | 5.776337 | 1 | 2.472784 | 512 | 0 | 1 | 0.07174439 |
| S1 | outer | 22.52897 | 3.758747 | 5.917237 | 1 | 2.89202 | 512 | 0 | 1 | 0.06711505 |
| S16 | inner | 18.99476 | 18.99476 | 10.16571 | 16 | 1 | 512 | 0 | 1 | 0.3327537 |
| S16 | outer | 19.5331 | 19.5331 | 12.76515 | 16 | 1 | 512 | 0 | 1 | 0.3355629 |

这里的covariance effective rank不是输入Jacobian秩；rank1仅指每程序输入摘要瓶颈，非线性token输出仍可具有更高协方差秩。J0系数投影仅是名义all-genes-measured欧氏几何，不是已解释的DSS信息或患者级信息保真下界。filters先前受DSS训练，诊断不读取新生存标签不等于表征无监督。

### 实测修正诊断（checkpoint中位数）

| arm | role | anchor_risk_SD | pair_risk_SD | correction_risk_SD | correction_SD_over_anchor_SD | po_selectivity_support_patient_mean | op_selectivity_support_patient_mean | po_selected_base_probability_mass_mean | op_selected_base_probability_mass_mean | po_TV_over_rho_M_mean | op_TV_over_rho_M_mean | matched_wrongC_risk_MAE | matched_wrongO_risk_MAE | wrongO_eligible_pair_loss_fraction_mean | isolated_correction_clinical_per_patient_gradient_L2_mean | isolated_correction_RNA_per_patient_gradient_L2_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D16 | inner | 0.7219528 | 0.1433883 | 5.834632e-05 | 8.600082e-05 | 1 | 1 | 0.2985451 | 0.3315302 | 0.003606475 | 0.00622832 | 3.415216e-05 | 1.027304e-05 | 0 | 5.169373e-05 | 4.07015e-06 |
| D16 | outer | 0.6887884 | 0.1270081 | 9.68142e-05 | 0.0001405572 | 1 | 1 | 0.2814614 | 0.3971064 | 0.003878979 | 0.007124595 | 5.816141e-05 | 5.730681e-06 | 0 | 7.478862e-05 | 2.445177e-06 |
| F16 | inner | 0.7219528 | 0.1434302 | 4.150062e-05 | 5.956079e-05 | 0.4513626 | 0.4261385 | 0.3242449 | 0.4619742 | 0.003194234 | 0.00619726 | 1.813774e-05 | 6.886452e-06 | 0 | 2.611875e-05 | 5.100383e-06 |
| F16 | outer | 0.6887884 | 0.1269844 | 7.971956e-05 | 0.0001146102 | 0.4616629 | 0.4250684 | 0.3254094 | 0.5243997 | 0.003537544 | 0.006758787 | 4.324169e-05 | 7.369403e-06 | 0 | 4.433918e-05 | 2.980777e-06 |
| S1 | inner | 0.7219528 | 0.1713432 | 6.645621e-05 | 8.533854e-05 | 0.5076966 | 0.4215949 | 0.2982627 | 0.3426531 | 0.003984908 | 0.005673334 | 4.072339e-05 | 2.488421e-05 | 0 | 3.17984e-05 | 3.407497e-06 |
| S1 | outer | 0.6887884 | 0.1294139 | 6.355856e-05 | 9.345024e-05 | 0.4839333 | 0.4252628 | 0.4259217 | 0.4686599 | 0.003968823 | 0.007285167 | 4.415904e-05 | 1.789072e-05 | 0 | 5.959655e-05 | 3.217163e-06 |
| S16 | inner | 0.7219528 | 0.1434204 | 3.932655e-05 | 5.732352e-05 | 0.4558264 | 0.426206 | 0.3302658 | 0.3913495 | 0.003273892 | 0.005717098 | 2.072881e-05 | 1.15654e-05 | 0 | 4.357815e-05 | 6.201463e-06 |
| S16 | outer | 0.6887884 | 0.1269892 | 0.0001057125 | 0.0001554292 | 0.4657688 | 0.4248227 | 0.311975 | 0.5038422 | 0.003429256 | 0.006459201 | 5.325401e-05 | 7.297673e-06 | 0 | 6.911423e-05 | 1.563995e-06 |
| WC16 | inner | 0.7219528 | 0.1434081 | 0.0001167048 | 0.0001498644 | 0.4656706 | 0.4258303 | 0.3044755 | 0.502575 | 0.00318186 | 0.006986056 | 4.755006e-05 | 6.777445e-06 | 0 | 6.882778e-05 | 2.715721e-06 |
| WC16 | outer | 0.6887884 | 0.1270138 | 7.857942e-05 | 0.0001155354 | 0.4775658 | 0.4215693 | 0.3074522 | 0.4704022 | 0.003462284 | 0.007009702 | 4.890145e-05 | 6.793105e-06 | 0 | 5.262502e-05 | 3.081638e-06 |
| WO16 | inner | 0.7219528 | 0.143427 | 3.861452e-05 | 5.564081e-05 | 0.4493006 | 0.4263199 | 0.2227635 | 0.4829417 | 0.003188549 | 0.006044899 | 1.870428e-05 | 4.4819e-06 | 0 | 2.673751e-05 | 2.390556e-06 |
| WO16 | outer | 0.6887884 | 0.1269882 | 9.905404e-05 | 0.0001511411 | 0.4689364 | 0.4248139 | 0.3029951 | 0.5274054 | 0.003192253 | 0.007135298 | 5.123532e-05 | 6.135688e-06 | 0 | 7.232927e-05 | 1.977401e-06 |

S/D共享未加权sum(g)及最终risk cap，不代表实际M=sum(a0*g)、TV或风险修正幅度相等。错Omics的eligible交集损失必须同时披露。这里是selected模型的all-fit匹配context探针；8查询的输入梯度只隔离新增修正。训练日志/summary保存的最后优化步参数梯度来自实际最后epoch，不可冒称selected epoch参数梯度。

S16的四种冻结wrong/off干预在本次Harrell与Uno60点值及20k区间中均为零差，**这不等于修正风险为零或梯度断路**。outer角色模型的all-fit探针中，correction risk SD均值为0.001133255，相对anchor SD均值为0.001631679；Clinical/RNA隔离输入梯度和attention修正也非零。应解释为“观察到非零但小的修正，未改变当前排名指标”；幅度小是实测表象，尚不能认定为唯一因果瓶颈。这些幅度/梯度是训练患者探针，不能冒充独立形态学真值。

## 训练预算

| arm | inner_cells | cap_fraction | maximum_cap_fraction | budget_gate_pass | selected_epoch_median | actual_epoch_median |
| --- | --- | --- | --- | --- | --- | --- |
| D16 | 15 | 0 | 0.2 | True | 17 | 57 |
| F16 | 15 | 0 | 0.2 | True | 17 | 57 |
| L0 | 15 | 0.06666667 | 0.2 | True | 33 | 73 |
| P1 | 15 | 0 | 0.2 | True | 16 | 56 |
| P16 | 15 | 0 | 0.2 | True | 17 | 57 |
| S1 | 15 | 0 | 0.2 | True | 15 | 55 |
| S16 | 15 | 0 | 0.2 | True | 17 | 57 |
| WC16 | 15 | 0 | 0.2 | True | 17 | 57 |
| WO16 | 15 | 0 | 0.2 | True | 17 | 57 |

预算门只检验预注册patience/cap，不能证明全局最优。

存在训练上限触顶的inner单元：

| arm | capped_inner_cells | inner_cells |
| --- | --- | --- |
| L0 | 1 | 15 |

低于预注册容许比例只代表预算资格通过，不能把这些触顶单元称为已收敛。

## 机制查询覆盖

| arm | role | checkpoint_count | isolated_input_gradient_available | risk_attention_probe_unique_query_patients | risk_attention_probe_unique_query_WSIs | risk_attention_probe_per_checkpoint_patients_min | risk_attention_probe_per_checkpoint_patients_max | risk_attention_probe_per_checkpoint_WSIs_min | risk_attention_probe_per_checkpoint_WSIs_max | isolated_input_gradient_probe_unique_query_patients | isolated_input_gradient_probe_unique_query_WSIs | isolated_input_gradient_probe_per_checkpoint_patients_min | isolated_input_gradient_probe_per_checkpoint_patients_max | isolated_input_gradient_probe_per_checkpoint_WSIs_min | isolated_input_gradient_probe_per_checkpoint_WSIs_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D16 | inner | 15 | True | 512 | 519 | 327 | 328 | 330 | 334 | 40 | 40 | 8 | 8 | 8 | 8 |
| D16 | outer | 15 | True | 512 | 519 | 409 | 410 | 413 | 417 | 32 | 33 | 8 | 8 | 8 | 9 |
| F16 | inner | 15 | True | 512 | 519 | 327 | 328 | 330 | 334 | 40 | 40 | 8 | 8 | 8 | 8 |
| F16 | outer | 15 | True | 512 | 519 | 409 | 410 | 413 | 417 | 32 | 33 | 8 | 8 | 8 | 9 |
| L0 | inner | 15 | False | 512 | 519 | 327 | 328 | 330 | 334 | 40 | 40 | 8 | 8 | 8 | 8 |
| L0 | outer | 15 | False | 512 | 519 | 409 | 410 | 413 | 417 | 32 | 33 | 8 | 8 | 8 | 9 |
| P1 | inner | 15 | False | 512 | 519 | 327 | 328 | 330 | 334 | 40 | 40 | 8 | 8 | 8 | 8 |
| P1 | outer | 15 | False | 512 | 519 | 409 | 410 | 413 | 417 | 32 | 33 | 8 | 8 | 8 | 9 |
| P16 | inner | 15 | False | 512 | 519 | 327 | 328 | 330 | 334 | 40 | 40 | 8 | 8 | 8 | 8 |
| P16 | outer | 15 | False | 512 | 519 | 409 | 410 | 413 | 417 | 32 | 33 | 8 | 8 | 8 | 9 |
| S1 | inner | 15 | True | 512 | 519 | 327 | 328 | 330 | 334 | 40 | 40 | 8 | 8 | 8 | 8 |
| S1 | outer | 15 | True | 512 | 519 | 409 | 410 | 413 | 417 | 32 | 33 | 8 | 8 | 8 | 9 |
| S16 | inner | 15 | True | 512 | 519 | 327 | 328 | 330 | 334 | 40 | 40 | 8 | 8 | 8 | 8 |
| S16 | outer | 15 | True | 512 | 519 | 409 | 410 | 413 | 417 | 32 | 33 | 8 | 8 | 8 | 9 |
| WC16 | inner | 15 | True | 512 | 519 | 327 | 328 | 330 | 334 | 40 | 40 | 8 | 8 | 8 | 8 |
| WC16 | outer | 15 | True | 512 | 519 | 409 | 410 | 413 | 417 | 32 | 33 | 8 | 8 | 8 | 9 |
| WO16 | inner | 15 | True | 512 | 519 | 327 | 328 | 330 | 334 | 40 | 40 | 8 | 8 | 8 | 8 |
| WO16 | outer | 15 | True | 512 | 519 | 409 | 410 | 413 | 417 | 32 | 33 | 8 | 8 | 8 | 9 |

只公开独立患者/WSI数量，不公开indices；checkpoint数不等于独立患者数。

## PFI次要终点

| arm | Harrell | Harrell_seed_SD | Uno60 | Uno60_seed_SD |
| --- | --- | --- | --- | --- |
| D16 | 0.7191566 | 0.003659049 | 0.7092182 | 0.005921592 |
| F16 | 0.7158122 | 0.001018801 | 0.7072301 | 0.004069114 |
| HIST_CS_S32_CONCH32 | 0.7206656 | 0 | 0.683313 | 0 |
| J0 | 0.7144255 | 0 | 0.7027215 | 0 |
| J_CS | 0.7119785 | 0 | 0.6702749 | 0 |
| L0 | 0.7145479 | 0.003554592 | 0.7007948 | 0.008006229 |
| P1 | 0.7150781 | 0.005581993 | 0.7079618 | 0.003264627 |
| P16 | 0.7169542 | 0.002373585 | 0.7075242 | 0.00177926 |
| P16__FULL_OMICS_WRONG | 0.715282 | 0.005157773 | 0.7070354 | 0.0001961383 |
| P16__UNIFORM_PAIR | 0.7147926 | 0.0009711555 | 0.7063654 | 0.006889979 |
| P1__FULL_OMICS_WRONG | 0.7231535 | 0.006135217 | 0.7130933 | 0.007924287 |
| P1__UNIFORM_PAIR | 0.7129165 | 0.0007960855 | 0.7056316 | 0.005760127 |
| S1 | 0.7109588 | 0.002109795 | 0.703658 | 0.005359081 |
| S16 | 0.7167095 | 0.002155424 | 0.7063785 | 0.005457441 |
| S16__CONSTANT_C | 0.7167095 | 0.002398681 | 0.7062937 | 0.005121432 |
| S16__CORRECTION_OFF | 0.7167095 | 0.002398681 | 0.7062937 | 0.005121432 |
| S16__WRONG_C | 0.7166687 | 0.002341837 | 0.7063432 | 0.00534656 |
| S16__WRONG_O | 0.7167095 | 0.002155424 | 0.7063785 | 0.005457441 |
| WC16 | 0.7191158 | 0.002109795 | 0.7045529 | 0.0001636463 |
| WO16 | 0.7177291 | 0.001726017 | 0.70564 | 0.001231991 |

仅迁移冻结DSS风险，未使用PFI训练、选择epoch或挑选架构。

## 执行与GPU（仅已观察区间）

| role | samples | sampled_interval_seconds | wall_clock_span_seconds | observed_interval_count | maximum_counted_interval_seconds | excluded_unknown_gap_count | excluded_unknown_gap_seconds | zero_length_interval_count | maximum_running_workers | GPU_percent_time_weighted_mean | GPU_percent_maximum | memory_MiB_time_weighted_mean | memory_MiB_maximum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inner | 608 | 6113.829 | 6113.829 | 607 | 60 | 0 | 0 | 0 | 8 | 97.98093 | 100 | 22367.77 | 24413 |
| outer | 337 | 3363.321 | 30763.11 | 334 | 60 | 2 | 27399.79 | 0 | 8 | 95.7279 | 100 | 24957.17 | 28971 |

GPU原始采样不改写。名义采样间隔10秒，仅对相邻间隔0<dt≤60秒做前值保持时间加权；超过60秒的故障、断电或审计间隙均记为unknown并整段剔除，报告其数量和秒数，不估填离线利用率。最大值只取实际采样点，不将已观察区间均值推广到完整墙钟时间。

正式拟合、实际发生更新的诊断短拟合和继承参考分别计数，见EXECUTION_COUNTS.json。无新增native/PCA/KMeans训练，历史官方方法仅作为背景，不混入当前统一协议榜单。患者级风险、gene信息、权重和所有原始数据仅保留私有目录。

账本可确认实际发生更新的神经拟合至少284次、至多291次；差额来自7个已提交0步的中断尝试，每个可能已有1步发生但尚未提交。已完成和正常失败尝试的优化步回调完整。15个被终止尝试的committed步数为记录下界，每个最多另有1步已发生但未提交；全部拟合优化步总量区间为[14346, 14361]，不把记录下界称为精确实际步数。

### 实际参数与计算

| arm | registered_parameters_mean | registered_parameters_max | last_step_gradient_parameters_mean | last_step_gradient_parameters_max | actual_epochs_mean | actual_epochs_max | seconds_mean | seconds_max | peak_GPU_MiB_mean | peak_GPU_MiB_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D16 | 86093.2 | 86128 | 86093.2 | 86128 | 52.8 | 136 | 285.0182 | 703.5854 | 2827.183 | 3122.046 |
| F16 | 86093.2 | 86128 | 86093.2 | 86128 | 49.8 | 119 | 273.5162 | 617.9197 | 2822.179 | 3117.124 |
| L0 | 70565.2 | 70576 | 69652.4 | 69656 | 66.4 | 160 | 200.5279 | 503.2892 | 2775.695 | 3071.664 |
| P1 | 86093.2 | 86128 | 85180.4 | 85208 | 48.86667 | 119 | 199.833 | 470.3606 | 2775.979 | 3071.876 |
| P16 | 86093.2 | 86128 | 85180.4 | 85208 | 51.66667 | 126 | 210.2987 | 502.8763 | 2775.979 | 3071.876 |
| S1 | 86093.2 | 86128 | 86093.2 | 86128 | 46.93333 | 111 | 257.0827 | 580.5726 | 2822.179 | 3117.124 |
| S16 | 86093.2 | 86128 | 86093.2 | 86128 | 48.53333 | 106 | 265.1147 | 542.1999 | 2822.179 | 3117.124 |
| WC16 | 86093.2 | 86128 | 86093.2 | 86128 | 55 | 121 | 297.6906 | 650.8459 | 2822.179 | 3117.124 |
| WO16 | 86093.2 | 86128 | 86093.2 | 86128 | 51.33333 | 131 | 277.9578 | 691.5563 | 2822.179 | 3117.124 |
