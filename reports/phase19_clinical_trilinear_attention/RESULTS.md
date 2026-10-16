# Phase19 — Clinical-conditioned Low-rank Patch–Pathway Attention

## 判定与证据范围

**MECHANISM_NOT_SUPPORTED_IN_THIS_ARCHITECTURE**。机制子矩阵判定为 `MECHANISM_NOT_SUPPORTED_IN_THIS_ARCHITECTURE`；完整方法预算与机制预算分别披露，不能把技术PASS误当优化充分。

父提交 `099f3d8cd0815a1f994f22b5ce5801b1f4621105`，训练来源 `ce14bd3053f51bbf277d21392419da70c41208581bb9580d78ca546f953873d5`。本轮完成 **150 inner + 150 fresh outer = 300 次正式神经网络拟合**；另有5次实际发生generator更新的短拟合、20个唯一凸控制拟合。旧125次内层Attention pilot没有新OOF，不计本轮300次，也不与完整外折表混排。

## DSS：完整患者级外折结果

| arm | Harrell | Uno60 | Harrell_seed_SD | Uno60_seed_SD | Harrell_seed_min | Uno60_seed_min | Harrell_seed_max | Uno60_seed_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 0.785384 | 0.781719 | 0.00490253 | 0.00394332 | 0.780052 | 0.777505 | 0.789697 | 0.785319 |
| A1 | 0.79495 | 0.783663 | 0.00442807 | 0.00997399 | 0.792049 | 0.772798 | 0.800047 | 0.792404 |
| A2 | 0.784757 | 0.781093 | 0.00509443 | 0.00727289 | 0.780052 | 0.772787 | 0.790167 | 0.786318 |
| A3 | 0.786638 | 0.784126 | 0.00300333 | 0.00390913 | 0.784051 | 0.781361 | 0.789932 | 0.788599 |
| A4 | 0.786246 | 0.78137 | 0.00319382 | 0.0060729 | 0.782875 | 0.774392 | 0.789226 | 0.785458 |
| A5 | 0.784757 | 0.781046 | 0.00470478 | 0.00362356 | 0.780052 | 0.777025 | 0.789461 | 0.784058 |
| A6 | 0.779581 | 0.772369 | 0.00164667 | 0.00387145 | 0.777699 | 0.769215 | 0.780757 | 0.77669 |
| MGCT | 0.770407 | 0.751232 | 0.0195588 | 0.0358449 | 0.747824 | 0.709932 | 0.781934 | 0.77425 |
| CAMR_REPAIRED | 0.793539 | 0.76012 | 0.012853 | 0.0115441 | 0.782875 | 0.747304 | 0.80781 | 0.769704 |
| MMP | 0.728299 | 0.688496 | 0.0120959 | 0.0122808 | 0.71889 | 0.674322 | 0.741943 | 0.695947 |
| J0 | 0.803576 | 0.792339 | 0 | 0 | 0.803576 | 0.792339 | 0.803576 | 0.792339 |
| J_CS | 0.814161 | 0.783318 | 0 | 0 | 0.814161 | 0.783318 | 0.814161 | 0.783318 |
| HIST_CS_S32_CONCH32 | 0.825923 | 0.806707 | 0 | 0 | 0.825923 | 0.806707 | 0.825923 | 0.806707 |

512患者、65 DSS events；每seed先加总五折同折可比较对分子/分母，再平均3个seed的C值；不是风险ensemble，也不是将15 cells叫seed mean。seed SD/范围描述算法随机性，3 seeds不是三个独立患者队列。历史Clinical+S32+CONCH32的视觉表征和训练协议不同，只是强历史参照。

## 六个预锁定主要比较

| comparison | metric | delta | CI95_low | CI95_high | Bonferroni_low | Bonferroni_high | positive_seeds | positive_fold_cells | fold_cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A3-A0 | Harrell | 0.00125461 | -0.0082645 | 0.0102746 | -0.0130851 | 0.0148685 | 2 | 9 | 15 |
| A3-A0 | Uno60 | 0.00240674 | -0.0140956 | 0.0217817 | -0.0232527 | 0.0364128 | 3 | 9 | 15 |
| A3-A1 | Harrell | -0.00831177 | -0.0237373 | 0.00592271 | -0.0319705 | 0.0122183 | 0 | 6 | 15 |
| A3-A1 | Uno60 | 0.000463624 | -0.0294221 | 0.0370357 | -0.0452953 | 0.0628069 | 1 | 8 | 15 |
| A3-A2 | Harrell | 0.00188191 | -0.00283765 | 0.0068923 | -0.00547314 | 0.0102116 | 2 | 4 | 15 |
| A3-A2 | Uno60 | 0.00303282 | -0.00612383 | 0.01479 | -0.0111446 | 0.0240825 | 2 | 3 | 15 |
| A3-A4 | Harrell | 0.000392065 | -0.00462832 | 0.00542349 | -0.0074104 | 0.00861085 | 2 | 5 | 15 |
| A3-A4 | Uno60 | 0.00275583 | -0.00703044 | 0.0146688 | -0.0120277 | 0.0235416 | 2 | 3 | 15 |
| A3-A5 | Harrell | 0.00188191 | -0.00398643 | 0.00776399 | -0.00706031 | 0.0104744 | 2 | 9 | 15 |
| A3-A5 | Uno60 | 0.00308047 | -0.00375878 | 0.0107958 | -0.00771274 | 0.016525 | 2 | 10 | 15 |
| A3-A6 | Harrell | 0.00705716 | -0.00844906 | 0.0228818 | -0.0158294 | 0.0311255 | 3 | 7 | 15 |
| A3-A6 | Uno60 | 0.011757 | -0.0110478 | 0.033877 | -0.0238833 | 0.0452617 | 3 | 11 | 15 |

实际有效bootstrap 20000/20000，零分母排除0。20,000次固定外折内患者重采样，所有模型及seed共享计数，pair权重count_i×count_j；Uno60使用对应outer-train反向KM，tau=60。12项家族校正区间覆盖率99.5833%，普通95%区间同列。点估计与每次bootstrap都计算同一个mean-seed-C配对统计量；不先平均风险。区间条件于已拟合模型，不包括完整重训练和反复研究者选择方差。

### 逐seed方向

| comparison | metric | seed | delta |
| --- | --- | --- | --- |
| A3-A0 | Harrell | 31 | 0.00588097 |
| A3-A0 | Harrell | 47 | 0.000235239 |
| A3-A0 | Harrell | 83 | -0.00235239 |
| A3-A0 | Uno60 | 31 | 0.00385616 |
| A3-A0 | Uno60 | 47 | 0.00327922 |
| A3-A0 | Uno60 | 83 | 8.48324e-05 |
| A3-A1 | Harrell | 31 | -0.00611621 |
| A3-A1 | Harrell | 47 | -0.0101153 |
| A3-A1 | Harrell | 83 | -0.00870383 |
| A3-A1 | Uno60 | 31 | 0.0085629 |
| A3-A1 | Uno60 | 47 | -0.00380586 |
| A3-A1 | Uno60 | 83 | -0.00336617 |
| A3-A2 | Harrell | 31 | 0.00188191 |
| A3-A2 | Harrell | 47 | -0.000235239 |
| A3-A2 | Harrell | 83 | 0.00399906 |
| A3-A2 | Uno60 | 31 | -0.00281377 |
| A3-A2 | Uno60 | 47 | 0.00228029 |
| A3-A2 | Uno60 | 83 | 0.00963193 |
| A3-A4 | Harrell | 31 | -0.000705716 |
| A3-A4 | Harrell | 47 | 0.000705716 |
| A3-A4 | Harrell | 83 | 0.00117619 |
| A3-A4 | Uno60 | 31 | -0.00289999 |
| A3-A4 | Uno60 | 47 | 0.00314062 |
| A3-A4 | Uno60 | 83 | 0.00802686 |
| A3-A5 | Harrell | 31 | 0.00588097 |
| A3-A5 | Harrell | 47 | 0.000470478 |
| A3-A5 | Harrell | 83 | -0.000705716 |
| A3-A5 | Uno60 | 31 | 0.00433631 |
| A3-A5 | Uno60 | 47 | 0.0065443 |
| A3-A5 | Uno60 | 83 | -0.00163921 |
| A3-A6 | Harrell | 31 | 0.00517525 |
| A3-A6 | Harrell | 47 | 0.0122324 |
| A3-A6 | Harrell | 83 | 0.00376382 |
| A3-A6 | Uno60 | 31 | 0.0101583 |
| A3-A6 | Uno60 | 47 | 0.0119088 |
| A3-A6 | Uno60 | 83 | 0.013204 |

### 工程推进门

| comparison | metric | delta | positive_seeds | direction_and_seed_pass | Bonferroni_positive |
| --- | --- | --- | --- | --- | --- |
| A3-A0 | Harrell | 0.00125461 | 2 | True | False |
| A3-A0 | Uno60 | 0.00240674 | 3 | True | False |
| A3-A1 | Harrell | -0.00831177 | 0 | False | False |
| A3-A1 | Uno60 | 0.000463624 | 1 | False | False |
| A3-A2 | Harrell | 0.00188191 | 2 | True | False |
| A3-A2 | Uno60 | 0.00303282 | 2 | True | False |
| A3-A4 | Harrell | 0.000392065 | 2 | True | False |
| A3-A4 | Uno60 | 0.00275583 | 2 | True | False |
| A3-A5 | Harrell | 0.00188191 | 2 | True | False |
| A3-A5 | Uno60 | 0.00308047 | 2 | True | False |
| A3-A6 | Harrell | 0.00705716 | 3 | True | False |
| A3-A6 | Uno60 | 0.011757 | 3 | True | False |

| reference | metric | delta_A3_minus_reference | no_mean_harm |
| --- | --- | --- | --- |
| J_CS | Harrell | -0.0275229 | False |
| J_CS | Uno60 | 0.000808647 | True |
| J0 | Harrell | -0.0169372 | False |
| J0 | Uno60 | -0.00821327 | False |

机制预算充分=True；全部方法预算充分=True；六比较×两指标方向/seed门=False；J0/J_CS无均值损害门=False。这些是预锁定工程门，不是生物学机制或独立效能确认。校正后下限>0的主要统计量：[]。未检出不等于无信息或统计等效。

## 强简单控制、官方核心适配与冻结干预

| comparison | metric | delta | CI95_low | CI95_high | Bonferroni_low | Bonferroni_high | family_tests |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A3-J0 | Harrell | -0.0169372 | -0.0381328 | 0.00305454 | -0.0512047 | 0.0136333 | 20 |
| A3-J0 | Uno60 | -0.00821327 | -0.0483918 | 0.0341453 | -0.0750383 | 0.0675426 | 20 |
| A3-J_CS | Harrell | -0.0275229 | -0.0751782 | 0.0197637 | -0.103918 | 0.04431 | 20 |
| A3-J_CS | Uno60 | 0.000808647 | -0.0733864 | 0.0771597 | -0.11683 | 0.126667 | 20 |
| A3-HIST_CS_S32_CONCH32 | Harrell | -0.0392849 | -0.0809053 | 0.00146539 | -0.105924 | 0.0237731 | 20 |
| A3-HIST_CS_S32_CONCH32 | Uno60 | -0.0225804 | -0.0841162 | 0.0401505 | -0.121737 | 0.0812109 | 20 |
| A3-MGCT | Harrell | 0.0162315 | -0.0143234 | 0.0467787 | -0.0326874 | 0.063975 | 20 |
| A3-MGCT | Uno60 | 0.0328945 | -0.015191 | 0.0837389 | -0.0426097 | 0.121602 | 20 |
| A3-CAMR_REPAIRED | Harrell | -0.00690034 | -0.0358342 | 0.0195938 | -0.0538097 | 0.0335386 | 20 |
| A3-CAMR_REPAIRED | Uno60 | 0.0240064 | -0.0278553 | 0.08745 | -0.0531031 | 0.133936 | 20 |
| A3-MMP | Harrell | 0.0583392 | 0.0128616 | 0.105604 | -0.0101879 | 0.132704 | 20 |
| A3-MMP | Uno60 | 0.0956298 | 0.0367213 | 0.161548 | 0.0057613 | 0.211921 | 20 |
| A3-A3__WRONG_C | Harrell | 0.00580256 | 0.00151639 | 0.0112714 | -0.000340392 | 0.0155403 | 20 |
| A3-A3__WRONG_C | Uno60 | 0.00268894 | -0.0031464 | 0.00897861 | -0.00751304 | 0.0146591 | 20 |
| A3-A3__WRONG_O | Harrell | -0.000235239 | -0.0020792 | 0.00150191 | -0.00344118 | 0.00270097 | 20 |
| A3-A3__WRONG_O | Uno60 | 0.000115131 | -0.00138422 | 0.00167196 | -0.00258771 | 0.00289859 | 20 |
| A3-A3__CONSTANT_C | Harrell | 0.00399906 | 0.000648803 | 0.00791296 | -0.00116149 | 0.0105854 | 20 |
| A3-A3__CONSTANT_C | Uno60 | 0.00292818 | -0.00175627 | 0.00959845 | -0.00463973 | 0.0162233 | 20 |
| A3-A3__INTERACTION_OFF | Harrell | -0.00988003 | -0.028434 | 0.00805574 | -0.0396165 | 0.0179329 | 20 |
| A3-A3__INTERACTION_OFF | Uno60 | -0.0065218 | -0.0461047 | 0.032208 | -0.0762769 | 0.0600006 | 20 |

J0为共同全输入线性Cox，J_CS为同源Clinical+S32；均固定ridge10，并不预称J0为最强基线。神经模型的共同线性主效应从J0初始化、联合优化，不以训练内风险再学习自由stacking系数。

A4/A5是分别用错误Clinical/Omics交互条件真实重新训练的模型；A3__WRONG_C/WRONG_O是冻结A3只换交互输入的推理干预。两者不能互称重训消融。本人Clinical/RNA/CNA主效应保持匹配。Clinical错配仅按癌种分区内置换，不按Stage/TNM/Age/Sex保留患者严重度；Omics错配沿用癌种/Stage规则，回退与平台/缺失差异见数据审计，不能把错配优势当因果语义证明。

MGCT、CAMR_REPAIRED、MMP已运行固定官方核心，但这是共同特征/Cox/Clinical适配，非原生论文训练协议严格复现。CAMR明确保留辅助目标及独立判别器，MMP训练分区拟合原型；不能宣称它们与Attention完全同算量或同优化任务。源码/许可/必要修复见BASELINE_REGISTRY与NATIVE_SOURCE_AUDIT，资格测试只证明登记的实现边界。

## 机制解释与表征边界

CP三元分数严格等价于Clinical调制的低秩双线性兼容度，以及增广Q/K；也可视作潜在维度上的Clinical-FiLM。等价实现的输出/梯度parity已测试，但它不是独立算法控制，也不是首次三元乘积或FiLM创新。既有工作已经存在Clinical前置交互，不能概括为此前Clinical都仅末端拼接。

A1是两两logit相加后softmax，其非线性本身可产生三模态依赖；A6才是PO、PC、OC独立标量风险线性相加的严格两两函数。A2保留增广常数1与可学习全局rank4项，因此固定Clinical均值不会把整个CP旁路置零。A3非截距Clinical系数初始0，但其真实参数梯度可学习；不能将零初始化响应误当实现断梯度。

24个Token分为RNA/CNA gain/CNA loss各8个；每个来自通路内一个可学习加权标量、覆盖率及slot embedding，并非完整512维RNA的无损32维通路表征。原始RNA512保留于共同线性主效应，只证明信息旁路可用，不能证明RNA→交互Token保留了全部预后排序。负结果不能推广为未压缩RNA或全部患者特异交互无效。

J0加入RNA/CNA各8个固定测量覆盖率主效应，降低把缺失模式增益误判为交互的风险；Token仍有逐基因测量mask，J0不是与其完全等表达的模型。平台来源/大块RNA缺失仍可能混杂，coverage控制不等于已识别并消除真实测序平台影响。UNI2-h只做冻结物理足迹/坐标审计，无原图形态真值，不能将attention分布当组织学标签。

### 前32名训练患者的冻结、无标签探针

| role | arm | main_risk_SD | interaction_risk_SD | matched_wrongC_risk_MAE | matched_wrongO_risk_MAE | clinical_attention_relative_L2 | base_scores_SD | trilinear_scores_SD | actual_combined_scores_SD | actual_score_correction_SD | p_to_o_entropy_mean | p_to_o_maximum_weight_mean | o_to_p_entropy_mean | o_to_p_maximum_weight_mean | isolated_interaction_clinical_gradient_L2 | isolated_interaction_RNA_gradient_L2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inner | A0 | 0.694422 | 0.131727 | 0 | 0.00208155 | 0 | 0.211989 | 0.144416 | 0.211989 | 0 | 3.16451 | 0.0501804 | 6.16963 | 0.00297793 | 0 | 0.000104998 |
| inner | A1 | 0.69264 | 0.182058 | 0.00741847 | 0.0014286 | 0.0402042 | 0.133077 | 0.495227 | 2.11957 | 2.06408 | 3.07444 | 0.0728342 | 4.88265 | 0.0771283 | 0.00118532 | 5.43692e-05 |
| inner | A2 | 0.69416 | 0.133332 | 0 | 0.00204138 | 0 | 0.208884 | 0.353758 | 0.539753 | 0.353758 | 3.1315 | 0.0593804 | 6.15485 | 0.00456262 | 0 | 0.000161757 |
| inner | A3 | 0.694143 | 0.13184 | 0.00147842 | 0.0020112 | 0.0234738 | 0.20824 | 0.415978 | 0.539669 | 0.415978 | 3.13988 | 0.0596601 | 6.15552 | 0.00464013 | 0.000518343 | 0.000204897 |
| inner | A4 | 0.694155 | 0.133036 | 0.000921295 | 0.00204503 | 0.0244965 | 0.2086 | 0.390186 | 0.545591 | 0.390186 | 3.14536 | 0.0587997 | 6.15512 | 0.00462302 | 0.000410982 | 0.000161864 |
| inner | A5 | 0.695051 | 0.12075 | 0.000654015 | 0.00214431 | 0.0244998 | 0.240442 | 0.408038 | 0.564684 | 0.408038 | 3.13486 | 0.0619497 | 6.16514 | 0.00477842 | 0.000197556 | 0.000300737 |
| inner | A6 | 0.667642 | 0.372972 | 0.0135218 | 0.00874642 | 0 | 0.129849 | 0.353769 | 0.129849 | 0 | 3.17141 | 0.0480687 | 6.21973 | 0.00237296 | 0.00319315 | 0.000189406 |
| inner | CAMR_REPAIRED | 0.680791 | 0.40337 | 0 | 0.326589 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| inner | MGCT | 0.735328 | 0.35461 | 0 | 0.0586334 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| inner | MMP | 0.672766 | 1.48293 | 0 | 1.34253 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| outer | A0 | 0.721694 | 0.102319 | 0 | 0.0013322 | 0 | 0.21943 | 0.130721 | 0.21943 | 0 | 3.16469 | 0.0509598 | 6.12676 | 0.00305621 | 0 | 6.49747e-05 |
| outer | A1 | 0.735816 | 0.147286 | 0.00924713 | 0.00163745 | 0.0437814 | 0.122938 | 0.601626 | 1.76283 | 1.74149 | 3.0687 | 0.0719517 | 5.11129 | 0.0675444 | 0.000610051 | 4.64283e-05 |
| outer | A2 | 0.720037 | 0.101429 | 0 | 0.00165498 | 0 | 0.205687 | 0.264731 | 0.416556 | 0.264731 | 3.14094 | 0.0583638 | 6.10708 | 0.004333 | 0 | 9.87782e-05 |
| outer | A3 | 0.719994 | 0.102183 | 0.000915925 | 0.00159193 | 0.0394791 | 0.207148 | 0.282358 | 0.435478 | 0.282358 | 3.13863 | 0.0596949 | 6.1076 | 0.00435703 | 0.000288341 | 0.000110748 |
| outer | A4 | 0.72003 | 0.101141 | 0.000448036 | 0.00171446 | 0.0329076 | 0.205876 | 0.27099 | 0.425171 | 0.27099 | 3.13893 | 0.0600791 | 6.10693 | 0.00433165 | 0.000420688 | 9.89965e-05 |
| outer | A5 | 0.730857 | 0.0981098 | 0.000509623 | 0.00182634 | 0.0218968 | 0.250806 | 0.294293 | 0.511985 | 0.294293 | 3.12946 | 0.0615876 | 6.10434 | 0.00461542 | 0.000287579 | 7.63749e-05 |
| outer | A6 | 0.696721 | 0.279468 | 0.00851379 | 0.0065046 | 0 | 0.123693 | 0.305078 | 0.123693 | 0 | 3.17411 | 0.0466692 | 6.12902 | 0.00259572 | 0.00600649 | 0.000242964 |
| outer | CAMR_REPAIRED | 0.730048 | 0.364979 | 0 | 0.177818 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| outer | MGCT | 0.773647 | 0.334294 | 0 | 0.013656 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| outer | MMP | 0.717567 | 1.7946 | 0 | 1.41257 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

每个checkpoint取固定顺序前32名训练患者，inner/outer分列；患者内WSI等权风险，与WSI级attention统计不混称独立病例。不同seed/fold会重复同一患者，表格是checkpoint中位数，不是独立队列均值。更尖锐attention、更大真/错响应或非零梯度只能证明算子敏感，不自动证明可泛化预测增益。

**隔离输入梯度仅使用fit[:1]的单个训练患者风险**；风险SD/attention/entropy使用fit[:32]。同fold不同seed首患者会重复，15 checkpoints不能当15名独立梯度探针患者。下表由检查点和owner元数据独立去重，无额外forward/gradient/优化；wrong donor身份不冒充query患者。

| arm | role | checkpoint_count | isolated_input_gradient_available | risk_attention_probe_unique_query_patients | risk_attention_probe_unique_query_WSIs | isolated_input_gradient_probe_unique_query_patients | isolated_input_gradient_probe_unique_query_WSIs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | inner | 15 | True | 141 | 142 | 5 | 5 |
| A0 | outer | 15 | True | 46 | 48 | 2 | 3 |
| A1 | inner | 15 | True | 141 | 142 | 5 | 5 |
| A1 | outer | 15 | True | 46 | 48 | 2 | 3 |
| A2 | inner | 15 | True | 141 | 142 | 5 | 5 |
| A2 | outer | 15 | True | 46 | 48 | 2 | 3 |
| A3 | inner | 15 | True | 141 | 142 | 5 | 5 |
| A3 | outer | 15 | True | 46 | 48 | 2 | 3 |
| A4 | inner | 15 | True | 141 | 142 | 5 | 5 |
| A4 | outer | 15 | True | 46 | 48 | 2 | 3 |
| A5 | inner | 15 | True | 141 | 142 | 5 | 5 |
| A5 | outer | 15 | True | 46 | 48 | 2 | 3 |
| A6 | inner | 15 | True | 141 | 142 | 5 | 5 |
| A6 | outer | 15 | True | 46 | 48 | 2 | 3 |
| CAMR_REPAIRED | inner | 15 | False | 141 | 142 | 0 | 0 |
| CAMR_REPAIRED | outer | 15 | False | 46 | 48 | 0 | 0 |
| MGCT | inner | 15 | False | 141 | 142 | 0 | 0 |
| MGCT | outer | 15 | False | 46 | 48 | 0 | 0 |
| MMP | inner | 15 | False | 141 | 142 | 0 | 0 |
| MMP | outer | 15 | False | 46 | 48 | 0 | 0 |

trilinear_scores_SD也可能来自某臂仅为诊断而计算的未使用CP值；实际进入attention的差分尺度看actual_score_correction_SD。A0/A6的CP分数不进入PO attention，A1实际修正是两两logit和；不能按未使用CP诊断宣称这些臂也训练了完整三元机制。原生适配baseline未实现同义的隔离输入梯度时保留NA，不冒充零梯度证据。

### 实际失败发生在哪一层

1. **已观察到条件分支连通和兼容度变化。** outer模型训练探针中，A3实际CP修正SD的checkpoint中位数为0.282358，真/错Clinical的attention相对L2为0.0394791。这些不是断路或全零旁路；但真/错Clinical及Omics的患者风险MAE只有0.000915925/0.00159193，相应交互/主效应风险SD为0.102183/0.719994。这是固定训练探针统计，跨15 checkpoints实际仅46名去重query患者；隔离输入梯度仅2名不同患者，不是全队列独立敏感性证据。

2. **条件响应未转成可靠held-out配对增量。** 冻结A3错Omics后，A3−wrongO的Harrell/Uno60为-0.000235239/+0.000115131，排序几乎不变。冻结错Clinical的Harrell名义95%区间[0.00151639,0.0112714]为正，但次要20项校正区间[-0.000340392,0.0155403]跨0；真实错Clinical重训对照A4的主要Harrell差仅+0.000392065、95%区间[-0.00462832,0.00542349]跨0。不能把单个冻结干预的名义区间当患者特异机制已确认。

3. **完整交互未超过保留主效应的简单控制。** 冻结关闭A3自身交互后Harrell为0.796518，A3−off的Harrell/Uno60为-0.00988003/-0.0065218；Harrell95%区间[-0.028434,0.00805574]仍跨0，不能宣称交互显著有害。J0的Harrell为0.803576，点估计仍高于该冻结off版本。

本轮支持的有限回答是：实现确实产生了Clinical条件变化，但该RNA/CNA联合配对路径没有形成可靠的held-out条件预后增量。通路标量压缩、joint优化中的主效应变化、事件数量及条件信息重叠仍是未被分别隔离的解释；当前证据不能指定唯一失败原因，也不授权据此盲目增加模块。

### 冻结MMP分布Token的无标签数值审计

| statistic | range_or_value |
| --- | --- |
| fit_sample_UNI2_mean_norm_range | [0.9999997787163011, 1.0000002658125595] |
| fit_sample_mean_per_dimension_variance_range | [0.0005020032075327665, 0.0005051256919655365] |
| prior_covariance_to_mean_feature_variance_ratio_range | [197.97052810931416, 199.20191444886905] |
| mean_normalized_probability_entropy_range | [0.42210586055643173, 0.42486702218913425] |
| mean_maximum_allocation_probability_range | [0.7640258042982795, 0.7660718826259055] |
| mean_prototype_pairwise_cosine_range | [0.4501919417211741, 0.4865292450171986] |
| covariance_prototype_pairwise_cosine_range | [0.9997829930587646, 0.9997864602577917] |
| prototype_mean_patient_effective_rank_range | [183.1497979481615, 220.57729816759684] |
| prototype_covariance_patient_effective_rank_range | [11.09156515473697, 12.739762018530174] |
| covariance_fraction_of_combined_feature_variance_range | [0.9518114526429786, 0.9526101750308313] |
| winning_prototypes_observed_range | [16.0, 16.0] |

最终覆盖10/10同源role/fold缓存，NaN/Inf计数均0；这是snapshot 3，先前较少缓存的可用性快照保留在JSON中，不是多次模型训练。概率熵以log(16)归一，uniform参考为熵1、最大分配概率0.0625；应依据上表实际概率和原型均值差异判断，不能仅因native prior .1相对单位范数UNI2逐维方差较大，就宣称分配均匀或整体collapse。

活跃患者combined Token的有效秩范围为[12.012096461417878, 13.815632832567943]；均值/协方差各自有效秩及原型余弦见上表。协方差向量近parallel或原始拼接方差由其尺度主导，是当前固定UNI2适配的限制；不等于均值/概率也无差异，更不是native MMP无效的证明。审计未读终点/OOF/风险，不改eps、模型、source或训练参数，也不据此追加调参。

## 训练预算与梯度

| arm | inner_cells | cap_fraction | maximum_cap_fraction | budget_gate_pass | selected_epoch_median | actual_epoch_median |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | 15 | 0 | 0.2 | True | 17 | 57 |
| A1 | 15 | 0 | 0.2 | True | 18 | 58 |
| A2 | 15 | 0 | 0.2 | True | 17 | 57 |
| A3 | 15 | 0 | 0.2 | True | 17 | 57 |
| A4 | 15 | 0 | 0.2 | True | 17 | 57 |
| A5 | 15 | 0 | 0.2 | True | 17 | 57 |
| A6 | 15 | 0 | 0.2 | True | 11 | 51 |
| CAMR_REPAIRED | 15 | 0 | 0.2 | True | 9 | 49 |
| MGCT | 15 | 0 | 0.2 | True | 10 | 50 |
| MMP | 15 | 0 | 0.2 | True | 8 | 48 |

固定max160、minimum8、patience40、EMA0.2、dropout0.1、AdamW lr0.001/WD0.01、主效应L2=10；没有依据外折成绩追加epoch/rank/scale搜索。触顶不超过20%只是预锁定预算门；零触顶不等于全局收敛。fresh outer逐单元使用本臂同seed/fold内层选择的准确epoch。

预外折预算检查原状态为 `PASS`；BUDGET_AUDIT.json原件保留，独立重算写BUDGET_VERIFICATION.json，未用事后覆盖掩盖原预算证据。

所选inner epoch、训练目标反传后保留的梯度组中位数（含主效应L2；CAMR另见下述辅助目标及D步边界）：

| arm | main_gradient | visual_gradient | molecular_gradient | clinical_factor_gradient | path_factor_gradient | omics_factor_gradient | qk_gradient | readout_gradient | native_gradient | clinical_nonintercept_gradient | CP_global_intercept_gradient |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 0.431973 | 0.198402 | 0.0134578 | 0 | 0 | 0 | 0.00842814 | 1.36535 | 0 | 0 | 0 |
| A1 | 0.477881 | 0.292151 | 0.0253565 | 0.051586 | 0.0812714 | 0.00588294 | 0.00649257 | 2.3479 | 0 | 0.047107 | 0.0241899 |
| A2 | 0.571054 | 0.207889 | 0.0203255 | 0.00224445 | 0.00943032 | 0.0192849 | 0.00776439 | 1.29931 | 0 | 8.88052e-10 | 0.00224445 |
| A3 | 0.606323 | 0.210899 | 0.0222662 | 0.0180933 | 0.0101654 | 0.0223144 | 0.00774475 | 1.31181 | 0 | 0.0176391 | 0.00224061 |
| A4 | 0.570673 | 0.209793 | 0.0215435 | 0.00823573 | 0.00977171 | 0.0212015 | 0.00774722 | 1.30227 | 0 | 0.00797449 | 0.00227747 |
| A5 | 0.469985 | 0.191291 | 0.0183021 | 0.0143075 | 0.0123544 | 0.0179946 | 0.00580415 | 1.3127 | 0 | 0.0139167 | 0.00308269 |
| A6 | 0.639546 | 0.771134 | 0.00372867 | 0.172025 | 0.18449 | 0.00229232 | 0.00147064 | 3.0705 | 0 | 0.149961 | 0.0542855 |
| CAMR_REPAIRED | 1.04611 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.61415 | 0 | 0 |
| MGCT | 0.864979 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.244118 | 0 | 0 |
| MMP | 2.38489 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3.41541 | 0 | 0 |

所选checkpoint的隔离输入梯度、训练损失梯度与不同训练长度下梯度是不同对象；不能仅据梯度较大归因唯一的竞争/机制原因。完整逐epoch loss、EMA、Harrell和梯度保存在TRAINING_CURVES.csv。

## 参数与计算

| role | arm | cells | registered_parameters | last_step_gradient_parameters_median | discriminator_parameters | actual_epochs_median | selected_epoch_median | seconds_sum | peak_GPU_MiB_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inner | A0 | 15 | 70369 | 69880 | 0 | 57 | 17 | 1225.63 | 2507.7 |
| inner | A1 | 15 | 70369 | 70300 | 0 | 58 | 18 | 1206.7 | 2507.7 |
| inner | A2 | 15 | 70369 | 70300 | 0 | 57 | 17 | 1262.9 | 2507.7 |
| inner | A3 | 15 | 70369 | 70300 | 0 | 57 | 17 | 1288.64 | 2507.7 |
| inner | A4 | 15 | 70369 | 70300 | 0 | 57 | 17 | 1270.73 | 2507.7 |
| inner | A5 | 15 | 70369 | 70300 | 0 | 57 | 17 | 1278.12 | 2507.7 |
| inner | A6 | 15 | 70369 | 70364 | 0 | 51 | 11 | 1182.81 | 2507.7 |
| inner | CAMR_REPAIRED | 15 | 2524813 | 2.52481e+06 | 11969 | 49 | 9 | 3257.22 | 2379.89 |
| inner | MGCT | 15 | 6385296 | 4.80347e+06 | 0 | 50 | 10 | 1569.91 | 3312.93 |
| inner | MMP | 15 | 2171785 | 2.17178e+06 | 0 | 48 | 8 | 1067.54 | 2449.88 |
| outer | A0 | 15 | 70369 | 69881 | 0 | 17 | 17 | 369.9 | 3076.21 |
| outer | A1 | 15 | 70369 | 70305 | 0 | 18 | 18 | 354.539 | 3076.21 |
| outer | A2 | 15 | 70369 | 70305 | 0 | 17 | 17 | 404.999 | 3076.21 |
| outer | A3 | 15 | 70369 | 70305 | 0 | 17 | 17 | 416.917 | 3076.21 |
| outer | A4 | 15 | 70369 | 70305 | 0 | 17 | 17 | 394.599 | 3076.21 |
| outer | A5 | 15 | 70369 | 70305 | 0 | 17 | 17 | 385.953 | 3076.21 |
| outer | A6 | 15 | 70369 | 70369 | 0 | 11 | 11 | 267.514 | 3076.21 |
| outer | CAMR_REPAIRED | 15 | 2524813 | 2.52481e+06 | 11969 | 9 | 9 | 699.127 | 2938.17 |
| outer | MGCT | 15 | 6385296 | 4.80347e+06 | 0 | 10 | 10 | 315.162 | 3821.48 |
| outer | MMP | 15 | 2171785 | 2.17178e+06 | 0 | 8 | 8 | 176.634 | 3031.9 |

registered_parameters为注册数；last_step_gradient_parameters为最后一次汇总时具有梯度tensor的参数数量，即使数值为0也会计入，不能直接称全部有效预测容量。CAMR独立D步之后保留D.grad，该计数及native_gradient混合了本epoch遗留D步与G步梯度，不能解释为单一Efron/G目标的活跃参数或与Attention作同目标梯度大小排名。实际G optimizer明确排除D参数，资格测试核验G/D更新隔离，因此这一诊断口径不改变风险或参数更新。A0–A6没有判别器，不受此项混合影响；共同main梯度也包含显式L2，CAMR另有辅助损失。各臂可有注册但未用路径；判别器参数及优化步骤另列。并行cell seconds之和不是墙钟时间或隔离速度排名，微批只管理显存，所有患者风险组成一次完整Efron风险集再反传，checkpoint保留dropout RNG。

### 整卡采样

| role | samples | sampled_interval_seconds | maximum_running_workers | maximum_configured_workers_limit | GPU_utilization_percent_time_weighted_mean | GPU_utilization_percent_sample_maximum | GPU_utilization_percent_sample_median | whole_GPU_memory_MiB_time_weighted_mean | whole_GPU_memory_MiB_sample_maximum | whole_GPU_memory_MiB_sample_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inner | 238 | 2389.25 | 8 | 8 | 93.5155 | 100 | 99 | 18744.8 | 25469 | 19133 |
| outer | 87 | 866.759 | 8 | 8 | 86.2687 | 100 | 99 | 16384.4 | 29247 | 17187 |

均值按相邻采样时段左值保持加权，仅覆盖记录到的正式inner/outer区间；最大值与中位数为采样统计，不是真实瞬时峰值或完整端到端负载。整卡可能包含并发进程，不能用显存占用率证明模型质量。

运行时private WORKERS可调整并发；上表记录实际worker上限/在跑数，不能把源码默认值当完整运行事实，也不把忙时99%称为始终99%。模型/优化/随机种子/数据及训练source未因调度改变。

### 实际软件与硬件环境

| field | value |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.5.1+cu124 |
| compiled CUDA | 12.4 |
| numpy | 2.1.3 |
| scipy | 1.14.1 |
| pandas | 2.2.3 |
| scikit-learn | 1.5.2 |
| scikit-survival | 0.23.1 |
| einops | 0.8.2 |
| GPU | {'name': 'NVIDIA GeForce RTX 4080 SUPER', 'driver_version': '595.71.05', 'memory_total_MiB': 32760.0} |

ENVIRONMENT.json只采集版本/GPU等白名单，训练固定deterministic=True、TF32=False、CUBLAS_WORKSPACE_CONFIG=:4096:8；另列报告进程当前开关，避免把独立报告进程默认值误称训练实际设置。未导出环境变量集合或凭据。

## 实际执行次数

| item | value |
| --- | --- |
| status | PASS |
| source | ce14bd3053f51bbf277d21392419da70c41208581bb9580d78ca546f953873d5 |
| new_formal_inner_neural_fits | 150 |
| new_formal_outer_neural_fits | 150 |
| new_formal_neural_fits | 300 |
| formal_generator_optimizer_steps | 10908 |
| formal_discriminator_optimizer_steps | 910 |
| unique_current_source_convex_fits | 20 |
| convex_seed_role_metric_references | 60 |
| historical_LOCAL_outer_metric_references | 15 |
| old_125_inner_pilot_new_fits_or_OOF | 0 |
| smoke_attempts_recorded | 5 |
| smoke_real_short_fits_with_generator_steps | 5 |
| smoke_returned_successfully | 5 |
| smoke_failed_after_generator_steps | 0 |
| smoke_generator_optimizer_steps | 10 |
| smoke_discriminator_optimizer_steps | 2 |
| all_recorded_real_neural_fits | 305 |
| all_recorded_generator_optimizer_steps | 10918 |
| all_recorded_discriminator_optimizer_steps | 912 |
| verified_completed_MMP_prototype_cache_producers_current_source | 10 |
| verified_completed_MMP_prototype_cache_producers_other_source | 0 |
| MMP_internal_kmeans_initializations_current_source | 10 |
| MMP_fit_attempt_ledger_available | False |
| MMP_failed_or_aborted_attempt_count | NOT_RECORDED |
| clustering_scope | verified completed fit-only KMeans cache producers only, not all attempts; three seed consumers reuse them; frozen per-WSI PANTHER EM is not another cross-patient KMeans fit |
| fit_definition | neural fit counted once if generator optimizer stepped; D optimizer steps separate; deterministic convex and unsupervised prototype fits separate |
| prior_stage_training_recounted_as_new | False |
| formal_failed_task_receipt_present | False |
| formal_fit_count_scope | 300 complete checkpoint+receipt fits; partial failed formal attempts, if any, require their own execution evidence and are not inferred as zero |

300个正式神经模型与诊断短拟合分计；20个唯一ridge fit跨seed引用，不乘成60次拟合。MMP十个当前source的训练分区原型缓存对应十个已完成KMeans producer，n_init为内部初始化次数，三个seed及冻结回放复用；每WSI PANTHER EM不另冒充跨患者聚类模型。原型生产没有尝试级账本，失败/中断聚类尝试记为NOT_RECORDED，不能从十个完成缓存推断总尝试数或失败数为0。诊断短拟合按实际optimizer step增量账本计数，包括失败但已更新的尝试，不保存其checkpoint或患者风险。完整患者OOF/正式checkpoint/PCA/原型权重仅在私有目录保留。

## PFI：冻结DSS风险迁移

510患者、129 events；只迁移DSS风险，无PFI训练/选配置，不用次要终点改写DSS判断。PFI来自同一研究队列，不是独立外部验证。

| arm | Harrell | Uno60 |
| --- | --- | --- |
| A0 | 0.702231 | 0.697559 |
| A1 | 0.710062 | 0.697575 |
| A2 | 0.704311 | 0.699494 |
| A3 | 0.704148 | 0.702834 |
| A4 | 0.704923 | 0.700159 |
| A5 | 0.70268 | 0.700241 |
| A6 | 0.703169 | 0.691255 |
| MGCT | 0.67829 | 0.662252 |
| CAMR_REPAIRED | 0.691627 | 0.675042 |
| MMP | 0.653371 | 0.640781 |
| J0 | 0.714426 | 0.702722 |
| J_CS | 0.711978 | 0.670275 |
| HIST_CS_S32_CONCH32 | 0.720666 | 0.683313 |

## 实现资格、优势与风险

| gate | status |
| --- | --- |
| DATA_AUDIT.json | PASS |
| DATA_LEAKAGE_AUDIT.json | PASS |
| OPERATOR_AND_PIPELINE_TESTS.json | PASS |
| NATIVE_SOURCE_AUDIT.json | PASS |
| NATIVE_QUALIFICATION.json | PASS |
| INDEPENDENT_VALIDATION.json | PASS |
| CHECKPOINT_REPLAY.json | PASS |
| BOOTSTRAP_AUDIT.json | PASS |
| CONTRAST_POINT_AUDIT.json | PASS |
| FIT_AND_REFERENCE_COUNTS.json | PASS |
| BUDGET_VERIFICATION.json | PASS |
| PROBE_COVERAGE_AUDIT.json | PASS |
| MMP_TOKEN_DIAGNOSTICS.json | PASS |

全部300 checkpoints的五种模式回放最大绝对误差=0，独立主效应重算最大误差=2.38419e-07。独立指标逐折与scikit-survival一致，配对区间同统计对象，外折标签突变在重建Data及完整内层短拟合期间保持隔离。技术资格不代替预后/生物机制效能。

优势：保留Clinical及原始RNA主效应；分离RNA/CNA证据与缺失；明确非零固定条件、严格两两及错配重训控制；患者级完整风险集、fresh outer、独立指标及配对统计可审计。

风险：65个DSS事件、反复开发的单队列、通路标量压缩、缺失/平台混杂、条件增量估计不稳定、预算门不证明全局最优，以及不同方法辅助任务/计算差异；本次10臂触顶均0，不将潜在截断写成本次已发生。无组织学原图核验和未触碰外部队列。算子/数值/执行计数置信度高，唯一生物学因果解释与外部效能置信度低。

**最终执行判断：MECHANISM_NOT_SUPPORTED_IN_THIS_ARCHITECTURE。** 完成本轮即停止自动扩展，不追加深度/rank/归一化/endpoint搜索；未证明一般Clinical前置或三模态互补不可能，也不把CP公式本身当新理论。

**拓展思考／你可能想知道：** Clinical改变兼容度、兼容度改变预测、预测增加Clinical+病理条件下的泛化排序，是三道不同证据门；任何一道单独通过都不能替代其余两道。
