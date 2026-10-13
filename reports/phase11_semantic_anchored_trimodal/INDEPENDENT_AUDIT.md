# 独立验收

450正式记录由隔离评价器直接读私有OOF/time/event/manifest重算，并逐cell核对scikit-survival。指标脚本不import训练数学或特征模块。另有train-only独立原矩阵重建50输入，420实际fit checkpoint回放，其中75Ridge保存系数与独立事件循环梯度通过；345神经checkpoint复放及模态/padding属性通过。25次bootstrap显式重复展开核对450项差，非先平均风险。

模型forward复放使用交付网络，不是第二个完整神经实现；独立小矩阵oracle、query shape hook和streamed全loss梯度校验负责其数学路径。公开读者仍需授权私有输入，不能从PASS文字独立复算。

## 强锚点新拟合核对

|arm|seed|fold|lambda|independent_fresh_fit_iterations|gradient_max|coefficient_L2_error|strong_convexity_L2_bound|max_risk_error|scope|
|---|---|---|---|---|---|---|---|---|---|
|A1|31|1|1.00000|20|0.00000|0.00000|0.00000|0.00000|5 unique outer-train fits; checked against all15 initialization records|
|A1|31|2|10.00000|28|0.00000|0.00000|0.00000|0.00000|5 unique outer-train fits; checked against all15 initialization records|
|A1|31|3|1.00000|25|0.00000|0.00000|0.00000|0.00000|5 unique outer-train fits; checked against all15 initialization records|
|A1|31|4|10.00000|31|0.00000|0.00000|0.00000|0.00000|5 unique outer-train fits; checked against all15 initialization records|
|A1|31|5|1.00000|20|0.00000|0.00000|0.00000|0.00000|5 unique outer-train fits; checked against all15 initialization records|
|A1|47|1|1.00000|20|0.00000|0.00000|0.00000|0.00000|5 unique outer-train fits; checked against all15 initialization records|
|A1|47|2|10.00000|28|0.00000|0.00000|0.00000|0.00000|5 unique outer-train fits; checked against all15 initialization records|
|A1|47|3|1.00000|25|0.00000|0.00000|0.00000|0.00000|5 unique outer-train fits; checked against all15 initialization records|
|A1|47|4|10.00000|31|0.00000|0.00000|0.00000|0.00000|5 unique outer-train fits; checked against all15 initialization records|
|A1|47|5|1.00000|20|0.00000|0.00000|0.00000|0.00000|5 unique outer-train fits; checked against all15 initialization records|
|A1|83|1|1.00000|20|0.00000|0.00000|0.00000|0.00000|5 unique outer-train fits; checked against all15 initialization records|
|A1|83|2|10.00000|28|0.00000|0.00000|0.00000|0.00000|5 unique outer-train fits; checked against all15 initialization records|
|A1|83|3|1.00000|25|0.00000|0.00000|0.00000|0.00000|5 unique outer-train fits; checked against all15 initialization records|
|A1|83|4|10.00000|31|0.00000|0.00000|0.00000|0.00000|5 unique outer-train fits; checked against all15 initialization records|
|A1|83|5|1.00000|20|0.00000|0.00000|0.00000|0.00000|5 unique outer-train fits; checked against all15 initialization records|

A1_INDEPENDENT_REFIT.csv记录独立事件循环目标的BFGS零初始化拟合。五个唯一outer-train问题核对15份记录，不更新任何正式checkpoint、OOF或模型选择。系数差通过λ强凸性与两侧梯度残差给出的可计算界验收。
