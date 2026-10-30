# B0–B7 统一生存评分器核验

日期：2026-10-30。状态：**数学合成核验完成；真实队列尚未评分**。

本次新增独立评分器，不修改历史实现或结果。完成 28 个测试用例，其中 **27 通过、1 跳过**；没有训练任何真实 DSS/PFI 风险模型，也没有读取患者预测。

## 固定合同

- 高风险分数表示更早事件；完整训练风险集采用 Efron ties，每个事件的负对数部分似然。事件时点的删失患者保留在 Cox 风险集。
- Harrell 对同时间事件—删失患者计可比较对，同时间事件—事件不计；风险差绝对值不大于 `1e-8` 计半分。
- Uno60 使用当前 fit-train 的 reverse-KM：同时间先移出事件，再计算删失跳变，采用观测时点的跳变后概率；事件资格为 `time < 60`。IPCW 是 `1/G(t)^2`，不裁剪零概率，不使用测试删失分布。
- 每折返回独立分子、分母；每个 seed 先加总各折分子分母，再等权平均 seed。绝不比较跨折风险。
- `tau_support` 独立记录 60 月支持；不足时执行门禁应报不支持/NA，不能仅因较早事件有支持就宣称 Uno60 合格。

## 独立参照与实际测试

已下载并固定 [官方 v0.25.0 metrics 源码](https://github.com/sebp/scikit-survival/blob/v0.25.0/sksurv/metrics.py)及 [nonparametric 源码](https://github.com/sebp/scikit-survival/blob/v0.25.0/sksurv/nonparametric.py)。官方文件只保留在本地临时参考目录，不随本实现分发。

当前环境没有编译版 scikit-survival。实际执行其未修改的 concordance/count/KM 核心函数与估计器主体，使用最小合成输入验证适配器替代外层 endpoint 检查。100 组合成随机样本完成 200 次加权/非加权 concordance 比较及 100 次 reverse-KM/IPCW 比较；完整包 API 测试明确跳过，不能称其已执行。

另核验独立 NumPy 标量 Efron、Torch 自动梯度、数值差分、Efron/Breslow 区别、极端风险、float32 风险下的 float64 时间精度、全风险集与错误微批损失的不等价、零/未知删失支持以及独立折分数平移。数值差分最大误差测试门槛为 `5e-10`，官方 concordance 比较绝对误差门槛为 `2e-15`。

## 历史实现边界

1. `src/phase8_v0/cox.py:km_left` 在精确相同时间使用左极限，合成反例中得到 1，而官方 post-time 值为 2/3。该差异可能影响权重；本次未在 TCGA 实物上量化其影响。
2. Phase8-V0R 修复了 Uno 分子/分母的聚合，但读取保存的患者权重；仅修复聚合不意味着旧权重的时间约定也已改变。
3. Phase10 及 BG-BEN R4/R8 的现有评分核心已经采用 post-time reverse-KM 和相同可比较对定义，不能把全部历史评分器统一认定为错误。新实现对超出正尾部支持的查询额外拒绝外推。
4. 历史 Efron log-weight 实现保留；本次新增 Torch 与独立标量 oracle，不重新拟合历史基线。

## 接口与复现

`harrell_components`、`uno60_components` 返回 `PairComponents`（含分子、分母、可比较对数、风险 ties 与同时间事件—删失对数）；`evaluate_fold` 返回两指标及其分子分母和 tau 支持；`aggregate_folds` 只聚合逐折分子/分母；`efron_cox_loss` 为可微完整风险集损失，`efron_numpy` 为独立标量参照。

运行：`python -m pytest tests/test_semantic_b0b7_survival.py -q`。官方源码核心对照依赖两个固定哈希的本地参考文件；缺失时测试明确跳过，其下载 URL 与 SHA 均记录在 JSON 报告中。

**优势：** 风险方向、ties、删失支持与逐折聚合统一且可审计。**限制：** 合成测试不能替代患者对齐、真实训练与旧 OOF 复算；不能从该核验推断 DSS 预测或语义机制增益。
