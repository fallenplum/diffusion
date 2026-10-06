# Phase4A-R 修复实验结果

135 个唯一正式 cells 已完整；每个 seed 拼接 5-fold OOF 后计算指标，再对 3 seeds 求均值。M4 为三个独立 shuffle 指标的均值，没有平均 logits。

## 核心结果

|模型|3-seed OOF G_TNM|T BA|N BA|M BA|
|---|---:|---:|---:|---:|
|M0|0.428734|0.261882|0.489545|0.534775|
|M1|0.526682|0.340181|0.616532|0.623333|
|M2|0.529401|0.337990|0.606225|0.643988|
|M3|0.530168|0.360595|0.603858|0.626052|
|M4|0.528076|0.354677|0.604610|0.624942|
|M5|0.528868|0.350745|0.602366|0.633494|
|M6|0.534245|0.351510|0.604500|0.646725|

## 同 seed 配对差值

|比较|seed|ΔG_TNM|患者配对 bootstrap 95% CI|
|---|---:|---:|---|
|M3_minus_M4_shuffle_mean|20260904|-0.004078|[-0.017466, 0.008505]|
|M3_minus_M2_naive|20260904|0.003015|[-0.025914, 0.032865]|
|M3_minus_M6_fixed_query|20260904|0.000023|[-0.010467, 0.010988]|
|M3_minus_M5_no_mask|20260904|0.004959|[-0.009052, 0.021121]|
|M3_minus_M4_shuffle_mean|20260905|0.007799|[-0.008072, 0.024270]|
|M3_minus_M2_naive|20260905|-0.001488|[-0.029983, 0.025539]|
|M3_minus_M6_fixed_query|20260905|-0.006307|[-0.023583, 0.008988]|
|M3_minus_M5_no_mask|20260905|-0.002976|[-0.012942, 0.006373]|
|M3_minus_M4_shuffle_mean|20260906|0.002555|[-0.009361, 0.015424]|
|M3_minus_M2_naive|20260906|0.000775|[-0.024184, 0.027029]|
|M3_minus_M6_fixed_query|20260906|-0.005946|[-0.020586, 0.009061]|
|M3_minus_M5_no_mask|20260906|0.001917|[-0.011380, 0.016638]|

## 修复与诊断

- 选择：inner-train 权重的 balanced validation CE，EMA α=0.2，至少10epochs，patience12；refit 使用 outer-train 权重和相同 AdamW/loss 规则。
- 正则：patch normalized entropy λ=0.001；pathway token off-diagonal squared cosine λ=0.001。M0/M2 的 masked tokens 保留 pathway slot。
- best epoch 中位数 24.0，范围 10–90。达到10epochs下界的 cells 为 31/135。
- 训练重建 MSE 首末下降的 cells：84/105。这是在线随机mask训练损失，不能替代固定mask测试增益。
- 测试 reconstruction MSE/Pearson、真实 raw cosine、中心化 correlation、effective rank、跨seed attention稳定性均单独列出。没有根据这些结果调整参数。

## prompt 的直接回答

1. 无 DSS supervision；原训练 loader 读取 DSS 仅检查固定队列（65events），DSS 不进入损失、epoch选择、表示或评估；汇总不读取 DSS。
2. 官方 MSigDB 2024.1.Hs 50 Hallmark，实际匹配4191个唯一genes，最小pathway32genes。
3. M3/M4 同参数量、相同初始化seed规则与训练规则；shuffle保持每个pathway规模和每gene度数。M6 的cond参数保留但在forward禁用，名义参数量匹配。
4–7. M0 RNA、M1 Pathology、M2 Naive、M3 DiseaseState 的 G_TNM 见上表。
8. True−shuffle mean ΔG_TNM=0.002092。
9. True−shuffle 的CI下界全正：False；正方向seeds：2/3。
10. T/N/M mean BA中正向任务：2/3。
11–12. M3−M2、M3−M6每seed差值与CI见配对表；均值为正不能独立证明稳定超越。
13. M3−M5(no mask) mean ΔG=0.001300；M3/M0/M2固定测试mask重建及M3−M5CI见CSV。
14. M3 raw token cosine均值=0.958190；跨患者variance floor collapse=False。方差不为零不证明pathway语义。
15. M3 attention top10% Jaccard=0.366572，Spearman=0.805531；稳定性不等于生物学grounding。
16. **STRUCTURE_HELPFUL_SEMANTICS_UNSUPPORTED**。
17. Phase4B：**不允许**。
18. 只有完整knowledge gate支持才允许下一阶段；届时疾病状态encoder冻结，仅训练小adapter/survival head，保留Clinical/Pathology/RNA signal paths。本次未运行transfer。

## 优势、风险与置信度

- 优势：类别均衡、平滑选择、matched controls、真正seed OOF和患者配对bootstrap，减少原先raw BA epoch选择噪声与mask identity不公平。
- 风险：稀有T/M类别仍限制统计精度；两项正则可能制造多样性而非真实语义；多项修复联合改变，不能拆分因果贡献。所有seed共享患者，CI不代表独立cohort泛化。
- inner RNA预处理统计来自outer-train（含inner-val features，无labels），inner验证并非完全feature隔离；validation CE按batch加权求均值，没有全局class-weight归一化。公开记录这一实现边界。
- 置信度：完成性和OOF覆盖为高（机械核验）；机制支持程度依据配对CI与seed一致性，见DECISION，不能由单一均值推断。

## 拓展思考

若M3−M6和True−shuffle缺乏稳定优势，增加cross-attention深度没有直接证据支持。应先区分监督分辨率不足、pathway表示压缩与设计限制。

## 你可能想知道

训练loss下降只证明优化器能拟合辅助目标；是否提供可迁移信息仍须固定mask测试重建、matched shuffle及guided−fixed比较共同支持。

## 产物边界

患者IDs、logits、attention ranks、embeddings、checkpoint均留在私有runtime；GitHub只提交源代码、官方GMT和聚合报告。烟雾/探索cell不进入正式结果。
