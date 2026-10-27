# BG-BEN R3 内层正式原型结果

状态：**COMPLETE**；已完成 72/72 个 unique cell。

数据层级：outer fold 1 的训练人群内，3 个固定内层分区 × 2 个 neural seeds（31、47）。这些是重复开发原型，不是完整5外折确认或独立队列。

每个 seed 先相加三个分区的 concordant/eligible 分子分母，再计算 Harrell 和 Uno60；最后均值两个 seed。绝不拼接跨分区风险作排序。缺少任一规定单元时，该臂平均分数留空。

| 模型 | 状态 | 完成unique cells | Harrell | Uno60 | 训练−验证 H | 训练ΔH/原参考 | 验证ΔH/原参考 | 保存诊断参考 | 参数量均值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | COMPLETE | 3 | 0.816796 | 0.783518 | — | — | 0.000000 | F0 | 102.000000 |
| F1 | COMPLETE | 3 | 0.825899 | 0.787859 | — | — | 0.009103 | F0 | 1.000000 |
| F2 | COMPLETE | 6 | 0.826127 | 0.789922 | -0.047860 | 0.007464 | 0.000228 | F1 | 88932.000000 |
| F3 | COMPLETE | 6 | 0.825330 | 0.789438 | -0.030575 | 0.023953 | -0.000569 | F1 | 92292.000000 |
| F4 | COMPLETE | 6 | 0.824078 | 0.789330 | -0.029777 | 0.023499 | -0.001821 | F1 | 92292.000000 |
| F5 | COMPLETE | 6 | 0.825558 | 0.790622 | -0.032619 | 0.022136 | -0.000341 | F1 | 93604.000000 |
| F6 | COMPLETE | 6 | 0.824989 | 0.789273 | -0.031794 | 0.022392 | -0.000910 | F1 | 93604.000000 |
| F7 | COMPLETE | 6 | 0.824761 | 0.788784 | -0.030318 | 0.023641 | -0.001138 | F1 | 93604.000000 |
| F8 | COMPLETE | 6 | 0.825671 | 0.790359 | -0.030575 | 0.024293 | -0.000228 | F1 | 123300.000000 |
| F9 | COMPLETE | 6 | 0.824875 | 0.789720 | -0.028899 | 0.025173 | -0.001024 | F1 | 104196.000000 |
| F10 | COMPLETE | 6 | 0.823623 | 0.788822 | -0.029804 | 0.023016 | -0.002276 | F1 | 94510.000000 |
| F5_WRONG_QUERY | COMPLETE | 6 | 0.826240 | 0.790993 | -0.031740 | 0.023697 | 0.000341 | F1 | 93604.000000 |
| F5_SHUFFLED_MEMBERSHIP | COMPLETE | 6 | 0.824761 | 0.790872 | -0.032645 | 0.021313 | -0.001138 | F1 | 93604.000000 |

## 预定配对比较

| 比较 | 状态 | ΔHarrell | ΔUno60 | H正向seeds/2 | U正向seeds/2 | H正向分区×seed/6 |
| --- | --- | --- | --- | --- | --- | --- |
| F0-F1 | COMPLETE | -0.009103 | -0.004341 | 0 | 0 | 0 |
| F2-F1 | COMPLETE | +0.000228 | +0.002063 | 1 | 2 | 3 |
| F3-F1 | COMPLETE | -0.000569 | +0.001579 | 1 | 2 | 3 |
| F4-F1 | COMPLETE | -0.001821 | +0.001470 | 0 | 2 | 1 |
| F5-F1 | COMPLETE | -0.000341 | +0.002763 | 1 | 2 | 3 |
| F6-F1 | COMPLETE | -0.000910 | +0.001414 | 0 | 2 | 2 |
| F7-F1 | COMPLETE | -0.001138 | +0.000924 | 0 | 2 | 3 |
| F8-F1 | COMPLETE | -0.000228 | +0.002499 | 1 | 2 | 2 |
| F9-F1 | COMPLETE | -0.001024 | +0.001861 | 1 | 2 | 2 |
| F10-F1 | COMPLETE | -0.002276 | +0.000962 | 1 | 1 | 3 |
| F5_WRONG_QUERY-F1 | COMPLETE | +0.000341 | +0.003134 | 1 | 2 | 4 |
| F5_SHUFFLED_MEMBERSHIP-F1 | COMPLETE | -0.001138 | +0.003012 | 0 | 2 | 1 |
| F1-F0 | COMPLETE | +0.009103 | +0.004341 | 2 | 2 | 6 |
| F5-F2 | COMPLETE | -0.000569 | +0.000700 | 1 | 2 | 2 |
| F5-F3 | COMPLETE | +0.000228 | +0.001185 | 1 | 2 | 3 |
| F5-F4 | COMPLETE | +0.001479 | +0.001293 | 1 | 1 | 4 |
| F6-F5 | COMPLETE | -0.000569 | -0.001349 | 1 | 0 | 2 |
| F7-F5 | COMPLETE | -0.000797 | -0.001839 | 0 | 0 | 2 |
| F8-F5 | COMPLETE | +0.000114 | -0.000264 | 1 | 1 | 3 |
| F9-F5 | COMPLETE | -0.000683 | -0.000902 | 1 | 0 | 3 |
| F10-F5 | COMPLETE | -0.001934 | -0.001801 | 1 | 0 | 1 |
| F5-F5_WRONG_QUERY | COMPLETE | -0.000683 | -0.000371 | 0 | 1 | 1 |
| F5-F5_SHUFFLED_MEMBERSHIP | COMPLETE | +0.000797 | -0.000249 | 1 | 1 | 4 |

## 结果解释与开销边界

- 20/40轮仅用于学习曲线诊断；正式原型成绩固定取80轮，未根据这些验证点改选checkpoint。
- 保存的risk SD比与repair/damage使用各自原始风险参考，不能用表中paired_reference代替：P0/O0/O1/O2/F0/F1均参照当前训练分区full LOCAL，即验证集上的F0；R1神经臂参照CS，R2神经臂参照RNA512/O0，R3神经臂参照F1。stored_risk_diagnostics_reference列明确记录该来源。
- paired_delta_H/paired_delta_U60由候选与对照的concordance分子分母分别聚合后直接相减；它们不是原repair/damage字段相减，也不自动等于相对保存诊断参考的增量。
- 学习曲线中训练/验证ΔH=repair−damage、ΔUno60=Uno_repair−Uno_damage，均相对各自保存的参考风险；训练使用cross-fitted offset，验证使用完整当前训练分区拟合的offset。原始train−valid H只作描述，train H低于valid H不能据此判定欠拟合。
- P0、O0、O1、各O2、F0、F1是确定性单元，只拟合seed31；seed47使用同一引用，未计为新的优化。
- 每个prepare分区为9个teacher Cox＋4个full Cox＋1个same-teacher β优化，共14个监督优化；原upstream_optimizer_fits字段的13只计Cox。
- 核心186cells全部完成时为12个无新优化baseline引用＋162个神经拟合＋12个新凸Cox拟合；另加3分区42个上游优化，共216个监督优化。解析风险decoder单列，外部ST拟合另见独立收据。
- 时间列为各进程耗时之和，不是并行调度的实际makespan。显存为对应进程记录的PyTorch峰值。
- 小于0.003的改善仅作很弱方向性线索；不生成CI、bootstrap或p值，也不把两个seed称为两个验证队列。

F5是本项目双向Attention控制，**不是官方SurvPath精确复现**；虽已核验公开源码，本轮尚未直接port其原生tokenizer/attention/离散生存头。

真实重训比较与冻结干预分别列出。冻结wrong-query/mean-query/whole-omics/attention-permutation/interaction-off/direct-off解释已训模型依赖，不能替代重新训练消融。

神经R3收据中的repair/damage参照F1；冻结控制中的repair/damage也参照F1。仅COMPLETE匿名汇总不能重建full模型与控制之间的repair/damage交集。若另有经saved-risk独立回放的MECHANISM_REPLAY.json，则full_control_*列给出该直接配对分解；缺少时明确NOT_RUN，不把相对F1的分解误当full-control分解。

无已通过资格的局部组织学真值时，不将attention/slot变化命名为医学事件或细胞来源。外部ST与文本追加候选未完成时不计入核心矩阵结果。
