# R8 复现顺序

需要先获得私有数据合同中列出的既有 R6 源状态、patch 缓存、三内层预处理、CP 风险和 RNA512 教师。公开仓库不含患者级输入、索引、标签、风险、权重或访问凭据；命令中的私有位置使用占位符。

固定配置：`configs/bgb_en_r8/inner_r6_s7_v1.json`。Python 依赖为 NumPy、SciPy、PyTorch 及既有 R6 项目环境。

## 顺序

1. 运行新模块的合成资格测试，不将其计入正式实验。正式 source/target 不因测试而优化。
2. 在独立私有命名空间运行 `scripts/bgb_en_r8_run.py --mode features`。使用既有相同资产参数；必须得到六组真实 READY 及 `R8_FEATURE_QUALIFICATION.json`。此阶段新增监督拟合为零。
3. 所有计算、报告代码、配置及测试部署完毕后运行 `--mode lock`。LOCK 绑定全部新 R8 科学脚本、测试、源文件、固定配置、旧 R6 源文件和六组资格字节。其后不得修改这些字节。
4. 对每个分区运行 `--mode references --partition P`；随后按固定矩阵运行 `--mode fit --arm A --partition P --seed S --device cpu`。允许至多两个 CPU 进程并发，每进程 BLAS/OMP 线程数2。行政调度器不属于科学脚本，不得修改 LOCK。
5. 81 个 COMPLETE 全部存在且无错误单元后，运行 `--mode replay --device cpu`，独立实际回放81个保存模型和18个引用。不得只检查计数、SHA或复制旧指标代替前向重算。
6. 运行 `python scripts/bgb_en_r8_report.py --private <R8_PRIVATE> --config configs/bgb_en_r8/inner_r6_s7_v1.json --output reports/bgb_en_r8`。仅在资格、LOCK、全部拟合及重放通过时生成完整结果标记。

`--mode features/lock/fit/references/replay` 使用相同必需参数：`--assets <PRIVATE_ASSETS> --private <R8_PRIVATE> --primary-private <OLD_PRIMARY> --source-private <OLD_SOURCE> --config <R8_CONFIG> --source-config <OLD_SOURCE_CONFIG> --r6-private <OLD_R6_TARGET> --r6-config <OLD_R6_CONFIG>`。正式 fit/replay 只消费新 scoped409 特征缓存，不重新打开完整旧队列。

## 验收

正式 ledger 应为81次凸拟合、0次神经更新、0次源/教师/系数重训，27个不同确定性问题按3个名义 seed重复。独立重放要求风险误差不超过 `2e-10`、排序贡献变化为0、KKT与训练SD证书通过；任何失败保留原目录，禁止自动重试覆盖。

匿名结果包含逐分区 DSS/PFI 指标、固定直接配对的修复/损伤贡献、训练成本、冻结干预和决策。PFI 不参与选择；新 outer 测试保持 `NOT_RUN`。本轮结果不得反馈修改并行 SPGEL 源训练配方。
