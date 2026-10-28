# R5 确定性修复版复现

基线提交：`ecd273d2036d98e40dd9e997dd45e8de9eba7947`。交付分支：`research/bgb-en-r5-readout`。最终性能仅来自新数值合同 `21ebbd65626d2fcdaf5023c184742527ba381b55091831975adda19873aa6e19`；历史v1合同为 `9d543b6b73861a163a40b8134a6e24f62094df91321c2202b11b6c0511cf123e`。

## 环境与私有输入

实际运行环境为Python3.12.3、PyTorch2.5.1+cu124、NumPy2.1.3、SciPy1.14.1、scikit-learn1.5.2、h5py3.16.0。驱动报告RTX4080SUPER与32760MiB设备容量；这里只记录驱动报告，不推断实体显卡规格。

数据与原有三个 `formal_v1` 教师制备包由数据持有人私下提供，不包含在仓库。设置 `ASSETS`、`PRIMARY` 和 `PRIVATE` 指向对应私有位置；`PRIVATE` 必须处于Git树之外，并以 `bgb_en_r5_deterministic_v2_private` 为目录名。已有原始输入和教师必须匹配 `PROTOCOL_LOCK.json`，不创建替代样本或重新挑基因。

## 正式入口

必须通过新包装器进入。旧入口会拒绝专用v2锁，防止在错误数值设置下运行。

```bash
python -B scripts/bgb_en_r5_deterministic_runtime.py --runtime-check
python -B scripts/bgb_en_r5_deterministic_runtime.py --entry qualify -- --assets "$ASSETS" --private "$PRIVATE" --primary-private "$PRIMARY" --device cuda
python -B scripts/bgb_en_r5_deterministic_runtime.py --entry run -- --assets "$ASSETS" --private "$PRIVATE" --primary-private "$PRIMARY" --device cuda --stage train --arm A4 --partition 1 --seed 31
python -B scripts/bgb_en_r5_deterministic_runtime.py --entry replay -- --assets "$ASSETS" --private "$PRIVATE" --primary-private "$PRIMARY" --device cuda --limit 1 --resume
python -B scripts/bgb_en_r5_deterministic_runtime.py --entry schedule -- --assets "$ASSETS" --private "$PRIVATE" --primary-private "$PRIMARY" --device cuda --stage all --max-workers 4 --resume
```

实际运行先完成矩阵中的A4/partition1/seed31，独立重放通过后才执行其余矩阵；该模型在63个主模型中只计一次。调度完成63个主模型后，按原锁定的Harrell主、Uno60次规则选择一个A4–A7候选；本次选中A5，再执行五种机制控制各9个模型，共45个。所有模型固定80个完整风险集Adam更新，保存20/40/80但只用80选择及报告。

```bash
python -B scripts/bgb_en_r5_deterministic_runtime.py --entry replay -- --assets "$ASSETS" --private "$PRIVATE" --primary-private "$PRIMARY" --device cuda --resume
python -B scripts/bgb_en_r5_deterministic_runtime.py --entry report -- --private "$PRIVATE" --stage final --output reports/bgb_en_r5
python -B scripts/bgb_en_r5_deterministic_runtime.py --entry publication -- --denylist "$QUALIFIED_PRIVATE_DENYLIST"
```

在本次归档执行中，R6启动收据已绑定四份R5公共证据的SHA；归档后不应原地重写它们。另行复现应使用独立目录，不改归档报告。

## 数值修复与公平性

包装器在import torch之前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，启用严格确定性算法、关闭TF32和cuDNN benchmark。25份原科学源码逐字节保持一致。新包装器、配置和报告兼容入口均进新锁，实际运行设置也进入合同；并发子进程必须回到包装器。

旧v1训练了108个模型，但保存状态独立重放95通过、13失败。三个零更新诊断检查点证明默认前向存在微量浮点差异；确定性前向自身稳定，部分仍与旧保存值不符。因此没有放宽2e-6门槛、没有重复前向直到碰巧通过、没有只重训失败臂；新版本完整执行63＋45并重新选择唯一候选，旧权重和选择未复用。旧模型、源码、收据及报告单独保留，匿名摘要见 `V1_REPLAY_FAILURE_SUMMARY.json`。

最终108/108保存模型已重新加载并实际前向，主要风险最大误差为0，排序信用改变为0；756组冻结控制风险也在同一门槛内回放。另重放原39个Cox系数模型及3个同教师β梯度，0新教师拟合。所有直接修复/损伤由同分区两条风险向量计算，不能以C-index差值代替。

每seed先相加三分区内部concordance分子/分母，再平均三个seed的比值；不拼不同分区风险排序。确定性A0/A1的18个逻辑引用不算18次优化。PFI仅使用冻结DSS风险，不参与选择；未执行bootstrap、CI或p值。

A3是共同新增风险幅度惩罚下的legacy函数重训，不声称复现历史F5优化轨迹。A7初始化严格等价A4线性初始输出，另检验真实初始batch与legacy足够接近，不声称rank4 GELU全域等同tanh。A8全模型可训练预算与A7相差7个参数，不声称相同FLOPs。

总实际R5成本为216个神经拟合、17280个更新：v1和v2各108/8640。正式v2性能分母仍仅108；旧v1不进入性能均值。四路并行资源窗口包含单A4及回放，详见 `R5_RESOURCE_SUMMARY.json`。所有患者风险、检查点、坐标、权重和凭据均留私有存储。
