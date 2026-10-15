# Phase15-R0 复现

父提交：`b06680da425527c1d1751b15ac27254eb2c69bb6`；本轮只新增目录与文件，不覆盖Phase14结果。

## 依赖

- Python3.12、torch2.5.1+CUDA、scikit-survival0.23.1，沿用Phase13独立评测依赖目录。
- 私有输入：`/root/phase7a_foundation_headroom_v1/private_dataset.npz`，Phase12/13固定患者分区、基因宇宙与预处理来源，Phase13 endpoints及原临床文件。
- 原强anchor的OOF及SHA用于CS_LEGACY原样对照。患者级材料不公开。

## 执行

```bash
cd /root/phase15_complement_repo
export PYTHONPATH=/root/phase10_trimodal_eval_deps
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python scripts/gate_phase15.py
python scripts/launch_phase15.py
# 等待90流程和EVALUATION_DONE.json，不能以日志/checkpoint存在替代完成
python scripts/audit_phase15_mechanisms.py
python scripts/report_phase15.py --finish
```

调度器先准备30组×inner/outer=60个anchor缓存，再执行90个基因/seed/外折/终点完整流程，最后自动独立评测。单元可断点续跑，只复用source/hash一致的正式产物。
31/47/83主要控制内部交叉拟合分块；凸求解器没有神经初始化随机性。相同RNA-only结果不是独立重复证据，bootstrap也不把seed当新患者。

`PRIVATE=/root/phase15_complement_private` 保留患者级风险、pair masks、教师、预处理与系数；这些不传到公开仓库。小型GPU pair求解与CPU Cox/临床嵌套选择并行12进程运行，凸优化用梯度验收，无神经网络epoch问题。

## 关键边界

1. 同一教师留出块内患者对；双方标签都不参与该教师/其内层选择。普通单患者OOF后跨教师拼pair并不满足此要求。
2. 全训练anchor与交叉拟合anchor的训练集规模不同，差异单独记录；不能用outer错误组重定义任务。
3. scalar RNA教师重新拟合子折缺失/缩放；基因宇宙固定于pipeline训练分区无监督选择。不是外折标签选择基因。
4. 表中1530为逐折模型评估记录，包含复用基线，不等于1530个新拟合神经网络。每seed先拼OOF，不能将15 cells称为seed mean。
5. 一维/512D对照改变容量和正则化几何；收益差异不能直接归因为参数量或事件不足。所有区间固定已训练模型，不包含重训练方差。
6. 错误组准确率、修复/损伤、teacher shift和selection stability均为模型锁定后的诊断，不是训练反馈。其区间未作为主确认检验。

清理仅涉及可重下载缓存和被正确修复替代的失效warmup，正确数据/实验哈希未变。清理摘要在公开DISK_CLEANUP.json，旧失效目录详细删除证据保留私有目录。
