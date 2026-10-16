# Phase18A 复现入口

## 环境与私有前提

服务器 repo 工作目录执行，Python `/root/miniconda3/bin/python`；环境包版本与父 Phase17R `ENVIRONMENT.json` 经 historical audit 核对。完整父数据、固定视觉 cache、fold manifests 与 checkpoints 为私有资产，不随公开仓库发布。只取得源码无法凭空重建这些资产。

```bash
export PYTHONPATH=.:/root/phase10_trimodal_eval_deps
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pytest tests/test_phase18a_readout.py tests/test_phase18a_model.py -q
```

## 按证据依赖运行

1. 核对 `PROTOCOL_LOCK.json` 与实际 source hash，保留 parent 历史文件；算子/完整模型 CPU/CUDA 输出与梯度通过后登记 `OPERATOR_PARITY.json`。benchmark 是合成实现核验，不是正式训练。
2. `python scripts/validate_phase18a.py --historical-only --replay`，随后 `--aggregate-historical --shards 1`。并行时每个 shard 显式提供 `--shard k --shards n`，聚合必须覆盖全部且无重复。F0/F1/A3 来源、SHA、患者分区、epoch、预处理、软件及四条件重放全部通过，才允许历史引用。
3. `python scripts/audit_phase18a_leakage.py` 运行两个短拟合，内存中只改 outer-test 标签；没有 smoke 权重/风险文件。生产执行曾因比较 bool buffer 的 audit helper 报错而重做该检查，累计四个真实短拟合、八次 optimizer steps；不能把复现一次的两次写成生产总数。
4. `python scripts/start_phase18a.py` 启动可恢复调度，先全部30 inner，锁定 epoch，再30 fresh outer。`WORKERS.json` 只调整并发，不更改数学定义或预算；正式生产未盲目增加 epoch。
5. `python scripts/validate_phase18a.py --historical-probes-only`，再 `--aggregate-historical-probes --shards 1`；冻结父权重，补齐患者级/WSI级及两种位置梯度目标。旧未完成探针 shard 已标为 superseded，不当完成，也不计优化。
6. `python scripts/validate_phase18a.py --new-replay` 覆盖60新 checkpoint（含 inner/outer）；`python scripts/validate_phase18a.py` 独立重建 OOF/anchor、核对 sksurv、重算20,000次患者配对区间和冻结 PFI。
7. `python scripts/report_phase18a.py` 只在各证据门 PASS 后生成结果和决策。可先用 `--protocol-only` 生成设计/复现说明，不输出未产生的成绩。

禁止在已经完成的正式 run 上为了“复现命令”无意义重复短拟合；如确实重做，应增加真实执行计数。已有正式 checkpoint 只有来源与文件摘要一致才复用，不把日志或文件存在当完成。

## 公开与私有边界

`CELL_MANIFEST.csv`、逐折/逐seed指标、配对区间与诊断汇总可审阅；患者级 OOF、原始 feature/gene identities、checkpoint 和对应私有 manifest 不上传。正式源码锁与评估/报告源码分别记录；修复评估 helper 不静默改变训练模型或旧报告。完成 Phase18A 后停止自动扩展。
