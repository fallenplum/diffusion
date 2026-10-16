# Phase19 复现与数据边界

训练来源：`ce14bd3053f51bbf277d21392419da70c41208581bb9580d78ca546f953873d5`。父提交：`099f3d8cd0815a1f994f22b5ce5801b1f4621105`。正式模型来自本轮fresh fit，不复用旧神经检查点。

## 前置条件

使用本项目既有、合法授权的私有512患者DSS/PFI矩阵、固定患者折、Phase12训练分区预处理缓存、Phase17物理足迹UNI2-h视觉缓存及经验证的历史LOCAL参照。患者数据、真实标识、PCA/原型状态、OOF和权重不包含在公开仓库，不能只下载代码就假定数据齐全。原图形态核验仍缺失。

环境为既有GPU环境及scikit-survival评测依赖；官方核心按`configs/phase19_baseline_sources.json`固定commit/文件内容校验。模型/输入/训练和适配边界见PROTOCOL.md与BASELINE_REGISTRY，不把本轮Cox适配称原生论文训练复现。

## 执行顺序

在仓库根目录设置`PYTHONPATH=.:/root/phase10_trimodal_eval_deps`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`及OMP/MKL/OPENBLAS线程数2。

```sh
python scripts/prepare_phase19.py
python scripts/schedule_phase19.py
python scripts/validate_phase19.py --replay --shard 0 --shards 4
python scripts/validate_phase19.py --replay --shard 1 --shards 4
python scripts/validate_phase19.py --replay --shard 2 --shards 4
python scripts/validate_phase19.py --replay --shard 3 --shards 4
python scripts/validate_phase19.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/diagnose_phase19.py
python scripts/report_phase19.py
```

prepare检查固定官方源、资格测试、10个role/fold预处理、20个唯一凸控制、真实短拟合及外折标签隔离；正式训练前锁最终source。scheduler依次执行150 inner与150 fresh outer，各outer精确使用对应inner所选epoch。不以PFI选择模型。本次使用上述四个互斥回放分片，每片75份检查点，共300份；可顺序执行，也可并行，但必须收齐四片再独立评估。不要混用分片总数，也不要混入其他分片方案留下的文件；collect会汇总全部CHECKPOINT_REPLAY_*.csv，重复/缺失/源hash不一致均拒绝完成。公开报告保留本次0/1/2/3四份，因此原目录复现须按相同四片完整覆盖，不能只把0改跑为shards=1后留下旧1/2/3。

缓存只在source、fit/heldout、特征、官方core与hash receipt一致时复用；不得通过删除receipt或覆盖旧结果绕过检查。GPU worker数只调度并发，不改变参数、样本、风险集或选择协议。完整患者风险形成一次Efron loss，微批不是微批Cox。

## 结果核对

逐折表先sum分子/分母生成真正3-seed指标；六主比较×两指标为Bonferroni12家族。bootstrap为固定折内共享患者计数，20,000 draws；只反映固定已训练模型的不确定性。PFI为同队列冻结风险迁移。报告程序仅在完整训练、回放、数据/指标/配对区间检查通过后生成RESULTS及decision，不生成缺失OOF或预填成绩。

复现新增拟合与历史指标引用分开计数，smoke增量账本保留失败已更新尝试。MMP只有完成缓存producer证据，没有尝试级KMeans账本，因此不能声称知道全部失败尝试数。最终只发布匿名聚合报告/代码，不发布私有患者产物。

实际Python、torch/CUDA、NumPy/SciPy/pandas/scikit-learn/scikit-survival/einops、GPU驱动与显存见本轮ENVIRONMENT.json，不直接继承父实验环境版本。只读diagnose不访问生存标签或改变训练；重复运行增加快照并改变诊断文件hash，需要重新生成report/package以保持引用一致。
