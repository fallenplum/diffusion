# 复算与运行

## 依赖与私有输入

真实版本见SOURCE_REGISTRY；Python3.12.3/torch2.5.1+cu124、sksurv0.23.1隔离目录/root/phase10_trimodal_eval_deps。设置PYTHONPATH该目录、CUBLAS_WORKSPACE_CONFIG=:4096:8、OPENBLAS/OMP/MKL_NUM_THREADS=2。大矩阵与519CONCH H5及Phase11嵌套缓存沿用原只读路径，完整hash见DATA_AUDIT。

## 新输出隔离

/root/phase12_c4_private/；不覆盖Phase11。复跑请使用新目录并修改run_phase12中PRIVATE及对应独立审计路径；同源COMPLETE会skip，禁止覆盖历史患者预测。

```bash
python -m unittest tests.test_phase12_c4 -v
python scripts/prepare_phase12.py
python scripts/audit_phase12_sources.py
python scripts/gate_phase12.py
python scripts/schedule_phase12.py
python scripts/validate_phase12_independent.py --statistics
python scripts/replay_phase12.py
python scripts/audit_phase12_statistics.py
python scripts/audit_phase12_test_sensitivity.py
python scripts/report_phase12.py
python scripts/package_phase12.py
python scripts/check_phase12_delivery.py
```

单cell：`python scripts/run_phase12.py --arms R0_512 --seed 31 --fold 1`。仅重建原OOF指标：`python scripts/validate_phase12_independent.py --statistics`。新矩阵/比较以phase12配置的arms、primary、secondary字段为准；合并协议中primary_comparisons/exact_aliases/patch_arms等旧Phase11字段只为历史继承记录，本轮不执行。发布分支允许Phase11 parent SHA的后继提交，仍逐文件核对原历史源码不变。矩阵配置中6个主要线性对比包括配对差中差；不能从不同risk尺度跨fold比较患者。原始临床DSS标签重建由独立raw_source脚本完成。

训练入口和原Phase11网络forward真实交付，不只PASS文件；授权外部审计员可读取私有OOF/checkpoint重新运行。
