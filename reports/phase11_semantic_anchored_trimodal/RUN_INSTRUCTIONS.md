# 运行与审计

## 环境

Python3.12.3 / torch2.5.1+cu124 / numpy2.1.3 / pandas2.2.3 / scipy1.14.1 / sklearn1.5.2 / h5py3.16.0 / scikit-survival0.23.1；完整依赖见SOURCE_REGISTRY。设置OPENBLAS/OMP/MKL_NUM_THREADS=2、CUBLAS_WORKSPACE_CONFIG=:4096:8、PYTHONPATH包含隔离sksurv依赖目录。

## 私有输入约定

/root/phase7a_foundation_headroom_v1/private_dataset.npz；/root/autodl-tmp/phase10_conch；/root/phase10_trimodal_private/cache/private_cohort_manifest.csv与private_fixed_patient_folds.csv；/root/phase6a_titan_runtime/cache_v1/patient_representations.npz。sha见来源表。原H5、矩阵和标签不在公开包，先取得用户授权。改变根路径时同时审查entrypoints中的PRIVATE和来源路径；不得拿不同队列悄悄替代。

## 顺序

```bash
python -m unittest tests.test_phase11_semantic -v
python scripts/prepare_phase11.py
python scripts/audit_phase11_sources.py
python scripts/gate_phase11.py
python scripts/schedule_phase11.py --workers 12
python scripts/replay_phase11.py
python scripts/audit_phase11_statistics.py
python scripts/report_phase11.py
python scripts/package_phase11.py
```

单cell：`python scripts/run_phase11.py --arms B0 --seed 31 --fold 1`。同源码已有COMPLETE会skip；重跑请使用全新独立输出目录并更新PRIVATE常量，不能覆盖旧OOF。只重建评测：`python scripts/validate_phase11_independent.py --statistics`。每stage evaluator --family A/B/C/D/P。`recheck_phase11_papers.py`在本地已存在固定官方git clone时运行，server归档其结果；不是训练依赖。

源码改变即拒绝旧cell/source hash；不能改HARD_GATES为PASS。P0必须归档旧版并按影响重训。父分支当前SHA9040f5c已fetch实际核对；不以旧报告分数作程序验证。

## 完成解释与最终校验

主要统计及回放完成后运行 `python scripts/audit_phase11_anchor_comparisons.py`；生成报告后运行 `python scripts/interpret_phase11.py`、`python scripts/package_phase11.py`、`python scripts/check_phase11_final_contract.py`。重新生成解释前先重跑report，避免重复追加。
