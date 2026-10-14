# 运行与复算

Python3.12.3/torch2.5.1+cu124/numpy2.1.3/scipy1.14.1/pandas2.2.3/sklearn1.5.2/sksurv0.23.1。环境PYTHONPATH=/root/phase10_trimodal_eval_deps，CUBLAS_WORKSPACE_CONFIG=:4096:8，OPENBLAS/OMP/MKL_NUM_THREADS=2。父提交96a25fd93da8ff39c7b081f078edd3880b0803af及Phase12私有目录只读依赖；原CDR来自SOURCE_REGISTRY公开链接并仅服务器保存。

正式文件与源hash绑定；复跑使用隔离输出目录，不覆盖原记录。原已有COMPLETE不重写。

```bash
python -m unittest tests.test_phase13_rna -v
python scripts/prepare_phase13.py
python scripts/gate_phase13.py
python scripts/schedule_phase13.py
python scripts/validate_phase13_independent.py
python scripts/replay_phase13.py
python scripts/audit_phase13_statistics.py
python scripts/report_phase13.py
python scripts/package_phase13.py
```

表征与head按producer/fold各自训练，不能拼接独立latent坐标做训练。ARTIFACT_HASH_INDEX提供匿名逐cell索引；私有formal/cells包含checkpoint/OOF/manifests路径与全部hash。模型forward和统计实现均实交付；授权外部审计员可在服务器重新复算。
