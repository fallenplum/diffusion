# 复现 Phase14-R0

## 已公开

独立分支基于 Phase13 `8403d77c8891dadea075feed7bcbd2224a12f4d0`。新增源码、锁定配置、协议、逐折/每seed OOF聚合指标、配对区间、训练及梯度诊断均可审查。OOF聚合表只含模型级指标，不含患者风险。

## 私有依赖

授权服务器使用：

- `/root/phase7a_foundation_headroom_v1/private_dataset.npz`
- `/root/phase13_rna_private/endpoints.npz`、`preprocessing/`
- Phase12 patient folds/membership，Phase13 Clinical+S32 anchor cells/OOF用于逐元素校验。
- 原始cBio临床表用于独立endpoint校验。

这些含患者级信息的文件不公开；需要数据所有者授权。读取父缓存时检查SHA/fit患者hash。RNA三组输入、PFI正时间交集与临床六字段与Phase13一致。不同encoder的latent coordinates不能跨fold拼训练。

## 运行

服务器目录为 `/root/phase14_rna_repo`；Python `/root/miniconda3/bin/python`。设置：

```bash
export PYTHONPATH=/root/phase10_trimodal_eval_deps
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python scripts/gate_phase14.py
python scripts/launch_phase14.py
# 等待 SCHEDULER_STATE.json 的90流程及 EVALUATION_DONE 校验完成
python scripts/audit_phase14.py
python scripts/budget_phase14.py --launch
# 等待budget_appendix/STATE.json的90流程完成
python scripts/evaluate_budget_phase14.py
python scripts/report_phase14.py --finish
```

实际完成文件是私有 `EVALUATION_DONE.json`、`budget_appendix/DONE.json` 和主 `DONE.json`；checkpoint/log存在不等于完成。

## 结果解释

主实验80-epoch，预算附录1280-epoch是看到具体上限诊断之后追加，分别报告；保留旧结果。3seeds分别拼5-fold OOF，然后聚合同折有效比较对。Uno60的权重来自各自outer-train，不能把fold C-index简单均值或用测试删失分布。

bootstrap共享患者重复数，统计对象是平均每seed pair credits，不是平均患者风险；固定模型bootstrap未覆盖全流程再训练不确定性。

教师label-OOF的子训练不看目标患者标签，但共享producer训练分区的无监督特征预处理，不是全预处理嵌套OOF。冻结encoder产生的读出训练表示可能in-sample适应，held-out评价仍保持隔离。

Public ZIP逐项SHA及敏感ID/列检查通过后发布。服务器保留必要私有producer/readout状态、OOF风险、教师目标及来源hash；不复制大原始数据，也不新增私有大ZIP。
