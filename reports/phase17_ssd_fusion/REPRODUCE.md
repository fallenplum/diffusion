# Phase17 重现与审计

## 资产与环境

本阶段从`7dca7cf9d72b61517baa85b7493b1f6ea750460d`建立独立分支。原数据、既有切分/endpoint与Phase12预处理、519个CONCH和UNI2-h H5及坐标映射均为外部私有资产。不能只凭公开汇总表声称完成私人数据复现。

新资产目录在配置的`private_root`，位于扩容后的数据盘。原始H5只读；所有患者预测、模型checkpoint和逐患者选择索引均不推送。`VISUAL_AUDIT.json`、`PREPROCESSING_AUDIT.json`与`ARTIFACT_MANIFEST.json`提供身份/公开文件校验。

环境：Python3.12、PyTorch2.5.1+cu124；scikit-survival来自既有独立评测依赖目录。论文方法依赖安装到本阶段私有`paper_deps`，没有覆盖现有项目环境。具体版本与官方仓库提交见`ENVIRONMENT.json`和`LITERATURE_BASELINES.md`。

## 运行顺序

以下从仓库根目录执行。服务器上的Python路径与`PYTHONPATH`按实际环境设置；`CUBLAS_WORKSPACE_CONFIG=:4096:8`、BLAS/OMP线程数2，禁止TF32。固定患者分区及预处理已经存在，不重新生成。

```bash
export PYTHONPATH=.:/root/phase10_trimodal_eval_deps:/root/autodl-tmp/phase17_ssd_private/paper_deps
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python scripts/run_phase17.py --prepare
python scripts/audit_phase17_operator.py
python scripts/audit_phase17_training_leakage.py
python scripts/audit_phase17_papers.py
python scripts/audit_phase17_mamba_port.py
python scripts/audit_phase17_statistics.py
python scripts/audit_phase17_preprocessing.py
python scripts/audit_phase17_ablations.py
```

### 预锁定历史说明

最初的22个pilot使用128个patch、minimum_epoch16；随后仅依据这些内层结果解除下限，A8重跑两折。共同正则比较T0=(dropout0.1,WD0.001)与T1=(0.25,0.01)，预定A3/A7/A8×两折的验证EMA均值选择T1；FiLM改为1664参数的rank8对照并补跑两折。首轮160epochs曲线未显示晚期恢复，因此锁定最多160、40轮无改善早停。实际顺序、两个旧source SHA和最终source SHA保存在`INNER_DIAGNOSIS.md`、`EXECUTION_COUNTS.json`及提交历史中。

当前配置已经是修复后的最终版本。**不要用最终配置重新跑旧pilot并覆盖其缓存**；如要重演原始pilot，检出第一次Phase17提交`37276d5040a032c54878d2f2a6ac393f8d69db0b`，使用隔离私有目录。最终训练source必须等于`PROTOCOL_LOCK.json`。锁内评测脚本曾追加历史强参考和独立核验，记录在`EVALUATION_AMENDMENT.json`；未更改训练source、主比较或bootstrap定义。

### 正式训练与评测

```bash
# 只有已存在且一致的PROTOCOL_LOCK才允许outer训练。
python scripts/schedule_phase17_formal.py
python scripts/run_phase17_simple.py
python scripts/replay_phase17.py --mechanism
python scripts/replay_phase17_simple.py
python scripts/validate_phase17_independent.py --statistics
python scripts/diagnose_phase17_mechanism.py
python scripts/audit_phase17_subgroups.py
python scripts/audit_phase17_inputs.py
python scripts/report_phase17.py
python scripts/plot_phase17.py
python scripts/finalize_phase17_evidence.py
python scripts/package_phase17.py
python scripts/publish_phase17.py
```

主调度器最多4个并行任务：3个S6与1个论文方法，按内层→选epoch→外层依次完成每cell，检查退出状态和checkpoint身份。没有用外层成绩选择方法。原始全训练患者风险收集后才构造Efron目标；患者microbatch只用于前向，不切断风险集。可从已完成cell恢复；如果source变化则拒绝复用。

`A0/A1`只有每折一次实际确定性拟合，3个seed作为引用；不要把同一个确定性结果数成3次训练。实际新外层拟合205、内层候选/诊断249，另外6次两epoch标签扰动检查；PFI没有新增拟合。算子与依赖冒烟没有持久化checkpoint。

## 限制

MGCT和MambaMIL均为固定官方模块的同任务适配，输入维数、通路数、Cox头和Clinical adapter的修改边界详述在结果中；TransSurv因未取得可核验作者实现及多尺度WSI输入而阻塞。未以自写通用Transformer代替，故三种正式论文方法的最低目标仍部分未完成。

缺少原始WSI QA与外部验证；结果只适用于当前candidate-pool-limited、geometry-only-validation与反复开发过的TCGA内部协议。
