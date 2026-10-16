# 复现说明

## 边界与依赖

本交付是 DSS 内层探索，不含患者级数据、所选基因、错配索引、OOF、风险或权重。
需要获授权的原始私有数据与历史特征缓存、已有患者折、NVIDIA GPU，以及与锁定训练环境兼容的 Python/PyTorch/numpy/scipy/pandas/scikit-learn/scikit-survival/pytest。
文件来源与真实预处理状态由 hash/分区审计核验；不可用其他 PCA、病例顺序或基因列表静默替代。

- 原训练 source：`f78fe314faa221de4145b28155a9e130b90caf475893e245c247043324782ab9`
- 160 敏感性 source：`43490d4a42d421b2f0757d68414008c1dd136812e4a3c0c042cf2d3fcf27b06a`
- A3F 实现来源：`72f0362bd26af732936f1f27b3299bdfd01655ca12cc5b94dc6814b5d7361395`
- 共同配置：`configs/trimodal_interaction_v1.json`
- 追加预算：`configs/trimodal_interaction_budget160.json`
- seeds=[31, 47, 83]，既有外折内部分区=[1, 3, 5]；没有新增 outer-test 评估。

## 执行顺序

下列命令从仓库根目录执行。私有路径由现有配置指向；已有 receipt 应先核验，禁止覆盖历史训练产物。

```bash
export PYTHONPATH=.:/root/phase10_trimodal_eval_deps
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pytest -q -p no:cacheprovider tests/test_trimodal_interaction.py
python scripts/audit_trimodal_environment.py
python scripts/schedule_trimodal_interaction.py
python scripts/validate_trimodal_interaction.py --device cuda
python scripts/audit_trimodal_query.py --device cuda
# 仅在已记录的逐臂触顶门触发后执行。原40结果不覆盖。
python scripts/extend_trimodal_budget.py
python scripts/validate_trimodal_interaction.py --device cuda --budget160
python scripts/audit_trimodal_query.py --device cuda --budget160
# A3F 是独立来源的 affine bugfix；按其脚本入口完成9个真实拟合及独立审计。
python scripts/repair_trimodal_diff_norm.py
# 以下报告命令会拒绝任何阶段未完成/审计未通过的状态。
python scripts/report_trimodal_interaction.py
python scripts/package_trimodal_interaction.py
```

资源监控使用 `scripts/monitor_trimodal_resources.py`；其实际开始于原40运行中，报告不把它冒充完整40资源轨迹。
A3F 上述无参数入口依次完成其9个真实拟合、独立验证与DONE，不支持 `--help`；不能跳过其 VALIDATION。

## 计数与统计复现

原40的81个神经单元各有一次真实 ridge 初始化，另有3次 A0 凸基线拟合：84个模型单元及84次 ridge solver调用。
160仅重训原40触顶的 M 个神经单元，其余作为原checkpoint引用；A3F再加9个神经拟合及9次 ridge初始化。
最终数量从实际 DONE/EXECUTION_COUNTS交叉核对；同一凸基线在3seed中的引用、checkpoint replay、query probe不计新训练。

指标先在每seed的内层分区累加可比较对分子/分母，再平均seed算法指标；不跨折直接排列不同尺度风险，不平均风险。
这些内层分区含重叠患者且已参与选择；只报告配对描述，不做置信区间/p值，不与历史完整外折OOF榜单比较。
160追加发生于观察到内层40触顶后，A3F发生于发现 affine实现缺陷后，二者均不属于原先独立确认。
