# 复核与受控重运行

公开交付包含源码、配置、匿名聚合报告和哈希证据。患者级数组、标识、临床记录、逐样本预测、变换参数和权重不公开。公开材料本身不是可独立重建全部患者结果的数据包。

## 配置身份

- 训练前设计：`configs/semantic_m123/design_precommit.json`。
- 原始私有执行锁 SHA256：`41f041bcb883d65285167a201183f40d41e9de1640ead2b40a9311f61f318af3`。
- 公开科学投影：`configs/semantic_m123/master_public.json`，仅把五个机器路径换成环境变量占位，其他科学字段、输入和源码哈希原样核验。
- 当前评测锁：`configs/semantic_m123/analysis_lock_v2.json`；此前接口版本不再启动。
- 冻结几何诊断锁：`configs/semantic_m123/diagnostic_lock.json`。
- 选择后的无结局检查点阶段归因锁：`configs/semantic_m123/readout_roles_lock.json`；它不更改任何性能候选或训练源。

公开投影不与原始执行 JSON 字节相同。原机器上的原始执行锁保留未改；在其他目录运行时，先从公开科学字段构造新的私有执行文件，绑定 `ASSETS_PATH`、`LEGACY_ROOT`、`PARENT_PRIVATE`、`PARENT_SPLIT_PLAN`、`FROZEN_IMAGES`。迁移会产生新的执行文件哈希；必须在任何新训练前同步构造新的评测/诊断主锁绑定并保存来源关系，不能继续冒用本轮原始执行哈希。

## 必需的私有依赖

需要同一资产合同、原患者级分区、全部当前训练分区 R4 教师与预处理、冻结 FM 缓存。程序逐项检查源码和资产哈希以及 fit/evaluation 身份。缺任一工件时应停止并先恢复资格，不得生成一个近似替代文件后宣称精确复现。

资格入口为 `scripts/semantic_m123_qualify.py`；训练入口为 `scripts/semantic_m123_schedule.py`。资格包括旧函数及 trainable warm 映射、掩码/切片读取和实际形状吞吐；正式运行必须使用新的私有输出目录，不接受无锁部分目录的自动续跑。

## 正式运行示例

在具备已核验原始私有锁及依赖的原环境中：

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
python scripts/semantic_m123_schedule.py \
  --lock "${MASTER_LOCK}" --expected-lock-sha256 "${MASTER_SHA256}" \
  --private "${NEW_PRIVATE_DIRECTORY}" --max-parallel 6
```

六个 GPU worker 是上限，实际准入同时检查 GPU、RAM 和磁盘余量；另允许一个 CPU Ridge worker。资源等待不更改种子、架构、轮数或候选。不能重复提交已有注册任务来提高显存数字。

程序先完成 945 个内层任务，保存全五外折共同选择屏障，再完成 155 个 final 任务。只有完整完成标记存在且 owner 已释放时，才运行 `scripts/semantic_m123_report.py`；必须显式传当前分析锁、私有目录、新报告目录、固定官方评分库目录和 `--formal-complete-authorized`。几何脚本在主报告完成后另行运行，不能覆盖主报告。

报告重建保存的变换，不重新 fit；保存各模型各 seed 的私有 OOF。早期内层 epoch 的预测保留，但仅保存内层最终 400 轮权重；没有声称对每个早期权重做了回放。Final 保存选择轮数的实际模型，含 epoch0。

## 软件与交付复核

运行全部 `tests/test_semantic_m123_*.py` 检查模型、完整 Efron、选择屏障、指标聚合、冻结几何和发布边界。实际运行记录见 `SOFTWARE_TEST_LEDGER.json`；合成测试不等于患者结果。

`scripts/semantic_m123_publication.py` 分为 `plan`、`stage`、`final-scan`。沿用既有私有排除名单与原扫描器，不重建或公开患者名单。扫描全部新增候选、控制文件、生成清单；原私有执行锁不进入候选。GitHub 新分支只新增文件，保留父提交 536 个文件的原始字节。最终另以 Git 对象流逐文件检查 SHA256 和 Git blob SHA1，不能用推送命令成功代替内容核验。

如果原始输入、选择合同或训练代码不同，应作为新实验标识，不能覆盖本轮历史结果。
