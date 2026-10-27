# R4 已锁定教师准备包的 CPU 提前计算

## 范围

仅提前执行原 R4 nested 协议本来就需要的 outer folds 2–5 × 3 inner partitions，共 12 份准备包；不增加候选结构、神经拟合或模型选择。原 `bgb_en_nested.py --resume` 后续直接校验并复用这些同路径文件。

每包原有 9 个教师 Cox + 4 个全训练分区 Cox，以及 1 个同教师 β 标量优化。因此计划内总计 **156 个 Cox + 12 个 β = 168 次监督优化**，并非额外的架构实验。原回执 `upstream_optimizer_fits=13` 不包含 β，须分别计数。

## 执行前资源资格

- 容器 cgroup 内存限额 66,571,993,088 bytes，检查时占用 40,118,214,656 bytes；宿主机逻辑 CPU/内存视图不作为容器可用上限。
- CPU quota 为 16 核；当时四个 GPU worker 各约占 1 个 CPU 核，GPU 利用率 99%。
- overlay 可用 4,239,233,024 bytes。三份已有准备包为 230,741,444–231,119,556 bytes；12 包按最大值加 5% 估算 2,912,106,396 bytes，预计仍可留约 1.33 GB。
- 限制为最多两个 CPU 子进程，每进程 2 个 BLAS/PyTorch 线程，`CUDA_VISIBLE_DEVICES=` 隐藏 GPU。
- 每次派发前复核容器内存余量至少 8 GiB、预计完成剩余包后 overlay 至少留 1 GiB；不满足即拒绝该单元，无自动重试。

## 源码与锁

独立调度器：`scripts/bgb_en_precompute_nested_prepare.py`，SHA256 `fd742d4159e0379b85fdcbe44f4515368f4c1a2f19b01f5863fc44554f8cff48`。

在任何拟合前调用原 `validate_selection_lock` 核验全局选择锁、冻结内层报告、配置和完整 scientific lineage。各子进程调用未修改的 `scripts/bgb_en_run.py --stage prepare`，不修改原 nested/outer 调度器。

可复现的每单元命令（路径由私有配置提供）：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
python scripts/bgb_en_run.py --assets "$ASSETS" --private "$NESTED_PRIVATE" \
  --config configs/bgb_en/prototype_v1.json --stage prepare \
  --outer-fold "$FOLD" --partition "$PARTITION" --seed 31 --resume --device cpu
```

`FOLD=2,3,4,5`，`PARTITION=0,1,2`；并发上限 2。

## 状态与限制

**已完成并独立核验 12/12 包，0 失败：原有 0 包、新建 12 包。**CPU 调度墙钟耗时 81.904 秒；各单元墙钟耗时之和 157.306 秒（不是 CPU time）。实际准备文件合计 2,769,732,784 bytes；审计时 overlay 剩余 1,469,014,016 bytes。

独立审计重新核验各准备文件 SHA、同教师成员隔离和全部预处理 fit hash。156 个 Cox 与 12 个 β 均为本来就锁定的 R4 上游优化；后续 nested 调度再次显示这些概念性模型数量时，**必须按同一批模型只计一次**。

执行状态、每包 SHA、已有/新建计数及 CPU 调度耗时以 `R4_CPU_PREPARE_PREFETCH_AUDIT.json` 为准。私有命令、准备包和教师风险均留在服务器，不公开患者数据。独立审计本身新增拟合、模型前向、GPU 调用均为 0；未访问外折风险数组。两份原选择文件 SHA 在调度前后及审计时一致。

**优势：**提前利用 CPU 余量，后续 GPU nested 训练可复用完全相同的上游准备结果。

**风险：**磁盘空间有限，需持续执行空间守护；若某准备过程失败，其已经发生但未留下完整收据的部分优化次数不得伪报为零。该调度不提供新的性能或医学机制证据。

**置信度：高（限原命令、来源锁及完成包哈希可核验的复用关系）。**
