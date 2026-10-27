# BG-BEN 复现与产物边界

## 真实环境

本轮服务器实际版本：Python 3.12.3、PyTorch 2.5.1+cu124、NumPy 2.1.3、SciPy 1.14.1、scikit-learn 1.5.2、官方 mamba-ssm 2.2.4。正式含 Mamba 的单元使用 CUDA 官方实现；合成测试中明确标记的 `ssm` 后端不作为正式 Mamba 结果。

Git 父提交：`3da53440e1160066d4bec79781d97764eb81d19f`。研究分支为 `research/bgb-en-spatial-omics-mechanism`。核心源文件、配置、输入与各训练单元收据的 SHA256 均留存；精确清单见 EXPERIMENT_LINEAGE.json 和 PUBLICATION_MANIFEST.json。

## 私有资产绑定

设置本机/服务器自己的路径变量，不能把真实患者文件、账号凭据或服务器映射提交到 Git。以下变量是操作员配置，不是可公开下载的数据：

先将 `configs/bgb_en/assets.example.json` 复制到自己的私有目录，将全部 `/private/...` 占位路径替换成已审计资产的实际位置，再令 `BGBEN_ASSETS` 指向该副本。示例逐项对应现有 `load_raw`、`load_inputs` 和缓存审计读取的字段；不改变科学配置。`visual_sources` 用于原 LOCAL/强 CP 复现，不能仅提供新 full-mean。四个 CONCH 缓存目录和 UNI2 缓存目录都必须含相应 `READY.json` 及受其哈希约束的数组。`conch_inventory` 和 `uni2_root` 用于原始来源/owner/几何资格核验；它们不进入公开发布。GMT 的公开 SHA 固定在示例中，cohort、endpoint、视觉源和患者顺序由 `data.py` 内既有常量继续核验。

```bash
export BGBEN_ASSETS=/private/asset_binding.json
export BGBEN_CORE=/private/core_run
export BGBEN_EXTRA=/private/sensitivity_run
export BGBEN_PATH=/private/path_budget_run
export BGBEN_OUTER=/private/outer_run
export BGBEN_F7=/private/F7_recovery_work
export BGBEN_RECHECK=/private/read_only_recheck
export BGBEN_BUDGET=/private/budget_sensitivity_run
```

资产配置键包含 canonical cohort、独立 DSS/PFI endpoint 表、视觉来源、旧八程序、Hallmark GMT 和四种 CONCH512/UNI2-128 缓存。`scripts/bgb_en_run.py::load_inputs` 是精确读取合同；R0 报告列出 hash、编码器尺度、坐标 hop 和患者数量。原始数据只读，派生缓存位于私有目录。

核心缓存来自全部原 H5；REGIONS 使用无结局的真实坐标连通窗口，GRID/RAND/DIVERSE 单独缓存。已有缓存只有通过 READY/hash 才能复用，不能仅靠目录名。

## R0 与共同教师

```bash
python scripts/bgb_en_run.py --assets "$BGBEN_ASSETS" --private "$BGBEN_CORE" --stage qualify --device cuda
python scripts/bgb_en_run.py --assets "$BGBEN_ASSETS" --private "$BGBEN_CORE" --stage prepare --partition 0
python scripts/bgb_en_run.py --assets "$BGBEN_ASSETS" --private "$BGBEN_CORE" --stage prepare --partition 1
python scripts/bgb_en_run.py --assets "$BGBEN_ASSETS" --private "$BGBEN_CORE" --stage prepare --partition 2
```

每个 prepare 由 9 个教师 Cox、4 个 full Cox、1 个同教师 held-out 患者对 β 优化组成。所有教师的基因选择、PCA、缺失处理和类别编码都只在其自己训练行拟合；标化只用其自己训练风险均值/SD。教师内两个 held-out 患者形成 β 校准对，避免某个患者的标签进入另一个教师训练后参与同一监督对。

历史 `upstream_optimizer_fits=13` 字段只数 Cox。该元数据边界和顺序说明修正见 PROTOCOL_ACCOUNTING_ADDENDUM.json；不覆盖已经存在的旧锁或收据。

## 核心 R1–R3 正式矩阵

以下调度入口用于新的复现实验目录。当前已完成F7独立恢复的原始运行不得再次启动原`--stage all`调度器：它会重新写调度收据，破坏必须保留的旧PARTIAL历史。已有运行应先检查进程与完成收据，只对明确未完成且可安全恢复的任务使用相应独立入口。

```bash
python scripts/bgb_en_schedule.py --assets "$BGBEN_ASSETS" --private "$BGBEN_CORE" --stage all --device cuda --max-workers 4 --resume
```

186 个唯一评估单元，其中 162 个神经拟合、12 个额外凸 Cox、12 个对新共同教师的确定性引用。加上 42 个上游优化，完整核心为 216 个监督优化；RNA 风险解析 decoder 不混算成额外 DSS Cox 模型。两个 seed 分别先汇总当前验证分区内 concordance 分子分母，再对 seed 均值。

所有神经单元固定 80 次完整训练风险集更新，20/40/80 checkpoints；科学端点固定取 80，不在已见验证集挑最优 epoch。训练前向可按 8 患者分块，两遍确定性梯度缓存精确计算同一个完整 Efron Cox；分块不会建立各自 Cox 风险集。

## 独立补充矩阵

额外锁 `ADDITIONAL_PLAN_LOCK.json` 对应 84 个预定拟合：组学因子/顺序 24、Patch 预算/编码器 36、相同 UNI2 输入的源监督对照 24。共同教师保持原来源，补充实验不更改核心源码或原 COMPLETE 文件。具体单元参数写在私有有限计划中；调度器仅执行明确任务列表，最多四个 GPU 进程，不做隐式参数搜索。

外部 source 原始尺度 7 个拟合经审计不能直接迁移到目标 L2 缓存；修复版另跑 7 个固定 20-step source 拟合。WRONG_DONOR 和 SPATIAL_SHUFFLE 各另跑一个全源拟合，source 总成本为 16。旧 raw 权重没有参与目标迁移。

外部迁移只初始化 UNI2 的 1536→64 投影，保留目标汇聚和风险头；不是把四个 ST 程序 query 直接移入 50 个 Hallmark token，也不是在 TCGA 上持续施加真实局部分子监督。两个 source 负对照使用原真实目标的固定尺度和同一初始化。

## 无训练审计与匿名报告

下列命令把新的复核结果写入独立目录，保留原CHECKPOINT_AUDIT的184/186及F7恢复证书。F7必须在独立确定性运行进程中回放，再将其六个已验证回放引入全量审计；不得在默认后端下直接重放修复版F7。`--resume`只复用通过且哈希未变的审计，不将失败改判通过。

```bash
python scripts/bgb_en_replay_feature_audit.py --assets "$BGBEN_ASSETS" --private "$BGBEN_CORE" --output "$BGBEN_RECHECK/SAVED_PREPROCESSING_REPLAY.json"
python scripts/bgb_en_checkpoint_audit.py --assets "$BGBEN_ASSETS" --private "$BGBEN_CORE" --stage R3 --arm F7 --runtime-repair --device cuda --resume --output "$BGBEN_RECHECK/F7_CHECKPOINT_AUDIT.json"
python scripts/bgb_en_checkpoint_audit.py --assets "$BGBEN_ASSETS" --private "$BGBEN_CORE" --stage all --f7-audit "$BGBEN_RECHECK/F7_CHECKPOINT_AUDIT.json" --device cuda --resume --output "$BGBEN_RECHECK/CHECKPOINT_AUDIT.json"
python scripts/bgb_en_pair_replay.py --assets "$BGBEN_ASSETS" --private "$BGBEN_CORE" --recovery-receipt "$BGBEN_F7/RECOVERY_RECEIPT.json" --output "$BGBEN_RECHECK/MECHANISM_REPLAY.json"
python scripts/bgb_en_report_inner.py --private "$BGBEN_CORE" --mechanism-replay "$BGBEN_RECHECK/MECHANISM_REPLAY.json" --output "$BGBEN_RECHECK/inner_reports"
```

机制 replay 的全验证风险修复/损伤与有限患者的注意力/梯度诊断分开；冻结干预不称真实重训。完整权重回放、OOF 覆盖、teacher fit hash 和 independently counted Harrell/Uno 有各自审计文件，合成 PASS 不代表真实模型已回放。

## 外层顺序

完整内层结束后锁定 OUTER_SELECTION_LOCK.json，绑定完整 inner 审计的 SHA 和科学来源。简单 CP/SCALAR 通过 `bgb_en_outer.py` 在五个外折分别训练；确定性 CP 向量跨三个 seed 引用，SCALAR 的教师分区与 β 按 seed 独立重训。

若神经 finalist 通过原型门，只能用固定 F2/F3/F5/F10 候选池在每个当前 outer-train 内重新完成三内折×两 seed，再锁该外折候选。不能把 outer1 原型的全局胜者直接用于其它外折。最终神经更新严格先完成 80 次，之后才计算该 outer-test 指标；PFI 仅读冻结 DSS 风险，不用 PFI 选参数。

```bash
python scripts/bgb_en_outer_schedule.py --assets "$BGBEN_ASSETS" --private "$BGBEN_OUTER" --primary-private "$BGBEN_CORE" --selection-lock reports/bgb_en/OUTER_SELECTION_LOCK.json --inner-report reports/bgb_en/INNER_REPORT_FOR_SELECTION.json --stage baselines --max-workers 2 --resume
python scripts/bgb_en_outer_schedule.py --assets "$BGBEN_ASSETS" --private "$BGBEN_OUTER" --primary-private "$BGBEN_CORE" --selection-lock reports/bgb_en/OUTER_SELECTION_LOCK.json --inner-report reports/bgb_en/INNER_REPORT_FOR_SELECTION.json --stage nested_neural --max-workers 4 --resume
python scripts/bgb_en_report_outer.py --private "$BGBEN_OUTER" --selection-lock reports/bgb_en/OUTER_SELECTION_LOCK.json --output reports/bgb_en
python scripts/bgb_en_outer_audit.py --assets "$BGBEN_ASSETS" --private "$BGBEN_OUTER" --primary-private "$BGBEN_CORE" --selection-lock reports/bgb_en/OUTER_SELECTION_LOCK.json --inner-report reports/bgb_en/INNER_REPORT_FOR_SELECTION.json --device cuda --resume
```

第一条调度五个外折的共同教师和基线，CPU并行上限为2；第二条依次完成五个外折各自的内层选择，全部结构锁定后再调度15个最终神经拟合，GPU并行上限为4。outer1复用核心的24个候选单元，outer2–5新训练96个候选单元。预先计算的12份教师准备包只复用到其自己的训练分区，不构成额外拟合或跨折复用。若这些准备包已存在，`--resume`核验哈希后复用，不应再次计数。

上述四条按依赖顺序执行，不能在训练尚未结束时将局部report视为完整R4。真实候选完成状态以收据为准；最后一条才进行保存权重、OOF覆盖与独立指标回放，不执行新优化。最终 R4 报告既保留逐折/seed，也保留同 seed 折内分子分母聚合；不直接拼接不同折风险排序。

### 本次R4中断恢复

2026-10-28更换GPU节点后，先核验科学源码、全局选择锁、既有30个内层完成模型、12份nested教师准备、5份外折baseline及前两份外折选择锁。全部通过原收据和产物SHA后，复用已有完成单元；`--resume`的作用是校验后跳过，而不是重新优化。

两个中断单元只留下START和空日志，没有末次权重或Adam状态。其原目录和日志原样归档，随后按同配置、同seed、新模型和新Adam重新执行80轮；不能称精确续训。两次历史尝试的已执行更新数没有证据，单列未知，不计为零，也不改写为原训练失败收据。新调度的66个剩余内层单元包含这两次重做，计划内层总数仍为96；另有15个最终拟合。

恢复使用原调度入口及4路GPU上限，没有改变科学源码、候选集合、epoch、γ、选择锁或数值阈值。旧30模型与两份折锁的独立恢复核验见R4_RESTART_INTEGRITY_20261028.json。新节点资源采样与旧节点分开，未采样时段不推测利用率；显卡容量只按驱动报告记录，不据此断言物理显存规格。

## 独立数值诊断与失败恢复

PCA 正则几何对照是在观察内层现象后单独锁定的探索性分析：复用原 PCA，比较单位方差 PC 与未再次缩放 PC 的同 λ Cox，不改变 R4 设计。原九次求解中七次通过、两次触发梯度守护；只对两次失败追加更严格收敛精度，保留原锁与失败记录。最终有效模型九个，实际求解尝试十一次，PCA 与风险教师均零重新拟合。其结果不能推广为所有低维或神经表示均保留预后信息。

F7 原六次拟合中三次触发完整风险集两遍前向 `2e-5` 检查。独立确定性恢复使用 `F7_deterministic_runtime_v1.json`：先逐单元 CUDA 零优化资格，再保留全部六个原目录，统一六个新模型从相同种子和新 Adam 开始。核心源码、Top64、阈值和 80 轮合同不变；无失败 epoch/Adam 状态，不能称精确续训。最终完成与独立认证以 F7 恢复收据为准，不改写原失败调度记录。其它模型保持原后端。

完整核心检查点首轮审计为184/186通过；O9 inner2的两个seed未满足原`2e-6`风险逐元素门槛。固定三个检查点、两个后端、每个五次的追加诊断中，30次完整验证风险向量的可比较患者对credit和Harrell/Uno60变化均为0，但数值门槛失败保留，不改判186/186。详情及全部重复见O9_FLOAT32_DRIFT_NOTE和两份诊断JSON。最初额外诊断误用默认γ=0.1的两份输出已经作废并私有归档；有效V2显式绑定原R2的γ=0.3，该诊断构造错误未影响原训练或首轮审计。

参数口径必须分开：旧核心神经单元COMPLETE中的parameters为registered参数总数，R3包含显式冻结的辅助头；已保存active_parameters才对应requires_grad计数，仍不等于实际梯度活跃度。凸Cox的参数为系数维数，P0/O0/F0/F1等教师引用也不代表新增神经优化。R4最终神经模型的parameters已经采用requires_grad计数。新增参数审计只补充metadata说明，不改旧完成收据、训练或冻结选择规则。

### 预算实物的只读复核

160/320预算运行由`configs/bgb_en/budget_sensitivity_v1.json`及私有有限任务计划约束。`scripts/bgb_en_make_budget_plan.py`生成24个明确任务，`scripts/bgb_en_budget_retrain.py --stage train`是每个任务的入口；当前已有24次尝试结束，不再次执行训练。各新拟合使用新模型和新Adam，不能称为从旧80轮模型精确续训。

严格汇总只接受23个原资格通过单元，必须显式`--allow-partial`；PARTIAL不因另有实物诊断而改变。新的私有复核目录不得指向原正式报告或原锁。

```bash
python scripts/bgb_en_summarize_budget.py --private "$BGBEN_BUDGET" --primary-private "$BGBEN_CORE" --assets "$BGBEN_ASSETS" --output-dir "$BGBEN_RECHECK/budget_strict23" --allow-partial
python scripts/bgb_en_budget_saved_chain_audit.py --stage lock --private "$BGBEN_BUDGET" --primary-private "$BGBEN_CORE" --assets "$BGBEN_ASSETS" --lock "$BGBEN_RECHECK/budget_chain/R2_BUDGET_SAVED_CHAIN_AUDIT_LOCK.json" --output-dir "$BGBEN_RECHECK/budget_chain"
python scripts/bgb_en_budget_saved_chain_audit.py --stage audit --private "$BGBEN_BUDGET" --primary-private "$BGBEN_CORE" --assets "$BGBEN_ASSETS" --lock "$BGBEN_RECHECK/budget_chain/R2_BUDGET_SAVED_CHAIN_AUDIT_LOCK.json" --output-dir "$BGBEN_RECHECK/budget_chain"
```

锁阶段不加载模型、不前向。audit阶段检查全24保存轨迹，并包含原失败160轮模型的一次CPU实际权重前向；不调用GPU、优化器或新decoder。全24物理工作量为5760次更新、48个decoder，其中合格5600/46、失败实物160/2；这些是同一批既有成本，不能再加到总账第二次。

## 私有备份与发布边界

先完成报告和诊断图，再执行显式白名单发布扫描。PNG 必须绑定可扫描 SVG、匿名 CSV 及绘图源码。患者/供体/spot 标识、行级风险、教师、权重、原始数据和坐标均留在私有目录，不进入公开分支。`bgb_en_backup_private.py` 可将私有产物流式存入经授权的本机大容量盘；不通过 PowerShell 文本管道传输二进制，不重复备份可重建的 cache feature blobs。

本轮不使用 bootstrap、置信区间或 p 值。已有研究反复使用同一 TCGA 队列，当前训练折隔离不构成新的独立验证队列。组织学监督是粗粒度真实组织标签，ST 程序是局部表达代理；TITAN 分数不是独立病理真值。最终正负结论和未执行项以 DECISION.md、EXPERIMENT_LINEAGE.json 为准。
