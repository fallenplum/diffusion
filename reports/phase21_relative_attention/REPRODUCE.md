# Phase21 复现与交付边界

## 固定来源

父提交为 `5b2bb3645b125ca09e2e6825706978fc691981f4`，实验分支为 `research/phase21-relative-attention-audit`。先读取 `PROTOCOL_LOCK.json` 和 `QUALIFICATION.json`，逐项核对配置、源文件与测试 SHA。以下命令在服务器仓库根目录执行；Python 为 `/root/miniconda3/bin/python`，`PYTHONPATH=.:/root/phase10_trimodal_eval_deps`，`CUBLAS_WORKSPACE_CONFIG=:4096:8`。实验期间保持已锁源码及全部 Phase20 文件不变。

本轮私有目录为 `/root/autodl-tmp/phase21_relative_private`。需要既有 Phase20/19 的保护数据、预处理缓存、官方源码固定版本及原 checkpoint；公开仓库不包含这些资料。缺文件时应停止，不能静默重新拟合历史对照。

## 已存 S16 的只读审计

以下为审计入口；已有完整 receipt 时校验源和 SHA 后复用，不重新训练。

```bash
python scripts/audit_phase21_saved.py --snapshot
python scripts/audit_phase21_saved.py --role inner --fold 1
python scripts/audit_phase21_saved.py --role outer --fold 1
```

对 folds 1–5 各执行 inner/outer job；每个 job 依次审计 seeds 31/47/83。全部10个 job 完成后执行：

```bash
python scripts/audit_phase21_saved.py --collect
```

每份旧 S16 核验原 receipt、实际 D1/D2 来源与 checkpoint SHA，回放四种 heldout 风险；随后分别遍历全部 fit 与 heldout 的匿名 attention 链。heldout 使用浅复制数据视图，仅改变诊断遍历索引，模型、训练均值、预处理及原分区保持冻结。不读取结局定义诊断，不优化模型；原模型此前接受过 DSS 训练，不能称无监督表示。风险、索引和地图保持私有。

## 新 DIRECT 训练

资格与正式训练由已锁脚本执行；**已经完成资格的原目录不得再次运行 prepare**，以免覆盖短拟合账本。新环境重建资格时，须先完整保存原实验来源及账本。

```bash
python scripts/prepare_phase21.py
python scripts/start_phase21.py
```

正式矩阵只有 DIRECT 的15 inner和15 fresh outer。P16/S16各30份原模型仅引用，不计本轮新训练。全部15 inner epoch 与 checkpoint SHA 锁定后才运行 outer。outer 从同 seed 初始化，不能加载已见 inner-val 标签的旧 outer 模型作为教师或初始化。`TRAINING_DONE.json`、全部逐 cell receipt、实际 optimizer 账本和完整评估共同决定完成状态，进程存在不代表完成。

若重启或失败，先归档原 GPU 采样及逐尝试账本；不得丢弃失败/中断步骤或把零已提交步骤直接当作零计算。正式与两次资格短拟合分开记账。

## 独立回放和评测

完成训练后可串行回放全部90份模型；不要混用不同分片总数产生的旧回放文件。

```bash
python scripts/validate_phase21.py --replay --shard 0 --shards 1
python scripts/validate_phase21.py
```

评估核对90份 checkpoint、540个风险向量回放、48个完整 seed OOF 向量与 DSS/PFI 各240逐折行。统计对象为每 seed 合并同折分子/分母后的 C，再平均三个 seed，不能先平均风险。主检验只有 DIRECT−S16、DIRECT−P16，各 Harrell/Uno60，共4项；20,000次同患者共享 bootstrap，报告 Bonferroni4 区间。PFI只迁移冻结 DSS 风险。

同 checkpoint 的 off→on 与独立训练模型比较分开解释。repair/damage 包含半分 ties，净差须等于相同统计对象的 C 差；不能用外折错误患者对生成训练监督。

## 报告与匿名包装

完整评估完成后由独立报告脚本生成最终结果及输入 SHA 审计：

```bash
python scripts/report_phase21.py
python scripts/package_phase21.py --dry-run
python scripts/package_phase21.py --dry-run --check-completion
```

`report_phase21.py` 为独立评估后报告入口，不能在结果尚未完整时预填成绩。检查最终 `RESULTS.md`、`decision.json`、`REPORT_INPUT_AUDIT.json`、实际 `EXECUTION_COUNTS.json` 后，由执行负责人显式发布包装：

```bash
python scripts/package_phase21.py --release
```

默认 dry run 不写文件；release 必须通过真实资格、训练、全回放、统计及报告技术门。预算不足或科学负结果可以如实发布，不能用技术 PASS 表述机制成立。包装逐字验证父提交2362个 tracked 文件，五个历史 ZIP 仅允许原 SHA 不变，不能重新加入本轮交付。

公开白名单仅 Phase21 新源码、测试、配置与匿名报告，单文件不超过8 MiB。患者级风险、权重、原始数组、地图、Token、患者/基因/供体身份及凭据不得公开。`ARTIFACT_MANIFEST.json` 列出逐文件 SHA/字节数且不自哈希；私有 `publication/phase21_public.tar.gz` 包含该 manifest，`publication/PACKAGE_RECEIPT.json` 记录转移包 SHA。包装程序不执行 Git commit/push，也不宣称全仓库隐私 PASS。

## 优势与限制

优势：旧结果字节可核验，只有一个新训练算子，风险修复与损伤可闭环核对。限制：原对照分别选 epoch，模型间比较含共同主干优化差异；同一队列反复开发且只有65个 DSS 事件，固定模型 bootstrap 不包含重新训练和研究选择的不确定性。预算门也不等同全局收敛证明。
