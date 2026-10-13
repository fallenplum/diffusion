# Phase10 三模态机制审计

正式420记录（405实际fits，15 D0复用）完成；源码审计、OOF、探针和回放齐全。主结论与十项研究问题见RESULTS.md，固定规则见DECISION.md。只公开代码、配置、聚合证据；患者数据保留用户服务器。

## 复现

使用新checkout，保留当前实验归档。当前服务器环境Python3.12 / torch2.5.1+cu124，真实GPU名称和依赖见运行环境记录；scikit-survival0.23.1/ecos2.0.14隔离安装在 `/root/phase10_trimodal_eval_deps`。需要具有读取下述私有输入的授权。

实测Python 3.12.3，GPU报告RTX4080SUPER/32760MiB，driver595.71.05；8路并行忙时观测利用率100%、显存17776MiB。该值为运行时快照，不是全程平均。

```bash
git clone -b codex/phase10-trimodal-mechanism-audit git@github.com:fallenplum/diffusion.git phase10_reproduce
cd phase10_reproduce
PYTHONPATH=/root/phase10_trimodal_eval_deps python scripts/run_phase10_complete.py --workers 8 --output /root/phase10_trimodal_private/reproduction_v3
```

该单命令执行合成门控、全部baseline+独立验收、全部机制+独立OOF/paired验收。要求现有已验收缓存 `/root/phase10_trimodal_private/cache` 和 `/root/phase7a_foundation_headroom_v1/private_dataset.npz`；若缓存缺失，先执行 `python scripts/run_phase10_trimodal.py data`，H5目录与cohort manifest可用 `--features/--manifest` 指定。脚本会拒绝非空的新output；scheduler单独调用支持同源逐cell恢复，源码变化拒绝恢复。不要从本机E盘路径直接运行服务器命令。

原cohort manifest/fold表已从新分支公开树移出；默认私有manifest `/root/phase10_trimodal_private/cache/private_cohort_manifest.csv`，原fold `/root/phase10_trimodal_private/cache/private_fixed_patient_folds.csv`。原文件与被遮名的历史汇总/legacy脚本均保存在 `/root/phase10_trimodal_private/inherited_identifiable_originals`，仅本分支公开树更改；历史分支未动。

当前真实正式目录 `/root/phase10_trimodal_private/formal`。旧训练v2源码精确归档 `TRAINING_SOURCE_V2.tgz`；交付源码只加固metadata、输入来源验证与无实际pad的防护，正式checkpoint回放已核对。用原版本复核前需隔离checkout恢复bundle，避免混合sourcehash。文件哈希均见public/private清单。

Git中也保留等内容的 `training_source_v2/` 文本快照，可直接检查旧训练源码与交付加固diff。二进制zip/tgz只交付本机/服务器，不跟踪Git；源码、汇总报告与包校验清单均推送。

```bash
PYTHONPATH=/root/phase10_trimodal_eval_deps python scripts/validate_phase10_independent.py --output /root/phase10_trimodal_private/formal
python scripts/audit_phase10_raw_sources.py
python scripts/audit_phase10_omics_source.py
python scripts/audit_phase10_final.py
# 每个正式seed/fold运行，支持在GPU并行；默认只读取现有formal目录
python scripts/audit_phase10_representations.py --seed 31 --fold 1
python scripts/test_phase10_real_test_perturbation.py
python scripts/supplement_phase10_fixed_representations.py
python scripts/add_phase10_descriptive_pairs.py
python scripts/collect_phase10_audit_tables.py
```

诊断脚本默认审计已归档formal目录；复核其它目录前在新checkout调整这些入口路径，禁止覆盖旧reports。修复D0旧元数据才需 `repair_phase10_reuse_metadata.py`，交付新runner已消除该缺陷。`TRAINING_SOURCE_V2.tgz` 不含私人数据。

## 阅读结果

主指标每seed先拼5-fold OOF，再用within-fold对的分子/分母micro；`pooled_*`仅风险尺度敏感性。Ridge三个seed是同一确定性拟合，没有三份独立信息。主family六比较×Harrell/Uno；其它附表明确描述性。固定表示补充25次fits不混入420正式计数。bootstrap不覆盖重新训练方差。

公开包 `PHASE10_PUBLIC_AUDIT.zip` 在本报告目录；服务器私有包 `/root/phase10_trimodal_private/PHASE10_PRIVATE_REPRO.zip`。公开包无法替代私有病例输入；患者OOF、标签、ID、checkpoint和transform禁止提交Git。
