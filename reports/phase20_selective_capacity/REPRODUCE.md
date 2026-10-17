# Phase20 reproduction

当前审计source: `e7b6383cfc23e4fbb8f94419a88bff5db1113babcb967672f87e0f13c74ba548`。原PROTOCOL_LOCK.json保留D1，AUDIT_REPAIR_LOCK.json锁定D2；旧记录必须经精确SHA兼容清单核验，不能仅允许旧source字符串。独立评估另绑定当前哈希。运行环境见ENVIRONMENT.json。

## 必需私有输入

沿用已核验的Phase19原始输入、冻结视觉特征、10个role/fold预处理bank和20个J0/J_CS凸参考；还必须持有本次135 inner+54 outer的D1 checkpoint/原receipt、81 outer的D2 checkpoint、原D1归档/资格/epoch锁、完整optimizer账本及私有ANCHOR_AUDIT_REPAIR_COMPATIBILITY清单。路径由configs/phase20_selective_attention.json及data.py确定。原始数据、患者级OOF、模型与gene列表不随公开仓库发布。祖先cache缺失或SHA不符必须停止，不能在旧目录静默重建或将旧神经模型冒充新OOF。

## 本发行包的执行边界

本次证据来自D1正式训练、anchor-only审计修复资格和D2续跑，当前validator严格校验189份原D1及81份D2的逐文件SHA兼容谱系。公开代码与本次私有历史共同支持结果审核/回放；不声称仅凭公开发行包即可从空目录完整重现同一混来源运行。

**本次原目录禁止重新运行prepare、start/scheduler、repair或audit_phase20_history_reference。** prepare会在末尾源检查之前重写资格记录并追加短拟合，破坏冻结D1清单；历史引用审计会写数学凭据，也不是可重复的只读检查。首次D1准备、D2恢复与历史引用技术修复均为已完成的执行，不提供再次启动它们的可执行命令。

fresh full-D2训练必须使用新目录和独立运行清单，重新定义/核验全D2来源及完整资格；当前严格189/81的validator不能直接用于宣称该新运行合格。这属于新的复现运行，不在本次已完成证据范围内。

## 已有私有结果的审核与回放

仅在正式TRAINING_DONE、EVALUATION_DONE真实完成、没有训练/回放任务运行时，由持有上述全部私有输入者在服务器仓库根目录执行。下列命令会重新计算并覆盖本次匿名评估表及私有OOF cache；审核已有结果可仅读取发布的审计文件，不必再次计算。需要保留原评估字节时，应先对输出作只读备份；不得迁移或重写兼容清单绑定的原绝对路径。

```bash
export PYTHONPATH=.:/root/phase10_trimodal_eval_deps
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

本次先完成135 inner并锁epoch，再fresh refit135 outer；D1已完成54 outer，D2只续跑剩余81。以原件SHA、兼容证明、TRAINING_DONE和checkpoint receipts为证据，不能以日志或checkpoint存在代替。审核时不再启动任何训练。

符合上述条件后，回放可四分片并行，必须全部成功退出：

```bash
pids=()
for shard in 0 1 2 3; do
  python scripts/validate_phase20.py --replay --shard "$shard" --shards 4 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [ "$failed" -ne 0 ]; then exit 1; fi
python scripts/validate_phase20.py
python scripts/report_phase20.py
```

正式validator收齐270份5context、270anchor和120额外表征干预，重建60个完整OOF向量/300折指标，独立sksurv核验、14主检验/16次检验/10探索统计及20k共享患者bootstrap，通过后自行写EVALUATION_DONE。失败/缺失/错源会拒绝完成，不靠手填marker。

## 计数与边界

270为完成的正式模型数；实际发生更新的失败/中断正式尝试与诊断短拟合由增量账本另计，20个凸模型只是继承参考、0新增求解。PFI只迁移冻结DSS风险。A、B、DID、选择性行为及实用收益分别判断；科学阴性/预算未决可以发布，技术审计不完整不能冒充完成。
