# Phase20 execution records

| id | arm | source | status | generator_steps | discriminator_steps | error_type |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | S16 | d1d7f8176f18417b8107767700f68ec0292181c3a0e76fce37894e06cc0c3192 | RETURNED_SUCCESSFULLY | 2 | 0 |  |
| 2 | S16 | d1d7f8176f18417b8107767700f68ec0292181c3a0e76fce37894e06cc0c3192 | RETURNED_SUCCESSFULLY | 2 | 0 |  |
| 3 | P16 | d1d7f8176f18417b8107767700f68ec0292181c3a0e76fce37894e06cc0c3192 | RETURNED_SUCCESSFULLY | 2 | 0 |  |
| 4 | D16 | d1d7f8176f18417b8107767700f68ec0292181c3a0e76fce37894e06cc0c3192 | RETURNED_SUCCESSFULLY | 2 | 0 |  |
| 5 | L0 | d1d7f8176f18417b8107767700f68ec0292181c3a0e76fce37894e06cc0c3192 | RETURNED_SUCCESSFULLY | 2 | 0 |  |

只按实际optimizer更新账本计数；诊断短拟合不保存checkpoint或患者预测，不冒充正式270模型。预备资格中的失败在此保留，不能从完成checkpoint数量反推总尝试次数。

本次读取时未发现正式scheduler FAILED.json；这不证明没有未记录的中断或失败尝试。

## 正式模型增量尝试账本

| arm | seed | fold | role | source | status | generator_steps | error_type |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D16 | 31 | 3 | outer | d1d7f8176f18417b8107767700f68ec0292181c3a0e76fce37894e06cc0c3192 | INTERRUPTED_BY_SCHEDULER_AFTER_PEER_FAILURE | 15 |  |
| D16 | 31 | 3 | outer | e7b6383cfc23e4fbb8f94419a88bff5db1113babcb967672f87e0f13c74ba548 | INTERRUPTED_BY_SERVER_REBOOT | 0 |  |
| F16 | 31 | 3 | outer | d1d7f8176f18417b8107767700f68ec0292181c3a0e76fce37894e06cc0c3192 | INTERRUPTED_BY_SCHEDULER_AFTER_PEER_FAILURE | 14 |  |
| F16 | 31 | 3 | outer | e7b6383cfc23e4fbb8f94419a88bff5db1113babcb967672f87e0f13c74ba548 | INTERRUPTED_BY_SERVER_REBOOT | 0 |  |
| L0 | 31 | 3 | outer | d1d7f8176f18417b8107767700f68ec0292181c3a0e76fce37894e06cc0c3192 | FAILED | 62 | ValueError |
| L0 | 31 | 3 | outer | e7b6383cfc23e4fbb8f94419a88bff5db1113babcb967672f87e0f13c74ba548 | INTERRUPTED_BY_SERVER_REBOOT | 0 |  |
| P16 | 31 | 3 | outer | d1d7f8176f18417b8107767700f68ec0292181c3a0e76fce37894e06cc0c3192 | INTERRUPTED_BY_SCHEDULER_AFTER_PEER_FAILURE | 25 |  |
| P16 | 31 | 3 | outer | e7b6383cfc23e4fbb8f94419a88bff5db1113babcb967672f87e0f13c74ba548 | INTERRUPTED_BY_SERVER_REBOOT | 1 |  |
| P1 | 31 | 3 | outer | d1d7f8176f18417b8107767700f68ec0292181c3a0e76fce37894e06cc0c3192 | INTERRUPTED_BY_SCHEDULER_AFTER_PEER_FAILURE | 36 |  |
| P1 | 31 | 3 | outer | e7b6383cfc23e4fbb8f94419a88bff5db1113babcb967672f87e0f13c74ba548 | INTERRUPTED_BY_SERVER_REBOOT | 0 |  |
| S16 | 31 | 3 | outer | d1d7f8176f18417b8107767700f68ec0292181c3a0e76fce37894e06cc0c3192 | INTERRUPTED_BY_SCHEDULER_AFTER_PEER_FAILURE | 14 |  |
| S16 | 31 | 3 | outer | e7b6383cfc23e4fbb8f94419a88bff5db1113babcb967672f87e0f13c74ba548 | INTERRUPTED_BY_SERVER_REBOOT | 0 |  |
| S1 | 31 | 3 | outer | d1d7f8176f18417b8107767700f68ec0292181c3a0e76fce37894e06cc0c3192 | INTERRUPTED_BY_SCHEDULER_AFTER_PEER_FAILURE | 15 |  |
| S1 | 31 | 3 | outer | e7b6383cfc23e4fbb8f94419a88bff5db1113babcb967672f87e0f13c74ba548 | INTERRUPTED_BY_SERVER_REBOOT | 0 |  |
| WC16 | 31 | 3 | outer | d1d7f8176f18417b8107767700f68ec0292181c3a0e76fce37894e06cc0c3192 | INTERRUPTED_BY_SCHEDULER_AFTER_PEER_FAILURE | 14 |  |
| WC16 | 31 | 3 | outer | e7b6383cfc23e4fbb8f94419a88bff5db1113babcb967672f87e0f13c74ba548 | INTERRUPTED_BY_SERVER_REBOOT | 0 |  |

完成270个正式模型与实际发生更新的失败/中断尝试分计，全部optimizer步数见EXECUTION_COUNTS.json；不将重试的partial步骤隐去。

## 历史引用评估修复

独立评估曾因historical LOCAL identity mismatch中止；这是历史float64独立重算与错误bitwise资格的差异，未触发训练失败或重新回放。原失败日志和旧evaluator/math凭据已按SHA保存；仅恢复既有Phase19的1e−9历史门，并新增全pair顺序/tie一致性、5项回归及原7项数学测试。具体15cell身份、AST不变和1740回放保留证据见HISTORICAL_REFERENCE_REPAIR_AUDIT.json，原失败不得静默省略。
