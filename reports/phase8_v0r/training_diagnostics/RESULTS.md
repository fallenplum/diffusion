# Phase8-V0R：A3/A4/A5 训练有效性诊断

服务器未保存模型 checkpoint；本审计按原 cell 记录的 selected epoch、seed 和 fold 重新执行 inner selection 与 outer fit，不增加正式训练预算。

|Arm|记录 epoch 均值|重放 inner Harrell|outer loss last|最大梯度|交互 q norm|k norm|i norm|风险修正 max abs|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|A3|28.53|0.8273|5.0302|1.3851|0.0000|0.0000|0.0000|0.2446|
|A4|27.80|0.8241|5.0323|1.3715|0.2144|0.1986|0.0543|0.2227|
|A5|37.40|0.8333|5.0162|1.4179|0.2691|0.2446|0.0763|0.3602|

- `A4/A5` 的 q、k、interaction head 权重均为非零，未出现整条交互分支梯度阻塞。
- `risk_minus_anchor` 是实际风险修正幅度，不能仅由输入尺度约束推断。
- 内部验证 C-index 来自重放的 inner-val，旧 cell JSON 未持久化该字段。
