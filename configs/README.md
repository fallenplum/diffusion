# 实验配置

这里保存**已执行 Phase P1 参数的声明式副本**。配置文件只描述运行参数，不包含数据绝对路径、患者标识或模型权重。

## 目录

- `phase1/common.yaml`：共同的数据、编码器、采样、复现和生存参数。
- `phase1/b0.yaml`：UNI2-h mean pooling + Cox 基线。
- `phase1/b1.yaml`：ABMIL + Cox。
- `phase1/b2.yaml`：从零初始化的 Context Transformer + Cox。
- `phase1/p0.yaml`：SSL encoder + frozen linear Cox probe。
- `phase1/p1.yaml`：SSL encoder + Cox fine-tuning。
- `phase1/p2.yaml`：SSL encoder + Cox + pairwise ranking。
- `phase1/p3.yaml`：SSL encoder + Cox + IPCW ranking。

配置中的 `extends` 使用相对配置文件继承；解析后必须保留 `common.yaml` 中的既有参数。正式运行 request 还必须记录提交 SHA、seed、fold 和输出 artifact 名称。

数据根目录只能通过运行机器上的 `DIFFUSION_DATA_ROOT` 环境变量提供，禁止把 `D:\`、`/mnt/` 等真实路径写入配置或 request。
