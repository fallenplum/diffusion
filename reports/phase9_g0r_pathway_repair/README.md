# Phase9-G0R

运行入口：

```bash
python scripts/run_phase9_g0r.py <private-output> <private-dataset-parent> <private-context-dir> <kegg-cache>
python scripts/validate_phase9_g0r.py <private-output> <private-dataset-parent>
```

训练和评测产生的患者级 OOF、索引、模型状态和原始数据只能保存在服务器私有目录。仓库只提交聚合指标、审计和配置。

R7(mean5) 是 5 个预先固定 degree-preserving null graph 的聚合，不是独立队列。
