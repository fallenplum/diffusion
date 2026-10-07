# CONCH v1.5 下载验收

- `RESULTS.md`：当前完成状态、缺失切片、patch 网格差异和方法适用边界。
- `download_manifest.csv`：624 个源文件的 SHA256、维度、坐标哈希和提取设置。
- `cohort_manifest.csv`：原始 520 张切片的对应关系，包含缺失的一张 DX1。
- `uni2_coordinate_inventory.json`：审计输入的原 UNI2 坐标哈希及提取元数据。
- `validation.json`：完整性、患者覆盖、切片覆盖、patch 对应关系的分别验收。
- `run_manifest.json` / `summary.json`：运行输入哈希、数据版本及实测汇总。

特征数组保存在本机 `E:\diffusion_conch_v15_features`；不进入 Git 仓库，也没有上传服务器。
