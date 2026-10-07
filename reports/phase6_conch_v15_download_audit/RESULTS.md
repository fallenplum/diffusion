# CONCH v1.5 下载与队列对应关系验收

审计日期：2026-10-07。范围：本机 E 盘上的预计算病理特征；没有训练模型、读取 DSS/OS 标签或上传特征到服务器。

## 下载及保存结果

| 项目 | 实测结果 |
|---|---:|
| 来源 | `sofieneb/conch_v15_features` |
| 固定版本 | `b19a10ae891b76407050c6b8a1cec4d8a2bd55f3` |
| 下载范围 | TCGA-COAD / TCGA-READ |
| 下载文件数 | 624（COAD 458，READ 166） |
| 实际传输文件体积 | 4,704,115,808 bytes；4.704 GB / 4.381 GiB |
| 下载耗时 | 1,790.3 秒；约 29 分 50 秒 |
| 对应当前队列的文件 | 519（COAD 389，READ 130） |
| 队列文件体积 | 4,075,428,024 bytes；4.075 GB / 3.796 GiB |
| 队列交付目录 | `E:\diffusion_conch_v15_features\TCGA` |
| 下载缓存目录 | `E:\hf_cache_conch_v15` |

E 盘为 exFAT，不支持硬链接。因此，验收后将 519 个匹配文件在同一 E 盘内移动至交付目录，文件内容不变，未创建第二份特征副本。其余 105 个下载文件留在缓存中，未进入当前队列清单。原 HF snapshot 已被整理，不再是完整快照；后续加载请使用交付目录和 `cohort_manifest.csv`。

## 全量完整性核验

对全部 624 个文件执行：

1. SHA256 与固定版本 HF tree 中的 `lfs_sha256` 比较，并核对文件大小。
2. H5 `features` 和 `coords` 结构检查；全部为 `features(N,768)`、float32，以及整数 `coords(N,2)`。
3. 读取全部特征数组检查有限值；全部无 NaN/Inf。
4. 检查坐标长度与特征长度一致，坐标无负值、无重复位置。
5. 核验 dataset 属性中的 encoder、patch size、target magnification。

**624/624 通过，错误数为 0。** 特征共 1,475,810 个 patch。提取属性保存在 `coords.attrs` / `features.attrs`，而不是 H5 根属性；此前仅检查根属性不能证明元数据缺失。

详细证据：`download_manifest.csv`、`validation.json`、`run_manifest.json`。报告不包含特征数组、权重或访问令牌。

## 切片与患者覆盖

精确匹配同时要求完整 TCGA slide barcode、DX 编号、UUID 和所属项目相同；没有按患者 ID 或短文件名前缀替换切片。

| 检查 | 结果 |
|---|---:|
| 原 UNI2 队列切片数 | 520 |
| 精确切片匹配 | 519/520（99.81%） |
| 原队列患者数 | 512 |
| 精确匹配切片对应的患者数 | 512/512（100%） |

缺失切片：

```text
TCGA-QG-A5Z1-01Z-00-DX1.F3157C57-0F35-42D3-9CA5-C72D93F1BF89.h5
```

该患者的下列 DX2 切片本来就在原始 520 张切片内，且已精确匹配：

```text
TCGA-QG-A5Z1-01Z-00-DX2.2CE72B6A-557F-43BD-BA4C-B252E14E46EF.h5
```

因此没有丢失患者，但不能声称原始 520 张切片全部齐备。`cohort_manifest.csv` 保留完整 520 行，其中缺失 DX1 标记为 `MISSING_CONCH_SLIDE`；没有复制 DX2 冒充 DX1，也没有将缺失切片填成零特征。

患者级多切片池化可在保持原患者划分的前提下使用现有 519 张；涉及原始切片集合完全相同的比较时，需明确采用共同 519 张切片，或补齐缺失 DX1。不能无声改变既有切片聚合协议。

## UNI2 与 CONCH 的 patch 对应关系

| 提取属性 | 原 UNI2 | 下载的 CONCH v1.5 |
|---|---|---|
| 目标放大倍数 | 20X | 20X |
| 目标图像 patch 大小 | 256×256 | 512×512 |
| 示例原切片放大倍数 | 40X | 40X |
| 示例 level-0 覆盖边长 | 512 | 1024 |
| 特征维度 | 1536 | 768 |
| 对匹配切片的完整坐标数组相同 | — | 0/519 |
| 忽略 patch 行顺序后坐标集合相同 | — | 0/519 |
| level-0 patch 大小相同 | — | 0/519 |

**不能将 CONCH 第 j 行权重直接用于 UNI2 第 j 行特征。** 坐标系统均可追溯到原切片，但覆盖范围、patch 数和提取网格不同。仅有相同左上角也不能证明同一图像区域；尚未建立跨尺度 patch 映射。

这里纠正先前资源评估中的“间距均为 512，表面一致”：40X 切片中，UNI2 的 level-0 间距 512 对应的是 20X 的 256-pixel patch；CONCH 的 20X 512-pixel patch 覆盖 level-0 的 1024-pixel 区域。提取设置实际不同。

## 对 CTF_2 路线的边界

- **可用：** 固定的 CONCH v1.5 patch 特征作为独立病理分支；可开展患者级/切片级池化和融合，不必为了该步骤下载整批原始 SVS。
- **尚不可宣称：** 完成原始 CONCH 的 CTF_2 复现、UNI2↔CONCH 的 patch 一一对应，或 patch↔text 语义空间校验。
- **文本语义门槛：** 已下载的是 768 维 `conch_v15` 图像表示，不含文本嵌入。旧 `phase_k1_conch_extract.py` 使用原始 `conch_ViT-B-16` 和 `proj_contrast=True`；不能直接把其文本向量与本次表示混用，也不能仅通过补零或线性层使维度相同就宣称语义对齐。
- 官方 TITAN 文档给出了从 CONCH v1.5 patch 特征及坐标生成 slide embedding，再使用 TITAN 的语言/切片编码器进行语义任务的路径。这是一个无需原始 SVS 的备选，但属于 slide-level 路线；仍需固定模型版本、确认访问权限并进行独立语义验收，不能冒充原始 CTF_2 patch-level 过程。

## 优势、限制与置信度

优势：当前 512 位患者全部有至少一张精确匹配的 CONCH 切片，实际文件仅约 4.7 GB；数据留在本机 E 盘，减少原始 WSI 存储及下载成本。

限制：一个原始 DX1 缺失；两个 encoder 的 patch 网格不同；768 维图像表示到文本概念的兼容性尚未验证。下载成功不证明生存预测收益。

置信度：**高**（文件完整性、覆盖率及 patch 设置为全量实测）；患者级预测收益及 CTF_2 文本语义兼容性仍未建立证据。

## 复现说明

执行脚本：`scripts/finalize_conch_v15_download.py`。输入包括固定版本 HF tree 元数据和原服务器 UNI2 坐标哈希清单；后者已按原始字节保存为 `uni2_coordinate_inventory.json`，其 SHA256 与运行 manifest 中的输入哈希一致。输出为文件 SHA256 清单、完整 520 行队列匹配清单、运行 manifest 和分项验收状态。脚本不读取生存标签。

```powershell
D:\Anaconda\envs\tslib\python.exe scripts\finalize_conch_v15_download.py `
  --snapshot-root E:\hf_cache_conch_v15\datasets--sofieneb--conch_v15_features\snapshots\b19a10ae891b76407050c6b8a1cec4d8a2bd55f3 `
  --tree-json E:\hf_cache_conch_v15\datasets--sofieneb--conch_v15_features\trees\b19a10ae891b76407050c6b8a1cec4d8a2bd55f3.json `
  --uni-inventory reports\phase6_conch_v15_download_audit\uni2_coordinate_inventory.json `
  --output-dir reports\phase6_conch_v15_download_audit `
  --delivery-root E:\diffusion_conch_v15_features `
  --download-seconds 1790.3
```

整理后重跑时，脚本会从交付目录读取已移动的文件进行校验。特征数据不进入 Git 仓库；Git 只保存报告、manifest 和审计脚本。

## 外部资料

- [预计算特征发布者数据卡](https://huggingface.co/datasets/sofieneb/conch_v15_features)
- [CONCH v1.5 官方模型卡](https://huggingface.co/MahmoodLab/conchv1_5)
- [TITAN 官方用法及 patch-to-slide 接口](https://github.com/mahmoodlab/TITAN)
- [TRIDENT 官方 encoder 实现](https://github.com/mahmoodlab/TRIDENT/blob/main/trident/patch_encoder_models/load.py)
