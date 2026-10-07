# CONCH 预计算特征与原始 SVS 可用性审计

**审计日期：** 2026-10-07  
**对象：** 当前 Phase5/Phase6 使用的 TCGA-COAD/READ 服务器队列  
**结论级别：** 资源可用性审计，不是 CONCH 生存结果

## 下载后更新（2026-10-07）

已获得数据集访问权限并完成 E 盘下载：COAD/READ 共 624 个文件，4.704 GB，约 29 分 50 秒；624 文件 SHA256 及全量数值检查通过。与当前队列精确匹配 519/520 张切片，覆盖 512/512 位患者。队列文件已整理到 `E:\diffusion_conch_v15_features`。

**更正此前估计：** 文件数量多于本队列不代表所有切片都能匹配；一个原始 DX1 缺失，但同一患者的原有 DX2 已匹配。实际提取设置为 UNI2 20X/256px 与 CONCH 20X/512px，所有 519 个匹配切片的完整坐标数组均不相同，不能按 patch 行号配对。

下文保留的是下载前资源评估（当时网络、权限及 E 盘状态）；当前实测状态、差异和方法限制以 `../phase6_conch_v15_download_audit/RESULTS.md` 为准。特征文件保留在 E 盘，没有上传服务器。

## 1. 当前服务器库存

通过只读 SSH 检查新服务器：

| 项目 | 结果 |
|---|---:|
| 病理 H5 文件 | 520 个 |
| COAD / READ | 390 / 130 个 |
| 唯一患者 | 512 个 |
| 多切片患者 | 7 个 |
| 现有 H5 总大小 | 20,545,482,072 bytes（约 19.13 GiB） |
| H5 keys | `annots`, `coords`, `coords_patching`, `features` |
| 示例特征形状 | `(1, N, 1536)`，`float32` |
| 原始 `.svs` / `.mrxs` | 未发现 |
| CONCH 特征文件 | 未发现 |
| CONCH checkpoint | 未发现 |
| CONCH 代码 | `/root/data/models/CONCH_repo`，仅代码和配置 |

现有 1536 维 H5 是已提取的 UNI2 类病理特征，不是 CONCH。服务器上没有可直接用于 CTF_2 的 CONCH patch 特征或原始切片。

## 2. 已找到的预计算 CONCH 数据

公开候选：[`sofieneb/conch_v15_features`](https://huggingface.co/datasets/sofieneb/conch_v15_features)。数据卡声明：

- 11,760 张 WSI，其中 TCGA 9,838 张、CPTAC 1,922 张；总大小 **121 GB**。
- TCGA-COAD 458 张、TCGA-READ 166 张，覆盖规模大于当前服务器的 390 + 130 张。
- 使用 CONCH **v1.5**、TRIDENT、20X、512×512 patch、无重叠。
- 每张 WSI 一个 H5，包含 `features`（`N×768`）和 level-0 `coords`。
- 数据集文件访问需要在 Hugging Face 登录并同意联系信息条款；许可为 CC BY-NC-ND 4.0。

### 对本项目的适用性

| 检查项 | 判断 |
|---|---|
| 患者/切片命名 | **有希望直接对齐**：数据卡保留完整 TCGA barcode + UUID，当前服务器 H5 也采用同一命名格式 |
| 项目覆盖 | **足够**：候选集 COAD/READ 共 624 张，当前队列 520 张 |
| 特征维度 | **需要适配**：候选为 768 维，当前 UNI2 为 1536 维 |
| patch 设置 | **表面一致**：当前 `coords_patching` 常见间距为 512；仍需逐文件核对候选坐标和 patch 数 |
| patient-level MIL / concept path | **可用候选**：不要求 UNI2 与 CONCH patch 一一对应，可按患者/切片独立池化 |
| patch-to-patch CTF_2 严格复现 | **尚未证明**：候选的组织分割和 patch 坐标可能与当前 UNI2 提取不同 |
| text-concept 相似度 | **不能仅靠该数据集完成**：它提供图像特征，不提供匹配的 CONCH 文本嵌入；需要同版本 text encoder 或预计算文本向量 |
| 严格论文复现 | **版本风险**：候选是 CONCH v1.5，不能直接宣称等于论文中使用的原始 CONCH 版本 |

因此，当前最合理的使用方式是：把它作为独立的 **CONCH visual-concept branch**，在患者级/切片级先池化，再与 Clinical、RNA 和 UNI2 分支融合；暂时不要假设它与 UNI2 的 patch 坐标完全相同。

## 3. 下载量与时间估算

### 3.1 预计算 CONCH 特征

按数据卡公布的全库平均值估算：

```text
121 GB × 520 / 11,760 ≈ 5.35 GB
```

由于每张切片的 patch 数差异很大，实际规划空间应按 **5–10 GB**，并额外预留校验/临时文件空间。预计传输时间（仅传输，不含解压、校验和上传）：

| 有效速度 | 约 5.35 GB |
|---:|---:|
| 1 MB/s | 约 89 分钟 |
| 5 MB/s | 约 18 分钟 |
| 10 MB/s | 约 9 分钟 |
| 50 Mbps | 约 14 分钟 |
| 100 Mbps | 约 7 分钟 |

这是平均值估算，不是已实测的单文件下载时间；HF 数据集本身支持断点续传。

### 3.2 原始 SVS 备选

GDC 官方说明 TCGA 组织/诊断切片可通过 GDC API 或 Data Transfer Tool 下载。公开的 GDC 文件清单审计中，474 张 TCGA SVS 的合计约为 472 GiB；按这个经验均值外推当前 520 张，约为 **518 GiB（约 557 GB）**。考虑项目差异，实际规划按 **0.5–0.7 TB** 更稳妥。

| 有效速度 | 约 518 GiB |
|---:|---:|
| 1 MB/s | 约 6.2 天 |
| 5 MB/s | 约 29.6 小时 |
| 10 MB/s | 约 14.8 小时 |
| 50 Mbps | 约 23.8 小时 |
| 100 Mbps | 约 11.9 小时 |

SVS 大小有明显长尾，单张文件可能从约 0.5 GB 到数 GB；正式下载前应先用 GDC API 按当前 520 个完整文件名生成 manifest 并读取 `file_size`，不要用上述外推值替代 manifest。

## 4. 网络与位置判断

本次只读测试中，服务器到 `huggingface.co:443` 和 `api.gdc.cancer.gov:443` 均在约 8–10 秒后超时。服务器上的实验 SSH 连接正常，但这不代表服务器具有公网下载能力。

当前建议顺序：

1. 在本机浏览器登录 Hugging Face，接受 `sofieneb/conch_v15_features` 的访问条款。
2. 将 **仅 COAD 和 READ** 的预计算特征下载到本机 E 盘；不下载全库 121 GB。
3. 下载完成后按完整 slide filename + UUID 做 520 行 manifest join，检查覆盖率、`features` 维度、`coords`、patch 数和缺失项。
4. 只有在覆盖率不足或模型版本不满足实验协议时，才为缺失切片生成 GDC SVS manifest；不要直接下载 0.5–0.7 TB 全量 SVS。
5. 若预计算特征通过校验，再分批上传约 5–10 GB 到服务器；若服务器出站网络仍不可用，这比在服务器端下载原始 SVS 更可控。

本次 Codex 沙箱没有看到本机 `E:` 盘挂载，因此无法替用户确认 E 盘剩余空间。下载前需要人工确认 E 盘至少有 **15 GB 可用空间**（预计算特征方案）或 **0.7 TB 以上可用空间**（原始 SVS 方案）。

## 5. 审计结论

**优先选择预计算 CONCH v1.5 特征，不立即下载原始 SVS。**

- 资源上：候选集覆盖数量足够，预计只需 5–10 GB。
- 方法上：可用于患者级 CONCH 分支和概念表示探索；不能未经验证就宣称完成 CTF_2 的 patch-level 严格复现。
- 风险上：需要 HF 条款访问、逐文件覆盖率核对、CONCH v1.5 与目标论文版本差异声明，以及匹配的 text encoder/文本嵌入。
- 只有当这些门槛不通过时，才转向 GDC 原始 SVS；原始 SVS 的存储和传输成本约高两个数量级。

**置信度：中高。** 服务器库存和网络结论是本次实测；预计算集规模、格式和许可来自其数据卡；520 张 SVS 的体积是基于公开 GDC 文件清单的外推，最终数值仍需 GDC manifest 确认。

## 参考来源

- [CONCH v1.5 预计算特征数据卡](https://huggingface.co/datasets/sofieneb/conch_v15_features)
- [CONCH 官方模型卡](https://huggingface.co/MahmoodLab/CONCH)
- [GDC Data Portal](https://gdc.cancer.gov/access-data/gdc-data-portal)
- [GDC API 文档](https://docs.gdc.cancer.gov/API/Users_Guide/Search_and_Retrieval/)
- [TCGA-COAD 项目页](https://portal.gdc.cancer.gov/projects/TCGA-COAD)
- [公开 GDC SVS 体积审计示例](https://the-obstacle-is-the-way.github.io/gigapixel-goblin/data/data-acquisition/)
