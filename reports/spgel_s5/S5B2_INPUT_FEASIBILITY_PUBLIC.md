# S5-B2：TRIPLEX 多尺度输入缺口的有限元数据补审

核验日期：2026-10-29。**找到八份可访问的完整 WSI 候选，总计 27,812,010,123 字节（25.902 GiB）；尚未验证它们能重现当前锁定 benchmark 的 target crop，TRIPLEX 仍为 NOT_REPRODUCED。** 本次没有下载完整 WSI、表达或权重，没有模型前向、训练、安装依赖或服务器传输。除 API/文本元数据外，只读取一次 64 KiB HTTP Range。

## 官方输入与已审计版本

TRIPLEX 固定为 `NEXGEM/TRIPLEX@4750051b9a8ba1122c9d4a30a40943edc418a1c5`。作者代码在 0.5 μm/px 下保存 224 target 和连续 1120 neighbor，邻域随后分为 5×5 tiles。其 HEST 模式先绑定目标 patch，再生成并按 barcode 对齐邻域。现有稀疏 target patch、UNI2/CONCH 邻域 embedding 均不能直接代替原生 CIGA 输入。[官方预处理](https://github.com/NEXGEM/TRIPLEX/blob/4750051b9a8ba1122c9d4a30a40943edc418a1c5/src/preprocess/prepare_data.py#L117-L171)

作者 README 指向 HEST 原始数据和预处理流程，实际 releases API 返回空列表；本次未找到与当前八张 target RGB 直接绑定的现成 1120/CIGA 发布包。这个结论仅覆盖已核查元数据，不表示网络上不存在其他文件。global 分支可以由同张切片的 target patch 经原生编码器得到，不要求额外下载另一份 dense global raster。[HEST 接入](https://github.com/NEXGEM/TRIPLEX/blob/4750051b9a8ba1122c9d4a30a40943edc418a1c5/src/preprocess/prepare_data.py#L246-L299)、[release API](https://api.github.com/repos/NEXGEM/TRIPLEX/releases)

## HEST 八份 WSI 候选

固定 `MahmoodLab/hest@7e8d5a0b0aace41d8c8ec0f6ecea80e4ad2a61ec`，八份 metadata JSON 和 24 条 WSI/metadata/patch paths-info 均返回 200，取得八份 WSI 的完整 LFS SHA256。初始标准 HF 缓存上下文返回 403；改用用户此前明确授权 HEST 的上下文后成功。因此不能把初始 403 写成用户原有 HEST 授权失效。令牌未写入报告或 E 盘快照。[固定 WSI 文件树](https://huggingface.co/datasets/MahmoodLab/hest/tree/7e8d5a0b0aace41d8c8ec0f6ecea80e4ad2a61ec/wsis)

下表 ordinal 仅供匿名绑定；完整 RGB SHA 在配套 JSON，实际 sample ID、逐文件大小、完整 WSI SHA 和 API 原文仅留 E 盘私有索引 `EIGHT_WSI_CANDIDATE_INVENTORY.private.json`。MPP 数字是官方元数据字段，尚未通过实际裁图核验。

| 匿名绑定 | embedded μm/px | estimated μm/px | 锁定 RGB SHA 前缀 |
|---|---:|---:|---|
| COAD-1 | 0.212500 | 0.273805 | 2b55b02a95bb2071 |
| COAD-2 | 0.273775 | 0.249702 | 60db06f90d56f092 |
| COAD-3 | 0.273776 | 0.273760 | 2bd7394a3c3481f8 |
| COAD-4 | 0.273777 | 0.274031 | a17254609041016e |
| READ-1 | 0.459802 | 0.459363 | 36186c62d13f29d2 |
| READ-2 | 0.459802 | 0.459840 | f9e8486c3ed24c2c |
| READ-3 | 0.459802 | 0.459130 | 48476d43b5c91f7e |
| READ-4 | 0.459802 | 0.459636 | 0561b4b92b50adbe |

COAD 合计 25,308,792,651 字节，READ 合计 2,503,217,472 字节。已有 target RGB 共 3,725,456,784 字节，无须重复下载。这里的 27.812 GB 是原始候选 WSI 文件量，不含未来派生邻域、临时解码、CIGA 权重和特征缓存；不能当作完整运行磁盘预算。

唯一范围请求为 READ-1 的 `bytes=0-65535`：HTTP 206、Content-Range 起止范围正确，实读 65,536 字节，总耗时 4.801 秒，其中响应头前 3.978 秒。它证明该候选可按范围访问；握手、缓存与小正文占主导，**不足以估计持续下载吞吐率或 27.812 GB 的完成时间**。下载时间保留 UNKNOWN，不据此外推小时数。

## 名称相同不等于 crop 匹配

目标始终是 `MahmoodLab/hest-bench@b039b5965ed7092f280d4f336da7972ef9a10e1a`。当前 main H5 的文件 SHA 与八份 benchmark H5 全部不同；这本身只能证明容器文件不相同，不能单独证明像素不同。已有几何审计还记录了部分 COAD 与 READ 的不同关系，READ 坐标不是可直接通用解释的 patch 左上角；实际 RGB 的组织学注册仍未独立验证。不能只按同 ID 或 barcode 把新 WSI 裁块宣布为当前 benchmark 的上下文。

特别是 COAD-1 的 embedded/pixel_size=0.2125 与 estimated≈0.273805，COAD-2 的 embedded≈0.273775 与 estimated≈0.249702；选择不同物理尺度会改变 560 μm 邻域对应的原始像素范围。下一步必须在不看模型测试成绩的条件下，预先锁定坐标约定、MPP、方向、缩放和边界 mask，并对 target RGB 做像素级回放，再生成同一 spot 的真实连续邻域。

## 作者 STP-Bench 新发布不能直接补齐当前八张

额外核对作者 `NEXGEM/STP-Bench@13ec2853cd63118bd5d79dca01d32b15cc7d8738` 和数据 `nexgem/STP-Bench@76510407a9ab565480895c9d37497298abf34486`。完整公共 WSI 树列 267 份，当前八张的精确 sample 文件名匹配为 0；没有取得可核验的别名 crosswalk，不能推断同来源或可替代。数据文件元信息访问返回 403，没有对该新 gated 仓库更换令牌。README 宣称的约 444 GB 是整库量，并非本项目所需量；本次不下载整库。该仓库默认共享 UNI2 表征也不能自动视为原版 CIGA 复现。[作者固定 README](https://github.com/NEXGEM/STP-Bench/blob/13ec2853cd63118bd5d79dca01d32b15cc7d8738/README.md)、[固定数据树](https://huggingface.co/datasets/nexgem/STP-Bench/tree/76510407a9ab565480895c9d37497298abf34486)

## 执行结论与边界

**状态为 PARTIAL：输入缺口有具体可下载候选，但原生多尺度输入资格仍未通过。** 优势是范围、体积、固定版本和访问路径已明确，不必无边界寻找新数据。风险是 target 像素/坐标/MPP 资格尚缺，完整 WSI 获取与解码成本尚未实测，TRIPLEX 的选轮数/测试隔离、编码器路径绑定和版本/许可边界仍须另行处理。

下一步如获准进入输入构建，应先选预先固定的一张做 WSI 全文件校验与 target crop 回放；通过后才按锁定规则生成其余邻域。未经资格确认不得启动原版 TRIPLEX 正式训练，不将本文作为生存、细胞来源或新性能证据。元数据存在性和有限访问事实置信度高；crop 匹配与完整复现可行性仍未知。

## 公开投影说明

原 Markdown SHA256：`579f1e33ee65ba75f26546d0afe6e97fd01143c6548b6cf987a4b4b58991bc47`。原始两份报告保持不变；本公开版的科学测量与结论保持一致。[配套公开 JSON](S5B2_INPUT_FEASIBILITY_PUBLIC.json) 仅将外部私有绝对目录替换为符号说明，并附原 JSON SHA256 与去识别谱系。私有目录不随投影发布，因此不声称逐字节可逆。
