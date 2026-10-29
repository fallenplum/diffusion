# 官方 HEST CRC 坐标字段语义审计（2026-10-29）

**结论：官方固定数据中的 COAD 与 READ 使用不同的存储坐标约定。READ 四份不是统一的 x/y 左上角坐标；它们严格等于提供方 spot 中心的行/列像素坐标取整。** 冻结特征提取器中的通用 `upper-left` 文字不能作为 READ 的合格空间合同。原始坐标、图像和冻结提取器均未修改。

匿名收据：[HEST_COORDINATE_SEMANTICS_AUDIT_20261029.json](HEST_COORDINATE_SEMANTICS_AUDIT_20261029.json)，SHA256 `adef67a27ea73dffdc9535b0609dfd08d57215ecaa9060dff3bfc9095ef93c77`。

## 实际核验范围

在 E 盘只读固定 hest-bench 版本 `b039b5965ed7092f280d4f336da7972ef9a10e1a` 的 8 份正确 RGB H5 的坐标/条码、8 份官方 H5AD 的观察索引、`obsm.spatial` 与显式像素坐标字段，按条码对齐 **24,058 个 patch**。没有读取表达 X、X 的原始文件字节或 RGB 像素，没有向服务器传输 RNA/obsm，没有新训练或模型加载。

首次脚本含完整 H5AD 哈希，被自动审批认定违反本次“不读 X”边界并在执行前拒绝；移除后完成了获准的坐标审计。完整 H5AD 和 RGB 的 SHA 引用此前实际固定版本资格/下载收据，本次没有重新读取这些大文件来计算哈希。授权坐标数组本身已重新计算 SHA；公开文件只保存摘要和误差，不含坐标数组或条码。

## 逐元素结果

8 份官方表达中的 `obsm.spatial` 均严格等于同一文件 `obs[[pxl_col_in_fullres, pxl_row_in_fullres]]`，最大绝对误差为 0。以下关系均在真实条码对齐后成立：

| 任务 | 正确 benchmark H5 的 `coords` 与提供方字段关系 | 支持量 | 最大误差 |
|---|---|---:|---:|
| COAD | `trunc([col,row] − floor(img.patch_size × img.factor / 2))` | 4 份，15,651 patch | 0 |
| READ | `trunc([pxl_row_in_fullres, pxl_col_in_fullres])` | 4 份，8,407 patch | 0 |

COAD 四份对应整数半裁剪值为 204、224、204、204 像素；它们由各 H5 的实际属性定义，没有根据外部成绩选择。READ 与未经取整的行/列中心最大误差小于 1 像素，取整后逐元素完全一致。这里的“中心”指提供方 spot 坐标字段的约定，不能扩大为我们已独立证明 RGB 裁剪真实组织学中心正确。

其他执行者已分别对旧主 HEST 与正确 benchmark 图像完成全数组比较，发现 READ 的 RGB 也不同。因此不能只交换坐标轴、减去常数后就将旧 RGB/FP32 特征当成正确 benchmark 版本。本审计没有重复读取其全套 RGB。

## 官方源码的解释范围

已通过 GitHub 插件读取并将原字节独立缓存核验，固定 HEST commit 为 `3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3`：

| 文件 | Git blob | 原字节 SHA256 |
|---|---|---|
| `HESTData.py` | `c2a76194ab23c690c8383eb24752f9229c728146` | `546260e7b94d7483f0ed80c290b69bedfadc3452526803adc532c3454e255e2c` |
| `readers.py` | `6d9b1cb7259abb3575dedc03b486001e7002371f` | `36db090a33c252b295a0537f8e3872ca7d9bc2d1def0bf0ce9796f20e5051d43` |
| `trident_compat.py` | `1aec76d2c3d870533f116d5ed9d859f5bedc1269` | `3c9874c15c9b4c8df03b897b808a86e069f838940f845547e59c425ca470c659` |

[官方 Visium 读取代码](https://github.com/mahmoodlab/HEST/blob/3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3/src/hest/readers.py#L444)按列、行组织 `obsm.spatial`。[同版本 patch 写出代码](https://github.com/mahmoodlab/HEST/blob/3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3/src/hest/HESTData.py#L395)将该坐标视为中心，减去半个原生裁剪尺寸后转为整数，再交给 patcher。**没有证据证明固定旧 benchmark 资产就是由该当前写出路径生成**，不能用当前代码覆盖真实文件中已确认的不同约定。

因此现在可以确认存储字段的提供方中心/角点关系，但仍未证明全部 RGB 的裁剪变换、原生图像配准或跨平台物理尺度。本轮不得将像素直接称为 μm，也不据此批准空间 GNN 或医学区域定位。

## 为什么不改变新 4-Ridge 数值基线

新 4-Ridge 的 loader 读取坐标只用于形状和有限数检查；返回的 `Partition` 只有 features、targets、barcodes、specimen_index、specimen_ids、donor_codes 和 genes，**不返回 coords**。Scaler、PCA、Ridge 与 `predict` 只消费图像 embeddings，预测接口没有 RNA 或坐标。因而未获资格的 `upper-left` 文字不进入当前数值预测；仍须以独立勘误保留该错误描述，不能据此解释空间机制。

独立只读核验的 11 份新实现/配置/测试/许可文件全部匹配冻结 manifest，canonical SHA 为 `0c33b463e556ea22d461bc7e2be7747dfa1d528ac815e62dbdc5534bc8b2ec3e`。实现者的 22 项合成测试日志原字节 SHA 为 `6909d401dcf83733dbd999f67bf9bfd8db8b6cef022de3f80dc0b1c6149f5df9`；本审查核验日志与 manifest，没有重复测试或模型拟合。回放后 artifact SHA、逐资产资格绑定、合法图像 QC 子集三项审查已闭环，**真实 8 组 BF16 特征和四个正式 Ridge 拟合仍需各自实际验收，不能由源码审查代替**。

该审计的优势是用精确字段等式限定坐标语义，避免靠固定偏移猜测；限制是没有原生 WSI/RGB 裁剪的独立组织学复核，也未建立 μm 物理空间合同。字段关系的置信度为高；真实组织位置与医学空间解释保持未验证。
