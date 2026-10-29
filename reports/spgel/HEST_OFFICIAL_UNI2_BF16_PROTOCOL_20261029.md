# 官方 HEST 八样本 UNI2-h BF16 特征合同

本轮重新提取当前官方 HEST-bench 八份 RGB 的冻结基础特征。它不训练编码器、程序模型或生存头，不计入原 SPGEL 的74次拟合、198个回放状态。原有16组 FP32 特征、原始数据与既有科学报告保持不变。

## 固定来源与算子

- 数据固定为 `MahmoodLab/hest-bench` revision `b039b5965ed7092f280d4f336da7972ef9a10e1a` 的 COAD/READ 八份官方 patch H5，总计3,725,456,784字节；逐文件大小与SHA256固定在新配置中。
- 模型固定为 `MahmoodLab/UNI2-h` revision `d517a8dd47902dd7c308b3c36f63bce47e7b9a43`，完整权重SHA256为 `6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e`。已有服务器权重整文件实测匹配，保留作后续使用。
- 直接执行固定 TRIDENT `UNIv2InferenceEncoder._build` 的原始本地权重分支，以及固定 HEST `H5PatchDataset` 类；只提取这些已核源码定义，不导入表达数据加载或预测代码。
- 模型参数为FP32，`eval()`、全参数冻结；前向在 `inference_mode` 和 CUDA BF16 autocast 下执行。默认TF32/确定性设置如实记录，不套用旧源训练程序的覆盖设置。
- 预处理为PIL RGB224、Resize224、CenterCrop224、ToTensor与ImageNet均值/标准差。资格阶段已逐像素确认本批224图像的Resize/Crop为恒等。
- 固定seed1、batch128、worker1；输出1536维原始embedding，不进行row-L2。实际模型输出为FP32；容器为FP32不代表前向使用FP32算术。

精确配置为 `configs/spgel/official8_uni2_bf16_v1.json`，计算脚本为 `scripts/extract_hest8_official_uni2_bf16.py`。主源码为 [HEST benchmark](https://github.com/mahmoodlab/HEST/blob/3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3/src/hest/bench/benchmark.py)、[HEST RGB dataset](https://github.com/mahmoodlab/HEST/blob/3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3/src/hest/bench/st_dataset.py) 和 [TRIDENT UNI2-h loader](https://github.com/mahmoodlab/TRIDENT/blob/bdeeb74b7eaad8d48bda232bde7de66cae980918/trident/patch_encoder_models/load.py)。

可用 `python scripts/fetch_hest8_pinned_primary.py --output <PRIMARY_CODE_DIR>` 获取三份公开原始源码，并以 `--primary-code <PRIMARY_CODE_DIR>` 传入提取器；逐文件固定SHA不符时停止，禁止静默采用更新版本。获取脚本不访问数据或模型凭据。

## 资格与存储

先以正式八份数据内固定128块真实RGB完成无优化器设备/算子资格，记录重复误差、峰值显存与吞吐。FP32/BF16差异只作当前数值诊断，不强迫官方BF16等价旧FP32；历史失败gate的具体数值未持久化，不能事后编造或声称重建。

E盘保存全部官方原件与新特征。经明确授权，服务器一次仅保留一份当前RGB临时副本；传输恢复须先核前缀SHA，完整RGB大小和SHA通过后才读取HDF/正式提取。每张特征返回E盘后重新计算完整SHA，并实际重载核对原顺序的barcode、coords、1536维有限原始特征及样本/源文件属性；通过后只删除该份服务器临时RGB。服务器保留体积较小、后续常用的特征与权重。

新H5仅包含 `embeddings`、`barcodes`、`coords`，另以属性绑定官方sample、patchSHA、精度与配置SHA。公开报告仅包含契约、计数与哈希；RGB、逐patch embedding、barcode、坐标和权重不进入Git。

## 坐标解释的明确限制

`coords` 数组必须原样保存官方输入，不能为了匹配旧缓存而交换轴或平移。当前提取器通用的upper-left文字标签**未构成物理坐标资格**；尤其READ官方数组与旧主HEST存在轴交换及偏移关系，应以追加的几何审计为解释依据，不能把通用标签当作经证实的中心、左上角或微米坐标。官方Ridge不使用coords作为预测变量；后续空间GNN或TCGA迁移必须另行确认真实坐标框架、MPP与取材范围。

最终权威资格为 `HEST_OFFICIAL8_BF16_FEATURE_QUALIFICATION_WITH_GEOMETRY_V2_20261029.json`；其绑定初始数值资格收据SHA，并增加提供方坐标关系与通用标签更正，没有改特征矩阵、barcode或coords。READ实际storedcoords对应提供方[row,col]中心取整，COAD为[col,row]减属性半裁剪后取整；这些字段等式不独立证明RGB组织学配准。

实际完成8份、24,058行，全部返回E盘后复算完整SHA及实际H5重载。GPU设备资格batch128约194.56patch/s、峰值约4.83GB；端到端另含授权的一次一slide网络传输，不将其当作模型训练时间。0次优化器拟合/更新。

此八样本涉及既有开发与历史评价使用记录，不能重新称为新的独立外部确认。特征提取不读取RNA、Clinical、DSS或PFI。
