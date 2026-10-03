# K1 CONCH / SVS 预检

## 状态

`CONCH_READY_SVS_REQUIRED`

## 已验证

- 服务器已存在 CONCH `conch_ViT-B-16` 权重，加载成功。
- CONCH 图像编码器输出为 512 维，文本编码器对固定 8 个 CRC concepts 输出 8×512 embedding。
- 从 GDC 文件元数据映射的一张原始 SVS 已完成 MD5 和 SHA-256 双重校验；校验通过后按 H5 的真实 `coords`、level 0、512×512 patch 读取。
- 32 个真实坐标 patch 的 CONCH embedding 已生成，均为有限值；L2 归一化误差在 float16 保存精度范围内。
- 原始 SVS 在提取结束后已从 staging 删除，未进入仓库。

## 尚未开始的部分

- 服务器当前没有原始 SVS 文件。
- GDC API 当前返回维护页面，尚未恢复下载。
- 现有 S3 公开镜像可访问，但当前线路对全量约 337 GiB 的切片下载速度不足以在短时间内完成。
- K1 Stage 1 grounding、负对照和 Stage 2 survival 尚未运行；本文件不是实验结果。

## 代码

`scripts/phase_k1_conch_extract.py` 按 H5 原始坐标逐片下载、校验、提取 CONCH embedding，并在单片处理后删除 SVS；只保存派生 embedding、坐标和审计元数据。
