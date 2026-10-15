# 授权后复现

父提交a19eb457bc8f7fcbf11ffaf752f00c5aa0d8cf61，分支codex/phase16-semantic-roi-routing。需要已授权个人账号下载的CONCH-Z固定revision、Phase10–15私有数据/原始H5；仓库不含这些数据或权重。

```bash
export PYTHONPATH=/root/phase10_trimodal_eval_deps
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/root/miniconda3/bin/python scripts/pipeline_phase16_authorized.py
```

该脚本按官方空间、模块pilot、formal、独立审计与汇总顺序执行。模型目录/root/phase16_conch_z/model，E盘原件E:/diffusion_phase16_conch_z/model。原始候选H5只读。已有同源formal cells复用；source/数据/缓存/权重SHA不匹配时拒绝。脚本不尝试绕过许可或自动解锁缺原图的G4。

私有目录：/root/phase16_semantic_private、/root/phase16_omics_private、/root/phase16_mil_private。语义模型与G5不同hash独立保存，旧Phase16文件不可变。重要checkpoint、ROI和小型重放胶囊持久化；/dev/shm中的标准化视觉/投影/完整文本缓存可重建，重放胶囊允许审计在完整文本缓存丢失后继续。

各模块实际命令见pipeline脚本；最终publish_phase16_authorized.py仅发布ARTIFACT_MANIFEST中的匿名列表，并校验远端commit/blob。
