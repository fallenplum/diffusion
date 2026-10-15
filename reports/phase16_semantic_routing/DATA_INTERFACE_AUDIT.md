# G0 数据与接口审计

{
  "patients": 512,
  "expected_slides": 519,
  "CONCH_slides_recovered": 519,
  "CONCH_patients": 512,
  "CONCH_original_patches": 1278642,
  "historical_128_patches": 66432,
  "historical_1024_patches": 474959,
  "UNI2_candidate_slides": 520,
  "UNI2_with_coords": 520,
  "raw_WSI_files_discovered": 0,
  "raw_RNA_shape": [
    512,
    20531
  ],
  "raw_CNA_shape": [
    512,
    25131
  ],
  "RNA_CNA_common_symbols": 18904,
  "RNA_missing_gene_counts_distribution": {
    "0": 309,
    "3024": 203
  },
  "platform_identity": "not confirmed;missing-pattern proxy only",
  "CNA_semantics": "gene-level log2 copy-number dosage;gain/loss separate;not RNA expression or mutation",
  "candidate_pool_limited": false,
  "tissue_mask": "source H5 candidate set only;no independent raw-image tissue mask verification",
  "dataset_sha256": "4aaeb4cc22374f32615148605d824abed679fcb957c64286b6a8e44f12a294a1",
  "inventory_errors": 0,
  "inventory_seconds": 54.84566378593445,
  "UNI2_patches_all_candidate_files": 3295129,
  "UNI2_coordinate_row_counts_match_features": true
}

## 语义空间

{
  "status": "SEMANTIC_ROUTE_BLOCKED_PENDING_CONCH_Z",
  "TITAN_text_to_raw_CONCH_v15_cosine": "PROHIBITED_NOT_SAME_SPACE",
  "TITAN_scope": "verified slide text alignment;retain S32/native baseline",
  "original_CONCH_v15_release": "patch visual weights;no compatible contrastive projection+text checkpoint in audited assets",
  "CONCH_Z_official_revision": "60211fced9c25a014494e248eedb6a6be80109f7",
  "CONCH_Z_required_operations": [
    "project_image_features(raw768)",
    "encode_text(tokenizer(prompts),normalize=True)"
  ],
  "CONCH_Z_access": "new user token authenticated config GET403:request awaiting review from repo authors;no bypass",
  "sources": [
    "https://github.com/mahmoodlab/LIMIT",
    "https://github.com/mahmoodlab/TITAN/issues/17",
    "https://github.com/mahmoodlab/CONCH"
  ]
}

## 坐标与授权

检查519张CONCH候选切片；UNI2有坐标切片数520；原始WSI文件数0。未完成30组原图/空间验证前不运行双视觉分支，不按行号匹配。
现有TITAN权重沿用此前用户下载/服务器常驻授权；官方模型限非商业研究。CONCH-Z需独立人工审批；不尝试绕过门禁。CONCH1.0模型若存在，不能直接替代CONCH1.5预计算特征的语义空间。

模型权重与每张H5的SHA和完整患者切片关联留在私有ASSET_INVENTORY.json；公开只含匿名覆盖与错误计数。原数据只读；RNA/CNA预处理及后续文本阈值均在对应训练分区拟合。

## E盘重新接入后的范围修复

**candidate-pool-limited=true**：519/520张原始切片有CONCH，覆盖512/512名患者。E盘全部CONCH及HF缓存检索未发现该队列额外文件。519个H5的所有原始patch行均恢复；一张原始UNI2切片缺少对应CONCH，同患者另一张切片已存在，不重命名/替换。最初将历史519个CONCH文件均恢复等同于完整原始WSI集合是范围标注错误，已记录BUG_LOG；不改变任何特征、训练、假设或预测。所有已训练arms统一使用这519张切片。
