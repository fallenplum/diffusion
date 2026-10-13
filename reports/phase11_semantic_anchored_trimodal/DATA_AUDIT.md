# 数据复查

当前512/65、519匹配CONCH H5、原RNA20531→20511、CNA25131→25101、TITAN native768/S32固定32均实查。各H5 SHA/整数非负唯一coords/finite/encoder512@20X审计通过。UNI2为256@20X，本轮没有任何按patch行号局部配对。缺失原DX1未替换，同患者原匹配DX2保留。患者全slide同折，等权slide聚合且budgets不变病例集合。

原临床/DSS逐元素重建、源RNA/CNA及原fold清单审计通过，详见raw_*json。训练预处理由独立第二程序重建共50输入矩阵，checkpoint全部逐患者回放。

{
  "status": "PASS",
  "patients": 512,
  "DSS_events": 65,
  "slides": 519,
  "original_patches": 1278642,
  "selected_1024": 474959,
  "budgets": {
    "128": 66432,
    "512": 261064,
    "1024": 474959
  },
  "subset_checked_slides": 519,
  "sampling": "coordinate lexsort recursive widest-interval midpoint prefixes",
  "patch_sha256": "47aaecd6c3a56bcd4bbe8217a7dff7b41c0c3b638bb3da10893b5fb6da49dc8f",
  "coordinates_sha256": "8fbd610beae11a469e33c9ae3488ff92e624755ef76e5d369e2bfc08c2551f13",
  "private_slide_manifest_sha256": "078bdf7ca1391d2011457b5456f1ad7a983676b27f2f10ceae8fa1015f216f8f",
  "dataset_sha256": "4aaeb4cc22374f32615148605d824abed679fcb957c64286b6a8e44f12a294a1",
  "source_manifest_sha256": "00986d86a760989f7a269edf1e38b3dafbe77386148c3c667f477f059e2bfc09",
  "TITAN_source_exact": true,
  "fixed_folds_source_exact": true,
  "rna_shapes": {
    "raw_rna": [
      512,
      20531
    ],
    "rna_genes": [
      512,
      20511
    ],
    "raw_cna": [
      512,
      25131
    ],
    "cna_genes": [
      512,
      25101
    ]
  },
  "missing_3024_patients": 203,
  "platform_identity": "not confirmed; missing-pattern proxy",
  "old128_sampling_reused": false
}
