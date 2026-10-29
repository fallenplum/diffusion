# Frozen-artifact public projection

This is a lossless, alias-labelled publication projection, not the original executable artifact. Restore and verify the original bytes with `scripts/materialize_hest8_public_projections.py` before use. Scientific computation used the original SHA listed in the projection manifest.

```json
{
  "schema": "hest_official8_uni2_bf16_feature_recipe_v1",
  "status": "LOCKED_BEFORE_TRUE_BF16_DEVICE_GATE_AND_EIGHT_FEATURE_EXTRACTIONS",
  "dataset_repo": "MahmoodLab/hest-bench",
  "dataset_revision": "b039b5965ed7092f280d4f336da7972ef9a10e1a",
  "patches": [
    {
      "task": "COAD",
      "section": "PUBLIC_COAD_SECTION_1",
      "path": "COAD/patches/PUBLIC_COAD_SECTION_1.h5",
      "bytes": 786498800,
      "sha256": "2b55b02a95bb207194402cf408643011bcc27a3df92c80e03faeb4cf5d2c491b"
    },
    {
      "task": "COAD",
      "section": "PUBLIC_COAD_SECTION_2",
      "path": "COAD/patches/PUBLIC_COAD_SECTION_2.h5",
      "bytes": 529126536,
      "sha256": "60db06f90d56f092e244b697e5391a37ab42b581c437c11e458ff059c63c227d"
    },
    {
      "task": "COAD",
      "section": "PUBLIC_COAD_SECTION_3",
      "path": "COAD/patches/PUBLIC_COAD_SECTION_3.h5",
      "bytes": 547397752,
      "sha256": "2bd7394a3c3481f8c7c6ff429e2a4310f26d608361f93dcac24c7032e1a04d70"
    },
    {
      "task": "COAD",
      "section": "PUBLIC_COAD_SECTION_4",
      "path": "COAD/patches/PUBLIC_COAD_SECTION_4.h5",
      "bytes": 560565320,
      "sha256": "a17254609041016e0459da70993bb3df7d23288b5d2c195b03f4d2cb92f2ae38"
    },
    {
      "task": "READ",
      "section": "PUBLIC_READ_SECTION_1",
      "path": "READ/patches/PUBLIC_READ_SECTION_1.h5",
      "bytes": 261863088,
      "sha256": "36186c62d13f29d22cc59d831b52114008ab623e5e28c2936d23805e52295e7d"
    },
    {
      "task": "READ",
      "section": "PUBLIC_READ_SECTION_2",
      "path": "READ/patches/PUBLIC_READ_SECTION_2.h5",
      "bytes": 329534992,
      "sha256": "f9e8486c3ed24c2cf5fb8c1fc85d5ebc5f20545978a8d604aac9ec478c5576b0"
    },
    {
      "task": "READ",
      "section": "PUBLIC_READ_SECTION_3",
      "path": "READ/patches/PUBLIC_READ_SECTION_3.h5",
      "bytes": 369325200,
      "sha256": "48476d43b5c91f7ef38bfb180bd8d950ef2fca88539eb7bc75f1c517b3b5f64d"
    },
    {
      "task": "READ",
      "section": "PUBLIC_READ_SECTION_4",
      "path": "READ/patches/PUBLIC_READ_SECTION_4.h5",
      "bytes": 341145096,
      "sha256": "0561b4b92b50adbe15160344f4ba7dec060392c4e3d7d7bc900238cefbff3bd4"
    }
  ],
  "raw_RGB_total_bytes": 3725456784,
  "raw_download_destination": "E_DATA_VOLUME_ONLY",
  "server_raw_retention": "one_current_slide_removed_only_after_feature_return_to_E_SHA_verified",
  "seed": 1,
  "batch_size": 128,
  "num_workers": 1,
  "forward_precision": "bfloat16",
  "parameter_dtype": "float32",
  "row_L2_normalized": false,
  "stored_feature_dtype": "float32_container_only",
  "model_repo": "MahmoodLab/UNI2-h",
  "model_revision": "d517a8dd47902dd7c308b3c36f63bce47e7b9a43",
  "model_weight_sha256": "6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e",
  "model_weights_reused": true,
  "model_weights_retained_on_server": true,
  "HEST_commit": "3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3",
  "TRIDENT_commit": "bdeeb74b7eaad8d48bda232bde7de66cae980918",
  "primary_code": {
    "HEST_benchmark_Git_blob": "f7179acb4462dfc84b949eb16f8728f72387af6a",
    "HEST_dataset_Git_blob": "57655967cc9d540c315806cec31948aa2c62d23f",
    "TRIDENT_loader_Git_blob": "07d8ca5197b1e527a3e04c78abd4f0af6de50223"
  },
  "model_build": "unchanged_pinned_UNIv2InferenceEncoder._build_local_FP32_weights_branch",
  "image_data": "unchanged_pinned_H5PatchDataset_AST_RGB224_PIL_Resize224_CenterCrop224_ToTensor_ImageNet",
  "runtime_flags": "official_defaults_recorded_not_overridden_by_SPGEL_training_runtime",
  "hardware_gate_scope": "fixed_first128_authorized_correct_bench_PUBLIC_COAD_SECTION_3_device_only_not_benchmark_feature",
  "FP32_BF16_difference": "diagnostic_only_not_false_requirement_of_FP32_equivalence",
  "old_FP32_caches": "preserved_not_used_as_current_official_BF16_features",
  "gene_expression_clinical_or_survival_inputs": false,
  "new_optimizer_fits": 0,
  "new_optimizer_updates": 0,
  "H5_keys": [
    "embeddings",
    "barcodes",
    "coords"
  ],
  "H5_sample_binding_attributes": [
    "official_sample_id",
    "source_patch_sha256",
    "forward_precision",
    "row_L2_normalized"
  ],
  "extractor_sha256": "7fb5571cfa80a8ec1a345cabbb35f23b60d8b650d461e4e3b912e3542a0e7938"
}

```
