"""Synthetic arithmetic only: no real cohorts, checkpoints or model forwards."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bgb_en_r6_attention_diagnostics import (
    ARMS, INDEX_NAME, MAX_REPORT_BYTES, PARTITIONS, SEEDS, activation_norms, canonical,
    entropy, gene_statistics, jaccard, load_diagnostics, paired_top_overlap,
    parameter_drift, patch_statistics, ratio, read_diagnostics,
    svd_effective_rank, top_indices, write_diagnostics,
)


def test_entropy_and_svd_raw_singular_probability():
    assert entropy([0, 1]) == 0
    assert np.isclose(entropy([.5, .5]), np.log(2))
    assert np.isclose(svd_effective_rank(np.eye(4) * .75), 4)
    assert np.isclose(svd_effective_rank(np.full((4, 6), .1)), 1)
    assert svd_effective_rank(np.zeros((4, 6))) is None
    assert svd_effective_rank(np.empty((0, 6))) is None


def test_top16_stable_ties_and_jaccard():
    weights = np.ones(5)
    valid = np.array([False, True, True, False, True])
    assert top_indices(weights, valid, topk=2).tolist() == [1, 2]
    assert top_indices(weights, valid).tolist() == [1, 2, 4]
    assert np.isclose(jaccard([1, 2], [2, 3]), 1 / 3)
    assert jaccard(None, [1]) is None
    assert jaccard([], []) is None
    assert top_indices(np.zeros(5), valid) is None


def test_all_masked_is_invalid_not_false_rank_zero():
    result = patch_statistics(np.zeros((1, 2, 4, 5)), np.ones((1, 2, 4)),
                              np.zeros((1, 2, 5), bool), np.zeros((1, 2, 4), bool))
    assert result["raw_patch_attention_svd_effective_rank"] == dict(n=0, mean=None, sd=None)
    assert result["invalid_rank_WSI"] == result["WSI_without_patches"] == 2
    for row in result["programs"]:
        assert row["valid_WSI_rows"] == 0 and row["invalid_WSI_rows"] == 2
        assert row["entropy_patch_plus_null"]["mean"] is None
    assert all(x["mean"] is None for x in result["different_program_top16_jaccard"].values())


def test_null_is_separate_from_raw_real_patch_rank():
    weights = np.eye(4)[None, None] * .75
    null = np.full((1, 1, 4), .25)
    result = patch_statistics(weights, null, np.ones((1, 1, 4), bool), np.ones((1, 1, 4), bool))
    assert np.isclose(result["raw_patch_attention_svd_effective_rank"]["mean"], 4)
    for row in result["programs"]:
        assert np.isclose(row["null"]["mean"], .25)
        assert np.isclose(row["real_patch_neff_exp_entropy"]["mean"], 1)
        assert np.isclose(row["entropy_patch_plus_null"]["mean"], entropy([.75, .25]))


def test_uniform_programs_rank_one_and_real_neff_four():
    weights = np.full((1, 1, 4, 4), .2)
    result = patch_statistics(weights, np.full((1, 1, 4), .2), np.ones((1, 1, 4), bool), np.ones((1, 1, 4), bool))
    assert np.isclose(result["raw_patch_attention_svd_effective_rank"]["mean"], 1)
    assert all(np.isclose(row["real_patch_neff_exp_entropy"]["mean"], 4) for row in result["programs"])
    assert all(x["mean"] == 1 for x in result["different_program_top16_jaccard"].values())


def test_equal_patient_weight_after_equal_valid_wsi():
    weights = np.zeros((2, 2, 4, 2)); null = np.ones((2, 2, 4))
    mask = np.zeros((2, 2, 2), bool); valid = np.zeros((2, 2, 4), bool)
    mask[0] = True; valid[0] = True; weights[0] = .25; null[0] = .5
    mask[1, 0] = True; valid[1, 0] = True; weights[1, 0] = .5; null[1, 0] = 0
    result = patch_statistics(weights, null, mask, valid)
    assert result["programs"][0]["null"]["mean"] == .25


def test_correct_wrong_top_overlap_uses_valid_intersection():
    a = np.ones((1, 1, 4, 20)) / 20
    mask = np.ones((1, 1, 20), bool)
    valid = np.ones((1, 1, 4), bool)
    result = paired_top_overlap(a, a, mask, valid, valid)
    assert all(row["mean"] == 1 for row in result)
    valid[..., 2] = False
    assert paired_top_overlap(a, a, mask, valid, valid)[2]["mean"] is None


def signed_example():
    ep = np.array([0, 0, 0, 1, 2, 3]); sign = np.array([1, 1, -1, 1, 1, 1])
    defined = np.array([[True, True], [True, False], [True, False], [True, False]])
    count = np.array([[[2, 1], [1, 0], [1, 0], [1, 0]]])
    return dict(weights=np.array([[.5, .5, 1, 1, 1, 1]]), measured_counts=count, retained_counts=count,
                coverage=defined[None].astype(float), component_mask=defined[None], component_defined=defined), ep, sign


def test_gene_up_down_separate_normalization_and_missing_defined_group():
    readout, ep, sign = signed_example()
    result = gene_statistics(readout, ep, sign, np.ones((1, 6), bool))
    assert result[0]["separately_normalized_weight_sum"]["mean"] == 1
    assert np.isclose(result[0]["N_eff_exp_entropy"]["mean"], 2)
    assert result[0]["N_eff_inverse_Simpson"]["mean"] == 2
    assert result[1]["N_eff_exp_entropy"]["mean"] == 1
    assert result[3]["status"] == "NO_DEFINED_MEMBERS"
    assert result[3]["measured_count"] is None
    assert gene_statistics(None, ep, sign) is None


def test_gene_missing_mask_and_weight_sum_fail_closed():
    readout, ep, sign = signed_example()
    mask = np.ones((1, 6), bool); mask[0, 0] = False
    with pytest.raises(ValueError, match="UNMEASURED_GENE"):
        gene_statistics(readout, ep, sign, mask)
    readout["weights"][0, :2] = .4
    with pytest.raises(ValueError, match="NORMALIZATION"):
        gene_statistics(readout, ep, sign)


def test_zero_offset_scale_is_not_artificially_inflated():
    assert ratio(.1, 0) is None
    assert ratio(.1, 2) == .05


def test_lossless_arm_shards_roundtrip_tamper_and_relative_path_guard(tmp_path):
    rows = [dict(arm=arm, partition=part, seed=seed, controls={"matched_query": {
        "null": {"n": 8, "mean": .25, "sd": .1}, "absent_direction": None}},
        original_order_value=[seed, part, None])
        for arm in reversed(ARMS) for seed in reversed(SEEDS) for part in reversed(PARTITIONS)]
    result = dict(schema="bgb_en_r6_attention_diagnostics_v1", contract_sha256="a" * 64,
                  script_sha256="b" * 64, metadata_with_null=None, rows=rows)
    index = write_diagnostics(result, tmp_path)
    index_path = tmp_path / INDEX_NAME
    assert len(index["rows_shards"]) == 10
    assert index["logical_object_sha256"] == canonical(result)
    assert all(part["row_count"] == 9 for part in index["rows_shards"])
    assert all(path.stat().st_size <= MAX_REPORT_BYTES for path in tmp_path.iterdir())
    assert read_diagnostics(index_path) == load_diagnostics(index_path) == result
    with pytest.raises(ValueError, match="NO_OVERWRITE"):
        write_diagnostics(result, tmp_path)
    shard = tmp_path / index["rows_shards"][0]["file"]
    original = shard.read_bytes()
    shard.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="SHA_OR_PATH"):
        read_diagnostics(index_path)
    shard.write_bytes(original)
    changed = json.loads(index_path.read_text())
    changed["rows_shards"][0]["file"] = "../outside.json"
    index_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="PATH_OR_COUNT"):
        read_diagnostics(index_path)


def test_parameter_drift_uses_all_named_trainable_groups_and_isolated_l2_only():
    initial = {"core.query_condition.weight": np.array([3., 4.]),
               "core.query_condition.bias": np.array([0.]),
               "core.spatial_adapter.weight": np.array([0.]),
               "hallmark.projection.weight": np.array([1.]),
               "core.foundation.frozen": np.array([100.])}
    final = {"core.query_condition.weight": np.array([6., 8.]),
             "core.query_condition.bias": np.array([0.]),
             "core.spatial_adapter.weight": np.array([2.]),
             "hallmark.projection.weight": np.array([4.]),
             "core.foundation.frozen": np.array([100.])}
    names = [name for name in initial if name != "core.foundation.frozen"]
    result = parameter_drift(initial, final, names)
    query = result["core.query_condition"]
    assert set(result) == {"core.query_condition", "core.spatial_adapter", "hallmark.projection"}
    assert query["trainable_numel"] == 3 and query["parameter_tensors"] == 2
    assert query["initial_L2"] == 5 and query["final_L2"] == 10
    assert query["drift_L2"] == 5 and query["relative_drift"] == 1
    assert query["isolated_parameter_L2_gradient_L2"] == .002
    assert result["core.spatial_adapter"]["relative_drift"] is None
    assert result["core.spatial_adapter"]["drift_L2"] == 2


def test_activation_norm_masks_zero_denominators_and_equal_wsi_scope():
    mask = np.array([[True, False, True, False]])
    vector = np.array([[[3., 4.], [np.nan, np.nan], [0., 0.], [np.nan, np.nan]]])
    delta = np.array([[[3., 4.], [np.nan, np.nan], [1., 0.], [np.nan, np.nan]]])
    patch_mask = np.array([[[True, True], [True, False]]])
    patch_values = np.array([[[[3., 4.], [3., 4.]], [[1., 0.], [np.nan, np.nan]]]])
    out = dict(own_program_mask=mask, query_source_mask=mask, G0=vector,
        query_context_used=vector, base_queries=vector, conditioned_queries=vector + delta,
        Ebase=vector, delta_E=np.zeros_like(vector), delta_G=delta,
        patch_values=patch_values, patch_base_read={"program_mask": mask}, patch_cond_read={"program_mask": mask},
        reverse_base_read={"tokens": vector, "program_mask": mask}, reverse_cond_read={"program_mask": mask},
        diagnostics={"reverse_kind": "gene"})
    result = activation_norms(out, patch_mask, forward_enabled=True)
    assert result["G0_L2"]["valid_items"] == 2
    assert result["G0_L2"]["patient_equal_program_mean"]["mean"] == 2.5
    query = result["base_query_and_condition_delta"]
    assert query["valid_zero_base_denominator_items"] == 1
    assert query["delta_over_base_L2"]["valid_items"] == 1
    assert query["delta_over_base_L2"]["patient_equal_program_mean"]["mean"] == 1
    assert result["patch_values_L2"]["patient_equal_WSI_patch_mean"]["mean"] == 3
    assert result["reverse_base_and_delta_G"]["delta_over_base_L2"]["patient_equal_program_mean"]["mean"] == 1
    assert result["reverse_difference_reference"].endswith("-Read(Ebase,own_original_gene_values)")
    out["diagnostics"]["reverse_kind"] = "none"
    out["own_program_mask"] = out["query_source_mask"] = np.zeros_like(mask)
    out["patch_base_read"]["program_mask"] = out["patch_cond_read"]["program_mask"] = np.zeros_like(mask)
    empty = activation_norms(out, np.zeros_like(patch_mask), forward_enabled=False)
    assert empty["reverse_base_and_delta_G"] is None and empty["reverse_difference_reference"] is None
    assert empty["G0_L2"]["patient_equal_program_mean"]["mean"] is None
    assert empty["base_query_and_condition_delta"]["delta_over_base_L2"]["patient_equal_program_mean"]["mean"] is None
    assert empty["patch_values_L2"]["patient_equal_WSI_patch_mean"]["mean"] is None
