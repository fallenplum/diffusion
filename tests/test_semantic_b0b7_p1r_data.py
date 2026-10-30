import copy
import numpy as np
import pytest
import torch

from src.semantic_b0b7.p1r_data import build_partition_inputs, require_no_outcomes


def fixture_inputs():
    n, g = 12, 6
    rng = np.random.default_rng(81)
    raw = {"folds": np.tile([1, 2, 3], 4), "rna": rng.normal(size=(n, g)),
           "cna": rng.normal(size=(n, g)), "rna_measured": np.ones((n, g), bool),
           "cna_measured": np.ones((n, g), bool), "cancer": np.array(["COAD"] * n),
           "platform": np.zeros(n, int), "clinical_raw": {
               "AGE": np.array([str(40 + i) for i in range(n)]),
               **{k: np.array(["I"] * n) for k in ("SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE")}},
           "audit": {"cohort_sha256": "synthetic", "endpoint_read": False}}
    image = {"patch_values": torch.randn(n, 3, 4), "patch_aligned": torch.randn(n, 3, 4),
             "patch_mask": torch.ones(n, 3, dtype=torch.bool), "patch_slide": torch.zeros(n, 3, dtype=torch.long),
             "titan_native": torch.randn(n, 4), "titan_aligned": torch.randn(n, 4), "titan_scores": torch.randn(n, 2)}
    names = [str(i) for i in range(50)]
    frozen = {"image": image, "global_rows": torch.arange(n), "program_names": names,
              **{k: torch.ones(2, 4) for k in ("conch_prototypes", "titan_prototypes")},
              **{k: torch.ones(2, 50) for k in ("relation_prior", "shuffled_relation_prior")}}
    return raw, frozen, (names, np.ones((50, g), bool), np.ones((50, g), bool))


def test_phase_a_refuses_endpoints_even_if_unused():
    for key in ("t", "e", "DSS", "pfi_t", "offset"):
        with pytest.raises(ValueError, match="ENDPOINT"):
            require_no_outcomes({key: np.zeros(3)})
    with pytest.raises(ValueError, match="ENDPOINT"):
        require_no_outcomes({"audit": {"endpoint_read": True}})


def test_validation_changes_cannot_change_training_statistics_or_tensors():
    raw, frozen, membership = fixture_inputs()
    fit, ev = np.arange(8), np.arange(8, 12)
    one, state_one = build_partition_inputs(raw, frozen, fit, ev, membership, split_name="unit")
    changed = copy.deepcopy(raw)
    changed["rna"][ev] = 100000
    changed["cna"][ev] = -100000
    changed["clinical_raw"]["AGE"][ev] = "999"
    two, state_two = build_partition_inputs(changed, frozen, fit, ev, membership, split_name="unit")
    assert torch.equal(one["omics"]["values"][:8], two["omics"]["values"][:8])
    assert torch.equal(one["image"]["clinical"][:8], two["image"]["clinical"][:8])
    assert np.array_equal(state_one["program_transform"].rna_scale.mean,
                          state_two["program_transform"].rna_scale.mean)
    assert np.array_equal(state_one["clinical_state"]["mean"], state_two["clinical_state"]["mean"])
    assert not torch.equal(one["omics"]["values"][8:], two["omics"]["values"][8:])


def test_incomplete_frozen_scope_fails_closed_and_no_overlap():
    raw, frozen, membership = fixture_inputs()
    with pytest.raises(ValueError, match="OVERLAP"):
        build_partition_inputs(raw, frozen, np.arange(8), np.arange(7, 12), membership, split_name="unit")
    frozen["global_rows"] = torch.arange(10)
    with pytest.raises(ValueError, match="INSUFFICIENT"):
        build_partition_inputs(raw, frozen, np.arange(8), np.arange(8, 12), membership, split_name="unit")
