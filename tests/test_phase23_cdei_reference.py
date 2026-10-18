"""Train-only reference contract; synthetic tensors only, zero model fitting."""
import copy
import json

import numpy as np
import pytest
import torch

from src.phase23_cdei.reference import FrozenPatientReferenceBank


def example(*, dtype=torch.float32):
    tokens = torch.arange(6 * 24 * 32, dtype=dtype).reshape(6, 24, 32) / 137
    measured = torch.ones(6, 24, dtype=torch.bool)
    clinical = torch.arange(18, dtype=dtype).reshape(6, 3) / 7
    return tokens, measured, clinical, torch.tensor([0, 2, 4])


def test_heldout_poison_is_never_inspected_or_copied():
    tokens, mask, clinical, fit = example()
    before = FrozenPatientReferenceBank(tokens, mask, clinical, fit)
    t, m, c = tokens.clone(), mask.clone(), clinical.clone()
    heldout = torch.tensor([1, 3, 5])
    t[heldout] = torch.nan
    c[heldout] = torch.inf
    m[heldout] = False
    after = FrozenPatientReferenceBank(t, m, c, fit)
    assert before.public_summary() == after.public_summary()
    for x, y in zip(before.query_reference_mean(range(6)),
                    after.query_reference_mean(range(6))):
        assert torch.equal(x, y)
    assert after._encoded_tokens.shape == (3, 24, 32)
    assert after._clinical.shape == (3, 3)


def test_fit_query_excludes_self_and_heldout_uses_all_fit_donors():
    tokens, mask, clinical, fit = example()
    bank = FrozenPatientReferenceBank(tokens, mask, clinical, fit)
    query = bank.query_reference_mean([2, 3, 2])
    torch.testing.assert_close(query.tokens[0], tokens[[0, 4]].double().mean(0).float())
    torch.testing.assert_close(query.tokens[1], tokens[fit].double().mean(0).float())
    assert torch.equal(query.tokens[0], query.tokens[2])
    assert torch.equal(query.counts[:, 0], torch.tensor([2, 3, 2]))
    assert query.mask.all()
    expected_c = clinical[fit].double().mean(0).float().expand(3, -1)
    assert torch.equal(query.clinical_mean, expected_c)


def test_patient_weighting_does_not_depend_on_query_repetition():
    tokens, mask, clinical, fit = example()
    tokens[0] = 1
    tokens[2] = 4
    tokens[4] = 10
    bank = FrozenPatientReferenceBank(tokens, mask, clinical, fit)
    singleton = bank.query_reference_mean([1])
    repeated = bank.query_reference_mean([1, 1, 1, 1])
    assert torch.equal(singleton.tokens[0], torch.full((24, 32), 5.))
    assert torch.equal(repeated.tokens, singleton.tokens.expand(4, -1, -1))


def test_slot_specific_missingness_keeps_mask_counts_and_no_pseudo_measurement():
    tokens, mask, clinical, fit = example()
    mask[fit, 0] = False
    mask[[2, 4], 1] = False
    tokens[~mask] = torch.nan
    bank = FrozenPatientReferenceBank(tokens, mask, clinical, fit)
    result = bank.query_reference_mean([0, 1, 2])
    assert torch.equal(result.counts[:, 0], torch.zeros(3, dtype=torch.int64))
    assert not result.mask[:, 0].any()
    assert torch.equal(result.tokens[:, 0], torch.zeros(3, 32))
    assert result.counts[0, 1] == 0
    assert not result.mask[0, 1]
    assert result.counts[1, 1] == 1
    assert result.counts[2, 1] == 1
    assert torch.equal(result.tokens[1, 1], tokens[0, 1])
    assert torch.isfinite(result.tokens).all()


def test_one_fit_patient_has_no_training_donor_but_valid_heldout_reference():
    tokens, mask, clinical, _ = example()
    bank = FrozenPatientReferenceBank(tokens, mask, clinical, [2])
    result = bank.query_reference_mean([2, 1])
    assert not result.mask[0].any()
    assert not result.counts[0].any()
    assert not result.tokens[0].any()
    assert torch.equal(result.tokens[1], tokens[2])


def test_matched_wrong_union_references_are_identical_with_donor_missingness():
    tokens, mask, clinical, fit = example()
    mask[0, 0] = False
    mask[2, 1] = False
    mask[4, 2] = False
    tokens[~mask] = torch.nan
    bank = FrozenPatientReferenceBank(tokens, mask, clinical, fit)
    own = [0, 1, 2, 5]
    wrong = [2, 2, 4, 0]
    matched = bank.query_reference_mean(own, np.array(wrong)[:, None])
    mismatched = bank.query_reference_mean(wrong, np.array(own)[:, None])
    for a, b in zip(matched, mismatched):
        assert torch.equal(a, b)


def test_union_exclusions_remove_self_and_extra_at_most_once():
    tokens, mask, clinical, fit = example()
    bank = FrozenPatientReferenceBank(tokens, mask, clinical, fit)
    result = bank.query_reference_mean([0, 1], [[0, 2, 2, 3], [2, 2, 3, 3]])
    assert torch.equal(result.tokens[0], tokens[4])
    assert torch.equal(result.counts[0], torch.ones(24, dtype=torch.int64))
    torch.testing.assert_close(result.tokens[1], tokens[[0, 4]].double().mean(0).float())
    assert torch.equal(result.counts[1], torch.full((24,), 2, dtype=torch.int64))
    all_excluded = bank.query_reference_mean([1], [[0, 2, 4]])
    assert not all_excluded.mask.any()
    assert not all_excluded.counts.any()
    assert not all_excluded.tokens.any()


def test_empty_extra_exclusions_keep_original_leave_self_out():
    bank = FrozenPatientReferenceBank(*example())
    expected = bank.query_reference_mean([0, 1])
    for extra in ([], np.empty((2, 0), dtype=np.int64), torch.empty((2, 0), dtype=torch.int64)):
        for a, b in zip(expected, bank.query_reference_mean([0, 1], extra)):
            assert torch.equal(a, b)


@pytest.mark.parametrize("extra,pattern", [
    ([[6]], "unknown"), ([[-1]], "unknown"), ([[1.0]], "integer"),
    ([[True]], "integer"), ([1], "shape"), ([[1], [2]], "shape"),
])
def test_invalid_additional_exclusion_ids_or_shape_fail(extra, pattern):
    bank = FrozenPatientReferenceBank(*example())
    with pytest.raises(ValueError, match=pattern):
        bank.query_reference_mean([0], extra)


@pytest.mark.parametrize("poison", [torch.nan, torch.inf, -torch.inf])
def test_nonfinite_measured_fit_token_rejected(poison):
    tokens, mask, clinical, fit = example()
    tokens[2, 0, 0] = poison
    with pytest.raises(ValueError, match="measured fit tokens"):
        FrozenPatientReferenceBank(tokens, mask, clinical, fit)


def test_nonfinite_fit_clinical_rejected():
    tokens, mask, clinical, fit = example()
    clinical[2, 0] = torch.nan
    with pytest.raises(ValueError, match="fit clinical"):
        FrozenPatientReferenceBank(tokens, mask, clinical, fit)


@pytest.mark.parametrize("fit,pattern", [
    ([], "nonempty"), ([0, 0], "unique"), ([0, -1], "out-of-bounds"),
    ([0, 6], "out-of-bounds"), ([0., 2.], "integer"),
    ([True, False], "integer"), ([[0, 2]], "one-dimensional"),
])
def test_bad_fit_contract_fails(fit, pattern):
    tokens, mask, clinical, _ = example()
    with pytest.raises(ValueError, match=pattern):
        FrozenPatientReferenceBank(tokens, mask, clinical, fit)


def test_independent_donor_contract_must_equal_current_fit_set():
    tokens, mask, clinical, fit = example()
    bank = FrozenPatientReferenceBank(tokens, mask, clinical, fit,
                                      donor_indices=[4, 0, 2])
    assert bank.public_summary()["fit_patient_count"] == 3
    for donors in ([0, 2], [0, 2, 3], [0, 2, 4, 5]):
        with pytest.raises(ValueError, match="donor set"):
            FrozenPatientReferenceBank(tokens, mask, clinical, fit,
                                       donor_indices=donors)
    with pytest.raises(ValueError, match="unique"):
        FrozenPatientReferenceBank(tokens, mask, clinical, fit,
                                   donor_indices=[0, 2, 2])


@pytest.mark.parametrize("query,pattern", [
    ([-1], "unknown"), ([6], "unknown"), ([1.0], "integer"),
    ([True], "integer"), ([[1]], "one-dimensional"),
])
def test_unknown_or_noninteger_query_fails(query, pattern):
    bank = FrozenPatientReferenceBank(*example())
    with pytest.raises(ValueError, match=pattern):
        bank.query_reference_mean(query)


def test_empty_queries_return_empty_well_shaped_outputs():
    bank = FrozenPatientReferenceBank(*example())
    result = bank.query_reference_mean([])
    assert result.tokens.shape == (0, 24, 32)
    assert result.mask.shape == result.counts.shape == (0, 24)
    assert result.clinical_mean.shape == (0, 3)


@pytest.mark.parametrize("alter", ["tokens_shape", "tokens_int", "mask_shape",
                                  "mask_int", "clinical_shape", "clinical_int"])
def test_tensor_shape_and_dtype_contract(alter):
    tokens, mask, clinical, fit = example()
    if alter == "tokens_shape": tokens = tokens[:, :23]
    elif alter == "tokens_int": tokens = tokens.long()
    elif alter == "mask_shape": mask = mask[:, :23]
    elif alter == "mask_int": mask = mask.long()
    elif alter == "clinical_shape": clinical = clinical[:5]
    elif alter == "clinical_int": clinical = clinical.long()
    with pytest.raises(ValueError):
        FrozenPatientReferenceBank(tokens, mask, clinical, fit)


def test_inputs_outputs_and_state_snapshot_cannot_alias_frozen_bank():
    tokens, mask, clinical, fit = example()
    bank = FrozenPatientReferenceBank(tokens, mask, clinical, fit)
    expected = bank.query_reference_mean([0, 1])
    tokens.fill_(700)
    mask.fill_(False)
    clinical.fill_(900)
    fit.fill_(1)
    output = bank.query_reference_mean([0, 1])
    output.tokens.fill_(999)
    output.mask.fill_(False)
    output.counts.fill_(77)
    output.clinical_mean.fill_(888)
    snapshot = bank.state_dict()
    snapshot["_encoded_tokens"].zero_()
    for x, y in zip(expected, bank.query_reference_mean([0, 1])):
        assert torch.equal(x, y)


def test_state_is_nontrainable_and_any_private_buffer_tamper_is_detected():
    tokens, mask, clinical, fit = example()
    tokens.requires_grad_()
    clinical.requires_grad_()
    bank = FrozenPatientReferenceBank(tokens, mask, clinical, fit)
    assert list(bank.parameters()) == []
    assert all(not x.requires_grad and x.grad_fn is None for x in bank.buffers())
    assert all(not x.requires_grad for x in bank.query_reference_mean([0, 1]))
    bank._encoded_tokens[0, 0, 0] += 1
    with pytest.raises(ValueError, match="immutable"):
        bank.query_reference_mean([1])


def test_loading_changed_bank_is_rejected_and_original_state_remains_intact():
    bank = FrozenPatientReferenceBank(*example(), provenance={"fold": 1, "role": "inner"})
    expected = bank.public_summary()
    snapshot = copy.deepcopy(bank.state_dict())
    bank.load_state_dict(snapshot, strict=True)
    snapshot["_encoded_tokens"][0, 0, 0] += 1
    with pytest.raises(RuntimeError, match="immutable reference bank state mismatch"):
        bank.load_state_dict(snapshot, strict=True)
    assert bank.public_summary() == expected


def test_public_summary_has_counts_and_hashes_only_and_no_tensor_values():
    bank = FrozenPatientReferenceBank(*example(), provenance={"role": "inner", "fold": 1})
    summary = bank.public_summary()
    assert summary["fit_patient_count"] == 3
    assert summary["observed_slots"] == 24
    assert summary["measured_patient_slot_count"] == 72
    assert all(isinstance(v, (int, str)) for v in summary.values())
    assert all(len(v) == 64 for k, v in summary.items() if k.endswith("sha256"))
    assert not any(key in summary for key in ("fit_indices", "tokens", "clinical", "mask"))
    json.dumps(summary, allow_nan=False)


def test_provenance_property_returns_fresh_nested_metadata_and_is_read_only():
    metadata = {"kernel_sha256": "a" * 64, "fold": 1,
                "nested": {"role": "inner", "flags": [True, False]}}
    expected = copy.deepcopy(metadata)
    bank = FrozenPatientReferenceBank(*example(), provenance=metadata)
    before = bank.public_summary()
    metadata["kernel_sha256"] = "b" * 64
    metadata["nested"]["flags"].append(False)
    returned = bank.provenance
    assert returned == expected
    returned["kernel_sha256"] = "c" * 64
    returned["nested"]["flags"].clear()
    assert bank.provenance == expected
    assert bank.public_summary() == before
    assert "kernel_sha256" not in bank.public_summary()
    with pytest.raises(AttributeError):
        bank.provenance = {"kernel_sha256": "d" * 64}


@pytest.mark.parametrize("field", ["_provenance_json", "_provenance_sha256", "_cohort_size"])
def test_private_metadata_tampering_rejected_before_any_query_or_getter(field):
    bank = FrozenPatientReferenceBank(*example(), provenance={"kernel_sha256": "a" * 64})
    setattr(bank, field, "tampered" if field != "_cohort_size" else 999)
    for action in (lambda: bank.provenance, lambda: bank.query_reference_mean([1]),
                   bank.public_summary):
        with pytest.raises(ValueError, match="immutable reference metadata"):
            action()


def test_state_provenance_tamper_is_rejected_without_changing_metadata_or_buffers():
    bank = FrozenPatientReferenceBank(*example(), provenance={"kernel_sha256": "a" * 64})
    before = bank.public_summary()
    metadata = bank.provenance
    snapshot = bank.state_dict()
    snapshot["_extra_state"]["provenance_json"] = '{"kernel_sha256":"tampered"}'
    with pytest.raises(RuntimeError, match="immutable reference bank state mismatch"):
        bank.load_state_dict(snapshot, strict=True)
    assert bank.public_summary() == before
    assert bank.provenance == metadata


def test_numpy_integer_inputs_supported_without_coercing_float_indices():
    tokens, mask, clinical, fit = example(dtype=torch.float64)
    bank = FrozenPatientReferenceBank(tokens.numpy(), mask.numpy(), clinical.numpy(),
                                      fit.numpy().astype(np.int32))
    output = bank.query_reference_mean(np.array([1, 2], np.int64))
    assert output.tokens.dtype == torch.float64
    assert output.counts.dtype == torch.int64


def test_cp_reference_mean_equals_explicit_patient_mean_of_frozen_kernel_scores():
    from src.phase19_attention.attention import RankConditionedAttention
    torch.manual_seed(23)
    tokens, mask, clinical, fit = example(dtype=torch.float64)
    mask[0, 0] = False
    mask[2, 1] = False
    bank = FrozenPatientReferenceBank(tokens, mask, clinical, fit)
    kernel = RankConditionedAttention(32, 32, 3, d=32, rank=4).double().eval()
    for parameter in kernel.parameters(): parameter.requires_grad_(False)
    query = torch.tensor([0, 3])
    patches = torch.randn(2, 5, 32, dtype=torch.float64)
    reference = bank.query_reference_mean(query)
    actual = kernel(patches, reference.tokens, clinical[query],
                    concept_mask=reference.mask)["scores"]
    for row, patient in enumerate(query):
        donors = fit[fit != patient]
        scores = kernel(patches[row:row+1].expand(len(donors), -1, -1),
                        tokens[donors], clinical[patient:patient+1].expand(len(donors), -1),
                        concept_mask=mask[donors])["scores"]
        count = mask[donors].sum(0)
        expected = (scores * mask[donors, None, :]).sum(0) / count.clamp_min(1)[None, :]
        expected = torch.where((count > 0)[None, :], expected, torch.zeros_like(expected))
        torch.testing.assert_close(actual[row], expected, rtol=1e-12, atol=1e-12)
