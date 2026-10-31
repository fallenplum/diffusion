"""Synthetic saved-risk/contract counterexamples; no historical or outer data."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import semantic_b0b7_p2_r4_history_audit as audit


def contract_fixture():
    fit, evaluation = np.asarray([0, 1, 2]), np.asarray([3, 4])
    master = {"assets": {"cohort_sha256":"a"*64,"endpoints_sha256":"b"*64,
                         "visual_sources_sha256":"c"*64,"GMT_sha256":"d"*64,"legacy_groups_sha256":"e"*64},
              "B0_R4_EXACT":{"full_CONCH_mean_sha256":"f"*64}}
    modules = {"protocol.py":"1"*64,"baselines.py":"2"*64}
    lineage = {"cohort_sha256":"a"*64,"endpoints_sha256":"b"*64,
               "GMT_sha256":"d"*64,"legacy_group_sha256":"e"*64}
    contract = {"cohort_sha256":"a"*64,"endpoints_sha256":"b"*64,"visual_sources_sha256":"c"*64,
        "full_CONCH_mean_sha256":"f"*64,"source_sha256":audit.HISTORICAL_PRODUCER_SHA,
        "train_sha256":audit.old_array_hash(fit),"test_sha256":audit.old_array_hash(evaluation),
        "module_sha256":modules,"source_lineage":lineage,"source_lineage_sha256":audit.lineage_hash(lineage)}
    return contract, master, modules, fit, evaluation


@pytest.mark.parametrize("field", ["cohort_sha256","endpoints_sha256","visual_sources_sha256",
                                    "full_CONCH_mean_sha256","source_sha256","train_sha256","test_sha256"])
def test_historical_source_and_row_hash_mismatches_are_rejected(field):
    contract, master, modules, fit, evaluation = contract_fixture()
    audit.validate_contract(contract, master, modules, fit, evaluation)
    changed = copy.deepcopy(contract); changed[field] = "0"*64
    with pytest.raises(ValueError, match="CONTRACT_BINDING_CHANGED"):
        audit.validate_contract(changed, master, modules, fit, evaluation)


def test_lineage_and_direct_module_changes_cannot_be_silently_treated_as_same_recipe():
    contract, master, modules, fit, evaluation = contract_fixture()
    changed = copy.deepcopy(contract); changed["module_sha256"]["protocol.py"] = "0"*64
    with pytest.raises(ValueError, match="MODULE_SOURCES_CHANGED"):
        audit.validate_contract(changed, master, modules, fit, evaluation)
    changed = copy.deepcopy(contract); changed["source_lineage"]["legacy_group_sha256"] = "0"*64
    changed["source_lineage_sha256"] = audit.lineage_hash(changed["source_lineage"])
    with pytest.raises(ValueError, match="GENE_MEMBERSHIP_CHANGED"):
        audit.validate_contract(changed, master, modules, fit, evaluation)


def test_vector_difference_is_reported_without_recalibration_or_hard_target_matching():
    before = np.asarray([.2, -.1, .3]); after = before.copy()
    equal = audit.vector_comparison(before, after, .2, .2)
    assert equal["risk_values_exact"] and equal["beta_exact"] and equal["max_abs_risk_difference"] == 0.
    assert "historical_beta" not in equal and "current_beta" not in equal
    assert equal["historical_beta_sha256"] == equal["current_beta_sha256"] == audit.fingerprint(.2)
    after[0] += .01
    different = audit.vector_comparison(before, after, .2, .21)
    assert not different["risk_values_exact"] and not different["beta_exact"]
    assert different["historical_beta_sha256"] != different["current_beta_sha256"]
    assert different["max_abs_risk_difference"] == pytest.approx(.01)
    np.testing.assert_array_equal(before, [.2, -.1, .3])
    with pytest.raises(ValueError, match="RISK_SHAPE_OR_FINITE"):
        audit.vector_comparison(before, np.asarray([1., 2.]), .2, .2)


def test_historical_and_current_rows_must_both_match_canonical_order():
    _, _, _, fit, evaluation = contract_fixture()
    historical = {"train":fit.copy(),"test":evaluation.copy()}
    anchor = SimpleNamespace(train_rows=fit.copy(), evaluation_rows=evaluation.copy())
    audit.check_row_order(historical, anchor, fit, evaluation)
    historical["test"] = historical["test"][::-1]
    with pytest.raises(ValueError, match="CANONICAL_ROW_ORDER_CHANGED"):
        audit.check_row_order(historical, anchor, fit, evaluation)


def test_single_seed_micro_uses_fold_pair_denominators_and_preserves_missing_support():
    rows=[]
    for fold in range(1, 6):
        den = 1. if fold == 1 else 100.
        value = 1. if fold == 1 else .5
        rows.append({"fold":fold,"metric":{"Harrell_num":value*den,"Harrell_den":den,
                                           "Uno_num":value*(den+10),"Uno_den":den+10}})
    result = audit.micro(rows,"metric")
    assert result["Harrell"] == pytest.approx(201/401)
    assert result["Harrell"] != pytest.approx(np.mean([1.,.5,.5,.5,.5]))
    assert result["independent_refit_seeds"] == 1 and not result["cross_fold_risk_pairs"]
    rows[-1]["metric"].update(Uno_num=None,Uno_den=None)
    assert audit.micro(rows,"metric")["Uno60"] is None
    with pytest.raises(ValueError, match="FIVE_FOLDS_REQUIRED"):
        audit.micro(rows[:-1],"metric")


def test_own_torch_artifact_is_checksum_qualified_before_decoding(tmp_path):
    import torch
    value={"contract":{"fixture":True},"teachers":{31:{"test_CP_SCALAR_risk":np.asarray([.1,.2]),"beta":.2}},
           "results":{"already_computed_PFI_aggregate_not_a_label_array":{"score":.7}}}
    path=tmp_path/"models.pt"; torch.save(value,path)
    complete={"status":"COMPLETE_OUTER_BASELINES","selection_used_outer_test":False,
              "artifact_sha256":audit.sha(path),"contract":value["contract"]}
    (tmp_path/"COMPLETE.json").write_text(json.dumps(complete),encoding="utf-8")
    loaded,_=audit.verified_historical_load(tmp_path)
    np.testing.assert_array_equal(loaded["teachers"][31]["test_CP_SCALAR_risk"],[.1,.2])
    with path.open("ab") as stream: stream.write(b"tamper")
    with pytest.raises(ValueError,match="NOT_HASH_QUALIFIED"):
        audit.verified_historical_load(tmp_path)


def test_absent_postformal_authority_prevents_any_file_hash_or_input_access(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError("Input touched before authority")
    monkeypatch.setattr(audit,"sha",forbidden)
    with pytest.raises(ValueError,match="BEFORE_INPUT_READ"):
        audit.run_audit("unread","unread",assets_path="unread",legacy_root="unread",historical_root="unread",
                        formal_private="unread",output="unread")
