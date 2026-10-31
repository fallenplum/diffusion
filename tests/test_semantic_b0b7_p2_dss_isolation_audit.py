"""Software counterexamples only; no TCGA arrays, neural fitting or outer reads."""
import copy
from dataclasses import dataclass
import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from scripts import semantic_b0b7_p2_dss_isolation_audit as audit


def test_joint_heldout_label_permutation_preserves_training_and_every_covariate():
    raw = {"t": np.asarray([1., 2., 3., 4., 5., 6.]), "e": np.asarray([1, 0, 1, 0, 1, 0]),
           "rna": np.arange(18.).reshape(6, 3), "clinical_raw": {"AGE": np.arange(6).astype(str)}}
    original = copy.deepcopy(raw)
    fit, held = np.asarray([0, 1, 2]), np.asarray([3, 4, 5])
    changed, receipt = audit.permute_heldout_dss(raw, fit, held, 27132)
    np.testing.assert_array_equal(changed["t"][fit], raw["t"][fit])
    np.testing.assert_array_equal(changed["e"][fit], raw["e"][fit])
    assert sorted(zip(changed["t"][held], changed["e"][held])) == sorted(zip(raw["t"][held], raw["e"][held]))
    assert not np.array_equal(changed["t"][held], raw["t"][held])
    assert audit.fingerprint(raw) == audit.fingerprint(original)
    assert receipt["training_label_values_changed"] == 0 and not receipt["covariates_changed"]
    with pytest.raises(ValueError, match="OVERLAP"):
        audit.permute_heldout_dss(raw, fit, fit, 27132)


def test_bitwise_state_fingerprint_checks_dtype_signed_zero_dataclass_and_ridge_state():
    @dataclass
    class State:
        mean: np.ndarray
    model = Ridge(alpha=1.)
    model.coef_ = np.asarray([.2, -.3]); model.intercept_ = 0.
    state = {"PCA": State(np.asarray([0., 1.])), "background": model}
    same = copy.deepcopy(state)
    assert audit.fingerprint(state) == audit.fingerprint(same)
    same["PCA"].mean[0] = -0.
    assert audit.fingerprint(state) != audit.fingerprint(same)
    same = copy.deepcopy(state); same["background"].coef_[0] += .01
    assert audit.fingerprint(state) != audit.fingerprint(same)
    assert audit.fingerprint(np.asarray([1.], np.float32)) != audit.fingerprint(np.asarray([1.], np.float64))


@pytest.mark.parametrize("component", ["teacher", "background", "PCA", "coefficient", "train_prediction", "eval_prediction"])
def test_any_learned_or_predictive_component_change_is_rejected(component):
    before = {name: np.asarray([.1, .2]) for name in
              ("teacher", "background", "PCA", "coefficient", "train_prediction", "eval_prediction")}
    after = copy.deepcopy(before)
    assert all(row["exact"] for row in audit.compare_components(before, after).values())
    after[component][0] += 1e-12
    with pytest.raises(ValueError, match="CHANGED_LEARNED_PIPELINE:"+component):
        audit.compare_components(before, after)


def test_absent_postformal_authority_prevents_all_input_and_source_reads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Input was read before explicit authority")
    monkeypatch.setattr(audit, "validate_lock", forbidden)
    with pytest.raises(ValueError, match="BEFORE_INPUT_READ"):
        audit.run_qualification("unread", "unread", "unread", assets_path="unread", legacy_root="unread",
            frozen_images_path="unread", formal_private="unread", private_output="unread", public_output="unread")


def locked_configuration():
    master = audit.read(audit.ROOT/"configs/semantic_b0b7/p2_lock.json")
    return {"schema":"semantic_b0b7_p2_dss_pipeline_isolation_lock_v1",
        "status":"LOCKED_BEFORE_ADDITIONAL_CPU_QUALIFICATION_FITS", "P2_master_lock_sha256":audit.MASTER_SHA,
        "formal_scientific_source_set_sha256":audit.qualification.canonical(master["frozen_sources"]),
        "outer":1,"family":"MAIN","arm":"B5","seed":31,"fit_patients":409,"evaluation_patients":103,
        "heldout_pair_permutation_seed":27132,"planned_cost":copy.deepcopy(audit.COST),
        "frozen_lambda_source":"existing_ALL_OUTER_SELECTIONS_outer1_B5_seed31_no_reselection",
        "performance_evidence_or_new_primary_cell":False,"does_not_replace_original_two_by_50_Phase_A_F1":True,
        "existing_F1_helper_source_sha256":audit.sha(audit.qualification.__file__),
        "audit_sources":{name:audit.sha(audit.ROOT/name) for name in (audit.SCRIPT,audit.TEST)}}


def test_locked_cpu_budget_and_source_closure_are_checked_without_data_access(tmp_path):
    lock = locked_configuration(); path = tmp_path/"synthetic_lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    audit.validate_lock(audit.ROOT/"configs/semantic_b0b7/p2_lock.json",path,audit.sha(path))
    lock["planned_cost"]["all_Cox_fits"] += 1
    path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="BUDGET_OR_SELECTION"):
        audit.validate_lock(audit.ROOT/"configs/semantic_b0b7/p2_lock.json",path,audit.sha(path))
