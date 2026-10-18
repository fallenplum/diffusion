"""Synthetic artifacts only: no endpoint files or actual predictor training."""
import json

import numpy as np
import pytest
import torch

from src.phase24_cdei_g1 import evaluation as ev


SOURCE = "a" * 64


def synthetic_cell(root, seed, fold):
    valid = np.array([2 * (fold - 1), 2 * (fold - 1) + 1])
    fit = np.setdiff1d(np.arange(10), valid)
    rng = np.random.default_rng(seed * 10 + fold)
    record = {"source": SOURCE, "role": "inner", "seed": seed, "fold": fold,
              "fit": fit, "valid": valid, "anchor_fit": np.zeros(len(fit)),
              "anchor_valid": np.zeros(len(valid)), "J_CS_valid": np.zeros(len(valid)), "arms": {}}
    for arm in ev.ARMS:
        arm_record = {"risk": rng.normal(size=len(valid)), "fit_risk": rng.normal(size=len(fit)),
            "model_state": {"weights": torch.ones(2)}, "clinical_dim": 2, "membership": torch.eye(2),
            "parameter_counts": {"total": 10, "backbone": 4},
            "training_audit": {"actual_epochs": 80, "optimizer_steps": 80,
                "fixed_final_epoch_no_validation_selection": True, "heldout_endpoint_reads": 0,
                "anchor_unchanged": True},
            "fit_only_reference_audit": {"raw_fit_patient_count": len(fit), "patient_equal": True,
                "fit_query_leave_self_out": True, "current_encoder_current_kernel": True, "no_heldout_reference": True},
            "history": [{"epoch": epoch, "training_loss": 1.1, "fit_loss": 1., "grad_norm": .1,
                         "fit_correction_SD": .1, "epoch_seconds": 1., "gradients": {"kernel": .1}}
                        for epoch in range(1, 81)]}
        if arm == "LOCAL":
            arm_record["sensitivities"] = {name: rng.normal(size=len(valid)) for name in ev.SENSITIVITIES}
        record["arms"][arm] = arm_record
    folder = root / "inner" / f"seed{seed}_fold{fold}"
    folder.mkdir(parents=True)
    torch.save(record, folder / "checkpoint.pt")
    ev.write_json(folder / "DONE_PRIVATE.json", {"status": "COMPLETE", "source": SOURCE,
        "role": "inner", "seed": seed, "fold": fold, "arms": 4, "epochs": 80,
        "checkpoint_sha256": ev.sha256_file(folder / "checkpoint.pt")})
    return folder


def full_stage(root):
    for seed in ev.SEEDS:
        for fold in ev.FOLDS:
            synthetic_cell(root, seed, fold)


def rewrite(folder, mutate):
    path = folder / "checkpoint.pt"
    record = torch.load(path, weights_only=False)
    mutate(record)
    torch.save(record, path)
    receipt = json.loads((folder / "DONE_PRIVATE.json").read_text())
    receipt["checkpoint_sha256"] = ev.sha256_file(path)
    ev.write_json(folder / "DONE_PRIVATE.json", receipt)


def endpoints():
    return {"DSS": {"time": np.arange(1., 11.), "event": np.ones(10), "valid": np.ones(10, bool)}}


def run(root, loader=endpoints, **kwargs):
    return ev.evaluate_stage(root, root / "public", "inner", SOURCE, loader,
                             cohort_size=10, expected_dss_events=10, **kwargs)


def test_missing_cell_blocks_before_any_endpoint_read(tmp_path):
    calls = []
    with pytest.raises(ValueError, match="missing complete"):
        run(tmp_path, lambda: calls.append("read"))
    assert calls == [] and not (tmp_path / "public").exists()


def test_all_hashes_checked_before_deserializing_any_checkpoint(tmp_path, monkeypatch):
    full_stage(tmp_path)
    (tmp_path / "inner/seed83_fold5/checkpoint.pt").write_bytes(b"tampered")
    loads = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: loads.append(1))
    with pytest.raises(ValueError, match="hash mismatch"):
        run(tmp_path, lambda: pytest.fail("endpoints requested"))
    assert loads == []


@pytest.mark.parametrize("change,match", [
    (lambda c: c["arms"]["REL"]["risk"].__setitem__(0, np.nan), "nonfinite"),
    (lambda c: c["arms"]["POOL"]["history"].pop(), "80 recorded"),
    (lambda c: c["arms"]["REL"]["history"][4].__setitem__("epoch", 7), "epoch sequence"),
    (lambda c: c["arms"]["LOCAL"]["training_audit"].__setitem__("heldout_endpoint_reads", 1), "training audit"),
    (lambda c: c["arms"]["LOCAL"]["fit_only_reference_audit"].__setitem__("current_encoder_current_kernel", False), "reference audit"),
    (lambda c: c["arms"]["LOCAL"]["sensitivities"].pop("OFF"), "four frozen-head"),
    (lambda c: c["arms"]["CTRL"]["parameter_counts"].__setitem__("total", 11), "identical total"),
    (lambda c: c.__setitem__("event", np.ones(10)), "endpoint arrays"),
    (lambda c: c["fit"].__setitem__(0, c["valid"][0]), "leakage"),
    (lambda c: c["arms"]["LOCAL"]["model_state"]["weights"].__setitem__(0, np.inf), "model state contains nonfinite"),
    (lambda c: c["arms"]["POOL"]["history"][0]["gradients"].__setitem__("kernel", np.nan), "gradient diagnostic"),
])
def test_rehashed_invalid_contract_blocks_before_labels(tmp_path, change, match):
    full_stage(tmp_path)
    rewrite(tmp_path / "inner/seed31_fold1", change)
    with pytest.raises(ValueError, match=match):
        run(tmp_path, lambda: pytest.fail("endpoints requested"))


def test_partition_and_frozen_anchor_must_match_across_algorithm_seeds(tmp_path):
    full_stage(tmp_path)
    rewrite(tmp_path / "inner/seed47_fold3", lambda c: c["anchor_valid"].__setitem__(0, .1))
    with pytest.raises(ValueError, match="differs across seeds"):
        run(tmp_path, lambda: pytest.fail("endpoints requested"))


def test_independent_anchor_reconstruction_precedes_endpoint_read(tmp_path):
    full_stage(tmp_path)
    calls = []
    def anchor(role, fold):
        calls.append((role, fold))
        return {"anchor_valid": np.ones(2)}
    with pytest.raises(ValueError, match="differs elementwise"):
        run(tmp_path, lambda: pytest.fail("endpoints requested"), anchor_loader=anchor)
    assert calls == [("inner", 1)]


def test_successful_inner_output_is_anonymous_not_complete_oof(tmp_path):
    full_stage(tmp_path)
    calls = []
    result, gate = run(tmp_path, lambda: calls.append(1) or endpoints())
    assert calls == [1]
    assert result["artifact_audit"]["complete_cells"] == 15
    assert result["artifact_audit"]["end_to_end_model_fits"] == 60
    assert not result["artifact_audit"]["encoder_feature_replay_claimed"]
    assert not result["complete_outer_OOF"] and not result["confirmatory_CI"]
    assert len(result["fold_rows"]) == 15 * 10
    assert len(result["seed_rows"]) == 3 * 10
    assert len(result["repair_damage"]) == 9 * 2
    assert len(result["seed_repair_damage"]) == 9 * 2 * 3
    text = json.dumps(result)
    for forbidden in ("anchor_valid", "fit_risk", "model_state", '"fit":', '"valid":'):
        assert forbidden not in text
    private = json.loads((tmp_path / "INNER_GATE.json").read_text())
    assert private["decision"] in ("GO", "STOP")
    assert private["public_gate_sha256"] == ev.sha256_file(tmp_path / "public/INNER_GATE.json")
    assert not gate["outer_automatically_started"]


def test_seed_aggregation_uses_comparable_pair_masses_not_fold_C_mean(tmp_path):
    full_stage(tmp_path)
    cells, _ = ev.verify_stage(tmp_path, "inner", SOURCE, cohort_size=10)
    for seed in ev.SEEDS:
        for fold in ev.FOLDS:
            cell = cells[seed, fold]
            cell["valid"] = np.arange(2 + fold)
            cell["fit"] = np.arange(7, 10)
            risk = -np.arange(2 + fold, dtype=float) if fold == 1 else np.arange(2 + fold, dtype=float)
            for arm in ev.ARMS:
                cell["arms"][arm]["risk"] = risk
            cell["anchor_valid"] = cell["J_CS_valid"] = risk
            cell["sensitivities"] = {name: risk for name in ev.SENSITIVITIES}
    result = ev.point_results(cells, endpoints()["DSS"], cohort_size=10)
    row = next(row for row in result["seed_rows"] if row["seed"] == 31 and row["arm"] == "LOCAL")
    folds = [row for row in result["fold_rows"] if row["seed"] == 31 and row["arm"] == "LOCAL"]
    assert row["Harrell"] == pytest.approx(sum(q["Harrell_num"] for q in folds) / sum(q["Harrell_den"] for q in folds))
    assert row["Harrell"] != pytest.approx(np.mean([q["Harrell"] for q in folds]))
    transition = ev.inner_pair_transitions(cells, endpoints()["DSS"], result, cohort_size=10)
    assert all(q["net"] == 0 and q["repair"] == 0 and q["damage"] == 0 for q in transition["repair_damage"])


def gate_rows():
    return [{"seed": seed, "arm": arm, "Harrell": .7 if arm == "LOCAL" else .6,
             "Uno60": .7 if arm == "LOCAL" else .6} for seed in ev.SEEDS for arm in ev.RISK_NAMES]


def test_gate_requires_three_core_controls_two_anchors_and_four_interventions():
    rows = gate_rows()
    gate = ev.inner_gate(rows, source=SOURCE, receipt_fingerprint="b" * 64)
    assert gate["status"] == "INNER_GATE_PASS"
    assert len(gate["primary_tests"]) == 6
    assert len(gate["anchor_noninferiority_screen"]) == 4
    assert len(gate["same_head_intervention_screen"]) == 4
    assert gate["outer_admitted"] and not gate["outer_automatically_started"]


@pytest.mark.parametrize("arm,metric,new", [("REL", "Uno60", .8), ("J_CS", "Harrell", .8),
    ("LOCAL_WRONG_O", "Uno60", .8), ("LOCAL_NO_C", "Harrell", .8)])
def test_each_gate_family_can_stop_current_operator(arm, metric, new):
    rows = gate_rows()
    for row in rows:
        if row["arm"] == arm:
            row[metric] = new
    gate = ev.inner_gate(rows, source=SOURCE, receipt_fingerprint="b" * 64)
    assert not gate["outer_admitted"]
    assert gate["status"] == "NOT_SUPPORTED_IN_THIS_END_TO_END_LOCAL_OPERATOR"


def test_gate_primary_needs_two_positive_seeds_even_with_positive_mean():
    rows = gate_rows()
    for row in rows:
        if row["arm"] == "CTRL":
            row["Harrell"] = .3 if row["seed"] == 31 else .75
    gate = ev.inner_gate(rows, source=SOURCE, receipt_fingerprint="b" * 64)
    assert not gate["outer_admitted"]


def test_intervention_ties_on_both_metrics_do_not_meet_mechanism_screen():
    rows = gate_rows()
    for row in rows:
        if row["arm"] == "LOCAL_OFF":
            row.update(Harrell=.7, Uno60=.7)
    gate = ev.inner_gate(rows, source=SOURCE, receipt_fingerprint="b" * 64)
    assert not gate["outer_admitted"]


def test_outer_is_not_implemented_or_automatically_evaluated(tmp_path):
    calls = []
    with pytest.raises(ValueError, match="separate admitted evaluation"):
        ev.evaluate_stage(tmp_path, tmp_path / "public", "outer", SOURCE, lambda: calls.append(1))
    assert calls == []


def test_invalid_endpoint_availability_fails_closed(tmp_path):
    full_stage(tmp_path)
    data = endpoints()
    data["DSS"]["valid"][0] = False
    with pytest.raises(ValueError, match="cannot exclude"):
        run(tmp_path, lambda: data)
