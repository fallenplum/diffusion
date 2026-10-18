"""Synthetic receipts/risks only; no predictor fitting or patient experiment."""
import json

import numpy as np
import pytest
import torch

from src.phase10_trimodal.reference import metrics
from src.phase23_cdei import evaluation as ev


SOURCE = "a" * 64


def synthetic_cell(root, seed, fold, *, role="inner", cohort_size=10):
    rng = np.random.default_rng(seed * 10 + fold)
    valid = np.array([2 * (fold - 1), 2 * (fold - 1) + 1])
    fit = np.setdiff1d(np.arange(cohort_size), valid)
    fit_relative = rng.normal(size=(len(fit), 64))
    valid_relative = rng.normal(size=(len(valid), 64))
    anchor_fit, anchor_valid = np.zeros(len(fit)), np.zeros(len(valid))
    checkpoint = {"source": SOURCE, "role": role, "seed": seed, "fold": fold,
                  "fit": fit, "valid": valid, "anchor_fit": anchor_fit,
                  "anchor_valid": anchor_valid, "J_CS_valid": np.zeros(len(valid)), "arms": {}}
    for arm in ev.ARMS:
        if arm == "REL":
            fit_x, valid_x = fit_relative.copy(), valid_relative.copy()
        else:
            fit_x = np.c_[fit_relative, rng.normal(size=(len(fit), 2))]
            valid_x = np.c_[valid_relative, rng.normal(size=(len(valid), 2))]
        mean, raw_sd = fit_x.mean(0), fit_x.std(0)
        constant = raw_sd < 1e-8
        scale = np.where(constant, 1., raw_sd)
        weights = rng.normal(size=fit_x.shape[1]) / 100
        checkpoint["arms"][arm] = {"fit_features": fit_x, "valid_features": valid_x,
            "weights": weights, "scaler": {"mean": mean, "scale": scale, "constant": constant, "fit_n": len(fit)},
            "fit_risk": anchor_fit + ((fit_x - mean) / scale) @ weights,
            "risk": anchor_valid + ((valid_x - mean) / scale) @ weights,
            "diagnostic": {"status": "CONVERGED", "gradient_max": 0., "penalty": 10.,
                           "parameters": len(weights), "offset_trainable": False, "intercept": False}}
    dual = checkpoint["arms"]["DUAL"]
    sensitivities = {}
    for name in ev.SENSITIVITIES:
        if name == "DUAL_ABS_OFF":
            normalized = (dual["valid_features"] - dual["scaler"]["mean"]) / dual["scaler"]["scale"]
            normalized[:, -2:] = 0
        else:
            raw = checkpoint["arms"][name[len("DUAL_"):]]["valid_features"]
            normalized = (raw - dual["scaler"]["mean"]) / dual["scaler"]["scale"]
        sensitivities[name] = anchor_valid + normalized @ dual["weights"]
    checkpoint["sensitivities"] = sensitivities
    folder = root / role / f"seed{seed}_fold{fold}"
    folder.mkdir(parents=True)
    torch.save({"synthetic": True}, folder / "kernel.pt")
    torch.save(checkpoint, folder / "readouts.pt")
    receipt = {"status": "COMPLETE", "source": SOURCE, "role": role, "seed": seed, "fold": fold,
               "kernel_epochs": 80, "readout_fits": 8,
               "kernel_sha256": ev.sha256_file(folder / "kernel.pt"),
               "readouts_sha256": ev.sha256_file(folder / "readouts.pt")}
    ev.write_json(folder / "DONE_PRIVATE.json", receipt)
    return folder


def full_stage(root, role="inner"):
    for seed in ev.SEEDS:
        for fold in ev.FOLDS:
            synthetic_cell(root, seed, fold, role=role)


def endpoints():
    return {"DSS": {"time": np.arange(1, 11, dtype=float),
                    "event": np.ones(10), "valid": np.ones(10, bool)}}


def test_missing_stage_never_calls_endpoint_loader(tmp_path):
    calls = []
    with pytest.raises(ValueError, match="missing complete"):
        ev.evaluate_stage(tmp_path, tmp_path / "public", "inner", SOURCE,
                          lambda: calls.append(1), cohort_size=10, expected_dss_events=10)
    assert calls == []
    assert not (tmp_path / "public").exists()


def test_tampered_artifact_never_calls_endpoint_loader(tmp_path):
    full_stage(tmp_path)
    (tmp_path / "inner" / "seed83_fold5" / "kernel.pt").write_bytes(b"changed")
    calls = []
    with pytest.raises(ValueError, match="hash mismatch"):
        ev.evaluate_stage(tmp_path, tmp_path / "public", "inner", SOURCE,
                          lambda: calls.append(1), cohort_size=10, expected_dss_events=10)
    assert calls == []


def test_15_receipts_preprocessing_and_all_saved_readouts_replay_before_metrics(tmp_path):
    full_stage(tmp_path)
    cells, audit = ev.verify_stage(tmp_path, "inner", SOURCE, cohort_size=10)
    assert len(cells) == 15 and audit["complete_cells"] == 15
    assert audit["maximum_elementwise_error"] == 0
    assert audit["all_replayed_values_elementwise_exact"]
    assert not audit["encoder_feature_replay_claimed"]
    calls = []
    result, gate = ev.evaluate_stage(tmp_path, tmp_path / "public", "inner", SOURCE,
        lambda: calls.append(1) or endpoints(), cohort_size=10, expected_dss_events=10)
    assert calls == [1] and not result["complete_outer_OOF"]
    assert not gate["confirmatory_confidence_intervals"]
    assert len(result["fold_rows"]) == 15 * (8 + 2 + 4)
    assert len(result["seed_rows"]) == 3 * (8 + 2 + 4)
    assert len(result["repair_damage"]) == 13 * 2
    assert len(result["seed_repair_damage"]) == 13 * 2 * 3
    assert (tmp_path / "public" / "INNER_GATE.json").is_file()
    text = json.dumps(result)
    assert "fit_features" not in text and "anchor_valid" not in text


@pytest.mark.parametrize("tamper", ["risk", "scale", "relative", "sensitivity"])
def test_inconsistent_but_rehashed_readout_still_fails_before_labels(tmp_path, tamper):
    full_stage(tmp_path)
    folder = tmp_path / "inner" / "seed31_fold1"
    checkpoint = torch.load(folder / "readouts.pt", weights_only=False)
    if tamper == "risk": checkpoint["arms"]["DUAL"]["risk"][0] += .1
    elif tamper == "scale": checkpoint["arms"]["DUAL"]["scaler"]["scale"][0] *= 2
    elif tamper == "relative": checkpoint["arms"]["ADD"]["fit_features"][0, 0] += .1
    elif tamper == "sensitivity": checkpoint["sensitivities"]["DUAL_ABS_OFF"][0] += .1
    torch.save(checkpoint, folder / "readouts.pt")
    receipt = json.loads((folder / "DONE_PRIVATE.json").read_text())
    receipt["readouts_sha256"] = ev.sha256_file(folder / "readouts.pt")
    ev.write_json(folder / "DONE_PRIVATE.json", receipt)
    calls = []
    with pytest.raises(ValueError):
        ev.evaluate_stage(tmp_path, tmp_path / "public", "inner", SOURCE,
                          lambda: calls.append(1), cohort_size=10, expected_dss_events=10)
    assert calls == []


def test_inner_sums_fold_pair_masses_instead_of_mean_of_fold_C(tmp_path):
    full_stage(tmp_path)
    cells, _ = ev.verify_stage(tmp_path, "inner", SOURCE, cohort_size=10)
    # Let inner validation overlap and give folds different sizes/pair masses.
    for seed in ev.SEEDS:
        for fold in ev.FOLDS:
            cell = cells[seed, fold]
            ix = np.arange(2 + fold)
            cell["valid"] = ix
            cell["fit"] = np.arange(7, 10)
            r = -np.arange(len(ix), dtype=float) if fold == 1 else np.arange(len(ix), dtype=float)
            for arm in ev.ARMS: cell["arms"][arm]["risk"] = r
            cell["anchor_valid"] = cell["J_CS_valid"] = r
            cell["sensitivities"] = {name: r for name in ev.SENSITIVITIES}
    result = ev.point_results(cells, endpoints()["DSS"], cohort_size=10)
    row = next(r for r in result["seed_rows"] if r["seed"] == 31 and r["arm"] == "DUAL")
    selected = [r for r in result["fold_rows"] if r["seed"] == 31 and r["arm"] == "DUAL"]
    assert row["Harrell"] == pytest.approx(sum(r["Harrell_num"] for r in selected) / sum(r["Harrell_den"] for r in selected))
    assert row["Harrell"] != pytest.approx(np.mean([r["Harrell"] for r in selected]))


def test_gate_requires_all_five_controls_both_metrics_and_two_positive_seeds():
    rows = [{"seed": seed, "arm": arm, "Harrell": .7 if arm == "DUAL" else .6,
             "Uno60": .7 if arm == "DUAL" else .6} for seed in ev.SEEDS for arm in [*ev.ARMS, "J0"]]
    good = ev.inner_gate(rows, source=SOURCE, receipt_fingerprint="b" * 64)
    assert good["status"] == "INNER_GATE_PASS" and len(good["mechanism_tests"]) == 10
    bad = [dict(row) for row in rows]
    for row in bad:
        if row["arm"] == "WRONG_C": row["Uno60"] = .8
    gate = ev.inner_gate(bad, source=SOURCE, receipt_fingerprint="b" * 64)
    assert gate["status"] == "NOT_SUPPORTED_IN_THIS_FIXED_KERNEL_READOUT" and not gate["outer_admitted"]


def test_outer_requires_pass_before_any_endpoint_read(tmp_path):
    calls = []
    with pytest.raises(ValueError, match="locked inner gate"):
        ev.evaluate_stage(tmp_path, tmp_path / "public", "outer", SOURCE,
                          lambda: calls.append(1), cohort_size=10)
    assert calls == []


def test_runner_nested_artifact_receipt_schema_is_supported(tmp_path):
    full_stage(tmp_path)
    for receipt_path in (tmp_path / "inner").glob("*/DONE_PRIVATE.json"):
        receipt = json.loads(receipt_path.read_text())
        receipt["artifacts"] = {"kernel.pt": receipt.pop("kernel_sha256"),
                                "readouts.pt": receipt.pop("readouts_sha256")}
        ev.write_json(receipt_path, receipt)
    cells, audit = ev.verify_stage(tmp_path, "inner", SOURCE, cohort_size=10)
    assert len(cells) == 15 and audit["complete_cells"] == 15


def test_inherited_anchor_callback_is_checked_before_reporting_results(tmp_path):
    full_stage(tmp_path)
    called = []
    def bad_anchor(role, fold):
        called.append((role, fold))
        return {"anchor_fit": np.zeros(8), "anchor_valid": np.ones(2), "J_CS_valid": np.zeros(2)}
    with pytest.raises(ValueError, match="frozen inherited anchor differs"):
        ev.evaluate_stage(tmp_path, tmp_path / "public", "inner", SOURCE, endpoints,
                          cohort_size=10, expected_dss_events=10, anchor_loader=bad_anchor)
    assert called == [("inner", 1)]
    assert not (tmp_path / "public" / "INNER_RESULTS.json").exists()


def test_outer_assembly_retains_patient_fold_identity_and_seed_specific_risks(tmp_path):
    full_stage(tmp_path, role="outer")
    cells, _ = ev.verify_stage(tmp_path, "outer", SOURCE, cohort_size=10)
    t, e, valid, folds, weights, risks = ev.assemble_outer(cells, endpoints()["DSS"], cohort_size=10)
    assert np.array_equal(folds, np.repeat(ev.FOLDS, 2))
    assert len(risks) == 3 * (8 + 2 + 4)
    assert np.isfinite(np.stack(list(risks.values()))).all()
    point = ev.point_results(cells, endpoints()["DSS"], cohort_size=10)
    for seed in ev.SEEDS:
        independently = metrics(t, e, risks["DUAL", seed], folds, weights)
        row = next(r for r in point["seed_rows"] if r["seed"] == seed and r["arm"] == "DUAL")
        assert row["Harrell"] == independently["Harrell"]
        assert row["Uno60"] == independently["Uno60"]


def test_secondary_missing_endpoint_values_are_excluded_only_by_explicit_mask(tmp_path):
    full_stage(tmp_path)
    cells, _ = ev.verify_stage(tmp_path, "inner", SOURCE, cohort_size=10)
    endpoint = endpoints()["DSS"]
    endpoint["time"][1] = np.nan
    endpoint["event"][1] = np.nan
    endpoint["valid"][1] = False
    # Endpoint validation permits masked NaN values, primary DSS cannot drop them.
    with pytest.raises(ValueError, match="silently exclude"):
        ev._endpoint(endpoint, 10, require_complete=True)
    _, _, mask = ev._endpoint(endpoint, 10, require_complete=False)
    assert not mask[1]


def test_inner_pair_credit_is_fold_mass_then_seed_mean_and_keeps_ties():
    endpoint = {"time": np.arange(1, 11, dtype=float),
                "event": np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 0]),
                "valid": np.ones(10, bool)}
    groups = ([0, 1], [2, 3, 4], [2, 3, 4, 5], [0, 2, 3, 4, 6], [0, 1, 2, 3, 4, 5, 6])
    cells = {}
    for seed in ev.SEEDS:
        for fold, group in zip(ev.FOLDS, groups):
            ix = np.asarray(group)
            fit = np.setdiff1d(np.arange(10), ix)
            base = np.arange(len(ix), dtype=float)
            dual = -base.copy() if fold % 2 else base.copy()
            if fold == 1:
                dual[:] = 0.  # wrong -> tied must carry exactly half-credit.
            risks = {arm: {"risk": dual.copy() if arm == "DUAL" else base.copy()} for arm in ev.ARMS}
            cells[seed, fold] = {"fit": fit, "valid": ix, "arms": risks,
                "anchor_valid": base.copy(), "J_CS_valid": base.copy(),
                "sensitivities": {name: base.copy() for name in ev.SENSITIVITIES}}
    point = ev.point_results(cells, endpoint, cohort_size=10)
    answer = ev.inner_pair_transitions(cells, endpoint, point, cohort_size=10)
    assert len(answer["repair_damage"]) == 26 and len(answer["seed_repair_damage"]) == 78
    metric_rows = {(r["seed"], r["arm"]): r for r in point["seed_rows"]}
    for row in answer["seed_repair_damage"]:
        control = row["comparison"][len("DUAL-"):]
        expected = metric_rows[row["seed"], "DUAL"][row["metric"]] - metric_rows[row["seed"], control][row["metric"]]
        assert row["net"] == pytest.approx(expected, abs=1e-12)
        assert row["repair"] - row["damage"] == pytest.approx(expected, abs=1e-12)
        assert row["wrong_to_tied_mass"] > 0
        assert row["wrong_to_tied_mass"] <= 1.
    harrell = next(row for row in answer["seed_repair_damage"]
                   if row["seed"] == 31 and row["comparison"] == "DUAL-REL" and row["metric"] == "Harrell")
    cell_deltas = []
    for fold in ev.FOLDS:
        row = next(r for r in point["fold_rows"] if r["seed"] == 31 and r["fold"] == fold and r["arm"] == "DUAL")
        old = next(r for r in point["fold_rows"] if r["seed"] == 31 and r["fold"] == fold and r["arm"] == "REL")
        cell_deltas.append(row["Harrell"] - old["Harrell"])
    assert harrell["net"] != pytest.approx(np.mean(cell_deltas))
    for row in answer["repair_damage"]:
        assert not any("CI" in key or "bootstrap" in key for key in row)
        seeds = [r for r in answer["seed_repair_damage"]
                 if r["comparison"] == row["comparison"] and r["metric"] == row["metric"]]
        assert row["net"] == pytest.approx(np.mean([r["net"] for r in seeds]), abs=1e-12)
