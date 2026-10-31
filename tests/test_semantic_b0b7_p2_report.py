"""Anonymous report qualification with synthetic receipts, never cohort results."""
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import semantic_b0b7_p2_report as report
from src.semantic_b0b7.p2_survival import fit_risk_cell, replay_risk_cell, array_sha


def metric_row(arm, fold, seed, hnum, hden, unum=None, uden=None):
    return {"arm": arm, "fold": fold, "seed": seed, "status": "COMPLETE",
        "Harrell_num": hnum, "Harrell_den": hden, "Harrell": hnum/hden,
        "Uno_num": unum if unum is not None else hnum, "Uno_den": uden if uden is not None else hden,
        "Uno60": (unum/uden) if unum is not None else hnum/hden}


def test_seed_micro_then_equal_seed_mean_is_not_fifteen_cell_mean():
    rows = []
    for seed, ratio in zip(report.SEEDS, (.4, .6, .8)):
        for fold in report.FOLDS:
            denominator = 1 if fold == 1 else 100
            score = 1. if fold == 1 else ratio
            rows.append(metric_row("B5", fold, seed, score*denominator, denominator))
    actual = report.aggregate_arm("B5", rows)
    expected = np.mean([(1 + 400*r)/401 for r in (.4, .6, .8)])
    assert actual["Harrell"] == pytest.approx(expected)
    assert actual["Harrell"] != pytest.approx(np.mean([r["Harrell"] for r in rows]))
    assert actual["cross_fold_risk_pairs"] is False


def test_uno_has_own_denominator_and_missing_fold_is_not_renormalized():
    rows = [metric_row("B1", f, 31, f, 10, 2*f, 20+f) for f in report.FOLDS]
    result = report.fold_micro(rows)
    assert result["Harrell"] == pytest.approx(15/50)
    assert result["Uno60"] == pytest.approx(30/115)
    partial = report.fold_micro(rows[:-1])
    assert partial["Harrell"] is None and partial["Uno60"] is None
    with pytest.raises(ValueError, match="DUPLICATE_FOLD"):
        report.fold_micro(rows + [rows[0]])
    rows[-1].update(Uno_num=None, Uno_den=None, Uno60=None)
    assert report.fold_micro(rows)["Harrell"] is not None
    assert report.fold_micro(rows)["Uno60"] is None


def test_deterministic_b0_reference_pairs_with_each_seed_without_fake_refits():
    rows = [metric_row("B0", f, 31, 6, 10) for f in report.FOLDS]
    rows += [metric_row("B5", f, s, 6.02, 10) for s in report.SEEDS for f in report.FOLDS]
    aggregate = {"B0": report.aggregate_arm("B0", rows, seeds=(31,)), "B5": report.aggregate_arm("B5", rows)}
    result = report.comparison("B5", "B0", aggregate, rows)
    assert result["Harrell_delta"] == pytest.approx(.002)
    assert result["Harrell_fold_seed_pairs"] == result["Harrell_positive_fold_seed_pairs"] == 15
    assert len(aggregate["B0"]["seed_results"]) == 1
    assert result["Harrell_interpretation"] == "WEAK_POSITIVE_DIRECTION_LT_0.005"
    assert result["confidence_intervals_computed"] is False and result["bootstrap_replicates"] == 0


def test_paired_cells_preserve_denominators_b0_lineage_and_metric_specific_worst_cells():
    rows = [metric_row("B0", f, 31, 6., 10., 12., 20.) for f in report.FOLDS]
    for f in report.FOLDS:
        for s in report.SEEDS:
            hdelta = -.03 if (f, s) == (2, 47) else .01
            udelta = -.04 if (f, s) == (4, 83) else .02
            rows.append(metric_row("B5", f, s, (0.6+hdelta)*10., 10., (0.6+udelta)*20., 20.))
    cells = report.paired_fold_seed_rows("B5", "B0", rows)
    assert len(cells) == 15 and all(x["reference_seed"] == 31 for x in cells)
    assert all(x["reference_B0_is_shared_deterministic_fit"] for x in cells)
    assert all(x["Harrell_den_left"] == x["Harrell_den_reference"] == 10. for x in cells)
    assert all(x["Uno_den_left"] == x["Uno_den_reference"] == 20. for x in cells)
    direction = report.paired_fold_directions("B5", "B0", cells)
    fold2 = next(x for x in direction if x["fold"] == 2)
    assert fold2["Harrell_positive_seeds"] == 2 and fold2["Harrell_negative_seeds"] == 1
    assert fold2["Harrell_fold_delta"] == pytest.approx((.01-.03+.01)/3)
    aggregates = {"B5": report.aggregate_arm("B5", rows), "B0": report.aggregate_arm("B0", rows, seeds=(31,))}
    summary = report.comparison("B5", "B0", aggregates, rows)
    assert summary["Harrell_worst_fold_seed_cell"]["fold"] == 2
    assert summary["Harrell_worst_fold_seed_cell"]["seed"] == 47
    assert summary["Uno60_worst_fold_seed_cell"]["fold"] == 4
    assert summary["Uno60_worst_fold_seed_cell"]["seed"] == 83
    assert summary["Harrell_worst_fold_equal_seed_mean_diagnostic"]["fold"] == 2
    assert summary["Uno60_worst_fold_equal_seed_mean_diagnostic"]["fold"] == 4


def test_paired_missing_metric_is_na_not_zero_or_fold_renormalized_and_duplicates_fail():
    rows = [metric_row("B0", f, 31, 6., 10.) for f in report.FOLDS]
    rows += [metric_row("B5", f, s, 6.1, 10.) for s in report.SEEDS for f in report.FOLDS]
    next(x for x in rows if x["arm"] == "B5" and x["fold"] == 3 and x["seed"] == 47).update(
        Uno60=None, Uno_num=None, Uno_den=None)
    cells = report.paired_fold_seed_rows("B5", "B0", rows)
    directions = report.paired_fold_directions("B5", "B0", cells)
    fold3 = next(x for x in directions if x["fold"] == 3)
    assert fold3["Uno60_available_seeds"] == 2 and fold3["Uno60_fold_delta"] is None
    assert fold3["Harrell_fold_delta"] == pytest.approx(.01)
    assert next(x for x in cells if (x["fold"], x["seed"]) == (3, 47))["Uno60_direction"] == "NA"
    with pytest.raises(ValueError, match="DUPLICATE_ARM_FOLD_SEED"):
        report.paired_fold_seed_rows("B5", "B0", rows + [rows[0]])
    missing = report.paired_fold_seed_rows("B5", "B3", rows)
    assert all(x["status"] == "NOT_RUN_OR_INCOMPLETE" and x["Harrell_delta"] is None for x in missing)


def test_risk_scale_metrics_are_readonly_aggregate_receipt_checks_not_new_fits():
    tt = np.linspace(1., 130., 30); te = (np.arange(30)%3 == 0).astype(int)
    et = np.asarray([1., 2., 4., 70., 100.]); ee = np.asarray([1, 1, 0, 1, 0])
    trbase = np.sin(np.arange(30)); evbase = np.asarray([0., .2, -.1, .8, .3])
    train = trbase + np.linspace(-.2, .2, 30)
    evaluation = evbase + np.asarray([.1, -.2, .1, .2, -.1])
    compact = {"train_time": tt, "train_event": te, "evaluation_time": et, "evaluation_event": ee,
               "training_offset": trbase, "evaluation_offset": evbase}
    receipt = {"role": "outer_final", "train_metrics": report.evaluate_safe(tt, te, tt, te, train),
               "evaluation_metrics": report.evaluate_safe(tt, te, et, ee, evaluation),
               "anchor_evaluation_metrics": report.evaluate_safe(tt, te, et, ee, evbase),
               "train_delta_sd": float(np.std(train-trbase)), "train_anchor_sd": float(np.std(trbase)),
               "repair_damage": {"risk_delta_sd": float(np.std(evaluation-evbase)), "anchor_sd": float(np.std(evbase))}}
    cell = SimpleNamespace(training_risk=train, evaluation_risk=evaluation)
    row = report.risk_scale_row(receipt, cell, compact, {"arm": "B5", "split": "outer1_final", "seed": 31})
    assert row["train_delta_sd"] == pytest.approx(np.std(train-trbase))
    assert row["evaluation_delta_sd_over_anchor_sd"] == pytest.approx(np.std(evaluation-evbase)/np.std(evbase))
    assert row["train_Harrell"] == receipt["train_metrics"]["Harrell"]
    assert row["evaluation_Harrell_delta_vs_r0"] == pytest.approx(
        receipt["evaluation_metrics"]["Harrell"]-receipt["anchor_evaluation_metrics"]["Harrell"])
    assert row["new_model_forwards"] == row["new_optimizer_steps"] == 0
    assert "not_end_to_end_OOF" in row["training_metrics_scope"]
    report.public_bytes(row)
    receipt["train_delta_sd"] += .1
    with pytest.raises(ValueError, match="TRAIN_RISK_SCALE_RECEIPT_CHANGED"):
        report.risk_scale_row(receipt, cell, compact, {"arm": "B5"})


def test_no_outer_authorization_means_no_input_read(monkeypatch, tmp_path):
    def forbidden(*a, **kw):
        raise AssertionError("private input was touched")
    monkeypatch.setattr(report, "sha", forbidden)
    with pytest.raises(ValueError, match="POSTCOMPLETION_AUTHORIZATION"):
        report.generate_reports("unread", "unread", "unread", tmp_path / "out")


def test_public_serialization_rejects_arrays_paths_identifiers_and_nonfinite():
    for value in ({"bad": np.ones(3)}, {"bad": "D:/private/measurements.npy"},
                  {"bad": "/root/" + "private/model.pt"}, {"bad": "TC" + "GA-AA-0001"}, {"bad": float("nan")}):
        with pytest.raises(ValueError):
            report.public_bytes(value)
    assert b"source_sha256" in report.public_bytes({"source_sha256": "a"*64})
    failure = report.public_failure({"arm": "B5"}, FileNotFoundError("/root/" + "private/patient.npy"))
    assert failure["status"] == "NOT_RUN" and "/root" not in json.dumps(failure)


def test_public_file_limits_and_append_only(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "MAX_BYTES", 1500)
    monkeypatch.setattr(report, "MAX_CSV_ROWS", 3)
    rows = [{"arm": "B5", "fold": i, "summary": "x"*100} for i in range(8)]
    outputs = report.write_records(tmp_path, "05_synthetic", rows)
    assert len([x for x in outputs if x["file"].endswith(".json")]) == 4
    for item in outputs:
        assert (tmp_path/item["file"]).stat().st_size <= 1500
        assert item.get("data_rows", 0) <= 2
    with pytest.raises(FileExistsError):
        report.write_records(tmp_path, "05_synthetic", rows)


def test_actual_cost_counts_shared_views_once_but_repeated_pca_each_lambda():
    phase = [{"optimizer_steps": 400, "actual_checkpoint_replays": 4, "elapsed_seconds": 1.,
              "peak_cuda_allocated_bytes": 1000} for _ in range(60)]
    heads = []
    for split in [f"outer{o}_inner{i}" for o in report.FOLDS for i in range(3)] + [f"outer{o}_final" for o in report.FOLDS]:
        for seed in report.SEEDS:
            for lam in ((.1, 1., 10.) if "inner" in split else (1.,)):
                heads.append({"split": split, "arm": "B5", "seed": seed, "ridge_lambda": lam,
                    "PCA_fits": 3, "background_Ridge_fits_for_this_view": 4})
    anchors = [{"Cox_optimizer_fits": 52, "scalar_pair_optimizer_fits": 4,
                "preprocessing_fits": 16, "elapsed_seconds": 1.} for _ in range(20)]
    costs = report.cost_report({"phase": phase, "heads": heads, "anchors": anchors, "arms": {"B5": "family"}},
                              {"family_readouts": {"family": ["B5"]}})
    assert costs["status"] == "COMPLETE"
    assert costs["actual_risk_Cox_fits"] == 150
    assert costs["actual_risk_map_PCA_fits_including_lambda_repeats"] == 450
    assert costs["actual_outcome_free_background_Ridge_fits"] == 240
    assert costs["actual_r0_Cox_fits"] == 1040 and costs["actual_r0_scalar_fits"] == 80
    assert costs["actual_neural_fits"] == 60 and costs["actual_phase_A_optimizer_steps"] == 24000


def make_private_risk(tmp_path):
    rng = np.random.default_rng(77)
    n, m = 42, 9
    f = lambda nn: {k: rng.normal(size=(nn, 12 if k != "mechanism" else 2))
                   for k in ("pathology", "omics", "clinical", "mechanism")}
    tr, ev = f(n), f(m)
    tt = np.linspace(1, 130, n); te = (np.arange(n)%3 == 0).astype(int)
    et = np.linspace(1, 90, m); ee = (np.arange(m)%3 == 0).astype(int)
    offset = rng.normal(size=n); eo = rng.normal(size=m)
    phase = {"source_checkpoint_sha256": "a"*64, "input_training_sha256": "b"*64}
    binding = {"lock_sha256": "c"*64, "split": "outer1_final", "arm": "B5", "seed": 31,
               "phase_a_source_family": "family", "phase_a_selected_state_sha256": "a"*64,
               "input_training_sha256": "b"*64}
    cell = fit_risk_cell(tr, ev, tt, te, et, ee, offset, eo, ridge_lambda=1., role="outer_final",
                         bindings=binding, selection_record={"uses_outer_test": False, "ridge_lambda": 1.})
    folder = tmp_path / "risk/outer1_final/B5_seed31"
    folder.mkdir(parents=True)
    path = folder / "lambda1.0.private.pkl"
    path.write_bytes(pickle.dumps(cell, protocol=5))
    compact = {"evaluation_time": et, "evaluation_event": ee, "evaluation_offset": eo,
        "train_time": tt, "train_event": te, "training_offset": offset,
        "train_design": cell.design.transform(tr), "evaluation_design": cell.design.transform(ev), "background_audit": None}
    (folder / "lambda1.0.compact.private.pkl").write_bytes(pickle.dumps(compact, protocol=5))
    row = {**cell.receipt, "artifact_sha256": report.sha(path),
           "saved_linear_head_replay": replay_risk_cell(cell, ev, eo, tt, te, et, ee)}
    (folder / "lambda1.0.private.json").write_text(json.dumps(row), encoding="utf-8")
    return folder, phase, compact


def test_real_small_synthetic_head_hash_and_arithmetic_recheck(tmp_path):
    folder, phase, compact = make_private_risk(tmp_path)
    row, cell, original, audit = report.checked_risk(tmp_path, "c"*64, "outer1_final", "B5", 31, 1., "family", phase)
    assert audit["new_optimizer_steps"] == 0 and audit["new_neural_forward"] is False
    assert np.array_equal(cell.evaluation_risk, cell.model.predict(compact["evaluation_design"], compact["evaluation_offset"]))
    with (folder/"lambda1.0.private.pkl").open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="PRIVATE_BYTES_CHANGED"):
        report.checked_risk(tmp_path, "c"*64, "outer1_final", "B5", 31, 1., "family", phase)


def test_compact_endpoint_or_design_tampering_is_rejected(tmp_path):
    folder, phase, compact = make_private_risk(tmp_path)
    compact["evaluation_time"] = compact["evaluation_time"] + 1
    (folder / "lambda1.0.compact.private.pkl").write_bytes(pickle.dumps(compact))
    with pytest.raises(ValueError, match="COMPACT_REPLAY_INPUT_HASH_CHANGED"):
        report.checked_risk(tmp_path, "c"*64, "outer1_final", "B5", 31, 1., "family", phase)


def test_repair_damage_is_same_paired_statistic_as_fold_difference():
    t = np.array([1., 2., 3., 4., 70.]); e = np.array([1, 1, 0, 1, 0])
    base = np.array([0., 1., .3, .4, -.2]); risk = np.array([1., 0., .1, .5, -.2])
    train_t = np.linspace(1, 130, 30); train_e = (np.arange(30)%3 == 0).astype(int)
    compact = {"evaluation_time": t, "evaluation_event": e, "train_time": train_t, "train_event": train_e}
    rows = report.repair_reports({"arms": {"B5": "family"}, "vectors": {
        ("B5", 1, 31): (SimpleNamespace(evaluation_risk=risk), compact),
        ("B0", 1, 31): (SimpleNamespace(evaluation_risk=base), compact)}})
    assert len(rows) == 1
    assert rows[0]["net_Harrell_change"] == pytest.approx((rows[0]["repair_credit"] - rows[0]["damage_credit"])/rows[0]["comparable_pairs"])


def test_full_public_report_generation_from_synthetic_aggregates_only(tmp_path, monkeypatch):
    # This fixture tests report plumbing; its mocked counters are not empirical
    # training evidence. No real cohort files or saved neural models are read.
    t=np.array([1., 2., 3., 4., 70.]); e=np.array([1, 1, 0, 1, 0])
    trt=np.linspace(1, 130, 30); tre=(np.arange(30)%3 == 0).astype(int)
    base=np.array([0., 1., .3, .4, -.2]); risk=base+.001
    compact={"evaluation_time":t, "evaluation_event":e, "train_time":trt, "train_event":tre}
    fm=report.evaluate_safe(trt,tre,t,e,risk)
    folds=[]; vectors={}
    for outer in report.FOLDS:
        folds.append(report.metric_row("B0",outer,31,fm))
        vectors[("B0",outer,31)]=(SimpleNamespace(evaluation_risk=base), compact)
        for arm in ("B1","B5"):
            for seed in report.SEEDS:
                folds.append(report.metric_row(arm,outer,seed,fm))
                vectors[(arm,outer,seed)]=(SimpleNamespace(evaluation_risk=risk), compact)
    collected={"arms":{"B1":"family","B5":"family"}, "inner":[], "selection":[],
        "fold":folds,"vectors":vectors,"phase":[],"heads":[],"anchors":[],"failures":[],
        "selection_file_sha256":"d"*64}
    monkeypatch.setattr(report,"collect_run",lambda *a:collected)
    out=tmp_path/"anonymous"
    lock=tmp_path/"lock.json"
    lock.write_text(json.dumps({"status":"LOCKED_BEFORE_OUTER_EVALUATION","frozen_sources":{},
                               "family_readouts":{"family":["B1","B5"]}}),encoding="utf-8")
    result=report.generate_reports("unused_synthetic",lock,report.sha(lock),out,allow_completed_outer_results=True)
    assert result["status"]=="FAILED_OR_PARTIAL_NO_CONFIRMATORY_CONCLUSION"  # no real fit evidence
    content=(out/"RESULTS.md").read_text(encoding="utf-8")
    assert "B5" in content and "置信区间" in content and "TITAN S32" in content
    aggregate=json.loads((out/"05_seed_aggregated_metrics.json").read_text(encoding="utf-8"))
    assert next(x for x in aggregate if x["arm"]=="B0")["required_seeds"]==[31]
    cells=json.loads((out/"06_fold_seed_paired_deltas.json").read_text(encoding="utf-8"))
    assert len(cells)==15*len(report.CORE)
    directions=json.loads((out/"06_fold_direction_summaries.json").read_text(encoding="utf-8"))
    assert len(directions)==5*len(report.CORE)
    extras=json.loads((out/"06_supplementary_reference_comparisons.json").read_text(encoding="utf-8"))
    assert any(x["comparison"]=="B5-B3" for x in extras)
    assert all(x["scope"]=="derived_reference_diagnostics_not_new_primary_model_selection" for x in extras)
    for path in out.iterdir():
        assert path.stat().st_size <= report.MAX_BYTES
        assert "TCGA-" not in path.read_text(encoding="utf-8")
    with pytest.raises(ValueError,match="APPEND_ONLY"):
        report.generate_reports("unused_synthetic",lock,report.sha(lock),out,allow_completed_outer_results=True)


def test_resource_profile_separates_forward_and_backward_compute_match(tmp_path):
    profiles=[]
    for role in ("inner","outer_final"):
        for arm in ("B4_DUAL","B2_MATCHED","B2_NATIVE"):
            profiles.append({"role":role,"arm":arm,"train_patients":40,"parameters":100,
                "gradient_norms":{"weight":.1},"forward_profile_seconds":.1,"forward_backward_profile_seconds":.3,
                "peak_cuda_allocated_bytes":1000,"forward_supported_matrix_FLOPs":100,
                "forward_backward_including_checkpoint_recomputation_supported_matrix_FLOPs":
                    130 if arm=="B2_MATCHED" else 100,
                "checkpoint_sha256":"a"*64,"FLOPs_exclude_unsupported_elementwise_operators":True,
                "actual_saved_reload_forward_replay":True,"optimizer_steps":0,"new_neural_fits":0,
                "profiler_event_collection":"CPU_dispatcher_shapes_for_real_CUDA_execution_no_CUPTI_trace"})
    value={"endpoints_read":False,"optimizer_updates":0,"new_neural_fits":0,"actual_saved_state_replays":6,
           "sources":{},"profiles":profiles}
    path=tmp_path/"PROFILE.json";path.write_text(json.dumps(value),encoding="utf-8")
    actual=report.resource_profile(path,report.sha(path))
    matched=next(x for x in actual["comparisons"] if x["arm"]=="B2_MATCHED")
    assert matched["forward_supported_matrix_FLOPs_within_10_percent"] is True
    assert matched["forward_backward_including_checkpoint_recomputation_supported_matrix_FLOPs_within_10_percent"] is False
    assert matched["matched_qualification"]=="NOT_QUALIFIED_AS_COMPUTE_MATCHED"
    value["endpoints_read"]=True;path.write_text(json.dumps(value),encoding="utf-8")
    with pytest.raises(ValueError,match="OUTCOME_OR_OPTIMIZER"):
        report.resource_profile(path,report.sha(path))
