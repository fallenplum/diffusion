"""Synthetic receipt aggregation/negative gates; no real data or model loading."""
from copy import deepcopy
import csv
import itertools
import json
from pathlib import Path

import pytest

from scripts import semantic_b0b7_p2s_report as report


def metric(h=.5, denominator=20., uno=None):
    u = h if uno is None else uno
    return {"Harrell": h, "Harrell_num": h * denominator, "Harrell_den": denominator,
            "Uno60": u, "Uno_num": u * denominator * 1.2, "Uno_den": denominator * 1.2,
            "Uno60_valid": True}


def rows_for(key, *, shift=0.):
    family, pair, epoch, d, arm, rank, mode, ridge = key
    rows = []
    for si, seed in enumerate(report.SEEDS):
        for fi, split in enumerate(report.SPLITS):
            rows.append({"family": family, "lambda_pair": pair, "epoch": epoch, "seed": seed, "split": split,
                         "spec": {"main_pca": d, "base_arm": arm, "factor_rank": rank, "mode": mode}, "ridge_lambda": ridge,
                         "status": report.STATUS_FIT, "evaluation_metrics": metric(.3 + si * .1 + fi * .03 + shift, denominator=(fi + 1) ** 2 * (si + 1) * 10.)})
    return rows


KEY = ("MAIN", .1, 400, 8, "B5", 2, "INTER", 1.)


def test_seed_micro_is_not_nine_cell_mean_or_global_component_pool():
    rows = rows_for(KEY)
    # Unequal event/comparable-pair supports are intentional.
    result = report.aggregate_config(rows)
    expected = sum(sum(r["evaluation_metrics"]["Harrell_num"] for r in rows if r["seed"] == seed)
                   / sum(r["evaluation_metrics"]["Harrell_den"] for r in rows if r["seed"] == seed) for seed in report.SEEDS) / 3
    assert result["Harrell"] == pytest.approx(expected)
    assert result["Harrell"] != pytest.approx(sum(r["evaluation_metrics"]["Harrell"] for r in rows) / 9)
    assert result["Harrell"] != pytest.approx(sum(r["evaluation_metrics"]["Harrell_num"] for r in rows)
                                             / sum(r["evaluation_metrics"]["Harrell_den"] for r in rows))
    assert result["Uno60"] == pytest.approx(expected)


def test_missing_na_or_duplicate_cells_do_not_silently_become_zero():
    rows = rows_for(KEY)
    with pytest.raises(ValueError, match="NINE_PAIRED"):
        report.aggregate_config(rows[:-1])
    with pytest.raises(ValueError, match="NINE_PAIRED"):
        report.aggregate_config(rows[:-1] + [rows[0]])
    rows[0]["status"] = report.STATUS_NA
    rows[0]["evaluation_metrics"] = None
    out = report.aggregate_config(rows)
    assert out["Harrell"] is None and out["fitted_cells"] == 8
    assert out["seed_results"][0]["Harrell"]["supported_partitions"] == 2


def test_paired_delta_requires_identical_within_partition_support():
    a, b = KEY, (*KEY[:2], 800, *KEY[3:])
    left, right = rows_for(a, shift=.02), rows_for(b)
    indexed = {a: {(r["seed"], r["split"]): r for r in left}, b: {(r["seed"], r["split"]): r for r in right}}
    summary, cells = report.compare_terms("test", [(1, a), (-1, b)], indexed)
    assert summary["Harrell_delta"] == pytest.approx(.02) and len(cells) == 9
    assert summary["Harrell_cell_positive"] == 9 and summary["Harrell_cell_NA"] == 0
    right[0]["evaluation_metrics"] = metric(.2, denominator=99.)
    with pytest.raises(ValueError, match="DIFFERENT_WITHIN_PARTITION_DENOMINATORS"):
        report.compare_terms("different", [(1, a), (-1, b)], indexed)


def test_conditional_increment_contrast_uses_same_nine_cells_and_seed_micro():
    keys = [KEY, ("MAIN", .1, 400, 8, "SHARED", 0, "NO_INTER", 1.),
            ("MAIN", .1, 800, 8, "B5", 2, "INTER", 1.), ("MAIN", .1, 800, 8, "SHARED", 0, "NO_INTER", 1.)]
    shifts = [.04, .03, .07, .02]
    indexed = {key: {(r["seed"], r["split"]): r for r in rows_for(key, shift=shift)} for key, shift in zip(keys, shifts)}
    # (.07-.02)-(.04-.03)=.04, not the full-risk change .03.
    summary, _ = report.compare_terms("conditional", [(1, keys[2]), (-1, keys[3]), (-1, keys[0]), (1, keys[1])], indexed)
    assert summary["Harrell_delta"] == pytest.approx(.04)
    next(iter(indexed[keys[1]].values()))["status"] = report.STATUS_NA
    summary, cells = report.compare_terms("withNA", [(1, keys[2]), (-1, keys[3]), (-1, keys[0]), (1, keys[1])], indexed)
    assert summary["Harrell_delta"] is None and summary["Harrell_cell_NA"] == 1
    assert len(cells) == 9


@pytest.mark.parametrize("name", ["Harrell", "Uno60"])
def test_metric_components_are_checked_without_recomputing_patient_metrics(name):
    m = metric()
    m[name] += .01
    with pytest.raises(ValueError, match="COMPONENTS_DISAGREE"):
        report.metric_value(m, name)
    m = metric(); m[name] = None
    nk, dk = report.METRIC_COMPONENTS[name]
    m[nk] = m[dk] = None
    if name == "Uno60":
        m["Uno60_valid"] = False
    assert report.metric_value(m, name) is None


def test_complete_fixed_comparison_set_has_no_extra_architecture_or_lambda_selection():
    indexed = {}
    for family, pair, epoch, ridge in itertools.product(report.FAMILIES, report.PAIR_WEIGHTS, report.EPOCHS, report.RIDGES):
        for d, arm, rank, mode in report.specs(family):
            key = family, pair, epoch, d, arm, rank, mode, ridge
            indexed[key] = {}
    definitions = report.comparisons(indexed)
    assert len(indexed) == 264 and len(definitions) == len({n for n, _ in definitions})
    assert any("COND_NO_INTER_800_minus400" in n for n, _ in definitions)
    assert any("COND_B7_pair003_minus010" in n for n, _ in definitions)
    assert any("MAIN_B5_vs_ATTN" in n for n, _ in definitions)
    assert any("R4_minus_R2" in n for n, _ in definitions)
    assert any("D16_minus_D8" in n for n, _ in definitions)
    assert len(report.specs("MAIN")) == 18 and len(report.specs("ATTN_MATCHED")) == 4
    assert {s[1] for s in report.specs("ATTN_MATCHED")} == {"SHARED", "ATTN_MATCHED"}
    removed = next(k for k in indexed if k[0] == "MAIN")
    del indexed[removed]
    with pytest.raises(ValueError, match="COMPARISON_CONFIGURATION_MISSING"):
        report.comparisons(indexed)


def test_csv_chunking_preserves_every_row_and_header_inclusive_line_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "MAX_CSV_LINES", 5)
    monkeypatch.setattr(report, "MAX_BYTES", 240)
    source = [{"number": i, "text": "quoted, value"} for i in range(13)]
    files = report.write_csv_parts(tmp_path, "curve", source)
    restored = []
    for item in files:
        data = (tmp_path / item["file"]).read_bytes()
        assert len(data) <= 240 and data.count(b"\n") <= 5 and report.sha(tmp_path / item["file"]) == item["sha256"]
        restored.extend(csv.DictReader(data.decode().splitlines()))
    assert [int(x["number"]) for x in restored] == list(range(13))
    assert all(x["text"] == "quoted, value" for x in restored)
    with pytest.raises(ValueError, match="MULTILINE"):
        report.write_csv_parts(tmp_path, "unsafe", [{"text": "two\nlines"}])


def test_plain_json_shards_preserve_all_values_and_unchanged_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "MAX_JSON_LIST", 3)
    monkeypatch.setattr(report, "MAX_BYTES", 270)
    original = [{"position": i, "result": {"point": i / 100, "NA": None}} for i in range(11)]
    files = report.write_json_parts(tmp_path, "parts", original)
    restored = []
    for item in files:
        data = (tmp_path / item["file"]).read_bytes()
        part = json.loads(data)
        assert len(part) <= 3 and len(data) <= 270
        restored.extend(part)
    assert restored == original


@pytest.mark.parametrize("bad", ["/" + "root/private/file", "D:\\private\\x", "hf_" + "a" * 20,
                                   "TCGA-" + "AA-" + "1234", list(range(129)), [None] * 1001, object(), float("nan")])
def test_privacy_and_array_guard_do_not_allow_opaque_public_escape(bad):
    with pytest.raises(ValueError):
        report.public_value({"field": bad})


def test_reporter_does_not_open_model_features_or_pickle(tmp_path):
    for suffix in (".pt", ".npz", ".pkl", ".h5"):
        with pytest.raises(ValueError, match="ONLY_OPENS_JSON"):
            report.read_json(tmp_path / ("forbidden" + suffix))
    with pytest.raises(ValueError, match="UNSAFE_RECEIPT"):
        report.safe_receipt(tmp_path, "../outside.json", "a" * 64)


def aux_metric(h, p=.2, gap=.1):
    return {"program_Huber": h, "macro_program_Pearson": p, "hard_gap": gap}


def test_auxiliary_comparisons_keep_pair_weights_separate_and_do_not_rank_total_loss():
    states = []
    for split, family, seed, pair, epoch in report.expected_states():
        h = .4 + (.03 if pair == .03 else 0) - (.02 if epoch == 800 else 0)
        states.append({"split": split, "family": family, "seed": seed, "lambda_pair": pair, "epoch": epoch, **aux_metric(h)})
    summaries, paired = report.auxiliary_comparisons(states)
    assert len(summaries) == 12 and len(paired) == 108
    assert all(not x["weighted_total_auxiliary_loss_compared"] and not x["selection_performed"] for x in summaries)
    selected = next(x for x in summaries if x["comparison"] == "MAIN_pair0.1_800_minus400")
    assert selected["program_Huber_mean_paired_delta"] == pytest.approx(-.02)


def job_fixture(tmp_path):
    meta = {"split": report.SPLITS[0], "family": "ATTN_MATCHED", "seed": 31, "lambda_pair": .1, "epoch": 400,
            "input_sha256": "a" * 64, "checkpoint_sha256": "b" * 64}
    rows = []
    for (d, arm, rank, mode), ridge in itertools.product(report.specs("ATTN_MATCHED"), report.RIDGES):
        rows.append({"spec": {"main_pca": d, "base_arm": arm, "factor_rank": rank, "mode": mode}, "ridge_lambda": ridge,
                     "primary_lambda": 1., "is_primary_lambda": ridge == 1., "status": report.STATUS_NA, "NA_reason": "INSUFFICIENT_REAL_RANK",
                     "evaluation_metrics": None})
    payload = {"schema": "semantic_b0b7_p2s_readout_state_v1", "status": "PARTIAL_PRIMARY_UNSUPPORTED_RANK", "meta": meta, "rows": rows,
               "actual_Cox_fits": 0, "NA_count": 12, "expected_fits": 12, "new_neural_fits": 0, "new_model_forwards": 0,
               "actual_PCA_fits": 0, "actual_bgRidge": 0, "elapsed_seconds": .1}
    path = tmp_path / "READOUT.private.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload, path


def test_readout_manifest_must_include_all72_states_and_unique_grid(tmp_path):
    _, path = job_fixture(tmp_path)
    manifest = {"schema": "semantic_b0b7_p2s_readout_manifest_v1", "status": "COMPLETE", "pretraining_lock_sha256": "p", "readout_lock_sha256": "r",
                "jobs": [{"split": report.SPLITS[0], "family": "ATTN_MATCHED", "seed": 31, "lambda_pair": .1, "epoch": 400,
                          "receipt_file": path.name, "receipt_sha256": report.sha(path)}]}
    with pytest.raises(ValueError, match="EXACT72_READOUT"):
        report.collect_readouts(tmp_path, manifest, pretrain_sha="p", readout_sha="r")


def synthetic_full_receipts(tmp_path):
    """A full locked ledger with explicit NA heads, never synthetic performance."""
    pre_root, risk_root = tmp_path / "neural", tmp_path / "readout"
    pre_root.mkdir(); risk_root.mkdir()
    design = {"schema": "semantic_b0b7_p2s_design_precommit_v1", "D": {"primary_ridge_lambda": 1., "expected_total_Cox_fits": 2376},
              "A_C": {"new_optimizer_updates": 21600}}
    design_path = tmp_path / "design.json"; design_path.write_text(json.dumps(design), encoding="utf-8")
    lock = {"schema": "semantic_b0b7_p2s_lock_v1", "status": "LOCKED_BEFORE_P2S_OPTIMIZER", "parent_master_sha256": "o" * 64,
            "study_design": design, "frozen_sources": {"src/anonymous.py": "s" * 64,
                "configs/semantic_b0b7/p2s_design_precommit.json": report.sha(design_path)}, "cells": []}
    for split, family, seed, pair in itertools.product(report.SPLITS, report.FAMILIES, report.SEEDS, report.PAIR_WEIGHTS):
        lock["cells"].append({"split": split, "family": family, "seed": seed, "lambda_pair": pair,
            "input_sha256": "i" * 64, "parent_receipt_sha256": "p" * 64, "parent_state400_sha256": "c" * 64})
    lock_path = tmp_path / "prelock.json"; lock_path.write_text(json.dumps(lock), encoding="utf-8")
    pre_sha = report.sha(lock_path)
    child_path = tmp_path / "child.json"
    child_path.write_text(json.dumps({"schema": "semantic_b0b7_p2s_readout_lock_v1", "status": "LOCKED_BEFORE_P2S_DSS_FITS",
        "parent_study_lock_sha256": pre_sha, "expected_feature_states": 72, "expected_Cox_fits": 2376,
        "primary_ridge_lambda": 1., "ridge_candidates": [.1, 1., 10.], "outer_evaluation": False,
        "anchors": {s: {"sha256": "a" * 64, "receipt_sha256": "b" * 64} for s in report.SPLITS},
        "readout_sources": {"scripts/anonymous.py": "a" * 64}}), encoding="utf-8")
    child_sha = report.sha(child_path)
    program = {"patient_equal_masked_huber": .3, "macro_program_pearson": None, "valid_pearson_programs": 0,
               "patients": 3, "patients_with_measurement": 3, "measured_target_cells": 150, "coverage_fraction": 1.,
               "per_program": [{"program_index": i, "patients": 3, "pearson": None, "spearman": None} for i in range(50)]}
    validation = {"program": program, "retrieval_hard": {d: {"paired_gap_mean": None} for d in ("image_to_gene", "gene_to_image")}, "strata": {}}
    manifest = {"schema": "semantic_b0b7_p2s_readout_manifest_v1", "status": "COMPLETE", "pretraining_lock_sha256": pre_sha,
                "readout_lock_sha256": child_sha, "jobs": []}
    for number, cell in enumerate(lock["cells"]):
        state_sha = {e: ("%064x" % (number * 2 + (1 if e == 400 else 2))) for e in report.EPOCHS}
        step_count = 400 if cell["lambda_pair"] == .1 else 800
        history = [{"epoch": e, "optimizer_step": e, "train_program_loss": .3, "train_pair_loss": .2,
                    "train_auxiliary_loss": .3 + cell["lambda_pair"] * .2, "active_parameter_elements": 5,
                    "inactive_parameters": [], "gradient_norms": {"weight": .01}, "weight_norms": {"weight": .1}}
                   for e in range(801 - step_count, 801)]
        receipt = {**{k: cell[k] for k in ("split", "family", "seed")}, "schema": "semantic_b0b7_p2s_cell_v1", "status": "COMPLETE",
            "parent_cell": cell, "master_lock_sha256": pre_sha, "base_p2_master_lock_sha256": lock["parent_master_sha256"],
            "DSS_OS_PFI_accessed": False, "selected_epoch": None, "no_DSS_or_auxiliary_epoch_selection": True,
            "spec": {"lambda_pair": cell["lambda_pair"]}, "input_sha256": cell["input_sha256"], "frozen_sources": lock["frozen_sources"],
            "optimizer_steps": step_count, "optimizer_lifetime_steps": 800, "resume_epoch": 800 - step_count, "history": history,
            "states": [{"epoch": e, "actual_replay": True, "replay_max_error": 0., "replay_preserved_training_rng": True,
                        "validation": validation, "sha256": state_sha[e], "identity": "SYNTHETIC_UNIT_FIXTURE"} for e in report.EPOCHS],
            "elapsed_seconds": .1, "peak_cuda_allocated_bytes": 100, "parameter_counts": {"total": 5, "dss_trainable_in_this_module": 0}}
        folder = pre_root / "phase_a" / cell["split"] / (cell["family"] + "_seed" + str(cell["seed"]) + "_pair%03d" % round(cell["lambda_pair"] * 100))
        folder.mkdir(parents=True)
        (folder / "cell.private.json").write_text(json.dumps(receipt), encoding="utf-8")
        for epoch in report.EPOCHS:
            meta = {k: cell[k] for k in ("split", "family", "seed", "lambda_pair")}
            meta.update(epoch=epoch, input_sha256=cell["input_sha256"], checkpoint_sha256=state_sha[epoch])
            records = [{"spec": {"main_pca": d, "base_arm": arm, "factor_rank": rank, "mode": mode}, "ridge_lambda": ridge,
                        "primary_lambda": 1., "is_primary_lambda": ridge == 1., "status": report.STATUS_NA,
                        "NA_reason": "SYNTHETIC_INSUFFICIENT_RANK", "evaluation_metrics": None}
                       for (d, arm, rank, mode), ridge in itertools.product(report.specs(cell["family"]), report.RIDGES)]
            payload = {"schema": "semantic_b0b7_p2s_readout_state_v1", "status": "PARTIAL_PRIMARY_UNSUPPORTED_RANK", "meta": meta, "rows": records,
                "actual_Cox_fits": 0, "NA_count": len(records), "expected_fits": len(records), "actual_PCA_fits": 0, "actual_bgRidge": 0,
                "actual_Cox_fit_attempts": 0, "actual_saved_head_replays": 0,
                "readout_lock_sha256": child_sha, "study_lock_sha256": pre_sha, "no_selection": True,
                "all_lambdas_reported_without_selection": True, "primary_lambda": 1., "outer_test_accessed": False,
                "actual_new_teacher_or_scalar_fits": 0, "elapsed_seconds": .1, "new_neural_fits": 0, "new_model_forwards": 0,
                "feature_extraction_compute": {"frozen_encoder_feature_extractions": 1, "elapsed_seconds": .2},
                "anchor": {"artifact_sha256": "a" * 64, "receipt_sha256": "b" * 64, "training_offset_sha256": "t" * 64,
                    "evaluation_offset_sha256": "e" * 64, "new_teacher_or_scalar_fits": 0,
                    "full_pipeline_crossfit_for_training_offset": True, "own_labels_excluded_from_teacher_and_beta": True,
                    "reused_original_teacher_Cox_fits": 52, "reused_original_scalar_fits": 4}}
            path = risk_root / ("job%02d_epoch%d.json" % (number, epoch))
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest["jobs"].append({**{k: meta[k] for k in ("split", "family", "seed", "lambda_pair", "epoch")},
                                     "receipt_file": path.name, "receipt_sha256": report.sha(path)})
    manifest_path = tmp_path / "manifest.json"; manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return pre_root, risk_root, lock_path, pre_sha, child_path, child_sha, design_path, report.sha(design_path), manifest_path


def test_full_synthetic_receipt_delivery_keeps_all2376_na_cells_and_all21600_new_updates(tmp_path):
    args = synthetic_full_receipts(tmp_path)
    output = tmp_path / "public"
    result = report.produce(*args, output)
    assert result["status"] == "COMPLETE_FIXED_GRID_WITH_EXPLICIT_NA"
    assert result["actual_Cox_fits"] == 0 and result["explicit_NA_head_cells"] == 2376
    costs = json.loads((output / "08_COMPUTE_COST.json").read_bytes())
    assert costs["new_optimizer_updates"] == 21600 and costs["reused_old_optimizer_updates_not_new"] == 7200
    assert costs["reporter_optimizer_fits"] == costs["reporter_model_forwards"] == 0
    index = json.loads((output / "INDEX.json").read_bytes())
    assert index["actual_pretraining_readout_state_crosscheck"]["states_compared"] == 72
    for item in index["files"]:
        path = output / item["file"]
        assert report.sha(path) == item["sha256"] and path.stat().st_size == item["bytes"]
        assert path.stat().st_size <= report.MAX_BYTES
        if path.suffix == ".csv":
            assert path.read_bytes().count(b"\n") <= report.MAX_CSV_LINES
        if path.suffix == ".json":
            report.public_value(json.loads(path.read_bytes()))
    for item in output.glob("05_seed_micro_aggregates*.json"):
        assert all(row["Harrell"] is None and row["Uno60"] is None for row in json.loads(item.read_bytes()))


def test_crosscheck_different_checkpoint_or_input_fails_without_opening_model():
    pre = {"provenance": [], "checkpoints": []}; risk = {"provenance": []}
    for split, family, seed, pair in itertools.product(report.SPLITS, report.FAMILIES, report.SEEDS, report.PAIR_WEIGHTS):
        meta = dict(split=split, family=family, seed=seed, lambda_pair=pair)
        pre["provenance"].append({**meta, "input_sha256": "i" * 64})
        for epoch in report.EPOCHS:
            pre["checkpoints"].append({**meta, "epoch": epoch, "checkpoint_sha256": "c" * 64})
            risk["provenance"].append({**meta, "epoch": epoch, "checkpoint_sha256": "c" * 64, "input_sha256": "i" * 64})
    assert report.crosscheck_state_sources(pre, risk)["states_compared"] == 72
    risk["provenance"][0]["checkpoint_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="CHECKPOINT_OR_INPUT_DIFFERS"):
        report.crosscheck_state_sources(pre, risk)


@pytest.mark.parametrize("value", [{"patient_ids": ["anonymous"]}, {"weights": [1., 2.]}, {"coordinates": []}])
def test_original_scanner_row_level_field_boundary_is_preserved(value):
    with pytest.raises(ValueError):
        report.public_value(value)


def test_immutable_child_cannot_rebind_old_scientific_source_or_select_new_primary():
    design = {"fixed": True}
    study = {"schema": "semantic_b0b7_p2s_lock_v1", "status": "LOCKED_BEFORE_P2S_OPTIMIZER", "study_design": design,
             "frozen_sources": {"old_source.py": "a" * 64, "configs/semantic_b0b7/p2s_design_precommit.json": "d" * 64}}
    child = {"schema": "semantic_b0b7_p2s_readout_lock_v1", "status": "LOCKED_BEFORE_P2S_DSS_FITS", "parent_study_lock_sha256": "s" * 64,
             "expected_feature_states": 72, "expected_Cox_fits": 2376, "primary_ridge_lambda": 1., "ridge_candidates": [.1, 1., 10.],
             "outer_evaluation": False, "anchors": {s: {} for s in report.SPLITS}, "readout_sources": {}}
    report.validate_report_locks(study, child, design, study_sha="s" * 64, design_sha="d" * 64)
    changed = deepcopy(child); changed["readout_sources"]["old_source.py"] = "b" * 64
    with pytest.raises(ValueError, match="MUST_NOT_REBIND"):
        report.validate_report_locks(study, changed, design, study_sha="s" * 64, design_sha="d" * 64)
    changed = deepcopy(child); changed["primary_ridge_lambda"] = 10.
    with pytest.raises(ValueError, match="PRIMARY_PROTOCOL_CHANGED"):
        report.validate_report_locks(study, changed, design, study_sha="s" * 64, design_sha="d" * 64)


def test_one_real_status_scalar_row_and_saved_replay_are_kept_among_explicit_na_cells(tmp_path):
    args = synthetic_full_receipts(tmp_path)
    manifest = report.read_json(args[-1]); job = manifest["jobs"][0]
    path = args[1] / job["receipt_file"]; payload = report.read_json(path)
    row = payload["rows"][0]
    row.update(status=report.STATUS_FIT, NA_reason=None, DSS_coefficients=24,
        train_metrics=metric(.6), evaluation_metrics=metric(.4), anchor_evaluation_metrics=metric(.45),
        risk_optimizer={"validation_used_for_optimizer_or_lambda": False, "lambda_selected_using_validation": False,
            "gradient_max": 1e-8, "optimized_coefficients": 24,
            "training_loss_curve": [{"iteration": 1, "objective": .2, "gradient_max": .1}]},
        design={"nominal_coefficients": 24, "zero_padding_columns": 0},
        saved_head_replay={"actual_saved_head_replay": True, "training_risk_byte_hash_equal": True,
            "evaluation_risk_byte_hash_equal": True, "scorer_components_exact": True, "max_abs_risk_error": 0.,
            "neural_forward": False, "new_optimizer_fits": 0})
    payload.update(actual_Cox_fits=1, actual_Cox_fit_attempts=1, actual_saved_head_replays=1, NA_count=payload["NA_count"] - 1)
    path.write_text(json.dumps(payload), encoding="utf-8"); job["receipt_sha256"] = report.sha(path)
    result = report.collect_readouts(args[1], manifest, pretrain_sha=args[3], readout_sha=args[5])
    assert sum(r["status"] == report.STATUS_FIT for r in result["rows"]) == 1
    assert len(result["rows"]) == 2376 and len(result["traces"]) == 1
    out = next(r for r in result["flat_rows"] if r["status"] == report.STATUS_FIT)
    assert out["evaluation_metrics.Harrell"] == .4 and out["DSS_coefficients"] == 24
    assert out["saved_head_replay.scorer_components_exact"] is True
    row["saved_head_replay"]["scorer_components_exact"] = False
    path.write_text(json.dumps(payload), encoding="utf-8"); job["receipt_sha256"] = report.sha(path)
    with pytest.raises(ValueError, match="REAL_SAVED_HEAD_REPLAY_NOT_QUALIFIED"):
        report.collect_readouts(args[1], manifest, pretrain_sha=args[3], readout_sha=args[5])


def test_same_partition_anchor_dependency_intervention_cannot_change_shared_r0():
    child = {"anchors": {s: {"sha256": "a" * 64, "receipt_sha256": "b" * 64} for s in report.SPLITS}}
    anchor = {"artifact_sha256": "a" * 64, "receipt_sha256": "b" * 64, "training_offset_sha256": "c" * 64,
              "evaluation_offset_sha256": "d" * 64, "new_teacher_or_scalar_fits": 0,
              "full_pipeline_crossfit_for_training_offset": True, "own_labels_excluded_from_teacher_and_beta": True,
              "reused_original_teacher_Cox_fits": 52, "reused_original_scalar_fits": 4}
    diagnostics = [{"split": s, "anchor": deepcopy(anchor)} for s in report.SPLITS for _ in range(24)]
    costs = report.crosscheck_shared_anchors(diagnostics, child)
    assert costs["original_teacher_Cox_fits_referenced_not_new"] == 156 and costs["new_teacher_or_scalar_fits"] == 0
    diagnostics[-1]["anchor"]["evaluation_offset_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="R0_NOT_EXACTLY_SHARED"):
        report.crosscheck_shared_anchors(diagnostics, child)
