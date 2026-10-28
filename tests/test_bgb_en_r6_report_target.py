"""Anonymous synthetic target receipts; no real outcomes/models/fits."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bgb_en_r6_report_target.py"
SPEC = importlib.util.spec_from_file_location("bgb_en_r6_report_target_tests", SCRIPT)
r = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(r)
CANARY = "PRIVATE_PATIENT_AND_RAW_RISK_MUST_NOT_EXPORT"
POINTS = dict(B0=(.70, .68), B1=(.72, .70), B2=(.71, .69), B3=(.73, .71), B4=(.74, .72),
    B5=(.75, .73), B6=(.76, .74), B7=(.80, .78), B8=(.73, .71), B9=(.74, .72),
    B11=(.78, .77), B13=(.72, .70))


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def metric(arm, partition, seed, *, secondary=False):
    a, b = POINTS[arm]
    offset = [0., .01, -.01][partition]
    seed_offset = [0., .005, -.005][r.SEEDS.index(seed)] if arm not in r.BASELINES else 0.
    if secondary:
        a, b = (.25, .22) if arm == "B7" else (.8, .77)
    a, b = a + offset + seed_offset, b + offset + seed_offset
    hd, ud = [10., 100., 1000.][partition], [20., 30., 60.][partition]
    out = dict(Harrell=a, Uno60=b, Harrell_num=a * hd, Harrell_den=hd, Uno_num=b * ud, Uno_den=ud)
    if not secondary:
        ba, bb = POINTS["B1"]
        dh, du = a - ba - offset, b - bb - offset
        out.update(repair=.1 + max(dh, 0), damage=.1 + max(-dh, 0),
            Uno_repair=.1 + max(du, 0), Uno_damage=.1 + max(-du, 0), risk_shift_sd_ratio=.05)
    return out


def metadata_fixture(tmp_path, *, complete=False):
    private = tmp_path / "private"; private.mkdir()
    config = json.loads((r.ROOT / "configs/bgb_en/r6_inner_v1.json").read_text(encoding="utf-8"))
    config_path = tmp_path / "target_config.json"; put(config_path, config)
    full = {control: dict(checkpoint_sha256=h("weights" + control), completion_sha256=h("done" + control),
        replay_sha256=h("replay" + control), complete_source_state_sha256=h("source" + control),
        source_config_sha256=h("sourceconfig"), source_control=control, external_full_source_seed=31)
        for control in r.FULL_CONTROLS}
    full["RANDOM_NO_ST"] = dict(complete_source_state_sha256=h("randomsource"), source_control="RANDOM_NO_ST",
        source_new_optimizer_fits=0, full_structure_freshly_initialized=True)
    lock = dict(schema="bgb_en_r6_target_protocol_lock_v1", status="LOCKED_BEFORE_TARGET_FITS",
        config_sha256=r.sha256(config_path), source_config_sha256=h("sourceconfig"), planned_neural_fits=90,
        teacher_refits=0, outer_test_access=False, source_full_states=full,
        fixed_deterministic_runtime=dict(deterministic_algorithms=True),
        prepared_receipt_sha256={f"outer1_inner{p}": h("prepared" + str(p)) for p in r.PARTITIONS},
        sources={"scripts/bgb_en_r6_report_target.py": r.sha256(SCRIPT)})
    lock["contract_sha256"] = r.canonical_hash(lock)
    put(private / "targetLOCK.json", lock)
    upstream = dict(status="PASS_R6_39_COEFFICIENTS_3_BETA_PREPROCESSING_REPLAY", contract_sha256=lock["contract_sha256"],
        new_optimizer_fits=0, new_preprocessing_fits=0,
        prepared_receipt_sha256={str(p): lock["prepared_receipt_sha256"][f"outer1_inner{p}"] for p in r.PARTITIONS},
        partitions=[dict(partition=p, status="PASS", coefficient_models=13, beta_gradient_abs=0.) for p in r.PARTITIONS])
    put(private / "UPSTREAM_REPLAY_AUDIT.json", upstream)
    qual = dict(status="PASS_R6_TARGET_REAL_DATA_AND_SOURCE_TRANSFER", contract_sha256=lock["contract_sha256"],
        new_optimizer_updates=0, new_neural_fits=0, source_teacher_or_scalar_fits=0, real_data_backward_passes=2,
        query_only_main_and_offset_exact=True, missing_gene_gradients_exact_zero=True,
        source_parameters_frozen=True, original_gene_values_in_dynamic_reread=True,
        shared_target_initialization_sha256=h("init_p0_s31"), upstream_replay_audit_sha256=r.sha256(private / "UPSTREAM_REPLAY_AUDIT.json"))
    put(private / "QUALIFICATION.json", qual)
    decision = dict(status="SUPPORTED", source_mechanism_only=True)
    source_decision = tmp_path / "source_decision.json"; put(source_decision, decision)
    source_report = tmp_path / "source_report.json"
    put(source_report, dict(status="COMPLETE_147_SOURCE_UNITS_AND_REPLAY", source_config_sha256=h("sourceconfig"),
        decision=decision, counts=dict(PASS_COMPLETE_AND_REPLAYED=147), confirmed_formal_optimizer_fits=135,
        confirmed_formal_optimizer_updates=10800, unknown_update_cost_units=0,
        full_source_transfer_units=[dict(arm="S7", control=control, heldout_donor=-1, seed=31,
            status="PASS_COMPLETE_AND_REPLAYED", receipt_sha256=full[control]["completion_sha256"],
            replay_sha256=full[control]["replay_sha256"]) for control in r.FULL_CONTROLS],
        private_source_rows=CANARY))
    if complete:
        write_complete_matrix(private, config, lock)
    return config_path, private, source_decision, source_report, lock


def origin(lock, partition):
    return dict(prepared_receipt_sha256=lock["prepared_receipt_sha256"][f"outer1_inner{partition}"],
        train_n=300, valid_n=100, teachers_reused=3, teacher_refits=0, upstream_supervised_fits_reused_not_new=14)


def cell_contract(private, config, lock, arm, partition, seed):
    return dict(contract_sha256=lock["contract_sha256"], prepared_origin=origin(lock, partition), arm=arm,
        partition=partition, seed=seed, qualification_sha256=r.sha256(private / "QUALIFICATION.json"),
        config_sha256=lock["config_sha256"], source_config_sha256=lock["source_config_sha256"],
        fixed_deterministic_runtime=lock["fixed_deterministic_runtime"], epochs=80,
        teacher_refits=0, source_training_updates=0, outer_test_used=False)


def write_complete_matrix(private, config, lock):
    reference_replays, replayed = [], []
    for arm in r.BASELINES + r.ARMS:
        for partition in r.PARTITIONS:
            for seed in r.SEEDS:
                score = metric(arm, partition, seed)
                all_metrics = dict(CP_SCALAR=score, CP_STRONG=deepcopy(score),
                    PFI_frozen_DSS_risk=metric(arm, partition, seed, secondary=True))
                name = r.unit_name(arm, partition, seed)
                if arm in r.BASELINES:
                    path = private / "references" / (name + ".json")
                    value = dict(status="R6_BASELINE_REFERENCE", group="REFERENCE", arm=arm, partition=partition, seed=seed,
                        prepared_origin=origin(lock, partition), contract_sha256=lock["contract_sha256"],
                        actual_new_optimizer_fits=0, teacher_refits=0, deterministic_seed_reference=seed != 31,
                        metrics=score, metrics_vs=all_metrics)
                    put(path, value)
                    reference_replays.append(dict(arm=arm, partition=partition, seed=seed, status="PASS", receipt_sha256=r.sha256(path),
                        independent_fixed_anchor_metrics_recomputed=True, new_optimizer_fits=0))
                else:
                    directory = private / "cells" / name; directory.mkdir(parents=True)
                    artifact = directory / "model.pt"; artifact.write_bytes(CANARY.encode())
                    source_key = "RANDOM_NO_ST" if arm in ("B2", "B13") else "WRONG_DONOR" if arm == "B8" else "MATCHED"
                    value = dict(status="COMPLETE_R6_TARGET_FIT", contract=cell_contract(private, config, lock, arm, partition, seed),
                        arm=arm, partition=partition, seed=seed, epochs=80, optimizer_updates=80, actual_new_optimizer_fits=1,
                        teacher_refits=0, source_training_updates=0, source_origin=dict(source_key=source_key,
                            source_origin=lock["source_full_states"][source_key]),
                        initialization=dict(target_state_sha256=h(f"init_p{partition}_s{seed}")),
                        parameters=dict(registered=1000, requires_grad=100), metrics=score, metrics_vs=all_metrics,
                        curves=[dict(epoch=epoch, optimizer_updates_completed=epoch, cox_loss=1 / epoch, gradient_norm=.1,
                            risk_cache_replay_max=0., cross_cache_replay_max=0., train_SD_delta_over_offset=.05,
                            valid_SD_delta_over_offset=.04) for epoch in (20, 40, 80)],
                        artifact_sha256=r.sha256(artifact), elapsed_seconds=1., peak_GPU_bytes=1024,
                        outer_test_used=False, primary_epoch_locked_before_training=80,
                        private_patient_rows=CANARY)
                    put(directory / "COMPLETE.json", value)
                    replayed.append(dict(unit=name, status="PASS", arm=arm, partition=partition, seed=seed,
                        contract_sha256=lock["contract_sha256"], completion_sha256=r.sha256(directory / "COMPLETE.json"),
                        artifact_sha256=value["artifact_sha256"], qualification_sha256=r.sha256(private / "QUALIFICATION.json"),
                        upstream_replay_audit_sha256=r.sha256(private / "UPSTREAM_REPLAY_AUDIT.json"),
                        actual_state_dict_forward_replayed=True, source_frozen_weight_hash_replayed=True,
                        saved_query_mapping_recomputed=True, changed_pair_credits=0, maximum_risk_error=0., new_optimizer_updates=0))
    paired, genes = [], []
    for partition in r.PARTITIONS:
        for seed in r.SEEDS:
            for candidate, reference in [("B7", ref) for ref in r.PAIR_REFERENCES] + [("B6", "B5")]:
                a, b = metric(candidate, partition, seed), metric(reference, partition, seed)
                dh, du = a["Harrell"] - b["Harrell"], a["Uno60"] - b["Uno60"]
                actual = dict(a, repair=.12 + max(dh, 0), damage=.12 + max(-dh, 0),
                    Uno_repair=.08 + max(du, 0), Uno_damage=.08 + max(-du, 0))
                row = dict(candidate=candidate, reference=reference, partition=partition, seed=seed,
                    candidate_vs_reference=actual, risk_SD_ratio=.05, direct_risk_pair_credits=True, bootstrap_or_CI_computed=False)
                (genes if candidate == "B6" else paired).append(row)
    put(private / "REPLAY_AUDIT.json", dict(status="PASS_COMPLETE_R6_TARGET_CHECKPOINT_REPLAY",
        contract_sha256=lock["contract_sha256"], qualification_sha256=r.sha256(private / "QUALIFICATION.json"),
        expected_units=90, available_units=90, passed_units=90, failed_units=0, new_optimizer_fits=0,
        new_preprocessing_fits=0, numerical_risk_tolerance=.000002, maximum_allowed_changed_pair_credits=0,
        upstream_replay_audit_sha256=r.sha256(private / "UPSTREAM_REPLAY_AUDIT.json"), units=replayed,
        zero_fit_baseline_reference_replays=reference_replays, paired_retrained_controls=paired, paired_raw_gene_controls=genes))


def test_component_aggregation_does_not_average_fold_C_or_rank_cross_partition_risks():
    rows = [metric("B7", p, 31) for p in r.PARTITIONS]
    actual = r.combine_components(rows)
    assert actual["Harrell"] == pytest.approx(sum(row["Harrell_num"] for row in rows) / 1110)
    assert actual["Harrell"] != pytest.approx(sum(row["Harrell"] for row in rows) / 3)
    with pytest.raises(ValueError, match="THREE_PARTITIONS"):
        r.combine_components(rows[:2])


def test_all_nine_cells_are_required_for_a_three_seed_mean():
    mapping = {("B7", p, s): metric("B7", p, s) for p in r.PARTITIONS for s in r.SEEDS}
    result = r.aggregate_arm("B7", mapping)
    assert result["complete_cells"] == 9 and len(result["per_seed"]) == 3
    mapping.pop(("B7", 2, 83))
    partial = r.aggregate_arm("B7", mapping)
    assert partial["complete_cells"] == 8 and partial["Harrell"] is None and not partial["per_seed"]


def test_complete_90_plus_18_report_requires_optional_and_never_exports_private_values(tmp_path):
    config, private, source_decision, source_report, _ = metadata_fixture(tmp_path, complete=True)
    report, folds, pairs = r.build_report(config, private, source_decision, source_report)
    assert report["confirmed_new_target_fits"] == 90 and report["confirmed_target_updates"] == 7200
    assert report["unknown_cost_units"] == 0 and len(folds) == 216 and len(pairs) == 90
    assert report["decision"]["decision"] == "INNER_GATE_PASSED_OPTIONAL_PENDING"
    assert all(value is True for value in report["decision"]["core_flags"].values())
    assert not report["decision"]["outer_start_allowed"] and report["outer_results"]["status"] == "NOT_RUN"
    assert report["decision"]["optional_followup"] == dict(B10="PENDING", B12="PENDING", B14="PENDING", B15="TEXT_BLOCKED")
    b7 = next(row for row in report["primary_DSS"] if row["arm"] == "B7")
    pfi = next(row for row in report["secondary_PFI_frozen_DSS_risk"] if row["arm"] == "B7")
    assert b7["Harrell"] > .78 and pfi["Harrell"] < .3
    credit = report["paired_comparisons"]["B7_minus_B1"]
    assert credit["direct_repair"] > .12 and credit["direct_damage"] == pytest.approx(.12)
    assert credit["direct_repair"] - credit["direct_damage"] == pytest.approx(credit["Harrell_delta"])
    output = tmp_path / "public"; output.mkdir()
    put(output / "SOURCE_AUDIT.json", dict(r6_source_supervision=dict(preserved=True)))
    r.write_report(report, folds, pairs, output)
    assert r.read(output / "SOURCE_AUDIT.json")["r6_source_supervision"]["preserved"]
    for path in output.iterdir():
        text = path.read_text(encoding="utf-8-sig")
        assert CANARY not in text and str(private) not in text and str(tmp_path) not in text
    assert "NOT_RUN" in (output / "R6_5_OUTER_RESULTS.csv").read_text(encoding="utf-8-sig")
    # Execute the scheduler's actual reporting CLI and its implicit source
    # filename resolution, not just the report's Python functions.
    (source_decision.parent / "R6_1_SOURCE_SUPERVISION.json").write_bytes(source_report.read_bytes())
    completed = subprocess.run([sys.executable, str(SCRIPT), "--private-root", str(private),
        "--config", str(config), "--source-decision", str(source_decision), "--output", str(output)],
        cwd=r.ROOT, text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["outer"] == "NOT_RUN"
    assert CANARY not in completed.stdout


def test_missing_or_error_units_are_not_fake_zero_cost_or_partial_means(tmp_path):
    config, private, source_decision, source_report, lock = metadata_fixture(tmp_path)
    contract = cell_contract(private, r.read(config), lock, "B7", 0, 31)
    put(private / "cells/B7_p0_seed31/ERROR_RECEIPT.json", dict(contract=contract,
        optimizer_updates_completed=7, error_type="RuntimeError", private_error=CANARY))
    report, _, pairs = r.build_report(config, private, source_decision, source_report)
    assert report["decision"]["decision"] == "INCONCLUSIVE" and not pairs
    assert report["confirmed_target_updates"] == 7 and report["unknown_cost_units"] == 107
    assert all(row["Harrell"] is None for row in report["primary_DSS"])
    assert CANARY not in json.dumps(report)


def test_direct_credit_cannot_be_fabricated_from_concordance_differences():
    a, b = metric("B7", 0, 31), metric("B1", 0, 31)
    replay = dict(complete=True, direct_rows=[dict(candidate="B7", reference="B1", partition=0, seed=31,
        candidate_vs_reference={key: a[key] for key in ("Harrell", "Uno60") + r.COMPONENTS},
        risk_SD_ratio=.1, direct_risk_pair_credits=True, bootstrap_or_CI_computed=False)], raw_gene_rows=[])
    indexed, public, audit = r.direct_pairs(replay, {("B7", 0, 31): a, ("B1", 0, 31): b})
    assert not indexed and not public and audit["validation_failures"] == ["TARGET_DIRECT_REPAIR_DAMAGE_MISSING"]
    assert a["Harrell"] > b["Harrell"]
    replay["complete"] = False
    assert not r.direct_pairs(replay, {})[0]


def decision_inputs():
    config = json.loads((r.ROOT / "configs/bgb_en/r6_inner_v1.json").read_text(encoding="utf-8"))
    source = dict(local_support=True, complete_source_and_full_transfer=True)
    qualification, replay, paired = dict(passed=True), dict(complete=True), dict(complete=True)
    ledger = [dict(status="PASS_FIT_AND_ACTUAL_REPLAY") for _ in range(90)]
    ledger += [dict(status="PASS_ZERO_FIT_REFERENCE") for _ in range(18)]
    comparisons = {f"B7_minus_{reference}": dict(complete=True, Harrell_delta=.01, Uno60_delta=.01) for reference in r.PAIR_REFERENCES}
    comparisons["B6_minus_B5"] = dict(complete=True, Harrell_delta=.005, Uno60_delta=0.)
    return config, source, qualification, replay, ledger, comparisons, paired


def test_source_negative_is_scoped_not_an_all_idea_no_go():
    args = decision_inputs(); args[1]["local_support"] = False
    decision = r.decide(*args)
    assert decision["decision"] == "NOT_SUPPORTED"
    assert decision["core_flags"]["ST_LOCAL_EVIDENCE_SUPPORTED"] is False
    assert not decision["all_medical_ideas_rejected"] and not decision["outer_start_allowed"]


def test_true_gene_reread_flag_uses_root_locked_or_then_reverse_rule():
    args = decision_inputs()
    args[5]["B6_minus_B5"] = dict(complete=True, Harrell_delta=-.01, Uno60_delta=-.01)
    assert r.decide(*args)["core_flags"]["TRUE_GENE_REREAD_WORKS"] is True
    args[5]["B7_minus_B11"] = dict(complete=True, Harrell_delta=0., Uno60_delta=0.)
    assert r.decide(*args)["core_flags"]["TRUE_GENE_REREAD_WORKS"] is False
    args[5]["B7_minus_B11"] = dict(complete=False, Harrell_delta=None, Uno60_delta=None)
    assert r.decide(*args)["core_flags"]["TRUE_GENE_REREAD_WORKS"] is None


def test_optional_not_activated_and_text_blocked_are_not_fit_failures():
    args = decision_inputs()
    args[5]["B7_minus_B5"] = dict(complete=True, Harrell_delta=0., Uno60_delta=.01)
    decision = r.decide(*args)
    assert decision["optional_followup"]["B10"] == "NOT_ACTIVATED"
    assert decision["optional_followup"]["B15"] == "TEXT_BLOCKED"
    assert decision["formal_target_matrix_denominator"] == 90 and decision["optional_not_counted_as_failure"]


def test_source_incomplete_and_undefined_primary_metric_stay_inconclusive():
    args = decision_inputs(); args[1]["complete_source_and_full_transfer"] = False
    assert r.decide(*args)["decision"] == "INCONCLUSIVE"
    args[1]["complete_source_and_full_transfer"] = True
    args[5]["B7_minus_B1"]["Uno60_delta"] = None
    assert r.decide(*args)["decision"] == "INCONCLUSIVE"


def test_lock_digest_and_actual_reference_identity_fail_closed(tmp_path):
    config, private, _, _, lock = metadata_fixture(tmp_path)
    changed = r.read(private / "targetLOCK.json"); changed["planned_neural_fits"] = 89
    put(private / "targetLOCK.json", changed)
    with pytest.raises(ValueError, match="LOCK_DIGEST"):
        r.validate_inputs(config, private / "targetLOCK.json")
    put(private / "targetLOCK.json", lock)
    path = private / "references/B1_p0_seed31.json"
    put(path, dict(status="R6_BASELINE_REFERENCE", group="REFERENCE", arm="B1", partition=0, seed=999))
    row, _, _ = r.read_reference(private, "B1", 0, 31, lock, dict(baseline_units={}))
    assert row["status"] == "INVALID"


def test_no_model_array_previous_phase_or_patient_data_imports():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in text and "import numpy" not in text and "import h5py" not in text
    assert "torch.load" not in text and "np.load" not in text
    assert "from scripts.bgb_en_r5" not in text and "from src.bgb_en.r5" not in text


def test_frozen_diagnostics_are_scalar_allowlisted_and_not_claimed_retraining_or_histology():
    value = dict(frozen_controls=dict(wrong_query=dict(risk_SD_ratio=.02, frozen_not_retrained=True,
        whole_modality_replaced=False, private_patient_values=CANARY)),
        attention_probe=dict(patients=8, patch_attention_TV=.03, risk_difference_SD=.01,
            original_values_and_main_effects_preserved=True, frozen_not_retrained=True,
            histology_or_gene_causality_claimed=False, private_attention_weights=CANARY),
        risk_diagnostics=dict(validation=dict(SD_cross_delta_over_offset=.02, private_risks=CANARY)))
    result = r.scalar_diagnostics(value)
    assert result["frozen_is_dependency_not_retraining"] and result["attention_is_not_histological_truth"]
    assert result["frozen_interventions"]["wrong_query"]["risk_SD_ratio"] == .02
    assert CANARY not in json.dumps(result)


def test_actual_replay_failure_keeps_cost_but_removes_complete_model_mean_and_direct_credit(tmp_path):
    config, private, source_decision, source_report, _ = metadata_fixture(tmp_path, complete=True)
    path = private / "REPLAY_AUDIT.json"
    replay = r.read(path)
    row = next(v for v in replay["units"] if (v["arm"], v["partition"], v["seed"]) == ("B7", 0, 31))
    row["status"] = "FAIL"
    replay.update(status="FAILED_R6_TARGET_CHECKPOINT_REPLAY", passed_units=89, failed_units=1)
    put(path, replay)
    report, _, pairs = r.build_report(config, private, source_decision, source_report)
    assert report["confirmed_target_updates"] == 7200
    assert report["decision"]["decision"] == "INCONCLUSIVE" and not pairs
    b7 = next(v for v in report["primary_DSS"] if v["arm"] == "B7")
    assert b7["complete_cells"] == 8 and b7["Harrell"] is None
    assert report["counts"]["ACTUAL_REPLAY_FAILED"] == 1
