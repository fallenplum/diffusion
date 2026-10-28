"""Metadata-only terminal accounting guards; synthetic receipts, no training."""
from __future__ import annotations

import csv
import copy
import json
from pathlib import Path

import pytest

from scripts import bgb_en_r6_execution_ledger as ledger


D = "a" * 64


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return ledger.sha(path)


def seal(value):
    return dict(value, contract_sha256=ledger.canonical(value))


@pytest.fixture
def complete_metadata(tmp_path):
    source, target, retention, reports = [tmp_path / n for n in ("source", "target", "retention", "reports")]
    srows = []
    for arm, control, donor, seed in sorted(ledger.source_keys()):
        directory = source / "source_cells" / f"{arm}_{control}_h{donor}_seed{seed}"
        cell = dict(arm=arm, control=control, heldout_donor=donor, seed=seed)
        fits, updates = (0, 0) if arm == "S0" else (1, 80)
        done = dict(cell=cell, config_sha256=D, source_code_sha256={"src/frozen.py": D},
            status="COMPLETE_R6_RANDOM_REFERENCE_NO_OPTIMIZER" if arm == "S0" else "COMPLETE_R6_SOURCE_FIT",
            epochs=80, optimizer_updates=updates, actual_new_optimizer_fits=fits,
            artifact_sha256={"state_epoch080.private.pt": D}, elapsed_seconds=1.)
        ds = write(directory / "COMPLETE.json", done)
        replay = dict(cell=cell, config_sha256=D, source_code_sha256=done["source_code_sha256"],
            status="PASS_COMPLETE_R6_SOURCE_CHECKPOINT_REPLAY", completion_sha256=ds,
            optimizer_updates=0, new_optimizer_fits=0, model_frozen=True,
            image_only_heldout_program_prediction=True, original_fit_artifacts_modified=False,
            forward_max_abs_errors={split: {"output": 1e-8} for split in (["train"] if donor < 0 else ["train", "heldout"])})
        rs = write(directory / "REPLAY.json", replay)
        srows.append(dict(cell, status="PASS_COMPLETE_AND_REPLAYED", metadata_validated=True,
            independent_replay_passed=True, actual_optimizer_fits=fits, actual_optimizer_updates=updates,
            receipt_sha256=ds, replay_sha256=rs, verified_artifact_files=1,
            artifact_manifest_sha256=ledger.canonical(done["artifact_sha256"]), elapsed_seconds=1.))
    sq = dict(schema="bgb_en_r6_real_source_compute_qualification_v1", status="PASS_TWO_REAL_SOURCE_GPU_UPDATES",
        config_sha256=D, qualification_updates=2, separate_qualification_optimizer_instances=1,
        formal_source_fits=0, saved_qualification_weights=False, TCGA_DSS_or_PFI_read=False,
        steps=[dict(qualification_update=i) for i in (1, 2)], elapsed_seconds=1.)
    sqsha = write(source / "REAL_SOURCE_COMPUTE_QUALIFICATION.private.json", sq)
    write(source / "SOURCE_SCHEDULE_COMPLETE.private.json", dict(status="COMPLETE_147_SOURCE_UNITS_AND_ACTUAL_REPLAY",
        source_config_sha256=D, planned_logical_units=147, planned_optimized_units=135, elapsed_seconds=200.,
        records=[dict(cell_index=i, status="FIT_AND_ACTUAL_REPLAY_COMPLETE") for i in range(147)]))
    sr = dict(schema="bgb_en_r6_source_supervision_report_v1", status="COMPLETE_147_SOURCE_UNITS_AND_REPLAY",
        source_config_sha256=D, source_code_sha256={"src/frozen.py": D}, counts={"PASS_COMPLETE_AND_REPLAYED": 147},
        ledger=srows, planned_logical_units=147, planned_formal_optimizer_fits=135,
        planned_formal_optimizer_updates=10800, planned_random_references=12, confirmed_formal_optimizer_fits=135,
        confirmed_formal_optimizer_updates=10800, unknown_fit_cost_units=0, unknown_update_cost_units=0,
        qualification_cost_separate_from_formal_matrix=dict(provided=True, receipt_sha256=sqsha,
            optimizer_instances=1, optimizer_updates=2, formal_source_fits=0))
    srsha = write(reports / "R6_1_SOURCE_SUPERVISION.json", sr)
    shared_states = {row["control"]: dict(checkpoint_sha256=D,
        completion_sha256=row["receipt_sha256"], replay_sha256=row["replay_sha256"],
        complete_source_state_sha256=D, source_config_sha256=D,
        source_control=row["control"], external_full_source_seed=31)
        for row in srows if row["heldout_donor"] == -1}
    tl = seal(dict(schema="bgb_en_r6_target_protocol_lock_v1", status="LOCKED_BEFORE_TARGET_FITS",
        config_sha256=D, source_config_sha256=D, sources={"scripts/bgb_en_r6_qualify_target.py":
            ledger.sha(ledger.ROOT / "scripts/bgb_en_r6_qualify_target.py")},
        teacher_refits=0, planned_neural_fits=90, outer_test_access=False,
        fixed_deterministic_runtime={"deterministic": True},
        source_full_states=shared_states, data_fingerprints=dict(cohort_sha256=D, endpoints_sha256=D),
        launch_gate_sha256=D,
        prepared_receipt_sha256={f"outer1_inner{p}": D for p in ledger.PARTITIONS}))
    tls = write(target / "targetLOCK.json", tl)
    up = dict(status="PASS_R6_39_COEFFICIENTS_3_BETA_PREPROCESSING_REPLAY", contract_sha256=tl["contract_sha256"],
        prepared_receipt_sha256={str(p): D for p in ledger.PARTITIONS}, new_optimizer_fits=0, new_preprocessing_fits=0,
        partitions=[dict(partition=p, status="PASS", coefficient_models=13, beta_gradient_abs=1e-9) for p in ledger.PARTITIONS])
    ups = write(target / "UPSTREAM_REPLAY_AUDIT.json", up)
    tq = dict(status="PASS_R6_TARGET_REAL_DATA_AND_SOURCE_TRANSFER", contract_sha256=tl["contract_sha256"],
        real_data_backward_passes=2, upstream_replay_audit_sha256=ups, new_optimizer_updates=0,
        new_neural_fits=0, source_teacher_or_scalar_fits=0, query_only_main_and_offset_exact=True,
        missing_gene_gradients_exact_zero=True, source_parameters_frozen=True,
        original_gene_values_in_dynamic_reread=True)
    tqs = write(target / "QUALIFICATION.json", tq)
    trows, replays, refs = [], [], []
    for arm, p, seed in sorted(ledger.target_keys() | ledger.target_keys(("B0", "B1"))):
        key = (arm, p, seed); name = ledger.unit_name(key)
        identity = dict(arm=arm, partition=p, seed=seed)
        c = dict(identity, contract_sha256=tl["contract_sha256"], config_sha256=D, source_config_sha256=D,
            qualification_sha256=tqs, prepared_origin=dict(prepared_receipt_sha256=D),
            fixed_deterministic_runtime=tl["fixed_deterministic_runtime"], epochs=80, teacher_refits=0,
            source_training_updates=0, outer_test_used=False)
        baseline = arm in ("B0", "B1")
        if baseline:
            done = dict(identity, group="REFERENCE", status="R6_BASELINE_REFERENCE", contract_sha256=tl["contract_sha256"],
                actual_new_optimizer_fits=0, teacher_refits=0, deterministic_seed_reference=seed != 31,
                prepared_origin=dict(prepared_receipt_sha256=D))
            ds = write(target / "references" / (name + ".json"), done)
            refs.append(dict(identity, status="PASS", receipt_sha256=ds,
                independent_fixed_anchor_metrics_recomputed=True, new_optimizer_fits=0))
            trows.append(dict(identity, group="REFERENCE", status="PASS_ZERO_FIT_REFERENCE", actual_new_optimizer_fits=0,
                actual_optimizer_updates=0, receipt_sha256=ds, actual_replay_passed=True))
        else:
            done = dict(identity, contract=c, status="COMPLETE_R6_TARGET_FIT", epochs=80, optimizer_updates=80,
                actual_new_optimizer_fits=1, primary_epoch_locked_before_training=80, outer_test_used=False,
                teacher_refits=0, source_training_updates=0, artifact_sha256=D, elapsed_seconds=1., peak_GPU_bytes=100)
            ds = write(target / "cells" / name / "COMPLETE.json", done)
            replay = dict(identity, unit=name, status="PASS", contract_sha256=tl["contract_sha256"],
                completion_sha256=ds, artifact_sha256=D, qualification_sha256=tqs, upstream_replay_audit_sha256=ups,
                actual_state_dict_forward_replayed=True, source_frozen_weight_hash_replayed=True,
                saved_query_mapping_recomputed=True, changed_pair_credits=0, maximum_risk_error=1e-8, new_optimizer_updates=0)
            replays.append(replay); write(target / "REPLAY_PRIVATE" / (name + ".PASS.json"), replay)
            trows.append(dict(identity, group="NEURAL", status="PASS_FIT_AND_ACTUAL_REPLAY", actual_new_optimizer_fits=1,
                actual_optimizer_updates=80, receipt_sha256=ds, actual_replay_passed=True, artifact_sha256=D,
                elapsed_seconds=1., peak_GPU_bytes=100))
    replay = dict(status="PASS_COMPLETE_R6_TARGET_CHECKPOINT_REPLAY", contract_sha256=tl["contract_sha256"],
        qualification_sha256=tqs, upstream_replay_audit_sha256=ups, expected_units=90, available_units=90,
        passed_units=90, failed_units=0, new_optimizer_fits=0, new_preprocessing_fits=0,
        missing_or_unreplayed_units=[], numerical_risk_tolerance=2e-6, maximum_allowed_changed_pair_credits=0,
        units=replays, zero_fit_baseline_reference_replays=refs)
    rps = write(target / "REPLAY_AUDIT.json", replay)
    tr = dict(schema="bgb_en_r6_target_inner_report_v1", contract_sha256=tl["contract_sha256"], target_lock_sha256=tls,
        config_sha256=D, target_sources_sha256=tl["sources"], source=dict(complete_source_and_full_transfer=True, report_sha256=srsha),
        qualification=dict(status="PASS", passed=True, sha256=tqs, real_data_backward_passes=2, optimizer_updates=0,
            neural_fits=0, upstream_replay_audit_sha256=ups), actual_replay=dict(status=replay["status"], complete=True, sha256=rps),
        ledger=trows, planned_new_target_fits=90, planned_baseline_reference_rows=18, planned_target_updates=7200,
        confirmed_new_target_fits=90, confirmed_target_updates=7200, unknown_cost_units=0,
        counts={"PASS_FIT_AND_ACTUAL_REPLAY": 90, "PASS_ZERO_FIT_REFERENCE": 18},
        decision=dict(complete_source_and_target_evidence=True, formal_target_matrix_denominator=90,
            optional_not_counted_as_failure=True, optional_followup=dict(B10="PENDING", B12="NOT_ACTIVATED",
                B14="NOT_ACTIVATED", B15="TEXT_BLOCKED")))
    trsha = write(reports / "R6_4_INNER_MATRIX.json", tr)
    labels = [ledger.unit_name(k) for k in sorted(ledger.target_keys())] + [f"references_p{p}" for p in ledger.PARTITIONS] + ["actual_checkpoint_replay"]
    write(target / "TARGET_SCHEDULE_COMPLETE.json", dict(status="COMPLETE_90_R6_TARGET_FITS_AND_REPLAY",
        contract_sha256=tl["contract_sha256"], new_unplanned_candidates=0,
        optional_controls_excluded_from_this_90_unit_denominator=True, worker_ceiling=4, elapsed_seconds=50.,
        records=[dict(unit=name, returncode=0) for name in labels]))
    entries = [dict(representation=r, partition=p, optimizer_representation=r, feature_values_sha256=D,
        measured_mask_sha256=D) for r in ledger.REPRESENTATIONS for p in ledger.PARTITIONS]
    old = [dict(representation=r, partition=p, status="REUSED_EXACT_PRIOR_ACTUAL_REPLAY",
        actual_optimizer_fits=0, historical_optimizer_fits_referenced_not_new=1) for r in ledger.OLD_RETENTION for p in ledger.PARTITIONS]
    rl = seal(dict(schema="bgb_en_r6_retention_lock_v1", status="LOCKED_BEFORE_ALL_RETENTION_FITS", config_sha256="b" * 64,
        source_config_sha256=D, representations=entries, historical_references=old, planned_logical_cells=18,
        source_full_states=shared_states, data_fingerprints=dict(cohort_sha256=D, endpoints_sha256=D), launch_gate_sha256=D,
        prepared_origins={f"outer1_inner{p}": dict(prepared_receipt_sha256=D, prepared_artifact_sha256=D) for p in ledger.PARTITIONS},
        planned_actual_optimizer_fits=18, new_teacher_or_PCA_fits=0))
    write(retention / "retentionLOCK.json", rl)
    rr = []
    for i, entry in enumerate(entries):
        r, p = entry["representation"], entry["partition"]
        c = dict(retention_contract_sha256=rl["contract_sha256"], **entry)
        it = 34 if i < 15 else 33  # total609; deliberately not the 80-update NN schedule.
        done = dict(status="COMPLETE_RETENTION_FIT", contract=c, actual_optimizer_fits=1, lambda_L2=1.,
            artifact_sha256=D, parameters=4, solver_iterations=it, solver=dict(iterations=it, elapsed_seconds=1.))
        directory = retention / "cells" / f"{r}_p{p}"
        ds = write(directory / "COMPLETE.json", done)
        rs = write(directory / "REPLAY.json", dict(contract=c, status="PASS_RETENTION_COEFFICIENT_AND_INDEPENDENT_EFRON_REPLAY",
            complete_sha256=ds, new_optimizer_fits=0, risk_max_abs_error=1e-15, independent_gradient_max=1e-8,
            lambda_L2=1., parameters=4))
        rr.append(dict(representation=r, partition=p, optimizer_representation=r, status="COMPLETE_AND_REPLAYED",
            actual_optimizer_fits=1, iterations=it, receipt_sha256=ds, replay_sha256=rs))
    write(reports / "R6_2_OMICS_RETENTION.json", dict(schema="bgb_en_r6_retention_report_v1",
        status="COMPLETE_18_LOGICAL_CELLS_AND_REPLAY", retention_contract_sha256=rl["contract_sha256"],
        planned_logical_cells=18, planned_actual_fits=18, confirmed_new_optimizer_fits=18, unknown_fit_count_cells=0,
        new_teacher_or_PCA_fits=0, new_Set_or_Mamba_training=0, rows=rr + old))
    attention = dict(schema="bgb_en_r6_attention_diagnostics_v1", status="COMPLETE_90_FIRST8_INFERENCE_DIAGNOSTICS",
        contract_sha256=tl["contract_sha256"], replay_audit_sha256=rps, completed_model_cells=90,
        controls_per_cell=5, diagnostic_forward_calls=450, fixed_first_valid_rows=8, new_optimizer_fits=0,
        new_backward_calls=0, cases_selected_using_outcomes=False, independent_patient_count_is_not_720=True,
        repeated_patient_evaluations=720, unique_probed_patients=24,
        rows=[dict(arm=a, partition=p, seed=s, fixed_validation_rows=8) for a in ledger.ARMS for p in ledger.PARTITIONS for s in ledger.SEEDS])
    attention_sha = ledger.canonical(attention); descriptors = []
    for arm in ledger.ARMS:
        positions = [i for i, r in enumerate(attention["rows"]) if r["arm"] == arm]
        name = f"R6_ATTENTION_DIAGNOSTICS_{arm}.json"
        ps = write(reports / name, dict(schema="bgb_en_r6_attention_diagnostics_arm_shard_v1", arm=arm,
            contract_sha256=tl["contract_sha256"], logical_object_sha256=attention_sha,
            rows=[attention["rows"][i] for i in positions]))
        descriptors.append(dict(arm=arm, file=name, row_count=9, sha256=ps,
            bytes=(reports / name).stat().st_size, row_positions=positions))
    write(reports / "R6_ATTENTION_DIAGNOSTICS.json", dict({k: v for k, v in attention.items() if k != "rows"},
        storage_schema="bgb_en_r6_attention_diagnostics_sharded_index_v1", rows_count=90,
        rows_shards=descriptors, logical_object_sha256=attention_sha))
    write(reports / "R6_RISK_DIAGNOSTICS.json", dict(schema="bgb_en_r6_risk_diagnostics_v1",
        status="COMPLETE_SAVED_METADATA_DIAGNOSTICS", contract_sha256=tl["contract_sha256"], report_sha256=trsha,
        model_or_array_or_label_files_read=False, units=[dict(arm=a, partition=p, seed=s) for a, p, s in sorted(ledger.target_keys())],
        new_optimizer_fits=0, new_optimizer_updates=0, new_model_forwards=0, new_backward_passes=0))
    write(source / "SOURCE_SCHEDULE_LAUNCH.private.json", dict(cgroup_limits={"memory.max": "66571993088", "cpu.max": "1600000 100000"}))
    for directory in (source, target):
        with (directory / "resource_samples.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=ledger.RESOURCE_FIELDS); writer.writeheader()
            for seconds in (0, 5, 10, 15, 100):
                writer.writerow(dict(elapsed_seconds=seconds, GPU_util_percent=90, GPU_memory_MiB=30000,
                    GPU_total_MiB=32768, CPU_util_percent=20, RAM_available_MiB=400000, disk_available_MiB=10000))
    return source, target, retention, reports


def test_complete_accounting_no_alias_no_optional_denominator_and_host_quota(complete_metadata):
    result, resource, arms = ledger.build(*complete_metadata)
    assert result["formal_neural"]["combined_actual_fits"] == 225
    assert result["formal_neural"]["combined_optimizer_updates"] == 18000
    assert result["separate_convex_retention"]["LBFGS_iterations"] == 609
    assert result["separate_convex_retention"]["actual_optimizer_fits"] == 18
    assert result["separate_qualification"]["source"]["optimizer_updates"] == 2
    assert result["separate_qualification"]["target"]["additional_RNA_input_autograd_grad_probes_code_derived"] == 1
    assert sum(r["actual_formal_neural_fits"] for r in arms) == 90
    assert next(r for r in arms if r["arm"] == "B10")["status"] == "PENDING"
    assert not any(r["counted_as_formal_failure"] for r in arms)
    capacity = resource["cgroup_capacity_from_source_launch"]
    assert capacity["memory_limit_GiB"] == 62 and capacity["CPU_quota_equivalent_cores"] == 16
    assert resource["windows"][0]["hostwide_RAM_available_min_MiB"] == 400000
    assert resource["windows"][0]["excluded_long_gap_seconds"] == 85
    assert resource["windows"][0]["supported_interpolation_seconds"] == 15
    assert result["accountant_runtime"]["model_array_label_files_read"] is False


def test_terminal_receipt_tamper_is_rejected_even_report_claims_complete(complete_metadata):
    source, target, retention, reports = complete_metadata
    path = target / "REPLAY_PRIVATE/B7_p1_seed47.PASS.json"
    value = json.loads(path.read_text()); value["changed_pair_credits"] = 1; write(path, value)
    with pytest.raises(ValueError, match="TARGET_ACTUAL_REPLAY_TERMINAL_OR_NUMERIC_FAILED"):
        ledger.build(source, target, retention, reports)


def test_distinct_task_recipe_hashes_accepted_only_with_shared_evidence_lineage(complete_metadata):
    source, target, retention, reports = complete_metadata
    result, _, _ = ledger.build(source, target, retention, reports)
    proof = result["shared_evidence_lineage"]
    assert proof["target_recipe_sha256"] != proof["retention_recipe_sha256"]
    assert proof["verified_full_source_states"] == proof["verified_prepared_receipt_origins"] == 3
    # Shared provenance remains a fail-closed requirement, independently of the
    # already checked canonical lock SHA and per-task receipt-chain hashes.
    sr = json.loads((reports / "R6_1_SOURCE_SUPERVISION.json").read_text())
    tl = json.loads((target / "targetLOCK.json").read_text())
    rl = json.loads((retention / "retentionLOCK.json").read_text())
    evidence = result["evidence_sha256"]
    for field, mutate, code in (
        ("source_full_states", lambda x: x["MATCHED"].update(checkpoint_sha256="c" * 64), "SHARED_FULL_SOURCE_PROVENANCE_CHANGED"),
        ("data_fingerprints", lambda x: x.update(cohort_sha256="c" * 64), "SHARED_COHORT_ENDPOINT_FINGERPRINT_CHANGED"),
        ("prepared_origins", lambda x: x["outer1_inner0"].update(prepared_receipt_sha256="c" * 64), "SHARED_PREPARED_ORIGIN_CHANGED"),
    ):
        changed = copy.deepcopy(rl); mutate(changed[field])
        with pytest.raises(ValueError, match=code):
            ledger.shared_lineage(source, sr, tl, changed, dict(evidence))
    changed = copy.deepcopy(rl); changed["launch_gate_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="SHARED_LAUNCH_GATE_CHANGED"):
        ledger.shared_lineage(source, sr, tl, changed, dict(evidence))


def test_incomplete_target_cannot_produce_final_ledger(complete_metadata):
    source, target, retention, reports = complete_metadata
    (target / "cells/B7_p2_seed83/COMPLETE.json").unlink()
    with pytest.raises(ValueError, match="REQUIRED_METADATA_FILE_ABSENT_OR_UNSAFE"):
        ledger.build(source, target, retention, reports)


def test_retention_exact_alias_not_counted_as_18_independent_fits(complete_metadata):
    source, target, retention, reports = complete_metadata
    path = retention / "cells/FIXED_HALLMARK_SUM50_p0/COMPLETE.json"
    value = json.loads(path.read_text()); value["actual_optimizer_fits"] = 0
    value["status"] = "REUSED_EXACT_REPRESENTATION_FIT"; write(path, value)
    with pytest.raises(ValueError, match="RETENTION_RECEIPT_OR_REPORTED_COST_CHANGED"):
        ledger.build(source, target, retention, reports)


def test_shard_hash_tamper_not_hidden_behind_complete_attention_index(complete_metadata):
    source, target, retention, reports = complete_metadata
    path = reports / "R6_ATTENTION_DIAGNOSTICS_B7.json"
    value = json.loads(path.read_text()); value["rows"][0]["fixed_validation_rows"] = 7; write(path, value)
    with pytest.raises(ValueError, match="ATTENTION_SHARD_HASH_OR_CONTRACT_CHANGED"):
        ledger.build(source, target, retention, reports)


def test_write_preserves_existing_files_and_exports_no_paths(complete_metadata, tmp_path):
    result, resource, arms = ledger.build(*complete_metadata)
    output = tmp_path / "export"
    ledger.write_outputs(output, result, resource, arms)
    assert set(p.name for p in output.iterdir()) == set(ledger.NAMES)
    content = (output / ledger.NAMES[0]).read_text(encoding="utf-8")
    assert str(tmp_path) not in content
    with pytest.raises(ValueError, match="LEDGER_EXISTING_OUTPUT_PRESERVED"):
        ledger.write_outputs(output, result, resource, arms)
