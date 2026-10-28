"""Synthetic metadata only; no real assets, model files, fit or inference."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import bgb_en_r7_delivery as delivery

D = "d" * 64


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")
    return delivery.sha(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def seal(lock):
    lock.pop("contract_sha256", None)
    lock["contract_sha256"] = delivery.canonical(lock)
    return lock


@pytest.fixture
def qualified(tmp_path, monkeypatch):
    root, private, reports = tmp_path / "repo", tmp_path / "private", tmp_path / "public"
    root.mkdir(); private.mkdir(); reports.mkdir()
    monkeypatch.setattr(delivery, "ROOT", root)
    config = dict(schema="bgb_en_r7_inner_v1", trainable_arms=list(delivery.ARMS), formal_target_neural_fits=99,
        paired_comparisons=[])
    config_name = "configs/bgb_en/r7_inner_v1.json"
    sources = {config_name: write(root / config_name, config)}
    for relative in ("src/bgb_en/r7_survival.py", "scripts/bgb_en_r7_common.py", "scripts/bgb_en_r7_replay.py",
        *(f"src/frozen_{i:02}.py" for i in range(36))):
        path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# SYNTHETIC SOURCE NOT A TRAINABLE MODEL\n" + relative, encoding="utf-8")
        sources[relative] = delivery.sha(path)
    lock = dict(schema="bgb_en_r7_target_protocol_lock_v1", status="LOCKED_BEFORE_R7_FITS", sources=sources,
        config_sha256=sources[config_name], source_config_sha256=D, prepared_receipt_sha256={str(p): D for p in delivery.PARTITIONS},
        source_full_states={"MATCHED": dict(complete_source_state_sha256=D)}, upstream_r6=dict(contract_sha256=D),
        calibration_receipt_sha256={}, fixed_deterministic_runtime=dict(synthetic=True), planned_neural_fits=99,
        planned_target_optimizer_updates=7920, teacher_refits=0, source_refits=0, outer_test_access=False)
    qcal = []
    for part in delivery.PARTITIONS:
        for seed in delivery.SEEDS:
            scales = {key: float(i + 1) for i, key in enumerate(delivery.SCALE_STATS)}
            statistics = {stage: {} for stage in ("raw", "base", "reverse")}
            for key, (stage, statistic) in delivery.SCALE_STATS.items():
                statistics[stage][statistic] = dict(count=100, sum_squares=100 * scales[key]**2)
            statistics["raw"]["raw_gene_value"] = dict(count=100, sum_squares=400.)
            origin = dict(partition=part, seed=seed, train_hash=D, prepared_receipt_sha256=D,
                source_state_sha256=D, patch_cache_receipt_sha256=D, initial_shared_state_sha256=D,
                core_source_sha256=sources["src/bgb_en/r7_survival.py"], common_source_sha256=sources["scripts/bgb_en_r7_common.py"],
                config_sha256=sources[config_name], calibration_model="untrained_NORM", validation_used=False, outcomes_used=False)
            cal = dict(status="PASS_R7_TRAIN_RMS_CALIBRATION", origin=origin, new_optimizer_updates=0,
                calibration=dict(train_hash=D, training_patients=300 + part, scales=scales, statistics=statistics,
                    train_only=True, outcomes_accessed=False, new_optimizer_updates=0, fixed_after_initialization=True))
            h = write(private / "calibration" / f"p{part}_seed{seed}.json", cal)
            lock["calibration_receipt_sha256"][f"p{part}_seed{seed}"] = h
            qcal.append(dict(partition=part, seed=seed, calibration=cal))
    write(private / "targetLOCK.json", seal(lock))
    q = dict(schema="bgb_en_r7_real_data_qualification_v1", status="PASS_R7_REAL_DATA_AND_READONLY_R6_TRANSFER",
        contract_sha256=lock["contract_sha256"], source_refits=0, teacher_refits=0, scalar_beta_refits=0,
        optimizer_instances=0, optimizer_updates=0, qualification_weights_saved=False, outer_test_access=False,
        explicit_backward_passes=11, complete_model_forward_calls=110, distinct_train_RMS_calibrations=9,
        scalar_scales_per_calibration=6, calibration_fits_are_unsupervised_not_neural_fits=True, calibrations=qcal,
        records=[dict(arm=a, explicit_backward_passes=1, optimizer_updates=0,
            complete_model_forward_calls=len(delivery.controls(a)) + 4, repeated_forward_max_error=0) for a in delivery.ARMS])
    write(private / "QUALIFICATION.json", q)
    return root, private, reports


def rebind_calibration(private, part, seed, mutate):
    path = private / "calibration" / f"p{part}_seed{seed}.json"
    cal = read(path); mutate(cal); h = write(path, cal)
    lock = read(private / "targetLOCK.json"); lock["calibration_receipt_sha256"][f"p{part}_seed{seed}"] = h
    write(private / "targetLOCK.json", seal(lock))
    q = read(private / "QUALIFICATION.json"); q["contract_sha256"] = lock["contract_sha256"]
    next(r for r in q["calibrations"] if (r["partition"], r["seed"]) == (part, seed))["calibration"] = cal
    write(private / "QUALIFICATION.json", q)


def test_qualification_exports_exact54_scales_without_origins_or_raw_measurements(qualified):
    root, private, reports = qualified
    result = delivery.export_qualification(private, reports)
    assert result["scalar_scale_count"] == 54 and len(result["rows"]) == 9
    assert all(len(r["scales"]) == len(r["statistic_counts"]) == 6 for r in result["rows"])
    data = (reports / "R7_SCALE_CALIBRATION_AUDIT.json").read_text()
    assert "sum_squares" not in data.replace("scales_recomputed_from_full_train_sum_squares_and_counts", "")
    assert "initial_shared_state_sha256" not in data and "train_hash" not in data.replace("full_current_train_receipt_and_train_hash_scope_verified", "")
    assert str(private) not in data and '"origin"' not in data
    assert result["public_preflight"]["actual_private_identifier_denylist_scanned"] is False
    assert not list(root.rglob("*.pt")) and not list(private.rglob("*.pt"))


def test_consistently_rehashed_but_incorrect_RMS_rejected(qualified):
    _, private, _ = qualified
    rebind_calibration(private, 0, 31, lambda c: c["calibration"]["scales"].update(context_rms=9.))
    with pytest.raises(ValueError, match="R7_CALIBRATION_RMS_FORMULA_OR_MINIMUM_CHANGED"):
        delivery.calibration_audit(private)


def test_consistently_rehashed_validation_calibration_rejected(qualified):
    _, private, _ = qualified
    rebind_calibration(private, 0, 31, lambda c: c["origin"].update(validation_used=True))
    with pytest.raises(ValueError, match="R7_CALIBRATION_SOURCE_OR_TRAIN_SCOPE_CHANGED"):
        delivery.calibration_audit(private)


def test_science_source_bytes_must_still_match_the_frozen40_file_lock(qualified):
    root, private, _ = qualified
    (root / "src/frozen_00.py").write_text("MODIFIED", encoding="utf-8")
    with pytest.raises(ValueError, match="R7_LOCKED_SOURCE_BYTES_CHANGED"):
        delivery.calibration_audit(private)


def test_completion_refuses_missing_matrix_and_does_not_publish_partial_ledger(qualified):
    _, private, reports = qualified
    with pytest.raises(ValueError, match="REQUIRED_JSON_METADATA_MISSING_OR_UNSAFE"):
        delivery.export_completion(private, reports)
    assert not list(reports.iterdir())


# Assemble deliberately fictitious rule probes at runtime so public test source
# cannot be mistaken for a recorded private identifier or an actual asset path.
@pytest.mark.parametrize("value", [dict(risk_vector=[1., 2.]), dict(note="TCGA-" + "AA-" + "0001"),
    dict(note="/" + "root" + "/private/model.pt")])
def test_replay_public_preflight_rejects_raw_arrays_identifiers_and_paths(value):
    with pytest.raises(ValueError):
        delivery.public_preflight(delivery.json_bytes(value))


def test_oversized_replay_cannot_be_trimmed_or_exported():
    with pytest.raises(ValueError, match="PUBLIC_JSON_EXCEEDS_2MIB"):
        delivery.public_preflight(b" " * (delivery.MAX_PUBLIC_BYTES + 1))


def test_partial_cgroup_window_is_not_backfilled():
    rows = [dict(unix_time=t, current=v) for t, v in ((100., 10.), (105., 20.), (110., 30.), (195., 40.))]
    result = delivery.summarize_window(rows, "unix_time", ("current",), "SYNTHETIC")
    assert result["supported_interpolation_seconds"] == 10
    assert result["excluded_long_gap_seconds"] == 85
    assert result["statistics"]["current"]["time_weighted_mean"] == 20
    assert result["unsampled_intervals_not_backfilled"] is True


def add_complete_matrix(tree):
    """All 99 metadata cells, 297 SHA-only states; deliberately no .pt files."""
    root, private, reports = tree
    lock, q = read(private / "targetLOCK.json"), read(private / "QUALIFICATION.json")
    upstream = dict(status="PASS_R7_REUSED_UPSTREAM_39_COX_AND_3_BETA", teacher_refits=0, source_refits=0,
        new_optimizer_fits=0, inherited_R6_contract_sha256=D,
        rows=[dict(partition=p, status="PASS", coefficient_models=13, beta_gradient_abs=1e-9,
            maximum_risk_error=1e-14, new_optimizer_fits=0, new_preprocessing_fits=0,
            teacher_heldout_labels_excluded_from_teacher_fits=True, validation_labels_excluded=True) for p in delivery.PARTITIONS])
    ups = write(private / "UPSTREAM_REPLAY_AUDIT.json", upstream)
    q["upstream_replay_audit_sha256"] = ups; qs = write(private / "QUALIFICATION.json", q)
    rows = []
    for arm, part, seed in sorted(delivery.units()):
        identity = dict(arm=arm, partition=part, seed=seed); unit = delivery.name((arm, part, seed))
        directory = private / "cells" / unit
        c = dict(identity, contract_sha256=lock["contract_sha256"], qualification_sha256=qs,
            config_sha256=lock["config_sha256"], source_config_sha256=D, upstream_r6=lock["upstream_r6"],
            fixed_deterministic_runtime=lock["fixed_deterministic_runtime"], prepared_origin=dict(prepared_receipt_sha256=D),
            epochs=80, teacher_refits=0, source_training_updates=0, outer_test_used=False)
        diagnostics = {}
        for epoch in (20, 40, 60, 80):
            gradient = dict(scope="complete_train_riskset_Cox_vector_accumulated_all_microbatches", microbatches=38,
                diagnostic_forward_calls=76, diagnostic_gradient_calls_per_microbatch=6,
                independent_parameter_L2_gradient_calls=1, new_optimizer_updates=0)
            query = dict(diagnostic_forward_calls=2 if arm in ("ADD", "RAW") else 3, new_backward_calls=0, new_optimizer_updates=0)
            diagnostics[str(epoch)] = dict(epoch=epoch, optimizer_updates_completed=epoch, gradient_probe=gradient,
                query_probe=query, validation_ranking_used_for_selection=False)
            write(directory / f"DIAGNOSTIC_epoch{epoch}.json", diagnostics[str(epoch)])
        done = dict(identity, status="COMPLETE_R7_TARGET_FIT", contract=c, epochs=80, optimizer_updates=80,
            actual_new_optimizer_fits=1, outer_test_used=False, primary_epoch_locked_before_training=80,
            teacher_refits=0, source_training_updates=0, artifact_sha256=D, elapsed_seconds=10., peak_GPU_bytes=100,
            source_origin=dict(source_key="MATCHED", source_origin=lock["source_full_states"]["MATCHED"],
                calibration_receipt_sha256=lock["calibration_receipt_sha256"][f"p{part}_seed{seed}"]),
            checkpoint_artifact_sha256={str(e): D for e in delivery.EPOCHS}, checkpoint_diagnostics=diagnostics)
        ds = write(directory / "COMPLETE.json", done)
        checkpoints = [dict(epoch=e, status="PASS", checkpoint_sha256=D, maximum_risk_error=1e-8,
            risk_errors_by_split=dict(train=1e-8, validation=1e-8), changed_pair_credits=0,
            frozen_control_risk_errors={control: 1e-8 for control in delivery.controls(arm)}, maximum_metric_error=1e-14,
            mean_query_error=1e-8, actual_state_dict_forward_replayed=True, new_optimizer_updates=0) for e in delivery.EPOCHS]
        rp = dict(identity, unit=unit, contract_sha256=lock["contract_sha256"], completion_sha256=ds, artifact_sha256=D,
            qualification_sha256=qs, upstream_replay_audit_sha256=ups, replay_script_sha256=lock["sources"]["scripts/bgb_en_r7_replay.py"],
            status="PASS", checkpoints=checkpoints, actual_checkpoint_count=3, maximum_risk_error=1e-8, changed_pair_credits=0,
            all_checkpoints_actual_forward_replayed=True, actual_state_dict_forward_replayed=True,
            source_frozen_weight_hash_replayed=True, saved_query_mapping_recomputed=True, new_optimizer_updates=0)
        rows.append(rp); write(private / "REPLAY_PRIVATE" / (unit + ".PASS.json"), rp)
    refs = []
    for key in sorted(delivery.units(("B0", "B1"))):
        arm, part, seed = key
        r = dict(arm=arm, partition=part, seed=seed, status="R7_BASELINE_REFERENCE", group="REFERENCE",
            contract_sha256=lock["contract_sha256"], actual_new_optimizer_fits=0, teacher_refits=0,
            deterministic_seed_reference=seed != 31, prepared_origin=dict(prepared_receipt_sha256=D))
        h = write(private / "references" / (delivery.name(key) + ".json"), r)
        refs.append(dict(arm=arm, partition=part, seed=seed, status="PASS", receipt_sha256=h,
            independent_fixed_anchor_metrics_recomputed=True, new_optimizer_fits=0))
    rp = dict(status="PASS_COMPLETE_R7_ALL_CHECKPOINT_REPLAY", contract_sha256=lock["contract_sha256"], qualification_sha256=qs,
        upstream_replay_audit_sha256=ups, expected_units=99, available_units=99, passed_units=99, failed_units=0,
        expected_checkpoints=297, passed_checkpoints=297, missing_or_unreplayed_units=[], units=rows, new_actual_replays=99,
        reused_terminal_receipts=0, upstream_reference_replay=upstream, zero_fit_baseline_reference_replays=refs,
        paired_retrained_controls=[], new_optimizer_fits=0, new_preprocessing_fits=0, primary_checkpoint_epoch=80,
        diagnostic_checkpoints_not_selected=True, patient_arrays_exported=False)
    rps = write(private / "REPLAY_AUDIT.json", rp)
    write(reports / "R7_RESULTS.json", dict(schema="bgb_en_r7_complete_report_v1", status="COMPLETE_99_FITS_297_CHECKPOINT_REPLAYS",
        contract_sha256=lock["contract_sha256"], config_sha256=lock["config_sha256"], qualification_sha256=qs, replay_sha256=rps,
        budget=dict(formal_neural_fits=99, formal_optimizer_updates=7920, source_refits=0, teacher_or_beta_refits=0,
            unsupervised_train_RMS_calibrations=9, zero_fit_baseline_reference_rows=18, diagnostics_optimizer_updates=0)))
    labels = [delivery.name(k) for k in sorted(delivery.units())] + [f"references_p{p}" for p in delivery.PARTITIONS] + ["actual_all_checkpoint_replay"]
    write(private / "SCHEDULE_COMPLETE.json", dict(status="COMPLETE_99_R7_TARGET_FITS_AND_297_CHECKPOINT_REPLAYS",
        contract_sha256=lock["contract_sha256"], formal_fit_count=99, formal_optimizer_updates=7920, unplanned_candidates=0,
        performance_adaptive_budget=False, elapsed_seconds=333., worker_ceiling=6, records=[dict(unit=n, returncode=0) for n in labels]))
    write(reports / "R7_0_R6_INTERFACE_AUDIT.json", dict(status="PASS_9_R6_CELLS_36_STATE_FORWARDS", actual_model_forwards=36,
        rows_count=36, new_backward_calls=0, new_optimizer_fits=0,
        rows=[dict(partition=p, seed=s, epoch=e) for p in delivery.PARTITIONS for s in delivery.SEEDS for e in (0, 20, 40, 80)]))
    write(private / "RESOURCE_QUALIFICATION.json", dict(status="PASS_R7_CONCURRENT_RESOURCE_QUALIFICATION", contract_sha256=lock["contract_sha256"],
        qualification_sha256=qs, observed_real_forward_backward=True, observed_concurrent_workers=4, qualified_max_workers=6,
        model_forward_calls=12, backward_calls=12, optimizer_updates=0, weights_saved=False,
        resource_measurements=dict(RAM_total_bytes=66571993088),
        workers=[dict(slot=i, status="PASS", arm="GATE2", model_forward_calls=3, backward_calls=3, optimizer_updates=0, weights_saved=False) for i in range(4)]))
    driver = [dict(UTC=f"2026-10-29T00:00:{s:02}+00:00", RAM_available_bytes=400000000000, RAM_total_bytes=500000000000,
        disk_free_bytes=10000000000, GPU=[dict(total_MiB=32000., used_MiB=27000., free_MiB=5000., utilization_percent=99.)]) for s in (0, 15, 30)]
    (private / "RESOURCE_MONITOR.jsonl").write_text("\n".join(json.dumps(r) for r in driver), encoding="utf-8")
    cgroup = [dict(unix_time=1000. + s, cgroup_memory_current_bytes=40000000000, cgroup_memory_limit_bytes=66571993088,
        data_disk_free_bytes=10000000000, monitor_started_formal_elapsed_seconds=729.) for s in (0, 5, 10)]
    (private / "cgroup_resource_samples.jsonl").write_text("\n".join(json.dumps(r) for r in cgroup), encoding="utf-8")
    write(private.parent / "CPU_CAPACITY_OBSERVATION.private.json", dict(cpu_max_raw="1600000 100000",
        quota_equivalent_cores=16., unix_time=1793204815.103323,
        scope="current_cgroup_capacity_snapshot_not_current_available_CPU_or_training_utilization"))
    return rp


def test_completion_metadata_all99_297states_distinct_diagnostics_and_exact_raw_copy(qualified):
    root, private, reports = qualified
    add_complete_matrix(qualified)
    original = (private / "REPLAY_AUDIT.json").read_bytes()
    result = delivery.export_completion(private, reports)
    assert result["formal"]["neural_fits"] == 99 and result["formal"]["optimizer_updates"] == 7920
    assert result["formal"]["independent_checkpoint_states_actually_replayed"] == 297
    assert result["separate_training_diagnostics"]["saved_diagnostic_receipts"] == 396
    assert result["separate_real_qualification"]["recorded_complete_model_forward_calls"] == 110
    assert result["separate_concurrent_resource_qualification"]["actual_backward_calls"] == 12
    assert (reports / "R7_CHECKPOINT_REPLAY_AUDIT.json").read_bytes() == original
    resource = read(reports / "R7_RESOURCE_SUMMARY.json")
    assert resource["schedule_makespan_seconds"] == 333. and resource["sum_individual_fit_seconds"] == 990.
    assert resource["cgroup_monitor"]["monitor_started_formal_elapsed_seconds"] == 729.
    assert resource["host_psutil_RAM_is_not_container_available_memory"] is True
    assert resource["cgroup_capacity"]["memory_limit_bytes"] == 66571993088
    assert resource["cgroup_capacity"]["CPU_quota_equivalent_cores"] == 16.
    assert resource["cgroup_capacity"]["CPU_quota_source"] == "current_cgroup_capacity_snapshot_not_current_available_CPU_or_training_utilization"
    assert not list(private.rglob("*.pt"))


def test_completion_actual_terminal_tamper_rejected_even_aggregate_still_claims_complete(qualified):
    _, private, reports = qualified
    add_complete_matrix(qualified)
    path = private / "REPLAY_PRIVATE/GATE1_p0_seed31.PASS.json"
    rp = read(path); rp["checkpoints"][0]["maximum_risk_error"] = .01; write(path, rp)
    with pytest.raises(ValueError, match="R7_ACTUAL_REPLAY_TERMINAL_OR_NUMERIC_FAILED"):
        delivery.export_completion(private, reports)
    assert not (reports / "R7_EXECUTION_LEDGER.json").exists()


def test_replay_scalar_structure_rejects_unknown_payload_even_if_raw_generic_scan_would_accept(qualified):
    replay = add_complete_matrix(qualified)
    replay["units"][0]["private_numeric_payload"] = [1., 2.]
    with pytest.raises(ValueError, match="REPLAY_PUBLIC_UNIT_STRUCTURE_CHANGED"):
        delivery.replay_public_structure(replay)
    replay["units"][0].pop("private_numeric_payload")
    replay["units"][0]["checkpoints"][0]["mean_query_error"] = [1., 2.]
    assert delivery.public_preflight(delivery.json_bytes(replay))["schema_and_generic_pattern_preflight_passed"] is True
    with pytest.raises(ValueError, match="REPLAY_PUBLIC_CHECKPOINT_NONSCALAR_LEAF"):
        delivery.replay_public_structure(replay)
