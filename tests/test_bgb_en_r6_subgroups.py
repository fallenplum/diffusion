"""Synthetic saved-risk subgroup audit only; no real data, models or fits."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bgb_en_r6_subgroups.py"
SPEC = importlib.util.spec_from_file_location("r6_subgroup_test_module", SCRIPT)
r = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(r)
CANARY = "PRIVATE_PATIENT_RISK_OR_PATH_SHOULD_NEVER_EXPORT"


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def test_groups_use_actual_finite_masks_not_proxy_codes_and_confirm_the_named_block():
    measured = np.ones((512, 20511), bool); measured[309:, :3024] = False
    cancer = np.where(np.arange(512) < 400, "TCGA-COAD", "READ")
    proxy = np.r_[np.ones(309, int), np.zeros(203, int)]  # Reverse nominal proxy codes deliberately.
    groups, audit = r.define_groups(measured, cancer, proxy)
    assert audit["status"] == "CANONICAL_309_203_BLOCK_CONFIRMED"
    assert audit["full_finite_patients"] == 309 and audit["block_3024_patients"] == 203
    assert audit["coverage_proxy_one_to_one"] and not audit["platform_is_verified_sequencing_identity"]
    full = next(mask for dim, name, mask in groups if name == "RNA_FULL_FINITE")
    assert full.sum() == 309 and proxy[full].min() == 1
    _, mixed = r.define_groups(measured, cancer, np.arange(512) % 2)
    assert not mixed["coverage_proxy_one_to_one"]
    assert sum(row["patients"] for row in mixed["RNA_coverage_platform_crosstab"]) == 512


def test_unexpected_missing_patterns_are_reported_as_observed_without_fake_309_203_labels():
    measured = np.ones((6, 5), bool); measured[2:4, :2] = False; measured[4:, 3:] = False
    groups, audit = r.define_groups(measured, np.array(["COAD", "READ"] * 3), np.arange(6) % 2)
    assert audit["status"] == "ACTUAL_MASK_DIFFERS_FROM_NAMED_EXPECTATION"
    assert audit["exact_finite_patterns"] and audit["other_missing_patients"] == 4
    assert not any(name == "RNA_BLOCK_MISSING_3024" for _, name, _ in groups)
    assert not audit["group_definitions_use_outcomes"]


def test_manual_harrell_ties_empty_and_tau60_cases_have_honest_missing_metrics():
    actual = r.subgroup_components(np.array([1., 2., 3.]), np.ones(3),
        np.array([1., 1., 2.]), np.array([1, 0, 1]), np.array([3., 1., 2.]))
    assert actual["Harrell_num"] == actual["Harrell_den"] == 2 and actual["Harrell"] == 1
    tied = r.subgroup_components(np.array([1., 2., 3.]), np.ones(3),
        np.array([1., 1., 2.]), np.array([1, 0, 1]), np.ones(3))
    assert tied["Harrell_num"] == 1 and tied["Harrell"] == .5
    empty = r.subgroup_components(np.array([1.]), np.array([1]), np.array([]), np.array([]), np.array([]))
    assert empty["patients"] == empty["events"] == empty["Harrell_den"] == empty["Uno_den"] == 0
    assert empty["Harrell"] is empty["Uno60"] is None
    late = r.subgroup_components(np.array([1., 70.]), np.ones(2), np.array([60., 70.]), np.array([1, 0]), np.array([1., 0.]))
    assert late["Harrell"] == 1 and late["Uno_den"] == 0 and late["Uno60"] is None


def test_uno_uses_full_training_partition_censor_KM_instead_of_group_refit():
    train_time, train_event = np.array([1., 2., 3., 10.]), np.array([0, 1, 0, 1])
    actual = r.subgroup_components(train_time, train_event, np.array([2., 5.]), np.array([1, 0]), np.array([1., 0.]))
    assert actual["Uno_num"] == actual["Uno_den"] == pytest.approx(1 / .75**2)
    incorrect_group_refit = r.subgroup_components(train_time[[1, 3]], train_event[[1, 3]],
        np.array([2., 5.]), np.array([1, 0]), np.array([1., 0.]))
    assert incorrect_group_refit["Uno_den"] == 1
    assert actual["IPCW_reference"] == r.IPCW_REFERENCE


def test_subgroup_support_failure_does_not_erase_harrell_or_fabricate_uno():
    actual = r.subgroup_components(np.array([1.]), np.array([0]), np.array([2., 3.]), np.array([1, 0]), np.array([2., 1.]))
    assert actual["Harrell"] == 1 and actual["Uno60"] is None and actual["Uno_den"] is None
    assert actual["Uno60_status"] == "IPCW_TRAINING_SUPPORT_FAILURE"


def test_only_within_partition_components_are_summed_and_seed_counts_are_not_tripled():
    rows = []
    for seed in r.SEEDS:
        for partition in r.PARTITIONS:
            rows.append(dict(arm="B7", endpoint="DSS", group_dimension="ALL", group="ALL", seed=seed, partition=partition,
                patients=2, events=1, censored=1, events_before_60=1,
                Harrell_num=1., Harrell_den=1., Uno_num=1., Uno_den=1.))
    seeds, summary = r.aggregate_rows(rows)
    assert len(seeds) == 3 and summary[0]["Harrell"] == summary[0]["Uno60"] == 1
    assert summary[0]["patients_per_seed"] == 6 and summary[0]["events_per_seed"] == 3
    pooled = r.subgroup_components(np.array([1., 2.]), np.ones(2), np.tile([1., 2.], 3),
        np.tile([1, 0], 3), np.array([100., 99., -100., -101., 0., -1.]))
    assert pooled["Harrell"] == pytest.approx(2 / 3)  # Forbidden cross-partition pairs would change the statistic.
    with pytest.raises(ValueError, match="MISSING_OR_DUPLICATE_PARTITION"):
        r.aggregate_rows(rows[:-1])


def test_scope_guard_catches_wrong_patient_order_even_when_constant_risks_hide_it():
    folds = np.array([1, 2, 3, 4, 5])
    train, valid = np.array([1, 2]), np.array([3, 4])
    r.verify_scope(train, valid, (train.copy(), valid.copy()), folds)
    with pytest.raises(ValueError, match="ROW_SCOPE_CHANGED"):
        r.verify_scope(train, valid[::-1], (train, valid), folds)
    with pytest.raises(ValueError, match="ROW_SCOPE_CHANGED"):
        r.verify_scope(np.array([0, 1]), valid, (np.array([0, 1]), valid), folds)


def fixture(tmp_path):
    from src.bgb_en.protocol import fixed_inner_splits
    from src.bgb_en.survival import evaluate
    n = 30
    raw = dict(t=np.arange(1., n + 1), e=(np.arange(n) % 4 == 0).astype(int), valid=np.ones(n, bool),
        pfi_t=np.arange(1., n + 1), pfi_e=(np.arange(n) % 3 == 0).astype(int), pfi_valid=np.ones(n, bool),
        folds=np.tile(np.arange(1, 6), 6), cancer=np.array(["COAD", "READ"] * 15), platform=np.arange(n) % 2,
        rna=np.zeros((n, 8)), audit=dict(cohort_sha256="a" * 64, endpoints_sha256="b" * 64),
        patient_ids=np.array([CANARY + str(i) for i in range(n)]))
    raw["rna"][15:, :3] = np.nan; raw["rna_measured"] = np.isfinite(raw["rna"])
    config = json.loads((r.ROOT / "configs/bgb_en/r6_inner_v1.json").read_text(encoding="utf-8"))
    config_path = tmp_path / "recipe.json"; put(config_path, config)
    private, primary = tmp_path / "target", tmp_path / "primary"
    private.mkdir(); primary.mkdir()
    splits = fixed_inner_splits(raw["folds"], raw["e"], outer=1)
    decoded, prepared_hashes = {}, {}
    for part, (train, valid) in enumerate(splits):
        stem = f"outer1_inner{part}"
        artifact = primary / "prepared" / (stem + ".pt"); artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes((CANARY + stem).encode())
        receipt = artifact.with_suffix(".json"); put(receipt, dict(status="COMPLETE", sha256=r.sha256(artifact)))
        prepared_hashes[stem] = r.sha256(receipt)
        decoded[artifact] = dict(train=train, valid=valid, CP=-raw["t"], base=-2 * raw["t"])
    mask = raw["rna_measured"]
    mask_sha = r.hashlib.sha256(str(mask.shape).encode() + np.packbits(mask, axis=1, bitorder="little").tobytes()).hexdigest()
    lock = dict(schema="bgb_en_r6_target_protocol_lock_v1", status="LOCKED_BEFORE_TARGET_FITS",
        config_sha256=r.sha256(config_path), planned_neural_fits=90, outer_test_access=False,
        data_fingerprints=raw["audit"], RNA_finite_mask_sha256=mask_sha,
        prepared_receipt_sha256=prepared_hashes, sources={"src/phase10_trimodal/reference.py": r.sha256(r.ROOT / "src/phase10_trimodal/reference.py")})
    lock["contract_sha256"] = r.canonical(lock); put(private / "targetLOCK.json", lock)
    upstream = private / "UPSTREAM_REPLAY_AUDIT.json"; put(upstream, dict(synthetic=True))
    qualification = dict(status="PASS_R6_TARGET_REAL_DATA_AND_SOURCE_TRANSFER", contract_sha256=lock["contract_sha256"],
        new_optimizer_updates=0, new_neural_fits=0, source_parameters_frozen=True,
        original_gene_values_in_dynamic_reread=True, upstream_replay_audit_sha256=r.sha256(upstream))
    put(private / "QUALIFICATION.json", qualification); qualification_sha = r.sha256(private / "QUALIFICATION.json")
    unit_replays, reference_replays = [], []
    for arm in r.ARMS:
        for part, (train, valid) in enumerate(splits):
            for seed in r.SEEDS:
                name = r.unit_name(arm, part, seed)
                full_risk = -raw["t"] * (1 if arm == "B0" else 2) + (seed + int(arm[1:])) / 1000
                risk = full_risk[valid]
                score = evaluate(raw["t"][train], raw["e"][train], raw["t"][valid], raw["e"][valid], risk)
                pfi = evaluate(raw["pfi_t"][train], raw["pfi_e"][train], raw["pfi_t"][valid], raw["pfi_e"][valid], risk)
                contract = dict(contract_sha256=lock["contract_sha256"], config_sha256=lock["config_sha256"],
                    qualification_sha256=qualification_sha, arm=arm, partition=part, seed=seed,
                    prepared_origin=dict(prepared_receipt_sha256=prepared_hashes[f"outer1_inner{part}"]))
                done = dict(arm=arm, partition=part, seed=seed, contract=contract, epochs=80, optimizer_updates=80,
                    prepared_origin=contract["prepared_origin"], contract_sha256=lock["contract_sha256"],
                    metrics_vs=dict(CP_SCALAR=score, PFI_frozen_DSS_risk=pfi), outer_test_used=False)
                if arm in r.NEURAL:
                    directory = private / "cells" / name; directory.mkdir(parents=True)
                    artifact = directory / "model.pt"; artifact.write_bytes((CANARY + name).encode())
                    done.update(status="COMPLETE_R6_TARGET_FIT", artifact_sha256=r.sha256(artifact))
                    path = directory / "COMPLETE.json"; put(path, done)
                    decoded[artifact] = dict(contract=contract, epochs=80, optimizer_updates=80, train=train.copy(), valid=valid.copy(),
                        predictions=dict(validation=dict(risk=risk)), private_note=CANARY)
                    unit_replays.append(dict(unit=name, status="PASS", arm=arm, partition=part, seed=seed,
                        contract_sha256=lock["contract_sha256"], qualification_sha256=qualification_sha,
                        completion_sha256=r.sha256(path), artifact_sha256=r.sha256(artifact),
                        maximum_risk_error=0., changed_pair_credits=0, actual_state_dict_forward_replayed=True))
                else:
                    done.update(status="R6_BASELINE_REFERENCE", actual_new_optimizer_fits=0, teacher_refits=0)
                    path = private / "references" / (name + ".json"); put(path, done)
                    reference_replays.append(dict(arm=arm, partition=part, seed=seed, status="PASS", receipt_sha256=r.sha256(path),
                        independent_fixed_anchor_metrics_recomputed=True, new_optimizer_fits=0))
    replay = dict(status="PASS_COMPLETE_R6_TARGET_CHECKPOINT_REPLAY", contract_sha256=lock["contract_sha256"], qualification_sha256=qualification_sha,
        expected_units=90, available_units=90, passed_units=90, failed_units=0, new_optimizer_fits=0, new_preprocessing_fits=0,
        numerical_risk_tolerance=2e-6, maximum_allowed_changed_pair_credits=0, upstream_replay_audit_sha256=r.sha256(upstream),
        units=unit_replays, zero_fit_baseline_reference_replays=reference_replays)
    put(private / "REPLAY_AUDIT.json", replay)
    return SimpleNamespace(config=config_path, private_root=private, primary_private=primary, assets=tmp_path / "assets.json"), raw, decoded


def test_postfit_full_pipeline_reuses_108_risks_keeps_sealed_outer_labels_out_and_exports_only_aggregates(tmp_path, monkeypatch):
    args, raw, decoded = fixture(tmp_path)
    calls = []
    def load_raw(_):
        calls.append("labels"); return deepcopy(raw)
    monkeypatch.setattr(r, "_load_raw", load_raw)
    monkeypatch.setattr(r, "_load_tensor_artifact", lambda path: deepcopy(decoded[Path(path)]))
    audit, rows, seeds, summary = r.run_audit(args)
    assert calls == ["labels"] and audit["certified_neural_risks"] == 90 and audit["certified_zero_fit_references"] == 18
    assert len(audit["risk_units"]) == 108 and audit["new_optimizer_fits"] == audit["new_optimizer_updates"] == audit["model_forward_calls"] == 0
    assert all(row["patients"] <= 8 for row in rows)
    all_summaries = [v for v in summary if v["group_dimension"] == "ALL" and v["endpoint"] == "DSS"]
    assert len(all_summaries) == 12 and all(v["patients_per_seed"] == 24 for v in all_summaries)
    assert not audit["outer_test_outcomes_evaluated"]
    output = tmp_path / "public"
    r.write_results(audit, rows, seeds, summary, output)
    assert len(list(output.iterdir())) == 5
    for path in output.iterdir():
        text = path.read_text(encoding="utf-8-sig")
        assert CANARY not in text and str(args.private_root) not in text


def test_incomplete_replay_stops_before_any_endpoint_or_risk_decode(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path)
    path = args.private_root / "REPLAY_AUDIT.json"; replay = r.read(path)
    replay["passed_units"] = 89; put(path, replay)
    monkeypatch.setattr(r, "_load_raw", lambda *_: pytest.fail("endpoint access before complete replay"))
    monkeypatch.setattr(r, "_load_tensor_artifact", lambda *_: pytest.fail("private tensor decoding before complete replay"))
    with pytest.raises(ValueError, match="COMPLETE_NINETY"):
        r.run_audit(args)


def test_tampered_private_risk_artifact_stops_before_label_access(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path)
    (args.private_root / "cells/B7_p0_seed31/model.pt").write_bytes(b"changed")
    monkeypatch.setattr(r, "_load_raw", lambda *_: pytest.fail("labels loaded before private hash verification"))
    with pytest.raises(ValueError, match="ARTIFACT_CHANGED"):
        r.run_audit(args)


def test_raw_fingerprint_and_finite_mask_must_match_immutable_target_lock(tmp_path):
    args, raw, _ = fixture(tmp_path)
    lock = r.read(args.private_root / "targetLOCK.json")
    r.verify_raw(raw, lock)
    invalid = deepcopy(raw); invalid["rna_measured"][0, 0] = False
    with pytest.raises(ValueError, match="ACTUAL_RNA_FINITE"):
        r.verify_raw(invalid, lock)
    invalid = deepcopy(raw); invalid["rna"][0, 0] = np.nan; invalid["rna_measured"][0, 0] = False
    with pytest.raises(ValueError, match="LOCKED_RNA_MASK_SHA"):
        r.verify_raw(invalid, lock)


def test_script_contains_no_fit_or_forward_and_only_new_output_namespaces():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "torch.optim" not in text and "minimize(" not in text and ".backward(" not in text
    assert "construct(" not in text and "common.context" not in text and "r5_" not in text
