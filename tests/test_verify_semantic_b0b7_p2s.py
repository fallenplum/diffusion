"""Synthetic saved-design audit qualification; no fitting or real DSS access."""
from copy import deepcopy
import gzip
import itertools
import json
import pickle

import numpy as np
import pytest

from scripts import verify_semantic_b0b7_p2s as audit


def record(**kwargs):
    value = audit.oracle.PassiveRecord(); value.__dict__.update(kwargs)
    return value


def fixture_head():
    """Algebraically stationary synthetic Cox point; no numerical optimizer."""
    rng = np.random.default_rng(314)
    n, nv, d = 44, 13, 24
    t = np.arange(1, n + 1, dtype=float) * 3; e = (np.arange(n) % 3 == 0).astype(int)
    vt = np.arange(1, nv + 1, dtype=float) * 7; ve = (np.arange(nv) % 3 == 0).astype(int)
    target = rng.normal(size=n) * .1; coefficient = np.linspace(-.01, .01, d)
    _, dr = audit.oracle.efron_gradient(target, t, e)
    x = rng.normal(size=(n, d)); x -= dr[:, None] * (dr @ x / (dr @ dr))[None, :]
    x -= dr[:, None] * (coefficient / (dr @ dr))[None, :]
    offset = target - x @ coefficient
    xx = rng.normal(size=(nv, d)); oo = rng.normal(size=nv) * .1
    train = offset + x @ coefficient; evaluation = oo + xx @ coefficient
    objective, gradient = audit.oracle.cox_objective(x, coefficient, offset, t, e, 1.)
    model_receipt = {"fit_design_sha256": audit.oracle.array_sha(x), "fit_time_sha256": audit.oracle.array_sha(t),
        "fit_event_sha256": audit.oracle.array_sha(e), "fit_offset_sha256": audit.oracle.array_sha(offset),
        "objective": objective, "gradient_max": float(np.abs(gradient).max()), "gradient_tolerance": 2e-5,
        "ridge_lambda": 1., "optimized_coefficients": d, "active_coefficients": d}
    train_metrics = audit.oracle.independent_metrics(t, e, t, e, train)
    evaluation_metrics = audit.oracle.independent_metrics(t, e, vt, ve, evaluation)
    model = record(coefficient=coefficient, ridge_lambda=1., receipt=model_receipt)
    main = record(width=8)
    design = record(spec=record(main_pca=8, base_arm="SHARED", factor_rank=0, mode="NO_INTER"), main=main)
    cell = record(model=model, design=design, training_risk=train, evaluation_risk=evaluation,
        receipt={"training_risk_sha256": audit.oracle.array_sha(train), "evaluation_risk_sha256": audit.oracle.array_sha(evaluation)})
    row = {"spec_name": "D8_SHARED_R0_NO_INTER", "spec": {"main_pca": 8, "base_arm": "SHARED", "factor_rank": 0, "mode": "NO_INTER"},
        "ridge_lambda": 1., "status": audit.FIT, "DSS_coefficients": d, "risk_optimizer": model_receipt,
        "train_metrics": train_metrics, "evaluation_metrics": evaluation_metrics}
    meta = {"split": audit.SPLITS[0], "family": "MAIN", "seed": 31, "lambda_pair": .1, "epoch": 400,
            "input_sha256": "a" * 64, "checkpoint_sha256": "b" * 64}
    payload = {"schema": "semantic_p2s_private_shared_design_cells_v1", "meta": meta,
        "cells_by_spec": {row["spec_name"]: {"1.0": cell}}, "cells": {row["spec_name"] + "/lambda1.0": cell},
        "designs": {row["spec_name"]: design}, "main_maps": {8: main},
        "design_matrices": {row["spec_name"]: {"train": x, "evaluation": xx}},
        "offsets": {"train": offset, "evaluation": oo}, "training_offset": offset, "evaluation_offset": oo,
        "train_time": t, "train_event": e, "evaluation_time": vt, "evaluation_event": ve}
    return payload, row


def manifest():
    return {"schema": "semantic_b0b7_p2s_readout_manifest_v1", "status": "COMPLETE", "pretraining_lock_sha256": "s", "readout_lock_sha256": "c",
        "jobs": [dict(zip(("split", "family", "seed", "lambda_pair", "epoch"), fields))
                 for fields in itertools.product(audit.SPLITS, audit.FAMILIES, audit.SEEDS, audit.PAIRS, audit.EPOCHS)]}


def test_exact72_states_must_be_complete_before_any_artifact_decode():
    assert len(audit.validate_manifest(manifest(), "s", "c")) == 72
    for kind in ("missing", "duplicate", "outer", "partial", "wrong_lock"):
        value = manifest()
        if kind == "missing": value["jobs"].pop()
        if kind == "duplicate": value["jobs"][-1] = value["jobs"][0]
        if kind == "outer": value["jobs"][0]["split"] = "outer1_final"
        if kind == "partial": value["status"] = "PARTIAL"
        if kind == "wrong_lock": value["readout_lock_sha256"] = "other"
        with pytest.raises(ValueError): audit.validate_manifest(value, "s", "c")


def test_direct_postsave_Z_coef_uses_no_predict_method_and_checks_both_risks():
    payload, row = fixture_head()
    cell = payload["cells_by_spec"][row["spec_name"]]["1.0"]
    def forbidden(*args, **kwargs):
        raise AssertionError("predict or scorer method must not execute")
    cell.model.predict = forbidden
    result, case, independent = audit.audit_head(payload, row)
    assert result["training_risk_bytes_equal"] and result["evaluation_risk_bytes_equal"]
    assert result["risk_max_abs_error"] == result["independent_objective_abs_error"] == 0.
    assert result["independent_gradient_max"] < 1e-12
    assert result["evaluation_Harrell"] == independent["Harrell"]
    np.testing.assert_array_equal(case["risk"], payload["offsets"]["evaluation"] + payload["design_matrices"][row["spec_name"]]["evaluation"] @ cell.model.coefficient)


@pytest.mark.parametrize("change", ["coefficient", "training_Z", "evaluation_Z", "offset", "risk_receipt", "objective", "gradient", "metric", "dtype", "flat_alias"])
def test_changed_postsave_values_or_claims_fail_closed(change):
    payload, row = fixture_head(); name = row["spec_name"]; cell = payload["cells_by_spec"][name]["1.0"]
    if change == "coefficient": cell.model.coefficient[0] += .01
    elif change == "training_Z": payload["design_matrices"][name]["train"][0, 0] += .01
    elif change == "evaluation_Z": payload["design_matrices"][name]["evaluation"][0, 0] += .01
    elif change == "offset": payload["offsets"]["evaluation"][0] += .01
    elif change == "risk_receipt": cell.receipt["training_risk_sha256"] = "c" * 64
    elif change == "objective": row["risk_optimizer"]["objective"] += .01
    elif change == "gradient": row["risk_optimizer"]["gradient_max"] += .01
    elif change == "metric": row["evaluation_metrics"]["Harrell_num"] += .5
    elif change == "dtype": payload["design_matrices"][name]["train"] = payload["design_matrices"][name]["train"].astype(np.float32)
    else: payload["cells"][name + "/lambda1.0"] = deepcopy(cell)
    with pytest.raises(ValueError): audit.audit_head(payload, row)


def test_independent_efron_ridge_gradient_agrees_with_finite_difference_at_nonstationary_point():
    payload, row = fixture_head(); name = row["spec_name"]
    x = payload["design_matrices"][name]["train"]; w = np.linspace(.01, -.03, 24)
    args = (payload["offsets"]["train"], payload["train_time"], payload["train_event"], .1)
    value, gradient = audit.oracle.cox_objective(x, w, *args)
    eps = 1e-6
    numerical = np.array([(audit.oracle.cox_objective(x, w + np.eye(24)[i] * eps, *args)[0]
                          - audit.oracle.cox_objective(x, w - np.eye(24)[i] * eps, *args)[0]) / (2 * eps) for i in range(24)])
    np.testing.assert_allclose(gradient, numerical, atol=2e-9, rtol=1e-6)
    assert np.isfinite(value)


def test_passive_gzip_keeps_shared_aliases_without_importing_study_model(tmp_path):
    payload, _ = fixture_head(); path = tmp_path / "models.pkl.gz"
    with gzip.open(path, "wb") as stream: pickle.dump(payload, stream, protocol=5)
    result = audit.passive_gzip(path, audit.oracle.sha(path))
    name = "D8_SHARED_R0_NO_INTER"
    assert isinstance(result["cells_by_spec"][name]["1.0"], audit.oracle.PassiveRecord)
    assert result["cells"][name + "/lambda1.0"] is result["cells_by_spec"][name]["1.0"]
    with pytest.raises(ValueError, match="GZIP_ARTIFACT_SHA_CHANGED"):
        audit.passive_gzip(path, "0" * 64)


@pytest.mark.parametrize("change", ["zero_metric", "hidden_fit", "zero_coefficient", "no_reason"])
def test_rank_na_cannot_be_replaced_with_zero_performance_or_hidden_fit(change):
    row = {"spec_name": "D16_B4_R4_INTER", "ridge_lambda": 1., "NA_reason": "INSUFFICIENT_RANK", "DSS_coefficients": None,
           "train_metrics": None, "evaluation_metrics": None, "risk_optimizer": None}
    bank = {}
    audit.assert_na_row(row, bank)
    if change == "zero_metric": row["evaluation_metrics"] = {"Harrell": 0.}
    elif change == "hidden_fit": bank[row["spec_name"]] = {"1.0": object()}
    elif change == "zero_coefficient": row["DSS_coefficients"] = 0
    else: row["NA_reason"] = ""
    with pytest.raises(ValueError): audit.assert_na_row(row, bank)


def test_payload_grid_checks_all54_head_positions_even_when_only_one_was_fit():
    payload, fitted = fixture_head(); rows = []
    for spec, lam in itertools.product(audit.specs("MAIN"), audit.LAMBDAS):
        d, arm, rank, mode = spec; name = f"D{d}_{arm}_R{rank}_{mode}"
        if name == fitted["spec_name"] and lam == 1.:
            rows.append(fitted); continue
        rows.append({"spec_name": name, "spec": {"main_pca": d, "base_arm": arm, "factor_rank": rank, "mode": mode},
            "ridge_lambda": lam, "status": audit.NA, "NA_reason": "SYNTHETIC_RANK_NA", "DSS_coefficients": None,
            "train_metrics": None, "evaluation_metrics": None, "risk_optimizer": None})
    receipt = {"meta": payload["meta"], "rows": rows, "actual_Cox_fits": 1, "actual_saved_head_replays": 1, "NA_count": 53}
    checks, cases, metrics = audit.audit_payload(payload, receipt, payload["meta"])
    assert len(checks) == 54 and len(cases) == 1 and len(metrics) == 54
    assert sum(r["status"] == audit.NA for r in checks) == 53
    receipt["rows"].pop()
    with pytest.raises(ValueError, match="EXACT_HEAD_GRID_CHANGED"):
        audit.audit_payload(payload, receipt, payload["meta"])


def test_264_config_arithmetic_is_seed_micro_not_equal_mean_of_nine_fold_scores():
    rows = []
    for family, pair, epoch, lam in itertools.product(audit.FAMILIES, audit.PAIRS, audit.EPOCHS, audit.LAMBDAS):
        for d, arm, rank, mode in audit.specs(family):
            for si, seed in enumerate(audit.SEEDS):
                for fi, split in enumerate(audit.SPLITS):
                    value, den = .2 + si * .1 + fi * .04, (fi + 1) ** 2 * (si + 1) * 20.
                    rows.append({"family": family, "lambda_pair": pair, "epoch": epoch, "ridge_lambda": lam,
                        "spec": {"main_pca": d, "base_arm": arm, "factor_rank": rank, "mode": mode}, "seed": seed, "split": split,
                        "metrics": {"Harrell": value, "Harrell_num": value * den, "Harrell_den": den,
                                    "Uno60": None, "Uno_num": None, "Uno_den": None}})
    result = audit.independent_aggregates(rows)
    assert len(result) == 264
    first = next(iter(result.values()))
    assert first["Harrell"] != pytest.approx(.2 + .1 + .04)
    assert first["Uno60"] is None
    rows[0]["metrics"] = None
    assert next(iter(audit.independent_aggregates(rows).values()))["Harrell"] is None


def test_full72_all_na_saved_artifacts_audit_is_metadata_not_fabricated_metric_or_official_api(tmp_path, monkeypatch):
    study = {"schema": "semantic_b0b7_p2s_lock_v1", "status": "LOCKED_BEFORE_P2S_OPTIMIZER", "parent_master_sha256": audit.P2_SHA,
        "cells": [{"split": s, "family": f, "seed": seed, "lambda_pair": pair, "input_sha256": "a" * 64,
                   "parent_state400_sha256": "b" * 64} for s, f, seed, pair in itertools.product(audit.SPLITS, audit.FAMILIES, audit.SEEDS, audit.PAIRS)]}
    study_path = tmp_path / "study.json"; study_path.write_text(json.dumps(study), encoding="utf-8")
    child = {"schema": "semantic_b0b7_p2s_readout_lock_v1", "status": "LOCKED_BEFORE_P2S_DSS_FITS",
        "parent_study_lock_sha256": audit.oracle.sha(study_path), "expected_feature_states": 72, "expected_Cox_fits": 2376,
        "primary_ridge_lambda": 1., "ridge_candidates": [.1, 1., 10.], "outer_evaluation": False,
        "anchors": {s: {"sha256": "c" * 64, "receipt_sha256": "d" * 64} for s in audit.SPLITS}}
    child_path = tmp_path / "child.json"; child_path.write_text(json.dumps(child), encoding="utf-8")
    full = manifest(); full["pretraining_lock_sha256"] = audit.oracle.sha(study_path); full["readout_lock_sha256"] = audit.oracle.sha(child_path)
    offsets = {"train": np.zeros(272, dtype=float), "evaluation": np.zeros(137, dtype=float)}
    for i, job in enumerate(full["jobs"]):
        job.update(input_sha256="a" * 64, source_checkpoint_sha256="b" * 64)
        identity = {k: job[k] for k in ("split", "family", "seed", "lambda_pair", "epoch")}
        meta = {**identity, "input_sha256": "a" * 64, "checkpoint_sha256": "b" * 64}
        payload = {"schema": "semantic_p2s_private_shared_design_cells_v1", "meta": meta, "cells_by_spec": {}, "offsets": offsets,
            "train_time": np.arange(272, dtype=float), "train_event": (np.arange(272) % 10 == 0).astype(int),
            "evaluation_time": np.arange(137, dtype=float), "evaluation_event": (np.arange(137) % 10 == 0).astype(int)}
        folder = tmp_path / "private" / ("job%02d" % i); folder.mkdir(parents=True)
        artifact = folder / "READOUT_MODELS.private.pkl.gz"
        with gzip.open(artifact, "wb") as stream: pickle.dump(payload, stream, protocol=5)
        rows = []
        for (d, arm, rank, mode), lam in itertools.product(audit.specs(job["family"]), audit.LAMBDAS):
            rows.append({"spec_name": f"D{d}_{arm}_R{rank}_{mode}", "spec": {"main_pca": d, "base_arm": arm, "factor_rank": rank, "mode": mode},
                "ridge_lambda": lam, "status": audit.NA, "NA_reason": "SYNTHETIC_RANK_NA", "DSS_coefficients": None,
                "train_metrics": None, "evaluation_metrics": None, "risk_optimizer": None})
        receipt = {"schema": "semantic_b0b7_p2s_readout_state_v1", "status": "PARTIAL_PRIMARY_UNSUPPORTED_RANK", "meta": meta, "rows": rows,
            "study_lock_sha256": full["pretraining_lock_sha256"], "readout_lock_sha256": full["readout_lock_sha256"],
            "artifact_sha256": audit.oracle.sha(artifact), "actual_Cox_fits": 0, "actual_saved_head_replays": 0, "NA_count": len(rows),
            "anchor": {"artifact_sha256": "c" * 64, "receipt_sha256": "d" * 64, "new_teacher_or_scalar_fits": 0,
                "full_pipeline_crossfit_for_training_offset": True, "own_labels_excluded_from_teacher_and_beta": True,
                "training_offset_sha256": audit.oracle.array_sha(offsets["train"]), "evaluation_offset_sha256": audit.oracle.array_sha(offsets["evaluation"])}}
        rp = folder / "READOUT.private.json"; rp.write_text(json.dumps(receipt), encoding="utf-8")
        job.update(receipt_file=rp.relative_to(tmp_path / "private").as_posix(), receipt_sha256=audit.oracle.sha(rp))
    mp = tmp_path / "manifest.json"; mp.write_text(json.dumps(full), encoding="utf-8")
    monkeypatch.setattr(audit, "STARTUP_THREADS", {k: "2" for k in audit.STARTUP_THREADS})
    def no_official_call(*args, **kwargs): raise AssertionError("No fitted risks; cannot claim or call official metric API")
    monkeypatch.setattr(audit.oracle, "official_oracle", no_official_call)
    result = audit.verify_run(tmp_path / "private", mp, study_path, full["pretraining_lock_sha256"], child_path,
        full["readout_lock_sha256"], tmp_path / "public", sksurv_target="unused", authorized=True)
    assert result["fitted_heads_audited"] == 0 and result["explicit_rank_NA"] == 2376 and result["actual_official_API_calls"] == 0
    summary = json.loads((tmp_path / "public" / "INDEPENDENT_SUMMARY.json").read_bytes())
    assert summary["max_risk_abs_error"] is None
    assert summary["official_API"]["package_version"] is None
    assert summary["model_predict_calls"] == summary["new_optimizer_fits"] == 0
    assert summary["unique_exact_shared_r0_partitions"] == 3


def test_locked_original400_encoder_binding_and_anchor_offsets_are_not_replaceable():
    jobs = manifest()["jobs"]
    cells = [{"split": s, "family": f, "seed": seed, "lambda_pair": p, "input_sha256": "a" * 64, "parent_state400_sha256": "b" * 64}
             for s, f, seed, p in itertools.product(audit.SPLITS, audit.FAMILIES, audit.SEEDS, audit.PAIRS)]
    for job in jobs: job.update(input_sha256="a" * 64, source_checkpoint_sha256="b" * 64)
    audit.validate_study_state_bindings(jobs, {"cells": cells})
    jobs[0]["source_checkpoint_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="ORIGINAL400_STATE_SOURCE_CHANGED"):
        audit.validate_study_state_bindings(jobs, {"cells": cells})
    payload, _ = fixture_head()
    locked = {"sha256": "a" * 64, "receipt_sha256": "b" * 64}
    receipt = {"anchor": {"artifact_sha256": "a" * 64, "receipt_sha256": "b" * 64, "new_teacher_or_scalar_fits": 0,
        "full_pipeline_crossfit_for_training_offset": True, "own_labels_excluded_from_teacher_and_beta": True,
        "training_offset_sha256": audit.oracle.array_sha(payload["offsets"]["train"]), "evaluation_offset_sha256": "c" * 64}}
    with pytest.raises(ValueError, match="OFFSETS_DIFFER_FROM_LOCKED"):
        audit.audit_anchor(payload, receipt, locked)


def test_exact_current_producer_manifest_checkpoint_field_is_canonical_and_alias_conflict_rejected():
    assert audit.checkpoint_hash({"checkpoint_sha256": "a" * 64}) == "a" * 64
    assert audit.checkpoint_hash({"source_checkpoint_sha256": "a" * 64}) == "a" * 64
    assert audit.checkpoint_hash({"checkpoint_sha256": "a" * 64, "source_checkpoint_sha256": "a" * 64}) == "a" * 64
    with pytest.raises(ValueError, match="ALIASES_CONFLICT"):
        audit.checkpoint_hash({"checkpoint_sha256": "a" * 64, "source_checkpoint_sha256": "b" * 64})
    value = manifest(); cells = []
    for job in value["jobs"]:
        job.update(input_sha256="a" * 64, checkpoint_sha256="b" * 64)
        key = {k: job[k] for k in ("split", "family", "seed", "lambda_pair")}
        if not any(all(c[k] == v for k, v in key.items()) for c in cells):
            cells.append({**key, "input_sha256": "a" * 64, "parent_state400_sha256": "b" * 64})
    audit.validate_study_state_bindings(value["jobs"], {"cells": cells})
