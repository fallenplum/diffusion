"""Synthetic math and mocked-solver tests; no real optimizer fits or endpoints."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import bgb_en_r6_retention as r


def recipe():
    return r.read(r.ROOT / r.CONFIG)


def synthetic_survival():
    return np.array([1., 1., 2., 3., 3., 4., 5., 5.]), np.array([1, 1, 0, 1, 0, 1, 1, 0])


def test_recipe_fixes_six_representations_eighteen_cells_and_nine_existing_references():
    config = r.validate_recipe(recipe())
    assert len(config["representations"]) == 6 and config["maximum_logical_Cox_cells"] == 18
    assert len(config["historical_references"]) == 9
    assert {row["partition"] for row in config["historical_references"]} == {0, 1, 2}
    assert config["normalization_geometry"]["learned_tokens_are_original_genes"] is False
    for field, changed in (("CNA_enabled", True), ("Set_or_Mamba_new_training_allowed", True),
                           ("source_full_seed", 47), ("maximum_logical_Cox_cells", 19)):
        invalid = deepcopy(config); invalid[field] = changed
        with pytest.raises(ValueError):
            r.validate_recipe(invalid)
    invalid = deepcopy(config); invalid["solver"]["lambda_L2"] = .1
    with pytest.raises(ValueError, match="NO_EXTRA_CANDIDATES_OR_TUNING"):
        r.validate_recipe(invalid)


def test_observed_train_scaler_ignores_validation_values_and_preserves_missing_zero():
    x = np.array([[0., 2., np.nan], [2., 4., np.nan], [4., np.nan, np.nan], [1e6, -1e6, 5.]])
    mask = np.isfinite(x); train = np.array([0, 1, 2])
    scaled, valid, state, audit = r.masked_feature_scale(x, mask, train)
    changed = x.copy(); changed[3] = [-1e12, 3e12, 1e18]
    alternate, _, other, _ = r.masked_feature_scale(changed, mask, train)
    for key in state:
        assert np.array_equal(state[key], other[key])
    assert np.array_equal(scaled[:3], alternate[:3])
    assert valid[0, 0] and scaled[0, 0] != 0  # A measured natural zero is an observation.
    assert not valid[:, 2].any() and scaled[:, 2].tolist() == [0., 0., 0., 0.]
    assert state["train_observed_count"].tolist() == [3, 2, 0]
    assert audit["source_ST_scaler_used"] is False and audit["train_only_fit"]
    assert np.isfinite(scaled).all()


def test_true_unit_SD_column_is_not_falsely_reported_as_constant():
    x = np.array([[-1., 0.], [1., 0.]])
    _, _, _, audit = r.masked_feature_scale(x, np.ones_like(x, bool), np.array([0, 1]))
    assert audit["train_constant_or_small_SD_columns"] == 1


def test_fixed_sum_and_mean_keep_real_measured_counts_and_all_missing_safe():
    x = np.array([[1., 3., np.nan], [2., np.nan, 5.], [np.nan, np.nan, np.nan]])
    measured = np.isfinite(x)
    member = np.array([[True, True, False], [False, True, True]])
    summed, mask, count = r.fixed_pool(x, measured, member)
    mean, mean_mask, _ = r.fixed_pool(x, measured, member, mean=True)
    np.testing.assert_array_equal(summed, [[4., 3.], [2., 5.], [0., 0.]])
    np.testing.assert_array_equal(mean, [[2., 3.], [2., 5.], [0., 0.]])
    assert count.tolist() == [[2, 1], [1, 1], [0, 0]]
    assert np.array_equal(mask, mean_mask) and not mask[2].any()


def test_signed_programs_normalize_up_and_down_separately_and_keep_empty_down_zero():
    values = np.array([[2., 6., 3., 8.], [2., np.nan, 3., np.nan]])
    mask = np.isfinite(values)
    ep, eg, sign = np.array([0, 0, 0, 1]), np.array([0, 1, 2, 3]), np.array([1, 1, -1, 1])
    out, valid, counts = r.signed_program_mean(values, mask, ep, eg, sign)
    np.testing.assert_array_equal(out, [[1., 8., 0., 0.], [-1., 0., 0., 0.]])
    assert counts[0, 0].tolist() == [2, 1] and counts[0, 1].tolist() == [1, 0]
    assert not valid[1, 1] and not valid[:, 2:].any()


def test_exact_sum_mean_alias_requires_identical_scaled_values_and_masks():
    x = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    m = np.ones_like(x, bool); train = np.array([0, 1, 2])
    a = r.masked_feature_scale(x * 2, m, train)
    b = r.masked_feature_scale(x, m, train)
    values = {name: a for name in r.REPRESENTATIONS}; values[r.REPRESENTATIONS[1]] = b
    assert r.exact_pool_alias(values)[r.REPRESENTATIONS[1]] == r.REPRESENTATIONS[0]
    changed = list(b); changed[0] = changed[0].copy(); changed[0][3, 0] += 1e-14
    values[r.REPRESENTATIONS[1]] = tuple(changed)
    assert r.exact_pool_alias(values)[r.REPRESENTATIONS[1]] == r.REPRESENTATIONS[1]
    changed = list(b); changed[1] = m.copy(); changed[1][3, 1] = False
    values[r.REPRESENTATIONS[1]] = tuple(changed)
    assert r.exact_pool_alias(values)[r.REPRESENTATIONS[1]] == r.REPRESENTATIONS[1]


def test_frozen_gene_tokens_recompute_own_values_without_ST_scaler_or_training():
    import torch
    from src.bgb_en.r6_omics import mapping_from_programs, GeneEvidenceTransform, SparseProgramEncoder, GeneValueBank
    from src.bgb_en.r6_source_model import SourceProgramModel, PROGRAMS
    mapping = mapping_from_programs([dict(id=name, up_genes=[str(k)], down_genes=[str(k + 4)]) for k, name in enumerate(PROGRAMS)])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(62131)
        source = SourceProgramModel("S7", tissue_prior=[.3, .4, .3], init_seed=31,
            gene_reader=SparseProgramEncoder(mapping, gene_encoder=GeneEvidenceTransform())).requires_grad_(False).eval()
    x = np.arange(48).reshape(6, 8).astype(float) / 10
    mask = np.ones_like(x, bool); mask[-1] = False; x[~mask] = np.nan
    flat, mean, _ = r.frozen_source_features(source, x, mask, mapping)
    assert flat[0].shape == (6, 256) and mean[0].shape == (6, 64)
    assert not flat[1][-1].any() and not mean[1][-1].any()
    assert np.all(flat[0][-1] == 0) and np.all(mean[0][-1] == 0)
    bank = GeneValueBank(mapping.gene_symbols, mapping.gene_symbols)
    with torch.no_grad():
        actual = source.encode_gene_programs(bank(torch.tensor(x, dtype=torch.float32), torch.tensor(mask)))
    np.testing.assert_array_equal(flat[0], actual["tokens"].numpy().astype(float).reshape(6, 256))
    changed = x.copy(); changed[-1] = 1e20
    again = r.frozen_source_features(source, changed, mask, mapping)
    assert np.array_equal(flat[0], again[0][0])
    assert all(p.grad is None and not p.requires_grad for p in source.parameters())
    source.program_queries.requires_grad_(True)
    with pytest.raises(ValueError, match="FROZEN_COMPLETE"):
        r.frozen_source_features(source, x, mask, mapping)


def test_independent_event_time_Efron_matches_matrix_gradient_and_finite_difference():
    x = np.random.default_rng(62131).normal(size=(8, 5)); t, e = synthetic_survival()
    w = np.array([.1, -.2, .05, .03, -.04])
    loss, gradient = r.matrix_efron_objective(x, t, e, w)
    own, independent = r.independent_efron_objective(x, t, e, w)
    assert abs(loss - own) < 2e-14
    np.testing.assert_allclose(gradient, independent, atol=2e-14, rtol=2e-13)
    finite = []
    for k in range(len(w)):
        step = np.zeros_like(w); step[k] = 1e-6
        finite.append((r.independent_efron_objective(x, t, e, w + step)[0] - r.independent_efron_objective(x, t, e, w - step)[0]) / 2e-6)
    np.testing.assert_allclose(gradient, finite, atol=3e-9, rtol=1e-7)


def test_Efron_includes_event_event_ties_and_event_censor_ties():
    t = np.array([1., 1., 1., 2.]); e = np.array([1, 1, 0, 1]); x = np.zeros((4, 2)); w = np.zeros(2)
    # t=1 has d=2 with risk denominators4,3; t=2 denominator1.
    loss, grad = r.independent_efron_objective(x, t, e, w)
    assert abs(loss - np.log(12) / 3) < 1e-15 and np.array_equal(grad, np.zeros(2))


def test_solver_is_called_once_with_fixed_precision_without_running_an_optimizer(monkeypatch):
    called = []
    def simulated(fun, initial, **kwargs):
        called.append(kwargs)
        return SimpleNamespace(x=initial, nit=0, nfev=1, success=True)
    monkeypatch.setattr(r, "minimize", simulated)
    t, e = synthetic_survival()
    coef, audit = r.fit_ridge_once(np.zeros((8, 3)), t, e)
    assert len(called) == 1 and called[0]["method"] == "L-BFGS-B"
    assert called[0]["options"] == dict(maxiter=2000, maxls=50, ftol=1e-15, gtol=1e-9)
    assert audit["gradient_max"] == audit["independent_gradient_max"] == 0 and np.all(coef == 0)


def test_solver_gradient_rejection_keeps_finite_coefficient_and_never_retries(monkeypatch):
    calls = []
    def simulated(fun, initial, **kwargs):
        calls.append(1)
        return SimpleNamespace(x=np.ones_like(initial), nit=7, nfev=10, success=False)
    monkeypatch.setattr(r, "minimize", simulated)
    t, e = synthetic_survival()
    with pytest.raises(r.RetentionSolverError) as failure:
        r.fit_ridge_once(np.zeros((8, 3)), t, e)
    assert len(calls) == 1 and failure.value.audit["iterations"] == 7
    assert failure.value.audit["gradient_max"] == 1 and np.isfinite(failure.value.coefficient).all()


def cache_fixture(tmp_path):
    args = SimpleNamespace(private=tmp_path, resume=False)
    train, valid = np.array([0, 2, 4, 6]), np.array([1, 3, 5])
    x = np.zeros((7, 50)); mask = np.ones_like(x, bool)
    entries = []
    for name in r.REPRESENTATIONS[:2]:
        path = r._cache_path(tmp_path, name, 0); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, features=x, measured_mask=mask, train_rows=train, valid_rows=valid, active_rows=np.r_[train, valid])
        entries.append(dict(representation=name, partition=0, optimizer_representation=r.REPRESENTATIONS[0],
            cache_sha256=r.sha256(path), feature_values_sha256=r.array_hash(x), measured_mask_sha256=r.array_hash(mask),
            train_rows_sha256=r.array_hash(train), valid_rows_sha256=r.array_hash(valid)))
    raw = dict(t=np.arange(2., 9.), e=np.array([1, 1, 1, 0, 0, 1, 1]))
    return args, dict(contract_sha256="1" * 64, representations=entries), entries, raw


def test_identical_sum_mean_share_one_mocked_fit_and_each_gets_independent_replay(tmp_path, monkeypatch):
    args, lock, entries, raw = cache_fixture(tmp_path)
    calls = []
    def simulated(x, t, e):
        calls.append(1); return np.zeros(x.shape[1]), dict(iterations=0, gradient_max=0.)
    monkeypatch.setattr(r, "fit_ridge_once", simulated)
    first = r.fit_unit(args, lock, entries[0], raw)
    second = r.fit_unit(args, lock, entries[1], raw)
    assert len(calls) == 1 and first["actual_optimizer_fits"] == 1 and second["actual_optimizer_fits"] == 0
    for entry in entries:
        result = r.replay_unit(args, lock, entry, raw)
        assert result["new_optimizer_fits"] == 0 and result["independent_gradient_max"] == 0
    with pytest.raises(ValueError, match="NOT_EXPLICIT"):
        r.fit_unit(args, lock, entries[0], raw)
    args.resume = True
    assert r.fit_unit(args, lock, entries[0], raw) == first and len(calls) == 1


def test_alias_source_receipt_changes_and_wrong_cell_replay_are_rejected(tmp_path, monkeypatch):
    args, lock, entries, raw = cache_fixture(tmp_path)
    monkeypatch.setattr(r, "fit_ridge_once", lambda x, t, e: (np.zeros(x.shape[1]), dict(iterations=0)))
    r.fit_unit(args, lock, entries[0], raw); r.fit_unit(args, lock, entries[1], raw)
    path = r._directory(args, entries[0]["representation"], 0) / "COMPLETE.json"
    payload = r.read(path); payload["extra"] = "synthetic_change"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="ALIAS_SOURCE_RECEIPT_CHANGED"):
        r.replay_unit(args, lock, entries[1], raw)
    payload["contract"]["partition"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="COMPLETION_LINEAGE"):
        r.replay_unit(args, lock, entries[0], raw)


def test_zero_risk_unchanged_metrics_cannot_hide_changed_training_label_bytes(tmp_path, monkeypatch):
    args, lock, entries, raw = cache_fixture(tmp_path)
    monkeypatch.setattr(r, "fit_ridge_once", lambda x, t, e: (np.zeros(x.shape[1]), dict(iterations=0)))
    for entry in entries:
        r.fit_unit(args, lock, entry, raw)
        r.replay_unit(args, lock, entry, raw)
    changed = deepcopy(raw)
    changed["t"][0] += .25  # An event time changes, preserving all ordering and censoring weights here.
    cache = r.load_cached(args, entries[0])
    original_metrics = r._evaluate(raw, cache, np.zeros(len(cache["valid_rows"])))
    assert r._evaluate(changed, cache, np.zeros(len(cache["valid_rows"]))) == original_metrics
    _, gradient = r.independent_efron_objective(cache["features"][:len(cache["train_rows"])],
        changed["t"][cache["train_rows"]], changed["e"][cache["train_rows"]], np.zeros(50))
    assert np.all(gradient == 0)
    args.resume = True
    for entry in entries:  # Both original-fit and alias must verify the original solver's labels.
        replay_path = r._directory(args, entry["representation"], entry["partition"]) / "REPLAY.json"
        previous = replay_path.read_bytes()
        with pytest.raises(ValueError, match="EXACT_TRAINING_LABEL_BYTES_CHANGED"):
            r.replay_unit(args, lock, entry, changed)
        with pytest.raises(ValueError, match="EXACT_TRAINING_LABEL_BYTES_CHANGED"):
            r.fit_unit(args, lock, entry, changed)
        assert replay_path.read_bytes() == previous


def test_partial_or_error_unit_is_preserved_without_any_solver_call(tmp_path, monkeypatch):
    args, lock, entries, raw = cache_fixture(tmp_path)
    directory = r._directory(args, entries[0]["representation"], 0); directory.mkdir(parents=True)
    sentinel = directory / "ERROR.json"; sentinel.write_text('{"synthetic":"preserved"}')
    monkeypatch.setattr(r, "fit_ridge_once", lambda *a: pytest.fail("must not start a solver"))
    with pytest.raises(ValueError, match="PRESERVED_NO_RETRY"):
        r.fit_unit(args, lock, entries[0], raw)
    assert sentinel.read_text() == '{"synthetic":"preserved"}'


def test_nonfinite_solver_failure_is_recorded_without_masking_original_error(tmp_path, monkeypatch):
    args, lock, entries, raw = cache_fixture(tmp_path)
    def failed(*_):
        raise r.RetentionSolverError(dict(actual_optimizer_fits=1, iterations=4,
            loss=float("nan"), gradient_max=float("inf")), np.zeros(50))
    monkeypatch.setattr(r, "fit_ridge_once", failed)
    with pytest.raises(r.RetentionSolverError):
        r.fit_unit(args, lock, entries[0], raw)
    directory = r._directory(args, entries[0]["representation"], 0)
    error = r.read(directory / "ERROR.json")
    assert error["known_solver_audit"]["loss"] is None and error["known_solver_audit"]["gradient_max"] is None
    assert error["known_solver_audit"]["iterations"] == 4
    assert (directory / "failed_coefficient.private.npz").is_file()
    assert not (directory / "COMPLETE.json").exists()


def test_full_metadata_report_counts_alias_once_and_partial_results_have_no_mean(tmp_path, monkeypatch):
    args = SimpleNamespace(private=tmp_path / "private", resume=False)
    args.private.mkdir()
    train, valid = np.array([0, 2, 4, 6]), np.array([1, 3, 5])
    raw = dict(t=np.arange(2., 9.), e=np.array([1, 1, 1, 0, 0, 1, 1]))
    entries = []
    for part in range(3):
        for name in r.REPRESENTATIONS:
            x = np.zeros((7, r.DIMENSIONS[name])); mask = np.ones_like(x, bool)
            path = r._cache_path(args.private, name, part); path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, features=x, measured_mask=mask, train_rows=train, valid_rows=valid, active_rows=np.r_[train, valid])
            entries.append(dict(representation=name, partition=part,
                optimizer_representation=r.REPRESENTATIONS[0] if name == r.REPRESENTATIONS[1] else name,
                cache_sha256=r.sha256(path), feature_values_sha256=r.array_hash(x), measured_mask_sha256=r.array_hash(mask),
                train_rows_sha256=r.array_hash(train), valid_rows_sha256=r.array_hash(valid), normalization=dict(dimension=x.shape[1])))
    references = [dict(representation=name, partition=part, status="REUSED_EXACT_PRIOR_ACTUAL_REPLAY",
                      parameters=512, lambda_L2=1., actual_optimizer_fits=0,
                      metrics=dict(Harrell=.5, Uno60=.5, Harrell_num=1., Harrell_den=2., Uno_num=1., Uno_den=2.))
                  for name in r.REFERENCE_NAMES for part in range(3)]
    lock = dict(contract_sha256="1" * 64, representations=entries, historical_references=references, planned_actual_optimizer_fits=15)
    calls = []
    def simulated(x, t, e):
        calls.append(1); return np.zeros(x.shape[1]), dict(iterations=0)
    monkeypatch.setattr(r, "fit_ridge_once", simulated)
    for entry in entries:
        r.fit_unit(args, lock, entry, raw)
    # Unknown receipt fields are never copied into public output.
    isolated = r._directory(args, r.REPRESENTATIONS[2], 0) / "COMPLETE.json"
    payload = r.read(isolated); payload["private_note"] = "DO_NOT_EXPORT_SYNTHETIC_PRIVATE_SENTINEL"
    isolated.write_text(json.dumps(payload))
    for entry in entries:
        r.replay_unit(args, lock, entry, raw)
    monkeypatch.setattr(r, "ROOT", tmp_path / "public")
    report = r.write_report(args, lock)
    assert len(calls) == report["confirmed_new_optimizer_fits"] == 15
    assert report["planned_logical_cells"] == 18 and report["status"] == "COMPLETE_18_LOGICAL_CELLS_AND_REPLAY"
    assert all(row["complete_partitions"] == 3 for row in report["summaries"])
    output = r.ROOT / "reports/bgb_en_r6"
    assert all("DO_NOT_EXPORT_SYNTHETIC_PRIVATE_SENTINEL" not in p.read_text(encoding="utf-8-sig") for p in output.iterdir())
    missing = r._directory(args, r.REPRESENTATIONS[2], 1) / "REPLAY.json"
    missing.rename(missing.with_name("REPLAY_SAVED.private.json"))
    partial = r.write_report(args, lock)
    summary = next(row for row in partial["summaries"] if row["representation"] == r.REPRESENTATIONS[2])
    assert partial["status"] == "INCONCLUSIVE_PARTIAL" and summary["complete_partitions"] == 2
    assert summary["Harrell"] is None and summary["Uno60"] is None


def test_gate_precedes_any_target_loader_and_source_failure_does_not_read_labels(tmp_path, monkeypatch):
    from scripts import bgb_en_r6_common as common, bgb_en_run as inherited
    args = SimpleNamespace(config=r.ROOT / r.CONFIG, launch_gate=tmp_path / "gate.json", launch_gate_sha256="")
    args.launch_gate.write_text(json.dumps(dict(schema="bgb_en_r6_source_launch_gate_v1", r6_formal_training_allowed=False)))
    args.launch_gate_sha256 = r.sha256(args.launch_gate)
    monkeypatch.setattr(inherited, "load_inputs", lambda *a: pytest.fail("target label loader entered before gate/source qualification"))
    with pytest.raises(ValueError, match="DOES_NOT_AUTHORIZE"):
        r.real_context(args)
    args.launch_gate.write_text(json.dumps(dict(schema="bgb_en_r6_source_launch_gate_v1", r6_formal_training_allowed=True,
        R5_decision="R5_READOUT_STOP_R6_ELIGIBLE", R5_public_evidence_sha256={name: "1" * 64 for name in (
            "decision.json", "R5_REPORT_AUDIT.json", "R5_CHECKPOINT_REPLAY_AUDIT.json", "PROTOCOL_LOCK.json")})))
    args.launch_gate_sha256 = r.sha256(args.launch_gate)
    monkeypatch.setattr(common, "configure_deterministic_runtime", lambda: {})
    def missing(*a):
        raise ValueError("SYNTHETIC_FULL_SOURCE_NOT_COMPLETE")
    monkeypatch.setattr(common, "source_foundations", missing)
    with pytest.raises(ValueError, match="FULL_SOURCE_NOT_COMPLETE"):
        r.real_context(args)


def test_private_cache_and_old_reference_paths_cannot_escape_allowed_contract(tmp_path):
    with pytest.raises(ValueError, match="OUTSIDE_GIT"):
        r.private_root(r.ROOT / "tmp/retention_patient_data")
    with pytest.raises(ValueError, match="HISTORICAL_DATA"):
        r.private_root(tmp_path / "source/new", readonly_roots=(tmp_path / "source",))
    args = SimpleNamespace(primary_private=tmp_path / "formal_v1")
    for cell in ("..", "../escape", "folder/escape"):
        with pytest.raises(ValueError, match="REFERENCE_CONTAINER"):
            r._reference_paths(args, dict(container="formal", cell=cell))


def test_no_target_trained_latent_or_Set_Mamba_optimizer_is_implemented():
    text = inspect.getsource(r)
    assert "SparsePathwayEncoder(" not in text and "torch.optim" not in text
    assert "clinical_raw" not in text and "cna_gain" not in text
    assert "configure_deterministic_runtime" in text
    assert "general_geneL2_equivalence_claimed=False" in text
