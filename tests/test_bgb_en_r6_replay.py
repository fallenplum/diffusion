"""Synthetic actual forwards plus structural 90-cell accounting; zero fits."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import bgb_en_r6_replay as r
from scripts import bgb_en_r6_common as c
from src.bgb_en.r6_training import predict, saved_state, state_hash, metrics_for
from src.bgb_en.survival import evaluate
_spec = importlib.util.spec_from_file_location("r6_training_test_helpers", Path(__file__).with_name("test_bgb_en_r6_training.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
wrapper = _helpers.wrapper


def actual_fixture(tmp_path, monkeypatch, *, tie=False):
    c.configure_deterministic_runtime()
    model, prepared = wrapper()
    if tie:
        with torch.no_grad():
            for head in (model.core.path_main, model.core.gene_main, model.core.hallmark_main,
                         model.core.path_cross, model.core.gene_cross):
                head.weight.zero_()
        model.offset[4:] = 0.
        prepared["base"][4:] = 0.
    raw = dict(t=np.array([1., 2., 2., 70., 5., 7.]), e=np.array([1, 1, 1, 0, 1, 0]),
        pfi_valid=np.ones(6, bool), pfi_t=np.array([1., 2., 2., 70., 5., 7.]), pfi_e=np.array([1, 1, 1, 0, 1, 0]))
    config, source_config = tmp_path/"target.json", tmp_path/"source.json"
    config.write_text("{}"); source_config.write_text("{}")
    args = SimpleNamespace(config=config, source_config=source_config, device="cpu")
    source = dict(complete_source_state_sha256=model.core.source_state_sha256)
    source_origin = dict(source_key="MATCHED", source_origin=source, pairing={}, patch_cache_receipt_sha256="c"*64)
    ctx = dict(raw=raw, lock=dict(contract_sha256="a"*64), source_audits=dict(MATCHED=source))
    origin = dict(prepared_receipt_sha256="b"*64)
    contract = dict(contract_sha256="a"*64, prepared_origin=origin, qualification_sha256="q"*64,
        config_sha256=c.sha256(config), source_config_sha256=c.sha256(source_config),
        fixed_deterministic_runtime=c.validate_deterministic_runtime(), arm="B7", partition=0, seed=31, epochs=80)
    initialization = dict(full_state_sha256=state_hash(saved_state(model)),
                          target_state_sha256=state_hash(saved_state(model), target_only=True))
    fresh = deepcopy(model)
    model.refresh_training_query_mean()
    tp, vp = predict(model, prepared["train"]), predict(model, prepared["valid"])
    controls = {tag: predict(model, prepared["valid"], control=tag) for tag in r.FROZEN_CONTROLS}
    vm, tm = metrics_for(raw, prepared, prepared["valid"], vp), metrics_for(raw, prepared, prepared["train"], tp)
    frozen = {}
    train, valid = prepared["train"], prepared["valid"]
    for tag, value in controls.items():
        frozen[tag] = dict(metrics_vs_base=evaluate(raw["t"][train], raw["e"][train], raw["t"][valid], raw["e"][valid],
            value["risk"], prepared["base"][valid]), relative_to_trained=evaluate(raw["t"][train], raw["e"][train],
            raw["t"][valid], raw["e"][valid], vp["risk"], value["risk"]))
    metadata = dict(status="COMPLETE_R6_TARGET_FIT", arm="B7", partition=0, seed=31, contract=contract,
        epochs=80, optimizer_updates=80, actual_new_optimizer_fits=1, teacher_refits=0, source_training_updates=0,
        outer_test_used=False, source_origin=source_origin, initialization=initialization,
        metrics=vm["CP_SCALAR"], metrics_vs=vm, train_metrics=tm, frozen_controls=frozen)
    artifact = dict(contract=contract, model_origin=source_origin, train=train, valid=valid,
        epochs=80, optimizer_updates=80, initialization=initialization,
        final_state_dict=saved_state(model), snapshots={e: dict(state_dict=saved_state(model)) for e in (20, 40, 80)},
        query_map=model.query_map.copy(), mean_query=model.mean_context.tokens.cpu(),
        mean_query_mask=model.mean_context.program_mask.cpu(), predictions=dict(train=tp, validation=vp),
        frozen_control_predictions=controls)
    receipt = tmp_path/"cells/B7_p0_seed31/COMPLETE.json"
    monkeypatch.setattr(r, "construct", lambda *a, **k: (deepcopy(fresh), source_origin))
    return args, ctx, prepared, origin, receipt, metadata, artifact


def test_actual_state_forward_replay_recomputes_all_controls_and_direct_metrics(tmp_path, monkeypatch):
    values = actual_fixture(tmp_path, monkeypatch)
    result, risk = r.replay_one(*values, "q"*64)
    assert result["status"] == "PASS" and result["actual_state_dict_forward_replayed"]
    assert result["changed_pair_credits"] == 0 and result["maximum_risk_error"] <= 2e-6
    assert set(result["frozen_control_risk_errors"]) == set(r.FROZEN_CONTROLS)
    assert risk.shape == (2,)


@pytest.mark.parametrize("kind, message", [("risk", "NUMERIC_TOLERANCE"),
    ("metric", "INDEPENDENT_METRICS"), ("split", "PATIENT_SCOPE"),
    ("source", "SOURCE_FAMILY"), ("control", "FROZEN_QUERY_RISK")])
def test_actual_replay_rejects_saved_corruption(tmp_path, monkeypatch, kind, message):
    args, ctx, prepared, origin, receipt, metadata, artifact = actual_fixture(tmp_path, monkeypatch)
    if kind == "risk":
        artifact["predictions"]["validation"]["risk"][0] += 1e-4
    elif kind == "metric":
        metadata["metrics_vs"]["CP_SCALAR"]["Harrell_num"] += 1.
    elif kind == "split":
        artifact["valid"] = artifact["valid"][::-1]
    elif kind == "source":
        metadata["source_origin"] = dict(metadata["source_origin"], source_key="WRONG_DONOR")
    else:
        artifact["frozen_control_predictions"]["wrong_query"]["risk"][0] += 1e-4
    with pytest.raises(ValueError, match=message):
        r.replay_one(args, ctx, prepared, origin, receipt, metadata, artifact, "q"*64)


def test_pair_credit_change_is_rejected_even_below_numeric_tolerance(tmp_path, monkeypatch):
    args, ctx, prepared, origin, receipt, metadata, artifact = actual_fixture(tmp_path, monkeypatch, tie=True)
    artifact["predictions"]["validation"]["risk"][0] += 5e-8
    with pytest.raises(ValueError, match="PAIR_CREDIT_CHANGED"):
        r.replay_one(args, ctx, prepared, origin, receipt, metadata, artifact, "q"*64)


def structural_fixture(tmp_path, monkeypatch, *, all_cells=True, fail=False):
    c.configure_deterministic_runtime()
    args = SimpleNamespace(private=tmp_path, limit=0, resume=False)
    c.write(tmp_path/"QUALIFICATION.json", dict(status="SYNTHETIC_TEST_ONLY"))
    ctx = dict(lock=dict(contract_sha256="a"*64,
        sources={"scripts/bgb_en_r6_replay.py": c.sha256(r.__file__)}), raw=dict(
            t=np.array([1., 2., 3., 4.]), e=np.array([1, 1, 1, 0]),
            pfi_valid=np.ones(4, bool), pfi_t=np.array([1., 2., 3., 4.]), pfi_e=np.array([1, 1, 1, 0])))
    prepared = dict(train=np.array([0, 1]), valid=np.array([2, 3]), base=np.zeros(4))
    units = r.planned_units()
    if not all_cells:
        units = dict(list(units.items())[:1])
    for unit, (arm, part, seed) in units.items():
        directory = tmp_path/"cells"/unit
        directory.mkdir(parents=True)
        artifact = directory/"model.pt"
        torch.save(dict(predictions=dict(validation=dict(risk=np.array([.7, .2])))), artifact)
        c.write(directory/"COMPLETE.json", dict(status="COMPLETE_R6_TARGET_FIT", arm=arm,
            partition=part, seed=seed, artifact_sha256=c.sha256(artifact)))
    calls = []
    monkeypatch.setattr(r, "validate_qualification", lambda *a: c.sha256(tmp_path/"QUALIFICATION.json"))
    monkeypatch.setattr(r, "qualified_prepared", lambda *a: (prepared, dict(prepared_receipt_sha256="b"*64)))
    monkeypatch.setattr(r, "upstream_audit", lambda *a: (dict(status="SYNTHETIC_TEST_ONLY"), "c"*64))
    monkeypatch.setattr(r, "reference_replay", lambda *a: [])
    monkeypatch.setattr(r, "public_snapshot_guard", lambda: {})
    monkeypatch.setattr(r, "validate_entry", lambda *a: None)

    def replay(*values):
        calls.append(values[4].parent.name)
        if fail:
            raise ValueError("SYNTHETIC_TEST_FAILURE")
        return dict(status="PASS", actual_state_dict_forward_replayed=True, new_optimizer_updates=0), np.array([.7, .2])
    monkeypatch.setattr(r, "replay_one", replay)
    return args, ctx, calls


def test_all_90_required_and_direct_81_plus_9_pair_rows(tmp_path, monkeypatch):
    args, ctx, calls = structural_fixture(tmp_path, monkeypatch)
    args.limit = 1
    partial = r.run(args, ctx)
    assert partial["status"].startswith("INCOMPLETE") and partial["passed_units"] == 1
    assert len(partial["missing_or_unreplayed_units"]) == 89
    args.resume, args.limit = True, 0
    complete = r.run(args, ctx)
    assert complete["status"] == "PASS_COMPLETE_R6_TARGET_CHECKPOINT_REPLAY"
    assert complete["passed_units"] == len(calls) == 90
    assert len(complete["paired_retrained_controls"]) == 81
    assert len(complete["paired_raw_gene_controls"]) == 9
    again = r.run(args, ctx)
    assert again["new_actual_replays"] == 0 and len(calls) == 90
    assert again["passed_units"] == 90


def test_missing_cells_cannot_get_subset_pass(tmp_path, monkeypatch):
    args, ctx, _ = structural_fixture(tmp_path, monkeypatch, all_cells=False)
    result = r.run(args, ctx)
    assert result["status"].startswith("INCOMPLETE") and result["available_units"] == 1
    assert result["passed_units"] == 1 and len(result["missing_or_unreplayed_units"]) == 89
    assert result["paired_retrained_controls"] == []


def test_failures_are_immutable_and_never_automatically_retried(tmp_path, monkeypatch):
    args, ctx, calls = structural_fixture(tmp_path, monkeypatch, all_cells=False, fail=True)
    result = r.run(args, ctx)
    assert result["failed_units"] == len(calls) == 1
    failure = next((tmp_path/"REPLAY_PRIVATE").glob("*.FAIL.json"))
    before = failure.read_bytes()
    args.resume = True
    result = r.run(args, ctx)
    assert result["failed_units"] == 1 and len(calls) == 1
    assert failure.read_bytes() == before


def test_resume_rejects_changed_artifact_or_completion_hash(tmp_path, monkeypatch):
    args, ctx, _ = structural_fixture(tmp_path, monkeypatch, all_cells=False)
    r.run(args, ctx)
    args.resume = True
    artifact = next((tmp_path/"cells").glob("*/model.pt"))
    with artifact.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="RESUMED_REPLAY_ORIGIN_CHANGED"):
        r.run(args, ctx)


def test_zero_fit_baseline_metrics_are_recomputed_and_tampering_fails(tmp_path, monkeypatch):
    args, ctx, prepared, origin, *_ = actual_fixture(tmp_path, monkeypatch)
    args.private = tmp_path
    from scripts.bgb_en_r6_run import make_references
    for part in r.PARTITIONS:
        args.partition = part
        make_references(args, ctx, prepared, origin)
    prepared_cache = {p: prepared for p in r.PARTITIONS}
    origins = {p: origin for p in r.PARTITIONS}
    rows = r.reference_replay(args, ctx, prepared_cache, origins)
    assert len(rows) == 18 and all(row["new_optimizer_fits"] == 0 for row in rows)
    path = tmp_path/"references/B1_p0_seed31.json"
    value = c.read(path)
    value["metrics"]["Harrell_num"] += 1.
    c.write(path, value)
    with pytest.raises(ValueError, match="REFERENCE_METRICS_CHANGED"):
        r.reference_replay(args, ctx, prepared_cache, origins)
