"""R7 report tests use fabricated metadata and deliberately invalid .pt bytes.

No experiment, private-cache guard, optimizer, model, or patient data is used.
The COMPLETE/PASS strings below are test fixtures, never formal evidence.
"""
from __future__ import annotations

import builtins
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bgb_r7_report_tested", ROOT / "scripts/bgb_en_r7_report.py")
report = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(report)
D = "d" * 64


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")
    return report.sha256(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stub_artifact(path, identity):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("SYNTHETIC_NOT_A_TORCH_CHECKPOINT\n" + identity).encode())
    return report.sha256(path)


def metric(harrell, uno, partition, *, undefined=False):
    # Unequal, distinct H/Uno supports catch fold means and wrong denominators.
    hd, ud = (10., 30., 90.)[partition], (5., 25., 120.)[partition]
    return dict(Harrell=None if undefined else harrell, Uno60=uno,
        Harrell_num=0. if undefined else harrell * hd, Harrell_den=0. if undefined else hd,
        Uno_num=uno * ud, Uno_den=ud)


def make_tree(root, *, mode="defined"):
    root.mkdir(parents=True, exist_ok=True)
    private, config_path = root / "private", root / "config.json"
    config = read_json(ROOT / "configs/bgb_en/r7_inner_v1.json")
    config_sha = write_json(config_path, config)
    origins = {str(p): hashlib.sha256(f"synthetic-prepared-{p}".encode()).hexdigest() for p in report.PARTITIONS}
    source = dict(checkpoint_sha256=D, completion_sha256=D, replay_sha256=D,
        complete_source_state_sha256=D, source_config_sha256=D, source_control="MATCHED", external_full_source_seed=31)
    lock = dict(schema="bgb_en_r7_target_protocol_lock_v1", status="LOCKED_BEFORE_R7_FITS",
        config_sha256=config_sha, source_config_sha256=D,
        sources={"scripts/bgb_en_r7_report.py": report.sha256(ROOT / "scripts/bgb_en_r7_report.py")},
        upstream_r6=dict(contract_sha256=D, replay_sha256=D, target_lock_sha256=D),
        prepared_receipt_sha256=origins, source_full_states={"MATCHED": source},
        calibration_receipt_sha256={f"p{p}_seed{s}":D for p in report.PARTITIONS for s in report.SEEDS},
        fixed_deterministic_runtime=dict(synthetic_runtime_only=True), planned_neural_fits=99,
        planned_target_optimizer_updates=7920, teacher_refits=0, source_refits=0, outer_test_access=False)
    lock["contract_sha256"] = report.canonical_hash(lock)
    write_json(private / "targetLOCK.json", lock)
    qualification = dict(status="PASS_R7_REAL_DATA_AND_READONLY_R6_TRANSFER", contract_sha256=lock["contract_sha256"],
        optimizer_instances=0, optimizer_updates=0, source_refits=0, teacher_refits=0, scalar_beta_refits=0,
        distinct_train_RMS_calibrations=9, calibration_fits_are_unsupervised_not_neural_fits=True,
        upstream_replay_audit_sha256=D, qualification_weights_saved=False, synthetic_metadata_only=True)
    qualification_sha = write_json(private / "QUALIFICATION.json", qualification)
    effects = dict(B0=(-.020, -.015), B1=(0., 0.), ADD=(.002, .001), RAW=(-.002, .003),
        NORM=(.003, .005), INDEP=(.006, .007), SLOT=(.009, .009), GATE1=(.020, .018),
        GATE2=(.019, .019), SLOT_WRONG_RNA=(.002, .002), GATE1_WRONG_RNA=(.008, .004),
        GATE1_WRONG_PATH=(.005, .003), GATE2_WRONG_RNA=(.003, .001))
    if mode == "mixed":
        effects["GATE1"] = (.020, -.020)
    scores = {}
    for ai, arm in enumerate(("B0", "B1", *report.ARMS)):
        for part in report.PARTITIONS:
            for sj, seed in enumerate(report.SEEDS):
                noise = 0. if arm in ("B0", "B1") else (sj - 1) * .001
                h = (.55, .68, .73)[part] + effects[arm][0] + noise
                u = (.53, .65, .70)[part] + effects[arm][1] + noise
                scores[(arm, part, seed)] = metric(h, u, part, undefined=mode == "undefined")
    units, references = [], []
    for arm in report.ARMS:
        ai = report.ARMS.index(arm)
        for part in report.PARTITIONS:
            for seed in report.SEEDS:
                unit = f"{arm}_p{part}_seed{seed}"
                directory = private / "cells" / unit
                artifact_sha = stub_artifact(directory / "model.pt", unit + "_model")
                checkpoint_hashes = {str(e):stub_artifact(directory / f"checkpoint_epoch{e}.pt", unit + f"_epoch{e}") for e in (20,40,80)}
                diagnostics = {str(e):dict(epoch=e, synthetic_metadata_only=True, optimizer_updates=0,
                    train_only=e == 60, interface_RMS_summary=dict(mean=1.),
                    gradient_probe=dict(scope="synthetic_scalar_metadata", L2=.01, new_optimizer_updates=0)) for e in (20,40,60,80)}
                for epoch, diagnostic in diagnostics.items():
                    write_json(directory / f"DIAGNOSTIC_epoch{epoch}.json", diagnostic)
                m = scores[(arm, part, seed)]
                c = dict(contract_sha256=lock["contract_sha256"], qualification_sha256=qualification_sha,
                    config_sha256=config_sha, source_config_sha256=D, arm=arm, partition=part, seed=seed,
                    prepared_origin=dict(prepared_receipt_sha256=origins[str(part)], prepared_artifact_sha256=D),
                    upstream_r6=lock["upstream_r6"], fixed_deterministic_runtime=lock["fixed_deterministic_runtime"],
                    epochs=80, teacher_refits=0, source_training_updates=0, outer_test_used=False)
                curves = [dict(epoch=e, optimizer_updates_completed=e, cox_loss=1. - e / 1000.,
                    parameter_regularizer=.02, train=copy.deepcopy(m), valid=copy.deepcopy(m),
                    checkpoint_selected_by_prelock=e == 80, outer_test_read=False) for e in (20,40,80)]
                cell = dict(status="COMPLETE_R7_TARGET_FIT", arm=arm, partition=part, seed=seed, contract=c,
                    epochs=80, optimizer_updates=80, actual_new_optimizer_fits=1, teacher_refits=0, source_training_updates=0,
                    outer_test_used=False, artifact_sha256=artifact_sha, checkpoint_artifact_sha256=checkpoint_hashes,
                    checkpoint_diagnostics=diagnostics, curves=curves, metrics=m,
                    metrics_vs=dict(CP_SCALAR=copy.deepcopy(m), CP_STRONG=copy.deepcopy(m), PFI_frozen_DSS_risk=copy.deepcopy(m)),
                    source_origin=dict(source_key="MATCHED", source_origin=source),
                    parameters=dict(requires_grad=20 + ai, registered=100 + ai), elapsed_seconds=.5, peak_GPU_bytes=0,
                    risk_diagnostics=dict(synthetic_metadata_only=True, train=dict(SD=.1), validation=dict(SD=.09)),
                    frozen_controls=dict(synthetic_metadata_only=True, intervention=dict(risk_SD_ratio=.01, frozen_not_retrained=True)),
                    primary_epoch_locked_before_training=80)
                completion_sha = write_json(directory / "COMPLETE.json", cell)
                checkpoints = [dict(epoch=e, status="PASS", checkpoint_sha256=checkpoint_hashes[str(e)],
                    actual_state_dict_forward_replayed=True, maximum_risk_error=1e-8, maximum_metric_error=0.,
                    changed_pair_credits=0, new_optimizer_updates=0) for e in (20,40,80)]
                units.append(dict(unit=unit, arm=arm, partition=part, seed=seed, status="PASS",
                    contract_sha256=lock["contract_sha256"], completion_sha256=completion_sha, artifact_sha256=artifact_sha,
                    qualification_sha256=qualification_sha, upstream_replay_audit_sha256=D,
                    actual_checkpoint_count=3, checkpoints=checkpoints, all_checkpoints_actual_forward_replayed=True,
                    actual_state_dict_forward_replayed=True, source_frozen_weight_hash_replayed=True,
                    saved_query_mapping_recomputed=True, changed_pair_credits=0, new_optimizer_updates=0))
    for arm in ("B0", "B1"):
        for part in report.PARTITIONS:
            for seed in report.SEEDS:
                value = dict(status="R7_BASELINE_REFERENCE", group="REFERENCE", arm=arm, partition=part, seed=seed,
                    contract_sha256=lock["contract_sha256"], actual_new_optimizer_fits=0, teacher_refits=0,
                    prepared_origin=dict(prepared_receipt_sha256=origins[str(part)]), deterministic_seed_reference=seed != 31,
                    metrics=scores[(arm, part, seed)], metrics_vs=dict(PFI_frozen_DSS_risk=scores[(arm, part, seed)]))
                digest = write_json(private / "references" / f"{arm}_p{part}_seed{seed}.json", value)
                references.append(dict(arm=arm, partition=part, seed=seed, status="PASS", receipt_sha256=digest,
                    independent_fixed_anchor_metrics_recomputed=True, new_optimizer_fits=0))
    pairs = []
    for comparison in config["paired_comparisons"]:
        candidate, reference = comparison["candidate"], comparison["reference"]
        for part in report.PARTITIONS:
            for seed in report.SEEDS:
                a, b = scores[(candidate,part,seed)], scores[(reference,part,seed)]
                value = copy.deepcopy(a)
                for label, metric_name, base in (("", "Harrell", .10), ("Uno_", "Uno60", .12)):
                    difference = None if a[metric_name] is None else a[metric_name] - b[metric_name]
                    value[label + "damage"] = None if difference is None else base + part * .01
                    value[label + "repair"] = None if difference is None else base + part * .01 + difference
                pairs.append(dict(candidate=candidate, reference=reference, partition=part, seed=seed,
                    candidate_vs_reference=value, direct_risk_pair_credits=True, primary_checkpoint_epoch=80,
                    bootstrap_or_CI_computed=False))
    replay = dict(status="PASS_COMPLETE_R7_ALL_CHECKPOINT_REPLAY", contract_sha256=lock["contract_sha256"],
        qualification_sha256=qualification_sha, expected_units=99, available_units=99, passed_units=99, failed_units=0,
        expected_checkpoints=297, passed_checkpoints=297, missing_or_unreplayed_units=[], units=units,
        zero_fit_baseline_reference_replays=references, paired_retrained_controls=pairs,
        new_optimizer_fits=0, new_preprocessing_fits=0, synthetic_metadata_only=True)
    write_json(private / "REPLAY_AUDIT.json", replay)
    return dict(root=root, private=private, config=config_path, scores=scores, lock=lock, replay=replay)


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    return make_tree(tmp_path_factory.mktemp("r7_synthetic_template"))


@pytest.fixture
def case(template, tmp_path):
    root = tmp_path / "case"; shutil.copytree(template["root"], root)
    return dict(template, root=root, private=root / "private", config=root / "config.json")


def pair_fixture(case, candidate="GATE1", reference="B1"):
    summaries = {arm:report.aggregate_arm(arm, case["scores"]) for arm in (candidate,reference)}
    rows = [r for r in case["replay"]["paired_retrained_controls"] if r["candidate"] == candidate and r["reference"] == reference]
    return summaries, copy.deepcopy(rows)


def test_complete_99_matrix_build_is_metadata_only_and_binds_33_anonymous_shards(case, tmp_path, monkeypatch):
    original_import, original_read = builtins.__import__, report.read
    reads = []
    def guarded_import(name, *args, **kwargs):
        assert name.split('.')[0] not in {'torch','numpy'}, 'report attempted model/array library import'
        return original_import(name, *args, **kwargs)
    def guarded_read(path):
        assert Path(path).suffix == '.json'
        reads.append(Path(path).name)
        return original_read(path)
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(report, 'read', guarded_read)
    output = tmp_path / "report"
    result = report.build(case["private"], case["config"], output)
    assert result["status"] == "COMPLETE_99_FITS_297_CHECKPOINT_REPLAYS"
    assert result["budget"]["formal_neural_fits"] == 99 and result["budget"]["formal_optimizer_updates"] == 7920
    assert result["budget"]["zero_fit_baseline_reference_rows"] == 18
    assert result["budget"]["source_refits"] == result["budget"]["teacher_or_beta_refits"] == 0
    assert result["decision"]["complete_actual_checkpoint_replays"] == 297
    assert result["decision"]["status"] == "DIRECTIONAL_PATIENT_CONTEXT_INCREMENT"
    assert result["decision"]["GATE2_selected_from_validation"] is False
    assert result["decision"]["outer_results"] == "NOT_RUN"
    parts = result["diagnostic_index"]["parts"]
    assert len(parts) == len({r["file"] for r in parts}) == 33
    raw_keys = {'patient_ids','barcodes','coordinates','risk_vector','state_dict','optimizer_state','predictions'}
    def anonymous(value):
        if isinstance(value, dict):
            assert not raw_keys.intersection(value)
            for item in value.values(): anonymous(item)
        elif isinstance(value, list):
            for item in value: anonymous(item)
    for part in parts:
        path = output / part["file"]
        assert path.stat().st_size == part["bytes"] <= 2 * 1024**2
        assert report.sha256(path) == part["sha256"]
        payload = read_json(path); anonymous(payload)
        assert payload["schema"] == "bgb_en_r7_diagnostic_shard_v1" and len(payload["rows"]) == part["rows"] == 3
        assert len({(r['arm'], r['partition']) for r in payload['rows']}) == 1
        assert {r['seed'] for r in payload['rows']} == set(report.SEEDS)
        assert all(set(r['checkpoint_diagnostics']) == {'20','40','60','80'} for r in payload['rows'])
    with (output / "R7_COSTS.csv").open(encoding='utf-8-sig',newline='') as stream:
        costs = list(csv.DictReader(stream))
    assert len(costs) == 99 and sum(int(r['fits']) for r in costs) == 99 and sum(int(r['updates']) for r in costs) == 7920
    with (output / "R7_LEARNING_CURVES.csv").open(encoding='utf-8-sig',newline='') as stream:
        curves = list(csv.DictReader(stream))
    assert len(curves) == 297 and sum(r['primary_checkpoint'] == 'True' for r in curves) == 99
    assert all(r['epoch'] == '80' for r in curves if r['primary_checkpoint'] == 'True')
    assert 'model.pt' not in reads and any(p.name == 'model.pt' and p.read_bytes().startswith(b'SYNTHETIC_NOT_A_TORCH') for p in case['private'].glob('cells/*/model.pt'))


def test_components_sum_before_seed_mean_not_fold_mean(template):
    summary = report.aggregate_arm('GATE1', template['scores'])
    expected = statistics.mean(sum(template['scores'][('GATE1',p,s)]['Harrell_num'] for p in report.PARTITIONS) /
        sum(template['scores'][('GATE1',p,s)]['Harrell_den'] for p in report.PARTITIONS) for s in report.SEEDS)
    wrong = statistics.mean(template['scores'][('GATE1',p,s)]['Harrell'] for p in report.PARTITIONS for s in report.SEEDS)
    assert summary['Harrell'] == pytest.approx(expected, abs=1e-12)
    assert abs(summary['Harrell'] - wrong) > .03


def test_nine_direct_pair_credits_use_their_own_metric_denominator(template):
    summaries, rows = pair_fixture(template)
    result = report.pair_summary('GATE1','B1',summaries,rows)
    first = result['per_seed'][0]
    source = [r['candidate_vs_reference'] for r in rows if r['seed'] == report.SEEDS[0]]
    expected_h = sum(r['repair'] * r['Harrell_den'] for r in source) / sum(r['Harrell_den'] for r in source)
    expected_u = sum(r['Uno_repair'] * r['Uno_den'] for r in source) / sum(r['Uno_den'] for r in source)
    assert first['repair'] == pytest.approx(expected_h) and first['Uno_repair'] == pytest.approx(expected_u)
    assert first['repair'] != pytest.approx(statistics.mean(r['repair'] for r in source))
    assert first['Uno_repair'] != pytest.approx(sum(r['Uno_repair'] * r['Harrell_den'] for r in source) / sum(r['Harrell_den'] for r in source))
    for row in result['per_seed']:
        assert row['repair'] - row['damage'] == pytest.approx(row['delta_Harrell'],abs=1e-12)
        assert row['Uno_repair'] - row['Uno_damage'] == pytest.approx(row['delta_Uno60'],abs=1e-12)
    assert result['repair_damage_are_normalized_pair_credits_not_unique_patient_pair_counts'] is True
    assert result['joint_directional_support'] is True


def test_mixed_metrics_never_pass_joint_or_final_gate(tmp_path):
    data = make_tree(tmp_path / 'mixed',mode='mixed')
    summaries, rows = pair_fixture(data)
    pair = report.pair_summary('GATE1','B1',summaries,rows)
    assert pair['delta_Harrell'] > 0 and pair['delta_Uno60'] < 0
    assert pair['mixed_metric_direction'] is True and pair['joint_directional_support'] is False
    result = report.build(data['private'],data['config'],tmp_path / 'mixed_report')
    assert result['decision']['status'] == 'CURRENT_GATE1_NOT_SUPPORTED'


def test_undefined_harrell_stays_none_and_report_is_inconclusive(tmp_path):
    data = make_tree(tmp_path / 'undefined',mode='undefined')
    summaries, rows = pair_fixture(data)
    pair = report.pair_summary('GATE1','B1',summaries,rows)
    assert pair['delta_Harrell'] is None and pair['delta_Uno60'] > 0
    assert pair['repair'] is pair['damage'] is None
    assert pair['joint_directional_support'] is False and pair['metric_status'] == 'INCONCLUSIVE_UNDEFINED_METRIC'
    result = report.build(data['private'],data['config'],tmp_path / 'undefined_report')
    assert result['decision']['status'] == 'INCONCLUSIVE_UNDEFINED_METRIC'
    assert all(r['Harrell'] is None for r in result['primary_DSS'].values())
    assert 'NA' in (tmp_path / 'undefined_report/RESULTS.md').read_text(encoding='utf-8')


@pytest.mark.parametrize('field', ('all_checkpoints_actual_forward_replayed','source_frozen_weight_hash_replayed','saved_query_mapping_recomputed'))
def test_required_actual_unit_replay_flags_missing_is_rejected(template,field):
    replay = copy.deepcopy(template['replay']); replay['units'][0].pop(field)
    with pytest.raises(ValueError,match='R7_UNIT_NOT_THREE_ACTUAL_REPLAYS'):
        report.validate_replay(replay,template['lock'])


def test_missing_actual_checkpoint_flag_is_rejected(template):
    replay = copy.deepcopy(template['replay']); replay['units'][0]['checkpoints'][1].pop('actual_state_dict_forward_replayed')
    with pytest.raises(ValueError,match='R7_SAVED_CHECKPOINT_NOT_ACTUALLY_REPLAYED'):
        report.validate_replay(replay,template['lock'])


@pytest.mark.parametrize('kind,code', (
    ('source_sha','R7_SCIENTIFIC_SOURCE_BYTES_CHANGED'),
    ('completion','R7_CELL_REPLAY_SHA_CHANGED'),
    ('artifact','R7_CELL_REPLAY_SHA_CHANGED'),
    ('checkpoint_replay_sha','R7_CHECKPOINT_SHA_CHANGED'),
    ('diagnostic','R7_DIAGNOSTIC_RECEIPT_CHANGED'),
))
def test_tampered_source_receipt_or_bound_checkpoint_rejected(case,tmp_path,kind,code):
    private = case['private']
    if kind == 'source_sha':
        lock = read_json(private/'targetLOCK.json'); lock['sources']['scripts/bgb_en_r7_report.py'] = '0' * 64
        lock.pop('contract_sha256'); lock['contract_sha256'] = report.canonical_hash(lock); write_json(private/'targetLOCK.json',lock)
    elif kind == 'completion':
        path = private/'cells/ADD_p0_seed31/COMPLETE.json'; value=read_json(path); value['elapsed_seconds']=99.; write_json(path,value)
    elif kind == 'artifact':
        path = private/'cells/ADD_p0_seed31/model.pt'; path.write_bytes(path.read_bytes()+b'TAMPER')
    elif kind == 'checkpoint_replay_sha':
        path = private/'REPLAY_AUDIT.json'; value=read_json(path); value['units'][0]['checkpoints'][0]['checkpoint_sha256']='0' * 64; write_json(path,value)
    else:
        path=private/'cells/ADD_p0_seed31/DIAGNOSTIC_epoch20.json'; value=read_json(path); value['interface_RMS_summary']['mean']=2.; write_json(path,value)
    with pytest.raises(ValueError,match=code):
        report.build(private,case['config'],tmp_path/'must_not_complete')
    assert not (tmp_path/'must_not_complete/R7_RESULTS.json').exists()


def test_pair_credit_range_identity_and_complete_nine_grid_are_required(template):
    summaries, rows = pair_fixture(template)
    bad = copy.deepcopy(rows); bad[0]['candidate_vs_reference']['repair']=1.01
    with pytest.raises(ValueError,match='R7_INVALID_NORMALIZED_PAIR_CREDIT'):
        report.pair_summary('GATE1','B1',summaries,bad)
    bad = copy.deepcopy(rows); bad[0]['candidate_vs_reference']['repair'] += .01
    with pytest.raises(ValueError,match='R7_DIRECT_HARRELL_REPAIR_DAMAGE_DOES_NOT_MATCH_PAIRED_DELTA'):
        report.pair_summary('GATE1','B1',summaries,bad)
    with pytest.raises(ValueError,match='R7_ALL_NINE_DIRECT_RISK_PAIR_RECORDS_REQUIRED'):
        report.pair_summary('GATE1','B1',summaries,rows[:-1])

