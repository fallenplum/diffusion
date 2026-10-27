"""Independent no-patient-data review of the frozen outer evaluation helpers."""
import copy
import json
import sys

import numpy as np
import pytest

from scripts import bgb_en_report_outer as report_outer
from scripts.bgb_en_report_outer import aggregate, comparisons
from scripts.bgb_en_nested import select_receipts
from src.bgb_en import protocol
from src.bgb_en.data import sha256
from src.bgb_en.replay import apply_saved_preprocessing, replay_full_risks


def _metric(h, u, dh, du):
    return dict(Harrell=h, Uno60=u, Harrell_num=h * dh, Harrell_den=dh,
                Uno_num=u * du, Uno_den=du)


def _rows():
    rows = []
    for seed in (31, 47, 83):
        for fold in range(1, 6):
            for name, improvement in [('CP_STRONG', 0.), ('CP_SCALAR', .01),
                                      ('NESTED_INTERACTION', .015)]:
                h = .5 + .01 * fold + .0001 * seed + improvement
                u = .45 + .02 * fold + .0002 * seed + improvement
                row = dict(model=name, endpoint='DSS', seed=seed, fold=fold,
                           **_metric(h, u, 11. * fold, 7. * fold * fold))
                if name != 'CP_STRONG':
                    row.update(repair=.02, damage=.02 - improvement,
                               Uno_repair=.03, Uno_damage=.03 - improvement,
                               risk_shift_sd_ratio=.08)
                if name == 'NESTED_INTERACTION':
                    row['_reference_metrics'] = dict(CP_SCALAR=dict(
                        repair=.007, damage=.002, Uno_repair=.009,
                        Uno_damage=.004, risk_shift_sd_ratio=.03))
                rows.append(row)
    return rows


def test_outer_aggregation_is_fold_pair_weighted_then_seed_mean():
    rows = _rows()
    result = aggregate(rows, 'CP_SCALAR', 'DSS')
    seed_h = []; seed_u = []
    for seed in (31, 47, 83):
        selected = [r for r in rows if r['model'] == 'CP_SCALAR' and r['seed'] == seed]
        expected_h = sum(r['Harrell_num'] for r in selected) / sum(r['Harrell_den'] for r in selected)
        expected_u = sum(r['Uno_num'] for r in selected) / sum(r['Uno_den'] for r in selected)
        assert result['seeds'][seed]['Harrell'] == expected_h
        assert result['seeds'][seed]['Uno60'] == expected_u
        seed_h.append(expected_h); seed_u.append(expected_u)
    assert result['Harrell'] == np.mean(seed_h)
    assert result['Uno60'] == np.mean(seed_u)
    naive = np.mean([r['Harrell'] for r in rows if r['model'] == 'CP_SCALAR'])
    assert abs(result['Harrell'] - naive) > .005


def test_missing_fold_cannot_be_reported_as_complete():
    rows = [r for r in _rows() if not (r['model'] == 'CP_SCALAR' and r['fold'] == 5 and r['seed'] == 47)]
    result = aggregate(rows, 'CP_SCALAR', 'DSS')
    assert result['status'] == 'NOT_ALL_FIVE_FOLDS_THREE_SEEDS'
    assert 47 not in result['seeds']
    assert 'Harrell' not in result


def test_outer_pairing_uses_same_fold_seed_and_correct_repair_reference():
    pairs = comparisons(_rows(), 'NESTED_INTERACTION', 'CP_SCALAR', 'DSS')
    assert len(pairs) == 18
    for row in pairs:
        assert row['candidate'] == 'NESTED_INTERACTION'
        assert row['reference'] == 'CP_SCALAR'
        assert abs(row['paired_delta_H'] - .005) < 1e-14
        assert abs(row['paired_delta_U60'] - .005) < 1e-14
        assert abs(row['repair'] - .007) < 1e-14
        assert abs(row['damage'] - .002) < 1e-14
        assert abs(row['Uno_repair'] - .009) < 1e-14
        assert abs(row['Uno_damage'] - .004) < 1e-14


def test_repair_fields_not_relabelled_against_another_reference():
    pairs = comparisons(_rows(), 'CP_SCALAR', 'NESTED_INTERACTION', 'DSS')
    for row in pairs:
        for key in ('repair', 'damage', 'Uno_repair', 'Uno_damage'):
            assert row.get(key) is None


def test_mismatched_eligible_pairs_or_ipcw_denominators_rejected():
    rows = _rows()
    next(r for r in rows if r['model'] == 'CP_SCALAR')['Uno_den'] += 1.
    with pytest.raises(ValueError, match='Different populations or IPCW'):
        comparisons(rows, 'CP_SCALAR', 'CP_STRONG', 'DSS')


def test_duplicate_fold_seed_is_rejected_in_aggregation_and_pairing():
    rows = _rows()
    rows.append(copy.deepcopy(next(r for r in rows if r['model'] == 'CP_SCALAR')))
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate(rows, 'CP_SCALAR', 'DSS')
    with pytest.raises(ValueError, match='Duplicate'):
        comparisons(rows, 'CP_SCALAR', 'CP_STRONG', 'DSS')


def test_all_three_pairwise_summaries_written_with_neural_finalist(tmp_path, monkeypatch):
    """Exercise actual report entrypoint using anonymous synthetic receipts only."""
    private = tmp_path / 'synthetic_private'; output = tmp_path / 'synthetic_report'
    lock = tmp_path / 'selection.json'
    lock.write_text(json.dumps(dict(finalists=['CP_STRONG', 'CP_SCALAR', 'NESTED_INTERACTION'])))
    lock_sha = sha256(lock)
    rows = _rows()
    for fold in range(1, 6):
        directory = private / 'outer' / f'fold{fold}'; directory.mkdir(parents=True)
        artifact = directory / 'models.pt'; artifact.write_bytes(b'synthetic-not-a-patient-model')
        results = {}
        for seed in (31, 47, 83):
            models = {}
            for name in ('CP_STRONG', 'CP_SCALAR'):
                row = next(r for r in rows if (r['fold'], r['seed'], r['model']) == (fold, seed, name))
                metric = {k: v for k, v in row.items() if k not in ('fold', 'seed', 'model', 'endpoint')}
                models[name] = dict(**metric, PFI_frozen_DSS_risk=metric)
            results[str(seed)] = dict(beta=.1, models=models)
        receipt = dict(status='COMPLETE_OUTER_BASELINES', fold=fold,
            contract=dict(selection_lock_sha256=lock_sha), artifact_sha256=sha256(artifact),
            results=results, actual_Cox_fits=25, actual_scalar_calibration_fits=3,
            elapsed_seconds=1., deterministic_CP_test_vector_fits=1, CP_seed_references=3)
        (directory / 'COMPLETE.json').write_text(json.dumps(receipt))
        selected = private / 'selection' / f'outer{fold}.json'; selected.parent.mkdir(exist_ok=True)
        selected.write_text(json.dumps(dict(outer_fold=fold, selected_arm='F5')))
        for seed in (31, 47, 83):
            directory = private / 'neural' / f'outer{fold}_seed{seed}'; directory.mkdir(parents=True)
            artifact = directory / 'model.pt'; artifact.write_bytes(b'synthetic-not-a-patient-neural-model')
            row = next(r for r in rows if (r['fold'], r['seed'], r['model']) == (fold, seed, 'NESTED_INTERACTION'))
            metric = {k: v for k, v in row.items() if k not in ('fold', 'seed', 'model', 'endpoint', '_reference_metrics')}
            reference = dict(metric, **row['_reference_metrics']['CP_SCALAR'])
            value = dict(status='COMPLETE_OUTER_NEURAL', fold=fold, seed=seed, epochs=80,
                selected_arm='F5', outer_test_read_for_selection=False,
                contract=dict(selection_lock_sha256=lock_sha, seed=seed, arm='F5', fold_lock_sha256=sha256(selected)),
                artifact_sha256=sha256(artifact), metrics=dict(CP_STRONG=metric, CP_SCALAR=reference),
                PFI_frozen_DSS_risk=metric, parameters=32, peak_GPU_bytes=0, elapsed_seconds=1.)
            (directory / 'COMPLETE.json').write_text(json.dumps(value))
    monkeypatch.setattr(sys, 'argv', ['bgb_en_report_outer.py', '--private', str(private),
                                     '--selection-lock', str(lock), '--output', str(output)])
    report_outer.main()
    text = (output / 'R4_FINAL_RESULTS.md').read_text(encoding='utf-8')
    for endpoint in ('DSS', 'PFI'):
        for candidate, reference in [('CP_SCALAR', 'CP_STRONG'),
                                     ('NESTED_INTERACTION', 'CP_STRONG'),
                                     ('NESTED_INTERACTION', 'CP_SCALAR')]:
            assert f'{endpoint}：{candidate}−{reference} ΔH=' in text
    audit = json.loads((output / 'R4_REPORT_AUDIT.json').read_text())
    assert audit['status'] == 'COMPLETE_LOCKED_FINALISTS'
    assert audit['actual_Cox_fits'] == 125
    assert audit['actual_scalar_fits'] == 15
    assert audit['actual_neural_fits'] == 15
    assert len(audit['outer_sources']) == 20


def _inner_receipts(tmp_path):
    config = dict(selection=dict(nested_interaction_candidates=['F2', 'F5']),
                  prototype=dict(neural_seeds=[31, 47]), training=dict(epochs=80))
    lineage = dict(synthetic_source='no_patient_data')
    for arm in ('F2', 'F5'):
        for seed in (31, 47):
            for part in range(3):
                directory = tmp_path / 'cells' / f'R3_outer2_inner{part}_{arm}_seed{seed}'
                directory.mkdir(parents=True)
                artifact = directory / 'model.pt'; artifact.write_bytes(b'synthetic-only')
                metric = _metric(.6 + .01 * (arm == 'F5'), .55, part + 1., 2. * part + 1.)
                receipt = dict(stage='R3', arm=arm, outer_fold=2, inner_partition=part, seed=seed,
                    status='COMPLETE', lineage=lineage, outer_test_evaluated=False, epochs=80,
                    artifact_sha256=sha256(artifact), metrics=metric, parameters=10)
                (directory / 'COMPLETE.json').write_text(json.dumps(receipt))
    return config, lineage


def test_nested_selection_uses_same_seed_partition_counts(tmp_path):
    config, lineage = _inner_receipts(tmp_path)
    chosen, candidates, receipts = select_receipts(tmp_path, 2, config, lineage)
    assert chosen['arm'] == 'F5'
    assert len(candidates) == 2
    assert len(receipts) == 12
    assert set(chosen['seeds']) == {'31', '47'}
    assert abs(chosen['Harrell'] - .61) < 1e-15


@pytest.mark.parametrize('field,replacement', [('stage', 'R2'), ('arm', 'F3'),
    ('outer_fold', 1), ('inner_partition', 2), ('seed', 83)])
def test_nested_rejects_receipt_swapped_between_units(tmp_path, field, replacement):
    config, lineage = _inner_receipts(tmp_path)
    path = tmp_path / 'cells' / 'R3_outer2_inner0_F2_seed31' / 'COMPLETE.json'
    value = json.loads(path.read_text()); value[field] = replacement
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='unit identity mismatch'):
        select_receipts(tmp_path, 2, config, lineage)


def _raw():
    rng = np.random.default_rng(27012)
    n, g = 64, 520
    clinical = {name: (np.arange(n) % 3).astype(str) for name in protocol.CLINICAL}
    clinical['AGE'] = (30 + np.arange(n) % 50).astype(str)
    clinical['AGE'][4] = 'missing'
    rna = rng.normal(size=(n, g)); rna[:, -1] = np.nan
    rna[3:7, 8] = np.nan
    symbols = np.array([f'SYNTHETIC_GENE_{j}' for j in range(g)])
    cna = rng.integers(-2, 3, size=(n, g)).astype(float); cna[2:4, 7] = np.nan
    cna_symbols = symbols.copy(); cna_symbols[-2] = 'SYNTHETIC_UNMATCHED'
    return dict(rna=rna, cna=cna, rna_symbols=symbols, cna_symbols=cna_symbols,
                clinical_raw=clinical, S32=rng.normal(size=(n, 32)),
                conch_local=rng.normal(size=(n, 40)), conch_mean=rng.normal(size=(n, 40)))


def test_saved_preprocessing_replays_all_channels_without_outcome_or_fit(monkeypatch):
    raw = _raw(); train = np.arange(48)
    expected = protocol.prepare_partition(raw, train, np.ones((2, 520), bool))
    # No endpoint fields exist; replacing all fitting functions must not matter.
    def refuse_fit(*args, **kwargs):
        raise AssertionError('Saved preprocessing unexpectedly refitted statistics')
    monkeypatch.setattr(protocol, 'scale_fit', refuse_fit)
    monkeypatch.setattr(protocol, 'clinical_fit', refuse_fit)
    monkeypatch.setattr(protocol, 'molecular_fit', refuse_fit)
    actual = apply_saved_preprocessing(raw, expected['state'])
    for name in ('C', 'CS', 'LOCAL', 'GLOBAL', 'R', 'rna', 'rna_mask',
                 'cna_gain', 'cna_loss', 'cna_mask'):
        if actual[name].dtype == bool:
            assert np.array_equal(actual[name], expected[name]), name
        else:
            assert np.allclose(actual[name], expected[name], rtol=0, atol=2e-14), name
    assert not actual['cna_mask'][:, -2].any()
    assert not actual['rna_mask'][:, -1].any()


def test_saved_preprocessing_keeps_train_rows_fixed_when_heldout_covariates_change():
    raw = _raw(); train = np.arange(48); held = np.arange(48, 64)
    state = protocol.prepare_partition(raw, train, np.ones((2, 520), bool))['state']
    original = apply_saved_preprocessing(raw, state)
    changed = copy.deepcopy(raw)
    changed['rna'][held] = 1e6
    changed['cna'][held] = -100.
    changed['S32'][held] *= 1e6
    changed['conch_local'][held] *= -1e6
    changed['conch_mean'][held] *= 1e6
    again = apply_saved_preprocessing(changed, state)
    for key in ('C', 'CS', 'LOCAL', 'GLOBAL', 'R', 'rna', 'rna_mask',
                'cna_gain', 'cna_loss', 'cna_mask'):
        assert np.array_equal(original[key][train], again[key][train]), key


def test_saved_preprocessing_rejects_changed_training_identity():
    raw = _raw()
    state = protocol.prepare_partition(raw, np.arange(48), np.ones((2, 520), bool))['state']
    state['fit'] = state['fit'][::-1].copy()
    with pytest.raises(ValueError, match='fit identity changed'):
        apply_saved_preprocessing(raw, state)


def test_full_risk_replay_uses_saved_teacher_scale_not_heldout_moments():
    features = dict(R=np.array([[1., 2.], [3., -1.], [10000., -20000.]]))
    models = dict(R=dict(coef=np.array([.2, -.3]), scale=dict(mean=.4, sd=2.)))
    actual = replay_full_risks(features, models)['R']
    assert np.array_equal(actual, (features['R'] @ models['R']['coef'] - .4) / 2.)
    features['R'][-1] *= 100.
    assert np.array_equal(replay_full_risks(features, models)['R'][:2], actual[:2])
