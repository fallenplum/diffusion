"""Synthetic temporary checkpoints exercise evaluation; no real predictor fits."""
import json

import numpy as np
import pytest
import torch

from src.phase25_local import evaluation as ev

SOURCE = 'a'*64
N = 12


def endpoint():
    return {'DSS': {'time': np.arange(1., N+1), 'event': np.ones(N), 'valid': np.ones(N, bool)}}


def moment(n):
    return {'count': n, 'mean': .1, 'population_SD': .02, 'RMS': .102, 'minimum': 0., 'maximum': .2}


def mechanism(arm, fit_n, valid_n):
    changes = {}
    for key in ev.SENSITIVITIES[arm]:
        changed = {'patient_risk_difference_over_fit_offset_SD': moment(valid_n)}
        if key != 'OFF':
            changed.update(attention_TV=moment(valid_n), top32_Jaccard=moment(valid_n),
                           scope='WSI_weighted_same_valid_patches_not_histological_ground_truth')
        changes[key] = changed
    return {'fit_unit_residual': moment(fit_n), 'valid_unit_residual': moment(valid_n),
            'attention': {key: moment(valid_n) for key in
                          ('normalized_entropy', 'effective_patch_fraction', 'TV_from_uniform')},
            'frozen_query_changes': changes}


def synthetic_cell(root, seed, fold):
    valid = np.arange(sum(range(2, fold+1)), sum(range(2, fold+2)), dtype=int)
    fit = np.setdiff1d(np.arange(N), valid)
    rng = np.random.default_rng(seed+100*fold)
    r0 = np.linspace(-.2, .3, len(valid))
    record = {'source': SOURCE, 'stage': 'prototype', 'seed': seed, 'fold': fold,
              'fit': fit, 'valid': valid, 'R0_valid': r0, 'CP_valid': r0+.01, 'arms': {}}
    for arm in ev.NEURAL_ARMS:
        record['arms'][arm] = {
            'risk': rng.normal(size=len(valid)), 'fit_risk': rng.normal(size=len(fit)),
            'model_state': {'weight': torch.ones(2)}, 'initial_state': {'weight': torch.zeros(2)},
            'history': [{'epoch': epoch, 'training_loss': 1.1, 'fit_loss': 1., 'fit_cox': .9,
                         'unit_residual_RMS': .1, 'grad_norm': .2, 'gradients': {'keys': .1},
                         'epoch_seconds': 1.} for epoch in range(1, 81)],
            'training_audit': {'actual_epochs': 80, 'optimizer_steps': 80, 'heldout_endpoint_reads': 0,
                              'fixed_final_epoch': True, 'main_offset_unchanged': True,
                              'fit_offset_kind': 'strict_threefold_crossfit_R0'},
            'parameter_counts': {'trainable_local': 8, 'common_frozen_main': 5},
            'sensitivities': {name: r0.copy() if name == 'OFF' else rng.normal(size=len(valid))
                              for name in ev.SENSITIVITIES[arm]},
            'mechanism': mechanism(arm, len(fit), len(valid)),
        }
    folder = root/'prototype'/f'seed{seed}_fold{fold}'
    folder.mkdir(parents=True)
    torch.save(record, folder/'checkpoint.pt')
    ev.write_json(folder/'DONE_PRIVATE.json', {'status': 'COMPLETE', 'source': SOURCE,
        'stage': 'prototype', 'seed': seed, 'fold': fold, 'model_slots': 4, 'epochs': 80,
        'neural_models': 3, 'optimizer_steps': 240, 'convex_main_shared_reference': True,
        'checkpoint_sha256': ev.sha256_file(folder/'checkpoint.pt')})
    return folder


def full_stage(root):
    for seed in ev.SEEDS:
        for fold in ev.FOLDS:
            synthetic_cell(root, seed, fold)


def rewrite(folder, change):
    record = torch.load(folder/'checkpoint.pt', weights_only=False)
    change(record)
    torch.save(record, folder/'checkpoint.pt')
    receipt = json.loads((folder/'DONE_PRIVATE.json').read_text())
    receipt['checkpoint_sha256'] = ev.sha256_file(folder/'checkpoint.pt')
    ev.write_json(folder/'DONE_PRIVATE.json', receipt)


def run(root, loader=endpoint, **kwargs):
    return ev.evaluate_stage(root, root/'public', 'prototype', SOURCE, loader,
                             cohort_size=N, expected_dss_events=N, **kwargs)


def test_missing_cell_blocks_endpoint_request(tmp_path):
    calls = []
    with pytest.raises(ValueError, match='missing complete'):
        run(tmp_path, lambda: calls.append(1))
    assert calls == [] and not (tmp_path/'public').exists()


def test_all_six_hashes_pass_before_any_deserialization(tmp_path, monkeypatch):
    full_stage(tmp_path)
    (tmp_path/'prototype/seed47_fold3/checkpoint.pt').write_bytes(b'tampered')
    calls = []
    monkeypatch.setattr(torch, 'load', lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError, match='hash mismatch'):
        run(tmp_path, lambda: pytest.fail('endpoints read'))
    assert calls == []


@pytest.mark.parametrize('key,value', [('model_slots', 3), ('epochs', 79), ('neural_models', 4),
    ('optimizer_steps', 239), ('convex_main_shared_reference', False), ('source', 'b'*64)])
def test_receipt_budget_identity_fails_closed_before_labels(tmp_path, key, value):
    full_stage(tmp_path)
    path = tmp_path/'prototype/seed31_fold1/DONE_PRIVATE.json'
    receipt = json.loads(path.read_text())
    receipt[key] = value
    ev.write_json(path, receipt)
    with pytest.raises(ValueError, match='receipt mismatch'):
        run(tmp_path, lambda: pytest.fail('endpoints read'))


@pytest.mark.parametrize('change,match', [
    (lambda r: r['arms']['R2']['risk'].__setitem__(0, np.nan), 'nonfinite'),
    (lambda r: r['arms']['R3']['history'].pop(), '80 history'),
    (lambda r: r['arms']['R1']['history'][2].__setitem__('epoch', 7), 'epoch sequence'),
    (lambda r: r['arms']['R2']['history'][1]['gradients'].__setitem__('keys', np.nan), 'nonfinite'),
    (lambda r: r['arms']['R1']['training_audit'].__setitem__('heldout_endpoint_reads', 1), 'isolation audit'),
    (lambda r: r['arms']['R3']['sensitivities'].pop('C_MEAN'), 'sensitivity set'),
    (lambda r: r['arms']['R2']['sensitivities']['OFF'].__setitem__(0, 100), 'OFF must equal'),
    (lambda r: r['arms']['R3']['model_state']['weight'].__setitem__(0, np.inf), 'finite tensors'),
    (lambda r: r['arms']['R1']['parameter_counts'].__setitem__('trainable_local', -1), 'counts invalid'),
    (lambda r: r['arms']['R2']['mechanism'].__setitem__('patient_indices', [1, 2]), 'not reviewed aggregates'),
    (lambda r: r['arms']['R2']['mechanism']['valid_unit_residual'].__setitem__('values', [1, 2]), 'aggregate moments only'),
    (lambda r: r['arms']['R3']['mechanism']['frozen_query_changes']['WRONG_C'].__setitem__('raw_risk', [1, 2]), 'not reviewed'),
    (lambda r: r['fit'].__setitem__(0, r['valid'][0]), 'leakage'),
    (lambda r: r.__setitem__('event', np.ones(N)), 'endpoint arrays'),
])
def test_rehashed_invalid_checkpoint_blocks_labels(tmp_path, change, match):
    full_stage(tmp_path)
    rewrite(tmp_path/'prototype/seed31_fold1', change)
    with pytest.raises(ValueError, match=match):
        run(tmp_path, lambda: pytest.fail('endpoints read'))


def test_shared_CP_guard_must_be_bit_identical_across_seeds(tmp_path):
    full_stage(tmp_path)
    rewrite(tmp_path/'prototype/seed47_fold2', lambda r: r['CP_valid'].__setitem__(0, .9))
    with pytest.raises(ValueError, match='baseline differs across seeds'):
        run(tmp_path, lambda: pytest.fail('endpoints read'))


def test_independent_baseline_is_checked_before_endpoint_request(tmp_path):
    full_stage(tmp_path)
    cells, _ = ev.verify_stage(tmp_path, 'prototype', SOURCE, cohort_size=N)
    def baseline(fold):
        cell = cells[31, fold]
        return {'fit': cell['fit'], 'valid': cell['valid'], 'R0_valid': cell['risks']['R0']+1,
                'CP_valid': cell['risks']['CP']}
    with pytest.raises(ValueError, match='differs elementwise'):
        run(tmp_path, lambda: pytest.fail('endpoints read'), baseline_loader=baseline)


def test_complete_prototype_outputs_anonymous_diagnostic_and_no_outer_launch(tmp_path):
    full_stage(tmp_path)
    calls = []
    result, gate = run(tmp_path, lambda: calls.append(1) or endpoint())
    assert calls == [1]
    assert result['artifact_audit']['complete_cells'] == 6
    assert result['artifact_audit']['evaluation_model_slots'] == 24
    assert result['artifact_audit']['neural_models'] == 18
    assert not result['unique_patient_OOF'] and not result['confirmatory_CI']
    assert not result['artifact_audit']['encoder_replay_claimed']
    assert len(result['fold_rows']) == 6*len(ev.RISK_NAMES)
    assert len(result['seed_rows']) == 2*len(ev.RISK_NAMES)
    assert len(result['mechanism_rows']) == 18
    assert result['mechanism_rows'][0]['training']['last10_fit_loss_improvement'] == 0.
    assert result['mechanism_rows'][0]['training']['gradient_groups']['keys']['zero_epochs'] == 0
    for forbidden in ('model_state', 'fit_risk', 'R0_valid', 'CP_valid', '"fit":', '"valid":'):
        assert forbidden not in json.dumps(result)
    private = json.loads((tmp_path/'INNER_GATE.json').read_text())
    assert private['decision'] in ('GO', 'STOP')
    assert private['public_gate_sha256'] == ev.sha256_file(tmp_path/'public/INNER_GATE.json')
    assert not gate['outer_automatically_started']
    with pytest.raises(ValueError, match='refusing overwrite'):
        run(tmp_path)


def test_known_orderings_and_half_credit_match_pair_accounting(tmp_path):
    full_stage(tmp_path)
    cells, _ = ev.verify_stage(tmp_path, 'prototype', SOURCE, cohort_size=N)
    labels = endpoint()['DSS']
    for cell in cells.values():
        t = labels['time'][cell['valid']]
        for name in ev.RISK_NAMES:
            cell['risks'][name] = t.copy()
        cell['risks']['R3'] = -t
        cell['risks']['R2'] = np.zeros_like(t)
    point = ev.point_results(cells, labels, cohort_size=N)
    by = {(r['seed'], r['model']): r for r in point['seed_rows']}
    assert by[31, 'R0']['Harrell'] == 0.
    assert by[31, 'R2']['Harrell'] == .5
    assert by[31, 'R3']['Harrell'] == 1.
    paired = ev.pair_results(cells, labels, point, cohort_size=N)
    row = next(r for r in paired['repair_damage'] if r['comparison'] == 'R3-R2' and r['metric'] == 'Harrell')
    assert row['repair'] == .5 and row['damage'] == 0. and row['net'] == .5


def test_fold_aggregation_sums_pair_masses_before_division(tmp_path):
    full_stage(tmp_path)
    cells, _ = ev.verify_stage(tmp_path, 'prototype', SOURCE, cohort_size=N)
    labels = endpoint()['DSS']
    for cell in cells.values():
        t = labels['time'][cell['valid']]
        for name in ev.RISK_NAMES:
            cell['risks'][name] = -t if cell['fold'] == 1 else t
    point = ev.point_results(cells, labels, cohort_size=N)
    row = next(r for r in point['seed_rows'] if r['seed'] == 31 and r['model'] == 'R3')
    folds = [r for r in point['fold_rows'] if r['seed'] == 31 and r['model'] == 'R3']
    assert row['Harrell'] == pytest.approx(1/10)
    assert row['Harrell'] != pytest.approx(np.mean([r['Harrell'] for r in folds]))


def gate_rows():
    values = {'R0': .4, 'CP': .4, 'R1': .5, 'R2': .6, 'R3': .7}
    return [{'seed': s, 'model': name, 'Harrell': values.get(name, .3), 'Uno60': values.get(name, .3)}
            for s in ev.SEEDS for name in ev.RISK_NAMES]


def gate(rows):
    return ev.prototype_gate(rows, source=SOURCE, receipt_fingerprint='b'*64)


def test_all_locked_gate_families_admit_GO_but_do_not_launch_outer():
    result = gate(gate_rows())
    assert result['classification'] == result['decision'] == 'GO'
    assert result['R1_effective'] and result['R2_effective']
    assert len(result['primary_tests']) == 4
    assert len(result['R3_anchor_guards']) == 4
    assert len(result['R3_query_pairing_guards']) == 3
    assert result['outer_admitted'] and not result['outer_automatically_started']
    assert not result['confirmatory_confidence_intervals']


@pytest.mark.parametrize('name,metric', [('R1', 'Uno60'), ('CP', 'Harrell'),
    ('R3_WRONG_RNA', 'Uno60'), ('R3_WRONG_C', 'Harrell'), ('R3_C_MEAN', 'Uno60')])
def test_each_main_gate_family_can_prevent_outer(name, metric):
    rows = gate_rows()
    for r in rows:
        if r['model'] == name:
            r[metric] = .9
    result = gate(rows)
    assert result['decision'] == 'STOP' and not result['outer_admitted']


def test_positive_mean_with_only_one_positive_seed_does_not_pass_primary():
    rows = gate_rows()
    for r in rows:
        if r['model'] == 'R2':
            r['Harrell'] = .2 if r['seed'] == 31 else .8
    result = gate(rows)
    item = next(r for r in result['primary_tests'] if r['comparison'] == 'R3-R2' and r['metric'] == 'Harrell')
    assert item['mean_seed_delta'] > 0 and item['positive_seeds'] == 1 and not item['pass']
    assert not result['outer_admitted']


def test_pairing_tie_on_both_metrics_fails_mechanism_screen():
    rows = gate_rows()
    for r in rows:
        if r['model'] == 'R3_C_MEAN':
            r.update(Harrell=.7, Uno60=.7)
    assert not gate(rows)['outer_admitted']


def test_R2_only_classification_does_not_admit_alternate_outer_candidate():
    rows = gate_rows()
    for r in rows:
        if r['model'] == 'R3':
            r.update(Harrell=.5, Uno60=.5)
    result = gate(rows)
    assert result['classification'] == 'R2_ONLY_DIAGNOSTIC'
    assert result['R2_effective'] and not result['outer_admitted']


def test_R1_only_classification_uses_locked_effective_definition():
    rows = gate_rows()
    for r in rows:
        if r['model'] in ('R2', 'R3'):
            r.update(Harrell=.45, Uno60=.45)
    result = gate(rows)
    assert result['classification'] == 'R1_ONLY_DIAGNOSTIC'
    assert result['R1_effective'] and not result['R2_effective'] and not result['outer_admitted']


def test_all_fail_means_STOP_without_model_search():
    rows = gate_rows()
    for r in rows:
        if r['model'] in ev.NEURAL_ARMS:
            r.update(Harrell=.1, Uno60=.1)
    result = gate(rows)
    assert result['classification'] == 'STOP'
    assert not result['automatic_hyperparameter_or_architecture_search']


def test_outer_is_rejected_before_any_endpoint_or_model_read(tmp_path):
    with pytest.raises(ValueError, match='separately locked'):
        ev.evaluate_stage(tmp_path, tmp_path/'public', 'outer', SOURCE,
                          lambda: pytest.fail('endpoint read'))


def test_current_endpoint_cannot_drop_any_patient(tmp_path):
    full_stage(tmp_path)
    data = endpoint()
    data['DSS']['valid'][0] = False
    with pytest.raises(ValueError, match='cannot exclude'):
        run(tmp_path, lambda: data)
