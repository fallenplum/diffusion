"""Synthetic locked-gate and within-fold arithmetic; no real endpoint access."""
import copy

import numpy as np
import pytest

from src.phase28_reread import evaluation as ev


def gate_rows():
    values = {'M0': .72, 'CP': .73, 'RNA': .55,
              'C00': .65, 'C10': .82, 'C01': .83, 'C11': .8, 'DIRECT': .7}
    return [{'seed': seed, 'model': name, 'Harrell': values.get(name, .74),
             'Uno60': values.get(name, .74)} for seed in ev.SEEDS for name in ev.RISK_NAMES]


def test_go_keeps_direction_synergy_off_and_whole_modality_diagnostic_and_outer_closed():
    rows = gate_rows()
    for row in rows:
        if row['model'] in ('C11_OFF', 'C11_WHOLE_OMICS_WRONG'):
            row.update(Harrell=.99, Uno60=.99)
    gate = ev.prototype_gate(rows, source='synthetic', receipt_fingerprint='fake')
    assert gate['decision'] == 'GO'
    assert gate['outer_admitted'] is False and gate['automatic_alternate_arm_promotion'] is False
    assert all(not row['pass'] for row in gate['direction_comparisons_diagnostic_only'])
    assert all(row['diagnostic_only'] and not row['biological_synergy_claimed']
               for row in gate['factorial_contrasts_diagnostic_only'])
    assert gate['OFF_and_WHOLE_OMICS_WRONG_are_diagnostic_only'] is True


@pytest.mark.parametrize('baseline', ['C00', 'DIRECT'])
@pytest.mark.parametrize('metric', ev.METRICS)
def test_primary_requires_two_positive_seeds_even_with_positive_mean(baseline, metric):
    rows = gate_rows(); by = {(r['seed'], r['model']): r for r in rows}
    by[31, baseline][metric] = by[31, 'C11'][metric]
    assert ev.prototype_gate(rows, source='fake', receipt_fingerprint='fake')['decision'] == 'STOP'


def test_external_guard_allows_equality_but_rejects_one_negative_metric():
    rows = gate_rows(); by = {(r['seed'], r['model']): r for r in rows}
    for seed in ev.SEEDS:
        for name in ('M0', 'CP'):
            by[seed, name].update(Harrell=.8, Uno60=.8)
    assert ev.prototype_gate(rows, source='fake', receipt_fingerprint='fake')['decision'] == 'GO'
    by[47, 'CP']['Uno60'] = .81
    assert ev.prototype_gate(rows, source='fake', receipt_fingerprint='fake')['decision'] == 'STOP'


@pytest.mark.parametrize('control', ev.GATE_CONTROLS)
def test_each_frozen_guard_requires_two_nonnegative_metric_means_and_one_strict(control):
    rows = gate_rows()
    for row in rows:
        if row['model'] == 'C11_'+control:
            row.update(Harrell=.81, Uno60=.79)
    assert ev.prototype_gate(rows, source='fake', receipt_fingerprint='fake')['decision'] == 'STOP'
    for row in rows:
        if row['model'] == 'C11_'+control:
            row.update(Harrell=.8, Uno60=.8)
    assert ev.prototype_gate(rows, source='fake', receipt_fingerprint='fake')['decision'] == 'STOP'


def test_gate_rejects_missing_duplicate_and_nonfinite_rows():
    for bad in (gate_rows()[:-1], gate_rows()+[gate_rows()[0]]):
        with pytest.raises(ValueError, match='complete unique'):
            ev.prototype_gate(bad, source='fake', receipt_fingerprint='fake')
    bad = gate_rows(); bad[0]['Harrell'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        ev.prototype_gate(bad, source='fake', receipt_fingerprint='fake')


def synthetic_cells():
    n = 18
    time = np.arange(1., n+1)
    event = np.ones(n); event[[2, 3, 7, 10, 13, 16, 17]] = 0
    valid_sets = (np.arange(0, 4), np.arange(4, 9), np.arange(9, 15))
    cells = {}
    for seed in ev.SEEDS:
        for fold, valid in zip(ev.FOLDS, valid_sets):
            fit = np.setdiff1d(np.arange(n), valid)
            baseline = -time[valid]
            risks = {name: baseline.copy() for name in ev.RISK_NAMES}
            # Different pair counts and within-fold performance rule out a
            # silent arithmetic mean of fold C-indices.
            candidate = baseline if fold == 1 else (time[valid] if fold == 2 else baseline*0)
            risks['C11'] = candidate.copy()
            risks['C11_OFF'] = baseline+.5  # own learned main, explicitly not M0
            cells[seed, fold] = {'fit': fit, 'valid': valid, 'risks': risks}
    return cells, {'time': time, 'event': event, 'valid': np.ones(n, bool)}, n


def test_point_sum_num_den_and_pair_credit_close_without_cross_fold_comparison():
    cells, endpoint, n = synthetic_cells()
    point = ev.point_results(cells, endpoint, cohort_size=n)
    seed = next(row for row in point['seed_rows'] if row['seed'] == 31 and row['model'] == 'C11')
    folds = [row for row in point['fold_rows'] if row['seed'] == 31 and row['model'] == 'C11']
    for metric, prefix in (('Harrell', 'Harrell'), ('Uno60', 'Uno')):
        expected = sum(row[prefix+'_num'] for row in folds)/sum(row[prefix+'_den'] for row in folds)
        assert seed[metric] == pytest.approx(expected, abs=1e-14)
    assert abs(seed['Harrell']-np.mean([row['Harrell'] for row in folds])) > 1e-3
    assert point['unique_patient_OOF'] is False and point['risk_ensemble'] is False
    paired = ev.pair_results(cells, endpoint, point, cohort_size=n)
    by = {(row['seed'], row['model']): row for row in point['seed_rows']}
    for row in paired['seed_repair_damage']:
        candidate, baseline = row['comparison'].split('-', 1)
        delta = by[row['seed'], candidate][row['metric']]-by[row['seed'], baseline][row['metric']]
        assert row['net'] == pytest.approx(delta, abs=1e-13)
        assert row['repair']-row['damage'] == pytest.approx(delta, abs=1e-13)
        assert row['repair_credit_mass']-row['damage_credit_mass'] == pytest.approx(
            row['net_credit_mass'], abs=1e-12)
    assert paired['OFF_is_same_arm_external_plus_learned_main_not_M0'] is True


def test_fold_offset_shift_does_not_change_within_fold_points_or_repairs():
    cells, endpoint, n = synthetic_cells()
    changed = copy.deepcopy(cells)
    for (seed, fold), cell in changed.items():
        for name in ev.RISK_NAMES:
            cell['risks'][name] += {1: 100., 2: -100., 3: 1000.}[fold]
    original = ev.point_results(cells, endpoint, cohort_size=n)
    shifted = ev.point_results(changed, endpoint, cohort_size=n)
    assert original['seed_rows'] == shifted['seed_rows']
    assert ev.pair_results(cells, endpoint, original, cohort_size=n)['seed_repair_damage'] == \
        ev.pair_results(changed, endpoint, shifted, cohort_size=n)['seed_repair_damage']


def test_missing_control_or_fit_valid_overlap_fail_closed():
    cells, endpoint, n = synthetic_cells()
    del cells[31, 1]['risks']['C11_GENE_WRONG']
    with pytest.raises(ValueError, match='all matched'):
        ev.point_results(cells, endpoint, cohort_size=n)
    cells, endpoint, n = synthetic_cells()
    cells[31, 1]['fit'][0] = cells[31, 1]['valid'][0]
    with pytest.raises(ValueError, match='overlap'):
        ev.point_results(cells, endpoint, cohort_size=n)
