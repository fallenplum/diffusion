import copy
import importlib.util
import json
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[2] / 'scripts/bgb_en_report_inner.py'
SPEC = importlib.util.spec_from_file_location('bgb_en_report_inner_test_module', SOURCE)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def config():
    return {'prototype': {'outer_fold': 1, 'inner_splits': 3, 'neural_seeds': [31, 47]},
            'training': {'epochs': 80},
            'R1': {'arms': ['P0', 'P1']}, 'R2': {'arms': ['O0', 'O3', 'O5', 'O6'], 'sequence_orders': 4},
            'R3': {'arms': ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10'],
                   'additional_real_retraining': ['F5_WRONG_QUERY', 'F5_SHUFFLED_MEMBERSHIP']}}


def metric(numerator=8., denominator=10.):
    return {'Harrell_num': numerator, 'Harrell_den': denominator,
            'Uno_num': numerator * 2, 'Uno_den': denominator * 2,
            'Harrell': numerator / denominator, 'Uno60': numerator / denominator,
            'repair': .1, 'damage': .03, 'Uno_repair': .08, 'Uno_damage': .01,
            'risk_shift_sd_ratio': .1}


def cell(stage, arm, part, seed, metrics=None):
    m = metric() if metrics is None else metrics
    neural = not report.deterministic(arm)
    return {'status': 'COMPLETE', 'lineage': {'source': 'synthetic-public-source'},
            'stage': stage, 'arm': arm, 'outer_fold': 1, 'inner_partition': part, 'seed': seed,
            'metrics': m, 'parameters': 12, 'peak_GPU_bytes': 1024 if neural else 0,
            'epochs': 80 if neural else 0, 'actual_optimizer_fits': 1 if neural else 0,
            'elapsed_seconds': 4., 'outer_test_evaluated': False,
            'curves': [{'epoch': epoch, 'train': metric(), 'valid': m, 'cox_loss': 2.}
                       for epoch in (20, 40, 80)] if neural else [],
            'frozen_controls': {'wrong_query': metric()},
            # These must never get copied to public artifacts.
            'private_patient_field': 'SYNTHETIC_PRIVATE_SENTINEL_DO_NOT_EXPORT'}


def write_cell(private, value):
    p = private / 'cells' / f"{value['stage']}_{value['arm']}_{value['inner_partition']}_{value['seed']}" / 'COMPLETE.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value), encoding='utf-8')


def test_micropairs_aggregate_before_seed_mean_not_average_cell_ratios():
    c = config(); cells = {}
    for seed in (31, 47):
        for p, (num, den) in enumerate([(1., 2.), (9., 10.), (1., 1.)]):
            cells['R1', 'P1', p, seed] = cell('R1', 'P1', p, seed, metric(num, den))
    summary = report.arm_summary(cells, c, 'R1', 'P1')
    assert summary['Harrell'] == 11. / 13.
    assert summary['Uno60'] == 11. / 13.
    assert summary['Harrell'] != (1 / 2 + 9 / 10 + 1) / 3


def test_incomplete_seed_never_produces_overall_scientific_mean():
    cells = {('R1', 'P1', p, 31): cell('R1', 'P1', p, 31) for p in range(3)}
    row = report.arm_summary(cells, config(), 'R1', 'P1')
    assert row['status'] == 'RUNNING'
    assert row['Harrell'] is row['Uno60'] is None


def test_deterministic_reference_reused_without_fabricating_extra_fit():
    cells = {('R1', 'P0', p, 31): cell('R1', 'P0', p, 31) for p in range(3)}
    row = report.arm_summary(cells, config(), 'R1', 'P0')
    assert row['status'] == 'COMPLETE'
    assert row['completed_unique_cells'] == 3
    assert row['logical_seed_references'] == 6
    assert row['actual_cell_optimizer_fits'] == 0
    assert row['Harrell_seed31'] == row['Harrell_seed47'] == .8


def test_report_reads_complete_only_excludes_private_unknown_keys_and_marks_missing(tmp_path):
    private, output = tmp_path / 'private', tmp_path / 'public'
    write_cell(private, cell('R1', 'P0', 0, 31))
    # These forbidden inputs would fail JSON decoding if they were read.
    (private / 'model.pt').write_bytes(b'not a pickle')
    (private / 'prepared').mkdir(); (private / 'prepared' / 'bad.json').write_text('bad')
    (private / 'cells' / 'FAILED.json').write_text('bad')
    result = report.generate(private, output, config())
    assert result['status'] == 'RUNNING'
    assert result['completed_unique_cells'] == 1
    for path in output.iterdir():
        text = path.read_text(encoding='utf-8')
        assert 'SYNTHETIC_PRIVATE_SENTINEL_DO_NOT_EXPORT' not in text
    assert 'NOT_RUN' in (output / 'R2_ORDER_SENSITIVITY.csv').read_text(encoding='utf-8')
    assert '| P0 | RUNNING | 1 | — | — |' in (output / 'R1_RESULTS.md').read_text(encoding='utf-8')


def test_lineage_or_final_epoch_metric_mismatch_fails_closed(tmp_path):
    private = tmp_path / 'private'
    write_cell(private, cell('R1', 'P0', 0, 31))
    different = cell('R1', 'P0', 1, 31); different['lineage']['source'] = 'different'
    write_cell(private, different)
    with pytest.raises(ValueError, match='Mixed source'):
        report.load_cells(private, config())
    second = tmp_path / 'other'
    bad = cell('R1', 'P1', 0, 31)
    bad['curves'][-1]['valid'] = metric(9., 10.)
    write_cell(second, bad)
    with pytest.raises(ValueError, match='final snapshot'):
        report.load_cells(second, config())


def test_pairing_preserves_frozen_vs_retraining_and_repair_reference():
    cells = {}
    for seed in (31, 47):
        for p in range(3):
            a = cell('R3', 'F5', p, seed, metric(9., 10.))
            a['frozen_controls'] = {'wrong_query': metric(8., 10.)}
            cells['R3', 'F5', p, seed] = a
            cells['R3', 'F2', p, seed] = cell('R3', 'F2', p, seed, metric(7., 10.))
    retrained = report.paired(cells, config(), 'R3', 'F5', 'F2')
    frozen = report.paired(cells, config(), 'R3', 'F5', 'F5', frozen='wrong_query')
    assert retrained['comparison_type'] == 'REAL_RETRAINING'
    assert frozen['comparison_type'] == 'FROZEN_INTERVENTION'
    assert abs(retrained['paired_delta_H'] - .2) < 1e-12
    assert abs(frozen['paired_delta_H'] - .1) < 1e-12
    assert frozen['direct_candidate_control_repair_damage'] == 'NOT_AVAILABLE_FROM_ANONYMOUS_RECEIPTS'
    assert frozen['candidate_repair_vs_F1'] == .1
    assert frozen['control_repair_vs_F1'] == .1
    assert 'F1' in frozen['repair_reference']


def test_nested_and_flat_decoder_fields_are_retained_without_claiming_missing_decoder():
    a = cell('R2', 'O3', 0, 31)
    a['retention'] = {'complete_tokens': {'dimension': 1600, 'Pearson': .7, 'R2': .4, 'target': 'honest_teacher'},
                      'pooled32': {'dimension': 32, 'Pearson': .3, 'R2': -.1, 'target': 'honest_teacher'}}
    b = cell('R2', 'O1', 0, 31)
    b['retention'] = {'decoder_available': False, 'risk_decoder_R2': None,
                      'risk_decoder_Pearson': None, 'prediction_risk_Pearson': .5}
    c = config(); c['R2']['arms'].append('O1')
    rows = report.retention_rows({('R2', 'O3', 0, 31): a, ('R2', 'O1', 0, 31): b}, c)
    assert any(r.get('representation') == 'complete_tokens' and r['dimension'] == 1600 for r in rows)
    assert any(r.get('representation') == 'pooled32' and r['decoder_R2'] == -.1 for r in rows)
    assert any(r['status'] == 'NOT_RUN_DECODER_HIGH_DIMENSION' and r['decoder_R2'] is None for r in rows)


def test_saved_risk_replay_merges_actual_full_control_repair_with_count_aggregation(tmp_path):
    cells, replay_rows = {}, []
    for seed in (31, 47):
        for part, denominator in enumerate([10., 20., 30.]):
            a = cell('R3', 'F5', part, seed, metric(.9 * denominator, denominator))
            a['frozen_controls'] = {'wrong_query': metric(.8 * denominator, denominator)}
            cells['R3', 'F5', part, seed] = a
            replay_rows.append({'arm': 'F5', 'control': 'wrong_query', 'seed': seed, 'inner_partition': part,
                                'pair_reference': 'full_trained_checkpoint_vs_frozen_control',
                                'Harrell_full': .9, 'Harrell_control': .8, 'delta_Harrell': .1,
                                'Uno60_full': .9, 'Uno60_control': .8, 'delta_Uno60': .1,
                                'repair_num': .12 * denominator, 'damage_num': .02 * denominator,
                                'Uno_repair_num': .15 * 2 * denominator, 'Uno_damage_num': .05 * 2 * denominator,
                                'Harrell_den': denominator, 'Uno_den': 2 * denominator, 'risk_shift_sd_ratio': .02})
    pairing = [report.paired(cells, config(), 'R3', 'F5', 'F5', frozen='wrong_query')]
    path = tmp_path / 'MECHANISM_REPLAY.json'
    path.write_text(json.dumps({'schema': 'bgb_en_mechanism_pair_replay_v1', 'optimizer_steps': 0,
                               'whole_matrix_complete': False, 'comparisons': replay_rows}), encoding='utf-8')
    audit = report.attach_mechanism_replay(pairing, cells, config(), path)
    row = pairing[0]
    assert audit['completed_pairing_comparisons'] == 1
    assert row['direct_replay_status'] == 'COMPLETE'
    assert row['direct_candidate_control_repair_damage'] == 'COMPLETE_SAVED_RISK_REPLAY'
    assert abs(row['full_control_repair'] - .12) < 1e-12
    assert abs(row['full_control_damage'] - .02) < 1e-12
    assert abs(row['full_control_Uno_repair'] - .15) < 1e-12
    assert abs(row['full_control_Uno_damage'] - .05) < 1e-12
    assert row['candidate_repair_vs_F1'] == .1  # Its distinct original reference is preserved.
    # An apparently complete replay cannot silently substitute another final risk.
    replay_rows[0]['Harrell_full'] = .7
    path.write_text(json.dumps({'schema': 'bgb_en_mechanism_pair_replay_v1', 'optimizer_steps': 0,
                               'comparisons': replay_rows}), encoding='utf-8')
    with pytest.raises(ValueError, match='fixed-final receipt'):
        report.attach_mechanism_replay(pairing, cells, config(), path)


@pytest.mark.parametrize('stage,arm,reference', [
    ('R1', 'P0', 'F0'), ('R1', 'P1', 'CS'), ('R1', 'P4', 'CS'),
    ('R2', 'O0', 'F0'), ('R2', 'O1', 'F0'), ('R2', 'O2_PCA32', 'F0'),
    ('R2', 'O2_PCA256', 'F0'), ('R2', 'O3', 'O0'), ('R2', 'O8', 'O0'),
    ('R3', 'F0', 'F0'), ('R3', 'F1', 'F0'), ('R3', 'F2', 'F1'),
    ('R3', 'F5', 'F1'), ('R3', 'F5_WRONG_QUERY', 'F1')])
def test_saved_diagnostic_reference_is_actual_writer_reference(stage, arm, reference):
    assert report.stored_risk_diagnostics_reference(stage, arm) == reference
    value = cell(stage, arm, 0, 31)
    row = report.cell_metric_rows({(stage, arm, 0, 31): value})[0]
    assert row['stored_risk_diagnostics_reference'] == reference
    assert row['repair_reference'] == reference


def test_paired_metric_difference_does_not_relabel_stored_repair_reference():
    cells = {}
    for seed in (31, 47):
        for partition in range(3):
            cells['R1', 'P1', partition, seed] = cell('R1', 'P1', partition, seed, metric(9., 10.))
    for partition in range(3):
        cells['R1', 'P0', partition, 31] = cell('R1', 'P0', partition, 31, metric(7., 10.))
    before = copy.deepcopy(cells)
    result = report.paired(cells, config(), 'R1', 'P1', 'P0')
    assert abs(result['paired_delta_H'] - .2) < 1e-12
    assert result['candidate_stored_risk_diagnostics_reference'] == 'CS'
    assert result['control_stored_risk_diagnostics_reference'] == 'F0'
    assert result['paired_delta_source'] == 'independent_concordance_numerator_denominator_difference'
    # Both synthetic saved diagnostic increments are .07, so subtracting those
    # would give zero rather than the correctly computed candidate/control .2.
    assert report.stored_risk_increment(cells['R1', 'P1', 0, 31]['metrics']) == pytest.approx(.07)
    assert report.stored_risk_increment(cells['R1', 'P0', 0, 31]['metrics']) == pytest.approx(.07)
    assert cells == before


def test_learning_curves_label_crossfit_offsets_and_both_increment_metrics():
    value = cell('R3', 'F5', 0, 31)
    for point in value['curves']:
        point['train'] = dict(metric(3., 10.), repair=.01, damage=.03, Uno_repair=.02, Uno_damage=.05)
    cells = {('R3', 'F5', 0, 31): value}
    before = copy.deepcopy(cells)
    rows = report.training_rows(cells)
    assert len(rows) == 3
    for row in rows:
        assert row['stored_risk_diagnostics_reference'] == 'F1'
        assert row['train_offset_scope'] == 'current_partition_crossfitted_offset'
        assert row['valid_offset_scope'] == 'current_partition_full_training_fit_offset'
        assert row['train_delta_H_vs_stored_reference'] == pytest.approx(-.02)
        assert row['train_delta_U60_vs_stored_reference'] == pytest.approx(-.03)
        assert row['valid_delta_H_vs_stored_reference'] == pytest.approx(.07)
        assert row['valid_delta_U60_vs_stored_reference'] == pytest.approx(.07)
        assert row['train_Harrell'] < row['valid_Harrell']
    assert cells == before


def test_summary_increments_keep_pair_weighting_and_report_interpretation_boundary():
    cells = {}
    for seed in (31, 47):
        for partition, den in enumerate((10., 20., 30.)):
            value = cell('R3', 'F5', partition, seed, metric(.8 * den, den))
            for point in value['curves']:
                point['train'] = dict(metric(.6 * den, den), repair=.01 * (partition + 1), damage=.01,
                                      Uno_repair=.02 * (partition + 1), Uno_damage=.02)
            cells['R3', 'F5', partition, seed] = value
    summary = report.arm_summary(cells, config(), 'R3', 'F5')
    assert summary['Harrell'] == .8 and summary['train_Harrell'] == .6
    assert summary['train_delta_H_vs_stored_reference'] == pytest.approx((0. * 10 + .01 * 20 + .02 * 30) / 60)
    assert summary['train_delta_U60_vs_stored_reference'] == pytest.approx((0. * 10 + .02 * 20 + .04 * 30) / 60)
    assert summary['valid_delta_H_vs_stored_reference'] == pytest.approx(.07)
    text = report.stage_report('R3', [summary], [], config(), 6, 6)
    assert '不能据此判定欠拟合' in text
    assert '不是原repair/damage字段相减' in text
    assert 'stored_risk_diagnostics_reference' in text
