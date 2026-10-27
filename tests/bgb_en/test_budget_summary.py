import numpy as np
from scripts.bgb_en_summarize_budget import aggregate_metrics, aggregate_rows, independent_metrics, mean_available


def metric(hn, hd, un, ud, repair=0., damage=0.):
    return dict(Harrell_num=hn, Harrell_den=hd, Uno_num=un, Uno_den=ud,
                Harrell=hn/hd, Uno60=un/ud, repair=repair, damage=damage, Uno_repair=repair, Uno_damage=damage)


def test_partition_aggregation_uses_numerators_not_mean_of_ratios():
    rows = [metric(1, 2, 1, 2), metric(9, 10, 9, 10), metric(2, 2, 2, 2)]
    result = aggregate_metrics(rows)
    assert result['Harrell'] == 12/14
    assert result['Harrell'] != np.mean([row['Harrell'] for row in rows])


def test_missing_decoder_score_does_not_silently_change_averaged_patients_or_seeds():
    assert mean_available([.2, None, .6]) is None
    assert mean_available([]) is None
    assert mean_available([.2, .6]) == .4


def test_independent_pair_metrics_counts_repairs_relative_to_same_risk_reference():
    t, e = np.array([1., 2., 3.]), np.ones(3, int)
    risk, reference = np.array([.9, .6, .1]), np.array([.6, .9, .1])
    result = independent_metrics(t, e, t, e, risk, reference)
    assert result['repair'] == 1/3 and result['damage'] == 0


def test_seed_mean_averages_seed_ratios_only_after_all_partitions():
    cells = []
    for seed, scores in [(31, [(1, 2), (9, 10), (2, 2)]), (47, [(1, 2), (1, 2), (1, 2)])]:
        for partition, (num, den) in enumerate(scores):
            m = metric(num, den, num, den)
            cells.append(dict(arm='O3', epoch=160, seed=seed, partition=partition,
                validation=m, training=m, parent80_comparison=m,
                retention={name: dict(Pearson=.5, R2=.2) for name in ('complete_tokens', 'pooled32')},
                valid_delta_SD_over_offset_SD=.1, training_walltime_seconds=2., total_cell_walltime_seconds=3.,
                parameters=17, peak_GPU_bytes=100))
    per_seed, summary = aggregate_rows(cells)
    assert len(per_seed) == 2 and summary[0]['complete_six_cells']
    assert summary[0]['validation']['Harrell'] == (12/14+.5)/2
