"""Read existing M7R/SPO artifacts; emit anonymous attribution statistics only.

No fitting, checkpoint/model construction, bootstrap, patient export or source
mutation. Execute on the authorized server with --recompute --stdout after
historical Git-order and current endpoint lineage checks pass.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def moments(values):
    x = np.asarray(values, float)
    require(x.ndim == 1 and len(x) and np.isfinite(x).all(), "invalid aggregate values")
    return {"count": len(x), "mean": float(x.mean()), "population_SD": float(x.std()),
            "minimum": float(x.min()), "maximum": float(x.max())}


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def descriptor_audit(root, reports):
    m7, spo = reports / 'phase_m7r_mechanism_confirmation', reports / 'phase_m8p1r_strong_anchor_residual_repair'
    film = rows(m7/'film_modulation_diagnostics.csv')
    profile = rows(m7/'compute_profile.csv')
    fields = ('mean_abs_gamma', 'mean_abs_beta', 'path_correction_norm', 'correction_over_path_norm')
    fsummary = []
    for arm in ('F3', 'F6'):
        groups = {}
        for row in film:
            if row['method'] == arm:
                groups.setdefault((int(row['seed']), int(row['outer_fold'])), []).append(row)
        require(len(groups) == 25, 'incomplete FiLM diagnostics')
        fsummary.append({'arm': arm, 'cells': 25, 'patient_rows_across_repeated_seeds': sum(map(len, groups.values())),
                         'cell_mean_statistics': {key: moments([np.mean([float(r[key]) for r in group])
                                                   for group in groups.values()]) for key in fields}})
    params = {arm: sorted({int(r['interaction_params']) for r in profile if r['method'] == arm})
              for arm in ('F2', 'F3')}
    diagnostic = rows(spo/'residual_diagnostics.csv')
    grouped = {}
    for r in diagnostic:
        if r['variant'] == 'SPO':
            grouped.setdefault((int(r['seed']), int(r['outer_fold'])), []).append(r)
    require(len(grouped) == 25, 'incomplete SPO residual diagnostics')
    output = []
    for (seed, fold), group in sorted(grouped.items()):
        anchor, dp, do, risk = [np.array([float(r[key]) for r in group]) for key in
                               ('clinical_offset', 'delta_pathology', 'delta_omics', 'risk')]
        require(all(np.isfinite(x).all() for x in (anchor, dp, do, risk)), 'nonfinite residual output')
        output.append({'seed': seed, 'fold': fold, 'patients': len(group),
                       'path_omics_Pearson': correlation(dp, do),
                       'path_anchor_Pearson': correlation(dp, anchor),
                       'omics_anchor_Pearson': correlation(do, anchor),
                       'path_output_SD': float(dp.std()), 'omics_output_SD': float(do.std()),
                       'total_residual_SD': float((dp+do).std()),
                       'recorded_reconstruction_max_abs': max(float(r['risk_reconstruction_abs_error']) for r in group),
                       'float64_reconstruction_from_saved_components_max_abs': float(np.max(np.abs(risk-anchor-dp-do)))})
    return {'FiLM': {'formula': 'p+tanh(gamma(o))*LayerNorm(p)+beta(o)',
                      'interaction_parameter_sets': params, 'modulation': fsummary,
                      'frozen_off_or_amplitude_replay': 'BLOCKED_MISSING_TRAINED_WEIGHTS',
                      'retrained_controls': ['F1', 'F2', 'F3', 'F6'],
                      'retraining_confound': 'encoder_fusion_head_joint_updates_and_selected_epoch'},
            'SPO': {'formula': 'frozen_anchor+path_head(path_encoder(P))+omics_head(omics_encoder(O))',
                    'P_O_encoder_parameters_shared': False, 'common_loss_jointly_optimized': True,
                    'clinical_encoder_has_risk_path': False,
                    'trained_gradient_replay': 'BLOCKED_MISSING_WEIGHTS_AND_GRADIENT_HISTORY',
                    'output_correlation_is_not_conditional_prognostic_information': True,
                    'rows': output,
                    'path_omics_cell_Pearson_summary': moments([r['path_omics_Pearson'] for r in output
                                                              if r['path_omics_Pearson'] is not None])}}


def lineage(root):
    from src.phase10_trimodal.features import ahash
    data = Path('/root/phase7a_foundation_headroom_v1/private_dataset.npz')
    ep_path = Path('/root/phase13_rna_private/endpoints.npz')
    with np.load(data, allow_pickle=False) as z:
        ids, folds = z['patient_ids'], z['folds']
        old = {k: z['DSS_'+k] for k in ('time', 'event', 'valid')}
    with np.load(ep_path, allow_pickle=False) as z:
        ep = {k: z['DSS_'+k] for k in ('time', 'event', 'valid')}
        patient_hash = str(z['patient_hash'])
    registry = json.loads((root/'reports/phase13_rna_information_audit/SOURCE_REGISTRY.json').read_text())
    require(sha(ep_path) == registry['endpoints_npz_sha256'], 'endpoint registry hash differs')
    require(patient_hash == ahash(ids), 'endpoint patient hash differs')
    require(all(np.array_equal(old[k], ep[k]) for k in old), 'canonical DSS differs')
    require(len(ids) == 512 and ep['valid'].all() and ep['event'].sum() == 65, 'cohort differs')
    checks = []
    for stage, commit in [('M7R', '6821df983928cb9cc09a56a30c18cc65eca29159'),
                          ('SPO', '59d558c45b6e9cc474838d5a7cfd547ea3a7cc69')]:
        raw = None
        for gitroot in (root, Path('/root/diffusion')):
            try:
                raw = subprocess.check_output(['git', 'show', commit+':outputs/phase_m1r/fixed_patient_folds.csv'],
                                               cwd=gitroot, stderr=subprocess.DEVNULL)
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        require(raw is not None, 'historical patient Git blob missing: '+stage)
        split = sorted(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))), key=lambda x: x['patient_id'][:12])
        require(np.array_equal([r['patient_id'][:12] for r in split], ids), 'historical patient order differs')
        require(np.array_equal([int(r['fold']) for r in split], folds), 'historical fold differs')
        checks.append({'stage': stage, 'commit': commit, 'split_blob_sha256': hashlib.sha256(raw).hexdigest(),
                       'patient_order_and_folds_elementwise': True})
    return ep, folds, {'status': 'PASS', 'patients': 512, 'DSS_events': 65,
                       'endpoint_hash_registry_verified': True, 'endpoint_patient_order_verified': True,
                       'canonical_DSS_elementwise_verified': True, 'historical_checks': checks}


def legacy_h(t, e, risk):
    mask = (e[:, None] == 1) & (t[None, :] > t[:, None])
    difference = risk[:, None]-risk[None, :]
    return float(((difference > 0)+.5*(difference == 0))[mask].mean())


def recompute(root, ep, folds, stage):
    from src.phase10_trimodal.reference import metrics, uno_weights
    from src.phase21_relative.evaluation import STATISTICS, pair_transition_summary
    seeds = range(20261002, 20261007) if stage == 'M7R' else range(20261012, 20261017)
    arms = ('F1', 'F2', 'F3') if stage == 'M7R' else ('S0', 'SP', 'SO', 'SPO')
    pairs = (('F3', 'F1'), ('F3', 'F2')) if stage == 'M7R' else (
        ('SP', 'S0'), ('SO', 'S0'), ('SPO', 'S0'), ('SPO', 'SP'), ('SPO', 'SO'))
    fold_rows, seed_rows, transitions, sources = [], [], [], []
    old_metric_error = 0.
    legacy_replayed_cells = 0
    t, e = ep['time'], ep['event']
    for seed in seeds:
        totals, pair_totals = {}, {}
        original_spo_rows = None
        if stage == 'SPO':
            original_spo_rows = {(r['variant'], int(r['outer_fold'])): r for r in rows(
                Path('/root/autodl-tmp/m8p1r_workers')/f'seed_{seed}'/'fold_metrics.csv')}
        for fold in range(1, 6):
            valid, fit = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
            weight = uno_weights(t[fit], e[fit], t[valid], e[valid], tau=60.)
            risks = {}
            for arm in arms:
                directory = Path('/root/autodl-tmp') / (f'diffusion_runtime_m7r_{seed}' if stage == 'M7R'
                                                        else f'm8p1r_workers/seed_{seed}') / 'cells'
                stem = f'{arm}{seed}{fold}' if stage == 'M7R' else f'{arm}_{seed}_{fold}'
                path = directory/(stem+'.npy')
                pred = np.asarray(np.load(path, allow_pickle=False), float)
                require(pred.shape == (len(valid),) and np.isfinite(pred).all(), 'invalid saved prediction')
                if stage == 'M7R':
                    saved = json.loads(path.with_suffix('.json').read_text())['row']
                    require((saved['method'], int(saved['seed']), int(saved['outer_fold'])) == (arm, seed, fold),
                            'prediction sidecar identity differs')
                else:
                    saved = original_spo_rows[arm, fold]
                    require(int(saved['seed']) == seed, 'SPO saved metric identity differs')
                require(int(saved['n_test']) == len(valid) and int(saved['events_test']) == int(e[valid].sum()),
                        'old heldout patient/event counts differ from canonical endpoint')
                error = abs(legacy_h(t[valid], e[valid], pred)-float(saved['harrell_c']))
                require(error < 1e-12, 'legacy prediction does not reproduce original Harrell')
                old_metric_error = max(old_metric_error, error)
                legacy_replayed_cells += 1
                risks[arm] = pred
                source_record = {'stage': stage, 'arm': arm, 'seed': seed, 'fold': fold, 'prediction_sha256': sha(path)}
                sources.append(source_record)
                value = metrics(t[valid], e[valid], pred, np.full(len(valid), fold), weight)
                require(all(np.isfinite(x) for x in value.values()), 'undefined current concordance')
                fold_rows.append({'arm': arm, 'seed': seed, 'fold': fold, **value})
                total = totals.setdefault(arm, {k: 0. for k in ('Harrell_num', 'Harrell_den', 'Uno_num', 'Uno_den')})
                for k in total:
                    total[k] += value[k]
            for candidate, baseline in pairs:
                record = pair_transition_summary(t[valid], e[valid], risks[baseline], risks[candidate],
                                                 folds=np.full(len(valid), fold), ipcw=weight)
                for r in record['rows']:
                    key = candidate+'-'+baseline, r['metric']
                    dest = pair_totals.setdefault(key, {'denominator': 0., **{s: 0. for s in STATISTICS}})
                    dest['denominator'] += r['denominator']
                    for stat in STATISTICS:
                        dest[stat] += r[stat]*r['denominator']
        for arm, total in totals.items():
            seed_rows.append({'seed': seed, 'arm': arm, **total,
                              'Harrell': total['Harrell_num']/total['Harrell_den'],
                              'Uno60': total['Uno_num']/total['Uno_den']})
        by = {r['arm']: r for r in seed_rows if r['seed'] == seed}
        for (comparison, metric), total in pair_totals.items():
            candidate, baseline = comparison.split('-')
            values = {k: total[k]/total['denominator'] for k in STATISTICS}
            require(abs(values['net']-(by[candidate][metric]-by[baseline][metric])) < 1e-12, 'pair credit mismatch')
            transitions.append({'seed': seed, 'comparison': comparison, 'metric': metric,
                                'denominator': total['denominator'], **values})
    means = [{'arm': arm, **{m: float(np.mean([r[m] for r in seed_rows if r['arm'] == arm]))
                             for m in ('Harrell', 'Uno60')}} for arm in arms]
    comparisons = []
    for candidate, baseline in pairs:
        for metric in ('Harrell', 'Uno60'):
            part = [r for r in transitions if r['comparison'] == candidate+'-'+baseline and r['metric'] == metric]
            comparisons.append({'comparison': candidate+'-'+baseline, 'metric': metric,
                                **{k: float(np.mean([r[k] for r in part])) for k in ('repair', 'damage', 'net')},
                                'positive_seeds': sum(r['net'] > 0 for r in part), 'seeds': 5})
    return {'status': 'PASS_EXISTING_PREDICTIONS_ONLY', 'fold_rows': fold_rows, 'seed_rows': seed_rows,
            'five_seed_means': means, 'paired_comparisons': comparisons, 'seed_repair_damage': transitions,
            'prediction_files': sources, 'legacy_Harrell_reproduction_max_error': old_metric_error,
            'legacy_prediction_cells_replayed': legacy_replayed_cells,
            'aggregation': 'within-fold numerator/denominator;sum5folds_per_seed_then_mean5seeds',
            'Uno60': 'tau60_event_time_strictly_less;reverseKM_from_outer_train;current_tie_handling',
            'bootstrap_replicates': 0, 'confidence_intervals': 'NOT_COMPUTED',
            'new_training': 0, 'historical_training_protocol_changed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--reports', type=Path)
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--stdout', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    result = {'schema': 'phase25_a_legacy_attribution_v1', 'audit_date': '2026-10-18',
              'new_training': 0, 'new_bootstrap_replicates': 0,
              'patient_arrays_or_weights_public': False, 'history_files_modified': False,
              'descriptors': descriptor_audit(root, args.reports or root/'reports'),
              'old_CI': {'M7R': 'NOT_PAIRED_and_cross_fold_pooled_not_current_metric',
                         'SPO': 'PAIRED_but_cross_fold_pooled_and_full_cohort_censor_not_current_metric'}}
    if Path('/root/autodl-tmp').exists():
        inventories = []
        for directory in [Path('/root/autodl-tmp')/f'diffusion_runtime_m7r_{seed}' for seed in range(20261002, 20261007)] + [
                Path('/root/autodl-tmp/m8p1r_workers')]:
            require(directory.is_dir(), 'historical run directory missing')
            inventories.append({'run_directory': directory.name,
                                'trained_weight_files_found': sum(p.is_file() and p.suffix in ('.pt', '.pth', '.ckpt')
                                                                 for p in directory.rglob('*'))})
        result['weight_inventory'] = {'scope': 'exact_known_historical_run_directories_only', 'rows': inventories}
    if args.recompute:
        ep, folds, result['endpoint_lineage'] = lineage(root)
        result['current_evaluation'] = {stage: recompute(root, ep, folds, stage) for stage in ('M7R', 'SPO')}
    else:
        result['current_evaluation'] = {'status': 'NOT_REQUESTED_requires_server_lineage'}
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if args.stdout:
        print(text, end='')
    else:
        require(args.output is not None and not args.output.exists(), 'explicit new output required')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
