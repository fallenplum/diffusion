"""Fixed-grid, no-fit post-hoc diagnosis of saved Phase25-B risk corrections."""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase25_local import data, runtime, train
from src.phase25_local.evaluation import verify_stage
from src.phase25_local.model import LocalRouting
from src.phase25_scale_diagnostic import moments, paired_distributions, residual_summary, scaled_risk
from src.phase10_trimodal.reference import metrics, uno_weights
from src.phase21_relative.evaluation import pair_transition_summary

CONFIG = ROOT/'configs/phase25_b_scale_diagnostic.json'
REPORT = ROOT/'reports/phase25_b_scale_diagnostic'
SOURCE = '91cad3e88814d92d9f43b3ebf8c0962d11098b3fa80a8ea81c275a6b83f2884c'
SCALES = (0, .1, .25, .5, 1, 2)
METRICS = ('Harrell', 'Uno60')


def require(value, message):
    if not bool(value): raise ValueError(message)


def inputs_snapshot():
    """Bind every original public artifact and every reused private receipt."""
    manifest = json.loads((runtime.REPORT/'ARTIFACT_MANIFEST.json').read_text())
    require(manifest['training_source'] == SOURCE and manifest['total_files'] == 47,
            'original Phase25 manifest changed')
    names = [ROOT/r['path'] for r in manifest['public_files']]
    names += [runtime.REPORT/'ARTIFACT_MANIFEST.json', CONFIG, REPORT/'PROTOCOL.md',
              Path(__file__), ROOT/'src/phase25_scale_diagnostic.py',
              data.parent.PRIVATE/'endpoints.npz', data.parent.REPORT/'SOURCE_REGISTRY.json']
    for row in manifest['public_files']:
        require(runtime.sha(ROOT/row['path']) == row['sha256'], 'original public artifact changed')
    for fold in (1, 2, 3):
        path = data.prepared_path(fold, 'prototype')
        names.extend((path, path.with_suffix('.json'),
            data.parent.PARENT/'formal/manifests'/f'R0_128_seed31_fold{fold}.json'))
        for seed in (31, 47):
            folder = runtime.cell_dir('prototype', seed, fold)
            names.extend((folder/'checkpoint.pt', folder/'DONE_PRIVATE.json'))
    return {str(p): runtime.sha(p) for p in names}


def verify_replay_evidence():
    for name, status, script in (
            ('MAIN_REPLAY_AUDIT.json', 'PASS_NO_FIT_MAIN_REPLAY', 'replay_phase25_main.py'),
            ('REPLAY_AUDIT.json', 'PASS', 'replay_phase25.py')):
        value = json.loads((runtime.REPORT/name).read_text())
        require(value['status'] == status and value['source'] == SOURCE,
                'prior independent replay is not complete/current')
        require(value['replay_script_sha256'] == runtime.sha(ROOT/'scripts'/script),
                'prior independent replay executable changed')
        require(value['maximum_risk_replay_error' if name.startswith('MAIN') else
                      'maximum_absolute_risk_error'] == 0, 'prior replay was not exact')
    gate = json.loads((runtime.REPORT/'decision.json').read_text())
    require(gate['decision'] == 'STOP' and not gate['outer_started'], 'original STOP changed')


def endpoint_and_partitions(prepared, cells):
    """Read only original DSS keys, after all checkpoint/prepared checks pass."""
    registry = json.loads((data.parent.REPORT/'SOURCE_REGISTRY.json').read_text())
    endpoint_path = data.parent.PRIVATE/'endpoints.npz'
    require(runtime.sha(endpoint_path) == registry['endpoints_npz_sha256'], 'DSS registry hash changed')
    with np.load(data.parent.DATA, allow_pickle=True) as z:
        ids = np.array(z['patient_ids'])
    require(hashlib.sha256('\n'.join(np.asarray(ids).astype(str)).encode()).hexdigest() ==
            '76a144d544b7461d669487625855dc7de65f115a6a2f65d7a20002b2a8e5008f',
            'canonical patient identity changed')
    with np.load(endpoint_path) as z:
        require(str(z['patient_hash']) == data.parent.ahash(ids), 'DSS patient order changed')
        t, e, valid = (np.array(z['DSS_'+key]) for key in ('time', 'event', 'valid'))
    require(t.shape == e.shape == valid.shape == (512,) and valid.dtype == bool and valid.all(),
            'original DSS cohort changed')
    require(np.isfinite(t).all() and (t >= 0).all() and np.isin(e, (0, 1)).all() and e.sum() == 65,
            'original DSS endpoint invalid')
    for fold, saved in prepared.items():
        split = json.loads((data.parent.PARENT/'formal/manifests'/f'R0_128_seed31_fold{fold}.json').read_text())
        require(np.array_equal(saved['fit'], split['inner_train']) and
                np.array_equal(saved['valid'], split['inner_val']), 'canonical inner partition changed')
        np.testing.assert_array_equal(t[saved['fit']], saved['fit_time'])
        np.testing.assert_array_equal(e[saved['fit']], saved['fit_event'])
        for seed in (31, 47):
            np.testing.assert_array_equal(cells[seed, fold]['fit'], saved['fit'])
            np.testing.assert_array_equal(cells[seed, fold]['valid'], saved['valid'])
    return t, e


def distribution_rows(prepared):
    rows, teachers = [], []
    for fold, saved in prepared.items():
        fit, valid, ms = saved['fit'], saved['valid'], saved['mainstate']
        cf = np.asarray(saved['base'][fit], np.float64)
        full = np.asarray(ms['full_X'][fit], np.float64) @ np.asarray(ms['full_R0_coefficients'], np.float64)
        np.testing.assert_allclose(full, ms['fullfit_risk_fit'], rtol=0, atol=1e-10)
        np.testing.assert_array_equal(cf, ms['crossfit_risk_fit'])
        require(float(cf.std()) == saved['scale'], 'saved residual unit differs from fit CF SD')
        rows.append({'fold': fold, 'same_fit_patients': len(fit),
                     **paired_distributions(cf, full),
                     'fullfit_valid_separate_patients': moments(saved['R0_valid']),
                     'valid_SD_over_fit_CF_SD': float(saved['R0_valid'].std()/cf.std()),
                     'valid_SD_over_fit_full_SD': float(saved['R0_valid'].std()/full.std()),
                     'fullfit_in_sample_is_ground_truth': False})
        for teacher in ms['teachers']:
            held = np.asarray(teacher['held'], int)
            full_held = np.asarray(ms['full_X'][held], np.float64) @ ms['full_R0_coefficients']
            np.testing.assert_array_equal(saved['base'][held], teacher['risk_held'])
            teachers.append({'fold': fold, 'teacher': int(teacher['teacher']),
                             'held_patients': len(held),
                             **paired_distributions(saved['base'][held], full_held)})
    return rows, teachers


def compute_grid(cells, prepared, t, e):
    fold_rows, residual_rows = [], []
    for seed in (31, 47):
        for fold in (1, 2, 3):
            cell, saved = cells[seed, fold], prepared[fold]
            fit, valid = cell['fit'], cell['valid']
            base = cell['risks']['R0']
            weights = uno_weights(t[fit], e[fit], t[valid], e[valid], tau=60.)
            labels = np.full(len(valid), fold)
            record = torch.load(runtime.cell_dir('prototype', seed, fold)/'checkpoint.pt',
                                map_location='cpu', weights_only=False)
            require(record['data_audit']['prepared_sha256'] == runtime.sha(data.prepared_path(fold, 'prototype')),
                    'checkpoint prepared-file lineage differs')
            for arm in ('R1', 'R2', 'R3'):
                trained = cell['risks'][arm]
                residual_rows.append({'seed': seed, 'fold': fold, 'arm': arm,
                    'fit_CF_offset': residual_summary(saved['base'][fit], record['arms'][arm]['fit_risk'], saved['scale']),
                    'valid_fullfit_offset': residual_summary(base, trained, saved['scale'])})
                for alpha in SCALES:
                    risk = scaled_risk(base, trained, alpha)
                    point = metrics(t[valid], e[valid], risk, labels, weights)
                    paired = pair_transition_summary(t[valid], e[valid], base, risk, folds=labels, ipcw=weights)
                    value = {'seed': seed, 'fold': fold, 'arm': arm, 'alpha': alpha,
                             'valid_patients': len(valid), **point}
                    require(all(np.isfinite(v) for v in point.values()), 'undefined grid metric')
                    for one in paired['rows']:
                        m = one['metric']; prefix = 'Harrell' if m == 'Harrell' else 'Uno'
                        require(abs(one['C1']-point[m]) < 1e-12, 'paired/point metrics differ')
                        value[m+'_repair_mass'] = one['repair_credit_mass']
                        value[m+'_damage_mass'] = one['damage_credit_mass']
                        require(abs(one['denominator']-point[prefix+'_den']) < 1e-10,
                                'paired/point denominators differ')
                        if alpha == 0:
                            require(one['repair'] == one['damage'] == 0, 'OFF pair changes must be zero')
                    fold_rows.append(value)
    seed_rows = []
    for seed in (31, 47):
        for arm in ('R1', 'R2', 'R3'):
            for alpha in SCALES:
                rows = [r for r in fold_rows if (r['seed'], r['arm'], r['alpha']) == (seed, arm, alpha)]
                require(len(rows) == 3, 'grid coverage incomplete')
                row = {'seed': seed, 'arm': arm, 'alpha': alpha}
                for metric, prefix in (('Harrell', 'Harrell'), ('Uno60', 'Uno')):
                    den = sum(r[prefix+'_den'] for r in rows)
                    row[metric] = sum(r[prefix+'_num'] for r in rows)/den
                    row[metric+'_repair'] = sum(r[metric+'_repair_mass'] for r in rows)/den
                    row[metric+'_damage'] = sum(r[metric+'_damage_mass'] for r in rows)/den
                    row[metric+'_delta'] = row[metric+'_repair']-row[metric+'_damage']
                seed_rows.append(row)
    means = []
    for arm in ('R1', 'R2', 'R3'):
        for alpha in SCALES:
            rows = [r for r in seed_rows if (r['arm'], r['alpha']) == (arm, alpha)]
            means.append({'arm': arm, 'alpha': alpha,
                          **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]
                             if k not in ('seed', 'arm', 'alpha')}})
    old = json.loads((runtime.REPORT/'INNER_RESULTS.json').read_text())
    for row in seed_rows:
        name = 'R0' if row['alpha'] == 0 else row['arm'] if row['alpha'] == 1 else None
        reference = next(r for r in old['seed_rows'] if r['seed'] == row['seed'] and r['model'] == (name or 'R0'))
        for m in METRICS:
            if name: require(abs(row[m]-reference[m]) < 1e-12, 'OFF/original metric did not replay')
            baseline = next(r for r in old['seed_rows'] if r['seed'] == row['seed'] and r['model'] == 'R0')
            require(abs(row[m]-baseline[m]-row[m+'_delta']) < 1e-12, 'repair-damage identity failed')
    return fold_rows, seed_rows, means, residual_rows


def main():
    config = json.loads(CONFIG.read_text())
    require(tuple(config['fixed_scales']) == SCALES and config['training_source'] == SOURCE,
            'fixed grid/source differs from lock')
    require(runtime.source() == SOURCE, 'original training source changed')
    target = REPORT/'DIAGNOSTIC.json'
    require(not target.exists(), 'append-only diagnostic refuses existing output')
    snapshot = inputs_snapshot()
    verify_replay_evidence()
    lock = {'schema': config['schema'], 'config_sha256': runtime.sha(CONFIG),
            'protocol_sha256': runtime.sha(REPORT/'PROTOCOL.md'),
            'first_diagnostic_script_sha256': runtime.sha(Path(__file__)),
            'fixed_scales': list(SCALES), 'prior_results_already_seen': True,
            'new_grid_outputs_seen_when_locked': False, 'alpha_selection_permitted': False}
    lock_path = runtime.PRIVATE/'scale_diagnostic_LOCK_PRIVATE.json'
    require(not lock_path.exists(), 'diagnostic lock already exists; review before retry')
    runtime.atomic_json(lock_path, lock)
    calls = {}
    def blocker(name):
        calls[name] = 0
        def blocked(*args, **kwargs):
            calls[name] += 1
            raise RuntimeError('no-fit frozen diagnostic forbids '+name)
        return blocked
    from src.phase10_trimodal import math as cox
    from src.phase13_rna import math as legacy_cox
    with contextlib.ExitStack() as stack:
        for obj, name in ((train, 'train_arm'), (train, 'run_cell'), (train, '_forward'),
                          (LocalRouting, 'forward'), (data, 'prepare'), (data, 'build_prepared'),
                          (data, 'fit_preprocessor'), (data, 'fit_ridge'), (data.PCA, 'fit'),
                          (data.parent, 'splits'), (data.parent, 'endpoints'),
                          (cox, 'fit_ridge'), (legacy_cox, 'fit_ridge'), (legacy_cox, 'block_fit'),
                          (torch.optim.AdamW, 'step'), (torch.optim.SGD, 'step')):
            stack.enter_context(patch.object(obj, name, blocker(obj.__name__+'.'+name)))
        cells, audit = verify_stage(runtime.PRIVATE, 'prototype', SOURCE)
        prepared = {fold: data.read_prepared(fold, 'prototype')[0] for fold in (1, 2, 3)}
        distributions, teachers = distribution_rows(prepared)
        t, e = endpoint_and_partitions(prepared, cells)
        fold_rows, seed_rows, means, residuals = compute_grid(cells, prepared, t, e)
    require(not any(calls.values()), 'forbidden API attempted')
    require(runtime.source() == SOURCE and all(runtime.sha(Path(p)) == h for p, h in snapshot.items()),
            'original artifact or lock bytes changed during diagnostic')
    result = {'schema': config['schema'], 'status': 'COMPLETE_POSTHOC_DIAGNOSTIC_ONLY',
        'training_source': SOURCE, 'diagnostic_script_sha256': runtime.sha(Path(__file__)),
        'helper_sha256': runtime.sha(ROOT/'src/phase25_scale_diagnostic.py'),
        'config_sha256': runtime.sha(CONFIG), 'protocol_sha256': runtime.sha(REPORT/'PROTOCOL.md'),
        'original_public_manifest_sha256': runtime.sha(runtime.REPORT/'ARTIFACT_MANIFEST.json'),
        'prior_MAIN_REPLAY_sha256': runtime.sha(runtime.REPORT/'MAIN_REPLAY_AUDIT.json'),
        'prior_REPLAY_sha256': runtime.sha(runtime.REPORT/'REPLAY_AUDIT.json'),
        'stage': 'prototype', 'cells': 6, 'stored_neural_models': 18,
        'fixed_scales': list(SCALES), 'aggregation': config['aggregation'],
        'fold_rows': fold_rows, 'seed_rows': seed_rows, 'two_seed_means': means,
        'fit_distribution_rows': distributions, 'teacher_rows': teachers,
        'residual_rows': residuals, 'forbidden_API_call_counts': calls,
        'original_inputs_checked_before_after': len(snapshot),
        'private_pre_output_lock_sha256': runtime.sha(lock_path),
        'original_STOP_unchanged': True, 'new_training_fits': 0, 'new_neural_forward_calls': 0,
        'new_calibration_fits': 0, 'new_bootstrap_draws': 0, 'alpha_selected': False,
        'scales_ranked': False, 'independent_performance_evidence': False,
        'outer_started': False, 'patient_arrays_or_weights_public': False,
        'prepared_existing_private_fit_label_bytes_deserialized': True,
        'fullfit_training_risk_is_in_sample_not_ground_truth': True,
        'source_audit_complete_cells': audit['complete_cells']}
    runtime.atomic_json(target, result)
    print(json.dumps({'status': result['status'], 'fold_scale_rows': len(fold_rows),
                      'new_training_fits': 0, 'alpha_selected': False, 'original_STOP_unchanged': True}), flush=True)


if __name__ == '__main__': main()
