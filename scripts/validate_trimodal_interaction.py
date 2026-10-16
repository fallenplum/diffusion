"""Independent validation of the locked INNER-ONLY interaction feasibility run.

No patient-level prediction is written publicly, no outer risk is calculated,
and no bootstrap/independence claim is made for overlapping inner partitions.
This evaluator is deliberately outside the frozen training-source identity.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from src.trimodal_interaction.data import P, R, ROOT, Data, cfg, digest, parent, source, write
from src.trimodal_interaction.model import Model


def independent_functions():
    path = ROOT / 'scripts/validate_phase17_independent.py'
    spec = importlib.util.spec_from_file_location('independent_phase17_pairs', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independent_efron(risk, time, event):
    """Literal Efron denominator calculation, not the training weight matrix."""
    risk = np.asarray(risk, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    if risk.shape != time.shape or time.shape != event.shape or not event.any():
        raise ValueError('invalid independent Efron input')
    shift = risk.max()
    exponent = np.exp(risk - shift)
    value = -risk[event].sum()
    for when in np.unique(time[event]):
        deaths = (time == when) & event
        n = int(deaths.sum())
        risk_sum = exponent[time >= when].sum()
        death_sum = exponent[deaths].sum()
        for j in range(n):
            denominator = risk_sum - j / n * death_sum
            if denominator <= 0:
                raise ValueError('nonpositive independent Efron denominator')
            value += np.log(denominator) + shift
    return float(value / event.sum())


def validate_history(checkpoint, config):
    summary, history = checkpoint['summary'], checkpoint['history']
    arm, tc = summary['arm'], config['training']
    if arm == 'A0':
        if history or summary['selected_epoch'] != 0 or summary['actual_epochs'] != 0:
            raise ValueError('convex reference incorrectly counted as neural epochs')
        return
    if not history or len(history) != summary['actual_epochs']:
        raise ValueError('missing/truncated training history')
    best, selected, ema = float('inf'), 0, None
    finite = ['train_loss', 'penalty', 'validation_loss', 'validation_loss_EMA',
              'validation_Harrell', 'gradient_norm', 'query_gradient',
              'gate_gradient', 'seconds']
    for epoch, row in enumerate(history, 1):
        if row['epoch'] != epoch or not all(np.isfinite(row[key]) for key in finite):
            raise ValueError('nonfinite or noncontiguous training curve')
        if any(row[key] < 0 for key in ['penalty', 'gradient_norm', 'query_gradient',
                                       'gate_gradient', 'seconds']):
            raise ValueError('invalid gradient/penalty/time diagnostic')
        ema = row['validation_loss'] if ema is None else .2 * row['validation_loss'] + .8 * ema
        if abs(ema - row['validation_loss_EMA']) > 1e-10:
            raise ValueError('validation EMA replay disagrees')
        if epoch >= tc['minimum_epoch'] and ema < best - 1e-8:
            best, selected = ema, epoch
        if epoch - selected >= tc['early_stopping_patience'] and epoch != len(history):
            raise ValueError('training continued after the locked stopping rule')
    if selected != summary['selected_epoch']:
        raise ValueError('selected epoch does not minimize the locked EMA criterion')
    if len(history) > tc['max_epochs'] or selected < tc['minimum_epoch']:
        raise ValueError('epoch outside locked schedule')
    if bool(summary['hit_maximum']) != (len(history) == tc['max_epochs']):
        raise ValueError('stored maximum-budget flag disagrees with actual history')
    if (len(history) < tc['max_epochs'] and
            len(history) - selected < tc['early_stopping_patience']):
        raise ValueError('training stopped without the locked patience or budget limit')


def tensor_arguments(data, indices, device, wrong=False):
    indices = data.check_indices(indices)
    own = torch.as_tensor(indices, dtype=torch.long)
    donor = torch.as_tensor(data.wrong[indices], dtype=torch.long) if wrong else own
    f = data.ft
    return tuple(value.to(device) for value in
                 [f['X'][own], f['C'][own], f['R'][donor], f['RM'][donor],
                  f['P'][own], f['PM'][own]])


def replay(checkpoint, data, config, device):
    """Rebuild the selected model and check saved risk, pairing and both nulls."""
    arm = checkpoint['summary']['arm']
    state = checkpoint['state']
    model = Model(data, arm, state['main'], config).to(device)
    if arm != 'A0':
        for name, expected in [('c0', data.C0), ('r0', data.R0), ('rm0', data.RM0)]:
            saved = state['interaction.' + name]
            if not torch.equal(saved.cpu(), torch.as_tensor(expected).to(saved.dtype).cpu()):
                raise ValueError('model reference is not the exact fit-partition mean: ' + name)
    model.load_state_dict(state)
    model.eval()
    chunk = config['training']['microbatch_patients']
    wrong = arm == 'A6'
    actual, swapped = [], []
    branch_bound, cnull, rnull = 0., 0., 0.
    with torch.no_grad():
        for start in range(0, len(data.valid), chunk):
            indices = data.valid[start:start + chunk]
            args = tensor_arguments(data, indices, device, wrong)
            risk, diagnostic = model(*args, diagnostic=True)
            actual.append(risk.cpu().numpy())
            other = tensor_arguments(data, indices, device, not wrong)
            swapped.append(model(*other).cpu().numpy())
            branch_bound = max(branch_bound, float(diagnostic['risk_abs'].max()))
            if arm in ['A4', 'A5', 'A6', 'A7']:
                main = args[0] @ model.main
                changed = list(args)
                changed[1] = model.interaction.c0.expand_as(changed[1])
                cnull = max(cnull, float((model(*changed) - main).abs().max()))
                changed = list(args)
                changed[2] = model.interaction.r0.expand_as(changed[2])
                changed[3] = model.interaction.rm0.expand_as(changed[3])
                rnull = max(rnull, float((model(*changed) - main).abs().max()))
    risk_error = float(np.max(np.abs(np.concatenate(actual) - checkpoint['risk'])))
    wrong_error = float(np.max(np.abs(np.concatenate(swapped) - checkpoint['wrong_risk'])))
    if risk_error > 2e-6 or wrong_error > 2e-6:
        raise ValueError(('checkpoint replay changed', arm, risk_error, wrong_error))
    if cnull != 0 or rnull != 0:
        raise ValueError(('mixed null is not exact zero', arm, cnull, rnull))
    if branch_bound > config['training']['interaction_risk_bound'] + 1e-7:
        raise ValueError(('interaction risk bound failed', arm, branch_bound))
    return {'replay_max_error': risk_error, 'wrong_replay_max_error': wrong_error,
            'clinical_null_max': cnull, 'complete_RNA_null_max': rnull,
            'interaction_abs_max': branch_bound}


def metric_rows_for_risk(independent, risk, data, arm, seed, actual_seed):
    from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
    from sksurv.util import Surv

    fit, valid = data.fit, data.valid
    t, e = data.ev['time'], data.ev['event'].astype(bool)
    train_t, train_e = t[fit], e[fit]
    valid_t, valid_e = t[valid], e[valid]
    ipcw = independent.weights(train_t, train_e, valid_t, valid_e, tau=60.)
    hn, hd, *_ = independent.components(valid_t, valid_e, risk, np.ones(len(valid)))
    un, ud, *_ = independent.components(valid_t, valid_e, risk, ipcw)
    if hd <= 0 or ud <= 0:
        raise ValueError('zero inner validation comparable-pair denominator')
    hr = concordance_index_censored(valid_e, valid_t, risk, tied_tol=1e-8)[0]
    ur = concordance_index_ipcw(Surv.from_arrays(train_e, train_t),
                                Surv.from_arrays(valid_e, valid_t), risk,
                                tau=60., tied_tol=1e-8)[0]
    if abs(hn / hd - hr) > 1e-10 or abs(un / ud - ur) > 1e-10:
        raise ValueError(('independent metric disagrees with sksurv', arm, seed, data.fold))
    return {'endpoint': 'DSS', 'role': 'inner_validation_NOT_OOF', 'arm': arm,
            'seed': seed, 'fold': data.fold, 'actual_fit_seed': actual_seed,
            'convex_reference_reused': seed != actual_seed,
            'fit_patients': len(fit), 'fit_events': int(train_e.sum()),
            'validation_patients': len(valid), 'validation_events': int(valid_e.sum()),
            'Harrell': hn / hd, 'Uno60': un / ud,
            'H_num': hn, 'H_den': hd, 'U_num': un, 'U_den': ud,
            'IPCW_training_partition': 'this_exact_inner_train',
            'IPCW_tau_months': 60., 'maximum_sksurv_error': max(abs(hn / hd - hr), abs(un / ud - ur))}


def summarize(config, fold_rows, checkpoint_rows):
    frames = pd.DataFrame(fold_rows)
    seeds = []
    for arm in config['arms']:
        for seed in config['seeds']:
            rows = frames[(frames.arm == arm) & (frames.seed == seed)]
            sums = {key: float(rows[key].sum()) for key in ['H_num', 'H_den', 'U_num', 'U_den']}
            seeds.append({'endpoint': 'DSS', 'role': 'overlapping_inner_validation_NOT_OOF',
                          'arm': arm, 'seed': seed, 'fold_cells': len(rows),
                          'Harrell': sums['H_num'] / sums['H_den'],
                          'Uno60': sums['U_num'] / sums['U_den'], **sums})
    seed_frame = pd.DataFrame(seeds)
    summaries = []
    actual = pd.DataFrame(checkpoint_rows)
    for arm in config['arms']:
        ss = seed_frame[seed_frame.arm == arm]
        aa = actual[actual.arm == arm]
        summaries.append({'arm': arm, 'role': 'inner_feasibility_NOT_OOF',
                          'Harrell_mean_over_seed_algorithms': float(ss.Harrell.mean()),
                          'Uno60_mean_over_seed_algorithms': float(ss.Uno60.mean()),
                          'Harrell_seed_min': float(ss.Harrell.min()),
                          'Harrell_seed_max': float(ss.Harrell.max()),
                          'Uno60_seed_min': float(ss.Uno60.min()),
                          'Uno60_seed_max': float(ss.Uno60.max()),
                          'real_training_cells_this_stage': int(aa.stage_real_fit.sum()),
                          'selected_model_cells': len(aa),
                          'evaluation_cells': int((frames.arm == arm).sum()),
                          'selected_epoch_median': float(aa.selected_epoch.median()),
                          'actual_epoch_median': float(aa.actual_epochs.median()),
                          'hit_maximum_fraction': float(aa.hit_maximum.mean()),
                          'parameters_registered_mean': float(aa.parameters_registered.mean()),
                          'parameters_gradient_active_mean': float(aa.parameters_gradient_active.mean()),
                          'seconds_mean_concurrent_run': float(aa.seconds.mean()),
                          'peak_GPU_MiB_max': float(aa.peak_GPU_MB.max())})
    frames.to_csv(R / 'FOLD_METRICS.csv', index=False)
    seed_frame.to_csv(R / 'SEED_INNER_METRICS.csv', index=False)
    pd.DataFrame(summaries).to_csv(R / 'ARM_SUMMARY.csv', index=False)
    return frames, seed_frame


def paired_diagnostics(config, folds, seeds):
    rows, decisions = [], []
    for family, contrasts in [('primary', config['primary_contrasts']),
                              ('secondary', config['secondary'])]:
        for arm_a, arm_b in contrasts:
            for metric in ['Harrell', 'Uno60']:
                aa = seeds[seeds.arm == arm_a].set_index('seed')[metric]
                bb = seeds[seeds.arm == arm_b].set_index('seed')[metric]
                delta_seed = aa - bb
                af = folds[folds.arm == arm_a].set_index(['seed', 'fold'])[metric]
                bf = folds[folds.arm == arm_b].set_index(['seed', 'fold'])[metric]
                delta_fold = af - bf
                row = {'comparison': arm_a + '-' + arm_b, 'family': family,
                       'metric': metric, 'mean_seed_pair_weighted_delta': float(delta_seed.mean()),
                       'positive_seed_algorithms': int((delta_seed > 0).sum()),
                       'positive_inner_cells': int((delta_fold > 0).sum()),
                       'inner_cells': len(delta_fold),
                       'positive_inner_cell_fraction': float((delta_fold > 0).mean()),
                       'mean_unweighted_cell_delta': float(delta_fold.mean()),
                       'minimum_cell_delta': float(delta_fold.min()),
                       'maximum_cell_delta': float(delta_fold.max()),
                       'scope': 'descriptive; overlapping inner patients; no confidence interval or p-value'}
                row.update({'seed_' + str(seed) + '_pair_weighted_delta': float(delta_seed.loc[seed])
                            for seed in config['seeds']})
                rows.append(row)
                if family == 'primary' and metric == 'Harrell':
                    threshold = config['gate']['required_each_seed_mean_Harrell_gain']
                    fraction = config['gate']['required_primary_positive_cell_fraction']
                    decisions.append({'comparison': row['comparison'],
                                      'each_seed_delta_above_locked_threshold': bool((delta_seed > threshold).all()),
                                      'positive_cells': int((delta_fold > 0).sum()),
                                      'required_fraction': fraction,
                                      'pass': bool((delta_seed > threshold).all() and
                                                   (delta_fold > 0).mean() >= fraction)})
    pd.DataFrame(rows).to_csv(R / 'PAIRED_INNER_DIAGNOSTICS.csv', index=False)
    return decisions


def main():
    global R
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--budget160', action='store_true')
    args = parser.parse_args()
    original_config = cfg()
    config = copy.deepcopy(original_config)
    expected_source = source()
    lock = json.loads((P / 'LOCK.json').read_text())
    original_done = json.loads((P / 'DONE.json').read_text())
    if lock['source'] != expected_source or original_done['source'] != expected_source:
        raise ValueError('training lock/DONE/source disagreement')
    if original_done.get('status') != 'ALL_LOCKED_INNER_CELLS_FINISHED':
        raise ValueError('formal inner queue has not finished')
    if original_done['outer_fits'] != 0 or original_done['patient_OOF_files'] != 0:
        raise ValueError('unexpected outer fits or OOF claimed for an inner-only run')
    if (P / 'OOF_PRIVATE.npz').exists() or (P / 'outer').exists():
        raise ValueError('unexpected outer prediction location in inner-only private root')
    budget_source, budget_done, selected_root = None, None, P
    if args.budget160:
        selected_root = P / 'budget160'
        R = R / 'budget160'
        path = ROOT / 'scripts/extend_trimodal_budget.py'
        spec = importlib.util.spec_from_file_location('locked_trimodal_budget160', path)
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
        budget_source = extension.budget_source()
        budget_lock = json.loads((selected_root / 'LOCK.json').read_text())
        budget_done = json.loads((selected_root / 'DONE.json').read_text())
        budget_config = json.loads((ROOT / 'configs/trimodal_interaction_budget160.json').read_text())
        if (budget_lock['budget_source'] != budget_source or
                budget_lock['original_source'] != expected_source or
                budget_lock['config'] != budget_config or
                budget_done['budget_source'] != budget_source or
                budget_done['source'] != expected_source):
            raise ValueError('budget160 config/script/source/lock disagreement')
        if budget_done['status'] != 'ALL_BUDGET_CELLS_VERIFIED' or budget_done['outer_fits'] != 0:
            raise ValueError('budget160 queue incomplete or reaches outer evaluation')
        config['training']['max_epochs'] = int(budget_config['max_epochs'])
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    R.mkdir(parents=True, exist_ok=True)
    independent = independent_functions()
    cohort = parent.load()
    endpoint = parent.endpoints(cohort)['DSS']
    if len(cohort['patient_ids']) != 512 or int(endpoint['event'][endpoint['valid']].sum()) != 65:
        raise ValueError('DSS cohort identity changed')
    checkpoints, curves, fold_rows, audits = [], [], [], []
    partition_audits, seen_valid = [], []
    cached = {}
    extension_reruns, extension_references = 0, 0
    for fold in config['folds']:
        parts = parent.splits(cohort, 'DSS', fold)
        if np.intersect1d(parts['inner_val'], parts['outer_test']).size:
            raise ValueError('inner validation reaches this fold outer test')
        seen_valid.extend(parts['inner_val'].tolist())
        partition_audits.append({'fold': fold, 'fit_patients': len(parts['inner_train']),
                                 'validation_patients': len(parts['inner_val']),
                                 'outer_test_patients_never_predicted_here': len(parts['outer_test'])})
        for seed in config['seeds']:
            data = Data(fold, 'inner', seed)
            for arm in config['arms']:
                actual_seed = config['seeds'][0] if arm == 'A0' else seed
                key = arm, actual_seed, fold
                if key not in cached:
                    label = f'{arm}_seed{actual_seed}_fold{fold}'
                    original_path = P / 'checkpoints' / (label + '.pt')
                    original_cell = json.loads((P / 'cells' / (label + '.json')).read_text())
                    if (original_cell['source'] != expected_source or
                            digest(original_path) != original_cell['checkpoint_sha256']):
                        raise ValueError('original40 source/checkpoint changed: ' + label)
                    cell = (json.loads((selected_root / 'cells' / (label + '.json')).read_text())
                            if args.budget160 else original_cell)
                    path = Path(cell['checkpoint_path']) if args.budget160 else original_path
                    reused = False
                    if args.budget160:
                        if cell['budget_source'] != budget_source:
                            raise ValueError('budget160 cell source changed: ' + label)
                        reused = bool(cell['reused_reference'])
                        reached_original_cap = original_cell['summary']['actual_epochs'] == original_config['training']['max_epochs']
                        if reused == reached_original_cap:
                            raise ValueError('budget reuse/rerun rule violated: ' + label)
                        expected_path = original_path if reused else selected_root / 'checkpoints' / (label + '.pt')
                        if path != expected_path:
                            raise ValueError('budget checkpoint points outside its locked location')
                        extension_references += int(reused)
                        extension_reruns += int(not reused)
                    if cell['source'] != expected_source or digest(path) != cell['checkpoint_sha256']:
                        raise ValueError('checkpoint receipt/source/SHA changed: ' + label)
                    q = torch.load(path, map_location='cpu', weights_only=False)
                    if q['source'] != expected_source or q['summary'] != cell['summary']:
                        raise ValueError('checkpoint payload and receipt disagree: ' + label)
                    summary = q['summary']
                    if (summary['arm'], summary['seed'], summary['fold'], summary['role']) != (arm, actual_seed, fold, 'inner_only'):
                        raise ValueError('checkpoint role or model identity changed')
                    if (not np.array_equal(q['fit'], parts['inner_train']) or
                            not np.array_equal(q['valid'], parts['inner_val'])):
                        raise ValueError('checkpoint partition differs from immutable manifest')
                    expected_keys = {'summary', 'history', 'source', 'fit', 'valid', 'risk',
                                     'baseline_risk', 'wrong_risk', 'state', 'preprocessing', 'ridge'}
                    if args.budget160 and not reused:
                        expected_keys |= {'budget_source', 'original_prefix_max_error'}
                        if q['budget_source'] != budget_source:
                            raise ValueError('budget checkpoint source differs from its receipt')
                        old_q = torch.load(original_path, map_location='cpu', weights_only=False)
                        if len(q['history']) < len(old_q['history']):
                            raise ValueError('extended history shorter than original40')
                        fields = ['epoch', 'train_loss', 'penalty', 'validation_loss',
                                  'validation_loss_EMA', 'validation_Harrell',
                                  'gradient_norm', 'query_gradient', 'gate_gradient']
                        prefix_error = max(abs(a[field] - b[field])
                                           for a, b in zip(old_q['history'], q['history'])
                                           for field in fields)
                        if (prefix_error > 1e-7 or
                                abs(prefix_error - cell['original_prefix_max_error']) > 1e-12 or
                                abs(prefix_error - q['original_prefix_max_error']) > 1e-12):
                            raise ValueError('first40 deterministic training fields changed')
                    if set(q) != expected_keys:
                        raise ValueError('unexpected checkpoint payload; cannot certify no outer risk')
                    for name in ['risk', 'baseline_risk', 'wrong_risk']:
                        values = np.asarray(q[name])
                        if values.shape != (len(data.valid),) or not np.isfinite(values).all():
                            raise ValueError('invalid inner-only risk vector: ' + name)
                    if q['preprocessing'] != data.audit:
                        raise ValueError('selected checkpoint preprocessing audit changed')
                    if not np.isfinite(q['ridge']['loss']) or q['ridge']['gradient_max'] > 2e-5:
                        raise ValueError('convex initializer has not converged')
                    validate_history(q, config)
                    replay_audit = replay(q, data, config, device)
                    loss = independent_efron(q['risk'], data.ev['time'][data.valid],
                                             data.ev['event'][data.valid])
                    if abs(loss - summary['validation_loss']) > 2e-6:
                        raise ValueError('independent Efron disagrees with stored selected-model loss')
                    if arm != 'A0':
                        selected_row = q['history'][summary['selected_epoch'] - 1]
                        if abs(loss - selected_row['validation_loss']) > 2e-6:
                            raise ValueError('selected checkpoint is inconsistent with epoch curve')
                    for row in q['history']:
                        curves.append({'arm': arm, 'seed': actual_seed, 'fold': fold,
                                       'role': 'inner_training', **row})
                    checkpoints.append({**summary, 'stage_real_fit': not args.budget160 or not reused})
                    audits.append({'arm': arm, 'seed': actual_seed, 'fold': fold,
                                   'checkpoint_sha256': cell['checkpoint_sha256'],
                                   'source': expected_source, 'independent_Efron': loss,
                                   'budget_source': budget_source,
                                   'reused_original40_reference': reused if args.budget160 else False,
                                   **replay_audit})
                    cached[key] = q
                q = cached[key]
                fold_rows.append(metric_rows_for_risk(independent, np.asarray(q['risk']),
                                                       data, arm, seed, actual_seed))
                if abs(fold_rows[-1]['Harrell'] - q['summary']['Harrell']) > 1e-10:
                    raise ValueError('independent Harrell disagrees with saved summary')
    expected_neural = (len(config['arms']) - 1) * len(config['seeds']) * len(config['folds'])
    expected_convex = len(config['folds'])
    if len(cached) != expected_neural + expected_convex or original_done['cells'] != len(cached):
        raise ValueError('real cell count is not the locked completed matrix')
    expected_labels = {f'{arm}_seed{seed}_fold{fold}'
                       for arm in config['arms'] for fold in config['folds']
                       for seed in ([config['seeds'][0]] if arm == 'A0' else config['seeds'])}
    if ({path.stem for path in (P / 'cells').glob('*.json')} != expected_labels or
            {path.stem for path in (P / 'checkpoints').glob('*.pt')} != expected_labels):
        raise ValueError('missing or unexpected real cell/checkpoint outside the locked matrix')
    if (original_done['real_neural_fits'] != expected_neural or
            original_done['real_convex_baseline_fits'] != expected_convex or
            original_done['evaluation_baseline_references'] != len(config['seeds']) * len(config['folds'])):
        raise ValueError('original DONE fit/reference counts disagree with actual matrix')
    if args.budget160 and (budget_done['cells'] != len(cached) or
                           budget_done['new_real_neural_fits'] != extension_reruns or
                           budget_done['evaluation_references'] != extension_references):
        raise ValueError('budget160 actual fit/reference counts disagree')
    if args.budget160 and {path.stem for path in (selected_root / 'cells').glob('*.json')} != expected_labels:
        raise ValueError('budget160 contains missing or unexpected evaluation receipts')
    pd.DataFrame(curves).to_csv(R / 'TRAINING_CURVES.csv', index=False)
    pd.DataFrame(audits).to_csv(R / 'CHECKPOINT_AUDIT.csv', index=False)
    folds, seeds = summarize(config, fold_rows, checkpoints)
    comparison_gates = paired_diagnostics(config, folds, seeds)
    neural = [row for row in checkpoints if row['arm'] != 'A0']
    cap_fraction = float(np.mean([row['hit_maximum'] for row in neural]))
    budget_per_arm = []
    for arm in config['arms']:
        if arm == 'A0':
            continue
        arm_rows = [row for row in neural if row['arm'] == arm]
        fraction = float(np.mean([row['hit_maximum'] for row in arm_rows]))
        budget_per_arm.append({'arm': arm, 'neural_cells': len(arm_rows),
                               'hit_maximum_cells': sum(row['hit_maximum'] for row in arm_rows),
                               'hit_maximum_fraction': fraction,
                               'maximum_epochs': config['training']['max_epochs'],
                               'budget_gate_pass': fraction <= config['gate']['budget_sufficient_max_cap_fraction']})
    budget_ok = all(row['budget_gate_pass'] for row in budget_per_arm)
    gradients = []
    for (arm, seed, fold), q in cached.items():
        if arm == 'A5':
            gradients.append({'arm': arm, 'seed': seed, 'fold': fold,
                              'query_path_nonzero': any(row['query_gradient'] > 1e-12 for row in q['history']),
                              'gate_path_nonzero': any(row['gate_gradient'] > 1e-12 for row in q['history'])})
    gradients_ok = all(row['query_path_nonzero'] and row['gate_path_nonzero'] for row in gradients)
    all_primary = all(row['pass'] for row in comparison_gates)
    go = budget_ok and gradients_ok and all_primary
    counts = {'real_neural_training_cells_this_stage': extension_reruns if args.budget160 else expected_neural,
              'real_baseline_only_convex_fits_this_stage': 0 if args.budget160 else expected_convex,
              'neural_initialization_ridge_solver_calls_this_stage': extension_reruns if args.budget160 else expected_neural,
              'total_ridge_solver_calls_this_stage': extension_reruns if args.budget160 else expected_neural + expected_convex,
              'real_completed_model_cells_this_stage': extension_reruns if args.budget160 else len(cached),
              'selected_model_cells_for_evaluation': len(cached),
              'original40_references_without_refit': extension_references if args.budget160 else 0,
              'cumulative_neural_training_cells': expected_neural + (extension_reruns if args.budget160 else 0),
              'cumulative_baseline_only_convex_fits': expected_convex,
              'cumulative_ridge_solver_calls': expected_neural + expected_convex + (extension_reruns if args.budget160 else 0),
              'convex_baseline_evaluation_references': len(config['seeds']) * len(config['folds']),
              'extra_convex_references_without_refit': (len(config['seeds']) - 1) * len(config['folds']),
              'metric_cells': len(fold_rows), 'checkpoint_replays': len(audits),
              'outer_fits': 0, 'outer_prediction_vectors': 0, 'patient_OOF_files': 0}
    decision = ('BUDGET_UNRESOLVED' if not budget_ok else
                'GO_TO_SEPARATELY_PREREGISTERED_FOLLOWUP' if go else
                'STOP_CURRENT_INTERACTION_PILOT')
    gate = {'decision': decision,
            'scope': 'locked engineering decision; overlapping inner validation; not independent efficacy inference',
            'budget_gate_pass': budget_ok, 'maximum_epoch_fraction': cap_fraction,
            'maximum_epoch_fraction_per_arm': budget_per_arm,
            'budget_gate_aggregation': 'every relevant neural arm must pass separately; pooled fraction is descriptive only',
            'nonzero_A5_query_and_gate_gradients_pass': gradients_ok,
            'primary_comparisons': comparison_gates,
            'gradient_path_checks': gradients,
            'does_not_imply_RNA_redundancy_or_no_biological_interaction': True}
    write(R / 'STAGE_DECISION.json', gate)
    write(R / 'EXECUTION_COUNTS.json', counts)
    write(R / 'BUDGET_AUDIT.json', {'maximum_epochs': config['training']['max_epochs'],
                                    'stage': 'budget160_sensitivity' if args.budget160 else 'original40',
                                    'all_relevant_arms_pass': budget_ok,
                                    'per_arm': budget_per_arm})
    write(R / 'VALIDATION.json', {
        'status': 'PASS', 'source': expected_source, 'evaluator_sha256': digest(Path(__file__)),
        'budget_source': budget_source,
        'stage': 'budget160_sensitivity' if args.budget160 else 'original40',
        'role': 'INNER_ONLY_FEASIBILITY_NOT_OOF', 'checkpoint_replay_device': device,
        'counts': counts, 'immutable_partition_checks': partition_audits,
        'unique_inner_validation_patients': len(set(seen_valid)),
        'sum_inner_validation_rows_before_seed_repetition': len(seen_valid),
        'overlapping_inner_patients_not_treated_as_independent': True,
        'no_confidence_intervals_or_p_values_computed': True,
        'no_cross_fold_risk_ranking_or_risk_ensembling': True,
        'IPCW_estimator': 'independent reverse KM fitted on each exact inner-train, tau60',
        'metrics_verified_against_sksurv': True,
        'maximum_sksurv_difference': float(folds.maximum_sksurv_error.max()),
        'all_selected_checkpoints_replayed': True,
        'maximum_checkpoint_risk_error': max(row['replay_max_error'] for row in audits),
        'both_complete_reference_nulls_exact_zero': True,
        'all_interaction_risks_within_fixed_point25_bound': True,
        'epoch_EMA_and_stopping_rules_recomputed': True,
        'formal_outer_evaluation': 'NOT STARTED',
        'stage_decision': gate['decision'],
        'evidence_boundary': 'PASS certifies this technical audit, not biological validity, efficacy, equivalence, or external generalization'})
    print('INNER_VALIDATION_PASS', json.dumps(counts), gate['decision'], flush=True)


if __name__ == '__main__':
    main()
