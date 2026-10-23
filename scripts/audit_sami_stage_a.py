"""Independent NumPy replay of the locked tissue-classification experiment.

Never fits, chooses a model, or loads a survival endpoint. Input arrays and
models are private. Only aggregate discrepancies and counts are exported.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def weights(donor, section, labels=None):
    """Independent group-size construction, without importing the trainer."""
    result = np.zeros(len(donor), dtype=np.float64)
    groups = {}
    for i, (d, s) in enumerate(zip(donor, section)):
        groups.setdefault(str(d), {}).setdefault(None if labels is None else int(labels[i]), {}).setdefault(str(s), []).append(i)
    for classes in groups.values():
        for slides in classes.values():
            for indices in slides.values():
                result[indices] = 1 / (len(groups) * len(classes) * len(slides) * len(indices))
    require(abs(result.sum()-1) < 1e-12, 'invalid independent weights')
    return result


def metrics(y, p, donor, section):
    require(p.shape == (len(y), 3) and np.isfinite(p).all(), 'invalid probabilities')
    require((p >= 0).all() and np.max(abs(p.sum(1)-1)) < 1e-10, 'invalid probability simplex')
    w = weights(donor, section)
    wb = weights(donor, section, y)
    pred = p.argmax(1)
    recalls = []
    cm = np.zeros((3, 3), dtype=np.int64)
    wcm = np.zeros((3, 3), dtype=np.float64)
    for c in range(3):
        mask = y == c
        require(mask.any(), 'evaluation class missing')
        recalls.append(float(wb[mask & (pred == c)].sum() / wb[mask].sum()))
        for k in range(3):
            cell = mask & (pred == k)
            cm[c, k] = cell.sum()
            wcm[c, k] = wb[cell].sum()
    confidence = p.max(1)
    ece = 0.
    # Reproduce the locked bin endpoints exactly. Rational b/10 and linspace
    # differ by one ULP at some edges and can change cancellation within bins.
    edges = np.linspace(0., 1., 11)
    for b, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= low) & ((confidence < high) if b < 9 else (confidence <= high))
        if mask.any():
            ece += abs(np.sum(w[mask] * ((pred[mask] == y[mask])-confidence[mask])))
    return dict(balanced_accuracy=float(np.mean(recalls)), accuracy=float(w @ (pred == y)),
                negative_log_likelihood=float(-w @ np.log(np.maximum(p[np.arange(len(y)), y], 1e-15))),
                brier_multiclass_sum=float(w @ np.sum((p-np.eye(3)[y])**2, axis=1)),
                ece_10_equal_width_bins=float(ece), class_recall=recalls,
                confusion_counts=cm.tolist(), balanced_weighted_confusion=wcm.tolist(), n_spots=len(y))


def numeric_equal(left, right, tolerance=1e-10):
    for key, value in left.items():
        require(key in right, 'missing reported metric')
        a, b = np.asarray(value), np.asarray(right[key])
        require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), 'invalid reported metric')
        require(np.max(np.abs(a-b), initial=0) <= tolerance, 'independent metric mismatch: '+key)


def replay_cell(directory, data, expected_train, expected_eval, lineage, accepted_gradient_inf=1e-5):
    receipt = read(directory/'COMPLETE.json')
    require(receipt['status'] == 'COMPLETE' and receipt['lineage'] == lineage, 'cell lineage mismatch')
    tr, ev = np.asarray(receipt['train_index']), np.asarray(receipt['evaluation_index'])
    require(np.array_equal(tr, expected_train) and np.array_equal(ev, expected_eval), 'cell split mismatch')
    require(not set(data['donor'][tr]) & set(data['donor'][ev]), 'donor crosses fit and evaluation')
    for name in ['model', 'predictions']:
        require(sha(directory/(name+'.npz')) == receipt[name+'_sha256'], 'cell artifact integrity failure')
    with np.load(directory/'model.npz', allow_pickle=False) as f:
        W, bias, mean, scale = [f[k].copy() for k in ['W', 'b', 'mean', 'scale']]
    train_x = data['X'][tr].astype(np.float64)
    w = weights(data['donor'][tr], data['section'][tr])
    expected_mean = np.average(train_x, axis=0, weights=w)
    expected_scale = np.sqrt(np.average((train_x-expected_mean)**2, axis=0, weights=w))
    expected_scale[expected_scale <= 1e-12] = 1
    scaler_error = max(float(np.max(abs(expected_mean-mean))), float(np.max(abs(expected_scale-scale))))
    require(scaler_error <= 1e-10, 'standardizer differs from training-only independent computation')
    logits = ((data['X'][ev].astype(np.float64)-mean)/scale) @ W + bias
    logits -= logits.max(1, keepdims=True)
    prob = np.exp(logits)
    prob /= prob.sum(1, keepdims=True)
    with np.load(directory/'predictions.npz', allow_pickle=False) as f:
        require(np.array_equal(f['row_index'], ev), 'prediction row order mismatch')
        prediction_error = float(np.max(abs(f['probabilities']-prob)))
    require(prediction_error <= 1e-10, 'independent checkpoint prediction mismatch')
    computed = metrics(data['y'][ev], prob, data['donor'][ev], data['section'][ev])
    numeric_equal(computed, receipt['metrics'])
    fit = receipt['fit_audit']
    require(np.isfinite(accepted_gradient_inf) and accepted_gradient_inf == 1e-5,
            'unexpected locked optimizer gradient threshold')
    reported_limit, gradient = float(fit['accepted_gradient_inf']), float(fit['gradient_inf'])
    require(np.isfinite(reported_limit) and reported_limit == accepted_gradient_inf,
            'optimizer acceptance threshold differs from locked config')
    require(fit['success'] is True and np.isfinite(gradient)
            and 0. <= gradient <= accepted_gradient_inf, 'optimizer not qualified')
    return receipt, prob, computed, scaler_error, prediction_error


def validate_public_header(public, lineage, class_order):
    require(public.get('lineage') == lineage, 'public lineage mismatch')
    require(public.get('schema_version') == 'sami_stage_a_results_v1'
            and public.get('status') == 'COMPLETE_NOT_YET_INDEPENDENTLY_AUDITED',
            'public result status or schema mismatch')
    require(public.get('n_donors') == 6 and public.get('n_sections') == 12
            and public.get('n_spots') == 13011 and public.get('completed_fits') == 96,
            'public cohort or fit count mismatch')
    require(public.get('class_order') == class_order, 'public class order mismatch')
    folds = public.get('outer_folds')
    require(isinstance(folds, list) and len(folds) == 6
            and [fold.get('outer_fold') for fold in folds] == list(range(6)),
            'public outer fold inventory or order mismatch')


def validate_public_fold(public_fold, outer_index, selected_C):
    require(public_fold.get('outer_fold') == outer_index, 'public outer fold index mismatch')
    require(public_fold.get('selected_C') == selected_C, 'public selected C mismatch')


def audit(args):
    before = {str(p): sha(p) for p in [args.prepared, args.prepared_manifest, args.config, args.public_results]}
    manifest, cfg = read(args.prepared_manifest), read(args.config)
    run = read(args.run_root/'run_manifest.json')
    lineage = run['lineage']
    require(lineage['prepared_npz_sha256'] == sha(args.prepared) == manifest['data_sha256'], 'prepared integrity mismatch')
    require(lineage['prepared_manifest_sha256'] == sha(args.prepared_manifest), 'prepared manifest mismatch')
    require(lineage['config_sha256'] == sha(args.config) == manifest['protocol_sha256'], 'protocol mismatch')
    require(manifest['original_split_lock_sha256'] == cfg['cohort']['split_lock_sha256'], 'original donor split lock mismatch')
    repo = Path(__file__).resolve().parents[1]
    expected_sources = {p: sha(repo/p) for p in ['src/sami_surv/morphology.py', 'scripts/run_sami_stage_a.py']}
    require(run['source_files'] == expected_sources, 'training source bytes differ from locked receipt')
    source_digest = hashlib.sha256(json.dumps(expected_sources, sort_keys=True).encode()).hexdigest()
    require(lineage['source_sha256'] == source_digest, 'source closure hash mismatch')
    with np.load(args.prepared, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    require(set(data) == {'X', 'y', 'donor', 'section', 'barcodes', 'coords_um'}, 'unexpected prepared fields')
    donors = np.unique(data['donor'])
    require(len(donors) == 6 and len(data['y']) == 13011, 'cohort mismatch')
    public = read(args.public_results)
    validate_public_header(public, lineage, cfg['label_contract']['class_order'])
    optimizer_limit = cfg['model']['accepted_gradient_inf']
    require(np.isfinite(optimizer_limit) and optimizer_limit == 1e-5,
            'unexpected locked optimizer gradient threshold')
    oof = np.full((len(data['y']), 3), np.nan)
    assigned = np.zeros(len(data['y']), int)
    rows, fit_count, scaler_max, pred_max = [], 0, 0., 0.
    for oi, held in enumerate(donors):
        directory = args.run_root/'cells'/f'outer_{oi:02d}'
        outer = read(directory/'COMPLETE.json')
        require(outer['status'] == 'COMPLETE' and outer['lineage'] == lineage, 'outer lineage mismatch')
        require(outer.get('outer_fold') == oi, 'outer completion index mismatch')
        expected = {}
        inner_scores = {float(c): [] for c in cfg['model']['C_candidates']}
        for ii, ih in enumerate(donors[donors != held]):
            tr = np.flatnonzero((data['donor'] != held) & (data['donor'] != ih))
            ev = np.flatnonzero(data['donor'] == ih)
            for c in inner_scores:
                name = f'inner_{ii:02d}_C_{c:g}'.replace('.', 'p')
                expected[name] = (tr, ev)
                receipt, prob, met, se, pe = replay_cell(directory/name, data, tr, ev, lineage, optimizer_limit)
                require(receipt['C'] == c, 'candidate identity mismatch')
                inner_scores[c].append(met['balanced_accuracy'])
                fit_count += 1
                scaler_max, pred_max = max(scaler_max, se), max(pred_max, pe)
        mean_scores = {c: np.mean(v) for c, v in inner_scores.items()}
        selected = min(c for c in mean_scores if mean_scores[c] >= max(mean_scores.values())-cfg['selection']['tie_tolerance'])
        require(selected == outer['selected_C'], 'inner-only model selection mismatch')
        validate_public_fold(public['outer_folds'][oi], oi, selected)
        for c, v in inner_scores.items():
            require(np.allclose(v, outer['inner_scores'][str(c)], atol=1e-10, rtol=0), 'saved inner scores mismatch')
        tr, ev = np.flatnonzero(data['donor'] != held), np.flatnonzero(data['donor'] == held)
        name = f'final_C_{selected:g}'.replace('.', 'p')
        expected[name] = (tr, ev)
        require(outer['final_cell'] == name and set(outer['cells']) == set(expected), 'outer cell inventory mismatch')
        require({p.parent.name for p in directory.glob('*/COMPLETE.json')} == set(expected), 'unexpected completed fits')
        receipt, prob, met, se, pe = replay_cell(directory/name, data, tr, ev, lineage, optimizer_limit)
        require(receipt['C'] == selected, 'final C mismatch')
        numeric_equal(met, outer['metrics'])
        fit_count += 1
        scaler_max, pred_max = max(scaler_max, se), max(pred_max, pe)
        w = weights(data['donor'][tr], data['section'][tr])
        prior = np.bincount(data['y'][tr], weights=w, minlength=3)
        require(np.allclose(prior, outer['prior'], atol=1e-12, rtol=0), 'train-only prior mismatch')
        row = dict(metrics=met)
        for arm, pp in [('prior_baseline', prior), ('majority_baseline', np.eye(3)[prior.argmax()])]:
            bm = metrics(data['y'][ev], np.tile(pp, (len(ev), 1)), data['donor'][ev], data['section'][ev])
            numeric_equal(bm, outer[arm])
            row[arm] = bm
        for arm in row:
            numeric_equal(row[arm], public['outer_folds'][oi][arm])
        rows.append(row)
        oof[ev] = prob
        assigned[ev] += 1
    require(fit_count == 96 and np.all(assigned == 1), 'incomplete formal coverage')
    complete = read(args.run_root/'COMPLETE.json')
    require(complete['lineage'] == lineage and complete['public_results_sha256'] == sha(args.public_results), 'completion report mismatch')
    require(complete['oof_predictions_sha256'] == sha(args.run_root/'oof_predictions.npz'), 'OOF cache hash mismatch')
    with np.load(args.run_root/'oof_predictions.npz', allow_pickle=False) as f:
        require(np.array_equal(f['row_index'], np.arange(len(oof))), 'OOF coverage order mismatch')
        for k in ['y', 'donor', 'section', 'barcodes']:
            require(np.array_equal(f[k], data[k]), 'OOF metadata mismatch')
        oof_error = float(np.max(abs(f['probabilities']-oof)))
        require(oof_error <= 1e-10, 'OOF independent replay failed')
    for arm in ['metrics', 'prior_baseline', 'majority_baseline']:
        for key, reported in public['donor_equal_aggregate'][arm].items():
            values = np.asarray([r[arm][key] for r in rows])
            expected = values.sum(0) if key == 'confusion_counts' else values.mean(0)
            require(np.allclose(expected, reported, atol=1e-10, rtol=0), 'donor-equal aggregation mismatch')
    require(before == {p: sha(p) for p in before}, 'input or official result changed during replay')
    evcfg = cfg['evaluation']
    require(evcfg['bootstrap_repeats'] == 20000 and evcfg['bootstrap_seed'] == 7031001, 'unexpected bootstrap contract')
    draws = np.random.default_rng(evcfg['bootstrap_seed']).integers(0, 6, size=(evcfg['bootstrap_repeats'], 6))
    intervals = {}
    for key in ['balanced_accuracy', 'accuracy', 'negative_log_likelihood', 'brier_multiclass_sum', 'ece_10_equal_width_bins']:
        per_arm = {arm: np.array([r[arm][key] for r in rows]) for arm in ['metrics', 'prior_baseline', 'majority_baseline']}
        quantities = dict(per_arm)
        quantities.update({f'model_minus_{arm}': per_arm['metrics']-per_arm[arm] for arm in ['prior_baseline', 'majority_baseline']})
        intervals[key] = {arm: dict(point=float(v.mean()), percentile_95=np.quantile(v[draws].mean(1), [.025, .975]).tolist())
                          for arm, v in quantities.items()}
    result = dict(schema_version='sami_stage_a_independent_audit_v1', status='PASS', completed_cells_verified=96,
                  nested_outer_donors=6, independently_replayed_predictions=96, OOF_rows=13011,
                  max_scaler_absolute_error=scaler_max, max_prediction_absolute_error=pred_max,
                  OOF_max_absolute_error=oof_error, source_lineage=lineage,
                  no_optimization_performed=True, DSS_accessed=False,
                  locked_optimizer_gradient_threshold_verified=optimizer_limit,
                  ECE_rule='within_each_donor_locked_numpy_linspace_edges_then_equal_donor_mean',
                  every_inner_and_outer_donor_separation_verified=True, original_results_unchanged=True,
                  descriptive_donor_bootstrap=dict(unit='held_out_donor', n_units=6, repeats=20000, seed=7031001,
                      scope='fixed_OOF_conditional_not_pipeline_refit_not_external_confirmation', estimates=intervals),
                  original_results_sha256=sha(args.public_results), audit_script_sha256=sha(__file__))
    args.public_report.parent.mkdir(parents=True, exist_ok=True)
    require(not args.public_report.exists(), 'refusing to overwrite independent audit')
    args.public_report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['prepared', 'prepared-manifest', 'config', 'run-root', 'public-results', 'public-report']:
        p.add_argument('--'+name, type=Path, required=True)
    audit(p.parse_args())
