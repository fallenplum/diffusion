"""Locked 96-fit frozen-UNI2 tissue qualification. All predictions stay private."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import scipy

from sami_surv.morphology import (classification_metrics, fit_model, outer_inner_splits,
                                  select_c, train_prior)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(text, encoding='utf-8')
    os.replace(temp, path)


def source_identity():
    sources = ['src/sami_surv/morphology.py', 'scripts/run_sami_stage_a.py']
    hashes = {s: sha256(ROOT/s) for s in sources}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return digest, hashes


def load_locked_inputs(prepared, prepared_manifest, config_path):
    config, manifest = json_read(config_path), json_read(prepared_manifest)
    if config['status'] != 'LOCKED_BEFORE_MODEL_FIT':
        raise ValueError('Protocol must be locked before any model fit')
    if config['schema_version'] != 'sami_surv_stage_a_protocol_v1':
        raise ValueError('Unsupported protocol schema')
    if (manifest['schema_version'] != 'sami_stage_a_prepared_v1'
            or manifest['status'] != 'PASS_PREPARED_FROZEN_TISSUE3'
            or manifest['data_sha256'] != sha256(prepared)
            or manifest['protocol_sha256'] != sha256(config_path)
            or manifest['classes'] != config['label_contract']['class_order']):
        raise ValueError('Prepared qualification or locked artifact identity mismatch')
    if (manifest['mapping_sha256'] != config['label_contract']['mapping_sha256']
            or manifest['DSS_read'] is not False
            or manifest['expression_array_accessed'] is not False
            or manifest['original_ST_test_arrays_read'] is not False):
        raise ValueError('Label mapping or forbidden-outcome/source access contract mismatch')
    if manifest['original_split_lock_sha256'] != config['cohort']['split_lock_sha256']:
        raise ValueError('Original development split lock differs from frozen protocol')
    model, selection = config['model'], config['selection']
    if (config['label_contract']['class_order'] != ['tumor', 'stroma', 'tumor_stroma_mixed']
            or model['feature_dimension'] != 1536 or model['C_candidates'] != [.01, .1, 1.]
            or model['solver'] != 'scipy_L-BFGS-B' or model['penalty'] != 'L2_weights_only_intercept_unpenalized'
            or selection['expected_fit_units'] != 96 or selection['tie_break'] != 'smallest_C_strongest_regularization'):
        raise ValueError('Unexpected locked model contract')
    with np.load(prepared, allow_pickle=False) as archive:
        required = {'X', 'y', 'donor', 'section', 'barcodes', 'coords_um'}
        if set(archive.files) != required:
            raise ValueError('Prepared schema mismatch; no outcome/additional arrays accepted')
        data = {k: archive[k] for k in archive.files}
    x, y = data['X'], data['y']
    n = len(x)
    if x.dtype != np.float32 or x.shape != (n, 1536) or not np.isfinite(x).all():
        raise ValueError('Frozen UNI2 matrix must be finite float32 [n,1536]')
    if y.dtype.kind not in 'iu' or y.shape != (n,) or not np.array_equal(np.unique(y), np.arange(3)):
        raise ValueError('Invalid tissue labels')
    if data['coords_um'].shape != (n, 2) or not np.isfinite(data['coords_um']).all():
        raise ValueError('Invalid aligned coordinates')
    for field in ['donor', 'section', 'barcodes']:
        if data[field].shape != (n,) or data[field].dtype.kind != 'U':
            raise ValueError('Identity columns must be aligned Unicode vectors')
    if len(np.unique(data['donor'])) != 6 or len(np.unique(data['section'])) != 12:
        raise ValueError('Locked six-donor/twelve-section cohort mismatch')
    if n != 13011:
        raise ValueError('Locked primary cohort must contain exactly 13011 observations')
    if len(set(zip(data['section'].tolist(), data['barcodes'].tolist()))) != n:
        raise ValueError('Duplicate section/barcode observation')
    for section in np.unique(data['section']):
        if len(np.unique(data['donor'][data['section'] == section])) != 1:
            raise ValueError('Section assigned to multiple donors')
    for donor in np.unique(data['donor']):
        if not np.array_equal(np.unique(y[data['donor'] == donor]), np.arange(3)):
            raise ValueError('Every locked donor must cover all three classes')
    if sorted(fold['outer_index'] for fold in manifest['outer_donor_folds']) != list(range(6)):
        raise ValueError('Prepared donor split manifest must contain exactly six outer folds')
    for fold in manifest['outer_donor_folds']:
        donor_order = np.unique(data['donor']).tolist()
        if (fold['heldout_donor'] != donor_order[fold['outer_index']]
                or sorted(fold['train_donors']) != sorted(d for d in donor_order if d != fold['heldout_donor'])):
            raise ValueError('Prepared donor partition differs from label-independent locked order')
    digest, sources = source_identity()
    lineage = dict(prepared_npz_sha256=sha256(prepared), prepared_manifest_sha256=sha256(prepared_manifest),
                   config_sha256=sha256(config_path), source_sha256=digest)
    return data, config, manifest, lineage, sources


def initialize_run(output, data, config, lineage, sources):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    splits = list(outer_inner_splits(data['donor']))
    run = dict(schema_version='sami_stage_a_run_v1', lineage=lineage, source_files=sources,
               numpy_version=np.__version__, scipy_version=scipy.__version__,
               n_spots=len(data['y']), feature_dimension=1536, n_classes=3,
               classes=config['label_contract']['class_order'], expected_fit_units=96,
               split_order='lexicographic donor identity; does not depend on labels',
               split_manifest=[dict(outer_fold=oi, train_index=tr.tolist(), test_index=te.tolist(),
                                    inner=[dict(train_index=it.tolist(), validation_index=iv.tolist()) for it,iv in inners])
                               for oi,tr,te,inners in splits])
    path = output/'run_manifest.json'
    try:
        with path.open('x', encoding='utf-8') as f:
            json.dump(run, f, indent=2, sort_keys=True, allow_nan=False)
    except FileExistsError:
        old = json_read(path)
        if old != run:
            raise ValueError('Existing run manifest does not match exact data/config/source/runtime')
    return splits


def cell_name(inner_index, C):
    prefix = 'final' if inner_index is None else f'inner_{inner_index:02d}'
    return f'{prefix}_C_{C:g}'.replace('.', 'p')


def fitted_cell(cell, data, train_index, evaluation_index, C, model_cfg, lineage):
    cell = Path(cell)
    cell.mkdir(parents=True, exist_ok=True)
    expected = dict(lineage=lineage, train_index=train_index.tolist(), evaluation_index=evaluation_index.tolist(), C=float(C))
    done = cell/'COMPLETE.json'
    if done.exists():
        result = json_read(done)
        if result['status'] != 'COMPLETE' or any(result[k] != v for k,v in expected.items()):
            raise ValueError('Completed cell lineage or split mismatch')
        for name in ['model', 'predictions']:
            if sha256(cell/f'{name}.npz') != result[f'{name}_sha256']:
                raise ValueError('Completed cell artifact hash mismatch')
        return result
    if (cell/'START.json').exists() or (cell/'FAILED.json').exists():
        raise RuntimeError('Incomplete/failed cell requires explicit investigation; no automatic refit')
    if set(data['donor'][train_index]) & set(data['donor'][evaluation_index]):
        raise ValueError('Donor leakage in cell')
    json_write(cell/'START.json', expected)
    start = time.monotonic()
    try:
        fit = fit_model(data['X'][train_index], data['y'][train_index], data['donor'][train_index],
                        data['section'][train_index], n_classes=3, C=C,
                        maxiter=model_cfg['maxiter'], ftol=model_cfg['ftol'], gtol=model_cfg['gtol'],
                        maxls=model_cfg['maxls'], accepted_gradient_inf=model_cfg['accepted_gradient_inf'])
        probability = fit.predict_proba(data['X'][evaluation_index])
        # First performance use of these labels; earlier accesses only qualify
        # the locked schema and class coverage, never choose or fit the model.
        metrics = classification_metrics(data['y'][evaluation_index], probability,
                                         data['donor'][evaluation_index], data['section'][evaluation_index], 3)
        np.savez_compressed(cell/'model.npz', **fit.arrays())
        np.savez_compressed(cell/'predictions.npz', row_index=evaluation_index, probabilities=probability)
        result = dict(**expected, status='COMPLETE', fit_audit=fit.fit_audit, metrics=metrics,
                      elapsed_seconds=time.monotonic()-start, model_sha256=sha256(cell/'model.npz'),
                      predictions_sha256=sha256(cell/'predictions.npz'))
        json_write(done, result)
        return result
    except Exception as exc:
        json_write(cell/'FAILED.json', dict(**expected, status='FAILED', exception_type=type(exc).__name__, message=str(exc)))
        raise


def run_outer(output, data, config, lineage, split):
    oi, outer_train, outer_test, inners = split
    directory = Path(output)/'cells'/f'outer_{oi:02d}'
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory/'RUNNING.lock'
    try:
        lock_fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Outer fold already running or requires stale-lock investigation')
    try:
        os.write(lock_fd, str(os.getpid()).encode())
        os.close(lock_fd)
        if (directory/'COMPLETE.json').exists():
            completed = json_read(directory/'COMPLETE.json')
            if completed['lineage'] != lineage or completed['status'] != 'COMPLETE':
                raise ValueError('Completed outer lineage mismatch')
            for name in completed['cells']:
                cell = directory/name
                receipt = json_read(cell/'COMPLETE.json')
                if receipt['lineage'] != lineage or receipt['status'] != 'COMPLETE':
                    raise ValueError('Completed outer has incomplete cells')
                for artifact in ['model', 'predictions']:
                    if sha256(cell/f'{artifact}.npz') != receipt[f'{artifact}_sha256']:
                        raise ValueError('Completed outer artifact changed')
            return completed
        candidates = config['model']['C_candidates']
        inner_scores = {float(c): [] for c in candidates}
        cells = []
        for ii, (train, valid) in enumerate(inners):
            for C in candidates:
                name = cell_name(ii, C)
                result = fitted_cell(directory/name, data, train, valid, C, config['model'], lineage)
                inner_scores[float(C)].append(result['metrics']['balanced_accuracy'])
                cells.append(name)
                print(json.dumps(dict(outer_fold=oi, completed_cell=name, status='COMPLETE')), flush=True)
        selected, mean_scores = select_c(inner_scores, candidates, config['selection']['tie_tolerance'])
        final_name = cell_name(None, selected)
        final = fitted_cell(directory/final_name, data, outer_train, outer_test, selected, config['model'], lineage)
        cells.append(final_name)
        prior = train_prior(data['y'][outer_train], data['donor'][outer_train], data['section'][outer_train], 3)
        prior_probs = np.tile(prior, (len(outer_test), 1))
        majority_probs = np.tile(np.eye(3)[int(np.argmax(prior))], (len(outer_test), 1))
        def baseline_metrics(prob):
            return classification_metrics(data['y'][outer_test], prob, data['donor'][outer_test], data['section'][outer_test], 3)
        result = dict(status='COMPLETE', lineage=lineage, outer_fold=oi, selected_C=selected,
                      inner_scores={str(c): s for c,s in inner_scores.items()},
                      inner_mean_scores={str(c): s for c,s in mean_scores.items()},
                      cells=cells, final_cell=final_name, metrics=final['metrics'],
                      prior=prior.tolist(), prior_baseline=baseline_metrics(prior_probs),
                      majority_baseline=baseline_metrics(majority_probs))
        complete = directory/'COMPLETE.json'
        if complete.exists() and json_read(complete) != result:
            raise ValueError('Completed outer result changed')
        if not complete.exists():
            json_write(complete, result)
        return result
    finally:
        lock.unlink(missing_ok=True)


def finalize(output, public_output, data, config, lineage, splits):
    output, public_output = Path(output), Path(public_output)
    if (output/'COMPLETE.json').exists():
        complete = json_read(output/'COMPLETE.json')
        if complete['lineage'] != lineage:
            raise ValueError('Final lineage changed')
        if sha256(output/'oof_predictions.npz') != complete['oof_predictions_sha256']:
            raise ValueError('Final OOF cache changed')
        if sha256(public_output/'RESULTS.json') != complete['public_results_sha256']:
            raise ValueError('Public results changed')
        return complete
    outer_results = []
    oof = np.full((len(data['y']), 3), np.nan)
    assigned = np.zeros(len(data['y']), np.int64)
    counts = 0
    for oi, train, test, inners in splits:
        directory = output/'cells'/f'outer_{oi:02d}'
        result = json_read(directory/'COMPLETE.json')
        if result['status'] != 'COMPLETE' or result['lineage'] != lineage or len(result['cells']) != 16:
            raise ValueError('Incomplete outer fold or incompatible lineage')
        for name in result['cells']:
            receipt = json_read(directory/name/'COMPLETE.json')
            if receipt['lineage'] != lineage or receipt['status'] != 'COMPLETE':
                raise ValueError('Incomplete cell')
            for artifact in ['model', 'predictions']:
                if sha256(directory/name/f'{artifact}.npz') != receipt[f'{artifact}_sha256']:
                    raise ValueError('Cell artifact hash mismatch')
            counts += 1
        with np.load(directory/result['final_cell']/'predictions.npz', allow_pickle=False) as a:
            if not np.array_equal(a['row_index'], test):
                raise ValueError('OOF index alignment failed')
            oof[test] = a['probabilities']
        assigned[test] += 1
        outer_results.append(result)
    if counts != 96 or not np.all(assigned == 1) or not np.isfinite(oof).all():
        raise ValueError('Final complete coverage/fit-count guard failed')
    np.savez_compressed(output/'oof_predictions.npz', row_index=np.arange(len(oof)), y=data['y'],
                        probabilities=oof, prediction=oof.argmax(1), donor=data['donor'],
                        section=data['section'], barcodes=data['barcodes'])
    keys = ['balanced_accuracy', 'accuracy', 'negative_log_likelihood', 'brier_multiclass_sum', 'ece_10_equal_width_bins']
    aggregate = {}
    for arm in ['metrics', 'prior_baseline', 'majority_baseline']:
        aggregate[arm] = {key: float(np.mean([r[arm][key] for r in outer_results])) for key in keys}
        aggregate[arm]['class_recall'] = np.mean([r[arm]['class_recall'] for r in outer_results], axis=0).tolist()
        aggregate[arm]['confusion_counts'] = np.sum([r[arm]['confusion_counts'] for r in outer_results], axis=0).tolist()
    public = dict(schema_version='sami_stage_a_results_v1', status='COMPLETE_NOT_YET_INDEPENDENTLY_AUDITED',
                  lineage=lineage, n_donors=6, n_sections=12, n_spots=len(oof), completed_fits=96,
                  class_order=config['label_contract']['class_order'], donor_equal_aggregate=aggregate,
                  outer_folds=[{k: r[k] for k in ['outer_fold','selected_C','metrics','prior_baseline','majority_baseline']} for r in outer_results],
                  interpretation='External tissue qualification only; no DSS access; uncalibrated balanced-fit probabilities; no automatic downstream admission.',
                  uncertainty='Six held-out donors; no spot-level confidence intervals or independent cross-study confirmation claimed.',
                  metric_definition=dict(balanced_accuracy='Within each donor: class-equal, then within class section-equal recall; donor-equal final mean.',
                                         NLL_Brier_ECE='Within each donor: section-equal, spot-equal; final donor-equal mean.',
                                         brier='Multiclass sum over classes, not divided by number of classes.',
                                         nll_floor=1e-15, ECE='10 fixed equal-width confidence bins; no calibration fitted.'))
    public_output.mkdir(parents=True, exist_ok=True)
    if (public_output/'RESULTS.json').exists():
        raise ValueError('Refusing to overwrite preexisting public results')
    json_write(public_output/'RESULTS.json', public)
    complete = dict(status='COMPLETE', lineage=lineage, completed_fits=96,
                    oof_predictions_sha256=sha256(output/'oof_predictions.npz'),
                    public_results_sha256=sha256(public_output/'RESULTS.json'))
    json_write(output/'COMPLETE.json', complete)
    return complete


def validate_output_paths(output, public_output, prepared, prepared_manifest):
    private, public = Path(output).resolve(), Path(public_output).resolve()
    if private.is_relative_to(ROOT) or 'raw' in [part.lower() for part in private.parts]:
        raise ValueError('Private model and OOF artifacts must be outside repository and raw directories')
    if not public.is_relative_to(ROOT/'reports'):
        raise ValueError('Public output must be a dedicated repository reports directory')
    if private.is_relative_to(public) or public.is_relative_to(private):
        raise ValueError('Private/public artifact roots must not overlap')
    for source in [prepared, prepared_manifest]:
        if Path(source).resolve().is_relative_to(private):
            raise ValueError('Run output must not contain or overwrite prepared source artifacts')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--prepared-manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='PRIVATE artifact root')
    parser.add_argument('--public-output', type=Path, required=True)
    parser.add_argument('--mode', choices=['plan','run','finalize'], default='plan')
    parser.add_argument('--outer-fold', type=int, choices=range(6))
    args = parser.parse_args()
    validate_output_paths(args.output, args.public_output, args.prepared, args.prepared_manifest)
    data, config, manifest, lineage, sources = load_locked_inputs(args.prepared, args.prepared_manifest, args.config)
    splits = initialize_run(args.output, data, config, lineage, sources)
    if args.mode == 'plan':
        print(json.dumps(dict(status='READY', expected_fit_units=96, lineage=lineage)), flush=True)
    elif args.mode == 'run':
        for split in splits:
            if args.outer_fold is None or split[0] == args.outer_fold:
                run_outer(args.output, data, config, lineage, split)
        print(json.dumps(dict(status='REQUESTED_OUTER_FOLDS_COMPLETE')), flush=True)
    else:
        print(json.dumps(finalize(args.output, args.public_output, data, config, lineage, splits)), flush=True)


if __name__ == '__main__':
    main()
