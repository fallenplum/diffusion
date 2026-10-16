"""Independent Phase19 source/checkpoint/OOF and paired-patient evaluation.

No Phase17/18 or interaction-pilot result is overwritten. The old 125 pilot
fits are not collected here: their predictions are overlapping inner-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_phase18a import (
    comparable_pairs, components, reverse_km_ipcw, resampled_c_indices,
    linear_c_contrast, metrics, assert_equal_structure,
)

PRIMARY_COMPARISONS = [("A3", arm) for arm in ["A0", "A1", "A2", "A4", "A5", "A6"]]
PRIMARY_FAMILY_TESTS = 12
PAPER_ADAPTATIONS = ["MGCT", "CAMR_REPAIRED", "MMP"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    def cast(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=cast, allow_nan=False) + "\n")
    temporary.replace(path)


def check_model_seed_coverage(risks, seeds):
    labels = sorted({arm for arm, _ in risks})
    for arm in labels:
        actual = sorted(seed for label, seed in risks if label == arm)
        if actual != sorted(seeds):
            raise ValueError(("model missing/repeating algorithm seeds", arm, actual))
        lengths = {np.asarray(risks[arm, seed]).shape for seed in seeds}
        if len(lengths) != 1 or not all(np.isfinite(risks[arm, seed]).all() for seed in seeds):
            raise ValueError(("invalid OOF risk vector", arm))
    return labels


def point_tensor(seed_rows, keys):
    lookup = {}
    for row in seed_rows:
        key = row["arm"], int(row["seed"])
        if key in lookup:
            raise ValueError("repeated seed C row")
        lookup[key] = [row["Harrell"], row["Uno60"]]
    if set(lookup) != set(keys):
        raise ValueError("point C rows do not describe the same models as bootstrap")
    return np.asarray([[lookup[key] for key in keys]], float)


def comparison_definitions(risks, seeds):
    labels = check_model_seed_coverage(risks, seeds)
    required = {arm for pair in PRIMARY_COMPARISONS for arm in pair}
    if not required <= set(labels):
        raise ValueError("primary architecture matrix incomplete")
    secondary = [("A3", arm) for arm in ["J0", "J_CS", "HIST_CS_S32_CONCH32", *PAPER_ADAPTATIONS]
                 if arm in labels]
    secondary += [("A3", "A3__" + tag) for tag in ["WRONG_C", "WRONG_O", "CONSTANT_C", "INTERACTION_OFF"]
                  if "A3__" + tag in labels]
    return [(a, b, "primary") for a, b in PRIMARY_COMPARISONS] + [(a, b, "secondary") for a, b in secondary]


def bootstrap(d, risks, endpoint_data, seed_rows, config, report_root):
    """Six fixed contrasts, H/Uno family12; same patient counts every model/seed."""
    import pandas as pd
    time, event, valid, ipcw = endpoint_data
    bc = config["bootstrap"]
    if bc["draws"] != 20000 or bc.get("family_tests", PRIMARY_FAMILY_TESTS) != PRIMARY_FAMILY_TESTS:
        raise ValueError("Phase19 primary multiplicity/bootstrap differs from locked protocol")
    seeds = config["seeds"]
    definitions = comparison_definitions(risks, seeds)
    count_audit = []
    keys, values, den, good = resampled_c_indices(time, event, valid, d["folds"], risks, ipcw,
        draws=bc["draws"], seed=bc["seed"], count_audit=count_audit)
    if int(good.sum()) < .99 * bc["draws"]:
        raise ValueError("too many no-comparable-pair resamples")
    points = point_tensor(seed_rows, keys)
    index = {key: i for i, key in enumerate(keys)}
    rows = []; seed_contrasts = []; audit = []
    secondary_tests = max(1, 2 * sum(family == "secondary" for _, _, family in definitions))
    fold_rows = pd.read_csv(Path(report_root) / "FOLD_METRICS.csv")
    for a, b, family in definitions:
        coefficients = {a: 1., b: -1.}
        contrast = linear_c_contrast(keys, values, coefficients, seeds)
        point = linear_c_contrast(keys, points, coefficients, seeds)[0]
        tests = PRIMARY_FAMILY_TESTS if family == "primary" else secondary_tests
        alpha = .05 / tests
        for m, metric in enumerate(["Harrell", "Uno60"]):
            per_seed = []
            for seed in seeds:
                delta = float(points[0, index[a, seed], m] - points[0, index[b, seed], m])
                per_seed.append(delta)
                seed_contrasts.append({"comparison": a + "-" + b, "metric": metric,
                                       "family": family, "seed": seed, "delta": delta})
            expected = float(np.mean(per_seed))
            if abs(expected - point[m]) > 1e-12:
                raise ValueError("point estimate is not same paired seed-C statistic as bootstrap")
            left = fold_rows[fold_rows.arm == a].set_index(["seed", "fold"])[metric]
            right = fold_rows[fold_rows.arm == b].set_index(["seed", "fold"])[metric]
            if not left.index.equals(right.index):
                left = left.sort_index(); right = right.sort_index()
            if not left.index.equals(right.index):
                raise ValueError("paired fold cells do not identify same seeds and folds")
            fold_delta = left - right
            lo, hi = np.quantile(contrast[:, m], [.025, .975])
            cl, ch = np.quantile(contrast[:, m], [alpha / 2, 1 - alpha / 2])
            rows.append({"comparison": a + "-" + b, "metric": metric, "family": family,
                "delta": point[m], "CI95_low": lo, "CI95_high": hi,
                "Bonferroni_low": cl, "Bonferroni_high": ch, "family_tests": tests,
                "positive_seeds": int(np.sum(np.asarray(per_seed) > 0)), "seeds": len(seeds),
                "positive_fold_cells": int((fold_delta > 0).sum()), "fold_cells": len(fold_delta),
                "draws_requested": bc["draws"], "draws_valid": int(good.sum()),
                "statistical_object": "mean algorithm-seed C after per-seed within-fold numerator/denominator sums; no risk ensemble"})
            audit.append({"comparison": a + "-" + b, "metric": metric, "point_from_complete_seed_C": expected,
                          "reported_point": point[m], "same_statistic_in_each_bootstrap_draw": True})
    pd.DataFrame(rows).to_csv(Path(report_root) / "PAIRED_COMPARISONS.csv", index=False)
    pd.DataFrame(seed_contrasts).to_csv(Path(report_root) / "SEED_PAIRED_CONTRASTS.csv", index=False)
    training_source = _api()[0].source()
    write(Path(report_root) / "CONTRAST_POINT_AUDIT.json", {"status": "PASS", "source": training_source,
        "primary_family_tests": 12, "rows": audit})
    write(Path(report_root) / "BOOTSTRAP_AUDIT.json", {"status": "PASS", "source": training_source, "draws": bc["draws"],
        "valid": int(good.sum()), "zero_denominator_draws": int((~good).sum()),
        "primary_family_tests": PRIMARY_FAMILY_TESTS,
        "shared_patient_count_matrices": count_audit,
        "minimum_H_denominator": float(den[:, 0].min()), "minimum_U_denominator": float(den[:, 1].min()),
        "sampling": "within fixed outer folds patient multinomial; one count vector shared all models/seeds",
        "pair_weight": "count_i*count_j; no independent pair/seed resampling", "IPCW": "frozen outer-train reverse KM",
        "risk_ensemble": False, "uncertainty_scope": "fixed trained models; excludes researcher selection and full model-refitting uncertainty",
        "historical_pilot_125_fits_used_as_new_OOF": False})
    return rows


def flatten_diagnostics(rows):
    """Publish aggregate scalars while retaining raw target values only privately."""
    denied = {"risk", "anchor_risk", "patient_risk", "risk_raw", "target_value", "prediction", "prediction_raw"}
    out = []
    for row in rows:
        scalar = {key: value for key, value in row.items()
                  if key not in denied and not isinstance(value, (list, tuple, dict, np.ndarray))}
        probes = row.get("position_gradient_probes", [])
        if probes:
            out.extend({**scalar, **{"position_probe_" + key: value for key, value in probe.items() if key not in denied}}
                       for probe in probes)
        else:
            out.append(scalar)
    return out


def update_probe_coverage(coverage, ck, visual_owner):
    """Metadata only: count distinct query patients/WSIs, never multiply by seeds."""
    available = ck['arm'].startswith('A')
    item = coverage.setdefault((ck['arm'], ck['role']), {
        'checkpoint_count': 0, 'risk_patients': set(), 'risk_WSIs': set(),
        'gradient_patients': set(), 'gradient_WSIs': set(),
        'risk_patient_counts': [], 'risk_WSI_counts': [],
        'gradient_patient_counts': [], 'gradient_WSI_counts': [],
        'isolated_input_gradient_available': available})
    item['checkpoint_count'] += 1
    for prefix, patients in [('risk', np.asarray(ck['fit'][:32])),
                             ('gradient', np.asarray(ck['fit'][:1]) if available else np.asarray([], int))]:
        slides = np.flatnonzero(np.isin(visual_owner, patients))
        item[prefix + '_patients'].update(patients.tolist())
        item[prefix + '_WSIs'].update(slides.tolist())
        item[prefix + '_patient_counts'].append(len(patients))
        item[prefix + '_WSI_counts'].append(len(slides))


def summarize_probe_coverage(coverage):
    rows = []
    for (arm, role), item in sorted(coverage.items()):
        row = {'arm': arm, 'role': role, 'checkpoint_count': item['checkpoint_count'],
               'isolated_input_gradient_available': item['isolated_input_gradient_available']}
        for prefix, label in [('risk', 'risk_attention_probe'), ('gradient', 'isolated_input_gradient_probe')]:
            row[label + '_unique_query_patients'] = len(item[prefix + '_patients'])
            row[label + '_unique_query_WSIs'] = len(item[prefix + '_WSIs'])
            for target, source_name in [('patients', 'patient'), ('WSIs', 'WSI')]:
                counts = item[prefix + '_' + source_name + '_counts']
                row[label + '_per_checkpoint_' + target + '_min'] = min(counts)
                row[label + '_per_checkpoint_' + target + '_max'] = max(counts)
        rows.append(row)
    return rows


def _api():
    from src.phase19_attention import data, train
    return data, train


def independent_feature_audit(ds):
    """Reconstruct Clinical/S32, CNA fixed pools, X layout and source visual PCA."""
    from scripts.validate_phase17_independent import independent_cs
    from src.phase19_attention.data import VISUAL
    c, s = independent_cs(ds.d, ds.fit)
    if not np.array_equal(c[ds.active], ds.f['C'][ds.active]) or not np.array_equal(s[ds.active], ds.f['S'][ds.active]):
        raise ValueError('independent Clinical/S32 reconstruction differs')
    if not np.array_equal(np.c_[ds.f['C'], ds.f['S']], ds.f['CS']):
        raise ValueError('J_CS design is not Clinical+S32')
    pool_error = 0.
    for output, input_name in [('GN8', 'G'), ('LN8', 'L')]:
        x = ds.f[input_name].astype(float); mask = ds.f['CM'].astype(bool)
        member = ds.f['membership'].astype(bool)
        raw = np.zeros((512, len(member)), float)
        for group, genes in enumerate(member):
            measured = mask[:, genes]
            numerator = np.where(measured, x[:, genes], 0.).sum(1)
            raw[:, group] = numerator / np.maximum(measured.sum(1), 1)
        raw = raw.astype(np.float32).astype(float)
        mean = raw[ds.fit].mean(0); sd = raw[ds.fit].std(0); sd[sd < 1e-8] = 1.
        actual = ((raw[ds.active] - mean) / sd).astype(np.float32)
        error = float(abs(actual - ds.f[output][ds.active]).max())
        pool_error = max(pool_error, error)
        if error > 2e-6:
            raise ValueError('independent measured CNA program pooling/scaling differs')
    coverage_error = 0.
    for output, mask_name in [('RCOV8', 'RM'), ('CCOV8', 'CM')]:
        if output not in ds.f:
            continue
        member = ds.f['membership'].astype(bool); measured = ds.f[mask_name].astype(bool)
        raw = np.stack([measured[:, genes].mean(1) for genes in member], axis=1).astype(np.float32).astype(float)
        mean = raw[ds.fit].mean(0); sd = raw[ds.fit].std(0); sd[sd < 1e-8] = 1.
        actual = ((raw[ds.active]-mean)/sd).astype(np.float32)
        error = float(abs(actual-ds.f[output][ds.active]).max()); coverage_error=max(coverage_error,error)
        if error > 2e-6:
            raise ValueError('independent observed-coverage main effect differs')
    owner = np.load(VISUAL / 'owner.npy'); mask = np.load(VISUAL / 'mask.npy')
    visual = np.load(VISUAL / 'u.npy', mmap_mode='r')
    means = np.zeros((512, 1536), float)
    for patient in ds.active:
        slides = np.flatnonzero(owner == patient)
        means[patient] = np.stack([np.asarray(visual[j], np.float32)[mask[j]].mean(0, dtype=np.float64) for j in slides]).mean(0)
    state = ds.state['UNI32']
    mean = means[ds.fit].mean(0); sd = means[ds.fit].std(0); sd[sd < 1e-8] = 1.
    if not np.allclose(mean, state['before_scale']['mean'], atol=1e-12, rtol=0) or not np.allclose(sd, state['before_scale']['sd'], atol=1e-12, rtol=0):
        raise ValueError('UNI patient means or fit-only scaling differs')
    before = ((means[ds.active] - mean) / sd).astype(np.float32)
    pca = state['pca']
    fit_before = ((means[ds.fit] - mean) / sd).astype(np.float32)
    if not np.allclose(pca.mean_, fit_before.mean(0), atol=1e-6, rtol=1e-6):
        raise ValueError('UNI PCA centroid is not its claimed training partition')
    pc = (before - pca.mean_) @ pca.components_.T
    pc = pc.astype(float)
    after = ((pc - state['after_scale']['mean']) / state['after_scale']['sd']).astype(np.float32)
    pca_error = float(abs(after - ds.f['UNI32'][ds.active]).max())
    if pca_error > 5e-5:
        raise ValueError(('independent UNI PCA replay differs', pca_error))
    keys = ds.audit['X_order']
    if not np.array_equal(np.concatenate([ds.f[key] for key in keys], axis=1), ds.f['X']):
        raise ValueError('joint main-effect design layout differs')
    if int(np.bincount(owner, weights=mask.sum(1), minlength=512).max()) > 512:
        raise ValueError('patient patch budget exceeded')
    if not np.array_equal(np.flatnonzero(np.isin(owner, ds.active)), ds.global_wsi_indices):
        raise ValueError('model WSI ownership differs from audited visual source')
    return {'status': 'PASS', 'fold': ds.fold, 'role': ds.role, 'CNA_pool_max_error': pool_error,
            'coverage_main_effect_max_error': coverage_error,
            'UNI32_matrix_replay_max_error': pca_error, 'Clinical_S32_elementwise_identical': True,
            'UNI_patient_pooling': 'valid patch mean per WSI then equal WSI mean per patient',
            'feature_cache_sha256': ds.audit['cache_sha256']}


def history_audit(ck, config):
    tc = config['training']; history = ck['history']; selected = int(ck['selected_epoch'])
    if not history or [q['epoch'] for q in history] != list(range(1, len(history) + 1)):
        raise ValueError('missing/noncontiguous training epochs')
    if not all(np.isfinite(q['train_loss']) and np.isfinite(q['penalty']) for q in history):
        raise ValueError('nonfinite training history')
    if ck['role'] == 'outer':
        if selected != len(history):
            raise ValueError('outer fresh refit length differs from selected epoch')
        return
    best, epoch, ema = float('inf'), 0, None
    for row in history:
        alpha = tc['validation_ema_alpha']
        ema = row['validation_loss'] if ema is None else alpha * row['validation_loss'] + (1-alpha)*ema
        if not np.isfinite(ema) or abs(ema-row['validation_loss_EMA']) > 1e-10:
            raise ValueError('inner EMA rule differs')
        if row['epoch'] >= tc['minimum_epoch'] and ema < best-1e-8:
            best, epoch = ema, row['epoch']
        if row['epoch']-epoch >= tc['early_stopping_patience'] and row['epoch'] != len(history):
            raise ValueError('inner run continued beyond locked patience')
    if epoch != selected or selected < tc['minimum_epoch'] or len(history) > tc['max_epochs']:
        raise ValueError('selected checkpoint not the locked inner-only EMA minimum')
    if len(history) < tc['max_epochs'] and len(history)-selected < tc['early_stopping_patience']:
        raise ValueError('inner run stopped before budget/patience completion')


def checkpoint_path(private, arm, seed, fold, role):
    return Path(private) / role / f'{arm}_seed{seed}_fold{fold}.pt'


def load_checkpoint(path, arm, seed, fold, role, training_source):
    import torch
    path = Path(path); receipt = path.with_suffix('.json')
    if not path.is_file() or not receipt.is_file():
        raise FileNotFoundError('checkpoint+receipt required: ' + str(path))
    metadata = json.loads(receipt.read_text()); ck = torch.load(path, map_location='cpu', weights_only=False)
    if metadata.get('status') != 'PASS':
        raise ValueError('checkpoint receipt is not a completed PASS artifact')
    if metadata['source'] != training_source or ck['source'] != training_source or metadata['checkpoint_sha256'] != sha(path):
        raise ValueError('checkpoint source/hash mismatch')
    for key, value in [('arm', arm), ('seed', seed), ('fold', fold), ('role', role)]:
        if ck[key] != value:
            raise ValueError(('checkpoint identity mismatch', key, path.name))
    if ck['selected_epoch'] != metadata['selected_epoch']:
        raise ValueError('checkpoint selected epoch differs from receipt')
    if role == 'outer':
        inner_path = path.parent.parent / 'inner' / path.name
        inner_hash = sha(inner_path)
        if ck.get('inner_checkpoint_sha256') != inner_hash or metadata.get('inner_checkpoint_sha256') != inner_hash:
            raise ValueError('fresh refit does not identify the exact inner selection artifact')
    return ck, metadata


def qualify_checkpoint(ck, ds, config):
    data, train = _api()
    if not np.array_equal(ck['fit'], ds.fit) or not np.array_equal(ck['validation_or_test'], ds.valid):
        raise ValueError('checkpoint patient fit/heldout partition mismatch')
    if np.intersect1d(ds.fit, ds.valid).size or (ds.role == 'inner' and np.intersect1d(ds.active, ds.part['outer_test']).size):
        raise ValueError('patient split leakage')
    assert_equal_structure(ck['preprocess_audit'], ds.audit, 'Phase19_preprocessing')
    history_audit(ck, config)
    assigned = train.assigned_mode(ck['arm'])
    if ck.get('assigned_mode') != assigned:
        raise ValueError('wrong-patient arm assignment differs from locked rule')
    assigned_key = {'matched': 'matched_risk', 'wrongC': 'wrongC_risk', 'wrongO': 'wrongO_risk'}[assigned]
    if not np.array_equal(ck['risk'], ck[assigned_key]):
        raise ValueError('primary prediction is not the training-assigned pairing mode')
    for key in ['risk', 'matched_risk', 'wrongC_risk', 'wrongO_risk', 'constantC_risk', 'interaction_off_risk']:
        if np.asarray(ck[key]).shape != (len(ds.valid),) or not np.isfinite(ck[key]).all():
            raise ValueError('nonfinite/missing heldout prediction ' + key)
    for mapping in [ds.wrong_c, ds.wrong_o]:
        for indices in [ds.fit, ds.valid]:
            if np.any(mapping[indices] == indices) or set(mapping[indices]) != set(indices):
                raise ValueError('mismatch map crosses partition or contains identity')
    if ds.role == 'outer':
        path = checkpoint_path(data.P, ck['arm'], ck['seed'], ck['fold'], 'inner')
        inner, _ = load_checkpoint(path, ck['arm'], ck['seed'], ck['fold'], 'inner', data.source())
        if inner['selected_epoch'] != ck['selected_epoch']:
            raise ValueError('outer epoch was not selected from same-source inner run')
        if not np.array_equal(inner['fit'], ds.part['inner_train']) or not np.array_equal(inner['validation_or_test'], ds.part['inner_val']):
            raise ValueError('outer selector used wrong inner partitions')


def load_simple(ds):
    import torch
    from scripts.validate_phase17_independent import anchor_gradient
    data, _ = _api()
    path = data.P / 'simple' / f'{ds.role}_fold{ds.fold}.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['source'] != data.source() or saved['fold'] != ds.fold or saved['role'] != ds.role:
        raise ValueError('convex reference identity/source mismatch')
    if not np.array_equal(saved['fit'], ds.fit) or not np.array_equal(saved['valid'], ds.valid):
        raise ValueError('convex reference fit/valid mismatch')
    if hasattr(ds,'audit'):
        expected_audit=json.loads(json.dumps({k:v for k,v in ds.audit.items() if k!='wrong_controls'},
            default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item(),sort_keys=True))
        assert_equal_structure(saved['preprocess_audit'],expected_audit,'convex_feature_audit')
    checks = []
    receipt = path.with_suffix('.json')
    if not receipt.is_file():
        raise FileNotFoundError('convex reference prior hash receipt required: ' + str(receipt))
    meta = json.loads(receipt.read_text())
    if (meta.get('status') != 'PASS' or meta['source'] != data.source()
            or meta['checkpoint_sha256'] != sha(path) or meta['fold'] != ds.fold or meta['role'] != ds.role):
        raise ValueError('convex reference hash receipt/identity mismatch')
    for name, key in [('J0', 'X'), ('J_CS', 'CS')]:
        row = saved['models'][name]; x = ds.f[key]
        expected = np.asarray(x[ds.valid] @ row['w'], np.float32)
        error = float(abs(expected-row['risk']).max())
        gradient = float(abs(anchor_gradient(x[ds.fit].astype(float), ds.t[ds.fit], ds.e[ds.fit],
                                             row['w'], row['lambda'])).max())
        if row['lambda'] != data.cfg()['training']['main_effect_l2'] or error >= 1e-6 or gradient >= 2e-5:
            raise ValueError(('convex reference numerical audit failed', name, error, gradient))
        checks.append({'arm': name, 'role': ds.role, 'fold': ds.fold, 'risk_reconstruction_max_error': error,
                       'independent_penalized_Efron_stationarity': gradient, 'lambda': row['lambda'],
                       'checkpoint_sha256': sha(path), 'prior_hash_receipt': receipt.exists()})
    return saved, checks


def replay(shard=0, shards=1):
    import torch
    import pandas as pd
    data, train = _api(); config = data.cfg(); training_source = data.source()
    lock = json.loads((data.R/'PROTOCOL_LOCK.json').read_text())
    if lock['source'] != training_source:
        raise ValueError('training source changed since protocol lock')
    jobs = [(arm, seed, fold, role) for arm in config['arms'] for role in ['inner', 'outer']
            for seed in config['seeds'] for fold in config['folds']]
    rows, features, seen_features = [], [], set()
    for arm, seed, fold, role in jobs[shard::shards]:
        path = checkpoint_path(data.P, arm, seed, fold, role)
        ck, receipt = load_checkpoint(path, arm, seed, fold, role, training_source)
        train.seed_all(seed+100*fold); ds = data.Data(fold, role, seed)
        qualify_checkpoint(ck, ds, config)
        if (role, fold) not in seen_features:
            features.append(independent_feature_audit(ds)); seen_features.add((role, fold))
        simple, _ = load_simple(ds)
        if not np.array_equal(ck['main_initial_w'], simple['models']['J0']['w']):
            raise ValueError('neural arm did not use common fit-local ridge initialization')
        model = train.make_model(ds, arm, ck['main_initial_w']); model.load_state_dict(ck['state'], strict=True); model.eval()
        errors = {}
        with torch.no_grad():
            for mode, key in [('matched', 'matched_risk'), ('wrongC', 'wrongC_risk'), ('wrongO', 'wrongO_risk'),
                              ('constantC', 'constantC_risk'), ('off', 'interaction_off_risk')]:
                actual = train.forward(model, ds, ds.valid, mode=mode, training=False).cpu().numpy()
                errors[mode] = float(abs(actual-ck[key]).max())
        maximum = max(errors.values())
        if maximum >= 2e-6:
            raise ValueError(('checkpoint prediction replay mismatch', arm, seed, fold, role, errors))
        own_main = np.asarray(ds.f['X'][ds.valid].astype(float) @ ck['state']['main'].cpu().numpy(), np.float32)
        off_error = float(abs(own_main-ck['interaction_off_risk']).max())
        if off_error >= 5e-6:
            raise ValueError('interaction-off changed own raw main effects')
        rows.append({'arm': arm, 'seed': seed, 'fold': fold, 'role': role, 'source': training_source,
                     'checkpoint_sha256': receipt['checkpoint_sha256'], 'assigned_mode': ck['assigned_mode'],
                     'max_replay_error': maximum, 'main_off_independent_max_error': off_error,
                     'selected_epoch': ck['selected_epoch'], 'same_fit_initial_main': True})
        print('PHASE19_REPLAY', arm, seed, fold, role, maximum, flush=True)
        del ds, model; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(data.R/f'CHECKPOINT_REPLAY_{shard}.csv', index=False)
    pd.DataFrame(features).to_csv(data.R/f'FEATURE_RECONSTRUCTION_{shard}.csv', index=False)
    write(data.P/f'REPLAY_SHARD_{shard}.json', {'status':'PASS','source':training_source,'shards':shards,'rows':len(rows)})


def collect():
    import torch
    import pandas as pd
    from types import SimpleNamespace
    data, train = _api(); config = data.cfg(); training_source = data.source()
    done = json.loads((data.P/'TRAINING_DONE.json').read_text())
    lock = json.loads((data.R/'PROTOCOL_LOCK.json').read_text())
    if done['status'] != 'PASS' or done['source'] != training_source or lock['source'] != training_source:
        raise ValueError('complete training and immutable source lock required')
    cohort = data.parent.load(); event = data.parent.endpoints(cohort)['DSS']
    if len(cohort['patient_ids']) != 512 or int(np.asarray(event['event']).sum()) != 65:
        raise ValueError('frozen DSS cohort changed')
    replay_rows = []
    for path in sorted(data.R.glob('CHECKPOINT_REPLAY_*.csv')):
        replay_rows.extend(pd.read_csv(path).to_dict('records'))
    expected_keys = {(arm, seed, fold, role) for arm in config['arms'] for role in ['inner', 'outer']
                     for seed in config['seeds'] for fold in config['folds']}
    actual_keys = [(q['arm'], q['seed'], q['fold'], q['role']) for q in replay_rows]
    if len(actual_keys) != len(expected_keys) or set(actual_keys) != expected_keys:
        raise ValueError('neural checkpoint replay coverage incomplete/duplicated')
    replay_index = {key: row for key, row in zip(actual_keys, replay_rows)}
    for row in replay_rows:
        if row['source'] != training_source or row['max_replay_error'] >= 2e-6 or row['main_off_independent_max_error'] >= 5e-6:
            raise ValueError('neural checkpoint replay failed/stale')
    feature_rows = []
    for path in sorted(data.R.glob('FEATURE_RECONSTRUCTION_*.csv')):
        feature_rows.extend(pd.read_csv(path).to_dict('records'))
    if {(q['role'], q['fold']) for q in feature_rows} != {(r, f) for r in ['inner', 'outer'] for f in config['folds']}:
        raise ValueError('independent feature reconstruction coverage incomplete')
    banks = {}; simple_models = {}; convex_checks = []; manifest = []
    for role in ['inner', 'outer']:
        for fold in config['folds']:
            path = data.P/'preprocessing'/f'fold{fold}_{role}.pt'
            receipt = json.loads(path.with_suffix('.json').read_text())
            bank = torch.load(path, map_location='cpu', weights_only=False)
            if sha(path) != receipt['sha256'] or bank['provenance']['source'] != training_source:
                raise ValueError('preprocessing cache receipt/source mismatch')
            feature_checks = [row for row in feature_rows if row['role'] == role and row['fold'] == fold]
            if any(row.get('status') != 'PASS' or row['feature_cache_sha256'] != sha(path)
                   for row in feature_checks):
                raise ValueError('independent feature reconstruction is failed/stale for current cache')
            parts = data.parent.splits(cohort, 'DSS', fold)
            fit, valid = parts['inner_train' if role == 'inner' else 'outer_train'], parts['inner_val' if role == 'inner' else 'outer_test']
            if not np.array_equal(fit, bank['fit']) or not np.array_equal(valid, bank['valid']):
                raise ValueError('preprocessing cache used wrong fit/heldout partition')
            ns = SimpleNamespace(f=bank['features'],fit=fit,valid=valid,t=event['time'],e=event['event'],role=role,fold=fold)
            simple, checks = load_simple(ns); convex_checks += checks
            banks[role, fold] = bank; simple_models[role, fold] = simple
            for reference in config['ridge_references']:
                manifest.append({'arm':reference,'seed':'deterministic','fold':fold,'role':role,
                    'fit_classification':'new_unique_convex_fit','source':training_source,
                    'checkpoint_sha256':sha(data.P/'simple'/f'{role}_fold{fold}.pt'),
                    'selected_epoch':0,'actual_epochs':0,'assigned_mode':'own_inputs'})
    risks = {}; curves = []; mechanisms = []; summaries = []; loaded_inner = {}
    coverage = {}; visual_owner = np.load(data.VISUAL / 'owner.npy')
    for arm in config['arms']:
        for role in ['inner', 'outer']:
            for seed in config['seeds']:
                for fold in config['folds']:
                    path = checkpoint_path(data.P, arm, seed, fold, role)
                    ck, receipt = load_checkpoint(path,arm,seed,fold,role,training_source)
                    key = arm, seed, fold, role
                    if replay_index[key]['checkpoint_sha256'] != receipt['checkpoint_sha256']:
                        raise ValueError('neural checkpoint changed since replay')
                    bank = banks[role,fold]; simple = simple_models[role,fold]
                    if not np.array_equal(ck['fit'],bank['fit']) or not np.array_equal(ck['validation_or_test'],bank['valid']):
                        raise ValueError('saved neural prediction partitions differ from audited preprocessing')
                    if ck['preprocess_audit']['cache_sha256'] != sha(data.P/'preprocessing'/f'fold{fold}_{role}.pt'):
                        raise ValueError('saved neural preprocessing identity differs')
                    if not np.array_equal(ck['main_initial_w'],simple['models']['J0']['w']):
                        raise ValueError('shared main-effect initialization differs')
                    history_audit(ck,config)
                    if role == 'inner':
                        loaded_inner[arm,seed,fold] = ck['selected_epoch']
                    elif ck['selected_epoch'] != loaded_inner[arm,seed,fold]:
                        raise ValueError('fresh outer fit length not selected within same fold/seed/arm')
                    if ck['assigned_mode'] != train.assigned_mode(arm):
                        raise ValueError('wrong-patient primary prediction assignment differs')
                    assigned_key = {'matched':'matched_risk','wrongC':'wrongC_risk','wrongO':'wrongO_risk'}[ck['assigned_mode']]
                    if not np.array_equal(ck['risk'],ck[assigned_key]):
                        raise ValueError('saved primary risk is not assigned-mode risk')
                    update_probe_coverage(coverage, ck, visual_owner)
                    for row in ck['history']:
                        curves.append({'arm':arm,'seed':seed,'fold':fold,'role':role,
                                       'selected_epoch':ck['selected_epoch'],**row})
                    mechanisms.append({'arm':arm,'seed':seed,'fold':fold,'role':role,
                                       'selected_epoch':ck['selected_epoch'],**ck['mechanism']})
                    summaries.append({'arm':arm,'seed':seed,'fold':fold,'role':role,
                        **{k:v for k,v in ck['summary'].items() if not isinstance(v,(dict,list,np.ndarray))}})
                    manifest.append({'arm':arm,'seed':seed,'fold':fold,'role':role,
                        'fit_classification':'new_neural_fit','source':training_source,
                        'checkpoint_sha256':receipt['checkpoint_sha256'],'selected_epoch':ck['selected_epoch'],
                        'actual_epochs':len(ck['history']),'assigned_mode':ck['assigned_mode']})
                    if role == 'outer':
                        test = bank['valid']; vector = risks.setdefault((arm,seed),np.full(512,np.nan))
                        if np.isfinite(vector[test]).any():
                            raise ValueError('duplicate outer patient risk assignment')
                        vector[test] = ck['risk']
                        if arm == 'A3':
                            for tag,field in [('WRONG_C','wrongC_risk'),('WRONG_O','wrongO_risk'),
                                              ('CONSTANT_C','constantC_risk'),('INTERACTION_OFF','interaction_off_risk')]:
                                risks.setdefault(('A3__'+tag,seed),np.full(512,np.nan))[test]=ck[field]
    for name in config['ridge_references']:
        for seed in config['seeds']:
            vector = np.full(512,np.nan)
            for fold in config['folds']:
                bank = banks['outer',fold]
                vector[bank['valid']]=simple_models['outer',fold]['models'][name]['risk']
            risks[name,seed]=vector
    # This immutable strong reference used a different visual backbone/protocol.
    historic = np.load('/root/phase16r_light_rna_private/OOF_PRIVATE.npz')
    historic_rows=[]
    for seed in config['seeds']:
        vector = historic['LOCAL__CV3__seed'+str(seed)]
        if vector.shape!=(512,) or not np.isfinite(vector).all():
            raise ValueError('historical LOCAL reference incomplete')
        for fold in config['folds']:
            artifact=Path('/root/phase16r_light_rna_private/artifacts')/f'seed{seed}_fold{fold}.pt'
            meta=json.loads((Path('/root/phase16r_light_rna_private/cells')/f'seed{seed}_fold{fold}.json').read_text())
            if sha(artifact)!=meta['sha256']:
                raise ValueError('historical LOCAL artifact hash mismatch')
            old=torch.load(artifact,map_location='cpu',weights_only=False);test=banks['outer',fold]['valid']
            if not np.array_equal(old['part']['outer_test'],test):
                raise ValueError('historical LOCAL patient fold differs')
            error=float(abs(old['models']['LOCAL__CV3']['risk']-vector[test]).max())
            if error>1e-9:
                raise ValueError('historical LOCAL OOF and saved cell disagree')
            historic_rows.append({'arm':'HIST_CS_S32_CONCH32','seed':seed,'fold':fold,
                'artifact_sha256':meta['sha256'],'maximum_error':error,
                'scope':'different visual representation and training protocol; not same-input ablation or new fit'})
        risks['HIST_CS_S32_CONCH32',seed]=vector.copy()
    check_model_seed_coverage(risks,config['seeds'])
    for key,value in risks.items():
        if value.shape!=(512,) or not np.isfinite(value).all():
            raise ValueError(('incomplete patient OOF',key))
    frame=pd.DataFrame(summaries); frame.to_csv(data.R/'COMPUTE_PROFILE.csv',index=False)
    budget=[]
    for arm,group in frame[frame.role=='inner'].groupby('arm'):
        fraction=float(group.hit_maximum.mean())
        budget.append({'arm':arm,'inner_cells':len(group),'cap_fraction':fraction,
            'maximum_cap_fraction':config['engineering_gate']['maximum_cap_fraction_per_arm'],
            'budget_gate_pass':fraction<=config['engineering_gate']['maximum_cap_fraction_per_arm'],
            'selected_epoch_median':float(group.selected_epoch.median()),'actual_epoch_median':float(group.actual_epochs.median())})
    original_budget=json.loads((data.R/'BUDGET_AUDIT.json').read_text())
    if original_budget['source']!=training_source or not original_budget['no_outer_results_used']:
        raise ValueError('pre-outer budget audit missing or wrong source')
    if any(abs(original_budget['inner_cap_fractions'][q['arm']]-q['cap_fraction'])>1e-12 for q in budget):
        raise ValueError('pre-outer budget assessment differs from final checkpoint histories')
    write(data.R/'BUDGET_VERIFICATION.json',{'status':'PASS','source':training_source,'rows':budget,
        'mechanism_budget_sufficient':all(q['budget_gate_pass'] for q in budget if q['arm'].startswith('A')),
        'all_method_budgets_sufficient':all(q['budget_gate_pass'] for q in budget),
        'original_preouter_status':original_budget['status'],'original_preouter_audit_preserved':True,
        'interpretation':'locked patience/cap adequacy only; not proof of global convergence; unresolved budget is not mechanism no-go'})
    pd.DataFrame(manifest).to_csv(data.R/'CELL_MANIFEST.csv',index=False)
    pd.DataFrame(curves).to_csv(data.R/'TRAINING_CURVES.csv',index=False)
    pd.DataFrame(flatten_diagnostics(mechanisms)).to_csv(data.R/'ATTENTION_MECHANISM.csv',index=False)
    pd.DataFrame(replay_rows).to_csv(data.R/'CHECKPOINT_REPLAY.csv',index=False)
    pd.DataFrame(feature_rows).to_csv(data.R/'FEATURE_RECONSTRUCTION.csv',index=False)
    pd.DataFrame(convex_checks).to_csv(data.R/'CONVEX_REFERENCE_AUDIT.csv',index=False)
    pd.DataFrame(historic_rows).to_csv(data.R/'HISTORICAL_LOCAL_REFERENCE_AUDIT.csv',index=False)
    np.savez_compressed(data.P/'OOF_PRIVATE.npz',**{arm+'__seed'+str(seed):value for (arm,seed),value in risks.items()})
    write(data.R/'CHECKPOINT_REPLAY.json',{'status':'PASS','source':training_source,
        'new_neural_checkpoints':len(replay_rows),'modes_per_checkpoint':5,
        'maximum_error':max(q['max_replay_error'] for q in replay_rows),
        'own_main_off_maximum_error':max(q['main_off_independent_max_error'] for q in replay_rows)})
    write(data.R/'FIT_AND_REFERENCE_COUNTS.json',{'status':'PASS','source':training_source,
        'new_inner_neural_fits':len(config['arms'])*len(config['seeds'])*len(config['folds']),
        'new_outer_neural_fits':len(config['arms'])*len(config['seeds'])*len(config['folds']),
        'new_unique_convex_models':len(config['ridge_references'])*len(config['folds'])*2,
        'convex_seed_role_metric_references':len(config['ridge_references'])*len(config['folds'])*len(config['seeds'])*2,
        'historical_LOCAL_outer_references':len(historic_rows),'historical_pilot_125_new_fit_or_OOF_count':0,
        'note':'diagnostic short fits and prototype/discriminator work separately recorded by training ledger; replay/probes are not neural refits'})
    coverage_rows = summarize_probe_coverage(coverage)
    if len(coverage_rows) != len(config['arms']) * 2 or any(row['checkpoint_count'] != 15 for row in coverage_rows):
        raise ValueError('probe metadata checkpoint coverage incomplete')
    write(data.R/'PROBE_COVERAGE_AUDIT.json', {'status':'PASS', 'source':training_source,
        'rows':coverage_rows, 'additional_forward_or_gradient_calls':0,
        'scope':'query patients from saved ck.fit; risk/attention uses fit[:32], isolated input-gradient target uses fit[:1]; native gradient unavailable',
        'interpretation':'query patient/WSI unions, not independent checkpoint counts; wrong-patient context donor identity is not the probe risk query identity',
        'raw_patient_indices_public':False})
    return cohort,risks


def formal_evaluation():
    data,_=_api();cohort,risks=collect();config=data.cfg()
    t,e,valid,w,seeds=metrics(cohort,risks,'DSS',data.R)
    bootstrap(cohort,risks,(t,e,valid,w),seeds,config,data.R)
    metrics(cohort,risks,'PFI',data.R)
    write(data.R/'INDEPENDENT_VALIDATION.json',{'status':'PASS','source':data.source(),
        'model_seed_vectors':len(risks),'outer_fold_metric_rows':len(risks)*len(config['folds']),
        'independent_sksurv_Harrell_Uno60_each_fold':True,'risk_ensembling':False,
        'primary_tests':12,'OOF':'only current full five-fold fresh outer fits, deterministic convex fits, qualified frozen LOCAL reference',
        'old_125_inner_pilot_fits_counted_as_OOF':False,
        'PFI':'frozen DSS predictions; no PFI fitting or selection',
        'Clinical_FiLM':'CP trilinear score is equivalent to diagonal Clinical scaling of latent query/key; no independent FiLM innovation claim'})
    print('PHASE19_INDEPENDENT_EVALUATION_PASS',len(risks)*len(config['folds']),flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--replay', action='store_true')
    parser.add_argument('--shard', type=int, default=0); parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    if args.replay:
        replay(args.shard, args.shards)
    else:
        formal_evaluation()


if __name__ == "__main__":
    main()
