"""Locked Phase7-P0 feasibility audit. Patient-level data stay on the server.

This is an evidence-level interaction audit, not TITAN token-prompt training.
All data-adaptive transforms are fitted on the current training partition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

REPO = Path('/root/diffusion_phase5_20261007')
BASE = Path('/root/autodl-tmp/diffusion_data/data/data_8/coadread_tcga_pan_can_atlas_2018')
CACHE = Path('/root/phase6a_titan_runtime/cache_v1')
CONFIG = REPO / 'configs/phase7_p0_biology_prompt_audit.json'
PRIOR = REPO / 'configs/phase7_p0_molecular_morphology_prior.json'
CONCEPTS = REPO / 'configs/phase6a_titan_crc_morphology_concepts.json'
HALLMARK = REPO / 'src/phase_m2/metadata/mmp_hallmarks_signatures.csv'
OUT_DEFAULT = Path('/root/phase7_p0_biology_audit_v4')
CLASSES = {'T': ['T1', 'T2', 'T3', 'T4'], 'N': ['N0', 'N1', 'N2'], 'M': ['M0', 'M1'], 'Stage': ['I', 'II', 'III', 'IV']}
REPS = ['R0_raw_rna_top512', 'R1_true_biology_state_rna', 'R2_matched_shuffled_state_rna', 'R3_true_biology_state_rna_cna']
BRIDGES = ['B0_static_titan_concepts', 'B1_raw_rna_additive', 'B1_molecular_state_additive', 'B2_biology_grounded_concept_bridge', 'B3_shuffled_concept_mapping', 'B4_wrong_patient_omics_bridge']
METHODS = REPS + BRIDGES
STATES = ['proliferation_g2m', 'invasion_emt', 'immune_ifng', 'hypoxia', 'tgfb_stroma', 'dna_repair', 'wnt_beta_catenin', 'angiogenesis']
HALLMARK_NAMES = ['HALLMARK_G2M_CHECKPOINT', 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION', 'HALLMARK_INTERFERON_GAMMA_RESPONSE', 'HALLMARK_HYPOXIA', 'HALLMARK_TGF_BETA_SIGNALING', 'HALLMARK_DNA_REPAIR', 'HALLMARK_WNT_BETA_CATENIN_SIGNALING', 'HALLMARK_ANGIOGENESIS']
PAIRS = [('R1-R0', REPS[1], REPS[0]), ('R1-R2', REPS[1], REPS[2]), ('R3-R1', REPS[3], REPS[1]),
         ('B2-B0', BRIDGES[3], BRIDGES[0]), ('B2-B1raw', BRIDGES[3], BRIDGES[1]), ('B2-B1mol', BRIDGES[3], BRIDGES[2]),
         ('B2-B3', BRIDGES[3], BRIDGES[4]), ('B2-B4', BRIDGES[3], BRIDGES[5])]
PATIENT_RE = re.compile(r'^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', re.I)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def ahash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def patient_id(s):
    m = PATIENT_RE.match(str(s).strip().upper())
    return m.group(1) if m else None


def load_omics(path, ids):
    tab = pd.read_csv(path, sep='\t', comment='#', low_memory=False)
    columns = {}
    for c in tab.columns[2:]:
        p = patient_id(c)
        if p in ids:
            columns.setdefault(p, []).append(c)
    assert all(p in columns for p in ids), 'Missing cohort patients'
    values = tab.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
    ci = {c: i for i, c in enumerate(tab.columns[2:])}
    selected = []
    for p in ids:
        primary = [c for c in columns[p] if str(c)[13:15] == '01']
        # Fail closed: do not substitute normal tissue or metastatic samples.
        assert primary, 'Patient lacks primary-tumor measurement'
        selected.append(sorted(primary))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        raw = np.stack([np.nanmean(values[:, [ci[c] for c in cs]], axis=1) for cs in selected])
    symbols = tab.iloc[:, 0].fillna('').astype(str).str.strip().str.upper().to_numpy(dtype='U64')
    valid_symbols = [s for s in symbols if s and s != 'NAN']
    finite = np.isfinite(raw)
    audit = {'patients': len(ids), 'aligned_patients': len(ids), 'missing_patients': 0,
             'selection': 'primary tumor 01 only; mean duplicate primary aliquots',
             'patients_with_multiple_primary_columns': sum(len(cs) > 1 for cs in selected),
             'source_samples': int(tab.shape[1] - 2), 'gene_rows': int(raw.shape[1]),
             'valid_unique_symbols': len(set(valid_symbols)), 'duplicate_symbol_rows': len(valid_symbols) - len(set(valid_symbols)),
             'blank_symbol_rows': int(sum(not s or s == 'NAN' for s in symbols)),
             'finite_fraction': float(finite.mean()), 'all_missing_gene_rows': int((~finite.any(0)).sum()),
             'all_missing_patient_rows': int((~finite.any(1)).sum()),
             'nonfinite_count': int((~finite).sum()), 'infinity_count': int(np.isinf(raw).sum()),
             'negative_finite_count': int((raw[finite] < 0).sum()),
             'finite_value_quantiles': np.quantile(raw[finite], [0, .01, .5, .99, 1]).tolist()}
    return raw, symbols, audit, selected


def collapse_symbols(raw, symbols, log_rna=False):
    x = raw.copy()
    if log_rna:
        # PanCancer batch-normalized RSEM can contain small negative values.
        # Preserve them; log2(x+1) is valid only when every finite x > -1.
        assert np.all(x[np.isfinite(x)] > -1), 'RNA log2(x+1) domain violation'
        x = np.log2(x + 1.)
    valid = sorted(set(s for s in symbols if s and s != 'NAN'))
    loc = {}
    for j, s in enumerate(symbols):
        loc.setdefault(s, []).append(j)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        out = np.stack([np.nanmean(x[:, loc[s]], axis=1) for s in valid], axis=1)
    return out.astype(np.float64), np.asarray(valid, dtype='U64')


def gene_z(raw, fit_idx):
    train = raw[fit_idx]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(train, axis=0)
    med[~np.isfinite(med)] = 0.
    filled = np.where(np.isfinite(raw), raw, med)
    mean = filled[fit_idx].mean(0)
    var = filled[fit_idx].var(0)
    sd = np.sqrt(var)
    sd[sd < 1e-8] = 1.
    usable = (np.isfinite(train).mean(0) >= .9) & (var > 1e-8)
    return (filled - mean) / sd, {'median': med, 'mean': mean, 'sd': sd, 'var': var, 'usable': usable}


def raw_top512(raw, fit_idx):
    x, st = gene_z(np.log2(raw.astype(np.float64) + 1.), fit_idx)
    eligible = np.flatnonzero(st['usable'])
    keep = eligible[np.argsort(-st['var'][eligible], kind='stable')[:512]]
    assert len(keep) == 512
    return x[:, keep], keep, st


def zscale(x, fit_idx):
    mean = x[fit_idx].mean(0)
    sd = x[fit_idx].std(0)
    sd[sd < 1e-8] = 1.
    return (x - mean) / sd, mean, sd


def matched_permutation(st, seed):
    eligible = np.flatnonzero(st['usable'])
    bins = []
    for key in ['mean', 'sd']:
        rank = np.argsort(np.argsort(st[key][eligible], kind='stable'), kind='stable')
        bins.append(np.minimum(4, rank * 5 // len(eligible)))
    code = bins[0] * 5 + bins[1]
    perm = np.arange(len(st['mean']))
    rng = np.random.default_rng(seed)
    for b in np.unique(code):
        members = rng.permutation(eligible[code == b])
        if len(members) > 1:
            perm[members] = np.roll(members, 1)
    assert len(set(perm[eligible])) == len(eligible)
    return perm


def group_scores(x, symbols, groups, usable, perm=None):
    loc = {s: j for j, s in enumerate(symbols)}
    sets = []
    for state in STATES:
        idx = np.asarray([loc[g] for g in groups[state] if g in loc and usable[loc[g]]], dtype=np.int64)
        assert len(idx) >= 3, 'Insufficient train-measured genes in ' + state
        sets.append(idx if perm is None else perm[idx])
    return np.stack([x[:, idx].mean(1) for idx in sets], axis=1), sets


def normalize_mask(binary):
    return binary / np.maximum(binary.sum(1, keepdims=True), 1)


def shuffled_mask(binary, seed):
    rng = np.random.default_rng(seed)
    sh = binary.copy()
    for _ in range(5000):
        edges = np.argwhere(sh > 0)
        (r1, c1), (r2, c2) = edges[rng.choice(len(edges), 2, replace=False)]
        if r1 != r2 and c1 != c2 and sh[r1, c2] == 0 and sh[r2, c1] == 0:
            sh[r1, c1] = sh[r2, c2] = 0
            sh[r1, c2] = sh[r2, c1] = 1
    assert np.array_equal(sh.sum(0), binary.sum(0)) and np.array_equal(sh.sum(1), binary.sum(1))
    assert (sh != binary).sum() >= 4
    return sh


def preprocessing(data, fit_idx, protocol):
    raw, keep, raw_stats = raw_top512(data['raw_rna'], fit_idx)
    rz, rs = gene_z(data['rna_genes'], fit_idx)
    cz, cs = gene_z(data['cna_genes'], fit_idx)
    permutation = matched_permutation(rs, protocol['shuffled_gene_seed'])
    true_r, true_sets = group_scores(rz, data['rna_gene_symbols'], data['groups'], rs['usable'])
    random_r, random_sets = group_scores(rz, data['rna_gene_symbols'], data['groups'], rs['usable'], permutation)
    true_c, cna_sets = group_scores(cz, data['cna_gene_symbols'], data['groups'], cs['usable'])
    # No RNA/CNA averaging: dosage and expression are distinct evidence.
    zr, _, _ = zscale(true_r, fit_idx)
    zrandom, _, _ = zscale(random_r, fit_idx)
    zc, _, _ = zscale(true_c, fit_idx)
    s, sm, ss = zscale(data['static'], fit_idx)
    out = {'raw': raw, 'rna': zr, 'random': zrandom, 'cna': zc, 's': s,
           'static_raw': data['static'], 'static_mean': sm, 'static_sd': ss,
           'fit_idx': fit_idx, 'raw_keep': keep, 'gene_permutation': permutation,
           'rna_usable': rs['usable'], 'cna_usable': cs['usable']}
    overlap = lambda sets: np.asarray([[len(set(a) & set(b)) for b in sets] for a in sets])
    assert np.array_equal(overlap(true_sets), overlap(random_sets))
    counts = {state: {'rna': len(r), 'shuffled_rna': len(rr), 'cna': len(c)} for state, r, rr, c in zip(STATES, true_sets, random_sets, cna_sets)}
    selected = np.unique(np.concatenate(true_sets))
    audit = {'fit_n': len(fit_idx), 'fit_indices_sha256': ahash(fit_idx), 'raw_top512_sha256': ahash(keep),
             'selection_before_standardization': True, 'counts': counts, 'gene_overlap_preserved': True,
             'gene_permutation_sha256': ahash(permutation),
             'fraction_signature_genes_replaced': float(np.mean(permutation[selected] != selected)),
             'matched_mean_absolute_difference': float(np.mean(np.abs(rs['mean'][selected] - rs['mean'][permutation[selected]]))),
             'matched_sd_absolute_difference': float(np.mean(np.abs(rs['sd'][selected] - rs['sd'][permutation[selected]]))),
             'rna_train_usable_symbols': int(rs['usable'].sum()), 'cna_train_usable_symbols': int(cs['usable'].sum())}
    return out, audit


def feature_matrix(method, pre, rows, mask, lam, conditioning_rows=None):
    s = pre['s'][rows]
    z = np.concatenate([pre['rna'], pre['cna']], axis=1)[rows]
    if method == REPS[0]: return pre['raw'][rows]
    if method == REPS[1]: return pre['rna'][rows]
    if method == REPS[2]: return pre['random'][rows]
    if method == REPS[3]: return z
    if method == BRIDGES[0]: return s
    if method == BRIDGES[1]: return np.concatenate([s, pre['raw'][rows]], axis=1)
    base = np.concatenate([s, z], axis=1)
    if method == BRIDGES[2]: return base
    assert method in BRIDGES[3:]
    context = rows if conditioning_rows is None else conditioning_rows
    # Keep raw cosine anchor. Divide corrections by the SAME static scale;
    # separately standardizing correction columns would cancel lambda.
    sr = pre['static_raw'][rows]
    rdelta = lam * sr * np.tanh(pre['rna'][context] @ mask.T)
    cdelta = lam * sr * np.tanh(pre['cna'][context] @ mask.T)
    assert np.all(np.abs(rdelta) <= lam * np.abs(sr) + 1e-12)
    assert np.all(np.abs(cdelta) <= lam * np.abs(sr) + 1e-12)
    return np.concatenate([base, rdelta / pre['static_sd'], cdelta / pre['static_sd']], axis=1)


def fit_lr(x, y, C):
    model = LogisticRegression(C=C, class_weight='balanced', tol=1e-6, max_iter=5000, solver='lbfgs')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, y)
    assert model.n_iter_.max() < 5000, 'Classifier training budget exhausted'
    return model


def source_files():
    return [CONFIG, PRIOR, CONCEPTS] + [REPO / 'scripts' / name for name in
           ['phase7_p0_biology_prompt_audit.py', 'validate_phase7_p0_biology_prompt_audit.py',
            'report_phase7_p0_biology_prompt_audit.py', 'run_phase7_p0_biology_prompt_audit.sh', 'finish_phase7_p0_biology_prompt_audit.sh']]


def check_lock(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    assert digest(out / 'protocol.json') == manifest['protocol_sha256']
    for item in manifest['sources']:
        assert digest(REPO / item['path']) == item['sha256'], 'Source changed after lock: ' + item['path']
    return manifest


def load_private(out):
    data = dict(np.load(out / 'private_dataset.npz', allow_pickle=False))
    manifest = check_lock(out)
    data['groups'] = manifest['groups']
    return data, manifest


def prepare(out, protocol):
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'manifest.json').exists(), 'Preparation is immutable; choose a new output root'
    rep = dict(np.load(CACHE / 'patient_representations.npz', allow_pickle=False))
    ids = rep['patient_ids'].astype(str)
    assert len(ids) == len(set(ids)) == 512
    folds_table = pd.read_csv(REPO / 'outputs/phase_m1r/fixed_patient_folds.csv')
    assert not folds_table.patient_id.duplicated().any()
    folds = folds_table.set_index('patient_id').loc[ids, 'fold'].to_numpy(dtype=np.int64)
    assert set(folds) == {1, 2, 3, 4, 5}
    r, rs, ra, rsel = load_omics(BASE / 'data_mrna_seq_v2_rsem.txt', ids.tolist())
    c, cs, ca, csel = load_omics(BASE / 'data_log2_cna.txt', ids.tolist())
    assert np.isfinite(rep['concept_cosine']).all()
    rg, rgs = collapse_symbols(r, rs, True)
    cg, cgs = collapse_symbols(c, cs, False)
    hm = pd.read_csv(HALLMARK, dtype=str)
    groups = {state: sorted(set(hm[name].dropna().str.strip().str.upper())) for state, name in zip(STATES, HALLMARK_NAMES)}
    concepts = json.loads(CONCEPTS.read_text())['concepts']
    assert len(concepts) == 32 and rep['concept_cosine'].shape == (512, 32)
    prior = json.loads(PRIOR.read_text())
    binary = np.zeros((32, 8), dtype=np.float64)
    concept_lookup = {v['id']: i for i, v in enumerate(concepts)}
    for entry in prior['allowed']:
        for state in entry['states']:
            binary[concept_lookup[entry['concept_id']], STATES.index(state)] = 1
    shuffled = shuffled_mask(binary, protocol['shuffled_mapping_seed'])
    # All external priors and source hashes locked before reading target values.
    write_json(out / 'protocol.json', protocol)
    lock = {'protocol_sha256': digest(out / 'protocol.json'),
            'sources': [{'path': str(p.relative_to(REPO)), 'sha256': digest(p)} for p in source_files()],
            'groups': groups, 'hallmark_names': HALLMARK_NAMES, 'concept_ids': [v['id'] for v in concepts],
            'concept_mask': normalize_mask(binary).tolist(), 'shuffled_mapping': normalize_mask(shuffled).tolist(),
            'mask_edge_count': int(binary.sum()), 'unmapped_concepts': int((binary.sum(1) == 0).sum()),
            'mapping_row_degrees_preserved': True, 'mapping_column_degrees_preserved': True,
            'prior_status': 'candidate unsigned associations, not patient annotations',
            'survival_columns_read': [], 'private_outputs_published': False,
            'invalidated_run': {'version': 'v3', 'reason': 'standardized variance Top512 and wrong whole-patient control; aggregate outcomes not inspected or used'},
            'frozen_utc': pd.Timestamp.now(tz='UTC').isoformat()}
    targets = pd.read_csv(CACHE / 'clinical_targets.csv', dtype=str, usecols=['patient_id'] + list(CLASSES)).set_index('patient_id').loc[ids]
    ys = {e: targets[e].map({name: i for i, name in enumerate(names)}).fillna(-1).to_numpy(dtype=np.int64) for e, names in CLASSES.items()}
    np.savez_compressed(out / 'private_dataset.npz', patient_ids=np.asarray(ids, dtype='U20'), folds=folds,
                        raw_rna=r, raw_cna=c, rna_genes=rg, cna_genes=cg, rna_gene_symbols=rgs, cna_gene_symbols=cgs,
                        static=rep['concept_cosine'].astype(np.float64), **{'y_' + e: y for e, y in ys.items()})
    paths = [BASE / 'data_mrna_seq_v2_rsem.txt', BASE / 'data_log2_cna.txt', BASE / 'meta_log2_cna.txt',
             BASE / 'meta_mrna_seq_v2_rsem.txt', HALLMARK, CACHE / 'clinical_targets.csv', CACHE / 'patient_representations.npz',
             REPO / 'outputs/phase_m1r/fixed_patient_folds.csv']
    lock.update({'inputs': [{'path': str(p), 'sha256': digest(p), 'bytes': p.stat().st_size} for p in paths],
                 'private_dataset_sha256': digest(out / 'private_dataset.npz'), 'patients': 512,
                 'targets': {e: {'valid_n': int((y >= 0).sum()), 'class_counts': np.bincount(y[y >= 0], minlength=len(CLASSES[e])).tolist()} for e, y in ys.items()}})
    write_json(out / 'manifest.json', lock)
    write_json(out / 'encoder_audit.json', {'status': 'PASS', 'rna': ra, 'cna': ca,
               'same_selected_aliquot_lists_rna_cna': sum(a == b for a, b in zip(rsel, csel)),
               'rna_encoding': 'Batch-normalized RSEM; log2(x+1)', 'cna_encoding': 'LOG2-VALUE SNP6; no second log transform',
               'cna_interpretation': 'signed gene dosage, not pathway activation; no arm-level burden claims',
               'static_concept_shape': [512, 32], 'mapping_shape': [32, 8], 'mapping_edges': int(binary.sum()),
               'targets': lock['targets'], 'survival_columns_read': [], 'clinical_predictors_used': []})
    for sub in ['cells', 'oof', 'private_preprocessing']:
        (out / sub).mkdir(exist_ok=True)
    print(json.dumps({'status': 'PREPARED', 'patients': 512, 'protocol_sha256': lock['protocol_sha256']}), flush=True)


def split_indices(data, endpoint, seed, fold, protocol):
    y = data['y_' + endpoint]
    tr = np.flatnonzero((y >= 0) & (data['folds'] != fold))
    te = np.flatnonzero((y >= 0) & (data['folds'] == fold))
    split_seed = seed + fold * 10000 + int.from_bytes(hashlib.sha256(endpoint.encode()).digest()[:4], 'little') % 100000
    a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=protocol['inner_validation_fraction'], random_state=split_seed).split(tr, y[tr]))
    return tr, te, tr[a], tr[b], split_seed


def worker(out, worker_id, workers):
    protocol = json.loads((out / 'protocol.json').read_text())
    assert workers == protocol['workers']
    data, manifest = load_private(out)
    assert digest(out / 'private_dataset.npz') == manifest['private_dataset_sha256']
    mask = np.asarray(manifest['concept_mask'])
    shuffled = np.asarray(manifest['shuffled_mapping'])
    blocks = [(s, e, f) for s in protocol['seeds'] for e in protocol['endpoints'] for f in range(1, 6)]
    for b, (seed, endpoint, fold) in enumerate(blocks):
        if b % workers != worker_id: continue
        prefix = f'{endpoint}__seed{seed}__fold{fold}'
        if all((out / 'cells' / f'{prefix}__{m}.json').exists() for m in METHODS): continue
        check_lock(out)
        started = time.monotonic()
        tr, te, it, iv, split_seed = split_indices(data, endpoint, seed, fold, protocol)
        inner, ia = preprocessing(data, it, protocol)
        outer, oa = preprocessing(data, tr, protocol)
        assert not (set(it) & set(iv) or set(tr) & set(te))
        assert set(it) | set(iv) == set(tr)
        pp = out / 'private_preprocessing' / (prefix + '.npz')
        arrays = {k + '_inner': v for k, v in inner.items()}
        arrays.update({k + '_outer': v for k, v in outer.items()})
        arrays.update({'tr': tr, 'te': te, 'it': it, 'iv': iv})
        b2_model = None
        b2_C = None
        for method in METHODS[:-1]:
            m = shuffled if method == BRIDGES[4] else mask
            xit = feature_matrix(method, inner, it, m, protocol['bridge_lambda'])
            xiv = feature_matrix(method, inner, iv, m, protocol['bridge_lambda'])
            trials = []
            best = None
            y = data['y_' + endpoint]
            for C in protocol['candidate_C']:
                model = fit_lr(xit, y[it], C)
                ba = float(balanced_accuracy_score(y[iv], model.predict(xiv)))
                trials.append({'C': C, 'inner_ba': ba, 'n_iter': int(model.n_iter_.max()), 'converged': True})
                key = (ba, -C)
                if best is None or key > best[0]: best = (key, C)
            C = best[1]
            xtr = feature_matrix(method, outer, tr, m, protocol['bridge_lambda'])
            xte = feature_matrix(method, outer, te, m, protocol['bridge_lambda'])
            model = fit_lr(xtr, y[tr], C)
            assert np.array_equal(model.classes_, np.arange(len(CLASSES[endpoint])))
            tag = f'{prefix}__{method}'
            op = out / 'oof' / (tag + '.npz')
            np.savez_compressed(op, test_indices=te, y_true=y[te], probabilities=model.predict_proba(xte), classes=model.classes_,
                                seed=seed, fold=fold, endpoint=endpoint, method=method, protocol_sha256=manifest['protocol_sha256'])
            record = {'status': 'PASS', 'method': method, 'endpoint': endpoint, 'seed': seed, 'fold': fold,
                      'selected_C': C, 'selected_lambda': protocol['bridge_lambda'] if method in BRIDGES[3:5] else 0.,
                      'feature_dim': xtr.shape[1], 'parameter_count': model.coef_.size + model.intercept_.size,
                      'final_n_iter': int(model.n_iter_.max()), 'converged': True, 'inner_trials': trials,
                      'inner_best_balanced_accuracy': best[0][0], 'inner_split_seed': split_seed,
                      'inner_train_n': len(it), 'inner_val_n': len(iv), 'outer_train_n': len(tr), 'outer_test_n': len(te),
                      'inner_preprocessing': ia, 'outer_preprocessing': oa, 'outer_test_used_for_selection': False,
                      'protocol_sha256': manifest['protocol_sha256'], 'oof_sha256': digest(op)}
            write_json(out / 'cells' / (tag + '.json'), record)
            if method == BRIDGES[3]:
                b2_model, b2_C = model, C
                arrays.update({'b2_coef': model.coef_, 'b2_intercept': model.intercept_, 'b2_classes': model.classes_})
            print('CELL_DONE', tag, flush=True)
        assert b2_model is not None
        shifted = np.roll(te, 1)
        assert np.all(shifted != te)
        matched = feature_matrix(BRIDGES[3], outer, te, mask, protocol['bridge_lambda'])
        identity = feature_matrix(BRIDGES[3], outer, te, mask, protocol['bridge_lambda'], te)
        wrong = feature_matrix(BRIDGES[3], outer, te, mask, protocol['bridge_lambda'], shifted)
        assert np.array_equal(matched, identity)
        assert np.array_equal(matched[:, :48], wrong[:, :48])
        assert not np.array_equal(matched[:, 48:], wrong[:, 48:])
        tag = f'{prefix}__{BRIDGES[-1]}'
        op = out / 'oof' / (tag + '.npz')
        np.savez_compressed(op, test_indices=te, conditioning_indices=shifted, y_true=y[te],
                            probabilities=b2_model.predict_proba(wrong), classes=b2_model.classes_, seed=seed, fold=fold,
                            endpoint=endpoint, method=BRIDGES[-1], protocol_sha256=manifest['protocol_sha256'])
        write_json(out / 'cells' / (tag + '.json'), {'status': 'PASS', 'method': BRIDGES[-1], 'endpoint': endpoint, 'seed': seed, 'fold': fold,
                   'selected_C': b2_C, 'selected_lambda': protocol['bridge_lambda'], 'feature_dim': 112, 'additional_fits': 0,
                   'same_fitted_weights_as_B2': True, 'pathology_unchanged': True, 'additive_omics_unchanged': True,
                   'identity_permutation_max_probability_error': 0., 'outer_test_used_for_selection': False,
                   'protocol_sha256': manifest['protocol_sha256'], 'oof_sha256': digest(op)})
        np.savez_compressed(pp, **arrays)
        print('BLOCK_DONE', prefix, 'seconds', round(time.monotonic() - started, 2), flush=True)
    check_lock(out)
    write_json(out / f'worker_{worker_id}_DONE.json', {'status': 'PASS', 'worker': worker_id, 'protocol_sha256': manifest['protocol_sha256']})


def metrics(y, p):
    pred = p.argmax(1)
    return {'balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro')),
            'macro_ovr_auroc': float(roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else roc_auc_score(y, p, multi_class='ovr'))}


def paired_bootstrap(y, a, b, seed, reps):
    # Shape seeds x patients x classes. Never treat 15 folds as 15 seed replicates.
    correct = (a.argmax(2) == y[None, :]).astype(float) - (b.argmax(2) == y[None, :]).astype(float)
    rng = np.random.default_rng(seed)
    boot = np.zeros(reps)
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        draws = rng.choice(idx, size=(reps, len(idx)), replace=True)
        boot += correct[:, draws].mean(axis=(0, 2)) / len(np.unique(y))
    point = np.mean([balanced_accuracy_score(y, aa.argmax(1)) - balanced_accuracy_score(y, bb.argmax(1)) for aa, bb in zip(a, b)])
    return float(point), float(np.quantile(boot, .00625)), float(np.quantile(boot, .99375))


def aggregate(out):
    data, manifest = load_private(out)
    protocol = json.loads((out / 'protocol.json').read_text())
    for w in range(protocol['workers']):
        done = json.loads((out / f'worker_{w}_DONE.json').read_text())
        assert done['status'] == 'PASS' and done['protocol_sha256'] == manifest['protocol_sha256']
    expected = len(protocol['seeds']) * len(CLASSES) * 5 * len(METHODS)
    assert len(list((out / 'oof').glob('*.npz'))) == len(list((out / 'cells').glob('*.json'))) == expected
    rows, comparisons, selections = [], [], []
    for endpoint in protocol['endpoints']:
        y = data['y_' + endpoint]
        valid = y >= 0
        preds = {}
        for method in METHODS:
            seed_preds = []
            for seed in protocol['seeds']:
                p = np.full((len(y), len(CLASSES[endpoint])), np.nan)
                counts = np.zeros(len(y), dtype=int)
                for fold in range(1, 6):
                    tag = f'{endpoint}__seed{seed}__fold{fold}__{method}'
                    op = out / 'oof' / (tag + '.npz')
                    cell = json.loads((out / 'cells' / (tag + '.json')).read_text())
                    assert cell['oof_sha256'] == digest(op) and cell['status'] == 'PASS'
                    z = dict(np.load(op, allow_pickle=False))
                    idx = z['test_indices']
                    assert np.array_equal(idx, np.flatnonzero(valid & (data['folds'] == fold)))
                    assert np.array_equal(z['y_true'], y[idx]) and np.isfinite(z['probabilities']).all()
                    p[idx] = z['probabilities']; counts[idx] += 1
                    selections.append({k: cell.get(k) for k in ['seed', 'endpoint', 'fold', 'method', 'selected_C', 'selected_lambda',
                                      'feature_dim', 'parameter_count', 'final_n_iter', 'inner_best_balanced_accuracy']})
                assert (counts[valid] == 1).all() and (counts[~valid] == 0).all()
                seed_preds.append(p[valid])
                rows.append({'seed': seed, 'endpoint': endpoint, 'method': method, 'n': int(valid.sum()), **metrics(y[valid], p[valid])})
            preds[method] = np.stack(seed_preds)
        for name, a, b in PAIRS:
            delta, low, high = paired_bootstrap(y[valid], preds[a], preds[b], 20261009 + int.from_bytes(hashlib.sha256((endpoint + name).encode()).digest()[:4], 'little'), protocol['bootstrap']['replicates'])
            seed_deltas = [balanced_accuracy_score(y[valid], pa.argmax(1)) - balanced_accuracy_score(y[valid], pb.argmax(1)) for pa, pb in zip(preds[a], preds[b])]
            comparisons.append({'endpoint': endpoint, 'comparison': name, 'mean_delta_ba': delta, 'sd_delta_ba': float(np.std(seed_deltas, ddof=1)),
                                'ci98_75_low': low, 'ci98_75_high': high, 'all_seed_delta_positive': all(d > 0 for d in seed_deltas),
                                'seed_deltas': json.dumps(seed_deltas), 'bootstrap_replicates': protocol['bootstrap']['replicates']})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(comparisons).to_csv(out / 'paired_comparisons.csv', index=False)
    pd.DataFrame(selections).to_csv(out / 'selection_summary.csv', index=False)
    frame.groupby(['endpoint', 'method'], as_index=False).agg(mean_oof_ba=('balanced_accuracy', 'mean'), sd_oof_ba=('balanced_accuracy', 'std'), mean_oof_macro_f1=('macro_f1', 'mean'), mean_oof_auroc=('macro_ovr_auroc', 'mean')).to_csv(out / 'aggregate_metrics.csv', index=False)
    val = {'status': 'PASS', 'fitted_outer_cells': expected - 60, 'outer_oof_files': expected, 'true_seed_oof_rows': len(rows),
           'worker_count': protocol['workers'], 'survival_columns_read': [], 'private_patient_predictions_published': False}
    write_json(out / 'validation.json', val)
    write_json(out / 'DONE.json', {'status': 'COMPUTATION_COMPLETE_PENDING_INDEPENDENT_VALIDATION', 'validation': val, 'protocol_sha256': manifest['protocol_sha256']})
    print(json.dumps(val), flush=True)


def self_check():
    rng = np.random.default_rng(51)
    r = rng.lognormal(size=(30, 700))
    r[:, 699] *= 100
    fit = np.arange(20)
    a, keep, st = raw_top512(r, fit)
    manual = np.log2(r[fit] + 1).var(0)
    assert np.array_equal(keep, np.argsort(-manual, kind='stable')[:512])
    changed = r.copy(); changed[20:] *= 100000
    a2, keep2, st2 = raw_top512(changed, fit)
    assert np.array_equal(keep, keep2) and np.array_equal(st['mean'], st2['mean'])
    assert np.array_equal(a[fit], a2[fit])
    pre = {'s': rng.normal(size=(30, 32)), 'rna': rng.normal(size=(30, 8)), 'cna': rng.normal(size=(30, 8)),
           'static_raw': rng.uniform(-.5, .5, size=(30, 32)), 'static_sd': np.ones(32)}
    mask = rng.uniform(size=(32, 8)); mask /= mask.sum(1, keepdims=True)
    rows = np.arange(20, 30)
    x = feature_matrix(BRIDGES[3], pre, rows, mask, .1)
    w = feature_matrix(BRIDGES[3], pre, rows, mask, .1, np.roll(rows, 1))
    assert np.array_equal(x[:, :48], w[:, :48]) and not np.array_equal(x[:, 48:], w[:, 48:])
    assert np.array_equal(x, feature_matrix(BRIDGES[3], pre, rows, mask, .1, rows))
    zero = feature_matrix(BRIDGES[3], pre, rows, mask, 0)
    assert (zero[:, 48:] == 0).all()
    print('SELF_CHECK_PASS: pre-standardization variance; held-out perturbation isolation; conditioning-only mismatch; zero correction; identity pairing')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['prepare', 'worker', 'aggregate', 'self-check'], required=True)
    ap.add_argument('--out', type=Path, default=OUT_DEFAULT)
    ap.add_argument('--worker-id', type=int, default=0)
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()
    if args.mode == 'self-check': self_check(); return
    try:
        if args.mode == 'prepare': prepare(args.out, json.loads(CONFIG.read_text()))
        elif args.mode == 'worker': worker(args.out, args.worker_id, args.workers)
        else: aggregate(args.out)
    except Exception:
        write_json(args.out / f'{args.mode}_{args.worker_id}_FAIL.json', {'status': 'FAIL', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
