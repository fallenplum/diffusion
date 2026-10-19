"""All-gene molecular evidence and original-UNI2 geometry, fit labels isolated.

Only canonical feature keys are read from the central NPZ. RNA is already
log2(x+1); CNA is signed continuous log2 dosage. Neither is logged here. The
frozen Phase26 CP+RNA offsets are reused byte for byte; no estimator is fitted
for those offsets. Full UNI2 rows are streamed from their original H5 files.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
COHORT_NPZ = Path('/root/phase7a_foundation_headroom_v1/private_dataset.npz')
COHORT_SHA256 = '4aaeb4cc22374f32615148605d824abed679fcb957c64286b6a8e44f12a294a1'
PATIENT_SHA256 = '76a144d544b7461d669487625855dc7de65f115a6a2f65d7a20002b2a8e5008f'
SLIDE_ROOT = Path('/root/autodl-tmp/diffusion_data/TCGA')
GMT_PATH = ROOT/'src/phase_m2/metadata/h.all.v2024.1.Hs.symbols.gmt'
GMT_SHA256 = 'ee2463540042078bfa3f67828e1e223bb354446d9fbb4d22845866835ba5c772'
PREPARED_ROOT = Path('/root/autodl-tmp/phase26_multiquery_private/prepared')
MAIN_AUDIT = ROOT/'reports/phase26_multiquery/MAIN_REPLAY_AUDIT.json'
MAIN_SOURCE = 'c4c29b105b3780a11ca8edc32c3a78bde8302ffd59f3c7eefd50a0fa4c13096c'
CHANNELS = ('RNA', 'CNA_gain', 'CNA_loss')
GEOMETRY_SEED = 2801


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value)
    header = json.dumps({'shape': list(a.shape), 'dtype': str(a.dtype)}, sort_keys=True).encode()
    return hashlib.sha256(header+a.tobytes()).hexdigest()


def _indices(value, n, name):
    x = np.asarray(value)
    require(x.ndim == 1 and len(x) and np.issubdtype(x.dtype, np.integer)
            and len(np.unique(x)) == len(x) and (x >= 0).all() and (x < n).all(),
            'invalid '+name+' indices')
    return x.astype(np.int64, copy=True)


def _symbols(value, width, name):
    values = np.asarray(value).astype(str)
    require(values.shape == (width,) and len(set(values)) == width
            and all(s and s.strip() == s and s.upper() != 'NAN' for s in values),
            'canonical '+name+' symbols must already be nonblank and unique')
    return values


def align_all_genes(rna, rna_symbols, cna, cna_symbols):
    """Align existing symbol-collapsed measurements; never compress/select genes.

    The canonical NPZ already merged duplicate symbols by available-measurement
    means. Repeating that operation using its raw20531 axis would misalign the
    existing20511 symbol registry. The sorted union defines immutable GID order.
    """
    r, c = np.asarray(rna), np.asarray(cna)
    require(r.ndim == c.ndim == 2 and len(r) == len(c) and len(r) > 0
            and np.issubdtype(r.dtype, np.number) and np.issubdtype(c.dtype, np.number)
            and not np.isinf(r).any() and not np.isinf(c).any(), 'invalid canonical molecular matrices')
    rs, cs = _symbols(rna_symbols, r.shape[1], 'RNA'), _symbols(cna_symbols, c.shape[1], 'CNA')
    symbols = np.asarray(sorted(set(rs)|set(cs), key=lambda s: s.encode('utf-8')))
    location = {s: k for k, s in enumerate(symbols)}
    ri, ci = np.asarray([location[s] for s in rs]), np.asarray([location[s] for s in cs])
    values = np.zeros((len(r), len(symbols), 3), np.float32)
    measured = np.zeros(values.shape, bool)
    rm, cm = np.isfinite(r), np.isfinite(c)
    values[:, ri, 0] = np.where(rm, r, 0.).astype(np.float32)
    # Crucially split signed dosage before any centering or standardization.
    values[:, ci, 1] = np.where(cm, np.maximum(c, 0.), 0.).astype(np.float32)
    values[:, ci, 2] = np.where(cm, np.maximum(-c, 0.), 0.).astype(np.float32)
    measured[:, ri, 0] = rm
    measured[:, ci, 1] = cm
    measured[:, ci, 2] = cm
    require(np.isfinite(values).all(), 'canonical molecular conversion overflowed float32')
    return values, measured, symbols


def fit_molecular_state(values, measured, fit):
    """Finite measured fit observations only; no medians or held-out statistics."""
    x, m = np.asarray(values), np.asarray(measured)
    require(x.ndim == 3 and x.shape[-1] == 3 and m.shape == x.shape and m.dtype == bool
            and np.isfinite(x[m]).all(), 'invalid molecular evidence/mask')
    fit = _indices(fit, len(x), 'fit')
    count = m[fit].sum(0, dtype=np.int64)
    mean, sd = np.zeros(x.shape[1:], np.float64), np.ones(x.shape[1:], np.float64)
    for channel in range(3):
        observed = m[fit, :, channel]
        finite = np.where(observed, x[fit, :, channel], 0.).astype(np.float64)
        mean[:, channel] = finite.sum(0)/np.maximum(count[:, channel], 1)
        centered = np.where(observed, finite-mean[:, channel], 0.)
        scale = np.sqrt((centered*centered).sum(0)/np.maximum(count[:, channel], 1))
        sd[:, channel] = np.where(scale > 0., scale, 1.)
    require(np.isfinite(mean).all() and np.isfinite(sd).all(), 'nonfinite fit-only molecular statistics')
    return {'mean': mean, 'sd': sd, 'count': count, 'fit_hash': array_hash(fit),
            'fit_n': len(fit), 'population_SD': True}


def transform_molecular(values, measured, state, active=None):
    x, raw_mask = np.asarray(values), np.asarray(measured)
    require(x.ndim == 3 and x.shape[-1] == 3 and raw_mask.shape == x.shape and raw_mask.dtype == bool,
            'invalid molecular transform input')
    mean, sd, count = (np.asarray(state[k]) for k in ('mean', 'sd', 'count'))
    require(mean.shape == sd.shape == count.shape == x.shape[1:] and np.isfinite(mean).all()
            and np.isfinite(sd).all() and (sd > 0).all() and (count >= 0).all(),
            'invalid fit-only molecular state')
    active = np.arange(len(x)) if active is None else _indices(active, len(x), 'active')
    output, mask = np.zeros(x.shape, np.float32), np.zeros(x.shape, bool)
    for channel in range(3):
        keep = raw_mask[active, :, channel] & (count[:, channel] > 0)
        require(np.isfinite(x[active, :, channel][keep]).all(), 'nonfinite measured molecular value')
        standardized = (x[active, :, channel].astype(np.float64)-mean[:, channel])/sd[:, channel]
        output[active, :, channel] = np.where(keep, standardized, 0.).astype(np.float32)
        mask[active, :, channel] = keep
    require(np.isfinite(output).all() and not output[~mask].any(), 'invalid masked standardized evidence')
    return output, mask


def hallmark_membership(path, symbols, expected_sha256=GMT_SHA256):
    """Use the pinned licensed GMT privately; publish hashes/counts only."""
    path = Path(path)
    require(sha256(path) == expected_sha256, 'official Hallmark2024.1 SHA mismatch')
    groups = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            pieces = line.split('\t')
            require(len(pieces) >= 3 and pieces[0].startswith('HALLMARK_'), 'invalid Hallmark GMT row')
            groups.append((pieces[0], set(pieces[2:])))
    require(len(groups) == 50 and len({g[0] for g in groups}) == 50, 'exactly50 named Hallmark sets required')
    symbols = np.asarray(symbols).astype(str)
    omega = np.asarray([[symbol in members for symbol in symbols] for _, members in groups], bool)
    require((omega.sum(1) > 0).all(), 'Hallmark has no canonical gene overlap')
    # Two slots for each identical biological support, then eight all-gene slots.
    slots = np.concatenate((np.repeat(omega, 2, axis=0), np.ones((8, len(symbols)), bool)), axis=0)
    return omega, slots, tuple(name for name, _ in groups)


def read_feature_npz(path=COHORT_NPZ):
    """Do not call historical loaders: they access DSS even for feature checks."""
    path = Path(path)
    require(sha256(path) == COHORT_SHA256, 'immutable canonical cohort NPZ SHA mismatch')
    keys = ('patient_ids', 'folds', 'rna_genes', 'rna_gene_symbols', 'cna_genes', 'cna_gene_symbols', 'cancer_type')
    with np.load(path, allow_pickle=True) as archive:
        result = {key: np.array(archive[key], copy=True) for key in keys}
    ids = result['patient_ids'].astype(str)
    require(len(ids) == len(set(ids)) == 512 and hashlib.sha256('\n'.join(ids).encode()).hexdigest() == PATIENT_SHA256,
            'canonical512 patient order changed')
    require(result['rna_genes'].shape == (512, 20511) and result['cna_genes'].shape == (512, 25101),
            'canonical all-gene dimensions changed')
    require(result['folds'].shape == (512,) and set(result['folds']) == {1, 2, 3, 4, 5}
            and result['cancer_type'].shape == (512,), 'canonical fold/metadata shape changed')
    return result


def read_frozen_main(fold, prepared_root=PREPARED_ROOT, main_audit=MAIN_AUDIT):
    """Only deserialize the original fit labels; valid endpoints are absent."""
    fold = int(fold)
    require(fold in (1, 2, 3), 'only the three locked prototype partitions are available')
    path = Path(prepared_root)/f'prototype_fold{fold}.pt'
    receipt = json.loads(path.with_suffix('.json').read_text(encoding='utf-8-sig'))
    digest = sha256(path)
    require(receipt.get('status') == 'COMPLETE' and receipt.get('source') == MAIN_SOURCE
            and receipt.get('stage') == 'prototype' and receipt.get('fold') == fold
            and receipt.get('sha256') == digest, 'frozen Phase26 prepared source/receipt/SHA mismatch')
    evidence = json.loads(Path(main_audit).read_text(encoding='utf-8-sig'))
    require(evidence.get('status') == 'PASS_NO_FIT_MAIN_REPLAY' and evidence.get('source') == MAIN_SOURCE
            and evidence.get('prepared_sha256_by_fold', {}).get(str(fold)) == digest
            and evidence.get('maximum_absolute_risk_error', np.inf) <= 1e-10
            and evidence.get('endpoint_API_reads') == evidence.get('new_predictor_fits') == 0,
            'independent frozen main replay evidence mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    require(saved.get('source') == MAIN_SOURCE and saved.get('stage') == 'prototype' and saved.get('fold') == fold,
            'frozen main payload identity changed')
    require(not set(saved) & {'valid_time', 'valid_event', 'test_time', 'test_event', 'DSS', 'endpoints'},
            'held-out supervision escaped into prepared main')
    n = len(saved['base'])
    fit, valid = _indices(saved['fit'], n, 'fit'), _indices(saved['valid'], n, 'valid')
    active = _indices(saved['active'], n, 'active')
    require(not np.intersect1d(fit, valid).size and np.array_equal(active, np.union1d(fit, valid)),
            'frozen main partition overlap or active scope mismatch')
    base = np.asarray(saved['base'], np.float64)
    require(base.shape == (n,) and np.isfinite(base[active]).all(), 'nonfinite frozen active offset')
    refs = {key: np.asarray(saved[key], np.float64).copy() for key in
            ('CP_cf_fit', 'RNA_cf_fit', 'CP_valid', 'RNA_valid', 'M0_valid')}
    require(refs['CP_cf_fit'].shape == refs['RNA_cf_fit'].shape == fit.shape
            and refs['CP_valid'].shape == refs['RNA_valid'].shape == refs['M0_valid'].shape == valid.shape
            and all(np.isfinite(v).all() for v in refs.values())
            and np.array_equal(base[fit], refs['CP_cf_fit']+refs['RNA_cf_fit'])
            and np.array_equal(base[valid], refs['M0_valid'])
            and np.array_equal(refs['M0_valid'], refs['CP_valid']+refs['RNA_valid']),
            'frozen CP+RNA risk sum or reference identity changed')
    time, event = np.asarray(saved['fit_time'], np.float64), np.asarray(saved['fit_event'], np.float64)
    require(time.shape == event.shape == fit.shape and np.isfinite(time).all() and (time >= 0).all()
            and np.isin(event, (0, 1)).all() and event.sum() > 0, 'invalid saved fit-only labels')
    return {'fit': fit, 'valid': valid, 'active': active, 'base': base.copy(), **refs,
            '_fit_time': time.copy(), '_fit_event': event.copy(), 'prepared_sha256': digest,
            'main_evidence_sha256': sha256(main_audit), 'legacy_prepared_sha256': saved['legacy_prepared_sha256']}


def _h5_shapes(handle):
    require('features' in handle and 'coords' in handle, 'native UNI2 H5 requires features and original coordinates')
    features, coordinates = handle['features'], handle['coords']
    require(features.ndim in (2, 3) and np.issubdtype(features.dtype, np.floating) and features.shape[-1] == 1536
            and (features.ndim == 2 or features.shape[0] == 1), 'native UNI2 feature shape mismatch')
    n = features.shape[-2]
    require(n > 0 and coordinates.shape in ((n, 2), (1, n, 2))
            and np.issubdtype(coordinates.dtype, np.integer), 'native UNI2 row/coordinate shape mismatch')
    for key, value in (('patch_size_level0', 512), ('target_magnification', 20), ('level0_magnification', 40)):
        require(coordinates.attrs.get(key) == value, 'native UNI2 coordinate scale mismatch: '+key)
    return n


def slide_catalog(root, patient_ids):
    """Private full-slide catalogue. Do not publish paths or per-patient rows."""
    ids = np.asarray(patient_ids).astype(str)
    require(len(set(ids)) == len(ids), 'nonunique patient IDs')
    lookup = {pid: i for i, pid in enumerate(ids)}
    rows, seen = [], set()
    for path in Path(root).glob('*/*.h5'):
        owner = lookup.get(path.stem[:12])
        if owner is None:
            continue
        require(path.stem not in seen and not path.is_symlink(), 'duplicate/symlink native H5 source')
        seen.add(path.stem)
        metadata = path.stat()
        with h5py.File(path, 'r') as handle:
            n = _h5_shapes(handle)
        rows.append({'path': str(path), 'owner': owner, 'patches': n,
                     'size': metadata.st_size, 'mtime_ns': metadata.st_mtime_ns})
    rows.sort(key=lambda row: (row['owner'], Path(row['path']).name))
    require(set(row['owner'] for row in rows) == set(range(len(ids))), 'a canonical patient has no complete native UNI2 H5')
    for index, row in enumerate(rows):
        row['index'] = index
    return rows


def _read_selected(handle, selected):
    """H5 fancy indexing is ascending; invert it back to geometry-plan order."""
    n = _h5_shapes(handle)
    selected = _indices(selected, n, 'geometry selected row')
    order = np.argsort(selected, kind='stable')
    sorted_rows = selected[order]
    source = handle['features']
    sorted_values = source[sorted_rows, :] if source.ndim == 2 else source[0, sorted_rows, :]
    inverse = np.empty(len(order), np.int64)
    inverse[order] = np.arange(len(order))
    values = np.asarray(sorted_values[inverse], np.float32)
    require(values.shape == (len(selected), 1536) and np.isfinite(values).all(), 'nonfinite selected native UNI2 evidence')
    return values


class Data:
    """No new main-effect fits, no legacy Data/512-Patch view, no valid labels.

    Molecular tensors and streamed Patch tensors default to CPU. Callers move
    only the current patient's evidence to the model device. Optional bounded
    slide LRU is RAM-only; original feature arrays are never copied to disk.
    """
    def __init__(self, fold, seed, device='cpu', *, cohort_npz=COHORT_NPZ, slide_root=SLIDE_ROOT,
                 gmt_path=GMT_PATH, prepared_root=PREPARED_ROOT, main_audit=MAIN_AUDIT,
                 cache_slides=0):
        require(not isinstance(cache_slides, bool) and int(cache_slides) == cache_slides and cache_slides >= 0,
                'cache_slides must be a nonnegative integer')
        self.fold, self.seed, self.stage = int(fold), int(seed), 'prototype'
        self.device = torch.device(device)
        raw = read_feature_npz(cohort_npz)
        main = read_frozen_main(fold, prepared_root, main_audit)
        require(len(main['base']) == len(raw['patient_ids']), 'frozen-main/canonical-cohort length mismatch')
        self.fit, self.valid, self.active = (main[k] for k in ('fit', 'valid', 'active'))
        self.patient_ids = raw['patient_ids'].astype(str)
        self.cancer_type = raw['cancer_type'].astype(str)
        self.canonical_folds = np.asarray(raw['folds'], np.int64)
        values, observed, symbols = align_all_genes(raw['rna_genes'], raw['rna_gene_symbols'],
                                                   raw['cna_genes'], raw['cna_gene_symbols'])
        require(len(symbols) == 26708, 'canonical all-gene union changed')
        self.gene_symbols = symbols  # private registry; GID itself is the tensor index
        self.molecular_state = fit_molecular_state(values, observed, self.fit)
        standardized, effective_mask = transform_molecular(values, observed, self.molecular_state, self.active)
        self.molecular = torch.as_tensor(standardized, device=self.device)
        self.molecular_mask = torch.as_tensor(effective_mask, device=self.device)
        self.gene_ids = torch.arange(len(symbols), dtype=torch.long, device=self.device)
        omega, slot_membership, self.pathway_names = hallmark_membership(gmt_path, symbols)
        self.omega = torch.as_tensor(omega, device=self.device)
        self.slot_membership = torch.as_tensor(slot_membership, device=self.device)
        self.base = torch.as_tensor(main['base'], dtype=torch.float64, device=self.device)
        for key in ('CP_cf_fit', 'RNA_cf_fit', 'CP_valid', 'RNA_valid', 'M0_valid'):
            setattr(self, key, main[key])
        self.R0_valid = self.M0_valid.copy()
        self._fit_time, self._fit_event = main['_fit_time'], main['_fit_event']
        self.slide_catalog = slide_catalog(slide_root, self.patient_ids)
        self.patient_slides = [[] for _ in range(len(self.patient_ids))]
        for row in self.slide_catalog:
            self.patient_slides[row['owner']].append(row['index'])
        self._plans, self._slide_cache, self._cache_slides = {}, OrderedDict(), int(cache_slides)
        self.prepared_sha256 = main['prepared_sha256']
        counts = self.molecular_state['count']
        self.audit = {'version': 'phase28_allgene_original_H5_fit_only_v1', 'patients': len(self.patient_ids),
            'RNA_genes': raw['rna_genes'].shape[1], 'CNA_genes': raw['cna_genes'].shape[1], 'union_genes': len(symbols),
            'molecular_shape': list(self.molecular.shape), 'gene_order': 'UTF8_sorted_canonical_HGNC_symbol_union',
            'gene_order_sha256': hashlib.sha256('\n'.join(symbols).encode()).hexdigest(),
            'RNA_already_log2_no_second_log': True, 'CNA_gain_loss_before_centering': True,
            'mask_before_imputation': True, 'channels': list(CHANNELS),
            'fit_zero_count_channels': {k: int((counts[:, j] == 0).sum()) for j, k in enumerate(CHANNELS)},
            'active_effective_missing': {k: int((~effective_mask[self.active, :, j]).sum()) for j, k in enumerate(CHANNELS)},
            'zero_fit_count_disabled_for_heldout': True, 'fit_statistics_hash': array_hash(np.stack(
                (self.molecular_state['mean'], self.molecular_state['sd'], counts), axis=0)),
            'fit_indices_hash': array_hash(self.fit), 'valid_indices_hash': array_hash(self.valid),
            'fit_statistics_patients': len(self.fit), 'valid_statistics_used': False,
            'pathways': 50, 'slots': 108, 'two_slots_share_same_pathway_support': True,
            'residual_allgene_slots': 8, 'pathway_gene_counts': omega.sum(1).tolist(), 'GMT_sha256': GMT_SHA256,
            'GMT_licensed_gene_membership_public': False, 'cohort_source_sha256': COHORT_SHA256,
            'prepared_source': MAIN_SOURCE, 'prepared_sha256': self.prepared_sha256,
            'main_evidence_sha256': main['main_evidence_sha256'], 'main_effect_new_fits': 0,
            'main_offset': 'exact_saved_Phase26_CP_plus_independent_RNA;fit_CF;valid_fullfit',
            'Clinical': 'legal_structured_clinical_already_in_frozen_external_CP;no_new_Clinical_encoder',
            'slides': len(self.slide_catalog), 'native_patch_count': sum(r['patches'] for r in self.slide_catalog),
            'patch_feature_dim': 1536, 'native_H5_singleton_axis_supported': True,
            'old_512_candidate_cache_used': False, 'new_raw_feature_disk_cache': False,
            'geometry_seed': '2801+stable_full_cohort_slide_index;independent_of_fold_seed_epoch',
            'fit_label_bytes_deserialized': True, 'valid_endpoint_loaded': False,
            'all_unobserved_value_requires_effective_measurement_mask': True}

    def fit_labels(self):
        return self._fit_time.copy(), self._fit_event.copy()

    def _plan(self, index, coordinates):
        if index not in self._plans:
            from .spatial import build_plan
            self._plans[index] = build_plan(coordinates, step=512, tile_size=16, max_regions=64,
                                             seed=GEOMETRY_SEED+index)
        return self._plans[index]

    def load_slide(self, index, device=None):
        require(not isinstance(index, bool) and int(index) == index and 0 <= index < len(self.slide_catalog),
                'invalid native slide index')
        index = int(index)
        if index in self._slide_cache:
            result = self._slide_cache.pop(index)
            self._slide_cache[index] = result
        else:
            row = self.slide_catalog[index]
            path = Path(row['path'])
            before = path.stat()
            require(before.st_size == row['size'] and before.st_mtime_ns == row['mtime_ns'],
                    'native H5 metadata changed after catalogue')
            with h5py.File(path, 'r') as handle:
                n = _h5_shapes(handle)
                coordinates = np.asarray(handle['coords']).reshape(-1, 2)
                require(len(coordinates) == n and np.isfinite(coordinates).all() and (coordinates >= 0).all(),
                        'invalid original native coordinates')
                plan = self._plan(index, coordinates)
                selected = _indices(plan['selected_indices'], n, 'plan row')
                require(len(selected) <= 64*16*16, 'geometry exceeded locked region/Patch capacity')
                values = _read_selected(handle, selected)
            after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    'native H5 changed during streaming')
            result = {'features': torch.from_numpy(values), 'plan': plan, 'owner': row['owner'], 'index': index}
            if self._cache_slides:
                self._slide_cache[index] = result
                while len(self._slide_cache) > self._cache_slides:
                    self._slide_cache.popitem(last=False)
        # Keep an optional RAM cache immutable from a caller's in-place edits.
        if device is None:
            return {**result, 'features': result['features'].clone()} if self._cache_slides else result
        return {**result, 'features': result['features'].to(device=device, copy=True)}

    def load_patient_slides(self, patient, device=None):
        require(not isinstance(patient, bool) and int(patient) == patient and 0 <= patient < len(self.patient_ids),
                'invalid patient index')
        return [self.load_slide(index, device=device) for index in self.patient_slides[int(patient)]]
