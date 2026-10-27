"""Read-only, coordinate-aware inputs for the locked BG-BEN experiment.

Private asset locations are supplied by configuration or environment variables.
This module fits no preprocessing estimator and does not select genes. Endpoint
reads are explicit; patch catalogues and geometry never require outcomes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

CLINICAL_COLUMNS = ('AGE', 'SEX', 'AJCC_PATHOLOGIC_TUMOR_STAGE',
                    'PATH_T_STAGE', 'PATH_N_STAGE', 'PATH_M_STAGE')
CANONICAL_COHORT_SHA256 = '4aaeb4cc22374f32615148605d824abed679fcb957c64286b6a8e44f12a294a1'
CANONICAL_PATIENT_SHA256 = '76a144d544b7461d669487625855dc7de65f115a6a2f65d7a20002b2a8e5008f'
CANONICAL_ENDPOINT_SHA256 = '91b5d15aa32489c825a8341aed8d9733281dc62386151c8c6246b6312975fd4e'
CANONICAL_VISUAL_SOURCES_SHA256 = '4898df75eb03539e85ef394772fe2bda991bfbe325154cc3241426e168c39194'
PUBLIC_GEOMETRY_KEYS = frozenset(('patch_size', 'patch_size_level0', 'target_magnification',
                                 'level0_magnification', 'overlap', 'downsample', 'mpp',
                                 'level', 'read_level', 'target_mpp'))


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    header = json.dumps({'dtype': str(value.dtype), 'shape': list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def _asset(mapping, key, *, required=True):
    value = mapping.get(key, mapping.get('BGBEN_' + key.upper(), os.environ.get('BGBEN_' + key.upper())))
    if value is None:
        require(not required, 'required asset configuration missing: ' + key)
        return None
    path = Path(value).expanduser()
    require(path.is_file() if key not in ('conch_root', 'uni2_root') else path.is_dir(),
            'configured asset is not readable: ' + key)
    return path


def load_raw(envmap=None, *, include_endpoints=True):
    """Read canonical covariates, with an explicit and separate endpoint read.

    ``clinical_raw`` is a mapping of the six original legal clinical fields to
    string arrays; it is deliberately not encoded here. ``platform`` is only a
    missing-pattern proxy, not a verified sequencing-platform identity. All
    returned arrays are private and must never be published.
    """
    mapping = {} if envmap is None else dict(envmap)
    path = _asset(mapping, 'cohort')
    digest = sha256(path)
    expected = mapping.get('cohort_sha256', CANONICAL_COHORT_SHA256)
    require(digest == expected, 'canonical cohort SHA256 mismatch')
    keys = ['patient_ids', 'folds', 'static', 'titan', 'uni2', 'rna_genes', 'rna_gene_symbols',
            'cna_genes', 'cna_gene_symbols', 'cancer_type', 'raw_rna']
    keys += ['clinical_' + name for name in CLINICAL_COLUMNS]
    with np.load(path, allow_pickle=True) as archive:
        require(all(key in archive.files for key in keys), 'canonical covariate key missing')
        source = {key: np.array(archive[key], copy=True) for key in keys}
    ids = source['patient_ids'].astype(str)
    n = len(ids)
    patient_digest = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    require(n == 512 and len(np.unique(ids)) == n and patient_digest == CANONICAL_PATIENT_SHA256,
            'canonical patient order mismatch')
    require(source['rna_genes'].shape == (n, 20511) and source['cna_genes'].shape == (n, 25101),
            'canonical all-gene matrix dimensions mismatch')
    require(source['folds'].shape == (n,) and set(source['folds']) == {1, 2, 3, 4, 5},
            'canonical outer patient folds mismatch')
    for key, width in [('static', 32), ('titan', 768), ('uni2', 1536)]:
        require(source[key].shape == (n, width) and np.isfinite(source[key]).all(),
                key + ' dimension/finite mismatch')
    for name in ('rna', 'cna'):
        symbols = source[name + '_gene_symbols'].astype(str)
        require(len(symbols) == source[name + '_genes'].shape[1]
                and len(np.unique(symbols)) == len(symbols) and (symbols != '').all(),
                name + ' symbol registry mismatch')
        require(not np.isinf(source[name + '_genes']).any(), name + ' contains infinite measurements')
    rna_missing = ~np.isfinite(source['raw_rna'])
    # Keep exact measurement patterns as private grouping covariates. No
    # outcome, cancer type or stage contributes to these pattern labels.
    _, patterns = np.unique(rna_missing, axis=0, return_inverse=True)
    result = {'patient_ids': ids, 'folds': source['folds'].astype(np.int64),
              'clinical_columns': list(CLINICAL_COLUMNS),
              'clinical_raw': {name: source['clinical_' + name].astype(str) for name in CLINICAL_COLUMNS},
              'S32': source['static'], 'titan': source['titan'], 'uni2_mean': source['uni2'],
              'rna': source['rna_genes'], 'rna_symbols': source['rna_gene_symbols'].astype(str),
              'cna': source['cna_genes'], 'cna_symbols': source['cna_gene_symbols'].astype(str),
              'platform': patterns, 'cancer': source['cancer_type'].astype(str),
              'rna_measured': np.isfinite(source['rna_genes']),
              'cna_measured': np.isfinite(source['cna_genes']),
              'audit': {'cohort_sha256': digest, 'patient_order_sha256': patient_digest,
                        'endpoint_read': False, 'preprocessing_fits': 0,
                        'RNA_already_log2': True, 'CNA_signed_log2_dosage': True,
                        'platform_field': 'exact raw-RNA missing-pattern proxy; platform identity unverified'}}
    visual = _asset(mapping, 'visual_sources', required=False)
    if visual is not None:
        require(sha256(visual) == mapping.get('visual_sources_sha256', CANONICAL_VISUAL_SOURCES_SHA256),
                'audited frozen visual-source SHA256 mismatch')
        with np.load(visual, allow_pickle=False) as archive:
            for key in ('LOCAL', 'GLOBAL_MATCHED', 'GLOBAL_ALL'):
                values = np.asarray(archive[key], np.float64)
                require(values.shape == (n, 768) and np.isfinite(values).all(), 'visual mean source mismatch')
                result['conch_' + key.lower()] = values.copy()
        result['conch_mean'] = result['conch_global_all']
        result['audit']['visual_sources_sha256'] = sha256(visual)
        result['audit']['conch_mean_scope'] = 'all original CONCH candidates; equal slide means per patient'
    if include_endpoints:
        endpoint_path = _asset(mapping, 'endpoints')
        require(sha256(endpoint_path) == mapping.get('endpoints_sha256', CANONICAL_ENDPOINT_SHA256),
                'audited endpoint-source SHA256 mismatch')
        with np.load(endpoint_path, allow_pickle=False) as archive:
            require(str(archive['patient_hash']) == hashlib.sha256(np.ascontiguousarray(source['patient_ids']).tobytes()).hexdigest(),
                    'endpoint canonical patient ordering mismatch')
            for endpoint in ('DSS', 'PFI'):
                fields = {key: np.array(archive[endpoint + '_' + key], copy=True) for key in ('time', 'event', 'valid')}
                valid = fields['valid'].astype(bool)
                require(all(v.shape == (n,) for v in fields.values()) and np.isfinite(fields['time'][valid]).all()
                        and (fields['time'][valid] >= 0).all() and np.isin(fields['event'][valid], (0, 1)).all(),
                        endpoint + ' endpoint contract mismatch')
                prefix = '' if endpoint == 'DSS' else 'pfi_'
                result[prefix + 't'], result[prefix + 'e'] = fields['time'], fields['event']
                result[prefix + 'valid'] = valid
        require(result['valid'].all() and int(result['e'].sum()) == 65, 'DSS cohort eligibility mismatch')
        result['audit'].update(endpoint_read=True, endpoints_sha256=sha256(endpoint_path), time_unit='months')
    return result


def outer_partition(folds, fold):
    folds = np.asarray(folds)
    require(folds.ndim == 1 and int(fold) in set(folds), 'invalid outer fold')
    return np.flatnonzero(folds != int(fold)), np.flatnonzero(folds == int(fold))


def inner_partitions(folds, event, *, outer_fold=1, split_seed=270001, n_splits=3):
    """Three fixed stratified partitions entirely inside one outer-train.

    Outer endpoints cannot affect assignments: only the outer-training event
    vector is supplied to StratifiedKFold. Every fit/valid row remains disjoint
    from the sealed outer-test set.
    """
    from sklearn.model_selection import StratifiedKFold
    fit, test = outer_partition(folds, outer_fold)
    event = np.asarray(event)
    require(event.shape == np.asarray(folds).shape and np.isin(event[fit], (0, 1)).all(), 'invalid inner event stratum')
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(split_seed))
    result = []
    for index, (a, b) in enumerate(splitter.split(fit, event[fit])):
        train, valid = fit[a], fit[b]
        require(not np.intersect1d(train, valid).size and not np.intersect1d(np.r_[train, valid], test).size,
                'inner/outer patient leakage')
        result.append({'partition': index, 'fit': train.copy(), 'valid': valid.copy(), 'outer_test': test.copy()})
    return result


def canonical_order(coords):
    """Geometry order, invariant to H5 row permutation; x ties ordered by y."""
    xy = np.asarray(coords)
    require(xy.ndim == 2 and xy.shape[1] == 2 and len(xy), 'coordinates require nonempty [N,2]')
    require(np.isfinite(xy).all() and (xy >= 0).all(), 'missing or negative coordinates')
    require(len(np.unique(xy, axis=0)) == len(xy), 'duplicate coordinates')
    return np.lexsort((xy[:, 0], xy[:, 1]))


def coordinate_step(coords):
    """Infer the original regular-grid hop from native XY, not patch size.

    The GCD uses all distinct coordinate differences on both axes. It remains
    valid when tissue masks remove intermediate grid locations. A coordinate
    set with no varying axis cannot identify a hop and must be rejected.
    """
    xy = np.asarray(coords)
    canonical_order(xy)
    require(np.issubdtype(xy.dtype, np.integer), 'grid hop requires native integer XY')
    deltas = [np.diff(np.unique(xy[:, axis])) for axis in range(2)]
    deltas = np.concatenate([value[value > 0] for value in deltas])
    require(len(deltas), 'one-location slide cannot identify a coordinate hop')
    hop = int(np.gcd.reduce(deltas))
    require(hop > 0 and ((xy-xy[0]) % hop == 0).all(), 'native coordinate hop/phase mismatch')
    return hop


def patient_quotas(slides, budget, n_patients):
    """Allocate a patient budget across actual slides using relative capacity."""
    require(int(budget) == budget and budget > 0, 'positive patient patch budget required')
    quota = np.zeros(len(slides), np.int64)
    for patient in range(n_patients):
        ix = np.array([j for j, row in enumerate(slides) if int(row['patient']) == patient], np.int64)
        require(len(ix), 'a patient has no patch slide')
        capacity = np.array([int(slides[j]['n']) for j in ix])
        require((capacity > 0).all(), 'empty patch slide')
        require(int(budget) >= len(ix), 'patient budget cannot retain all actual slides')
        used = np.ones(len(ix), np.int64)
        target = min(int(budget), int(capacity.sum()))
        while int(used.sum()) < target:
            priority = np.where(used < capacity, capacity / (used + 1), -np.inf)
            used[int(np.argmax(priority))] += 1
        quota[ix] = used
    return quota


def select_patch_rows(features, coords, count, *, selector='GRID', seed=0):
    """Outcome-blind patch selectors with a coordinate-canonical starting order.

    GRID is deterministic farthest-point coverage in normalized true XY.
    DIVERSE uses a fixed random feature projection plus the same XY. Neither
    selector treats UNI2 and CONCH row indices as a shared patch registry.
    """
    x, xy = np.asarray(features), np.asarray(coords)
    require(x.ndim == 2 and len(x) == len(xy) and np.isfinite(x).all(), 'invalid patch feature matrix')
    order = canonical_order(xy)
    require(0 < int(count) <= len(x) and int(count) == count, 'invalid selected patch count')
    if selector == 'REGIONS':
        return select_region_rows(xy, int(count))[0]
    if selector == 'RAND':
        return order[np.random.default_rng(int(seed)).permutation(len(order))[:int(count)]]
    require(selector in ('GRID', 'DIVERSE'), 'unknown patch selector')
    if int(count) == len(x):
        return order
    pos = xy[order].astype(np.float64)
    pos = (pos - pos.min(0)) / np.maximum(np.ptp(pos, axis=0), 1.)
    if selector == 'DIVERSE':
        projection = np.random.default_rng(270002).normal(size=(x.shape[1], 16)) / 4.
        feature = np.asarray(x[order], np.float64) @ projection
        norm = np.linalg.norm(feature, axis=1, keepdims=True)
        feature = feature / np.maximum(norm, 1e-12)
        pos = np.c_[feature, .1 * pos]
    center = pos.mean(0)
    current = int(np.argmin(((pos - center)**2).sum(1)))
    distance = np.full(len(pos), np.inf)
    selected = []
    for _ in range(int(count)):
        selected.append(current)
        distance = np.minimum(distance, ((pos - pos[current])**2).sum(1))
        distance[np.asarray(selected)] = -np.inf
        current = int(np.argmax(distance))
    # Encode scans from true XY rather than from the order of FPS discovery.
    chosen = order[np.asarray(selected)]
    return chosen[np.lexsort((xy[chosen, 0], xy[chosen, 1]))]


def select_region_rows(coords, count, *, max_window=64):
    """Spatially dispersed seeds, each expanded only along real four-neighbor XY.

    Windows retain at most 64 original patches. Selection is independent of H5
    row order and all outcomes. Returned window IDs require a recurrent reset
    at every window, in addition to actual row/column gaps and slide boundaries.
    """
    xy = np.asarray(coords)
    order = canonical_order(xy)
    require(0 < count <= len(xy) and max_window > 0, 'invalid region patch budget')
    hop = coordinate_step(xy)
    pos = xy[order].astype(np.int64)
    grid = (pos-pos[0])//hop
    lookup = {tuple(point): index for index, point in enumerate(grid)}
    remaining = np.ones(len(pos), bool)
    distance = np.full(len(pos), np.inf)
    centroid = pos.astype(np.float64).mean(0)
    seed = int(np.argmin(((pos-centroid)**2).sum(1)))
    windows, chosen, window_number = [], [], 0
    while len(chosen) < count:
        cap = min(int(max_window), int(count)-len(chosen))
        queue, members, enqueued = [seed], [], {seed}
        for at in queue:
            if not remaining[at]:
                continue
            members.append(at)
            if len(members) == cap:
                break
            x, y = grid[at]
            for point in ((x-1,y),(x,y-1),(x+1,y),(x,y+1)):
                neighbor = lookup.get(point)
                if neighbor is not None and remaining[neighbor] and neighbor not in enqueued:
                    queue.append(neighbor); enqueued.add(neighbor)
        require(members, 'region BFS made no progress')
        remaining[members] = False
        chosen.extend(members)
        windows.extend([window_number] * len(members))
        window_number += 1
        distance = np.minimum(distance, ((pos-pos[seed])**2).sum(1).astype(np.float64))
        distance[~remaining] = -np.inf
        if len(chosen) < count:
            seed = int(np.argmax(distance))
    chosen, windows = np.asarray(chosen, np.int64), np.asarray(windows, np.int64)
    at = np.lexsort((pos[chosen,0], pos[chosen,1]))
    return order[chosen[at]], windows[at]


def read_h5_slide(path, *, feature_dim=None, normalize=True):
    """Original H5 rows and their own coordinates; support UNI2 singleton axis."""
    with h5py.File(path, 'r') as handle:
        require('features' in handle and 'coords' in handle, 'patch H5 missing features or coordinates')
        x = np.asarray(handle['features'], np.float32)
        xy = np.asarray(handle['coords'])
        attrs = {str(k): v.item() if isinstance(v, np.generic) else v for k, v in handle['coords'].attrs.items()}
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if xy.ndim == 3 and xy.shape[0] == 1:
        xy = xy[0]
    require(x.ndim == 2 and x.shape[1] in (768, 1536) and (feature_dim is None or x.shape[1] == feature_dim),
            'unsupported native patch feature shape')
    require(np.issubdtype(xy.dtype, np.integer), 'native patch coordinates must be integer pixels')
    canonical_order(xy)
    require(len(x) == len(xy) and np.isfinite(x).all(), 'patch/coordinate alignment or finite mismatch')
    norm = np.linalg.norm(x, axis=1)
    require((norm > 1e-12).all(), 'zero patch feature norm')
    if normalize:
        x = x / norm[:, None]
    return {'features': x, 'coords': xy.astype(np.int64), 'coordinate_attrs': attrs,
            'feature_dim': x.shape[1], 'normalized': bool(normalize)}


def read_candidate_cache(manifest_path, *, feature_override=None, coords_override=None, verify_hash=True):
    """Read the feature-only Phase16 full candidate cache with geometry binding."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding='utf-8-sig'))
    feature_path = Path(feature_override) if feature_override else Path(manifest['features'])
    coord_path = Path(coords_override) if coords_override else path.parent / 'coords.npy'
    if verify_hash:
        require(sha256(feature_path) == manifest['cache_sha256'], 'candidate feature cache hash mismatch')
        require(sha256(coord_path) == manifest['coordinate_sha256'], 'candidate coordinate hash mismatch')
    x, coords = np.load(feature_path, mmap_mode='r'), np.load(coord_path, mmap_mode='r')
    require(x.shape == (int(manifest['patches']), 768) and coords.shape == (len(x), 2), 'full candidate shape mismatch')
    slides = manifest['slides']
    require(sum(int(row['n']) for row in slides) == len(x), 'slide row coverage mismatch')
    cursor = 0
    for row in slides:
        require(int(row['offset']) == cursor and int(row['n']) > 0, 'candidate slide offsets are not contiguous')
        cursor += int(row['n'])
    require(set(int(row['patient']) for row in slides) == set(range(int(manifest['patients']))),
            'candidate patient coverage mismatch')
    return {'features': x, 'coords': coords, 'slides': slides, 'manifest': manifest,
            'manifest_sha256': sha256(path), 'feature_path': feature_path, 'coord_path': coord_path}


def build_patch_cache(candidate, *, budget=512, selector='GRID', seed=0):
    """Build slide-padded CPU tensors; retain explicit patient and slide ownership.

    Output ``features`` is [slides,max_quota,768], ``coords`` [slides,max_quota,2],
    ``mask`` [slides,max_quota], and ``owner`` [slides]. Patient pooling must first
    pool within a slide then weight that patient's slides equally.
    """
    slides = candidate['slides']
    quota = patient_quotas(slides, int(budget), int(candidate['manifest']['patients']))
    width = int(quota.max())
    features = np.zeros((len(slides), width, 768), np.float32)
    coords = np.zeros((len(slides), width, 2), np.int64)
    mask = np.zeros((len(slides), width), bool)
    rowids = np.full((len(slides), width), -1, np.int64)
    owner = np.asarray([int(row['patient']) for row in slides], np.int64)
    for index, row in enumerate(slides):
        start, count = int(row['offset']), int(row['n'])
        x = candidate['features'][start:start + count]
        xy = candidate['coords'][start:start + count]
        selected = select_patch_rows(x, xy, int(quota[index]), selector=selector, seed=int(seed) + index * 104729)
        n = len(selected)
        features[index, :n] = x[selected]
        coords[index, :n] = xy[selected]
        mask[index, :n] = True
        rowids[index, :n] = start + selected
    require(int(mask.sum()) == int(quota.sum()), 'selected patch coverage mismatch')
    return {'features': features, 'coords': coords, 'mask': mask, 'owner': owner, 'rowids': rowids,
            'quota': quota, 'audit': {'source_manifest_sha256': candidate['manifest_sha256'],
            'budget_per_patient': int(budget), 'selector': selector, 'selector_seed': int(seed),
            'selected_patches': int(mask.sum()), 'slides': len(slides),
            'patient_pool': 'mean within slide then equal slide weighting',
            'outcome_inputs': False, 'new_preprocessing_fits': 0}}


def native_conch_catalog(inventory_path, *, root_override=None):
    """Recover original CONCH sources after a volatile cache was lost.

    The historical source inventory binds each actual H5 to a canonical patient
    index and SHA256. A relocation may change the source root, never the relative
    TCGA cancer directory and filename or the scientific source identity.
    """
    inventory = json.loads(Path(inventory_path).read_text(encoding='utf-8-sig'))
    require(not inventory.get('errors') and not inventory.get('missing_conch'), 'historical CONCH inventory incomplete')
    source = inventory['chosen_conch']
    require(len(source) == 519, 'expected canonical CONCH slide coverage')
    rows = []
    for row in sorted(source, key=lambda r: (int(r['patient_index']), str(r['stem']))):
        path = Path(row['path'])
        if root_override:
            path = Path(root_override) / path.parent.name / path.name
        require(path.is_file() and not path.is_symlink(), 'original CONCH H5 unavailable')
        require(path.stat().st_size == int(row['bytes']), 'original CONCH H5 byte count changed')
        require(int(row['dim']) == 768 and int(row['patches']) > 0, 'CONCH source shape mismatch')
        rows.append({'path': path, 'patient': int(row['patient_index']), 'n': int(row['patches']),
                     'sha256': row['sha256'], 'bytes': int(row['bytes']), 'geometry': row['coordinate_attrs']})
    require(set(row['patient'] for row in rows) == set(range(512)), 'CONCH patient coverage changed')
    return rows


def prepare_native_patch_cache(inventory_path, output, *, budget=512, selector='GRID', seed=0,
                               root_override=None):
    """Stream raw H5 once per slide into a small, independent feature-only cache.

    Original files stay read-only. One selected cache is scientifically one
    fixed selector/budget/seed; no RNA, clinical or outcome values are read.
    Its manifest and row IDs remain private. Full-source mean is retained as a
    fair all-candidate pooling reference before any patch subsampling.
    """
    output = Path(output)
    require(not output.exists(), 'refuse to overwrite a patch cache or failed partial output')
    rows = native_conch_catalog(inventory_path, root_override=root_override)
    quota = patient_quotas(rows, budget, 512)
    output.mkdir(parents=True, exist_ok=False)
    shape = (len(rows), int(quota.max()), 768)
    features = np.lib.format.open_memmap(output/'features.npy', mode='w+', dtype=np.float32, shape=shape)
    features[:] = 0
    coords = np.zeros(shape[:2]+(2,), np.int64)
    mask = np.zeros(shape[:2], bool)
    rowids = np.full(shape[:2], -1, np.int64)
    region_ids = np.full(shape[:2], -1, np.int64)
    owner = np.asarray([row['patient'] for row in rows], np.int64)
    steps = np.zeros(len(rows), np.int64)
    full_slide_mean = np.zeros((len(rows), 768), np.float64)
    selected_slide_mean = np.zeros_like(full_slide_mean)
    patch_count, attrs, fingerprints = 0, {}, []
    for index, row in enumerate(rows):
        require(sha256(row['path']) == row['sha256'], 'native CONCH H5 SHA mismatch')
        slide = read_h5_slide(row['path'], feature_dim=768, normalize=True)
        x, xy = slide['features'], slide['coords']
        steps[index] = coordinate_step(xy)
        require(len(x) == row['n'], 'native CONCH patch count mismatch')
        for key, value in slide['coordinate_attrs'].items():
            if key in PUBLIC_GEOMETRY_KEYS and isinstance(value, (str, int, float, bool)):
                attrs.setdefault(str(key), set()).add(str(value))
        if selector == 'REGIONS':
            selected, windows = select_region_rows(xy, int(quota[index]))
        else:
            selected = select_patch_rows(x, xy, int(quota[index]), selector=selector, seed=int(seed)+index*104729)
            windows = np.zeros(len(selected), np.int64)
        n = len(selected)
        features[index, :n] = x[selected]
        coords[index, :n] = xy[selected]
        mask[index, :n] = True
        rowids[index, :n] = selected
        region_ids[index, :n] = windows
        full_slide_mean[index] = np.asarray(x, np.float64).mean(0)
        selected_slide_mean[index] = np.asarray(x[selected], np.float64).mean(0)
        patch_count += len(x)
        fingerprints.append(row['sha256'])
    features.flush()
    patient_full, patient_selected = np.zeros((512, 768)), np.zeros((512, 768))
    counts = np.bincount(owner, minlength=512)
    for index, patient in enumerate(owner):
        patient_full[patient] += full_slide_mean[index] / counts[patient]
        patient_selected[patient] += selected_slide_mean[index] / counts[patient]
    for name, value in [('coords', coords), ('mask', mask), ('rowids', rowids), ('owner', owner),
                        ('quota', quota), ('full_mean', patient_full), ('selected_mean', patient_selected),
                        ('coordinate_step', steps), ('region_ids', region_ids)]:
        np.save(output/(name+'.npy'), value)
    files = ['features', 'coords', 'mask', 'rowids', 'owner', 'quota', 'full_mean', 'selected_mean', 'coordinate_step', 'region_ids']
    receipt = {'schema': 'bgb_en_native_patch_cache_v1', 'status': 'COMPLETE', 'patients': 512,
               'slides': len(rows), 'original_patches': patch_count, 'selected_patches': int(mask.sum()),
               'feature_shape': list(shape), 'budget_per_patient': int(budget), 'selector': selector,
               'seed': int(seed), 'normalization': 'float32 per-patch L2; full/selected mean float64',
               'pooling': 'slide means then equal patient slide average', 'outcome_reads': 0,
               'source_inventory_sha256': sha256(inventory_path),
               'source_h5_collection_sha256': hashlib.sha256(''.join(fingerprints).encode()).hexdigest(),
               'coordinate_attributes': {k: sorted(v) for k, v in attrs.items()},
               'coordinate_step_counts': {str(int(step)): int((steps == step).sum()) for step in np.unique(steps)},
               'coordinate_step_source': 'GCD of all positive native distinct XY differences, before selection',
               'region_selection': 'dispersed-seed exact-four-neighbor BFS windows<=64' if selector == 'REGIONS' else 'not region selected',
               'region_boundary_reset_required': selector == 'REGIONS',
               'array_sha256': {name: sha256(output/(name+'.npy')) for name in files}}
    (output/'READY.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return receipt


def read_patch_cache(output, *, verify_hash=True):
    output = Path(output)
    receipt = json.loads((output/'READY.json').read_text(encoding='utf-8-sig'))
    require(receipt.get('schema') == 'bgb_en_native_patch_cache_v1' and receipt.get('status') == 'COMPLETE',
            'native patch cache is incomplete')
    result = {'audit': receipt}
    for name, digest in receipt['array_sha256'].items():
        path = output/(name+'.npy')
        if verify_hash:
            require(sha256(path) == digest, 'native cache array hash mismatch: '+name)
        result[name] = np.load(path, mmap_mode='r')
    require(result['features'].shape == tuple(receipt['feature_shape'])
            and result['mask'].shape == result['features'].shape[:2]
            and result['coords'].shape == result['mask'].shape+(2,), 'native patch cache shape mismatch')
    return result


def prepare_uni2_patch_cache(root, patient_ids, output, *, budget=128, selector='GRID', seed=270003):
    """Read original UNI2 H5 on its own grid, preserving its own row identity."""
    patient_ids = np.asarray(patient_ids).astype(str)
    require(len(patient_ids) == 512 and len(np.unique(patient_ids)) == 512, 'UNI2 canonical cohort mismatch')
    lookup = {pid: index for index, pid in enumerate(patient_ids)}
    paths = [path for path in Path(root).glob('*/*.h5') if path.stem[:12] in lookup]
    require(paths and len({path.stem for path in paths}) == len(paths), 'native UNI2 slide source duplicate/missing')
    rows = []
    for path in paths:
        require(path.is_file() and not path.is_symlink(), 'invalid native UNI2 source')
        with h5py.File(path, 'r') as handle:
            shape = handle['features'].shape
            require(len(shape) in (2, 3) and shape[-1] == 1536 and (len(shape) == 2 or shape[0] == 1),
                    'UNI2 source shape mismatch')
            rows.append({'path': path, 'patient': lookup[path.stem[:12]], 'n': shape[-2]})
    rows.sort(key=lambda row: (row['patient'], row['path'].name))
    require(set(row['patient'] for row in rows) == set(range(512)), 'UNI2 canonical patient coverage incomplete')
    quota = patient_quotas(rows, budget, 512)
    output = Path(output)
    require(not output.exists(), 'refuse to overwrite native UNI2 cache')
    output.mkdir(parents=True, exist_ok=False)
    shape = (len(rows), int(quota.max()), 1536)
    features = np.lib.format.open_memmap(output/'features.npy', mode='w+', dtype=np.float32, shape=shape)
    features[:] = 0
    coords = np.zeros(shape[:2]+(2,), np.int64)
    mask = np.zeros(shape[:2], bool)
    rowids = np.full(shape[:2], -1, np.int64)
    owner = np.asarray([row['patient'] for row in rows], np.int64)
    steps = np.zeros(len(rows), np.int64)
    region_ids = np.full(shape[:2], -1, np.int64)
    counts = np.bincount(owner, minlength=512)
    full_mean, selected_mean = np.zeros((512, 1536)), np.zeros((512, 1536))
    fingerprints, native_count, native_bytes, attrs = [], 0, 0, {}
    for index, row in enumerate(rows):
        before = row['path'].stat()
        fingerprints.append(sha256(row['path']))
        slide = read_h5_slide(row['path'], feature_dim=1536, normalize=True)
        x, xy = slide['features'], slide['coords']
        steps[index] = coordinate_step(xy)
        require(len(x) == row['n'], 'native UNI2 patch count changed')
        after = row['path'].stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'native UNI2 changed during read')
        for key, value in slide['coordinate_attrs'].items():
            if key in PUBLIC_GEOMETRY_KEYS and isinstance(value, (str, int, float, bool)):
                attrs.setdefault(key, set()).add(str(value))
        if selector == 'REGIONS':
            selected, windows = select_region_rows(xy, int(quota[index]))
        else:
            selected = select_patch_rows(x, xy, int(quota[index]), selector=selector, seed=int(seed)+index*104729)
            windows = np.zeros(len(selected), np.int64)
        n = len(selected)
        features[index, :n], coords[index, :n] = x[selected], xy[selected]
        mask[index, :n], rowids[index, :n] = True, selected
        region_ids[index, :n] = windows
        full_mean[row['patient']] += np.asarray(x, np.float64).mean(0)/counts[row['patient']]
        selected_mean[row['patient']] += np.asarray(x[selected], np.float64).mean(0)/counts[row['patient']]
        native_count += len(x)
        native_bytes += before.st_size
    features.flush()
    arrays = {'coords': coords, 'mask': mask, 'rowids': rowids, 'owner': owner, 'quota': quota,
              'full_mean': full_mean, 'selected_mean': selected_mean, 'coordinate_step': steps,
              'region_ids': region_ids}
    for name, value in arrays.items():
        np.save(output/(name+'.npy'), value)
    files = ['features'] + list(arrays)
    receipt = {'schema': 'bgb_en_native_patch_cache_v1', 'status': 'COMPLETE', 'encoder': 'UNI2-h',
               'patients': 512, 'slides': len(rows), 'original_patches': native_count,
               'native_h5_bytes': native_bytes, 'selected_patches': int(mask.sum()), 'feature_shape': list(shape),
               'budget_per_patient': int(budget), 'selector': selector, 'seed': int(seed),
               'normalization': 'float32 per-patch L2; full/selected mean float64',
               'pooling': 'slide means then equal patient slide average', 'outcome_reads': 0,
               'source_h5_collection_sha256': hashlib.sha256(''.join(fingerprints).encode()).hexdigest(),
               'source_order': 'canonical patient index then original slide filename',
               'coordinate_attributes': {key: sorted(value) for key, value in attrs.items()},
               'coordinate_step_counts': {str(int(step)): int((steps == step).sum()) for step in np.unique(steps)},
               'coordinate_step_source': 'GCD of all positive native distinct XY differences, before selection',
               'region_selection': 'dispersed-seed exact-four-neighbor BFS windows<=64' if selector == 'REGIONS' else 'not region selected',
               'region_boundary_reset_required': selector == 'REGIONS',
               'CONCH_correspondence': 'independent original grid; no by-row pairing',
               'array_sha256': {name: sha256(output/(name+'.npy')) for name in files}}
    (output/'READY.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return receipt
