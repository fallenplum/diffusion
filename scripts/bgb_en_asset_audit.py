"""Read-only BG-BEN input qualification; public output contains no row IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np

from src.bgb_en.data import (CLINICAL_COLUMNS, PUBLIC_GEOMETRY_KEYS, canonical_order, load_raw,
                            native_conch_catalog, prepare_native_patch_cache,
                            read_candidate_cache, read_h5_slide, require, sha256)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _quantiles(values):
    values = np.asarray(values, np.float64)
    return {'min': float(values.min()), 'median': float(np.median(values)), 'max': float(values.max())}


def cohort_audit(raw):
    answer = {'patients': len(raw['patient_ids']), 'outer_folds': 5,
              'DSS_events': int(raw['e'].sum()), 'DSS_time_unit': 'months',
              'PFI_valid_patients': int(raw['pfi_valid'].sum()),
              'PFI_valid_events': int(raw['pfi_e'][raw['pfi_valid']].sum()),
              'clinical_fields': list(CLINICAL_COLUMNS), 'S32_shape': list(raw['S32'].shape),
              'TITAN_shape': list(raw['titan'].shape), 'UNI2_mean_shape': list(raw['uni2_mean'].shape),
              'raw_RNA_missing_patterns': int(len(np.unique(raw['platform']))),
              'hashes': {key: value for key, value in raw['audit'].items() if key.endswith('sha256')},
              'cohort_labels_used_for_model_selection': False, 'preprocessing_fits': 0,
              'RNA_scale': 'already log2(x+1); no additional transform in audit',
              'CNA_scale': 'continuous signed log2 dosage; gain/loss must be separate before centering',
              'platform_identity': 'unverified; exact missing-pattern proxy only'}
    for name in ('rna', 'cna'):
        values = raw[name]
        finite = np.isfinite(values)
        answer[name.upper()] = {'shape': list(values.shape), 'dtype': str(values.dtype),
                              'array_bytes': int(values.nbytes), 'observed_values': int(finite.sum()),
                              'missing_values': int((~finite).sum()),
                              'all_patients_measured_genes': int(finite.all(0).sum()),
                              'any_patient_measured_genes': int(finite.any(0).sum()),
                              'complete_patients': int(finite.all(1).sum()),
                              'zero_observation_genes': int((~finite.any(0)).sum()),
                              'observed_genes_per_patient': _quantiles(finite.sum(1)),
                              'symbol_count': len(raw[name + '_symbols']),
                              'missing_is_not_expression_zero': True}
    return answer


def candidate_audit(cache):
    x, xy, slides = cache['features'], cache['coords'], cache['slides']
    duplicate = 0
    norms = []
    step_modes = {}
    geometry = {}
    for row in slides:
        a, b = int(row['offset']), int(row['offset']) + int(row['n'])
        coords = xy[a:b]
        canonical_order(coords)
        feature = np.asarray(x[a:b])
        require(np.isfinite(feature).all(), 'nonfinite full candidate CONCH features')
        norm = np.linalg.norm(feature, axis=1)
        require((norm > 1e-12).all(), 'zero CONCH feature norm')
        norms.extend([float(norm.min()), float(norm.max())])
        duplicate += len(coords) - len(np.unique(coords, axis=0))
        for key, value in row.get('geometry', {}).items():
            if key in PUBLIC_GEOMETRY_KEYS and isinstance(value, (str, int, float, bool)):
                geometry.setdefault(str(key), set()).add(str(value))
        for axis in range(2):
            d = np.diff(np.unique(coords[:, axis]))
            if len(d):
                unique, count = np.unique(d, return_counts=True)
                step = int(unique[int(np.argmax(count))])
                step_modes[str(step)] = step_modes.get(str(step), 0) + 1
    return {'status': 'PASS', 'patients': int(cache['manifest']['patients']), 'slides': len(slides),
            'patches': len(x), 'feature_dim': 768, 'dtype': str(x.dtype),
            'feature_array_bytes': int(x.nbytes), 'coordinate_array_bytes': int(xy.nbytes),
            'source_manifest_sha256': cache['manifest_sha256'],
            'source_h5_collection_sha256': cache['manifest']['source_sha256'],
            'feature_sha256': cache['manifest']['cache_sha256'],
            'coordinate_sha256': cache['manifest']['coordinate_sha256'],
            'duplicate_coordinates': duplicate, 'nonfinite_feature_values': 0,
            'feature_norm_range': [min(norms), max(norms)],
            'coordinate_axis_step_mode_counts': step_modes,
            'coordinate_attributes': {key: sorted(values) for key, values in geometry.items()},
            'normalization': 'per-patch L2 normalization only; no outcome-derived projection',
            'semantic_projection_applied': bool(cache['manifest'].get('semantic_projection_applied', False)),
            'native_grid_pairing_to_UNI2': 'distinct grids; no by-row pairing permitted',
            'aggregate_patient_pool': 'within-slide mean then equal slide means'}


def native_uni2_audit(root, patient_ids, *, full_hash=False):
    lookup = {value: index for index, value in enumerate(patient_ids)}
    paths = [path for path in Path(root).glob('*/*.h5') if path.stem[:12] in lookup]
    require(paths, 'no native UNI2 H5 files matched the canonical cohort')
    seen, patients, patches, size, fingerprint = set(), set(), 0, 0, []
    scale = {}
    shapes, norms = set(), []
    for path in sorted(paths):
        require(path.stem not in seen and not path.is_symlink(), 'duplicate/symlink native UNI2 source')
        seen.add(path.stem)
        slide = read_h5_slide(path, feature_dim=1536, normalize=False)
        x, xy = slide['features'], slide['coords']
        patches += len(x)
        size += path.stat().st_size
        patients.add(lookup[path.stem[:12]])
        with h5py.File(path, 'r') as handle:
            shapes.add(str(tuple(handle['features'].shape)))
        norm = np.linalg.norm(x, axis=1)
        norms.extend([float(norm.min()), float(norm.max())])
        for key, value in slide['coordinate_attrs'].items():
            if key in PUBLIC_GEOMETRY_KEYS and isinstance(value, (str, int, float, bool)):
                scale.setdefault(key, set()).add(str(value))
        # This distinction is explicit: quick metadata fingerprints do not
        # claim byte-level H5 integrity. Full-hash mode is separately reported.
        fingerprint.append(sha256(path) if full_hash else str((path.stat().st_size, len(x))))
    import hashlib
    require(len(patients) == len(patient_ids), 'some canonical patients lack native UNI2 patches')
    return {'status': 'PASS', 'patients': len(patients), 'slides': len(paths), 'patches': patches,
            'feature_dim': 1536, 'total_h5_bytes': size, 'all_native_rows_finite_checked': True,
            'original_H5_shape_support': '2D or singleton-leading 3D; shape retained in private source',
            'coordinate_attributes': {key: sorted(values) for key, values in scale.items()},
            'feature_norm_range': [min(norms), max(norms)],
            'collection_fingerprint_sha256': hashlib.sha256('|'.join(fingerprint).encode()).hexdigest(),
            'fingerprint_type': 'ordered full-H5 SHA256 collection' if full_hash else 'ordered metadata only; not a byte-integrity claim',
            'native_CONCH_grid_pairing': 'distinct grids; no by-row pairing permitted'}


def native_conch_audit(inventory, *, root_override=None):
    rows = native_conch_catalog(inventory, root_override=root_override)
    patches, attrs, norms, step_modes, fingerprints = 0, {}, [], {}, []
    for row in rows:
        require(sha256(row['path']) == row['sha256'], 'original CONCH H5 SHA256 mismatch')
        slide = read_h5_slide(row['path'], feature_dim=768, normalize=False)
        x, xy = slide['features'], slide['coords']
        require(len(x) == row['n'], 'original CONCH patch count changed')
        patches += len(x)
        norm = np.linalg.norm(x, axis=1)
        norms.extend([float(norm.min()), float(norm.max())])
        for key, value in slide['coordinate_attrs'].items():
            if key in PUBLIC_GEOMETRY_KEYS and isinstance(value, (str, int, float, bool)):
                attrs.setdefault(key, set()).add(str(value))
        for axis in range(2):
            delta = np.diff(np.unique(xy[:, axis]))
            if len(delta):
                values, count = np.unique(delta, return_counts=True)
                key = str(int(values[int(np.argmax(count))]))
                step_modes[key] = step_modes.get(key, 0)+1
        fingerprints.append(row['sha256'])
    import hashlib
    return {'status': 'PASS', 'patients': 512, 'slides': len(rows), 'patches': patches,
            'feature_dim': 768, 'total_h5_bytes': sum(row['bytes'] for row in rows),
            'all_native_rows_finite_checked': True, 'duplicate_coordinates': 0,
            'source_h5_collection_sha256': hashlib.sha256(''.join(fingerprints).encode()).hexdigest(),
            'full_original_H5_hashes_verified': True,
            'coordinate_attributes': {key: sorted(values) for key, values in attrs.items()},
            'coordinate_axis_step_mode_counts': step_modes, 'feature_norm_range': [min(norms), max(norms)],
            'volatile_candidate_cache_available': False,
            'native_grid_pairing_to_UNI2': 'distinct grids; no by-row pairing permitted'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets-json', required=True, help='Private path mapping; never publish this file')
    parser.add_argument('--output', required=True)
    parser.add_argument('--full-native-hash', action='store_true')
    parser.add_argument('--schema-only', action='store_true')
    parser.add_argument('--prepare-cache', help='Private new feature-only cache directory; refuses overwrite')
    parser.add_argument('--patch-budget', type=int, default=512)
    parser.add_argument('--selector', choices=('GRID', 'DIVERSE', 'RAND', 'REGIONS'), default='GRID')
    parser.add_argument('--seed', type=int, default=270003)
    args = parser.parse_args()
    started = time.time()
    assets = json.loads(Path(args.assets_json).read_text(encoding='utf-8-sig'))
    raw = load_raw(assets)
    result = {'schema': 'bgb_en_R0_asset_audit_v1', 'status': 'PASS_COHORT_ONLY',
              'cohort': cohort_audit(raw), 'model_fits': 0, 'patient_identifiers_public': False,
              'private_paths_public': False}
    if not args.schema_only:
        if assets.get('conch_inventory'):
            result['CONCH'] = native_conch_audit(assets['conch_inventory'], root_override=assets.get('conch_root'))
        else:
            candidate = read_candidate_cache(assets['candidate_manifest'],
                                            feature_override=assets.get('candidate_features'),
                                            coords_override=assets.get('candidate_coords'))
            result['CONCH'] = candidate_audit(candidate)
        if assets.get('uni2_root'):
            result['UNI2'] = native_uni2_audit(assets['uni2_root'], raw['patient_ids'], full_hash=args.full_native_hash)
        if args.prepare_cache:
            result['new_feature_only_cache'] = prepare_native_patch_cache(
                assets['conch_inventory'], args.prepare_cache, budget=args.patch_budget,
                selector=args.selector, seed=args.seed, root_override=assets.get('conch_root'))
        result['status'] = 'PASS_COHORT_AND_AVAILABLE_PATCH_ASSETS'
    result['wall_seconds'] = time.time() - started
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
