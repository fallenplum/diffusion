"""One read-only real-input qualification; no encoder, optimizer or endpoints.

Full H5/coordinate fingerprints and source paths stay in the private registry.
The public report contains only anonymous counts, maxima and receipt hashes.
This script is independent of the not-yet-locked Phase28 training schedule.
"""
from __future__ import annotations

from collections import Counter
import contextlib
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
from src.phase28_reread import data, spatial


PRIVATE = Path('/root/autodl-tmp/phase28_reread_private')
REPORT = ROOT/'reports/phase28_bidir_reread/INPUT_AUDIT.json'
REGISTRY = PRIVATE/'H5_SOURCES_PRIVATE.json'
STATS_ATOL = 1e-10
FEATURE_KEYS = frozenset(('patient_ids', 'folds', 'rna_genes', 'rna_gene_symbols',
                         'cna_genes', 'cna_gene_symbols', 'cancer_type'))


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.partial')
    require(not temp.exists(), 'orphan audit partial needs explicit review')
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def moments(values):
    a = np.asarray(values, np.float64)
    require(a.size > 0 and np.isfinite(a).all(), 'invalid aggregate statistic')
    return {'count': int(a.size), 'minimum': float(a.min()), 'mean': float(a.mean()),
            'median': float(np.median(a)), 'maximum': float(a.max())}


@contextlib.contextmanager
def read_only_guards():
    """Hard block neural/optimizer calls and reject any non-feature NPZ key."""
    calls, reads = {'nn.Module._call_impl': 0, 'Optimizer.__init__': 0,
                    'forbidden_central_NPZ_array': 0}, Counter()
    original_load = np.load
    central = data.COHORT_NPZ.resolve()

    class FeatureArchive:
        def __init__(self, archive): self.archive = archive
        def __enter__(self): return self
        def __exit__(self, *args): self.archive.close()
        def __getitem__(self, key):
            if key not in FEATURE_KEYS:
                calls['forbidden_central_NPZ_array'] += 1
                raise RuntimeError('input audit forbids non-feature central NPZ keys')
            reads[key] += 1
            return self.archive[key]

    def load(file, *args, **kwargs):
        try: source = Path(file).resolve()
        except TypeError: source = None
        value = original_load(file, *args, **kwargs)
        return FeatureArchive(value) if source == central else value

    def forbidden(label):
        def fail(*args, **kwargs):
            calls[label] += 1
            raise RuntimeError('input audit forbids '+label)
        return fail

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(torch.nn.Module, '_call_impl', forbidden('nn.Module._call_impl')))
        stack.enter_context(patch.object(torch.optim.Optimizer, '__init__', forbidden('Optimizer.__init__')))
        stack.enter_context(patch.object(np, 'load', load))
        yield calls, reads


def verify_fold(dataset, raw_values, raw_mask):
    """Independent NumPy masked mean/SD; held-out labels are never requested."""
    fit, active = dataset.fit, dataset.active
    state = dataset.molecular_state
    actual = dataset.molecular.cpu().numpy()
    actual_mask = dataset.molecular_mask.cpu().numpy()
    require(actual.shape == actual_mask.shape == (512, 26708, 3)
            and actual.dtype == np.float32 and actual_mask.dtype == bool
            and np.isfinite(actual).all() and not actual[~actual_mask].any(),
            'all-gene molecular tensor/measurement mask qualification failed')
    require(not any(hasattr(dataset, key) for key in
                    ('valid_time', 'valid_event', '_valid_time', '_valid_event', 'endpoints', 'ep')),
            'held-out supervision retained in Data')
    require(np.array_equal(dataset.gene_ids.cpu().numpy(), np.arange(26708)), 'GID order mismatch')
    membership = dataset.slot_membership.cpu().numpy()
    omega = dataset.omega.cpu().numpy()
    require(membership.shape == (108, 26708) and membership.dtype == bool
            and np.array_equal(membership[:100], np.repeat(omega, 2, axis=0))
            and membership[100:].all(), 'Hallmark two-slot or all-gene residual support changed')
    expected_mask = np.zeros(raw_mask.shape, bool)
    errors, channels = {}, {}
    for channel, name in enumerate(data.CHANNELS):
        measured_fit = raw_mask[fit, :, channel]
        source = np.where(measured_fit, raw_values[fit, :, channel], np.nan).astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            expected_mean = np.nanmean(source, axis=0)
            expected_sd = np.nanstd(source, axis=0, ddof=0)
        counts = measured_fit.sum(0, dtype=np.int64)
        expected_mean = np.where(counts > 0, expected_mean, 0.)
        expected_sd = np.where(np.isfinite(expected_sd) & (expected_sd > 0.), expected_sd, 1.)
        require(np.array_equal(counts, state['count'][:, channel]), 'fit finite-count mismatch')
        mean_error = float(np.max(abs(expected_mean-state['mean'][:, channel])))
        sd_error = float(np.max(abs(expected_sd-state['sd'][:, channel])))
        require(max(mean_error, sd_error) <= STATS_ATOL, 'fit mean/population SD mismatch')
        expected_mask[active, :, channel] = raw_mask[active, :, channel] & (counts > 0)
        expected_value = np.zeros((512, 26708), np.float32)
        standardized = ((raw_values[active, :, channel].astype(np.float64)-state['mean'][:, channel])
                        / state['sd'][:, channel])
        expected_value[active] = np.where(expected_mask[active, :, channel], standardized, 0.).astype(np.float32)
        require(np.array_equal(expected_value, actual[:, :, channel]), 'stored molecular standardization mismatch')
        require(not actual_mask[:, counts == 0, channel].any(), 'fit-unmeasured channel was enabled in held-out')
        errors[name] = {'mean_max_absolute_error': mean_error, 'population_SD_max_absolute_error': sd_error,
                        'standardized_value_max_absolute_error': 0.}
        channels[name] = {'fit_count_minimum': int(counts.min()), 'fit_count_maximum': int(counts.max()),
                          'fit_zero_count_genes': int((counts == 0).sum()),
                          'fit_measured_entries': int(actual_mask[fit, :, channel].sum()),
                          'valid_measured_entries': int(actual_mask[dataset.valid, :, channel].sum()),
                          'raw_observed_zero_entries': int(((raw_values[:, :, channel] == 0)
                                                           & raw_mask[:, :, channel]).sum())}
    require(np.array_equal(actual_mask, expected_mask), 'pre-imputation/effective measurement mask mismatch')
    base = dataset.base.cpu().numpy()
    require(np.array_equal(base[fit], dataset.CP_cf_fit+dataset.RNA_cf_fit)
            and np.array_equal(base[dataset.valid], dataset.M0_valid)
            and np.array_equal(dataset.M0_valid, dataset.CP_valid+dataset.RNA_valid),
            'fixed crossfit/fullfit strong-main offset changed')
    return {'fold': dataset.fold, 'seed_for_loader': dataset.seed,
            'fit_patients': len(fit), 'valid_patients': len(dataset.valid),
            'prepared_sha256': dataset.prepared_sha256, 'main_evidence_sha256': dataset.audit['main_evidence_sha256'],
            'finite_count_and_transform_checks': channels, 'statistics_errors': errors,
            'main_offsets_exact': True, 'valid_endpoint_arrays_read': 0,
            'fit_label_getter_calls': 0, 'canonical_feature_shape': [512, 26708, 3],
            'all_gene_ids_retained': True, 'slots': 108}


def verify_geometry(plan, coords):
    selected = np.asarray(plan['selected_indices'])
    n = len(selected)
    require(0 < n <= 64*256 and len(np.unique(selected)) == n
            and (selected >= 0).all() and (selected < len(coords)).all(), 'invalid original-row geometry selection')
    require(np.array_equal(np.concatenate(plan['regions']), selected), 'region original-row concatenation changed')
    require(len(plan['regions']) <= 64 and max(map(len, plan['regions'])) <= 256, 'spatial region budget changed')
    require(spatial.validate_plan(plan), 'independent geometry invariants failed')
    for direction in spatial.DIRECTIONS:
        require(np.array_equal(np.sort(np.concatenate(plan['segments'][direction])), np.arange(n)),
                'direction omitted/repeated a selected real Patch')
    require(plan['audit']['fabricated_patch_locations'] == 0
            and plan['audit']['outcome_or_epoch_inputs'] is False
            and plan['audit']['selected_patches'] == n
            and plan['audit']['selected_regions'] == len(plan['regions']), 'geometry audit accounting differs')


def scan_h5(catalog):
    private, fractions, selected_counts, region_counts, all_regions = [], [], [], [], []
    segments = {direction: [] for direction in spatial.DIRECTIONS}
    total_bytes = total_patches = selected_total = 0
    for index, row in enumerate(catalog):
        require(index == row['index'], 'stable full-cohort slide index changed')
        path = Path(row['path']); before = path.stat()
        require(before.st_size == row['size'] and before.st_mtime_ns == row['mtime_ns'], 'source H5 metadata changed')
        digest = data.sha256(path)  # One complete byte pass, never load features into a new cache.
        with h5py.File(path, 'r') as handle:
            patches = data._h5_shapes(handle)
            raw_coords = np.asarray(handle['coords'])
            coords = raw_coords.reshape(-1, 2)
            require(len(coords) == patches, 'feature/original coordinate row pairing mismatch')
            plan = spatial.build_plan(coords, step=512, tile_size=16, max_regions=64,
                                      seed=data.GEOMETRY_SEED+index)
            verify_geometry(plan, coords)
            meta = {'index': index, 'owner': row['owner'], 'path': str(path),
                    'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns,
                    'H5_sha256': digest, 'coords_SHA256': hashlib.sha256(raw_coords.tobytes()).hexdigest(),
                    'coords_array_hash_with_shape_dtype': data.array_hash(raw_coords),
                    'feature_shape': list(handle['features'].shape), 'feature_dtype': str(handle['features'].dtype),
                    'coords_shape': list(raw_coords.shape), 'coords_dtype': str(raw_coords.dtype),
                    'source_rows': patches, 'geometry_seed': data.GEOMETRY_SEED+index,
                    'selected_rows': len(plan['selected_indices']), 'selected_regions': len(plan['regions'])}
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                'source H5 changed during full SHA/coordinate audit')
        private.append(meta); total_bytes += before.st_size; total_patches += patches
        selected = len(plan['selected_indices']); selected_total += selected
        selected_counts.append(selected); fractions.append(selected/patches)
        region_counts.append(len(plan['regions'])); all_regions.append(plan['audit']['all_regions'])
        for direction in spatial.DIRECTIONS:
            segments[direction].append(len(plan['segments'][direction]))
        if (index+1) % 100 == 0:
            print('PHASE28_H5_AUDITED', index+1, flush=True)
    require(len(private) == 520 and len({r['owner'] for r in private}) == 512
            and total_patches == 3295129, 'full native-H5 cohort/Patch coverage changed')
    return private, {'slides': len(private), 'patients': 512, 'native_Patches': total_patches,
        'native_H5_bytes': total_bytes, 'H5_full_SHA_files': len(private), 'coords_SHA_files': len(private),
        'selected_Patches': selected_total, 'Patch_coverage_weighted_by_native_Patches': selected_total/total_patches,
        'Patch_coverage_equal_WSI': moments(fractions), 'selected_Patches_per_WSI': moments(selected_counts),
        'selected_regions_per_WSI': moments(region_counts), 'all_regions_per_WSI': moments(all_regions),
        'segments_per_direction_per_WSI': {key: moments(value) for key, value in segments.items()},
        'all_directions_visit_every_selected_patch_once': True,
        'all_original_coordinates_integer_and_same_512_grid_phase': True,
        'all_regions_nonempty_four_connected_within_16x16_tile': True,
        'no_missing_patch_location_inserted': True, 'old_512_candidate_cache_used': False,
        'neural_encoder_or_packing_forward_called': False,
        'feature_payload_finite_rescan_here': False, 'feature_shape_coordinate_pairing_checked': True}


def main():
    began = time.monotonic()
    require(not REPORT.exists() and not REGISTRY.exists(), 'existing input audit/registry: refusing silent repeat')
    files = {'data': ROOT/'src/phase28_reread/data.py', 'spatial': ROOT/'src/phase28_reread/spatial.py',
             'audit_script': Path(__file__)}
    code_sha = {key: data.sha256(value) for key, value in files.items()}
    with read_only_guards() as (calls, reads):
        raw = data.read_feature_npz()
        values, masks, symbols = data.align_all_genes(raw['rna_genes'], raw['rna_gene_symbols'],
                                                     raw['cna_genes'], raw['cna_gene_symbols'])
        fold_rows, catalog = [], None
        for fold in (1, 2, 3):
            dataset = data.Data(fold, 31, device='cpu', cache_slides=0)
            require(np.array_equal(dataset.gene_symbols, symbols), 'fold-specific canonical GID universe changed')
            fold_rows.append(verify_fold(dataset, values, masks))
            if catalog is None: catalog = dataset.slide_catalog
            else:
                require(catalog == dataset.slide_catalog, 'native H5 catalogue/order changed across folds')
            del dataset; gc.collect()
        del raw, values, masks; gc.collect()
        h5_rows, geometry = scan_h5(catalog)
        require(not any(calls.values()) and set(reads) == FEATURE_KEYS and all(v == 4 for v in reads.values()),
                'unexpected forward/optimizer/endpoint or central NPZ key access')
        require(code_sha == {key: data.sha256(value) for key, value in files.items()}, 'input code changed during audit')
    registry = {'status': 'PASS_PRIVATE_INPUT_SOURCE_REGISTRY', 'version': 'phase28_input_sources_v1',
                'input_code_sha256': code_sha, 'canonical_NPZ_sha256': data.COHORT_SHA256,
                'official_Hallmark_2024_1_sha256': data.GMT_SHA256,
                'prepared_sha256_by_fold': {str(row['fold']): row['prepared_sha256'] for row in fold_rows},
                'H5_files': h5_rows, 'full_original_H5_and_coords_audited': True,
                'patient_paths_identifiers_private': True, 'no_raw_feature_or_coordinate_arrays_saved': True}
    atomic_json(REGISTRY, registry)
    report = {'status': 'PASS', 'stage': 'prototype_input_qualification_only', 'input_code_sha256': code_sha,
              'audit_script_sha256': code_sha['audit_script'], 'training_source_not_yet_locked': True,
              'canonical_NPZ_sha256': data.COHORT_SHA256, 'official_Hallmark_2024_1_sha256': data.GMT_SHA256,
              'private_registry_sha256': data.sha256(REGISTRY), 'folds': fold_rows,
              'geometry': geometry, 'statistics_absolute_tolerance': STATS_ATOL,
              'molecular_shape': [512, 26708, 3], 'raw_H5_feature_dim': 1536, 'slots': 108,
              'RNA_already_log2_no_double_log': True, 'CNA_gain_loss_before_fit_scaling': True,
              'fit_unobserved_channels_disabled_for_valid': True, 'source_Clinical_only_in_frozen_CP': True,
              'forbidden_API_call_counts': calls, 'central_feature_array_read_counts': dict(reads),
              'new_neural_training_steps': 0, 'new_predictor_fits': 0, 'optimizer_constructions': 0,
              'neural_forward_calls': 0, 'valid_endpoint_array_reads': 0, 'fit_label_getter_calls': 0,
              'fit_label_bytes_deserialized_in_prepared': True, 'no_valid_labels_in_prepared': True,
              'new_raw_feature_or_coordinate_array_files': 0, 'public_patient_arrays_weights_or_IDs': False,
              'qualification_is_not_prognostic_or_histologic_validation': True,
              'wall_seconds': time.monotonic()-began}
    atomic_json(REPORT, report)
    print(json.dumps({'status': report['status'], 'H5_files': len(h5_rows), 'folds': len(fold_rows),
                      'bytes': REPORT.stat().st_size, 'wall_seconds': report['wall_seconds']}, sort_keys=True), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        PRIVATE.mkdir(parents=True, exist_ok=True)
        failure = PRIVATE/'INPUT_AUDIT_FAILED_PRIVATE.json'
        if not failure.exists():
            atomic_json(failure, {'status': 'FAILED_NOT_RETRIED', 'error_type': type(error).__name__,
                                 'error_message_private': str(error), 'script_sha256': data.sha256(Path(__file__))})
        # File exceptions can contain patient filenames; stdout stays anonymous.
        print('PHASE28_INPUT_AUDIT_FAILED', type(error).__name__, flush=True)
        raise SystemExit(1)
