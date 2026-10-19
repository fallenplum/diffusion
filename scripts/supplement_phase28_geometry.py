"""Append geometry risks using only registry-bound coordinates, not H5 rehash.

The original input-audit script and its source receipt remain unchanged. No
neural forward, fitting, central NPZ or endpoint access occurs in this pass.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
from src.phase28_reread import data, spatial
from scripts.audit_phase28_inputs import REPORT, REGISTRY, require, read_only_guards


def distribution(values):
    x = np.asarray(values, np.float64)
    require(len(x) > 0 and np.isfinite(x).all(), 'nonfinite geometry summary')
    return {'count': len(x), 'minimum': float(x.min()), 'p05': float(np.quantile(x, .05)),
            'p25': float(np.quantile(x, .25)), 'median': float(np.median(x)),
            'mean': float(x.mean()), 'p75': float(np.quantile(x, .75)),
            'p95': float(np.quantile(x, .95)), 'maximum': float(x.max())}


def main():
    began, script_sha = time.monotonic(), data.sha256(Path(__file__))
    previous_sha = data.sha256(REPORT)
    report = json.loads(REPORT.read_text())
    require(report['status'] == 'PASS' and 'geometry_risks' not in report, 'one unsupplemented completed input audit required')
    registry_sha = data.sha256(REGISTRY)
    require(report['private_registry_sha256'] == registry_sha, 'original private H5 registry changed')
    registry = json.loads(REGISTRY.read_text())
    require(len(registry['H5_files']) == 520, 'full520 source registry required')
    for key in ('data', 'spatial'):
        require(data.sha256(ROOT/f'src/phase28_reread/{key}.py') == report['input_code_sha256'][key],
                'input/geometry implementation changed after original audit')
    total_regions = singleton_regions = selected_patches = 0
    per_WSI_singleton, per_WSI_eligible, coverage = [], [], []
    totals = {key: {'segments': 0, 'length_one': 0} for key in spatial.DIRECTIONS}
    segment_fractions = {key: [] for key in spatial.DIRECTIONS}
    with read_only_guards() as (calls, reads):
        for row in registry['H5_files']:
            path = Path(row['path']); before = path.stat()
            require(before.st_size == row['bytes'] and before.st_mtime_ns == row['mtime_ns'],
                    'registered source metadata changed')
            with h5py.File(path, 'r') as handle:
                raw = np.asarray(handle['coords'])
            require(hashlib.sha256(raw.tobytes()).hexdigest() == row['coords_SHA256'],
                    'coordinate SHA differs from original full audit')
            coords = raw.reshape(-1, 2)
            plan = spatial.build_plan(coords, step=512, tile_size=16, max_regions=64,
                                      seed=row['geometry_seed'])
            n = len(plan['selected_indices']); lengths = np.asarray([len(v) for v in plan['regions']])
            require(n == row['selected_rows'] and len(lengths) == row['selected_regions'],
                    'coordinate plan counts differ from original audit')
            single = int((lengths == 1).sum())
            total_regions += len(lengths); singleton_regions += single; selected_patches += n
            per_WSI_singleton.append(single/len(lengths)); per_WSI_eligible.append(1-single/len(lengths))
            coverage.append(n/len(coords))
            for direction in spatial.DIRECTIONS:
                runs = plan['segments'][direction]
                one = sum(len(run) == 1 for run in runs)
                totals[direction]['segments'] += len(runs); totals[direction]['length_one'] += one
                segment_fractions[direction].append(one/len(runs))
            after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    'source metadata changed during coordinate-only supplement')
        require(not any(calls.values()) and not reads, 'unexpected neural/optimizer/NPZ read')
    require(selected_patches == report['geometry']['selected_Patches'], 'selected Patch aggregate changed')
    for key in ('data', 'spatial'):
        require(data.sha256(ROOT/f'src/phase28_reread/{key}.py') == report['input_code_sha256'][key],
                'input/geometry source changed during supplement')
    require(data.sha256(REGISTRY) == registry_sha and data.sha256(REPORT) == previous_sha
            and data.sha256(Path(__file__)) == script_sha, 'supplement input/source bytes changed')
    report['geometry_risks'] = {
        'supplement_script_sha256': script_sha, 'original_input_audit_sha256_before_supplement': previous_sha,
        'H5_full_SHA_passes_repeated': 0, 'coordinate_arrays_rechecked_against_saved_SHA': 520,
        'central_NPZ_arrays_accessed': 0, 'neural_forward_calls': 0, 'new_predictor_fits': 0,
        'selected_regions_total': total_regions, 'singleton_regions': singleton_regions,
        'singleton_region_fraction_all_selected_regions': singleton_regions/total_regions,
        'singleton_region_fraction_equal_WSI': distribution(per_WSI_singleton),
        'read_query_slots_total': 2*total_regions,
        'read_query_slots_with_at_least_two_real_candidates': 2*(total_regions-singleton_regions),
        'query_slot_fraction_with_at_least_two_candidates': 1-singleton_regions/total_regions,
        'query_eligibility_fraction_equal_WSI': distribution(per_WSI_eligible),
        'read_query_scope': 'two planned queries per region; candidate eligibility, not learned attention usage',
        'scan_segments_by_direction': {key: {**value,
            'length_one_fraction_of_segments': value['length_one']/value['segments'],
            'length_one_fraction_of_selected_Patch_visits': value['length_one']/selected_patches,
            'length_one_segment_fraction_equal_WSI': distribution(segment_fractions[key])}
            for key, value in totals.items()},
        'per_WSI_selected_Patch_coverage_distribution': distribution(coverage),
        'coverage_denominator': 'all available original H5 feature rows; not original WSI tissue/pixel coverage',
        'raw_WSI_full_tissue_coverage_established': False,
        'interpretation': 'real coordinate continuity only within existing contiguous runs; singleton regions cannot test candidate reallocation; length-one runs have no recurrent within-run spatial context',
        'wall_seconds': time.monotonic()-began}
    temp = REPORT.with_suffix('.geometry_supplement.partial')
    require(not temp.exists(), 'orphan geometry supplement partial')
    temp.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n')
    temp.replace(REPORT)
    print(json.dumps({'status': 'PASS_GEOMETRY_RISK_SUPPLEMENT', 'singleton_regions': singleton_regions,
                      'selected_regions': total_regions, 'eligible_query_fraction': 1-singleton_regions/total_regions,
                      'bytes': REPORT.stat().st_size}, sort_keys=True), flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print('PHASE28_GEOMETRY_SUPPLEMENT_FAILED', type(error).__name__, flush=True)
        raise SystemExit(1)
